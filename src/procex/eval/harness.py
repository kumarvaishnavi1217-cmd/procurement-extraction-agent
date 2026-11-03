"""Field-level evaluation against a labelled document set.

    procex-eval --dataset eval/dataset --extractor claude          # live, needs ANTHROPIC_API_KEY
    procex-eval --dataset eval/dataset --extractor claude-code     # live via `claude -p`, Claude subscription
    procex-eval --dataset eval/dataset --extractor fixture --fixtures eval/fixtures

    --save-outputs DIR         keep the raw model output per document, so failures can be replayed offline
    --langfuse-dataset NAME    sync the documents to a Langfuse dataset and record the run as an experiment

Besides raw accuracy it measures the review queue itself: how many wrong values it catches
(error catch rate) and how accurate the values are that skip review (auto-accepted accuracy) -
the number that matters when tuning the review threshold, prompt, model or effort.
"""

import argparse
import asyncio
import hashlib
import json
import sys
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langfuse import Evaluation

from procex import tracing
from procex.config import get_settings
from procex.extractor import ClaudeCodeExtractor, ClaudeExtractor, ExtractionError, Extractor, FixtureExtractor
from procex.pdf import extract_text, normalise
from procex.pipeline import extract_and_validate
from procex.validation import ValidationResult

HEADER_FIELDS = ("supplier_name", "document_number", "document_date", "currency", "subtotal", "total")
LINE_FIELDS = ("part_number", "quantity", "unit_price", "line_total")
NUMERIC = {"subtotal", "total", "quantity", "unit_price", "line_total"}

# USD per million tokens (input, output), for the cost column only
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}


def values_match(name: str, expected: Any, actual: Any) -> bool:
    if expected is None or actual is None:
        return expected is None and actual is None
    if name in NUMERIC:
        try:
            return abs(float(expected) - float(actual)) < 0.005
        except (TypeError, ValueError):
            return False
    return normalise(str(expected)).rstrip(".") == normalise(str(actual)).rstrip(".")


@dataclass
class FieldScore:
    total: int = 0
    correct: int = 0
    flagged: int = 0
    flagged_wrong: int = 0  # wrong values the review queue caught
    accepted: int = 0
    accepted_correct: int = 0

    def add(self, correct: bool, flagged: bool) -> None:
        self.total += 1
        self.correct += correct
        self.flagged += flagged
        self.flagged_wrong += flagged and not correct
        self.accepted += not flagged
        self.accepted_correct += (not flagged) and correct


@dataclass
class Report:
    documents: int = 0
    failed_documents: list[str] = field(default_factory=list)
    per_field: dict[str, FieldScore] = field(default_factory=lambda: defaultdict(FieldScore))
    errors: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: list[int] = field(default_factory=list)
    model: str = ""

    def overall(self) -> FieldScore:
        total = FieldScore()
        for s in self.per_field.values():
            for k in asdict(total):
                setattr(total, k, getattr(total, k) + getattr(s, k))
        return total

    def summary(self) -> dict[str, Any]:
        o = self.overall()
        wrong = o.total - o.correct
        summary: dict[str, Any] = {
            "model": self.model,
            "documents": self.documents,
            "failed_documents": self.failed_documents,
            "fields_scored": o.total,
            "field_accuracy": _ratio(o.correct, o.total),
            "review_rate": _ratio(o.flagged, o.total),
            "auto_accepted_accuracy": _ratio(o.accepted_correct, o.accepted),
            "error_catch_rate": _ratio(o.flagged_wrong, wrong) if wrong else 1.0,
            "per_field": {
                name: {
                    "accuracy": _ratio(s.correct, s.total),
                    "review_rate": _ratio(s.flagged, s.total),
                    "n": s.total,
                }
                for name, s in sorted(self.per_field.items())
            },
            "errors": self.errors,
        }
        if self.latency_ms:
            summary["mean_latency_ms"] = int(sum(self.latency_ms) / len(self.latency_ms))
            summary["input_tokens"] = self.input_tokens
            summary["output_tokens"] = self.output_tokens
            if self.model in PRICES:  # API runs only; "(via Claude Code)" runs are on a subscription
                pin, pout = PRICES[self.model]
                cost = (self.input_tokens * pin + self.output_tokens * pout) / 1e6
                summary["cost_usd"] = round(cost, 4)
                summary["cost_per_document_usd"] = round(cost / max(1, self.documents), 4)
        return summary


def _ratio(a: int, b: int) -> float:
    return round(a / b, 4) if b else 0.0


@dataclass
class Runner:
    """Runs one document through extraction + validation and scores it into a shared report."""

    extractor: Extractor
    review_threshold: float
    buyer_names: list[str] | None
    save_outputs: Path | None = None
    report: Report = field(default_factory=Report)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def run(self, pdf_path: Path, label: dict[str, Any]) -> dict[str, Any]:
        """Returns what the experiment records as the task output: predicted values and what was flagged."""
        data = pdf_path.read_bytes()
        try:
            outcome = extract_and_validate(
                self.extractor, data, pdf_path.name, extract_text(data), self.review_threshold, self.buyer_names
            )
        except ExtractionError as e:
            with self._lock:
                self.report.documents += 1
                self.report.failed_documents.append(f"{pdf_path.name}: {e}")
            return {"error": str(e)}
        result, checks = outcome.result, outcome.checks
        if self.save_outputs:
            self.save_outputs.mkdir(parents=True, exist_ok=True)
            (self.save_outputs / f"{pdf_path.stem}.json").write_text(result.document.model_dump_json(indent=2) + "\n")

        output = task_output(checks, self.review_threshold)
        with self._lock:
            self.report.documents += 1
            self.report.model = result.model
            self.report.input_tokens += result.input_tokens
            self.report.output_tokens += result.output_tokens
            if result.latency_ms:
                self.report.latency_ms.append(result.latency_ms)
            score_document(self.report, pdf_path.name, label, output)
        return output


def evaluate(
    dataset: Path,
    extractor: Extractor,
    review_threshold: float,
    buyer_names: list[str] | None = None,
    save_outputs: Path | None = None,
) -> Report:
    runner = Runner(extractor, review_threshold, buyer_names, save_outputs)
    for pdf_path in sorted(dataset.glob("*.pdf")):
        label = json.loads(pdf_path.with_suffix(".json").read_text())
        with tracing.span("evaluate_document", input={"file": pdf_path.as_posix()}) as span:
            span.update(output=runner.run(pdf_path, label))
    return runner.report


def task_output(checks: ValidationResult, review_threshold: float) -> dict[str, Any]:
    """Predicted values and what was sent to review: everything scoring needs, JSON-serialisable."""
    flagged = {f.path for f in checks.needs_review(review_threshold)}
    return {
        "data": checks.data(),
        "flagged": sorted(flagged),
        # a skipped line item has no field to flag; it is caught if the totals check sent the doc to review
        "lines_flagged": bool(checks.document_issues) or "subtotal" in flagged,
    }


def score_document(report: Report, doc: str, label: dict[str, Any], output: dict[str, Any]) -> None:
    predicted, flagged, lines_flagged = output["data"], set(output["flagged"]), output["lines_flagged"]
    for name in HEADER_FIELDS:
        _score(report, doc, name, name, label[name], predicted.get(name), name in flagged)
    pred_lines = predicted.get("line_items", [])
    for i, expected_line in enumerate(label["line_items"]):
        missing = i >= len(pred_lines)
        actual_line = {} if missing else pred_lines[i]
        for name in LINE_FIELDS:
            path = f"line_items[{i}].{name}"
            _score(
                report,
                doc,
                name,
                path,
                expected_line[name],
                actual_line.get(name),
                path in flagged or (missing and lines_flagged),
            )


# ---------- Langfuse datasets and experiments ----------


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def sync_dataset(name: str, dataset: Path) -> int:
    """Upsert the labelled documents of ``dataset`` into the Langfuse dataset ``name``, keyed by folder and
    file name: a label fix updates its item instead of duplicating it. Unchanged items are not rewritten,
    so parallel runs (one per model) on the same dataset don't conflict. Returns the number written."""
    lf = tracing.get_langfuse()
    if lf is None:
        raise SystemExit("Langfuse is not configured: set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL")
    tracing.ensure_dataset(name, "Labelled procurement documents from eval/; expected_output is the label")
    existing = {item.id: item for item in lf.get_dataset(name).items}
    written = 0
    for pdf_path in sorted(dataset.glob("*.pdf")):
        item_id = f"{name}-{dataset.name}-{pdf_path.stem}"
        fields = {
            "input": {"file": _rel(pdf_path), "sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest()},
            "expected_output": json.loads(pdf_path.with_suffix(".json").read_text()),
            "metadata": {"split": dataset.name},
        }
        old = existing.get(item_id)
        if old is not None and all(getattr(old, k) == v for k, v in fields.items()):
            continue
        lf.create_dataset_item(dataset_name=name, id=item_id, **fields)
        written += 1
    return written


def document_evaluations(
    *, input: dict[str, Any], output: dict[str, Any], expected_output: dict[str, Any], **_: Any
) -> list[Evaluation]:
    if "error" in output:
        return [Evaluation(name="extraction_failed", value=1, comment=output["error"])]
    r = Report()
    score_document(r, input["file"], expected_output, output)
    o = r.overall()
    missed = [e for e in r.errors if not e["flagged"]]
    return [
        Evaluation(name="field_accuracy", value=_ratio(o.correct, o.total)),
        Evaluation(name="review_rate", value=_ratio(o.flagged, o.total)),
        # wrong values that skipped review: the ones that reach the ERP
        Evaluation(
            name="missed_errors",
            value=len(missed),
            comment="; ".join(f"{e['field']}: expected {e['expected']!r}, got {e['actual']!r}" for e in missed) or None,
        ),
    ]


def evaluate_langfuse(
    dataset: Path, runner: Runner, dataset_name: str, run_name: str, concurrency: int
) -> tuple[Report, str | None]:
    lf = tracing.get_langfuse()
    assert lf is not None  # sync_dataset exits otherwise
    sync_dataset(dataset_name, dataset)
    items = [
        item
        for item in lf.get_dataset(dataset_name).items
        if (item.metadata or {}).get("split") == dataset.name and Path(item.input["file"]).exists()
    ]

    async def task(*, item: Any, **_: Any) -> dict[str, Any]:
        # to_thread copies the context, so the extraction spans nest under the experiment's item trace
        return await asyncio.to_thread(runner.run, Path(item.input["file"]), item.expected_output)

    def run_evaluations(**_: Any) -> list[Evaluation]:
        s = runner.report.summary()
        return [
            Evaluation(name=k, value=s[k])
            for k in ("field_accuracy", "auto_accepted_accuracy", "error_catch_rate", "review_rate")
        ]

    result = lf.run_experiment(
        name="procex-eval",
        run_name=run_name,
        description=f"{dataset.name}, review threshold {runner.review_threshold}",
        data=items,
        task=task,
        evaluators=[document_evaluations],
        run_evaluators=[run_evaluations],
        max_concurrency=concurrency,
        metadata={"split": dataset.name, "review_threshold": str(runner.review_threshold)},
    )
    lf.flush()
    return runner.report, result.experiment_url


def _score(report: Report, doc: str, name: str, path: str, expected: Any, actual: Any, is_flagged: bool) -> None:
    ok = values_match(name, expected, actual)
    report.per_field[name].add(ok, is_flagged)
    if not ok:
        report.errors.append(
            {"document": doc, "field": path, "expected": expected, "actual": actual, "flagged": is_flagged}
        )


def print_table(summary: dict[str, Any]) -> None:
    print(f"\nModel: {summary['model']}   documents: {summary['documents']}   fields: {summary['fields_scored']}")
    print(f"{'field':<16}{'accuracy':>10}{'review rate':>13}{'n':>6}")
    for name, s in summary["per_field"].items():
        print(f"{name:<16}{s['accuracy']:>10.1%}{s['review_rate']:>13.1%}{s['n']:>6}")
    print(f"\nfield accuracy          {summary['field_accuracy']:.1%}")
    print(f"auto-accepted accuracy  {summary['auto_accepted_accuracy']:.1%}   (values that skip human review)")
    print(f"error catch rate        {summary['error_catch_rate']:.1%}   (wrong values sent to review)")
    print(f"review rate             {summary['review_rate']:.1%}")
    if "cost_usd" in summary:
        print(f"cost                    ${summary['cost_usd']} (${summary['cost_per_document_usd']}/document)")
        print(f"mean latency            {summary['mean_latency_ms']} ms")
    for err in summary["errors"]:
        mark = "caught" if err["flagged"] else "MISSED"
        print(f"  [{mark}] {err['document']} {err['field']}: expected {err['expected']!r}, got {err['actual']!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("eval/dataset"))
    parser.add_argument("--extractor", choices=["claude", "claude-code", "fixture"], default="claude")
    parser.add_argument("--model", help="model for --extractor claude-code, e.g. opus or sonnet")
    parser.add_argument("--fixtures", type=Path, default=Path("eval/fixtures"))
    parser.add_argument("--out", type=Path, help="write the JSON report here")
    parser.add_argument("--min-auto-accepted-accuracy", type=float, help="exit 1 below this (CI gate)")
    parser.add_argument("--save-outputs", type=Path, help="write each raw model output to DIR/<doc>.json")
    parser.add_argument("--langfuse-dataset", help="sync to this Langfuse dataset and run as an experiment")
    parser.add_argument("--run-name", help="experiment run name (default: <model or extractor>-<timestamp>)")
    parser.add_argument("--concurrency", type=int, default=1, help="documents in parallel (Langfuse mode)")
    args = parser.parse_args(argv)

    settings = get_settings()
    extractor: Extractor
    if args.extractor == "claude":
        extractor = ClaudeExtractor(settings)
    elif args.extractor == "claude-code":
        extractor = ClaudeCodeExtractor(args.model or settings.claude_code_model)
    else:
        extractor = FixtureExtractor(args.fixtures)

    experiment_url = None
    if args.langfuse_dataset:
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        run_name = args.run_name or f"{args.dataset.name}-{args.model or args.extractor}-{stamp}"
        runner = Runner(extractor, settings.review_threshold, settings.buyer_names, args.save_outputs)
        report, experiment_url = evaluate_langfuse(
            args.dataset, runner, args.langfuse_dataset, run_name, args.concurrency
        )
    else:
        report = evaluate(args.dataset, extractor, settings.review_threshold, settings.buyer_names, args.save_outputs)
        tracing.flush()
    summary = report.summary()
    print_table(summary)
    if experiment_url:
        print(f"\nLangfuse experiment: {experiment_url}")
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
    gate = args.min_auto_accepted_accuracy
    if gate is not None and summary["auto_accepted_accuracy"] < gate:
        print(f"\nFAIL: auto-accepted accuracy below {gate:.1%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
