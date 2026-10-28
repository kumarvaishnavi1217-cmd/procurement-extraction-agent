"""Field-level evaluation against a labelled document set.

    procex-eval --dataset eval/dataset --extractor claude          # live, needs ANTHROPIC_API_KEY
    procex-eval --dataset eval/dataset --extractor claude-code     # live via `claude -p`, Claude subscription
    procex-eval --dataset eval/dataset --extractor fixture --fixtures eval/fixtures

Besides raw accuracy it measures the review queue itself: how many wrong values it catches
(error catch rate) and how accurate the values are that skip review (auto-accepted accuracy) -
the number that matters when tuning the review threshold, prompt, model or effort.
"""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from procex.config import get_settings
from procex.extractor import ClaudeCodeExtractor, ClaudeExtractor, ExtractionError, Extractor, FixtureExtractor
from procex.pdf import extract_text, normalise
from procex.validation import validate

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


def evaluate(
    dataset: Path, extractor: Extractor, review_threshold: float, buyer_names: list[str] | None = None
) -> Report:
    report = Report()
    for pdf_path in sorted(dataset.glob("*.pdf")):
        label = json.loads(pdf_path.with_suffix(".json").read_text())
        data = pdf_path.read_bytes()
        report.documents += 1
        try:
            result = extractor.extract(data, pdf_path.name)
        except ExtractionError as e:
            report.failed_documents.append(f"{pdf_path.name}: {e}")
            continue
        report.model = result.model
        report.input_tokens += result.input_tokens
        report.output_tokens += result.output_tokens
        if result.latency_ms:
            report.latency_ms.append(result.latency_ms)

        checks = validate(result.document, extract_text(data), buyer_names)
        flagged = {f.path for f in checks.needs_review(review_threshold)}
        predicted = checks.data()
        # a skipped line item has no field to flag; it is caught if the totals check sent the doc to review
        lines_flagged = bool(checks.document_issues) or "subtotal" in flagged

        for name in HEADER_FIELDS:
            _score(report, pdf_path.name, name, name, label[name], predicted.get(name), name in flagged)
        pred_lines = predicted.get("line_items", [])
        for i, expected_line in enumerate(label["line_items"]):
            missing = i >= len(pred_lines)
            actual_line = {} if missing else pred_lines[i]
            for name in LINE_FIELDS:
                path = f"line_items[{i}].{name}"
                _score(
                    report,
                    pdf_path.name,
                    name,
                    path,
                    expected_line[name],
                    actual_line.get(name),
                    path in flagged or (missing and lines_flagged),
                )
    return report


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
    args = parser.parse_args(argv)

    settings = get_settings()
    extractor: Extractor
    if args.extractor == "claude":
        extractor = ClaudeExtractor(settings)
    elif args.extractor == "claude-code":
        extractor = ClaudeCodeExtractor(args.model or settings.claude_code_model)
    else:
        extractor = FixtureExtractor(args.fixtures)
    summary = evaluate(args.dataset, extractor, settings.review_threshold, settings.buyer_names).summary()
    print_table(summary)
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2))
    gate = args.min_auto_accepted_accuracy
    if gate is not None and summary["auto_accepted_accuracy"] < gate:
        print(f"\nFAIL: auto-accepted accuracy below {gate:.1%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
