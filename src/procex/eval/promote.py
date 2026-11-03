"""Turn failures into regression cases.

    procex-promote eval --dataset eval/dataset-hard --outputs eval/results/hard-opus/outputs --source opus
    procex-promote langfuse --pdf-dir /path/to/original/pdfs

A case is the PDF, its correct label and the model output that got it wrong, stored in eval/regression/.
``tests/test_regression_cases.py`` replays every case offline and asserts the validator still sends each
wrong value to review. A case whose error the rules do not catch yet is recorded as a known gap and runs as
a strict xfail: it documents the gap, and starts failing (telling you to flip the flag) once a rule fixes it.

``eval`` promotes documents that a live evaluation got wrong (saved with ``procex-eval --save-outputs``).
``langfuse`` promotes production documents a reviewer had to correct: the API records them, linked to their
trace, in the Langfuse dataset ``PROCEX_LANGFUSE_CORRECTIONS_DATASET``. Reviewers correct fields, they
cannot add a line item the model skipped, so check the label of a promoted review case before committing it.
"""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from procex import tracing
from procex.config import get_settings
from procex.eval.harness import Report, score_document, task_output
from procex.pdf import extract_text
from procex.schema import ExtractedDocument
from procex.validation import validate

REGRESSION_DIR = Path("eval/regression")


def replay(
    pdf: bytes, label: dict[str, Any], output: dict[str, Any], buyer_names: list[str], threshold: float
) -> list[dict[str, Any]]:
    """The wrong values in a recorded model output, each with whether validation sends it to review."""
    checks = validate(ExtractedDocument.model_validate(output), extract_text(pdf), buyer_names)
    report = Report()
    score_document(report, "case", label, task_output(checks, threshold))
    return report.errors


def write_case(
    root: Path, case_id: str, pdf: bytes, label: dict[str, Any], output: dict[str, Any], meta: dict[str, Any]
) -> dict[str, Any] | None:
    settings = get_settings()
    errors = replay(pdf, label, output, settings.buyer_names, settings.review_threshold)
    if not errors:
        return None  # the output matches the label: nothing to regress on
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    (root / f"{case_id}.pdf").write_bytes(pdf)
    (root / f"{case_id}.json").write_text(json.dumps(label, indent=2) + "\n")
    (root / "outputs" / f"{case_id}.json").write_text(json.dumps(output, indent=2) + "\n")

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    entry = {
        **meta,
        "promoted_at": datetime.now(UTC).date().isoformat(),
        "buyer_names": settings.buyer_names,
        "review_threshold": settings.review_threshold,
        "errors": [
            {"field": e["field"], "expected": e["expected"], "actual": e["actual"], "caught": e["flagged"]}
            for e in errors
        ],
        "known_gap": any(not e["flagged"] for e in errors),
    }
    manifest[case_id] = entry
    manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=2) + "\n")
    return entry


def promote_eval(args: argparse.Namespace) -> int:
    promoted = 0
    for output_path in sorted(args.outputs.glob("*.json")):
        stem = output_path.stem
        if args.only and stem not in args.only:
            continue
        pdf_path = args.dataset / f"{stem}.pdf"
        if not pdf_path.exists():
            print(f"skip {stem}: {pdf_path} not found")
            continue
        case_id = f"{args.dataset.name}-{stem}-{args.source}"
        entry = write_case(
            args.root,
            case_id,
            pdf_path.read_bytes(),
            json.loads(pdf_path.with_suffix(".json").read_text()),
            json.loads(output_path.read_text()),
            {"source": f"eval:{args.source}", "origin": pdf_path.as_posix()},
        )
        promoted += _report(case_id, entry)
    print(f"\n{promoted} case(s) written to {args.root}")
    return 0


def promote_langfuse(args: argparse.Namespace) -> int:
    lf = tracing.get_langfuse()
    if lf is None:
        print("Langfuse is not configured: set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL")
        return 2
    pdfs = {hashlib.sha256(p.read_bytes()).hexdigest(): p for p in args.pdf_dir.glob("*.pdf")}
    promoted = 0
    for item in lf.get_dataset(args.dataset_name).items:
        sha = item.input["sha256"]
        pdf_path = pdfs.get(sha)
        if pdf_path is None:
            print(f"skip {item.input['filename']}: no PDF with sha256 {sha[:12]}… in {args.pdf_dir}")
            continue
        meta = item.metadata or {}
        case_id = f"review-{sha[:12]}"
        entry = write_case(
            args.root,
            case_id,
            pdf_path.read_bytes(),
            item.expected_output,
            meta["raw_output"],
            {
                "source": f"review:{meta.get('model')}",
                "origin": item.input["filename"],
                "trace_id": item.source_trace_id,
                "corrections": meta.get("corrections", []),
            },
        )
        promoted += _report(case_id, entry)
    print(f"\n{promoted} case(s) written to {args.root}; check each label before committing")
    return 0


def _report(case_id: str, entry: dict[str, Any] | None) -> int:
    if entry is None:
        print(f"{case_id}: output matches the label, not promoted")
        return 0
    gap = "  <- KNOWN GAP: validation does not catch this yet" if entry["known_gap"] else ""
    print(f"{case_id}{gap}")
    for e in entry["errors"]:
        mark = "caught" if e["caught"] else "MISSED"
        print(f"  [{mark}] {e['field']}: expected {e['expected']!r}, got {e['actual']!r}")
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REGRESSION_DIR, help="where cases are written")
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("eval", help="promote documents a live evaluation got wrong")
    ev.add_argument("--dataset", type=Path, required=True, help="labelled set the outputs belong to")
    ev.add_argument("--outputs", type=Path, required=True, help="directory written by procex-eval --save-outputs")
    ev.add_argument("--source", required=True, help="short model/run name, part of the case id (e.g. opus)")
    ev.add_argument("--only", nargs="*", help="document stems to promote (default: every failure)")

    lf = sub.add_parser("langfuse", help="promote production documents a reviewer corrected")
    lf.add_argument("--pdf-dir", type=Path, required=True, help="where the original PDFs are (matched by hash)")
    lf.add_argument("--dataset-name", default=get_settings().langfuse_corrections_dataset)

    args = parser.parse_args(argv)
    return promote_eval(args) if args.command == "eval" else promote_langfuse(args)


if __name__ == "__main__":
    sys.exit(main())
