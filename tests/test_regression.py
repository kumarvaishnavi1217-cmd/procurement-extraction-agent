"""Regression gate over the labelled set: fails if a change to prompts, rules or thresholds alters which
fields are flagged, or lets accuracy of auto-accepted values drop.

Regenerate the snapshot after an intentional change:  UPDATE_SNAPSHOTS=1 pytest tests/test_regression.py
"""

import json
import os
from pathlib import Path

from procex.eval.harness import evaluate, values_match
from procex.extractor import FixtureExtractor
from procex.pdf import extract_text
from procex.schema import ExtractedDocument
from procex.validation import validate
from tests.conftest import BUYER, DATASET, FIXTURES

SNAPSHOT = Path(__file__).parent / "snapshots" / "flagged_fields.json"


def current_flags() -> dict[str, list[str]]:
    flags = {}
    for pdf_path in sorted(DATASET.glob("*.pdf")):
        doc = ExtractedDocument.model_validate_json((FIXTURES / f"{pdf_path.stem}.json").read_text())
        result = validate(doc, extract_text(pdf_path.read_bytes()), [BUYER])
        flags[pdf_path.stem] = sorted(f.path for f in result.needs_review(0.8))
    return flags


def test_flagged_fields_match_snapshot() -> None:
    flags = current_flags()
    if os.environ.get("UPDATE_SNAPSHOTS") or not SNAPSHOT.exists():
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(json.dumps(flags, indent=2) + "\n")
    assert flags == json.loads(SNAPSHOT.read_text())


def test_eval_quality_gate() -> None:
    summary = evaluate(DATASET, FixtureExtractor(FIXTURES), 0.8, [BUYER]).summary()
    assert summary["documents"] == 12
    assert summary["auto_accepted_accuracy"] == 1.0
    assert summary["error_catch_rate"] == 1.0
    assert summary["review_rate"] < 0.10


def test_value_matching_rules() -> None:
    assert values_match("total", 1234.5, 1234.50)
    assert not values_match("total", 1234.5, 1234.6)
    assert values_match("supplier_name", "Apex Fasteners Ltd", " apex  fasteners ltd. ")
    assert not values_match("part_number", "HX-M8-25-ZN", None)
    assert values_match("part_number", None, None)
