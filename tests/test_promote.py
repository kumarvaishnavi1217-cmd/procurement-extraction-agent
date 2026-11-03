"""procex-promote: a failure becomes a regression case, and an error no rule catches becomes a known gap."""

import copy
import json
from pathlib import Path
from typing import Any

from procex.eval.promote import write_case
from tests.conftest import ROOT

HARD = ROOT / "eval" / "dataset-hard"
# Opus's real (correct) output for deposit-2, recorded by `procex-eval --save-outputs`
OPUS_OUTPUT = ROOT / "eval" / "results" / "hard-opus" / "outputs" / "deposit-2.json"


def deposit_case() -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    return (
        (HARD / "deposit-2.pdf").read_bytes(),
        json.loads((HARD / "deposit-2.json").read_text()),
        json.loads(OPUS_OUTPUT.read_text()),
    )


def test_correct_output_is_not_promoted(tmp_path: Path) -> None:
    pdf, label, output = deposit_case()
    assert write_case(tmp_path, "c", pdf, label, output, {"source": "test"}) is None
    assert not (tmp_path / "manifest.json").exists()


def test_uncaught_error_is_recorded_as_known_gap(tmp_path: Path) -> None:
    pdf, label, output = deposit_case()
    # Synthetic mistake: the balance due (after a 10% deposit) taken as the invoice total. It is printed in
    # the document and still larger than the subtotal, so no current rule catches it.
    wrong = copy.deepcopy(output)
    wrong["total"] = {"value": 26126.86, "confidence": 0.95, "evidence": "EUR 26.126,86"}

    entry = write_case(tmp_path, "deposit-gap", pdf, label, wrong, {"source": "test"})

    assert entry is not None and entry["known_gap"]
    assert entry["errors"] == [{"field": "total", "expected": 29029.85, "actual": 26126.86, "caught": False}]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["deposit-gap"]["known_gap"]
    for path in ("deposit-gap.pdf", "deposit-gap.json", "outputs/deposit-gap.json"):
        assert (tmp_path / path).exists()


def test_caught_error_is_not_a_gap(tmp_path: Path) -> None:
    pdf, label, output = deposit_case()
    wrong = copy.deepcopy(output)
    wrong["line_items"][0]["quantity"]["value"] *= 10  # breaks qty x price = line total

    entry = write_case(tmp_path, "deposit-qty", pdf, label, wrong, {"source": "test"})

    assert entry is not None and not entry["known_gap"]
    assert all(e["caught"] for e in entry["errors"])
