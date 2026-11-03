"""Every failure promoted with `procex-promote` stays a test: replaying the recorded model output must still
send each wrong value to review. Known gaps (errors no rule catches yet) run as strict xfails, so fixing one
fails this test until its ``known_gap`` flag in eval/regression/manifest.json is set to false.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from procex.eval.promote import replay
from tests.conftest import ROOT

REGRESSION = ROOT / "eval" / "regression"
MANIFEST = REGRESSION / "manifest.json"
CASES: dict[str, dict[str, Any]] = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def _params() -> list[Any]:
    params = []
    for case_id, meta in sorted(CASES.items()):
        missed = [e["field"] for e in meta["errors"] if not e["caught"]]
        marks = [pytest.mark.xfail(strict=True, reason=f"known gap: {', '.join(missed)}")] if meta["known_gap"] else []
        params.append(pytest.param(case_id, marks=marks, id=case_id))
    return params


def _replay(case_id: str) -> list[dict[str, Any]]:
    meta = CASES[case_id]
    return replay(
        (REGRESSION / f"{case_id}.pdf").read_bytes(),
        json.loads((REGRESSION / f"{case_id}.json").read_text()),
        json.loads((REGRESSION / "outputs" / f"{case_id}.json").read_text()),
        meta["buyer_names"],
        meta["review_threshold"],
    )


@pytest.mark.parametrize("case_id", _params())
def test_failure_is_sent_to_review(case_id: str) -> None:
    errors = _replay(case_id)
    assert errors, "the recorded output matches the label; this case no longer reproduces a failure"
    missed = [f"{e['field']}: expected {e['expected']!r}, got {e['actual']!r}" for e in errors if not e["flagged"]]
    assert not missed, "wrong values that would skip review: " + "; ".join(missed)


def test_manifest_matches_files() -> None:
    for case_id in CASES:
        for path in (REGRESSION / f"{case_id}.pdf", REGRESSION / f"{case_id}.json"):
            assert path.exists(), path
        assert (REGRESSION / "outputs" / f"{case_id}.json").exists()
    assert {p.stem for p in Path(REGRESSION).glob("*.pdf")} == set(CASES) if REGRESSION.exists() else True
