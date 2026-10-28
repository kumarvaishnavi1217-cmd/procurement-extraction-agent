import json

import pytest

from procex.pdf import extract_text
from procex.schema import ExtractedDocument
from procex.validation import ERROR_CAP, parse_number_candidates, set_path, validate
from tests.conftest import BUYER, FIXTURES, pdf


def load(doc_id: str) -> tuple[ExtractedDocument, str]:
    return ExtractedDocument.model_validate_json((FIXTURES / f"{doc_id}.json").read_text()), extract_text(pdf(doc_id))


def flagged(doc_id: str, threshold: float = 0.8) -> dict[str, list[str]]:
    doc, text = load(doc_id)
    return {f.path: f.issues for f in validate(doc, text, [BUYER]).needs_review(threshold)}


@pytest.mark.parametrize("doc_id", ["doc01", "doc03", "doc05", "doc07", "doc09"])
def test_correct_extractions_pass_without_review(doc_id: str) -> None:
    assert flagged(doc_id) == {}


def test_misread_quantity_breaks_line_arithmetic() -> None:
    issues = flagged("doc02")
    assert "line_arithmetic_mismatch" in issues["line_items[0].quantity"]
    assert "value_not_in_evidence" in issues["line_items[0].quantity"]
    assert "line_arithmetic_mismatch" in issues["line_items[0].line_total"]


def test_buyer_taken_as_supplier_is_flagged() -> None:
    assert flagged("doc04")["supplier_name"] == ["supplier_is_buyer"]


def test_po_number_taken_as_document_number_is_flagged() -> None:
    assert flagged("doc08")["document_number"] == ["looks_like_po_number"]


def test_hallucinated_price_is_not_grounded() -> None:
    assert "evidence_not_in_document" in flagged("doc10")["line_items[1].unit_price"]


def test_skipped_line_item_breaks_subtotal() -> None:
    assert flagged("doc11")["subtotal"] == ["subtotal_mismatch"]


def test_low_model_confidence_alone_triggers_review() -> None:
    assert flagged("doc12") == {"line_items[0].part_number": []}


def test_scanned_pdf_without_text_layer_is_always_reviewed() -> None:
    doc, _ = load("doc01")
    result = validate(doc, pdf_text="")
    assert all(
        f.confidence <= 0.75 for f in result.fields.values() if f.value is not None and f.path != "document_type"
    )


def test_missing_required_fields_and_bad_formats() -> None:
    doc, text = load("doc01")
    raw = doc.model_dump()
    raw["supplier_name"]["value"] = None
    raw["currency"].update(value="EURO")
    raw["document_date"].update(value="14/03/2026")
    raw["line_items"][0]["part_number"].update(value="bolt with spaces", evidence="bolt with spaces")
    raw["line_items"] = raw["line_items"][:1]
    result = validate(ExtractedDocument.model_validate(raw), text)
    f = result.fields
    assert "missing_required" in f["supplier_name"].issues
    assert "unknown_currency" in f["currency"].issues
    assert "invalid_date" in f["document_date"].issues
    assert "invalid_part_number" in f["line_items[0].part_number"].issues
    assert f["supplier_name"].confidence == ERROR_CAP


def test_no_line_items_is_a_document_issue() -> None:
    doc, text = load("doc01")
    raw = doc.model_dump()
    raw["line_items"] = []
    assert validate(ExtractedDocument.model_validate(raw), text).document_issues == ["no_line_items"]


@pytest.mark.parametrize(
    ("printed", "expected"),
    [("1,234.50", 1234.5), ("1.234,50", 1234.5), ("1'234.50", 1234.5), ("0.42", 0.42), ("250", 250.0)],
)
def test_number_formats(printed: str, expected: float) -> None:
    assert expected in parse_number_candidates(printed)


def test_set_path_builds_nested_structure() -> None:
    data: dict[str, object] = {}
    set_path(data, "line_items[1].quantity", 5)
    set_path(data, "supplier_name", "Acme")
    assert data == {"line_items": [{}, {"quantity": 5}], "supplier_name": "Acme"}


def test_fixture_files_are_valid_schema() -> None:
    for path in FIXTURES.glob("*.json"):
        ExtractedDocument.model_validate(json.loads(path.read_text()))
