"""REST API against a real PostgreSQL (Testcontainers), with the LLM replaced by stored fixtures."""

from fastapi.testclient import TestClient

from procex.api import app, get_extractor
from procex.extractor import ExtractionError, ExtractionResult
from tests.conftest import pdf


def upload(client: TestClient, doc_id: str, content: bytes | None = None, name: str | None = None):  # type: ignore[no-untyped-def]
    files = {"file": (name or f"{doc_id}.pdf", content if content is not None else pdf(doc_id), "application/pdf")}
    return client.post("/documents", files=files)


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_clean_document_is_auto_approved(client: TestClient) -> None:
    r = upload(client, "doc01")
    assert r.status_code == 201
    body = r.json()
    assert body["status"] == "auto_approved"
    assert body["open_review_tasks"] == []
    assert body["data"]["supplier_name"] == "Nordwerk Industrietechnik GmbH"
    assert body["data"]["line_items"][0]["part_number"]
    assert r.headers["X-Request-ID"]


def test_duplicate_upload_returns_existing_document(client: TestClient) -> None:
    first = upload(client, "doc01").json()
    again = upload(client, "doc01")
    assert again.status_code == 200
    assert again.json()["id"] == first["id"]
    assert len(client.get("/documents").json()) == 1


def test_low_confidence_fields_go_to_review_and_corrections_are_applied(client: TestClient) -> None:
    doc = upload(client, "doc02").json()
    assert doc["status"] == "needs_review"
    tasks = client.get("/review-tasks").json()
    paths = {t["field_path"] for t in tasks}
    assert {"line_items[0].quantity", "line_items[0].line_total", "line_items[0].unit_price"} <= paths

    # reviewer fixes the quantity and confirms the other two flagged values as extracted
    for task in tasks:
        value = 10 if task["field_path"] == "line_items[0].quantity" else task["extracted_value"]
        r = client.post(f"/review-tasks/{task['id']}/resolve", json={"value": value, "reviewer": "anna"})
        assert r.status_code == 200
        assert r.json()["status"] == "resolved"

    final = client.get(f"/documents/{doc['id']}").json()
    assert final["status"] == "reviewed"
    assert final["data"]["line_items"][0]["quantity"] == 10
    assert client.get("/review-tasks").json() == []
    assert len(client.get("/review-tasks", params={"status": "resolved"}).json()) == len(tasks)


def test_review_queue_is_ordered_least_confident_first(client: TestClient) -> None:
    upload(client, "doc06")
    upload(client, "doc12")
    confidences = [t["confidence"] for t in client.get("/review-tasks").json()]
    assert confidences == sorted(confidences)


def test_numeric_correction_must_be_a_number(client: TestClient) -> None:
    upload(client, "doc02")
    task = next(t for t in client.get("/review-tasks").json() if t["field_path"].endswith("quantity"))
    r = client.post(f"/review-tasks/{task['id']}/resolve", json={"value": "ten", "reviewer": "anna"})
    assert r.status_code == 409


def test_resolving_twice_is_rejected(client: TestClient) -> None:
    upload(client, "doc12")
    task = client.get("/review-tasks").json()[0]
    body = {"value": task["extracted_value"], "reviewer": "anna"}
    assert client.post(f"/review-tasks/{task['id']}/resolve", json=body).status_code == 200
    assert client.post(f"/review-tasks/{task['id']}/resolve", json=body).status_code == 409


def test_filter_documents_by_status(client: TestClient) -> None:
    upload(client, "doc01")
    upload(client, "doc02")
    review = client.get("/documents", params={"status": "needs_review"}).json()
    assert [d["filename"] for d in review] == ["doc02.pdf"]


def test_non_pdf_is_rejected(client: TestClient) -> None:
    assert upload(client, "x", content=b"MZ\x90\x00 not a pdf", name="invoice.pdf").status_code == 415


def test_corrupt_pdf_is_rejected(client: TestClient) -> None:
    assert upload(client, "x", content=b"%PDF-1.4 garbage", name="broken.pdf").status_code == 422


def test_unknown_ids_are_404(client: TestClient) -> None:
    missing = "00000000-0000-0000-0000-000000000000"
    assert client.get(f"/documents/{missing}").status_code == 404
    assert client.post(f"/review-tasks/{missing}/resolve", json={"value": 1, "reviewer": "a"}).status_code == 404


def test_extraction_failure_is_recorded(client: TestClient) -> None:
    class FailingExtractor:
        def extract(self, pdf_bytes: bytes, filename: str) -> ExtractionResult:
            raise ExtractionError("Model declined to process the document (category=None)")

    previous = app.dependency_overrides[get_extractor]
    app.dependency_overrides[get_extractor] = FailingExtractor
    try:
        body = upload(client, "doc03").json()
    finally:
        app.dependency_overrides[get_extractor] = previous
    assert body["status"] == "failed"
    assert "declined" in body["error"]
    assert body["data"] is None
