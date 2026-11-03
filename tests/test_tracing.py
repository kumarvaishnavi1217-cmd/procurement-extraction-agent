"""Langfuse tracing, against an in-memory span exporter: no Langfuse server needed."""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from procex import tracing
from tests.conftest import pdf

TYPE = "langfuse.observation.type"
LEVEL = "langfuse.observation.level"


class Recorder:
    def __init__(self) -> None:
        self.exporter = InMemorySpanExporter()
        self.client = Langfuse(
            public_key="pk-lf-test",
            secret_key="sk-lf-test",  # noqa: S106 - dummy key, nothing is sent
            base_url="http://localhost:9",  # nothing listens; spans go to the in-memory exporter
            span_exporter=self.exporter,
            tracer_provider=TracerProvider(),
            flush_at=1,
        )
        self.scores: list[dict[str, Any]] = []
        self.dataset_items: list[dict[str, Any]] = []
        self.client.create_score = lambda **kw: self.scores.append(kw)  # type: ignore[method-assign]
        self.client.create_dataset_item = lambda **kw: self.dataset_items.append(kw)  # type: ignore[method-assign]
        self.client.get_dataset = lambda *a, **kw: None  # type: ignore[method-assign]

    def spans(self) -> dict[str, Any]:
        self.client.flush()
        return {s.name: s for s in self.exporter.get_finished_spans()}


@pytest.fixture
def recorder() -> Iterator[Recorder]:
    rec = Recorder()
    tracing.use_client(rec.client)
    yield rec
    tracing.use_client(None)


def test_tracing_is_a_noop_without_keys(client: TestClient) -> None:
    doc = client.post("/documents", files={"file": ("doc01.pdf", pdf("doc01"), "application/pdf")}).json()
    assert doc["trace_id"] is None


def test_extraction_is_traced(client: TestClient, recorder: Recorder) -> None:
    doc = client.post("/documents", files={"file": ("doc02.pdf", pdf("doc02"), "application/pdf")}).json()
    spans = recorder.spans()

    root, gen, guard = spans["process_document"], spans["extract"], spans["validate"]
    assert gen.attributes is not None and guard.attributes is not None
    assert gen.attributes[TYPE] == "generation"
    assert gen.attributes["langfuse.observation.model.name"] == "fixture"
    assert guard.attributes[TYPE] == "guardrail"
    assert guard.attributes[LEVEL] == "WARNING"  # doc02's misread quantity is flagged
    assert gen.parent is not None and guard.parent is not None
    assert gen.parent.span_id == root.context.span_id == guard.parent.span_id
    assert doc["trace_id"] == format(root.context.trace_id, "032x")


def test_review_scores_the_trace_and_records_the_correction(client: TestClient, recorder: Recorder) -> None:
    doc = client.post("/documents", files={"file": ("doc02.pdf", pdf("doc02"), "application/pdf")}).json()
    tasks = {t["field_path"]: t for t in doc["open_review_tasks"]}
    for path, task in tasks.items():
        value = 10 if path == "line_items[0].quantity" else task["extracted_value"]
        client.post(f"/review-tasks/{task['id']}/resolve", json={"value": value, "reviewer": "anna"})

    scores = {s["name"]: s for s in recorder.scores if s["name"] != "human_review"}
    assert scores["needs_review"]["value"] == 1
    assert scores["flagged_fields"]["value"] == len(tasks)
    reviews = [s for s in recorder.scores if s["name"] == "human_review"]
    assert sorted(s["value"] for s in reviews) == ["confirmed"] * (len(tasks) - 1) + ["corrected"]
    assert all(s["trace_id"] == doc["trace_id"] for s in reviews)

    tracing.flush()  # the dataset item is sent in the background
    [item] = recorder.dataset_items  # one production failure, once all its tasks are resolved
    assert item["source_trace_id"] == doc["trace_id"]
    assert item["metadata"]["corrections"] == [{"field": "line_items[0].quantity", "extracted": 1.0, "corrected": 10}]
    assert item["expected_output"]["line_items"][0]["quantity"] == 10
