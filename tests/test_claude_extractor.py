"""The Claude request and response handling, against a stub client (no network, no API key)."""

import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from procex.config import Settings
from procex.extractor import ClaudeExtractor, ExtractionError
from procex.schema import ExtractedDocument
from tests.conftest import FIXTURES, pdf


class StubClient:
    def __init__(self, response: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._parse))
        self._response = response

    def _parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._response


def response(stop_reason: str = "end_turn", parsed: ExtractedDocument | None = None, **extra: Any) -> Any:
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=extra.get("stop_details"),
        parsed_output=parsed,
        model="claude-opus-5-5",
        usage=SimpleNamespace(input_tokens=3100, output_tokens=850),
        _request_id="req_test",
    )


def fixture_doc() -> ExtractedDocument:
    return ExtractedDocument.model_validate_json((FIXTURES / "doc01.json").read_text())


def test_sends_pdf_as_document_block_with_structured_output() -> None:
    stub = StubClient(response(parsed=fixture_doc()))
    data = pdf("doc01")
    result = ClaudeExtractor(Settings(effort="high"), client=stub).extract(data, "doc01.pdf")  # type: ignore[arg-type]

    call = stub.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_format"] is ExtractedDocument
    assert call["output_config"] == {"effort": "high"}
    assert call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert "temperature" not in call and "thinking" not in call
    document_block = call["messages"][0]["content"][0]
    assert document_block["type"] == "document"
    assert base64.b64decode(document_block["source"]["data"]) == data

    assert result.document == fixture_doc()
    assert (result.input_tokens, result.output_tokens, result.request_id) == (3100, 850, "req_test")


def test_refusal_raises() -> None:
    stub = StubClient(response("refusal", stop_details=SimpleNamespace(category="cyber")))
    with pytest.raises(ExtractionError, match="declined"):
        ClaudeExtractor(Settings(), client=stub).extract(b"%PDF-1.4", "x.pdf")  # type: ignore[arg-type]


def test_truncation_raises() -> None:
    stub = StubClient(response("max_tokens"))
    with pytest.raises(ExtractionError, match="max_tokens"):
        ClaudeExtractor(Settings(), client=stub).extract(b"%PDF-1.4", "x.pdf")  # type: ignore[arg-type]


def test_missing_structured_output_raises() -> None:
    stub = StubClient(response(parsed=None))
    with pytest.raises(ExtractionError, match="no structured output"):
        ClaudeExtractor(Settings(), client=stub).extract(b"%PDF-1.4", "x.pdf")  # type: ignore[arg-type]


def test_missing_credentials_become_extraction_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_CONFIG_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no `ant auth login` profile either, as in a fresh container
    with pytest.raises(ExtractionError, match="not configured"):
        ClaudeExtractor(Settings()).extract(pdf("doc01"), "doc01.pdf")
