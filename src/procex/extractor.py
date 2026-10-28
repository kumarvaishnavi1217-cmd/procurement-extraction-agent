"""LLM extraction step. ``ClaudeExtractor`` is the production implementation; tests and the offline
evaluation use ``FixtureExtractor`` so they run without an API key."""

import base64
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import anthropic

from procex.config import Settings
from procex.schema import ExtractedDocument

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You extract structured procurement data from supplier quotes and invoices for a purchasing team.

The extracted values are booked into an ERP system, so a wrong value is worse than a missing one:
- Only extract what is printed in the document. If a field is absent or illegible, return null and a low \
confidence rather than guessing.
- For every field, copy the supporting text verbatim into `evidence`, exactly as printed (same digits, \
separators and spelling). Evidence is checked against the document automatically.
- `supplier_name` is the company issuing the document, not the customer it is addressed to.
- Numbers: return plain decimals (1234.5) regardless of the printed format ("1.234,50", "1,234.50", "$1,234.50").
- Dates: normalise to YYYY-MM-DD. If the day/month order is ambiguous, pick the reading consistent with the \
document's locale and lower the confidence.
- Include every line item, in document order. Skip pure text rows such as "Shipping notes".
- Confidence reflects how sure you are the value is correct: 0.95+ only when it is printed clearly and \
unambiguously; 0.5-0.8 when you had to interpret layout, OCR noise or ambiguity; below 0.5 when guessing.
"""

USER_PROMPT = "Extract the procurement data from this document."


class ExtractionError(RuntimeError):
    """The model could not produce a usable extraction (refusal, truncation, invalid output)."""


@dataclass(frozen=True)
class ExtractionResult:
    document: ExtractedDocument
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    request_id: str | None = None


class Extractor(Protocol):
    def extract(self, pdf_bytes: bytes, filename: str) -> ExtractionResult: ...


class ClaudeExtractor:
    def __init__(self, settings: Settings, client: anthropic.Anthropic | None = None) -> None:
        self._settings = settings
        self._client = client  # created on first use, so missing credentials surface as a failed document

    def extract(self, pdf_bytes: bytes, filename: str) -> ExtractionResult:
        if self._client is None:
            self._client = anthropic.Anthropic()
        started = time.perf_counter()
        try:
            response = self._client.beta.messages.parse(
                model=self._settings.model,
                max_tokens=self._settings.max_tokens,
                output_config={"effort": self._settings.effort},
                # if a safety classifier declines, the API retries on a fallback model in the same call
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": base64.standard_b64encode(pdf_bytes).decode("ascii"),
                                },
                                "title": filename,
                            },
                            {"type": "text", "text": USER_PROMPT},
                        ],
                    }
                ],
                output_format=ExtractedDocument,
            )
        except TypeError as e:
            # the SDK raises TypeError when no API key / auth token / profile is configured
            raise ExtractionError(f"Anthropic client is not configured: {e}") from e
        latency_ms = int((time.perf_counter() - started) * 1000)

        if response.stop_reason == "refusal":
            category = response.stop_details.category if response.stop_details else None
            raise ExtractionError(f"Model declined to process the document (category={category})")
        if response.stop_reason == "max_tokens":
            raise ExtractionError("Extraction truncated at max_tokens; raise PROCEX_MAX_TOKENS")
        if response.parsed_output is None:
            raise ExtractionError("Model returned no structured output")

        log.info(
            "llm_extraction",
            extra={
                "model": response.model,
                "request_id": response._request_id,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "latency_ms": latency_ms,
            },
        )
        return ExtractionResult(
            document=response.parsed_output,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            request_id=response._request_id,
        )


class FixtureExtractor:
    """Replays stored model outputs keyed by file name (``<stem>.json``) for offline tests and evals."""

    def __init__(self, fixtures_dir: Path) -> None:
        self._dir = fixtures_dir

    def extract(self, pdf_bytes: bytes, filename: str) -> ExtractionResult:
        path = self._dir / f"{Path(filename).stem}.json"
        if not path.exists():
            raise ExtractionError(f"No fixture for {filename}")
        return ExtractionResult(document=ExtractedDocument.model_validate_json(path.read_text()), model="fixture")
