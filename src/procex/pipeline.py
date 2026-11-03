"""PDF -> LLM extraction -> validation -> persisted result + review queue."""

import copy
import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import anthropic
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from procex import tracing
from procex.config import Settings, get_settings
from procex.extractor import USER_PROMPT, ExtractionError, ExtractionResult, Extractor
from procex.models import Document, DocumentStatus, Extraction, ReviewTask, TaskStatus, utcnow
from procex.pdf import extract_text
from procex.validation import FieldResult, ValidationResult, set_path, validate

log = logging.getLogger(__name__)

NUMERIC_FIELDS = ("quantity", "unit_price", "line_total", "subtotal", "total")
LABEL_LINE_FIELDS = ("part_number", "quantity", "unit_price", "line_total")


@dataclass(frozen=True)
class Outcome:
    result: ExtractionResult
    checks: ValidationResult
    flagged: list[FieldResult]

    @property
    def needs_review(self) -> bool:
        return bool(self.flagged or self.checks.document_issues)


def extract_and_validate(
    extractor: Extractor,
    data: bytes,
    filename: str,
    pdf_text: str,
    review_threshold: float,
    buyer_names: list[str] | None,
) -> Outcome:
    """The core shared by the API and the evaluation harness, traced as one generation (the model call)
    and one guardrail (the validation rules) under whatever span the caller has open."""
    with tracing.generation("extract", input={"filename": filename, "prompt": USER_PROMPT}) as gen:
        try:
            result = extractor.extract(data, filename)
        except Exception as e:
            gen.update(level="ERROR", status_message=f"{type(e).__name__}: {e}")
            raise
        gen.update(
            model=result.model,
            output=result.document.model_dump(mode="json"),
            usage_details={"input": result.input_tokens, "output": result.output_tokens},
            metadata={"llm_request_id": result.request_id, "latency_ms": result.latency_ms},
        )

    with tracing.guardrail(
        "validate", input={"text_layer_chars": len(pdf_text), "review_threshold": review_threshold}
    ) as guard:
        checks = validate(result.document, pdf_text, buyer_names)
        outcome = Outcome(result, checks, checks.needs_review(review_threshold))
        guard.update(
            output={
                "needs_review": outcome.needs_review,
                "flagged": {f.path: f.issues or ["low_model_confidence"] for f in outcome.flagged},
                "document_issues": checks.document_issues,
            },
            level="WARNING" if outcome.needs_review else "DEFAULT",
        )
    return outcome


def process_document(
    session: Session, extractor: Extractor, settings: Settings, filename: str, data: bytes
) -> tuple[Document, bool]:
    """Returns the document and whether it was newly created (identical files are processed once)."""
    sha256 = hashlib.sha256(data).hexdigest()
    existing = session.scalar(select(Document).where(Document.sha256 == sha256))
    if existing is not None:
        log.info("duplicate_upload", extra={"document_id": str(existing.id)})
        return existing, False

    pdf_text = extract_text(data)  # raises InvalidPdfError before anything is stored
    document = Document(filename=filename, sha256=sha256, size_bytes=len(data), status=DocumentStatus.FAILED)
    session.add(document)
    session.flush()

    with tracing.span(
        "process_document", input={"filename": filename, "sha256": sha256, "size_bytes": len(data)}
    ) as root:
        document.trace_id = root.trace_id
        try:
            outcome = extract_and_validate(
                extractor, data, filename, pdf_text, settings.review_threshold, settings.buyer_names
            )
        except (ExtractionError, anthropic.AnthropicError) as e:
            document.error = f"{type(e).__name__}: {e}"
            session.commit()
            root.update(output={"document_id": str(document.id), "status": document.status}, level="ERROR")
            log.warning("extraction_failed", extra={"document_id": str(document.id), "error": document.error})
            return document, True

        result, checks, flagged = outcome.result, outcome.checks, outcome.flagged
        session.add(
            Extraction(
                document_id=document.id,
                model=result.model,
                llm_request_id=result.request_id,
                raw_output=result.document.model_dump(mode="json"),
                data=checks.data(),
                field_checks={
                    path: {"confidence": f.confidence, "model_confidence": f.model_confidence, "issues": f.issues}
                    for path, f in checks.fields.items()
                },
                document_issues=checks.document_issues,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=result.latency_ms,
            )
        )
        for f in flagged:
            session.add(
                ReviewTask(
                    document_id=document.id,
                    field_path=f.path,
                    extracted_value=f.value,
                    confidence=f.confidence,
                    reasons=f.issues or ["low_model_confidence"],
                )
            )
        for issue in checks.document_issues:
            session.add(ReviewTask(document_id=document.id, field_path="line_items", confidence=0.0, reasons=[issue]))

        document.status = DocumentStatus.NEEDS_REVIEW if outcome.needs_review else DocumentStatus.AUTO_APPROVED
        session.commit()
        root.update(
            output={"document_id": str(document.id), "status": document.status},
            metadata={"document_id": str(document.id)},
        )
        root.score_trace(name="needs_review", value=int(outcome.needs_review), data_type="BOOLEAN")
        root.score_trace(name="flagged_fields", value=len(flagged), data_type="NUMERIC")
    log.info(
        "document_processed",
        extra={
            "document_id": str(document.id),
            "status": document.status,
            "fields": len(checks.fields),
            "flagged_fields": [f.path for f in flagged],
            "trace_id": document.trace_id,
        },
    )
    return document, True


class InvalidCorrectionError(ValueError):
    pass


def resolve_task(session: Session, task: ReviewTask, value: Any, reviewer: str) -> ReviewTask:
    """Apply a reviewer's decision. ``value`` is the corrected value; pass the extracted one to confirm it."""
    if task.status == TaskStatus.RESOLVED:
        raise InvalidCorrectionError("Task already resolved")
    is_numeric = task.field_path.rsplit(".", 1)[-1] in NUMERIC_FIELDS
    if is_numeric and value is not None and (isinstance(value, bool) or not isinstance(value, int | float)):
        raise InvalidCorrectionError(f"{task.field_path} must be a number")

    extraction = task.document.extraction
    if extraction is not None and task.field_path != "line_items":
        data = copy.deepcopy(extraction.data)  # reassign so SQLAlchemy sees the JSON change
        set_path(data, task.field_path, value)
        extraction.data = data

    task.corrected_value = value
    task.status = TaskStatus.RESOLVED
    task.resolved_by = reviewer
    task.resolved_at = utcnow()
    session.flush()

    open_tasks = session.scalar(
        select(func.count())
        .select_from(ReviewTask)
        .where(ReviewTask.document_id == task.document_id, ReviewTask.status == TaskStatus.OPEN)
    )
    if open_tasks == 0:
        task.document.status = DocumentStatus.REVIEWED
    session.commit()
    log.info("review_task_resolved", extra={"task_id": str(task.id), "field": task.field_path})

    corrected = task.field_path != "line_items" and value != task.extracted_value
    tracing.score(
        task.document.trace_id,
        "human_review",
        "corrected" if corrected else "confirmed",
        comment=f"{task.field_path}: {task.extracted_value!r} -> {value!r}" if corrected else task.field_path,
        metadata={"field": task.field_path, "reviewer": reviewer, "reasons": task.reasons},
    )
    if open_tasks == 0:
        record_corrections(task.document)
    return task


def corrections(document: Document) -> list[dict[str, Any]]:
    return [
        {"field": t.field_path, "extracted": t.extracted_value, "corrected": t.corrected_value}
        for t in document.review_tasks
        if t.status == TaskStatus.RESOLVED and t.field_path != "line_items" and t.corrected_value != t.extracted_value
    ]


def label_from_data(data: dict[str, Any]) -> dict[str, Any]:
    """Reviewed extraction data in the shape of an evaluation label (eval/dataset/*.json)."""
    label = {k: data.get(k) for k in ("document_type", "supplier_name", "document_number", "document_date")}
    label |= {k: data.get(k) for k in ("currency", "subtotal", "total")}
    label["line_items"] = [{k: li.get(k) for k in LABEL_LINE_FIELDS} for li in data.get("line_items", [])]
    return label


def record_corrections(document: Document) -> None:
    """A document a reviewer had to correct is a production failure. Keep it in a Langfuse dataset, linked
    to its trace, so `procex-promote langfuse` can turn it into a regression case."""
    lf = tracing.get_langfuse()
    fixes = corrections(document)
    if lf is None or document.extraction is None or not fixes:
        return
    name = get_settings().langfuse_corrections_dataset
    document_id = str(document.id)
    item: dict[str, Any] = {  # built now, from the session's objects; sent later, outside the request
        "dataset_name": name,
        "id": f"{name}-{document.sha256[:32]}",
        "input": {"filename": document.filename, "sha256": document.sha256},
        "expected_output": label_from_data(document.extraction.data),
        "metadata": {
            "document_id": document_id,
            "model": document.extraction.model,
            "raw_output": document.extraction.raw_output,
            "corrections": fixes,
        },
        "source_trace_id": document.trace_id,
    }

    def send() -> None:
        try:
            tracing.ensure_dataset(name, "Documents a reviewer had to correct; expected_output is the reviewed data")
            lf.create_dataset_item(**item)
        except Exception:  # tracing must never break a review
            log.warning("langfuse_dataset_item_failed", extra={"document_id": document_id}, exc_info=True)

    tracing.in_background(send)
