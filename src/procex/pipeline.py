"""PDF -> LLM extraction -> validation -> persisted result + review queue."""

import copy
import hashlib
import logging
from typing import Any

import anthropic
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from procex.config import Settings
from procex.extractor import ExtractionError, Extractor
from procex.models import Document, DocumentStatus, Extraction, ReviewTask, TaskStatus, utcnow
from procex.pdf import extract_text
from procex.validation import set_path, validate

log = logging.getLogger(__name__)

NUMERIC_FIELDS = ("quantity", "unit_price", "line_total", "subtotal", "total")


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

    try:
        result = extractor.extract(data, filename)
    except (ExtractionError, anthropic.AnthropicError) as e:
        document.error = f"{type(e).__name__}: {e}"
        session.commit()
        log.warning("extraction_failed", extra={"document_id": str(document.id), "error": document.error})
        return document, True

    checks = validate(result.document, pdf_text, settings.buyer_names)
    flagged = checks.needs_review(settings.review_threshold)

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

    needs_review = bool(flagged or checks.document_issues)
    document.status = DocumentStatus.NEEDS_REVIEW if needs_review else DocumentStatus.AUTO_APPROVED
    session.commit()
    log.info(
        "document_processed",
        extra={
            "document_id": str(document.id),
            "status": document.status,
            "fields": len(checks.fields),
            "flagged_fields": [f.path for f in flagged],
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
    return task
