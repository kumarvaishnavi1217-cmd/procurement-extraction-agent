import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from procex.config import Settings, get_settings
from procex.db import get_session, init_db
from procex.extractor import ClaudeExtractor, Extractor, FixtureExtractor
from procex.logging_config import configure_logging, new_request_id, request_id_var
from procex.models import Document, DocumentStatus, ReviewTask, TaskStatus
from procex.pdf import PDF_MAGIC, InvalidPdfError
from procex.pipeline import InvalidCorrectionError, process_document, resolve_task

log = logging.getLogger("procex.api")


@lru_cache
def get_extractor() -> Extractor:
    settings = get_settings()
    if settings.extractor == "fixture":
        return FixtureExtractor(Path(settings.fixtures_dir))
    return ClaudeExtractor(settings)


SessionDep = Annotated[Session, Depends(get_session)]
ExtractorDep = Annotated[Extractor, Depends(get_extractor)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


# ---------- response / request models ----------


class ReviewTaskOut(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID
    field_path: str
    extracted_value: Any
    confidence: float
    reasons: list[str]
    status: str
    corrected_value: Any
    resolved_by: str | None
    resolved_at: datetime | None


class UsageOut(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int


class DocumentOut(BaseModel):
    id: uuid.UUID
    filename: str
    status: str
    error: str | None
    created_at: datetime
    data: dict[str, Any] | None
    field_checks: dict[str, Any] | None
    usage: UsageOut | None
    open_review_tasks: list[ReviewTaskOut]

    @classmethod
    def of(cls, doc: Document) -> "DocumentOut":
        ex = doc.extraction
        return cls(
            id=doc.id,
            filename=doc.filename,
            status=doc.status,
            error=doc.error,
            created_at=doc.created_at,
            data=ex.data if ex else None,
            field_checks=ex.field_checks if ex else None,
            usage=UsageOut(
                model=ex.model,
                input_tokens=ex.input_tokens,
                output_tokens=ex.output_tokens,
                latency_ms=ex.latency_ms,
            )
            if ex
            else None,
            open_review_tasks=[
                ReviewTaskOut.model_validate(t, from_attributes=True)
                for t in doc.review_tasks
                if t.status == TaskStatus.OPEN
            ],
        )


class ResolveRequest(BaseModel):
    value: str | float | None = Field(description="Correct value; send the extracted value to confirm it")
    reviewer: str = Field(min_length=1, max_length=100)


# ---------- app ----------


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    configure_logging(get_settings().log_level)
    init_db()
    yield


app = FastAPI(title="Procurement Document Extraction Agent", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def request_logging(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    request_id = new_request_id(request.headers.get("x-request-id"))
    token = request_id_var.set(request_id)
    started = time.perf_counter()
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        log.info(
            "http_request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": int((time.perf_counter() - started) * 1000),
            },
        )
        return response
    finally:
        request_id_var.reset(token)


@app.get("/health")
def health(session: SessionDep) -> dict[str, str]:
    session.execute(text("select 1"))
    return {"status": "ok"}


@app.post("/documents", status_code=status.HTTP_201_CREATED, response_model=DocumentOut)
def upload_document(
    file: UploadFile, response: Response, session: SessionDep, extractor: ExtractorDep, settings: SettingsDep
) -> DocumentOut:
    data = file.file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "File too large")
    if not data.startswith(PDF_MAGIC):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only PDF files are accepted")
    try:
        document, created = process_document(session, extractor, settings, file.filename or "upload.pdf", data)
    except InvalidPdfError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e
    if not created:
        response.status_code = status.HTTP_200_OK
    return DocumentOut.of(document)


@app.get("/documents", response_model=list[DocumentOut])
def list_documents(
    session: SessionDep,
    status_filter: Annotated[DocumentStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[DocumentOut]:
    query = select(Document).order_by(Document.created_at.desc()).limit(limit)
    if status_filter is not None:
        query = query.where(Document.status == status_filter)
    return [DocumentOut.of(d) for d in session.scalars(query)]


@app.get("/documents/{document_id}", response_model=DocumentOut)
def get_document(document_id: uuid.UUID, session: SessionDep) -> DocumentOut:
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return DocumentOut.of(document)


@app.get("/review-tasks", response_model=list[ReviewTaskOut])
def list_review_tasks(
    session: SessionDep,
    status_filter: Annotated[TaskStatus, Query(alias="status")] = TaskStatus.OPEN,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[ReviewTask]:
    query = (
        select(ReviewTask)
        .where(ReviewTask.status == status_filter)
        .order_by(ReviewTask.confidence, ReviewTask.created_at)  # least confident first
        .limit(limit)
    )
    return list(session.scalars(query))


@app.post("/review-tasks/{task_id}/resolve", response_model=ReviewTaskOut)
def resolve_review_task(task_id: uuid.UUID, body: ResolveRequest, session: SessionDep) -> ReviewTask:
    task = session.get(ReviewTask, task_id)
    if task is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Review task not found")
    try:
        return resolve_task(session, task, body.value, body.reviewer)
    except InvalidCorrectionError as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e)) from e
