import enum
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

JsonType = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class DocumentStatus(enum.StrEnum):
    AUTO_APPROVED = "auto_approved"  # every field passed validation with high confidence
    NEEDS_REVIEW = "needs_review"  # at least one open review task
    REVIEWED = "reviewed"  # all review tasks resolved by a human
    FAILED = "failed"  # extraction failed (refusal, truncation, API error)


class TaskStatus(enum.StrEnum):
    OPEN = "open"
    RESOLVED = "resolved"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    filename: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    size_bytes: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), index=True)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    extraction: Mapped["Extraction | None"] = relationship(back_populates="document", uselist=False)
    review_tasks: Mapped[list["ReviewTask"]] = relationship(back_populates="document", order_by="ReviewTask.field_path")


class Extraction(Base):
    __tablename__ = "extractions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id"), unique=True)
    model: Mapped[str] = mapped_column(String(100))
    llm_request_id: Mapped[str | None] = mapped_column(String(100))
    raw_output: Mapped[dict[str, Any]] = mapped_column(JsonType)  # exactly what the model returned
    data: Mapped[dict[str, Any]] = mapped_column(JsonType)  # current values, including human corrections
    field_checks: Mapped[dict[str, Any]] = mapped_column(JsonType)  # path -> confidence + issues
    document_issues: Mapped[list[str]] = mapped_column(JsonType, default=list)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    document: Mapped[Document] = relationship(back_populates="extraction")


class ReviewTask(Base):
    __tablename__ = "review_tasks"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id"), index=True)
    field_path: Mapped[str] = mapped_column(String(100))
    extracted_value: Mapped[Any] = mapped_column(JsonType, nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    reasons: Mapped[list[str]] = mapped_column(JsonType, default=list)
    status: Mapped[str] = mapped_column(String(20), default=TaskStatus.OPEN, index=True)
    corrected_value: Mapped[Any] = mapped_column(JsonType, nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(100))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    document: Mapped[Document] = relationship(back_populates="review_tasks")
