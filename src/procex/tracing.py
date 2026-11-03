"""Langfuse tracing for extractions, review decisions and evaluation runs.

Enabled when ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are set (plus ``LANGFUSE_BASE_URL`` for a
self-hosted instance). Without them every helper here is a no-op, so the API, the tests and the offline
evaluation run unchanged.

Traces carry the file name, hash, extracted values and validation results, never the PDF bytes.
"""

import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from langfuse import Langfuse

from procex.config import get_settings

_override: Langfuse | None = None
# Spans and scores are batched by the SDK; dataset calls are synchronous HTTP, so they run here instead
# of inside a reviewer's request.
_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="procex-langfuse")
_pending: list[Future[None]] = []
_pending_lock = threading.Lock()


@lru_cache
def _from_settings() -> Langfuse | None:
    s = get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        return None
    return Langfuse(
        public_key=s.langfuse_public_key,
        secret_key=s.langfuse_secret_key,
        base_url=s.langfuse_base_url,
        environment=s.langfuse_environment,
    )


def get_langfuse() -> Langfuse | None:
    return _override or _from_settings()


def use_client(client: Langfuse | None) -> None:
    """Tests: route tracing to a client with an in-memory exporter (None restores the configured one)."""
    global _override
    _override = client


class _NoOp:
    """Stands in for a Langfuse observation when tracing is off."""

    id: str | None = None
    trace_id: str | None = None

    def update(self, **_: Any) -> "_NoOp":
        return self

    def score_trace(self, **_: Any) -> None:
        return None


@contextmanager
def span(name: str, **kwargs: Any) -> Iterator[Any]:
    lf = get_langfuse()
    if lf is None:
        yield _NoOp()
        return
    with lf.start_as_current_observation(name=name, as_type="span", **kwargs) as obs:
        yield obs


@contextmanager
def generation(name: str, **kwargs: Any) -> Iterator[Any]:
    lf = get_langfuse()
    if lf is None:
        yield _NoOp()
        return
    with lf.start_as_current_observation(name=name, as_type="generation", **kwargs) as obs:
        yield obs


@contextmanager
def guardrail(name: str, **kwargs: Any) -> Iterator[Any]:
    lf = get_langfuse()
    if lf is None:
        yield _NoOp()
        return
    with lf.start_as_current_observation(name=name, as_type="guardrail", **kwargs) as obs:
        yield obs


def score(
    trace_id: str | None,
    name: str,
    value: float | str,
    comment: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Score a trace after the fact, e.g. when a reviewer resolves a task hours later. String values are
    categorical scores, numbers numeric ones."""
    lf = get_langfuse()
    if lf is None or trace_id is None:
        return
    if isinstance(value, str):
        lf.create_score(
            trace_id=trace_id, name=name, value=value, data_type="CATEGORICAL", comment=comment, metadata=metadata
        )
    else:
        lf.create_score(
            trace_id=trace_id, name=name, value=value, data_type="NUMERIC", comment=comment, metadata=metadata
        )


def ensure_dataset(name: str, description: str) -> None:
    lf = get_langfuse()
    if lf is None:
        return
    try:
        lf.get_dataset(name, fetch_items_page_size=1)
    except Exception:  # the SDK raises its NotFoundError (via the generated API client)
        lf.create_dataset(name=name, description=description)


def in_background(job: Callable[[], None]) -> None:
    """Run a blocking Langfuse API call off the request path. ``job`` handles its own errors."""
    with _pending_lock:
        _pending[:] = [f for f in _pending if not f.done()]
        _pending.append(_worker.submit(job))


def flush() -> None:
    """Wait for background jobs, then send buffered spans and scores."""
    with _pending_lock:
        jobs = list(_pending)
    wait(jobs)
    lf = get_langfuse()
    if lf is not None:
        lf.flush()
