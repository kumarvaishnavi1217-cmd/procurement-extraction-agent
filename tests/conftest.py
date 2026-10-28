import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from testcontainers.community.postgres import PostgresContainer

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "eval" / "dataset"
FIXTURES = ROOT / "eval" / "fixtures"
BUYER = "Meridian Manufacturing GmbH"


def pdf(doc_id: str) -> bytes:
    return (DATASET / f"{doc_id}.pdf").read_bytes()


@pytest.fixture(scope="session")
def postgres() -> Iterator[str]:
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as pg:
        yield pg.get_connection_url()


@pytest.fixture(scope="session")
def app_client(postgres: str) -> Iterator[TestClient]:
    os.environ["PROCEX_DATABASE_URL"] = postgres
    os.environ["PROCEX_BUYER_NAMES"] = f'["{BUYER}"]'
    from procex.api import app, get_extractor
    from procex.config import get_settings
    from procex.db import get_engine
    from procex.extractor import FixtureExtractor

    get_settings.cache_clear()
    get_engine.cache_clear()
    app.dependency_overrides[get_extractor] = lambda: FixtureExtractor(FIXTURES)
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
def client(app_client: TestClient) -> TestClient:
    from procex.db import get_engine

    with get_engine().begin() as conn:
        conn.execute(text("TRUNCATE review_tasks, extractions, documents"))
    return app_client
