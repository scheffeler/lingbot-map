"""D0 smoke tests for the FastAPI dashboard backend.

Two minimal contracts:
  - The app starts and responds to GET /api/health.
  - DB schema is created idempotently in a temp SQLite file.

These tests intentionally don't touch the real `polevision.db`; the
fixture below redirects the app to a per-test temp file via the
`POLEVISION_DB` env var.
"""

import os
import sqlite3
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def temp_db_path(monkeypatch, tmp_path):
    db_path = tmp_path / "polevision.db"
    monkeypatch.setenv("POLEVISION_DB", str(db_path))
    return str(db_path)


@pytest.fixture
def client(temp_db_path):
    """Build a TestClient against a fresh app instance — re-import to
    pick up the env-var DB path."""
    import importlib
    import app.main as main
    importlib.reload(main)
    return TestClient(main.app)


def test_health_endpoint_returns_200_with_version(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert "version" in body
    assert "db_path" in body


def test_db_schema_created_on_startup(client, temp_db_path):
    """After the app starts, the SQLite file exists and has the
    expected tables."""
    assert os.path.exists(temp_db_path), \
        f"DB file not created at {temp_db_path}"
    conn = sqlite3.connect(temp_db_path)
    tables = {row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    conn.close()
    expected = {"captures", "runs", "measurements", "ref_taps"}
    missing = expected - tables
    assert not missing, f"Missing tables: {missing}; got {tables}"


def test_db_init_is_idempotent(client, temp_db_path):
    """Hitting the app twice should not error or duplicate schema."""
    r1 = client.get("/api/health")
    r2 = client.get("/api/health")
    assert r1.status_code == 200 and r2.status_code == 200
