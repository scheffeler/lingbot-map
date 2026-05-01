"""D6: pytest runner endpoint.

POST /api/test runs the project's test suite as a subprocess and
returns a structured summary so the dashboard's "Run tests" modal
can show pass/fail counts and the raw output.

Stubbed via `_run_pytest` which tests monkeypatch.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("POLEVISION_DB", str(tmp_path / "polevision.db"))
    monkeypatch.setenv("POLEVISION_WORKSPACE", str(tmp_path))
    import importlib
    import app.main as main
    import app.test_runner as test_runner
    importlib.reload(main)
    importlib.reload(test_runner)
    return TestClient(main.app), test_runner


def test_test_runner_returns_pass_count(client, monkeypatch):
    tc, mod = client

    def fake_run(args=None):
        return {
            "exit_code": 0,
            "passed": 21,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "duration_s": 1.1,
            "output": "============================== 21 passed in 1.10s ==============================",
        }
    monkeypatch.setattr(mod, "_run_pytest_subprocess", fake_run)
    r = tc.post("/api/test")
    assert r.status_code == 200
    body = r.json()
    assert body["passed"] == 21
    assert body["failed"] == 0
    assert body["exit_code"] == 0


def test_test_runner_handles_failure(client, monkeypatch):
    tc, mod = client

    def fake_run(args=None):
        return {
            "exit_code": 1,
            "passed": 18,
            "failed": 3,
            "skipped": 0,
            "errors": 0,
            "duration_s": 1.4,
            "output": "FAILED tests/foo.py::test_bar - AssertionError",
        }
    monkeypatch.setattr(mod, "_run_pytest_subprocess", fake_run)
    r = tc.post("/api/test")
    assert r.status_code == 200
    body = r.json()
    assert body["failed"] == 3
    assert body["exit_code"] == 1
    # The endpoint should not raise on non-zero exit; instead return
    # the breakdown so the UI can render the failure detail.


def test_test_runner_summary_parser_extracts_counts():
    """Direct unit-test of the parser used by the endpoint."""
    from app.test_runner import parse_pytest_summary
    summary = parse_pytest_summary(
        "============= 18 passed, 3 failed, 1 skipped in 1.40s ============="
    )
    assert summary["passed"] == 18
    assert summary["failed"] == 3
    assert summary["skipped"] == 1
    assert abs(summary["duration_s"] - 1.4) < 1e-6
