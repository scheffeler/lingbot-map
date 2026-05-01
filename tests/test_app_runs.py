"""D2 tests for pipeline-stage subprocess orchestration + SSE log
streaming.

Behaviors covered:
  - POST /api/captures/{id}/run/{stage} returns a run id and creates a
    DB row with status='running'.
  - When the subprocess exits 0, the run row transitions to
    'succeeded' and the log file is written.
  - When it exits non-zero, the row becomes 'failed' with exit_code
    populated.
  - The /api/runs/{id}/stream endpoint emits each log line as an SSE
    `data:` event, in order, terminating with a `[done]` line.
  - GET /api/runs/{id} returns the latest row and (when finished) the
    log file content.

Test isolation: subprocess invocations go through `STAGE_COMMANDS`
which we monkeypatch to point at a tiny stub script that prints
known lines and exits with a controllable code. No real Modal calls.
"""

from __future__ import annotations

import os
import sys
import time
import json
import textwrap

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def temp_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    pole = workspace / "pole_001"
    pole.mkdir()
    for i in range(3):
        (pole / f"IMG_{i:04d}.JPEG").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
    monkeypatch.setenv("POLEVISION_DB", str(workspace / "polevision.db"))
    monkeypatch.setenv("POLEVISION_WORKSPACE", str(workspace))
    return workspace


def _stub_script(workspace, *, lines: list[str], exit_code: int) -> list[str]:
    """Build an argv that runs Python with a small inline program that
    prints `lines` (one per line, with small sleeps to exercise the
    streaming code path) and exits with `exit_code`."""
    body = "\n".join([
        "import sys, time",
        f"lines = {lines!r}",
        "for line in lines:",
        "    print(line, flush=True)",
        "    time.sleep(0.02)",
        f"sys.exit({exit_code})",
    ])
    return [sys.executable, "-c", body]


@pytest.fixture
def client(temp_workspace, monkeypatch):
    """Build a TestClient. Reload only `app.main` so it picks up the
    test's POLEVISION_DB env var; do NOT reload `app.runs` because
    that would clobber the module-level `_QUEUES` dict and create a
    stale reference in `main.runs`."""
    import importlib
    import app.runs as runs
    import app.main as main
    importlib.reload(main)
    # Reset runs' module-level state so prior tests don't leak.
    runs._QUEUES.clear()
    return TestClient(main.app), runs


def test_post_run_creates_running_row(client, temp_workspace, monkeypatch):
    tc, runs = client
    cap_id = tc.get("/api/captures").json()["captures"][0]["id"]
    monkeypatch.setattr(
        runs, "build_stage_command",
        lambda stage, cap, params=None: _stub_script(
            temp_workspace,
            lines=["hello", "world", "done"],
            exit_code=0,
        ),
    )
    r = tc.post(f"/api/captures/{cap_id}/run/exif")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "run_id" in body and "status" in body
    # Right after submission the status is queued or running (race);
    # we only require that it's a valid in-flight state.
    assert body["status"] in ("queued", "running")


def test_run_finishes_succeeded_with_log(client, temp_workspace, monkeypatch):
    tc, runs = client
    cap_id = tc.get("/api/captures").json()["captures"][0]["id"]
    monkeypatch.setattr(
        runs, "build_stage_command",
        lambda stage, cap, params=None: _stub_script(
            temp_workspace,
            lines=["starting exif", "found 3 photos", "wrote pole_001.gps.json"],
            exit_code=0,
        ),
    )
    run_id = tc.post(f"/api/captures/{cap_id}/run/exif").json()["run_id"]
    # Wait for completion (poll the run row).
    deadline = time.time() + 5.0
    final = None
    while time.time() < deadline:
        final = tc.get(f"/api/runs/{run_id}").json()
        if final["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)
    assert final is not None
    assert final["status"] == "succeeded", f"final={final}"
    assert final["exit_code"] == 0
    # Log file should exist and contain the printed lines.
    assert os.path.exists(final["log_path"])
    log_content = open(final["log_path"]).read()
    assert "starting exif" in log_content
    assert "wrote pole_001.gps.json" in log_content


def test_run_finishes_failed_with_exit_code(client, temp_workspace, monkeypatch):
    tc, runs = client
    cap_id = tc.get("/api/captures").json()["captures"][0]["id"]
    monkeypatch.setattr(
        runs, "build_stage_command",
        lambda stage, cap, params=None: _stub_script(
            temp_workspace,
            lines=["boom"],
            exit_code=42,
        ),
    )
    run_id = tc.post(f"/api/captures/{cap_id}/run/exif").json()["run_id"]
    deadline = time.time() + 5.0
    final = None
    while time.time() < deadline:
        final = tc.get(f"/api/runs/{run_id}").json()
        if final["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)
    assert final["status"] == "failed"
    assert final["exit_code"] == 42


def test_sse_stream_emits_lines_in_order(client, temp_workspace, monkeypatch):
    tc, runs = client
    cap_id = tc.get("/api/captures").json()["captures"][0]["id"]
    monkeypatch.setattr(
        runs, "build_stage_command",
        lambda stage, cap, params=None: _stub_script(
            temp_workspace,
            lines=["alpha", "beta", "gamma"],
            exit_code=0,
        ),
    )
    run_id = tc.post(f"/api/captures/{cap_id}/run/exif").json()["run_id"]

    # Tail the stream; collect data: lines until we see [done].
    seen = []
    with tc.stream("GET", f"/api/runs/{run_id}/stream") as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        for line in resp.iter_lines():
            if line.startswith("data: "):
                payload = line[len("data: "):]
                seen.append(payload)
                if payload == "[done]":
                    break
    # At least the three printed lines should have made it through, in
    # order, before the [done] sentinel.
    body_lines = [s for s in seen if s != "[done]"]
    assert "alpha" in body_lines
    assert "beta" in body_lines
    assert "gamma" in body_lines
    # Order check: indices should be ascending.
    a, b, g = (body_lines.index(x) for x in ("alpha", "beta", "gamma"))
    assert a < b < g, f"out of order: {body_lines}"


def test_capture_detail_reflects_run_status(client, temp_workspace, monkeypatch):
    """After a successful exif run, GET /api/captures/{id} should
    show stages.exif.status == 'succeeded'."""
    tc, runs = client
    cap_id = tc.get("/api/captures").json()["captures"][0]["id"]

    # Have the stub create the artifact file the stage_status helper
    # looks for, so the disk-artifact path also lights up.
    artifact = str(temp_workspace / "pole_001.gps.json")
    body = "\n".join([
        "import sys",
        f"open({artifact!r}, 'w').write('{{}}')",
        "print('wrote artifact')",
        "sys.exit(0)",
    ])
    monkeypatch.setattr(
        runs, "build_stage_command",
        lambda stage, cap, params=None: [sys.executable, "-c", body],
    )
    run_id = tc.post(f"/api/captures/{cap_id}/run/exif").json()["run_id"]
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if tc.get(f"/api/runs/{run_id}").json()["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.05)
    detail = tc.get(f"/api/captures/{cap_id}").json()
    assert detail["stages"]["exif"]["status"] == "succeeded"
