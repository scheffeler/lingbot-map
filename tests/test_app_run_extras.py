"""Cancel + stage-params extensions to the /run endpoint."""

from __future__ import annotations

import sys
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def temp_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    pole = workspace / "pole_001"
    pole.mkdir()
    for i in range(3):
        (pole / f"IMG_{i:04d}.JPEG").write_bytes(b"\xff\xd8")
    monkeypatch.setenv("POLEVISION_DB", str(workspace / "polevision.db"))
    monkeypatch.setenv("POLEVISION_WORKSPACE", str(workspace))
    return workspace


@pytest.fixture
def client(temp_workspace):
    import importlib
    import app.runs as runs
    import app.main as main
    importlib.reload(main)
    runs._QUEUES.clear()
    return TestClient(main.app), runs


def _slow_stub(exit_code: int = 0, sleep_s: float = 1.5):
    body = "\n".join([
        "import sys, time",
        "print('starting', flush=True)",
        f"time.sleep({sleep_s})",
        "print('done', flush=True)",
        f"sys.exit({exit_code})",
    ])
    return [sys.executable, "-c", body]


def test_run_cancel_terminates_subprocess(client, monkeypatch):
    tc, runs = client
    cap_id = tc.get("/api/captures").json()["captures"][0]["id"]
    monkeypatch.setattr(runs, "build_stage_command",
                        lambda stage, cap, params=None: _slow_stub(0, 4.0))
    run_id = tc.post(f"/api/captures/{cap_id}/run/exif").json()["run_id"]
    # Give the thread a moment to actually start the subprocess.
    time.sleep(0.3)
    r = tc.post(f"/api/runs/{run_id}/cancel")
    assert r.status_code == 200, r.text
    # Wait briefly for the cancellation to land.
    deadline = time.time() + 3.0
    final = None
    while time.time() < deadline:
        final = tc.get(f"/api/runs/{run_id}").json()
        if final["status"] == "failed":
            break
        time.sleep(0.05)
    assert final["status"] == "failed"
    # We don't strictly require exit_code == -15 (Windows uses
    # different codes), only that it's non-zero.
    assert final["exit_code"] != 0


def test_run_cancel_404_on_unknown_run(client):
    tc, _ = client
    r = tc.post("/api/runs/99999/cancel")
    assert r.status_code == 404


def test_stage_params_passed_to_build_command(client, monkeypatch):
    """POST body's `params` field is forwarded to build_stage_command."""
    tc, runs = client
    cap_id = tc.get("/api/captures").json()["captures"][0]["id"]
    captured_params = []

    def fake_builder(stage, cap, params=None):
        captured_params.append(params or {})
        return [sys.executable, "-c", "print('ok')"]

    monkeypatch.setattr(runs, "build_stage_command", fake_builder)
    r = tc.post(
        f"/api/captures/{cap_id}/run/fit",
        json={"params": {"object_id": 2}},
    )
    assert r.status_code == 200, r.text
    assert captured_params == [{"object_id": 2}]


def test_fit_stage_command_includes_object_id(temp_workspace, monkeypatch):
    """The real build_stage_command for `fit` substitutes the
    object_id param into the argv."""
    monkeypatch.setenv("POLEVISION_WORKSPACE", str(temp_workspace))
    import importlib
    import app.runs as runs
    importlib.reload(runs)
    cap = {"name": "pole_001", "folder_path": str(temp_workspace / "pole_001")}
    argv = runs.build_stage_command("fit", cap, params={"object_id": 2})
    assert "--object-id" in argv
    assert argv[argv.index("--object-id") + 1] == "2"
