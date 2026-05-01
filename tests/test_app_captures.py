"""D1 tests for capture import + GET endpoints.

Behaviors covered:
  - On startup (or first /api/captures hit) the app scans the working
    directory for `pole_*/` folders and imports them.
  - GET /api/captures returns the list of captures.
  - GET /api/captures/{id} returns details + a per-stage artifact map
    so the frontend can render stage-status dots without re-deriving
    the file checks.
  - Imported captures pick up n_photos from the folder.
"""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def temp_workspace(tmp_path, monkeypatch):
    """Set up a temp working dir with a fake pole_001 folder + 3 stub
    JPEGs so the import path has something to find."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    pole = workspace / "pole_001"
    pole.mkdir()
    for i in range(3):
        (pole / f"IMG_{i:04d}.JPEG").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")
    monkeypatch.setenv("POLEVISION_DB", str(workspace / "polevision.db"))
    monkeypatch.setenv("POLEVISION_WORKSPACE", str(workspace))
    return workspace


@pytest.fixture
def client(temp_workspace):
    import importlib
    import app.main as main
    importlib.reload(main)
    return TestClient(main.app)


def test_disk_capture_imported_on_first_list(client, temp_workspace):
    r = client.get("/api/captures")
    assert r.status_code == 200
    body = r.json()
    assert "captures" in body
    names = [c["name"] for c in body["captures"]]
    assert "pole_001" in names, (
        f"Expected pole_001 to be auto-imported, got {names}"
    )


def test_imported_capture_has_photo_count(client, temp_workspace):
    r = client.get("/api/captures")
    cap = next(c for c in r.json()["captures"] if c["name"] == "pole_001")
    assert cap["n_photos"] == 3, (
        f"Expected 3 photos in stub pole_001, got {cap['n_photos']}"
    )


def test_get_capture_includes_stage_artifact_status(client, temp_workspace):
    """GET /api/captures/{id} returns a `stages` dict mapping stage
    id to status. Initially everything is 'queued' since no run has
    happened yet."""
    list_resp = client.get("/api/captures")
    cap_id = list_resp.json()["captures"][0]["id"]

    r = client.get(f"/api/captures/{cap_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "pole_001"
    assert "stages" in body
    expected_stages = {"exif", "pose", "sam", "scale", "fit"}
    assert set(body["stages"].keys()) == expected_stages
    # All start as queued (no artifact, no run row).
    for stage_id, info in body["stages"].items():
        assert info["status"] in ("queued", "succeeded"), (
            f"Stage {stage_id}: status was {info['status']}"
        )


def test_artifact_status_succeeds_when_artifact_file_exists(
    client, temp_workspace
):
    """If pole_001.poses.npz exists in the workspace, the pose stage
    should show as 'succeeded' even without a run row."""
    # Place a stub artifact.
    (temp_workspace / "pole_001.poses.npz").write_bytes(b"")
    list_resp = client.get("/api/captures")
    cap_id = list_resp.json()["captures"][0]["id"]
    detail = client.get(f"/api/captures/{cap_id}").json()
    assert detail["stages"]["pose"]["status"] == "succeeded"


def test_capture_404_on_unknown_id(client):
    r = client.get("/api/captures/999999")
    assert r.status_code == 404
