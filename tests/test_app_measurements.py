"""D7: /api/captures/{id}/measurements endpoint.

Parses `{name}.triangulation.json` (the fit stage's output) into a
list of `{kind, value, sigma, source}` measurement records the
dashboard's Measurements tab can render.

When triangulation.json doesn't exist yet, the endpoint returns an
empty list (the tab shows the "Run fit to populate" empty state).
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def temp_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    pole = workspace / "pole_001"
    pole.mkdir()
    (pole / "IMG_0001.JPEG").write_bytes(b"\xff\xd8\xff\xe0")
    monkeypatch.setenv("POLEVISION_DB", str(workspace / "polevision.db"))
    monkeypatch.setenv("POLEVISION_WORKSPACE", str(workspace))
    return workspace


@pytest.fixture
def client(temp_workspace):
    import importlib
    import app.main as main
    importlib.reload(main)
    return TestClient(main.app)


def test_measurements_empty_when_no_triangulation(client):
    cap_id = client.get("/api/captures").json()["captures"][0]["id"]
    r = client.get(f"/api/captures/{cap_id}/measurements")
    assert r.status_code == 200
    assert r.json() == {"measurements": []}


def test_measurements_parsed_from_triangulation_json(client, temp_workspace):
    """Drop a realistic triangulation.json next to pole_001 and
    confirm the endpoint extracts height + lean as separate
    measurement records."""
    triang = {
        "pole_top_xyz": [0, -0.3, 0.7],
        "pole_bottom_xyz": [0, 0.3, 0.4],
        "height_model_units": 0.6,
        "height_m": 9.24,
        "metric_scale": 15.4,
        "axis_lean_deg": 14.5,
        "axis_lean_cam_up_deg": 13.8,
        "frames_used": 8,
        "frames_total": 8,
    }
    (temp_workspace / "pole_001.triangulation.json").write_text(
        json.dumps(triang)
    )
    cap_id = client.get("/api/captures").json()["captures"][0]["id"]
    r = client.get(f"/api/captures/{cap_id}/measurements")
    body = r.json()
    kinds = {m["kind"]: m for m in body["measurements"]}
    assert "height_m" in kinds, body
    assert abs(kinds["height_m"]["value"] - 9.24) < 1e-9
    assert kinds["height_m"]["unit"] == "m"
    assert "lean_deg" in kinds
    assert abs(kinds["lean_deg"]["value"] - 14.5) < 1e-9
    # Lean has both ground-normal and camera-up references; the
    # endpoint should expose at least one and label it.
    assert "source" in kinds["height_m"]


def test_measurements_includes_scale_when_present(client, temp_workspace):
    """If gps_scale.json exists, expose the metric scale itself as
    a measurement (so the UI can show 'GPS scale: 15.4 m/unit ± ...')."""
    (temp_workspace / "pole_001.scale.json").write_text(json.dumps({
        "scale": 15.4,
        "sigma": 4.27,
        "method": "weighted_least_squares",
    }))
    cap_id = client.get("/api/captures").json()["captures"][0]["id"]
    r = client.get(f"/api/captures/{cap_id}/measurements")
    kinds = {m["kind"]: m for m in r.json()["measurements"]}
    assert "metric_scale" in kinds
    assert abs(kinds["metric_scale"]["value"] - 15.4) < 1e-9
    assert abs(kinds["metric_scale"]["sigma"] - 4.27) < 1e-9
