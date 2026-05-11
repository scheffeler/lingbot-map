"""T1: backend endpoints for the in-browser three.js 3D viewer.

  GET /api/captures/{id}/scene        — JSON: poses + triangulation +
                                        ground plane + metric scale
  GET /api/captures/{id}/ply           — streams the dense PLY
  GET /api/captures/{id}/frame/{idx}  — JPEG of one source photo
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def temp_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    pole = workspace / "pole_001"
    pole.mkdir()
    # Real-ish JPEG (PIL must be able to open it for the frame test).
    from PIL import Image
    for i in range(2):
        Image.new("RGB", (640, 480), (180 + 30 * i, 80, 80)).save(
            pole / f"IMG_{i:04d}.JPEG", "JPEG"
        )
    monkeypatch.setenv("POLEVISION_DB", str(workspace / "polevision.db"))
    monkeypatch.setenv("POLEVISION_WORKSPACE", str(workspace))
    return workspace


@pytest.fixture
def client(temp_workspace):
    import importlib
    import app.main as main
    importlib.reload(main)
    return TestClient(main.app)


def _write_stub_poses(path, n_frames=2):
    np.savez(
        path,
        extrinsic=np.tile(
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                     dtype=np.float32),
            (n_frames, 1, 1),
        ),
        intrinsic=np.tile(
            np.array([[400, 0, 259], [0, 400, 259], [0, 0, 1]],
                     dtype=np.float32),
            (n_frames, 1, 1),
        ),
        image_paths=np.array([f"IMG_{i:04d}.JPEG" for i in range(n_frames)]),
        image_hw=np.array([518, 518], dtype=np.int32),
    )


def _write_stub_ply(path, n=400):
    """Write a small binary PLY (same format `test_pole.write_ply_binary`
    produces) with `n` random points around y≈0 so a ground-plane
    fit has something to lock onto."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-2, 2, size=(n, 3)).astype(np.float32)
    pts[:, 1] = rng.normal(0, 0.02, size=n).astype(np.float32)  # ~flat plane
    cols = rng.integers(50, 250, size=(n, 3), dtype=np.uint8)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dt = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    rows = np.empty(n, dtype=dt)
    rows["x"], rows["y"], rows["z"] = pts.T
    rows["r"], rows["g"], rows["b"] = cols.T
    with open(path, "wb") as f:
        f.write(header)
        f.write(rows.tobytes())


def test_scene_404_when_pose_missing(client):
    cap_id = client.get("/api/captures").json()["captures"][0]["id"]
    r = client.get(f"/api/captures/{cap_id}/scene")
    assert r.status_code == 404
    assert "pose" in r.json()["detail"].lower()


def test_scene_returns_pose_arrays(client, temp_workspace):
    _write_stub_poses(temp_workspace / "pole_001.poses.npz", n_frames=2)
    cap_id = client.get("/api/captures").json()["captures"][0]["id"]
    r = client.get(f"/api/captures/{cap_id}/scene")
    assert r.status_code == 200, r.text
    body = r.json()
    assert "cameras" in body and len(body["cameras"]) == 2
    cam0 = body["cameras"][0]
    assert "extrinsic" in cam0 and len(cam0["extrinsic"]) == 3
    assert "intrinsic" in cam0 and len(cam0["intrinsic"]) == 3
    assert "image_path" in cam0
    assert body["image_hw"] == [518, 518]


def test_scene_includes_triangulation_when_present(client, temp_workspace):
    _write_stub_poses(temp_workspace / "pole_001.poses.npz", n_frames=2)
    (temp_workspace / "pole_001.triangulation.json").write_text(json.dumps({
        "pole_top_xyz": [0.0, -0.3, 0.7],
        "pole_bottom_xyz": [0.0, 0.3, 0.4],
        "axis_direction": [0.0, -1.0, 0.0],
        "height_m": 8.96,
        "metric_scale": 15.4,
        "axis_lean_deg": 14.5,
    }))
    cap_id = client.get("/api/captures").json()["captures"][0]["id"]
    body = client.get(f"/api/captures/{cap_id}/scene").json()
    assert "pole" in body
    assert body["pole"]["top"] == [0.0, -0.3, 0.7]
    assert body["pole"]["bottom"] == [0.0, 0.3, 0.4]
    assert abs(body["pole"]["height_m"] - 8.96) < 1e-9
    assert abs(body["pole"]["axis_lean_deg"] - 14.5) < 1e-9
    assert abs(body["metric_scale"] - 15.4) < 1e-9


def test_scene_includes_diameters_and_attachments(client, temp_workspace):
    """When triangulation.json carries `diameters` and `attachments`
    (S2 + S3 outputs), the /scene endpoint surfaces them so the
    in-browser viewer can render cylinders + attachment markers."""
    _write_stub_poses(temp_workspace / "pole_001.poses.npz", n_frames=2)
    (temp_workspace / "pole_001.triangulation.json").write_text(json.dumps({
        "pole_top_xyz": [0.0, -0.3, 0.7],
        "pole_bottom_xyz": [0.0, 0.3, 0.4],
        "axis_direction": [0.0, -1.0, 0.0],
        "height_m": 8.96,
        "metric_scale": 15.4,
        "diameters": [
            {"height_m": 1.5, "diameter_m": 0.37, "n_frames_used": 4},
            {"height_m": 5.0, "diameter_m": None, "n_frames_used": 0},
        ],
        "attachments": [
            {"name": "crossarm", "object_index": 1,
             "height_m": 7.2, "n_frames_used": 6},
            {"name": "wire", "object_index": 2,
             "height_m": None, "n_frames_used": 0},
        ],
    }))
    cap_id = client.get("/api/captures").json()["captures"][0]["id"]
    body = client.get(f"/api/captures/{cap_id}/scene").json()
    assert "pole" in body
    assert "diameters" in body["pole"]
    # Only the populated entry should pass through; nulls are filtered.
    assert len(body["pole"]["diameters"]) == 1
    d = body["pole"]["diameters"][0]
    assert d["height_m"] == 1.5 and abs(d["diameter_m"] - 0.37) < 1e-9
    assert "attachments" in body["pole"]
    assert len(body["pole"]["attachments"]) == 1
    a = body["pole"]["attachments"][0]
    assert a["name"] == "crossarm" and abs(a["height_m"] - 7.2) < 1e-9


def test_scene_includes_ground_plane_when_ply_present(client, temp_workspace):
    _write_stub_poses(temp_workspace / "pole_001.poses.npz", n_frames=2)
    _write_stub_ply(temp_workspace / "pole_001.ply", n=600)
    cap_id = client.get("/api/captures").json()["captures"][0]["id"]
    body = client.get(f"/api/captures/{cap_id}/scene").json()
    assert "ground_plane" in body and body["ground_plane"] is not None
    n = body["ground_plane"]["normal"]
    assert len(n) == 3
    # Synthesised cloud's plane is roughly the y=0 plane → normal
    # dominant in y axis.
    assert abs(n[1]) > abs(n[0]) and abs(n[1]) > abs(n[2])


def test_ply_endpoint_streams_bytes(client, temp_workspace):
    _write_stub_poses(temp_workspace / "pole_001.poses.npz", n_frames=2)
    _write_stub_ply(temp_workspace / "pole_001.ply", n=300)
    expected = (temp_workspace / "pole_001.ply").read_bytes()
    cap_id = client.get("/api/captures").json()["captures"][0]["id"]
    r = client.get(f"/api/captures/{cap_id}/ply")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/octet-stream")
    assert r.content == expected


def test_frame_endpoint_returns_jpeg(client, temp_workspace):
    _write_stub_poses(temp_workspace / "pole_001.poses.npz", n_frames=2)
    cap_id = client.get("/api/captures").json()["captures"][0]["id"]
    r = client.get(f"/api/captures/{cap_id}/frame/0")
    assert r.status_code == 200
    assert "image/jpeg" in r.headers["content-type"]
    # Confirm it's a real JPEG by sniffing magic bytes.
    assert r.content[:3] == b"\xff\xd8\xff"


def test_frame_endpoint_404_on_out_of_range(client, temp_workspace):
    _write_stub_poses(temp_workspace / "pole_001.poses.npz", n_frames=2)
    cap_id = client.get("/api/captures").json()["captures"][0]["id"]
    r = client.get(f"/api/captures/{cap_id}/frame/99")
    assert r.status_code == 404


def test_ply_endpoint_404_when_missing(client):
    cap_id = client.get("/api/captures").json()["captures"][0]["id"]
    r = client.get(f"/api/captures/{cap_id}/ply")
    assert r.status_code == 404
