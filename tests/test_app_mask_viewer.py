"""D4a: /api/captures/{id}/mask-viewer returns the standalone HTML
mask-viewer for that capture. 404 when masks NPZ doesn't exist yet.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def temp_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    pole = workspace / "pole_001"
    pole.mkdir()
    # Drop a real-ish JPEG so PIL can open it (1x1 white pixel).
    from PIL import Image
    img = Image.new("RGB", (640, 480), (200, 200, 200))
    img.save(pole / "IMG_0001.JPEG", "JPEG")
    img.save(pole / "IMG_0002.JPEG", "JPEG")
    monkeypatch.setenv("POLEVISION_DB", str(workspace / "polevision.db"))
    monkeypatch.setenv("POLEVISION_WORKSPACE", str(workspace))
    return workspace


@pytest.fixture
def client(temp_workspace):
    import importlib
    import app.main as main
    importlib.reload(main)
    return TestClient(main.app)


def test_mask_viewer_404_when_no_masks(client):
    cap_id = client.get("/api/captures").json()["captures"][0]["id"]
    r = client.get(f"/api/captures/{cap_id}/mask-viewer")
    assert r.status_code == 404


def test_mask_viewer_returns_html_when_masks_exist(client, temp_workspace):
    """Drop a synthetic 2-frame masks NPZ and a matching capture
    folder; the endpoint must produce the embedded HTML."""
    H, W = 64, 48
    masks = np.zeros((1, 2, H, W), dtype=bool)
    masks[0, 0, 10:30, 20:25] = True
    masks[0, 1, 12:28, 22:27] = True
    np.savez(
        temp_workspace / "pole_001.masks.npz",
        masks=masks,
        obj_ids=np.array([0], dtype=np.int64),
        text_prompt=np.array("utility pole"),
        image_hw=np.array([H, W], dtype=np.int32),
        image_paths=np.array(["IMG_0001.JPEG", "IMG_0002.JPEG"]),
        ref_frame=np.int32(0),
    )
    cap_id = client.get("/api/captures").json()["captures"][0]["id"]
    r = client.get(f"/api/captures/{cap_id}/mask-viewer")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    body = r.text
    assert "SAM 3.1 mask viewer" in body, "Expected the design's title in the HTML"
    assert "data:image/jpeg;base64," in body, "Expected embedded JPEG data URL"
