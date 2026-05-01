"""Capture upload: POST /api/captures/upload accepts a zip of photos
plus a name, extracts to workspace/{name}/, and creates a DB row.
"""

from __future__ import annotations

import io
import zipfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def temp_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setenv("POLEVISION_DB", str(workspace / "polevision.db"))
    monkeypatch.setenv("POLEVISION_WORKSPACE", str(workspace))
    return workspace


@pytest.fixture
def client(temp_workspace):
    import importlib
    import app.main as main
    importlib.reload(main)
    return TestClient(main.app)


def _make_zip(file_names: list[str]) -> bytes:
    """Build a zip containing one tiny stub JPEG per filename."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in file_names:
            zf.writestr(name, b"\xff\xd8\xff\xe0\x00\x10JFIF")
    return buf.getvalue()


def test_upload_creates_capture_and_extracts_photos(client, temp_workspace):
    zip_bytes = _make_zip([
        "IMG_0001.JPEG", "IMG_0002.JPEG", "IMG_0003.JPEG",
    ])
    r = client.post(
        "/api/captures/upload",
        files={"file": ("test.zip", zip_bytes, "application/zip")},
        data={"name": "pole_007"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["name"] == "pole_007"
    assert body["n_photos"] == 3
    # Folder + JPEGs exist on disk.
    folder = temp_workspace / "pole_007"
    assert folder.is_dir()
    assert (folder / "IMG_0001.JPEG").exists()
    # Listed by /api/captures.
    listed = [c["name"] for c in client.get("/api/captures").json()["captures"]]
    assert "pole_007" in listed


def test_upload_rejects_zip_with_no_photos(client, temp_workspace):
    zip_bytes = _make_zip(["readme.txt"])  # no .jpg/.jpeg/.png
    # Replace the JFIF stub bytes with text body for clarity.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", b"hello")
    r = client.post(
        "/api/captures/upload",
        files={"file": ("empty.zip", buf.getvalue(), "application/zip")},
        data={"name": "pole_empty"},
    )
    assert r.status_code == 400
    assert "photo" in r.json()["detail"].lower() or "image" in r.json()["detail"].lower()


def test_upload_rejects_duplicate_name(client, temp_workspace):
    z = _make_zip(["a.JPEG", "b.JPEG"])
    r1 = client.post(
        "/api/captures/upload",
        files={"file": ("x.zip", z, "application/zip")},
        data={"name": "pole_dup"},
    )
    assert r1.status_code == 200
    r2 = client.post(
        "/api/captures/upload",
        files={"file": ("x.zip", z, "application/zip")},
        data={"name": "pole_dup"},
    )
    assert r2.status_code == 409


def test_upload_rejects_invalid_name(client, temp_workspace):
    z = _make_zip(["a.JPEG"])
    r = client.post(
        "/api/captures/upload",
        files={"file": ("x.zip", z, "application/zip")},
        data={"name": "../escape"},
    )
    assert r.status_code == 400


def test_upload_normalizes_name_with_pole_prefix(client, temp_workspace):
    """A capture name without 'pole_' prefix gets prefixed automatically
    so disk-import scanning later finds it."""
    z = _make_zip(["a.JPEG", "b.JPEG"])
    r = client.post(
        "/api/captures/upload",
        files={"file": ("x.zip", z, "application/zip")},
        data={"name": "myrig"},
    )
    assert r.status_code == 200
    assert r.json()["name"].startswith("pole_")
