"""D4b: /api/captures/{id}/view-3d starts the viser viewer for that
capture as a subprocess on port 8080.

`subprocess.Popen` is monkeypatched to a stub so tests don't actually
launch viser.
"""

from __future__ import annotations

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
def client(temp_workspace, monkeypatch):
    import importlib
    import app.main as main
    import app.viewer_proc as vp
    importlib.reload(main)
    # Stub Popen so we don't fork viser. Reset the module state.
    vp._CURRENT.clear()

    class FakeProc:
        def __init__(self, argv, **kwargs):
            self.argv = argv
            self._returncode = None
            self.pid = 12345
        def poll(self):
            return self._returncode
        def terminate(self):
            self._returncode = -15
        def wait(self, timeout=None):
            self._returncode = self._returncode or 0
            return self._returncode

    fakes_created = []
    def fake_popen(*args, **kwargs):
        argv = args[0] if args else kwargs.get("args")
        p = FakeProc(argv, **kwargs)
        fakes_created.append(p)
        return p

    monkeypatch.setattr(vp.subprocess, "Popen", fake_popen)
    return TestClient(main.app), vp, fakes_created


def test_view_3d_400_when_pose_missing(client, temp_workspace):
    tc, vp, fakes = client
    cap_id = tc.get("/api/captures").json()["captures"][0]["id"]
    r = tc.post(f"/api/captures/{cap_id}/view-3d")
    assert r.status_code == 400, r.text
    assert "pose" in r.json()["detail"].lower()
    assert len(fakes) == 0, "should not start subprocess when pose missing"


def test_view_3d_starts_subprocess_when_artifacts_exist(
    client, temp_workspace,
):
    tc, vp, fakes = client
    # Lay down fake artifacts.
    (temp_workspace / "pole_001.poses.npz").write_bytes(b"")
    (temp_workspace / "pole_001.ply").write_bytes(b"ply\n")
    cap_id = tc.get("/api/captures").json()["captures"][0]["id"]
    r = tc.post(f"/api/captures/{cap_id}/view-3d")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["url"].startswith("http://localhost:")
    assert len(fakes) == 1
    argv = fakes[0].argv
    # Argv should reference the capture's artifacts.
    assert any("pole_001.ply" in a for a in argv)
    assert any("pole_001.poses.npz" in a for a in argv)


def test_view_3d_kills_prior_subprocess_on_new_call(
    client, temp_workspace,
):
    tc, vp, fakes = client
    (temp_workspace / "pole_001.poses.npz").write_bytes(b"")
    (temp_workspace / "pole_001.ply").write_bytes(b"ply\n")
    cap_id = tc.get("/api/captures").json()["captures"][0]["id"]
    tc.post(f"/api/captures/{cap_id}/view-3d")
    tc.post(f"/api/captures/{cap_id}/view-3d")
    assert len(fakes) == 2
    # The first one should have been terminated.
    assert fakes[0].poll() is not None, (
        f"first subprocess returncode={fakes[0].poll()}; expected non-None "
        f"after termination"
    )
