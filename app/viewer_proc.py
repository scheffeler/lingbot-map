"""Manage the viser 3D viewer as a subprocess. One process at a time
on port 8080; switching captures replaces the running viewer.

Why: the dashboard runs in a different process from viser. Letting
the dashboard own viser's lifecycle means the user never starts a
CLI manually, and there's a deterministic answer to "which capture
is the 3D tab showing?".
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# Module-level "current" subprocess tracker. A dict so test fixtures
# can `.clear()` it without touching globals.
_CURRENT: dict[str, object] = {}

VISER_PORT = 8080


def _workspace() -> Path:
    return Path(os.environ.get("POLEVISION_WORKSPACE", "."))


def _kill_current(timeout: float = 2.0) -> None:
    proc = _CURRENT.get("proc")
    if proc is None:
        return
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
    _CURRENT.pop("proc", None)
    _CURRENT.pop("capture_name", None)


def start_for_capture(capture: dict) -> dict:
    """Validate that the capture has the artifacts viser needs, kill
    any prior viser process, start a new one bound to this capture.

    Returns `{url, capture, pid}`.

    Raises FileNotFoundError if the pose stage hasn't produced its
    artifacts yet.
    """
    ws = _workspace()
    name = capture["name"]
    poses = ws / f"{name}.poses.npz"
    ply = ws / f"{name}.ply"
    if not poses.exists() or not ply.exists():
        raise FileNotFoundError(
            f"viser viewer needs the pose stage's outputs "
            f"({name}.poses.npz + {name}.ply); run pose first."
        )
    masks = ws / f"{name}.masks.npz"
    triang = ws / f"{name}.triangulation.json"
    scale = ws / f"{name}.scale.json"

    argv = [
        sys.executable, "scripts/visualize_pole.py",
        "--ply", str(ply),
        "--poses", str(poses),
        "--port", str(VISER_PORT),
        "--align-ground",
    ]
    if masks.exists():
        argv.extend(["--masks", str(masks)])
    if triang.exists():
        argv.extend(["--triangulation", str(triang)])
    if scale.exists():
        argv.extend(["--gps-scale", str(scale)])
    argv.extend(["--image-folder", capture["folder_path"]])

    _kill_current()

    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    proc = subprocess.Popen(
        argv,
        cwd=str(ws),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _CURRENT["proc"] = proc
    _CURRENT["capture_name"] = name
    _CURRENT["started_at"] = time.time()
    return {
        "url": f"http://localhost:{VISER_PORT}",
        "capture": name,
        "pid": getattr(proc, "pid", None),
    }


def status() -> dict:
    proc = _CURRENT.get("proc")
    if proc is None:
        return {"running": False}
    rc = proc.poll()
    return {
        "running": rc is None,
        "capture": _CURRENT.get("capture_name"),
        "pid": getattr(proc, "pid", None),
        "url": f"http://localhost:{VISER_PORT}",
        "exit_code": rc,
    }


def stop() -> None:
    _kill_current()
