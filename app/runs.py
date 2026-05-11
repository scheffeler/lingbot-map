"""Pipeline-stage subprocess orchestration + SSE log streaming.

Each stage (`exif`, `pose`, `sam`, `scale`, `fit`) maps to a CLI
command via `build_stage_command`. Tests monkeypatch this function
to inject stub commands instead of invoking the real (Modal-bound)
pipeline scripts.

We run the subprocess in a daemon **thread** rather than via
`asyncio.create_subprocess_exec`. Reason: FastAPI's TestClient
(httpx + anyio) runs each request in a fresh event loop, so
asyncio.Task work doesn't survive across requests. A plain thread
runs in the background of the test process and remains addressable
via the run id.

Live log streaming uses a thread-safe `queue.Queue` per run, with
the SSE endpoint pulling from it via `run_in_executor`. Late
subscribers (connecting after the subprocess finished) get the log
file content from disk plus a `[done]` sentinel.
"""

from __future__ import annotations

import asyncio
import os
import queue
import shlex
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from app import captures, db

# Per-run live broadcast queues. List-of-Queues per run so multiple
# tabs/clients can each tail the same run independently.
_QUEUES: dict[int, list[queue.Queue]] = {}
_LOCK = threading.Lock()


def _workspace() -> Path:
    return Path(os.environ.get("POLEVISION_WORKSPACE", "."))


def _log_dir() -> Path:
    return _workspace() / "logs"


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def build_stage_command(
    stage: str, capture: dict, params: dict | None = None,
) -> list[str]:
    """Map a stage name + capture row + per-stage params to the
    subprocess argv. Tests monkeypatch this; production routes through
    the real CLI scripts. All paths are relative to the workspace dir.

    Recognized params (per stage):
      fit:  object_id (int) — which tracked object to triangulate
      sam:  text_prompts (list[str]) — extra prompts for multi-class
            tracking (defers to default 'utility pole' if absent)
      scale: pixel_sigma_px (float)
    """
    p = params or {}
    name = capture["name"]
    folder = capture["folder_path"]
    if stage == "exif":
        return [
            sys.executable, "scripts/exif_gps.py",
            "--folder", folder, "--out", f"{name}.gps.json",
        ]
    if stage == "pose":
        return [
            "modal", "run", "scripts/phase0_modal_imageset.py",
            "--image-folder", folder, "--output", f"{name}.ply",
        ]
    if stage == "sam":
        cmd = [
            "modal", "run", "scripts/phase1_sam_imageset.py",
            "--image-folder", folder,
            "--output", f"{name}.masks.npz",
            "--preview", f"{name}.masks_preview.jpg",
        ]
        text = p.get("text")
        if text:
            cmd.extend(["--text", str(text)])
        return cmd
    if stage == "scale":
        cmd = [
            sys.executable, "scripts/scale_solver.py",
            "--poses", f"{name}.poses.npz",
            "--out", f"{name}.scale.json",
        ]
        gps_path = _workspace() / f"{name}.gps.json"
        if gps_path.exists():
            cmd.extend(["--gps", str(gps_path)])
        return cmd
    if stage == "fit":
        object_id = int(p.get("object_id", 0))
        bootstrap = int(p.get("bootstrap", 0))
        diameter_heights = p.get("diameter_at_heights")
        cmd = [
            sys.executable, "scripts/fit_pole_axis.py",
            "--masks", f"{name}.masks.npz",
            "--poses", f"{name}.poses.npz",
            "--object-id", str(object_id),
            "--gps-scale", f"{name}.scale.json",
            "--ply", f"{name}.ply",
            "--output", f"{name}.triangulation.json",
        ]
        if bootstrap > 0:
            cmd.extend(["--bootstrap", str(bootstrap)])
        if diameter_heights:
            if isinstance(diameter_heights, (list, tuple)):
                heights_str = ",".join(str(float(h)) for h in diameter_heights)
            else:
                heights_str = str(diameter_heights)
            cmd.extend(["--diameter-at-heights", heights_str])
        attachment_objects = p.get("attachment_objects")
        if attachment_objects:
            if isinstance(attachment_objects, (list, tuple)):
                # Accept ["crossarm=0","wire=1"] or [0, 1].
                tokens = [str(x) for x in attachment_objects]
                arg = ",".join(tokens)
            else:
                arg = str(attachment_objects)
            cmd.extend(["--attachment-objects", arg])
        return cmd
    raise ValueError(f"Unknown stage: {stage}")


# Map of run_id → subprocess.Popen so cancellation can find the
# right process. Cleaned out when the subprocess finishes.
_PROCS: dict[int, subprocess.Popen] = {}


def start_run(
    capture_id: int, stage: str, params: dict | None = None,
) -> dict:
    """Insert a `running` row, spawn the subprocess thread, return
    `{run_id, status}`. The thread updates the row when the
    subprocess completes."""
    cap = captures.get_capture(capture_id)
    if cap is None:
        raise ValueError(f"capture {capture_id} not found")
    _log_dir().mkdir(parents=True, exist_ok=True)

    started = _utcnow()
    import json as _json
    params_json = _json.dumps(params or {})
    with db.connect() as conn:
        cur = conn.execute(
            """INSERT INTO runs (capture_id, stage, status, started_at, params)
               VALUES (?, ?, 'running', ?, ?)""",
            (capture_id, stage, started, params_json),
        )
        run_id = cur.lastrowid
        log_path = str(_log_dir() / f"run_{run_id}.log")
        conn.execute(
            "UPDATE runs SET log_path = ? WHERE id = ?", (log_path, run_id)
        )

    argv = build_stage_command(stage, cap, params)
    with _LOCK:
        _QUEUES[run_id] = []
    # Capture the active DB path now so the thread updates the same
    # database it was launched against, even if the env var changes
    # later (matters for tests where each test uses a fresh tmp DB).
    db_path_snapshot = db.db_path()
    thread = threading.Thread(
        target=_run_subprocess_in_thread,
        args=(run_id, argv, log_path, db_path_snapshot),
        daemon=True,
    )
    thread.start()
    return {"run_id": run_id, "status": "running",
            "command": " ".join(shlex.quote(a) for a in argv)}


def _run_subprocess_in_thread(
    run_id: int, argv: list[str], log_path: str, db_path: str,
):
    """Body of the subprocess thread. Reads stdout line-by-line,
    writes to the per-run log file, fanouts to live SSE queues, and
    updates the DB row when finished."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(_workspace()),
            env=env,
        )
        _PROCS[run_id] = proc
    except FileNotFoundError as e:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"failed to launch subprocess: {e}\n"
                    f"argv: {argv}\n")
        _update_run_in_db(db_path, run_id, "failed", -1)
        _broadcast(run_id, f"failed to launch subprocess: {e}")
        _broadcast(run_id, "[done]")
        return

    with open(log_path, "w", encoding="utf-8") as logf:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            logf.write(line)
            logf.flush()
            _broadcast(run_id, line.rstrip("\n"))
    rc = proc.wait()
    _PROCS.pop(run_id, None)
    final_status = "succeeded" if rc == 0 else "failed"
    _update_run_in_db(db_path, run_id, final_status, rc)
    _broadcast(run_id, "[done]")


def cancel_run(run_id: int) -> dict:
    """Terminate the subprocess for `run_id`, mark the row failed.
    Returns the updated row, or raises ValueError if the run doesn't
    exist."""
    run = get_run(run_id)
    if run is None:
        raise ValueError(f"run {run_id} not found")
    proc = _PROCS.get(run_id)
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            pass
    return get_run(run_id) or run


def _update_run_in_db(db_path: str, run_id: int, status: str, exit_code: int):
    """Direct sqlite3 update — bypasses `db.connect()` so the thread
    writes to the database it was launched against, regardless of any
    later env-var changes."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """UPDATE runs SET status = ?, finished_at = ?, exit_code = ?
               WHERE id = ?""",
            (status, _utcnow(), exit_code, run_id),
        )
        conn.commit()
    finally:
        conn.close()


def _broadcast(run_id: int, line: str) -> None:
    with _LOCK:
        for q in _QUEUES.get(run_id, []):
            q.put(line)


def get_run(run_id: int) -> dict | None:
    with db.connect() as conn:
        row = conn.execute(
            """SELECT id, capture_id, stage, status, started_at,
                      finished_at, exit_code, log_path, artifact_path
               FROM runs WHERE id = ?""",
            (run_id,),
        ).fetchone()
    return dict(row) if row else None


def list_runs(capture_id: int, limit: int = 50) -> list[dict]:
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT id, capture_id, stage, status, started_at,
                      finished_at, exit_code
               FROM runs
               WHERE capture_id = ?
               ORDER BY id DESC
               LIMIT ?""",
            (capture_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


async def stream_run_log(run_id: int):
    """Async generator yielding SSE `data: ...\\n\\n` events.

    Late subscribers (run already finished) get the log file content
    plus a final `[done]`. Live subscribers get a replay-then-tail.
    """
    run = get_run(run_id)
    if run is None:
        yield f"data: [error] run {run_id} not found\n\n"
        yield "data: [done]\n\n"
        return

    log_path = run.get("log_path")
    if log_path and os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                yield f"data: {line.rstrip(chr(10))}\n\n"

    # Refresh: between the file replay and the queue subscribe, the
    # run may have completed.
    run = get_run(run_id) or run
    if run["status"] in ("succeeded", "failed"):
        yield "data: [done]\n\n"
        return

    q: queue.Queue = queue.Queue()
    with _LOCK:
        _QUEUES.setdefault(run_id, []).append(q)
    loop = asyncio.get_event_loop()
    try:
        while True:
            line = await loop.run_in_executor(None, q.get)
            yield f"data: {line}\n\n"
            if line == "[done]":
                break
    finally:
        with _LOCK:
            subs = _QUEUES.get(run_id, [])
            if q in subs:
                subs.remove(q)
