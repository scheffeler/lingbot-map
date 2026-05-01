"""Capture domain logic: import from disk, list, and per-stage
artifact status.

A "capture" is a folder of photos for one utility pole, plus the
artifact files produced by the pipeline (poses NPZ, masks NPZ, scale
JSON, triangulation JSON, etc.). The DB row is the source of truth
for identity (id, name) and metadata; artifact existence on disk is
read live so the dashboard reflects new files as soon as they
appear.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

from app import db

POLE_FOLDER_RE = re.compile(r"^pole_[A-Za-z0-9_-]+$")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff")
STAGES = ("exif", "pose", "sam", "scale", "fit")

# Maps each stage to the artifact filename pattern (relative to
# workspace, with `{name}` substituted). When the file exists, the
# stage is treated as succeeded — this lets the dashboard reflect
# pre-existing pipeline output without requiring a DB run row.
STAGE_ARTIFACTS = {
    "exif":  "{name}.gps.json",
    "pose":  "{name}.poses.npz",
    "sam":   "{name}.masks.npz",
    "scale": "{name}.scale.json",
    "fit":   "{name}.triangulation.json",
}


def workspace_path() -> Path:
    return Path(os.environ.get("POLEVISION_WORKSPACE", "."))


def _count_photos(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(
        1 for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def import_disk_captures() -> int:
    """Scan the workspace for `pole_*/` folders and INSERT OR IGNORE
    any new ones into the captures table. Returns the count of newly
    imported rows."""
    ws = workspace_path()
    if not ws.is_dir():
        return 0
    new_count = 0
    with db.connect() as conn:
        for child in sorted(ws.iterdir()):
            if not child.is_dir() or not POLE_FOLDER_RE.match(child.name):
                continue
            existing = conn.execute(
                "SELECT id FROM captures WHERE name = ?", (child.name,),
            ).fetchone()
            if existing:
                # Refresh photo count in case more photos were added.
                conn.execute(
                    "UPDATE captures SET n_photos = ? WHERE id = ?",
                    (_count_photos(child), existing["id"]),
                )
                continue
            conn.execute(
                """INSERT INTO captures (name, folder_path, n_photos)
                   VALUES (?, ?, ?)""",
                (child.name, str(child), _count_photos(child)),
            )
            new_count += 1
    return new_count


def upload_zip(name: str, file_bytes: bytes) -> dict:
    """Extract `file_bytes` (a zip) into `workspace/{name}/` and create
    a capture row. Raises ValueError on invalid input.

    The capture name is sanitized: only alphanumerics, dashes, and
    underscores are allowed; if it doesn't start with 'pole_' the
    prefix is added so later disk-import scans pick it up too.
    """
    import re as _re
    import zipfile as _zip
    # Only allow alphanumerics, dashes, underscores. Reject anything
    # else (../, slashes, spaces, dots) outright — sanitizing-then-
    # accepting hides path traversal attempts.
    if not _re.fullmatch(r"[A-Za-z0-9_\-]+", name or ""):
        raise ValueError(
            f"invalid capture name {name!r}: only [A-Za-z0-9_-] allowed"
        )
    safe = name.strip("-_")
    if not safe:
        raise ValueError(f"invalid capture name: {name!r}")
    if not safe.startswith("pole_"):
        safe = f"pole_{safe}"

    ws = workspace_path()
    folder = ws / safe
    if folder.exists():
        raise FileExistsError(f"capture {safe} already exists at {folder}")

    # Validate zip contents have at least one image.
    try:
        zf = _zip.ZipFile(__import__("io").BytesIO(file_bytes))
    except _zip.BadZipFile:
        raise ValueError("uploaded file is not a valid zip archive")
    image_members = [
        m for m in zf.namelist()
        if not m.endswith("/") and Path(m).suffix.lower() in IMG_EXTS
    ]
    if not image_members:
        raise ValueError(
            f"zip contains no image files (expected one of {IMG_EXTS})"
        )

    folder.mkdir(parents=True, exist_ok=False)
    for member in image_members:
        # Flatten nested zip layouts into the capture folder. Strip
        # parent dirs so we don't honor `../` traversal.
        target_name = Path(member).name
        out_path = folder / target_name
        with zf.open(member) as src, open(out_path, "wb") as dst:
            dst.write(src.read())

    n_photos = _count_photos(folder)
    with db.connect() as conn:
        cur = conn.execute(
            """INSERT INTO captures (name, folder_path, n_photos)
               VALUES (?, ?, ?)""",
            (safe, str(folder), n_photos),
        )
        cap_id = cur.lastrowid

    return {
        "id": cap_id,
        "name": safe,
        "folder_path": str(folder),
        "n_photos": n_photos,
    }


def list_captures() -> list[dict]:
    """Returns every non-deleted capture as a list of plain dicts,
    sorted by created_at desc."""
    import_disk_captures()
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT id, name, folder_path, created_at, n_photos,
                      gps_centroid_lat, gps_centroid_lon, gps_baseline_m,
                      notes
               FROM captures
               WHERE deleted = 0
               ORDER BY datetime(created_at) DESC, id DESC"""
        ).fetchall()
    return [_capture_row_to_dict(r) for r in rows]


def get_capture(capture_id: int) -> dict | None:
    """Return the full capture detail incl. per-stage artifact status."""
    with db.connect() as conn:
        row = conn.execute(
            """SELECT id, name, folder_path, created_at, n_photos,
                      gps_centroid_lat, gps_centroid_lon, gps_baseline_m,
                      notes
               FROM captures
               WHERE id = ? AND deleted = 0""",
            (capture_id,),
        ).fetchone()
    if row is None:
        return None
    detail = _capture_row_to_dict(row)
    detail["stages"] = stage_status(row["name"])
    return detail


def stage_status(capture_name: str) -> dict:
    """Resolve per-stage status by checking disk artifacts + the most
    recent run row. Disk artifact present → 'succeeded'. Otherwise
    fall back to the latest run's status, or 'queued' if there's no
    run row."""
    ws = workspace_path()
    out = {}
    with db.connect() as conn:
        latest_runs = {
            row["stage"]: row
            for row in conn.execute(
                """SELECT stage, status, started_at, finished_at,
                          exit_code, log_path, artifact_path
                   FROM runs r
                   WHERE capture_id = (
                     SELECT id FROM captures WHERE name = ?
                   )
                   ORDER BY id DESC""",
                (capture_name,),
            ).fetchall()
        }

    for stage in STAGES:
        artifact_name = STAGE_ARTIFACTS[stage].format(name=capture_name)
        artifact_path = ws / artifact_name
        run = latest_runs.get(stage)
        if artifact_path.exists():
            status = "succeeded"
        elif run is not None:
            status = run["status"]
        else:
            status = "queued"
        out[stage] = {
            "status": status,
            "artifact_path": str(artifact_path) if artifact_path.exists() else None,
            "exit_code": run["exit_code"] if run else None,
            "started_at": run["started_at"] if run else None,
            "finished_at": run["finished_at"] if run else None,
            "log_path": run["log_path"] if run else None,
        }
    return out


def _capture_row_to_dict(row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "folder_path": row["folder_path"],
        "created_at": row["created_at"],
        "n_photos": row["n_photos"],
        "gps_centroid_lat": row["gps_centroid_lat"],
        "gps_centroid_lon": row["gps_centroid_lon"],
        "gps_baseline_m": row["gps_baseline_m"],
        "notes": row["notes"],
    }
