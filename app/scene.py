"""Backend support for the in-browser three.js 3D viewer.

`build_scene(name)` bundles every piece of geometry the viewer needs
into one JSON: per-camera poses, the fitted pole axis, the ground-
plane normal/offset (RANSAC on the dense PLY), and the metric scale.

The dense PLY is *not* embedded — it's served separately by the
/api/captures/{id}/ply endpoint to avoid bloating the JSON.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from app import captures

# Re-use the well-tested ground-plane RANSAC from the viser viewer.
# Add scripts/ to sys.path so the import works regardless of how the
# app was launched.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def _load_ground_plane(ply_path: Path, cam_centres: np.ndarray) -> dict | None:
    """Run the same RANSAC fit visualize_pole.py uses; return
    {normal, offset} or None if no plausible plane."""
    try:
        from visualize_pole import fit_ground_plane, load_ply_binary
    except Exception:
        return None
    try:
        pts, _ = load_ply_binary(ply_path)
    except Exception:
        return None
    res = fit_ground_plane(pts.astype(np.float64), cam_centres)
    if res is None:
        return None
    normal, offset, _ = res
    return {
        "normal": [float(x) for x in normal],
        "offset": float(offset),
    }


def build_scene(capture_name: str, capture_id: int | None = None) -> dict | None:
    """Returns a dict with every piece of geometry the in-browser
    viewer needs, or None if the capture doesn't have pose
    artifacts yet (the viewer is gated on pose stage success).

    `capture_id` (int) is used to construct the URLs for /ply and
    /frame/{idx} since those endpoints take the integer id, not the
    name. If omitted, the URLs are emitted with the name and the
    frontend has to rewrite them — keep this for backward
    compatibility but pass the id when you can.
    """
    ws = captures.workspace_path()
    poses_path = ws / f"{capture_name}.poses.npz"
    if not poses_path.exists():
        return None

    poses = np.load(poses_path, allow_pickle=True)
    extrinsics = poses["extrinsic"]               # (S, 3, 4)
    intrinsics = poses["intrinsic"]               # (S, 3, 3)
    image_paths = [str(p) for p in poses["image_paths"]]
    pad_h, pad_w = (int(x) for x in poses["image_hw"])

    url_key = capture_id if capture_id is not None else capture_name
    cameras = []
    for i in range(extrinsics.shape[0]):
        cameras.append({
            "index": i,
            "extrinsic": extrinsics[i].tolist(),
            "intrinsic": intrinsics[i].tolist(),
            "image_path": image_paths[i] if i < len(image_paths) else "",
            "frame_url": f"/api/captures/{url_key}/frame/{i}",
        })

    out: dict = {
        "capture": capture_name,
        "cameras": cameras,
        "image_hw": [pad_h, pad_w],
        "ply_url": f"/api/captures/{url_key}/ply",
    }

    triang_path = ws / f"{capture_name}.triangulation.json"
    if triang_path.exists():
        try:
            t = json.loads(triang_path.read_text())
        except json.JSONDecodeError:
            t = {}
        if t:
            out["pole"] = {
                "top": t.get("pole_top_xyz"),
                "bottom": t.get("pole_bottom_xyz"),
                "axis_dir": t.get("axis_direction"),
                "height_model_units": t.get("height_model_units"),
                "height_m": t.get("height_m"),
                "axis_lean_deg": t.get("axis_lean_deg"),
                "axis_lean_cam_up_deg": t.get("axis_lean_cam_up_deg"),
                "object_id": t.get("object_id"),
                "ci": t.get("ci"),
            }
            if t.get("metric_scale") is not None:
                out["metric_scale"] = float(t["metric_scale"])

    scale_path = ws / f"{capture_name}.scale.json"
    if scale_path.exists() and "metric_scale" not in out:
        try:
            s = json.loads(scale_path.read_text())
            if "scale" in s:
                out["metric_scale"] = float(s["scale"])
        except json.JSONDecodeError:
            pass

    ply_path = ws / f"{capture_name}.ply"
    if ply_path.exists():
        cam_centres = extrinsics[:, :3, 3].astype(np.float64)
        out["ground_plane"] = _load_ground_plane(ply_path, cam_centres)
    else:
        out["ground_plane"] = None

    return out


def ply_path(capture_name: str) -> Path:
    return captures.workspace_path() / f"{capture_name}.ply"


def frame_path(capture_name: str, idx: int) -> tuple[Path | None, list[str]]:
    """Return the path of the `idx`-th source photo for `capture_name`,
    plus the alphabetical list of all image basenames (for tests).
    Returns (None, names) when idx is out of range."""
    ws = captures.workspace_path()
    poses_path = ws / f"{capture_name}.poses.npz"
    names: list[str] = []
    if poses_path.exists():
        try:
            poses = np.load(poses_path, allow_pickle=True)
            names = [str(p) for p in poses["image_paths"]]
        except Exception:
            names = []
    if not names:
        # Fall back to disk scan in alphabetical order — matches the
        # ordering phase1_sam_imageset uses.
        with captures.db.connect() as conn:
            row = conn.execute(
                "SELECT folder_path FROM captures WHERE name = ?",
                (capture_name,),
            ).fetchone()
        if row is None:
            return None, []
        folder = Path(row["folder_path"])
        if not folder.is_dir():
            return None, []
        IMG_EXTS = (".jpg", ".jpeg", ".png", ".heic", ".heif")
        names = sorted(
            p.name for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )
    if idx < 0 or idx >= len(names):
        return None, names

    # Resolve to actual disk path.
    with captures.db.connect() as conn:
        row = conn.execute(
            "SELECT folder_path FROM captures WHERE name = ?",
            (capture_name,),
        ).fetchone()
    if row is None:
        return None, names
    p = Path(row["folder_path"]) / names[idx]
    return (p if p.exists() else None), names


def frame_jpeg_bytes(capture_name: str, idx: int,
                     max_width: int = 1600,
                     quality: int = 85) -> bytes | None:
    """Return JPEG bytes for the `idx`-th photo. Transcodes HEIC →
    JPEG via pillow-heif so browsers can display it."""
    p, _ = frame_path(capture_name, idx)
    if p is None:
        return None
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
    from PIL import Image
    import io as _io
    with Image.open(p) as im:
        im = im.convert("RGB")
        if im.width > max_width:
            new_h = int(round(im.height * max_width / im.width))
            im = im.resize((max_width, new_h), Image.BILINEAR)
        buf = _io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
        return buf.getvalue()
