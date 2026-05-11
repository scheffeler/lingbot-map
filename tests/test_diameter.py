"""Phase 1.5 S2 — diameter at named heights along the fitted pole axis.

Synthetic test: a 0.30 m wide vertical bar viewed from 8 cameras on a
horizontal ring at 8 m radius. `measure_diameter` should recover the
planted diameter to within 5% at any height inside the pole's extent.

Real-fixture test: on `pole_001`, the diameter at z=1.5 m must fall in
[0.20, 0.40] m — the typical range for distribution poles. This is a
sanity gate, not a correctness claim; ground-truth validation is S6.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fit_pole_axis import (  # noqa: E402
    fit_axis_from_planes,
    measure_diameter,
    mask_centerline_2d,
    viewing_plane_normal,
)
from triangulate_pole import (  # noqa: E402
    fit_largest_component,
    native_to_pad518,
)
from tests.synthetic import make_pole_scene


def _fit_axis_from_scene(scene):
    """Fit axis from a synthetic scene and orient it to point from
    gt_bottom toward gt_top (matches the convention `measure_diameter`
    expects: heights >0 lie along +axis_dir from `pole_bottom_xyz`)."""
    masks = scene["masks"]
    extrinsics = scene["extrinsics"]
    intrinsics = scene["intrinsics"]
    H, W = scene["image_hw"]
    normals, offsets = [], []
    for i in range(masks.shape[0]):
        pts = fit_largest_component(masks[i])
        if pts is None or pts.shape[0] < 50:
            continue
        centroid, major, _ = mask_centerline_2d(pts)
        c_uv = native_to_pad518(centroid, (H, W))
        m_uv = native_to_pad518(centroid + major, (H, W))
        major_uv = m_uv - c_uv
        n, d = viewing_plane_normal(c_uv, major_uv,
                                    intrinsics[i], extrinsics[i])
        if np.linalg.norm(n) < 1e-9:
            continue
        normals.append(n)
        offsets.append(d)
    axis_point, axis_dir = fit_axis_from_planes(
        np.stack(normals), np.array(offsets))
    if axis_dir @ scene["gt_axis_dir"] < 0:
        axis_dir = -axis_dir
    return axis_point, axis_dir


def test_diameter_synthetic_bar_recovers_within_5pct():
    scene = make_pole_scene(
        height_m=10.0, diameter_m=0.30,
        n_cameras=8, radius_m=8.0, noise_px=0.0,
    )
    axis_point, axis_dir = _fit_axis_from_scene(scene)

    # In the synthetic frame the world is metric, so scale = 1.0.
    diameters = measure_diameter(
        masks=scene["masks"],
        extrinsics=scene["extrinsics"],
        intrinsics=scene["intrinsics"],
        native_hw=scene["image_hw"],
        axis_point=axis_point,
        axis_dir=axis_dir,
        pole_bottom_xyz=scene["gt_bottom_xyz"],
        heights_m=[1.5, 5.0, 8.0],
        scale=1.0,
    )

    assert len(diameters) == 3
    for entry in diameters:
        assert "height_m" in entry and "diameter_m" in entry
        assert "n_frames_used" in entry
        # Synthetic rectangle has width pole_pix_width =
        # diameter_m * focal_px / depth, so recovery should be tight.
        assert entry["n_frames_used"] >= 4, (
            f"too few frames for h={entry['height_m']}: "
            f"{entry['n_frames_used']}"
        )
        assert abs(entry["diameter_m"] - 0.30) / 0.30 < 0.05, (
            f"h={entry['height_m']}: recovered "
            f"{entry['diameter_m']:.3f} m, expected 0.30 ±5%"
        )


def test_diameter_height_outside_pole_returns_low_n_frames():
    """Heights above the pole tip or below ground shouldn't crash;
    they just return n_frames_used=0 (no mask intersection)."""
    scene = make_pole_scene(
        height_m=10.0, diameter_m=0.30,
        n_cameras=8, radius_m=8.0, noise_px=0.0,
    )
    axis_point, axis_dir = _fit_axis_from_scene(scene)

    diameters = measure_diameter(
        masks=scene["masks"],
        extrinsics=scene["extrinsics"],
        intrinsics=scene["intrinsics"],
        native_hw=scene["image_hw"],
        axis_point=axis_point,
        axis_dir=axis_dir,
        pole_bottom_xyz=scene["gt_bottom_xyz"],
        heights_m=[20.0],   # well above the 10 m pole
        scale=1.0,
    )
    assert len(diameters) == 1
    assert diameters[0]["n_frames_used"] == 0
    assert diameters[0]["diameter_m"] is None


@pytest.mark.skipif(
    not (REPO_ROOT / "pole_001.masks.npz").exists()
    or not (REPO_ROOT / "pole_001.poses.npz").exists()
    or not (REPO_ROOT / "pole_001.triangulation.json").exists(),
    reason="pole_001 artifacts not present",
)
def test_diameter_real_pole_001_in_plausible_range():
    """Sanity gate on the real capture: diameter at 1.5 m above ground
    sits inside the typical distribution-pole range."""
    import json

    masks_data = np.load(REPO_ROOT / "pole_001.masks.npz", allow_pickle=True)
    poses_data = np.load(REPO_ROOT / "pole_001.poses.npz", allow_pickle=True)
    triang = json.loads(
        (REPO_ROOT / "pole_001.triangulation.json").read_text()
    )

    masks = masks_data["masks"]
    if masks.ndim == 4:
        # Multi-object NPZ from SAM video predictor — pick the
        # object id used by the saved triangulation.
        oid = int(triang.get("object_id", 0))
        masks = masks[oid]
    native_hw = tuple(int(x) for x in masks_data["image_hw"])
    extrinsics = poses_data["extrinsic"]
    intrinsics = poses_data["intrinsic"]

    axis_point = np.array(triang["pole_bottom_xyz"], dtype=np.float64)
    axis_dir = np.array(triang["axis_direction"], dtype=np.float64)
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    pole_bottom = np.array(triang["pole_bottom_xyz"], dtype=np.float64)
    scale = float(triang["metric_scale"])

    diameters = measure_diameter(
        masks=masks,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        native_hw=native_hw,
        axis_point=axis_point,
        axis_dir=axis_dir,
        pole_bottom_xyz=pole_bottom,
        heights_m=[1.5],
        scale=scale,
    )
    assert len(diameters) == 1
    d = diameters[0]
    assert d["n_frames_used"] >= 2, (
        f"only {d['n_frames_used']} frames usable at h=1.5 m "
        f"(8-frame capture has limited base coverage; expect ≥2)"
    )
    assert 0.15 <= d["diameter_m"] <= 0.50, (
        f"recovered diameter {d['diameter_m']:.3f} m at 1.5 m is "
        f"outside plausible distribution-pole range [0.15, 0.50] m"
    )
