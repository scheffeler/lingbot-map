"""S0 sanity test: a planted vertical pole + ring of cameras must
recover the planted axis through the existing viewing-plane fit.

This is the foundation for every later test. If it doesn't pass, the
synthetic generator is wrong and downstream tests on diameter,
attachments, uncertainty etc. cannot be trusted.
"""

import numpy as np
import pytest

from tests.synthetic import make_pole_scene
from triangulate_pole import (
    fit_largest_component,
    native_to_pad518,
    pixel_to_world_ray,
)
from fit_pole_axis import (
    mask_centerline_2d,
    viewing_plane_normal,
    fit_axis_from_planes,
)


def _run_pipeline(scene: dict):
    """Run the existing fit_pole_axis viewing-plane pipeline on a
    synthetic scene and return (axis_point, axis_dir) plus per-frame
    plane data for diagnostics."""
    masks = scene["masks"]
    extrinsics = scene["extrinsics"]
    intrinsics = scene["intrinsics"]
    native_hw = scene["image_hw"]

    normals, offsets = [], []
    for i in range(masks.shape[0]):
        pts = fit_largest_component(masks[i])
        assert pts is not None, f"Frame {i}: empty mask"
        centroid, major, _ = mask_centerline_2d(pts)
        c_uv = native_to_pad518(centroid, native_hw)
        m_uv = native_to_pad518(centroid + major, native_hw)
        major_uv = m_uv - c_uv
        n, d = viewing_plane_normal(c_uv, major_uv, intrinsics[i], extrinsics[i])
        assert np.linalg.norm(n) > 1e-6, f"Frame {i}: degenerate plane"
        normals.append(n)
        offsets.append(d)
    axis_point, axis_dir = fit_axis_from_planes(
        np.stack(normals), np.array(offsets),
    )
    return axis_point, axis_dir


def test_synthetic_scene_has_correct_shapes():
    """Generator returns shape-correct arrays."""
    scene = make_pole_scene(n_cameras=8)
    assert scene["masks"].shape == (8, 518, 518)
    assert scene["masks"].dtype == bool
    assert scene["extrinsics"].shape == (8, 3, 4)
    assert scene["intrinsics"].shape == (8, 3, 3)
    assert scene["gt_top_xyz"].shape == (3,)
    assert scene["gt_bottom_xyz"].shape == (3,)


def test_synthetic_scene_has_visible_pole_in_every_frame():
    """A pole at the centre of an 8-camera ring should be visible in
    every frame (mask not empty)."""
    scene = make_pole_scene(n_cameras=8)
    masks = scene["masks"]
    for i in range(8):
        assert masks[i].sum() > 100, (
            f"Frame {i} has only {masks[i].sum()} mask pixels — "
            f"pole must be visible in every frame for the ring camera "
            f"setup."
        )


def test_planted_pole_is_vertical_in_world_frame():
    """The synthetic generator's gt direction must be (0, 1, 0)."""
    scene = make_pole_scene()
    gt_dir = scene["gt_axis_dir"]
    assert np.allclose(gt_dir, [0.0, 1.0, 0.0]), (
        f"Planted axis direction is {gt_dir}, expected (0,1,0)"
    )


def test_axis_direction_recovered_within_one_milliradian():
    """Run the viewing-plane axis fit on a noiseless synthetic scene;
    the recovered direction must match the planted direction (up to
    sign) to within 0.001 rad ≈ 0.057°."""
    scene = make_pole_scene(noise_px=0.0)
    axis_point, axis_dir = _run_pipeline(scene)
    gt_dir = scene["gt_axis_dir"]
    cos = abs(float(np.dot(axis_dir, gt_dir)))
    angle = np.arccos(min(cos, 1.0))
    assert angle < 1e-3, (
        f"Axis-direction error {np.degrees(angle):.4f}° exceeds 0.057° "
        f"tolerance. Recovered {axis_dir}, planted {gt_dir}."
    )


def test_planted_top_lies_on_recovered_axis_within_one_mm():
    """The 3D top of the planted pole must lie within 1 mm of the
    recovered axis line."""
    scene = make_pole_scene(noise_px=0.0)
    axis_point, axis_dir = _run_pipeline(scene)
    gt_top = scene["gt_top_xyz"]
    # Distance from gt_top to the line through (axis_point, axis_dir).
    perp = np.cross(gt_top - axis_point, axis_dir)
    dist = float(np.linalg.norm(perp) / max(np.linalg.norm(axis_dir), 1e-12))
    assert dist < 1e-3, (
        f"Planted top deviates {dist*1000:.2f} mm from the recovered "
        f"axis (tolerance 1 mm)."
    )


def test_planted_bottom_lies_on_recovered_axis_within_one_mm():
    """Same but for the planted pole bottom."""
    scene = make_pole_scene(noise_px=0.0)
    axis_point, axis_dir = _run_pipeline(scene)
    gt_bot = scene["gt_bottom_xyz"]
    perp = np.cross(gt_bot - axis_point, axis_dir)
    dist = float(np.linalg.norm(perp) / max(np.linalg.norm(axis_dir), 1e-12))
    assert dist < 1e-3, (
        f"Planted bottom deviates {dist*1000:.2f} mm from the recovered "
        f"axis (tolerance 1 mm)."
    )
