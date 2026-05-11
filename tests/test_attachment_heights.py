"""Phase 1.5 S3 — attachment heights along the fitted pole axis.

Each non-pole tracked SAM object is an attachment candidate (crossarm,
transformer, wire). For each frame we project the attachment mask's
centroid through the camera, find the closest point on the pole axis
to that ray, and convert the axis-t value to "metres above pole base".

Synthetic test: a 2.44 m horizontal crossarm rasterized at z=6 m above
the pole base → recovered height within 0.2 m.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fit_pole_axis import measure_attachments  # noqa: E402
from tests.synthetic import make_pole_scene
from tests.test_diameter import _fit_axis_from_scene


def test_attachment_synthetic_horizontal_bar_within_0_2m():
    scene = make_pole_scene(
        height_m=10.0, diameter_m=0.30,
        n_cameras=8, radius_m=8.0, noise_px=0.0,
        attachments=[
            {"name": "crossarm", "height_m": 6.0, "length_m": 2.44},
        ],
    )
    axis_point, axis_dir = _fit_axis_from_scene(scene)

    atts = measure_attachments(
        attachment_masks=scene["attachment_masks"],
        attachment_names=scene["attachment_names"],
        extrinsics=scene["extrinsics"],
        intrinsics=scene["intrinsics"],
        native_hw=scene["image_hw"],
        axis_point=axis_point,
        axis_dir=axis_dir,
        pole_bottom_xyz=scene["gt_bottom_xyz"],
        scale=1.0,
    )
    assert len(atts) == 1
    a = atts[0]
    assert a["name"] == "crossarm"
    assert a["height_m"] is not None
    assert a["n_frames_used"] >= 4
    assert abs(a["height_m"] - 6.0) < 0.2, (
        f"recovered crossarm height {a['height_m']:.3f} m, expected ~6.0 ±0.2"
    )


def test_attachment_two_attachments_returns_two_records():
    scene = make_pole_scene(
        height_m=10.0, diameter_m=0.30,
        n_cameras=8, radius_m=8.0, noise_px=0.0,
        attachments=[
            {"name": "crossarm", "height_m": 7.0, "length_m": 2.44},
            {"name": "transformer", "height_m": 5.0, "length_m": 0.40},
        ],
    )
    axis_point, axis_dir = _fit_axis_from_scene(scene)

    atts = measure_attachments(
        attachment_masks=scene["attachment_masks"],
        attachment_names=scene["attachment_names"],
        extrinsics=scene["extrinsics"],
        intrinsics=scene["intrinsics"],
        native_hw=scene["image_hw"],
        axis_point=axis_point,
        axis_dir=axis_dir,
        pole_bottom_xyz=scene["gt_bottom_xyz"],
        scale=1.0,
    )
    assert [a["name"] for a in atts] == ["crossarm", "transformer"]
    by_name = {a["name"]: a for a in atts}
    assert abs(by_name["crossarm"]["height_m"] - 7.0) < 0.2
    assert abs(by_name["transformer"]["height_m"] - 5.0) < 0.2


def test_attachment_empty_mask_returns_none_height():
    """An attachment object with no True pixels in any frame yields
    height=None and n_frames_used=0 (rather than crashing or guessing)."""
    scene = make_pole_scene(
        height_m=10.0, diameter_m=0.30,
        n_cameras=8, radius_m=8.0, noise_px=0.0,
        attachments=[
            {"name": "crossarm", "height_m": 6.0, "length_m": 2.44},
        ],
    )
    masks = np.zeros_like(scene["attachment_masks"])  # empty
    axis_point, axis_dir = _fit_axis_from_scene(scene)

    atts = measure_attachments(
        attachment_masks=masks,
        attachment_names=scene["attachment_names"],
        extrinsics=scene["extrinsics"],
        intrinsics=scene["intrinsics"],
        native_hw=scene["image_hw"],
        axis_point=axis_point,
        axis_dir=axis_dir,
        pole_bottom_xyz=scene["gt_bottom_xyz"],
        scale=1.0,
    )
    assert len(atts) == 1
    assert atts[0]["height_m"] is None
    assert atts[0]["n_frames_used"] == 0
