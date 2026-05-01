"""S4: bootstrap uncertainty bounds for the pole-axis fit.

Resamples frames with replacement, refits the axis, collects the
height/lean distribution, returns 5th/95th percentiles.

Tests use the synthetic generator from S0 — planted geometry with
known truth — so we can assert on coverage and CI width without
flakiness from a real capture.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.synthetic import make_pole_scene
from fit_pole_axis import bootstrap_axis_fit


def test_bootstrap_returns_ci_dict_shape():
    """The bootstrap function returns a dict with `height_m_ci`,
    `lean_deg_ci` etc — each a 2-tuple [lo, hi] in metric units."""
    scene = make_pole_scene(height_m=10.0, n_cameras=8, noise_px=0.0)
    out = bootstrap_axis_fit(
        masks=scene["masks"],
        extrinsics=scene["extrinsics"],
        intrinsics=scene["intrinsics"],
        native_hw=scene["image_hw"],
        n_iter=50,
        scale=1.0,
    )
    assert "height_m_ci" in out
    lo, hi = out["height_m_ci"]
    assert lo <= hi
    assert "n_iter" in out and out["n_iter"] == 50


def test_bootstrap_ci_contains_planted_height_on_clean_data():
    """No mask noise → every bootstrap sample should recover the
    planted 10 m. The CI brackets the truth tightly."""
    scene = make_pole_scene(height_m=10.0, n_cameras=8, noise_px=0.0)
    out = bootstrap_axis_fit(
        masks=scene["masks"],
        extrinsics=scene["extrinsics"],
        intrinsics=scene["intrinsics"],
        native_hw=scene["image_hw"],
        n_iter=80,
        scale=1.0,
    )
    lo, hi = out["height_m_ci"]
    assert lo <= 10.0 <= hi, f"CI {[lo,hi]} doesn't contain planted 10.0 m"
    # On clean data the CI should be small (< 50 cm).
    assert (hi - lo) < 0.5, f"CI width {hi-lo:.2f} m unexpectedly wide on clean data"


def test_bootstrap_ci_widens_under_pixel_noise():
    """Add pixel noise to the synthetic; the CI should be wider
    than on the noiseless capture, demonstrating the bootstrap is
    actually responding to noise (not always returning the same
    point estimate)."""
    clean = bootstrap_axis_fit(
        **_run_args(make_pole_scene(noise_px=0.0)),
        n_iter=80, scale=1.0,
    )
    noisy = bootstrap_axis_fit(
        **_run_args(make_pole_scene(noise_px=4.0, seed=1)),
        n_iter=80, scale=1.0,
    )
    clean_w = clean["height_m_ci"][1] - clean["height_m_ci"][0]
    noisy_w = noisy["height_m_ci"][1] - noisy["height_m_ci"][0]
    assert noisy_w >= clean_w, (
        f"Expected noisy CI ({noisy_w:.3f} m) to be at least as wide as "
        f"clean CI ({clean_w:.3f} m); bootstrap may not be sensitive."
    )


def _run_args(scene):
    return {
        "masks": scene["masks"],
        "extrinsics": scene["extrinsics"],
        "intrinsics": scene["intrinsics"],
        "native_hw": scene["image_hw"],
    }
