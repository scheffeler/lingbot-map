"""S1 tests: reference-length scale prior + fused multi-prior solver.

The reference tap is the cheapest path to engineering-grade scale on
existing captures: user marks a known length (crossarm, tape) in two
or more frames, the solver triangulates the 3D segment and recovers
the model-to-metres ratio from `length_m / measured_model_units`.

Tests:
  1. Synthetic-only sanity: a known 10 m segment marked in two frames
     with no noise produces scale = 1.0 to within 0.1 %.
  2. Two-prior fusion: noisy GPS (5 % bias) + clean reference; the
     fused estimate is closer to truth than the GPS alone.
  3. Pixel noise: 2 px Gaussian jitter on the marked points produces
     a scale within 1 % of truth (cheap reality check on the
     triangulation conditioning).
"""

import numpy as np
import pytest

from tests.synthetic import make_pole_scene
from scripts.scale_solver import (
    ScaleObservation,
    reference_length_observation,
    solve_fused,
)


def test_reference_length_recovers_unit_scale_synthetic():
    """Plant a 10 m pole, mark its top + bottom in two frames, expect
    `scale == 1.0` since the synthetic geometry already lives in
    metric world units."""
    scene = make_pole_scene(height_m=10.0, n_cameras=8, noise_px=0.0)
    # Use frames 0 and 4 — opposite sides of the ring → wide baseline,
    # well-conditioned triangulation.
    fa, fb = 0, 4
    obs = reference_length_observation(
        pixel_pairs=[
            (fa, scene["gt_top_uv"][fa], scene["gt_bot_uv"][fa]),
            (fb, scene["gt_top_uv"][fb], scene["gt_bot_uv"][fb]),
        ],
        length_m=scene["gt_height_m"],
        extrinsics=scene["extrinsics"],
        intrinsics=scene["intrinsics"],
        native_hw=scene["image_hw"],
    )
    assert obs.name == "reference_length"
    assert abs(obs.scale - 1.0) < 1e-3, (
        f"Recovered scale {obs.scale:.6f} differs from 1.0 by more "
        f"than 0.1 %; check the ref-tap triangulation."
    )


def test_fused_solver_pulls_toward_lower_sigma():
    """Two scale observations: a 'noisy GPS' biased 5 % low, and a
    clean reference. The fused scale must lie *closer to 1.0* (the
    reference) than to 0.95 (the GPS bias)."""
    gps_obs = ScaleObservation(
        name="gps_sim3", scale=0.95, sigma=0.10, metadata={},
    )
    ref_obs = ScaleObservation(
        name="reference_length", scale=1.00, sigma=0.01, metadata={},
    )
    fused = solve_fused([gps_obs, ref_obs])
    assert "scale" in fused and "sigma" in fused
    assert abs(fused["scale"] - 1.00) < abs(fused["scale"] - 0.95), (
        f"Fused scale {fused['scale']:.4f} is closer to GPS (0.95) "
        f"than to the lower-sigma reference (1.00). The weighting is "
        f"wrong."
    )
    # And it should land close to the reference (within ~3 % of 1.0
    # given the 1 % vs 10 % sigma split).
    assert abs(fused["scale"] - 1.00) < 0.03


def test_reference_length_is_robust_to_small_pixel_noise():
    """2 px Gaussian noise on the marked points yields a scale within
    1 % of truth — sanity check that the triangulation isn't
    pathologically sensitive."""
    rng = np.random.default_rng(42)
    scene = make_pole_scene(height_m=10.0, n_cameras=8, noise_px=0.0)
    fa, fb = 0, 4
    top_a = scene["gt_top_uv"][fa] + rng.normal(0, 2.0, 2)
    bot_a = scene["gt_bot_uv"][fa] + rng.normal(0, 2.0, 2)
    top_b = scene["gt_top_uv"][fb] + rng.normal(0, 2.0, 2)
    bot_b = scene["gt_bot_uv"][fb] + rng.normal(0, 2.0, 2)
    obs = reference_length_observation(
        pixel_pairs=[(fa, top_a, bot_a), (fb, top_b, bot_b)],
        length_m=scene["gt_height_m"],
        extrinsics=scene["extrinsics"],
        intrinsics=scene["intrinsics"],
        native_hw=scene["image_hw"],
    )
    assert abs(obs.scale - 1.0) < 0.01, (
        f"With 2 px noise, recovered scale {obs.scale:.4f} drifted "
        f"more than 1 % from 1.0."
    )


def test_solve_fused_with_single_prior_returns_that_prior():
    """Edge case: a single observation should pass through unchanged."""
    obs = ScaleObservation(name="only", scale=3.14, sigma=0.5, metadata={})
    fused = solve_fused([obs])
    assert abs(fused["scale"] - 3.14) < 1e-9
    assert abs(fused["sigma"] - 0.5) < 1e-9


def test_solve_fused_rejects_empty_input():
    """Zero observations is a programming error — solver should raise."""
    with pytest.raises(ValueError):
        solve_fused([])
