"""Phase 1.4b: scale-prior solver.

Each scale "prior" is reduced to a single `ScaleObservation` — an
implied metres-per-model-unit value plus an uncertainty on that
value. The fused estimator is weighted least squares assuming the
observations are independent and Gaussian:
    s_hat = sum(s_i / sigma_i^2) / sum(1 / sigma_i^2)
    sigma_hat = 1 / sqrt(sum(1 / sigma_i^2))

Two priors are concrete today; more (`PoleDiameterPrior`,
`VlmScaleCandidatesPrior`, ...) can be added by writing another
`*_observation()` helper that returns a `ScaleObservation`.

Run as a CLI:
    python scripts/scale_solver.py \\
        --gps pole_001.gps.json --poses pole_001.poses.npz \\
        --ref-pixels "0,850,300,920,800;4,820,310,950,790" \\
        --ref-length-m 2.44 \\
        --out pole_001.scale.json

`--ref-pixels` is `frame,top_x,top_y,bot_x,bot_y;frame,top_x,...` —
each segment names a frame and the two pixel coords of the reference
object's endpoints in that frame.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Import sibling helpers without depending on package install.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from triangulate_pole import (  # noqa: E402
    closest_point_to_rays,
    native_to_pad518,
    pixel_to_world_ray,
)


@dataclass
class ScaleObservation:
    """A single scale observation: an implied metres-per-model-unit
    value plus an uncertainty on that value. Solvers fuse these via
    weighted least squares (see `solve_fused`)."""
    name: str
    scale: float
    sigma: float
    metadata: dict[str, Any] = field(default_factory=dict)


def reference_length_observation(
    pixel_pairs: Sequence[tuple[int, np.ndarray, np.ndarray]],
    length_m: float,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    native_hw: tuple[int, int],
    pixel_sigma_px: float = 1.0,
    sigma_floor_rel: float = 0.005,
) -> ScaleObservation:
    """Triangulate the two endpoints of a known-length reference object
    from `pixel_pairs` (a list of `(frame_idx, top_uv, bot_uv)` tuples
    naming where the user marked the two ends in each frame), then
    compute the implied scale `length_m / measured_model_units`.

    Needs >=2 distinct frames to triangulate; ideally 3-4 frames spaced
    around the object for conditioning. Sigma is propagated from a
    nominal pixel-error magnitude (`pixel_sigma_px`) through the
    triangulation conditioning, with a floor at `sigma_floor_rel *
    scale` to avoid pathologically small uncertainties.
    """
    if len(pixel_pairs) < 2:
        raise ValueError(
            f"Need >=2 marked frames to triangulate a reference "
            f"object; got {len(pixel_pairs)}."
        )

    top_origins, top_dirs = [], []
    bot_origins, bot_dirs = [], []
    for frame_idx, top_uv_native, bot_uv_native in pixel_pairs:
        top_uv = native_to_pad518(np.asarray(top_uv_native), native_hw)
        bot_uv = native_to_pad518(np.asarray(bot_uv_native), native_hw)
        K = intrinsics[frame_idx]
        E = extrinsics[frame_idx]
        o_t, d_t = pixel_to_world_ray(top_uv, K, E)
        o_b, d_b = pixel_to_world_ray(bot_uv, K, E)
        top_origins.append(o_t); top_dirs.append(d_t)
        bot_origins.append(o_b); bot_dirs.append(d_b)

    p_top = closest_point_to_rays(np.array(top_origins), np.array(top_dirs))
    p_bot = closest_point_to_rays(np.array(bot_origins), np.array(bot_dirs))
    measured_units = float(np.linalg.norm(p_top - p_bot))
    if measured_units < 1e-9:
        raise ValueError(
            "Reference endpoints triangulated to the same point. "
            "Pixel marks may be co-linear with the camera rays."
        )
    scale = length_m / measured_units

    # Crude sigma propagation: a `pixel_sigma_px` shift on each marked
    # point shifts the triangulated point by ~ pixel_sigma_px * Z / f
    # where Z is the typical viewing distance and f the focal length.
    # Aggregate over both endpoints and convert to a relative scale
    # error.
    f_mean = float(np.mean([intrinsics[fi][0, 0] for fi, _, _ in pixel_pairs]))
    z_typical = float(np.median([
        np.linalg.norm(extrinsics[fi][:3, 3] - 0.5 * (p_top + p_bot))
        for fi, _, _ in pixel_pairs
    ]))
    pos_sigma_units = pixel_sigma_px * z_typical / max(f_mean, 1e-6)
    rel_sigma = (pos_sigma_units / max(measured_units, 1e-6)) * np.sqrt(2.0)
    sigma = max(scale * rel_sigma, scale * sigma_floor_rel)

    return ScaleObservation(
        name="reference_length",
        scale=scale,
        sigma=sigma,
        metadata={
            "length_m": float(length_m),
            "measured_model_units": measured_units,
            "p_top_world": p_top.tolist(),
            "p_bot_world": p_bot.tolist(),
            "n_frames": int(len(pixel_pairs)),
            "pixel_sigma_px": float(pixel_sigma_px),
        },
    )


def gps_sim3_observation(
    gps_entries: list[dict],
    cam_centres: np.ndarray,
    cam_image_paths: list[str],
    sigma_floor_rel: float = 0.05,
) -> ScaleObservation:
    """Wrap `solve_gps_scale` and turn its (scale, residual_m,
    baseline_m) triple into a ScaleObservation. Sigma is the
    per-photo residual divided by the baseline, scaled to the same
    units as `scale`."""
    # Local import to keep the test path light when GPS isn't used.
    from scale_from_gps import solve_gps_scale
    result = solve_gps_scale(gps_entries, cam_centres, cam_image_paths)
    scale = float(result["scale"])
    rel_sigma = float(result["residual_m"]) / max(float(result["baseline_m"]), 1e-6)
    sigma = max(scale * rel_sigma, scale * sigma_floor_rel)
    return ScaleObservation(
        name="gps_sim3",
        scale=scale,
        sigma=sigma,
        metadata=result,
    )


def solve_fused(observations: list[ScaleObservation]) -> dict:
    """Weighted-least-squares fusion of independent scale observations.

    Returns a dict with `scale`, `sigma`, and `priors` (per-observation
    debug info)."""
    if not observations:
        raise ValueError("solve_fused needs at least one observation")
    if len(observations) == 1:
        o = observations[0]
        return {
            "scale": float(o.scale),
            "sigma": float(o.sigma),
            "priors": [_obs_dict(o)],
            "method": "single_prior_passthrough",
        }
    weights = np.array([1.0 / max(o.sigma, 1e-12) ** 2 for o in observations])
    scales = np.array([o.scale for o in observations])
    s_hat = float(np.sum(scales * weights) / np.sum(weights))
    sigma_hat = float(1.0 / np.sqrt(np.sum(weights)))
    return {
        "scale": s_hat,
        "sigma": sigma_hat,
        "priors": [_obs_dict(o) for o in observations],
        "method": "weighted_least_squares",
    }


def _obs_dict(o: ScaleObservation) -> dict:
    return {
        "name": o.name,
        "scale": float(o.scale),
        "sigma": float(o.sigma),
        "metadata": o.metadata,
    }


# ---------- CLI ----------

def _parse_ref_pixels(spec: str | None) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Parse '0,850,300,920,800;4,820,310,950,790' into pixel_pairs."""
    if not spec:
        return []
    out = []
    for seg in spec.split(";"):
        parts = [p.strip() for p in seg.split(",") if p.strip()]
        if len(parts) != 5:
            raise SystemExit(
                f"--ref-pixels segment must have 5 fields "
                f"(frame,top_x,top_y,bot_x,bot_y), got {parts}"
            )
        frame_idx = int(parts[0])
        top_uv = np.array([float(parts[1]), float(parts[2])])
        bot_uv = np.array([float(parts[3]), float(parts[4])])
        out.append((frame_idx, top_uv, bot_uv))
    return out


def main():
    ap = argparse.ArgumentParser(description="Fused scale-prior solver")
    ap.add_argument("--gps", help="GPS JSON from exif_gps.py (optional)")
    ap.add_argument("--poses", required=True,
                    help="*.poses.npz from phase0")
    ap.add_argument("--ref-pixels", default=None,
                    help="'frame,top_x,top_y,bot_x,bot_y;...' — at least "
                         "two segments to triangulate the reference.")
    ap.add_argument("--ref-length-m", type=float, default=None,
                    help="Known length of the reference object in metres.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    poses = np.load(args.poses, allow_pickle=True)
    extrinsics = poses["extrinsic"]
    intrinsics = poses["intrinsic"]
    native_hw = tuple(int(x) for x in poses["image_hw"])
    image_paths = [str(p) for p in poses["image_paths"]]

    obs_list: list[ScaleObservation] = []

    if args.gps:
        gps_data = json.loads(Path(args.gps).read_text())
        cam_centres = extrinsics[:, :3, 3].astype(np.float64)
        try:
            obs_list.append(gps_sim3_observation(
                gps_data["images"], cam_centres, image_paths,
            ))
            print(f"Added GPS prior: scale={obs_list[-1].scale:.4f} "
                  f"+/- {obs_list[-1].sigma:.4f}")
        except ValueError as e:
            print(f"GPS prior skipped: {e}")

    if args.ref_pixels:
        if args.ref_length_m is None:
            sys.exit("--ref-pixels requires --ref-length-m")
        pixel_pairs = _parse_ref_pixels(args.ref_pixels)
        obs_list.append(reference_length_observation(
            pixel_pairs=pixel_pairs,
            length_m=args.ref_length_m,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            native_hw=native_hw,
        ))
        print(f"Added reference prior: scale={obs_list[-1].scale:.4f} "
              f"+/- {obs_list[-1].sigma:.4f}")

    if not obs_list:
        sys.exit("No priors specified. Pass --gps and/or --ref-pixels.")

    fused = solve_fused(obs_list)
    print(f"\nFused scale: {fused['scale']:.6f} +/- {fused['sigma']:.6f}  "
          f"({fused['method']})")
    Path(args.out).write_text(json.dumps(fused, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
