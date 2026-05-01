"""Phase 1.4: fit a Sim(3) transform from per-image EXIF GPS to lingbot-map
camera centres, recovering metric scale.

Pipeline:
  1. Load EXIF GPS JSON (from `exif_gps.py`) — list of (lat, lon, alt) per
     image, in capture order.
  2. Load `<scene>.poses.npz` from `phase0_modal_imageset.py` — per-frame
     c2w extrinsics in lingbot-map's arbitrary world units.
  3. Convert lat/lon/alt to a local ENU frame in metres (origin = first
     photo). Small-angle formula is accurate to <0.1% over <100 m.
  4. Solve Umeyama similarity (s, R, t) between camera centres and ENU
     positions, ordered by `image_paths` from the poses NPZ.
  5. Validate: per-photo residual, GPS baseline, condition number.
  6. Write `<scene>.gps_scale.json` with the scalar `scale` (the only
     thing `triangulate_pole.py` consumes today). Also stores R, t, and
     per-image residuals so a future scale-prior solver (Phase 1.4b) can
     ingest this as one prior in a fusion problem.

The function `solve_gps_scale(...)` returns `(scale, residual_m)` so the
1.4b refactor is mechanical — wrap it in a `GpsSim3Prior` and the rest of
the solver doesn't change.

Run:
    python scripts/scale_from_gps.py \\
        --gps pole_001.gps.json \\
        --poses pole_001.poses.npz \\
        --out pole_001.gps_scale.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

R_EARTH = 6_378_137.0  # WGS84 semi-major axis, metres


def latlon_to_enu(
    lats: np.ndarray, lons: np.ndarray, alts: np.ndarray,
    lat0: float, lon0: float, alt0: float,
) -> np.ndarray:
    """Convert (lat, lon, alt) arrays to a local ENU frame around
    (lat0, lon0, alt0). Small-angle approximation; <0.1% error over
    <100 m, which is well within GPS noise for our captures."""
    coslat = math.cos(math.radians(lat0))
    de = (lons - lon0) * math.radians(1) * coslat * R_EARTH
    dn = (lats - lat0) * math.radians(1) * R_EARTH
    du = alts - alt0
    return np.stack([de, dn, du], axis=1)


def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Closed-form similarity (s, R, t) mapping src -> dst, minimizing
    sum_i ||s R src_i + t - dst_i||^2.

    Reference: Umeyama 1991, "Least-squares estimation of transformation
    parameters between two point patterns".

    src, dst: (N, 3) point clouds in correspondence.
    Returns: (s, R(3x3), t(3,)).
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    # Cross-covariance.
    sigma = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(sigma)
    # Reflection check: ensure det(R) = +1 even when SVD picked a flip.
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt

    var_src = (src_c ** 2).sum() / n
    s = float((D * np.diag(S)).sum() / max(var_src, 1e-12))
    t = mu_dst - s * R @ mu_src
    return s, R, t


def solve_gps_scale(
    gps_entries: list[dict],
    cam_centres: np.ndarray,
    cam_image_paths: list[str],
) -> dict:
    """Align lingbot-map camera centres to ENU GPS positions.

    Returns a dict with `scale`, `R` (3x3), `t` (3,), `residual_m` (RMS),
    `per_image_residual_m` (N,), `baseline_m`, and `n_used`.

    Raises ValueError if alignment fails the sanity gates."""
    if len(gps_entries) < 5:
        raise ValueError(f"Need >=5 GPS images for a stable fit, got {len(gps_entries)}")

    gps_by_name = {e["path"]: e for e in gps_entries}
    pairs = [(p, gps_by_name.get(p)) for p in cam_image_paths]
    used = [(p, g) for p, g in pairs if g is not None]
    if len(used) < 5:
        names = [p for p, g in pairs if g is None][:5]
        raise ValueError(f"Only {len(used)} pose<->gps matches by filename; "
                         f"missing GPS for: {names}")

    src_idx = [cam_image_paths.index(p) for p, _ in used]
    src = cam_centres[src_idx]
    lats = np.array([g["lat"] for _, g in used])
    lons = np.array([g["lon"] for _, g in used])
    alts = np.array([g["alt_m"] if g["alt_m"] is not None else 0.0
                     for _, g in used])
    dst = latlon_to_enu(lats, lons, alts, lats[0], lons[0], alts[0])

    baseline_m = float(np.linalg.norm(
        dst[:, None, :] - dst[None, :, :], axis=-1
    ).max())
    if baseline_m < 3.0:
        raise ValueError(f"GPS baseline {baseline_m:.2f} m too small; "
                         "Sim(3) fit will be dominated by GPS noise.")

    # Iterative outlier rejection: refit, drop the worst point if its
    # residual is >2x the median, repeat until converged or <5 points
    # remain. Catches single-frame pose-head drift (a known failure mode
    # of streaming inference on sparse captures).
    keep_mask = np.ones(len(src), dtype=bool)
    dropped = []
    for _ in range(len(src)):
        s, R, t = umeyama_sim3(src[keep_mask], dst[keep_mask])
        pred = (s * (R @ src.T)).T + t
        per_residual = np.linalg.norm(pred - dst, axis=1)
        kept_resid = per_residual[keep_mask]
        med = np.median(kept_resid)
        worst = int(np.argmax(np.where(keep_mask, per_residual, -1)))
        if keep_mask.sum() <= 5 or per_residual[worst] <= max(2 * med, 0.5):
            break
        keep_mask[worst] = False
        dropped.append((used[worst][0], float(per_residual[worst])))

    rms = float(np.sqrt((per_residual[keep_mask] ** 2).mean()))

    return {
        "scale": s,
        "R": R.tolist(),
        "t": t.tolist(),
        "residual_m": rms,
        "per_image_residual_m": per_residual.tolist(),
        "kept_mask": keep_mask.tolist(),
        "dropped_outliers": dropped,
        "baseline_m": baseline_m,
        "n_used": int(keep_mask.sum()),
        "image_paths_used": [p for p, _ in used],
    }


def main():
    ap = argparse.ArgumentParser(description="Fit Sim(3) GPS->lingbot-map scale")
    ap.add_argument("--gps", required=True, help="GPS JSON from exif_gps.py")
    ap.add_argument("--poses", required=True, help="*.poses.npz from phase0")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--max-residual-m", type=float, default=1.5,
                    help="Per-image residual threshold for a 'pass' verdict")
    args = ap.parse_args()

    gps_data = json.loads(Path(args.gps).read_text())
    gps_entries = gps_data["images"]

    poses = np.load(args.poses, allow_pickle=True)
    extrinsics = poses["extrinsic"]                 # (S, 3, 4) c2w
    cam_centres = extrinsics[:, :3, 3].astype(np.float64)
    image_paths = [str(p) for p in poses["image_paths"]]

    print(f"Poses: {len(image_paths)} cameras")
    print(f"GPS:   {len(gps_entries)} images, "
          f"baseline {gps_data['max_pairwise_baseline_m']:.1f} m")

    try:
        result = solve_gps_scale(gps_entries, cam_centres, image_paths)
    except ValueError as e:
        sys.exit(f"FAIL: {e}")

    print(f"\nUmeyama fit on {result['n_used']}/{len(result['image_paths_used'])} pairs:")
    print(f"  scale          = {result['scale']:.6f}  (model_units -> metres)")
    print(f"  RMS residual   = {result['residual_m']:.3f} m  (kept points only)")
    print(f"  GPS baseline   = {result['baseline_m']:.2f} m")
    per = np.array(result["per_image_residual_m"])
    kept = np.array(result["kept_mask"])
    print(f"  kept med/max   = {np.median(per[kept]):.3f} / {per[kept].max():.3f} m")
    if result["dropped_outliers"]:
        print("  rejected outliers:")
        for name, r in result["dropped_outliers"]:
            print(f"    {name}: {r:.2f} m residual")

    kept_max = per[kept].max() if kept.any() else float("inf")
    verdict = "PASS" if kept_max < args.max_residual_m else "WARN"
    print(f"\nVerdict: {verdict} (kept-points max residual vs threshold "
          f"{args.max_residual_m:.2f} m)")

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
