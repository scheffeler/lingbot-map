"""Phase 1.5: refined pole axis via per-frame viewing-plane intersection.

Improves on `triangulate_pole.py` (which uses only the two extreme PCA
points per frame) by using each frame's full mask centerline as a 2D
line, which back-projects to a 3D plane that contains the pole axis.
The axis is then the line that lies in *all* such planes:
  * direction = right null space of the stacked plane normals (SVD)
  * position  = least-norm point satisfying n_i . p = d_i for all i

This handles partial-mask frames cleanly: a frame that only sees the
bottom half of the pole still constrains the axis to lie in its
viewing plane, even though its mask endpoints are biased. The previous
endpoint-based triangulator would pull the recovered "top" downward
on those frames.

Outputs the same JSON schema as triangulate_pole.py (so visualize_pole.py
consumes it unchanged), plus:
  * axis_lean_deg  — angle between the fitted axis and the ground-plane
                     normal computed from the dense PLY (if available)
  * n_planes_used  — frames whose mask was usable for the plane fit

Run:
    python scripts/fit_pole_axis.py \\
        --masks pole_001.masks.npz \\
        --poses pole_001.poses.npz \\
        --object-id 2 \\
        --gps-scale pole_001.gps_scale.json \\
        --ply pole_001.ply \\
        --output pole_001.triangulation.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Reuse helpers from the legacy triangulator so the per-frame mask
# preprocessing stays identical.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triangulate_pole import (  # noqa: E402
    fit_largest_component,
    native_to_pad518,
    pixel_to_world_ray,
)


def mask_centerline_2d(pts_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """PCA on mask pixels: returns (centroid (2,), major_dir (2,) unit,
    half_extent float along the major axis)."""
    centroid = pts_xy.mean(axis=0)
    centered = pts_xy - centroid
    _, sigma, vt = np.linalg.svd(centered, full_matrices=False)
    major = vt[0]
    proj = centered @ major
    half_extent = float(0.5 * (proj.max() - proj.min()))
    return centroid, major, half_extent


def viewing_plane_normal(
    centroid_uv: np.ndarray, major_uv: np.ndarray,
    K: np.ndarray, E_c2w: np.ndarray,
) -> tuple[np.ndarray, float]:
    """For a 2D line passing through `centroid_uv` with direction
    `major_uv` in pixel space, return the world-frame plane (n, d) such
    that the plane contains the camera centre and back-projects the 2D
    line. n is unit-norm; the plane equation is n . x = d."""
    p1 = centroid_uv
    p2 = centroid_uv + major_uv  # any second point on the 2D line
    o, r1 = pixel_to_world_ray(p1, K, E_c2w)
    _, r2 = pixel_to_world_ray(p2, K, E_c2w)
    n = np.cross(r1, r2)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        # Degenerate (rays parallel — should never happen for a real
        # 2D line in pixel space). Return zero plane to be filtered out.
        return np.zeros(3), 0.0
    n = n / norm
    d = float(n @ o)            # camera centre lies in the plane
    return n, d


def fit_axis_from_planes(
    normals: np.ndarray, offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Given N viewing-plane (normal, offset) pairs, return (axis_point,
    axis_dir) for the line that lies in all planes (least-squares).
      axis_dir = right null space of the stacked normals (smallest SVD).
      axis_point = least-norm solution of normals @ p = offsets, then
                   reduced to its component perpendicular to axis_dir."""
    _, sigma, vt = np.linalg.svd(normals, full_matrices=False)
    axis_dir = vt[-1]
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    # Least-norm point satisfying N p = d (pseudoinverse).
    axis_point = np.linalg.pinv(normals) @ offsets
    # Subtract the component along axis_dir so axis_point is the closest
    # point on the line to the origin — gives a stable canonical anchor.
    axis_point = axis_point - axis_dir * (axis_dir @ axis_point)
    return axis_point.astype(np.float64), axis_dir.astype(np.float64)


def project_ray_to_axis_t(
    ray_o: np.ndarray, ray_d: np.ndarray,
    axis_point: np.ndarray, axis_dir: np.ndarray,
) -> float:
    """For a ray and an axis line, return the t-value along the axis
    where the ray and the axis come closest. Used to convert per-frame
    mask top/bottom pixels into "distance along axis" values that we
    take percentiles of for the final pole top/bottom."""
    a = axis_dir
    b = ray_d
    w0 = axis_point - ray_o
    a_dot_a = float(a @ a)
    b_dot_b = float(b @ b)
    a_dot_b = float(a @ b)
    a_dot_w0 = float(a @ w0)
    b_dot_w0 = float(b @ w0)
    denom = a_dot_a * b_dot_b - a_dot_b ** 2
    if abs(denom) < 1e-9:
        # Parallel — pick any point.
        return 0.0
    t_axis = (a_dot_b * b_dot_w0 - b_dot_b * a_dot_w0) / denom
    return float(t_axis)


def _per_frame_planes(
    masks, extrinsics, intrinsics, native_hw,
):
    """Run the per-frame mask → viewing-plane step once and cache the
    plane (n, d) plus the per-frame mask endpoints used for top/bottom
    triangulation. Bootstrap iterations resample these cached entries
    rather than re-segmenting each iteration."""
    S = min(masks.shape[0], extrinsics.shape[0])
    cache = []                    # one entry per usable frame
    for i in range(S):
        pts = fit_largest_component(masks[i])
        if pts is None or pts.shape[0] < 50:
            continue
        centroid, major, _ = mask_centerline_2d(pts)
        c_uv = native_to_pad518(centroid, native_hw)
        m_uv = native_to_pad518(centroid + major, native_hw)
        major_uv = m_uv - c_uv
        if np.linalg.norm(major_uv) < 1e-3:
            continue
        K = intrinsics[i]
        E = extrinsics[i]
        n, d = viewing_plane_normal(c_uv, major_uv, K, E)
        if np.linalg.norm(n) < 1e-9:
            continue
        proj = (pts - centroid) @ major
        p_min_native = pts[int(np.argmin(proj))]
        p_max_native = pts[int(np.argmax(proj))]
        p_min_uv = native_to_pad518(p_min_native, native_hw)
        p_max_uv = native_to_pad518(p_max_native, native_hw)
        o_min, d_min = pixel_to_world_ray(p_min_uv, K, E)
        o_max, d_max = pixel_to_world_ray(p_max_uv, K, E)
        cache.append({
            "frame": i, "n": n, "d": d,
            "ray_min_o": o_min, "ray_min_d": d_min,
            "ray_max_o": o_max, "ray_max_d": d_max,
            "K": K, "E": E,
        })
    return cache


def _fit_height_and_lean(per_frame_cache, indices, ref_up: np.ndarray):
    """Given a subset of per-frame entries, run the axis fit and
    return (height_units, lean_deg) where lean is against `ref_up`."""
    chosen = [per_frame_cache[i] for i in indices]
    if len(chosen) < 4:
        return None
    normals = np.stack([c["n"] for c in chosen])
    offsets = np.array([c["d"] for c in chosen])
    axis_point, axis_dir = fit_axis_from_planes(normals, offsets)
    t_top_vals = []
    t_bot_vals = []
    for c in chosen:
        t_top_vals.append(project_ray_to_axis_t(
            c["ray_min_o"], c["ray_min_d"], axis_point, axis_dir))
        t_bot_vals.append(project_ray_to_axis_t(
            c["ray_max_o"], c["ray_max_d"], axis_point, axis_dir))
    t_top = float(np.median(t_top_vals))
    t_bot = float(np.median(t_bot_vals))
    p_a = axis_point + axis_dir * t_top
    p_b = axis_point + axis_dir * t_bot
    height = float(np.linalg.norm(p_a - p_b))
    if height < 1e-9:
        return None
    signed = (p_a - p_b) / height
    if signed @ ref_up < 0:
        signed = -signed
    cos = float(abs(signed @ ref_up))
    lean = float(np.degrees(np.arccos(min(cos, 1.0))))
    return height, lean


def bootstrap_axis_fit(
    masks: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    native_hw,
    n_iter: int = 200,
    scale: float | None = None,
    percentiles: tuple[float, float] = (5.0, 95.0),
    seed: int = 0,
) -> dict:
    """Bootstrap-resample frames with replacement, refit the axis,
    return CI percentiles for height (in metres if `scale` provided,
    else model units) and lean degrees.

    `scale` is the metres-per-model-unit factor from the GPS
    Sim(3) / fused scale solver; when None, the height CI stays in
    model units (and is labelled with `_units` instead of `_m`)."""
    cache = _per_frame_planes(masks, extrinsics, intrinsics, native_hw)
    if len(cache) < 4:
        raise ValueError(
            f"need >=4 usable frames for bootstrap, got {len(cache)}"
        )
    cam_ups = -extrinsics[:, :3, 1]
    ref_up = cam_ups.mean(axis=0)
    ref_up = ref_up / max(np.linalg.norm(ref_up), 1e-9)

    rng = np.random.default_rng(seed)
    heights = []
    leans = []
    n = len(cache)
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        out = _fit_height_and_lean(cache, idx, ref_up)
        if out is None:
            continue
        h, lean = out
        heights.append(h * scale if scale is not None else h)
        leans.append(lean)
    if not heights:
        raise ValueError("bootstrap produced zero successful fits")
    p_lo, p_hi = percentiles
    h_lo, h_hi = np.percentile(heights, [p_lo, p_hi])
    l_lo, l_hi = np.percentile(leans, [p_lo, p_hi])
    h_key = "height_m_ci" if scale is not None else "height_units_ci"
    return {
        h_key: [float(h_lo), float(h_hi)],
        "lean_deg_ci": [float(l_lo), float(l_hi)],
        "n_iter": int(n_iter),
        "n_successful": int(len(heights)),
        "n_frames_pool": int(n),
        "percentiles": [float(p_lo), float(p_hi)],
        "median_height": float(np.median(heights)),
        "median_lean_deg": float(np.median(leans)),
    }


def fit_ground_normal_from_ply(ply_path: Path) -> np.ndarray | None:
    """RANSAC plane on the lowest-Y subset of the dense PLY. Returns
    the unit normal pointing roughly opposite the cameras (i.e., 'up'
    if the cloud has a clear ground), or None if no plausible plane."""
    try:
        # Reuse the loader + plane fit from visualize_pole.
        from visualize_pole import load_ply_binary, fit_ground_plane
    except Exception:
        return None
    pts, _ = load_ply_binary(ply_path)
    plane = fit_ground_plane(pts.astype(np.float64),
                             np.zeros((1, 3), dtype=np.float64))
    if plane is None:
        return None
    normal, _, _ = plane
    return normal


def main():
    ap = argparse.ArgumentParser(
        description="Phase 1.5 refined pole axis (viewing-plane intersection)"
    )
    ap.add_argument("--masks", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--object-id", type=int, default=0,
                    help="Multi-object masks: which tracked object to fit.")
    ap.add_argument("--gps-scale", default=None)
    ap.add_argument("--ply", default=None,
                    help="If provided, fit ground plane and report lean.")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="Bootstrap iterations for height/lean CI; "
                         "0 disables. 200 is a reasonable default.")
    ap.add_argument("--output", default="pole_axis.json")
    args = ap.parse_args()

    masks_data = np.load(args.masks, allow_pickle=True)
    poses_data = np.load(args.poses, allow_pickle=True)
    masks = masks_data["masks"]
    if masks.ndim == 4:
        n_obj = masks.shape[0]
        if args.object_id >= n_obj:
            sys.exit(f"--object-id {args.object_id} out of range; have {n_obj}")
        obj_ids = (masks_data["obj_ids"]
                   if "obj_ids" in masks_data.files else None)
        oid = (int(obj_ids[args.object_id]) if obj_ids is not None
               else args.object_id)
        print(f"Multi-object masks: using object {args.object_id} (id={oid})")
        masks = masks[args.object_id]
    native_hw = masks_data["image_hw"]
    extrinsics = poses_data["extrinsic"]
    intrinsics = poses_data["intrinsic"]
    pad_hw = poses_data["image_hw"]

    S = min(masks.shape[0], extrinsics.shape[0])
    print(f"Frames: {S}; native hw {tuple(native_hw)}; "
          f"recon hw {tuple(pad_hw)}")

    normals = []
    offsets = []
    per_frame_endpoints_native = []
    used_frames = []
    for i in range(S):
        pts = fit_largest_component(masks[i])
        if pts is None or pts.shape[0] < 50:
            per_frame_endpoints_native.append(None)
            continue
        centroid, major, half_extent = mask_centerline_2d(pts)
        # Convert centroid + a second point on the line from native
        # mask resolution to the 518-pixel pad space the intrinsics
        # match.
        c_uv = native_to_pad518(centroid, native_hw)
        m_uv = native_to_pad518(centroid + major, native_hw)
        major_uv = m_uv - c_uv
        if np.linalg.norm(major_uv) < 1e-3:
            per_frame_endpoints_native.append(None)
            continue
        K = intrinsics[i]
        E = extrinsics[i]
        n, d = viewing_plane_normal(c_uv, major_uv, K, E)
        if np.linalg.norm(n) < 1e-9:
            per_frame_endpoints_native.append(None)
            continue
        normals.append(n)
        offsets.append(d)
        used_frames.append(i)

        # Cache PCA endpoints (top/bottom of mask along the major axis)
        # in native mask pixel space — used later to bound the pole's
        # extent along the fitted axis.
        proj = (pts - centroid) @ major
        p_min = pts[int(np.argmin(proj))]
        p_max = pts[int(np.argmax(proj))]
        per_frame_endpoints_native.append((p_min, p_max))

    if len(normals) < 4:
        sys.exit(f"Only {len(normals)} usable frame(s); need 4+ for axis fit.")

    normals_arr = np.stack(normals)
    offsets_arr = np.array(offsets)
    axis_point, axis_dir = fit_axis_from_planes(normals_arr, offsets_arr)
    print(f"\nFitted pole axis:")
    print(f"  direction = {axis_dir}")
    print(f"  point on axis (closest to origin) = {axis_point}")

    # Convert per-frame mask endpoints into axis-t values.
    t_vals_top = []
    t_vals_bot = []
    for i, endpts in enumerate(per_frame_endpoints_native):
        if endpts is None:
            continue
        p_min_native, p_max_native = endpts
        p_min_uv = native_to_pad518(p_min_native, native_hw)
        p_max_uv = native_to_pad518(p_max_native, native_hw)
        K = intrinsics[i]
        E = extrinsics[i]
        o_min, d_min = pixel_to_world_ray(p_min_uv, K, E)
        o_max, d_max = pixel_to_world_ray(p_max_uv, K, E)
        t_min = project_ray_to_axis_t(o_min, d_min, axis_point, axis_dir)
        t_max = project_ray_to_axis_t(o_max, d_max, axis_point, axis_dir)
        # `p_min` (smaller image y in the original axis_endpoints code) was
        # treated as the "top" — preserve that ordering. Whichever t is
        # larger here corresponds to "further along axis_dir from
        # axis_point"; we resolve the top/bottom convention below.
        t_vals_top.append(t_min)
        t_vals_bot.append(t_max)

    # Use the median across frames so a single bad mask doesn't drag the
    # result. We then identify which side (more positive t or more
    # negative t) corresponds to the "top" by checking which is more in
    # the +up direction in lingbot-map's world frame: lingbot-map uses
    # camera +Y down, so world -Y is roughly "up" for typical captures.
    t_top = float(np.median(t_vals_top))
    t_bot = float(np.median(t_vals_bot))
    if t_top < t_bot:
        # Already ordered: smaller t = lower along axis_dir; pick the
        # convention that "top" has larger world-up component.
        pass

    pole_a = axis_point + axis_dir * t_top
    pole_b = axis_point + axis_dir * t_bot
    # Resolve top vs bottom by world-up: the endpoint with the
    # *smaller* world-y is the top (lingbot-map y-down convention).
    if pole_a[1] > pole_b[1]:
        pole_top, pole_bot = pole_b, pole_a
        # also flip axis to point "up"
        signed_dir = (pole_top - pole_bot)
        signed_dir = signed_dir / max(np.linalg.norm(signed_dir), 1e-9)
    else:
        pole_top, pole_bot = pole_a, pole_b
        signed_dir = (pole_top - pole_bot)
        signed_dir = signed_dir / max(np.linalg.norm(signed_dir), 1e-9)

    height = float(np.linalg.norm(pole_top - pole_bot))
    print(f"\nPole top:    {pole_top}")
    print(f"Pole bottom: {pole_bot}")
    print(f"Height:      {height:.4f} model units")

    metric_scale = None
    if args.gps_scale and Path(args.gps_scale).exists():
        gs = json.loads(Path(args.gps_scale).read_text())
        metric_scale = float(gs["scale"])
        print(f"Height:      {height * metric_scale:.3f} m  (metric)")

    lean_deg = None
    if args.ply and Path(args.ply).exists():
        ground_normal = fit_ground_normal_from_ply(Path(args.ply))
        if ground_normal is not None:
            cos = abs(float(signed_dir @ ground_normal))
            lean_deg = float(np.degrees(np.arccos(min(cos, 1.0))))
            # `cos == 1` would mean axis perfectly aligned with ground
            # normal == 0 deg lean. Print the deviation (lean).
            print(f"Ground normal: {ground_normal}")
            print(f"Pole lean:     {lean_deg:.2f} deg from ground normal")

    # Alternative lean reference: average camera-up vector across all
    # poses. People hold phones roughly upright; averaging over 8+
    # frames is a more stable vertical than a noisy ground-plane fit
    # on a sparse cloud. lingbot-map uses the OpenCV/COLMAP
    # convention where the camera's +Y axis points DOWN in image
    # space, so world -Y_cam (== -ext[:3, 1]) is "up" from the
    # camera's frame.
    cam_ups = -extrinsics[:, :3, 1]                 # (S, 3)
    cam_up_mean = cam_ups.mean(axis=0)
    cam_up_mean = cam_up_mean / max(np.linalg.norm(cam_up_mean), 1e-9)
    cos_cam = abs(float(signed_dir @ cam_up_mean))
    lean_cam_deg = float(np.degrees(np.arccos(min(cos_cam, 1.0))))
    print(f"Mean cam-up:   {cam_up_mean}")
    print(f"Pole lean:     {lean_cam_deg:.2f} deg from mean camera-up")

    ci = None
    if args.bootstrap > 0:
        print(f"\nRunning bootstrap with {args.bootstrap} iterations...")
        try:
            ci = bootstrap_axis_fit(
                masks=masks,
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                native_hw=native_hw,
                n_iter=args.bootstrap,
                scale=metric_scale,
            )
            unit = "m" if metric_scale is not None else "model units"
            h_key = "height_m_ci" if metric_scale is not None else "height_units_ci"
            print(f"  height 90 % CI: {ci[h_key][0]:.3f} – "
                  f"{ci[h_key][1]:.3f} {unit}  (n={ci['n_successful']})")
            print(f"  lean   90 % CI: {ci['lean_deg_ci'][0]:.2f} – "
                  f"{ci['lean_deg_ci'][1]:.2f} deg")
        except ValueError as e:
            print(f"  bootstrap failed: {e}")

    out = {
        "pole_top_xyz": pole_top.tolist(),
        "pole_bottom_xyz": pole_bot.tolist(),
        "axis_direction": signed_dir.tolist(),
        "height_model_units": height,
        "height_m": height * metric_scale if metric_scale else None,
        "metric_scale": metric_scale,
        "axis_lean_deg": lean_deg,
        "axis_lean_cam_up_deg": lean_cam_deg,
        "n_planes_used": int(len(normals)),
        "frames_used": int(len(normals)),
        "frames_total": int(S),
        "object_id": int(args.object_id),
        "method": "viewing_plane_intersection_phase_1_5",
        "native_hw": [int(x) for x in native_hw],
        "pad_hw": [int(x) for x in pad_hw],
        "ci": ci,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
