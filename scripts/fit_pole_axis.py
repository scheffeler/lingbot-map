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


def _project_world_to_pixel(P: np.ndarray, K: np.ndarray,
                            E_c2w: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Project a world point through a c2w extrinsic + intrinsic.
    Returns (pixel_uv, depth) in the image plane the intrinsic
    matches (i.e., 518x518 pad space here). None if behind camera."""
    R = E_c2w[:3, :3]
    t = E_c2w[:3, 3]
    P_cam = R.T @ (P - t)
    if P_cam[2] <= 1e-6:
        return None, float(P_cam[2])
    p = K @ P_cam
    return np.array([p[0] / p[2], p[1] / p[2]], dtype=np.float64), float(P_cam[2])


def _pad518_uv_to_native(p_uv_pad, native_hw, target=518, patch=14):
    """Inverse of native_to_pad518: map a pixel from the 518x518 pad
    image back to the native cv2 mask resolution."""
    Hn, Wn = native_hw
    if Wn >= Hn:
        new_w = target
        new_h = round(Hn * (target / Wn) / patch) * patch
        pad_top = (target - new_h) // 2
        u = p_uv_pad[0] * (Wn / new_w)
        v = (p_uv_pad[1] - pad_top) * (Hn / new_h)
    else:
        new_h = target
        new_w = round(Wn * (target / Hn) / patch) * patch
        pad_left = (target - new_w) // 2
        u = (p_uv_pad[0] - pad_left) * (Wn / new_w)
        v = p_uv_pad[1] * (Hn / new_h)
    return np.array([u, v], dtype=np.float64)


def _diameter_at_one_frame(
    mask: np.ndarray, K: np.ndarray, E: np.ndarray,
    native_hw,
    P_h_world: np.ndarray, axis_dir: np.ndarray,
) -> float | None:
    """Measure metric diameter at the world point P_h on a frame's
    mask. Returns None if the projected point falls outside the mask
    or the perpendicular slice has no mask pixels.

    Procedure:
      1. Project P_h to pixel space (518 pad) → uv_pad.
      2. Project a second axis point (P_h + ε·axis_dir) to get the
         axis direction in pixel space; perpendicular = its 90° rotation.
      3. Convert uv_pad → uv_native (mask resolution).
      4. At uv_native, sample the mask along the perpendicular direction;
         find the leftmost and rightmost True pixel.
      5. Convert those two native edges back to pad space, build world
         rays, and intersect each with the plane (n=axis_dir, d=axis_dir·P_h).
      6. Diameter = ||left_world - right_world||.
    """
    uv_pad, depth = _project_world_to_pixel(P_h_world, K, E)
    if uv_pad is None or depth <= 0:
        return None
    P_h2 = P_h_world + axis_dir * 0.1
    uv_pad2, _ = _project_world_to_pixel(P_h2, K, E)
    if uv_pad2 is None:
        return None
    axis_pix = uv_pad2 - uv_pad
    axis_pix_norm = np.linalg.norm(axis_pix)
    if axis_pix_norm < 1e-3:
        return None
    axis_pix = axis_pix / axis_pix_norm
    perp_pix_pad = np.array([-axis_pix[1], axis_pix[0]], dtype=np.float64)

    uv_native = _pad518_uv_to_native(uv_pad, native_hw)
    Hn, Wn = native_hw
    u0, v0 = uv_native
    if not (0 <= u0 < Wn and 0 <= v0 < Hn):
        return None

    # The native↔pad mapping might rotate/scale axes slightly; recompute
    # perpendicular in native space by mapping (uv_pad ± 1·perp) back.
    a_pad = uv_pad + perp_pix_pad
    a_native = _pad518_uv_to_native(a_pad, native_hw)
    perp_native = a_native - uv_native
    perp_norm = np.linalg.norm(perp_native)
    if perp_norm < 1e-6:
        return None
    perp_native = perp_native / perp_norm

    # Sample the perpendicular line over a wide range and find every
    # True→False / False→True transition. Then pick the connected True
    # segment closest to u0 (or containing it) and return that segment's
    # left/right offsets. This handles the case where the fitted axis
    # projects slightly outside the per-frame mask (common for thin
    # poles where the pose-only axis fit can be off by 5–20 px).
    step_px = 0.1
    max_t = max(Hn, Wn)
    n_steps = int(2 * max_t / step_px)
    ts = (np.arange(n_steps) - n_steps / 2) * step_px
    us = u0 + ts * perp_native[0]
    vs = v0 + ts * perp_native[1]
    ius = np.round(us).astype(int)
    ivs = np.round(vs).astype(int)
    in_img = (ius >= 0) & (ius < Wn) & (ivs >= 0) & (ivs < Hn)
    sampled = np.zeros_like(ts, dtype=bool)
    sampled[in_img] = mask[ivs[in_img], ius[in_img]]
    if not sampled.any():
        return None

    # Find connected True segments.
    diff = np.diff(sampled.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if sampled[0]:
        starts = np.concatenate(([0], starts))
    if sampled[-1]:
        ends = np.concatenate((ends, [len(sampled)]))

    # Only accept a segment that contains t=0 — i.e., the projected
    # axis center is itself inside the mask. This guards against a
    # bad axis fit accidentally picking up an unrelated mask segment
    # (different blob, neighbouring object, etc.) and reporting a
    # nonsense diameter.
    best = None
    for s, e in zip(starts, ends):
        if e <= s:
            continue
        if ts[s] <= 0.0 <= ts[e - 1]:
            best = (s, e)
            break
    if best is None:
        return None
    s, e = best
    left_t = float(ts[s])
    right_t = float(ts[e - 1])
    if right_t <= left_t:
        return None
    # If the segment is on one side of t=0, both bounds will have the
    # same sign — that's fine, we still want |right - left|. But to
    # match the perp-distance formula's convention, treat (left, right)
    # as the segment's two world-space edges and let the
    # _line_to_axis_dist sum cover the rest.

    left_native = uv_native + left_t * perp_native
    right_native = uv_native + right_t * perp_native
    left_pad = native_to_pad518(left_native, native_hw)
    right_pad = native_to_pad518(right_native, native_hw)

    # Diameter = perpendicular distance from the left-edge ray to the
    # pole axis + same for the right-edge ray. For a true cylinder, each
    # edge ray is tangent to the surface, so its perpendicular distance
    # to the axis line equals the radius. This formulation is robust to
    # camera-axis-parallel-to-pole-axis-plane geometry (the case the
    # ray-plane intersection blows up on).
    o_l, d_l = pixel_to_world_ray(left_pad, K, E)
    o_r, d_r = pixel_to_world_ray(right_pad, K, E)

    def _line_to_axis_dist(o, d):
        cross = np.cross(axis_dir, d)
        n = np.linalg.norm(cross)
        if n < 1e-9:
            return None
        return abs(float((o - axis_point_local) @ cross)) / n

    # Use P_h as a point on the axis (any axis point works for the
    # perpendicular-distance formula).
    axis_point_local = P_h_world
    r_l = _line_to_axis_dist(o_l, d_l)
    r_r = _line_to_axis_dist(o_r, d_r)
    if r_l is None or r_r is None:
        return None
    return float(r_l + r_r)


def measure_diameter(
    masks: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    native_hw,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    pole_bottom_xyz: np.ndarray,
    heights_m: list[float],
    scale: float,
) -> list[dict]:
    """Measure pole diameter at each requested metric height above
    `pole_bottom_xyz`, in metres. Returns a list of
    `{height_m, diameter_m, n_frames_used}` dicts (one per requested
    height). `diameter_m` is the per-frame median; None if no frame
    yielded a valid measurement.

    `scale` is metres-per-model-unit (from the GPS / fused scale
    solver). Heights are converted to model units via `1/scale` to
    locate the world point on the axis.
    """
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    S = min(masks.shape[0], extrinsics.shape[0])
    results = []
    for h_m in heights_m:
        h_units = float(h_m) / float(scale)
        P_h = pole_bottom_xyz + axis_dir * h_units
        diams = []
        for i in range(S):
            d = _diameter_at_one_frame(
                masks[i], intrinsics[i], extrinsics[i],
                native_hw, P_h, axis_dir,
            )
            if d is None:
                continue
            diams.append(d * scale)        # to metres
        if diams:
            results.append({
                "height_m": float(h_m),
                "diameter_m": float(np.median(diams)),
                "n_frames_used": int(len(diams)),
            })
        else:
            results.append({
                "height_m": float(h_m),
                "diameter_m": None,
                "n_frames_used": 0,
            })
    return results


def measure_attachments(
    attachment_masks: np.ndarray,
    attachment_names: list[str],
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    native_hw,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    pole_bottom_xyz: np.ndarray,
    scale: float,
) -> list[dict]:
    """Measure each attachment's height above pole base.

    `attachment_masks` is shape (N_attach, S, H, W) — same convention as
    multi-object SAM output indexed by object id. For each attachment
    object and each frame:
      1. Take the mask centroid (mean of True pixel coords).
      2. Convert native→pad pixel space, back-project to world ray.
      3. Find the closest point on the pole axis to that ray
         (project_ray_to_axis_t).
      4. Convert axis-t to "metres above pole base":
         height = scale * (t_attach - axis_dir · (pole_bottom - axis_point))
    Take the median across frames as the attachment's height.

    Returns one dict per attachment:
        {name, height_m, n_frames_used, object_index}.
    """
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    t_bottom = float(axis_dir @ (pole_bottom_xyz - axis_point))

    N = attachment_masks.shape[0]
    S = min(attachment_masks.shape[1], extrinsics.shape[0])
    out: list[dict] = []
    for ai in range(N):
        ts = []
        for i in range(S):
            m = attachment_masks[ai, i]
            if not m.any():
                continue
            ys, xs = np.where(m)
            cx = float(xs.mean())
            cy = float(ys.mean())
            uv_pad = native_to_pad518(np.array([cx, cy]), native_hw)
            ray_o, ray_d = pixel_to_world_ray(uv_pad, intrinsics[i],
                                              extrinsics[i])
            t = project_ray_to_axis_t(ray_o, ray_d, axis_point, axis_dir)
            ts.append(t)
        if not ts:
            out.append({
                "name": attachment_names[ai] if ai < len(attachment_names) else f"obj_{ai}",
                "object_index": int(ai),
                "height_m": None,
                "n_frames_used": 0,
            })
            continue
        t_med = float(np.median(ts))
        height_m = float(scale) * (t_med - t_bottom)
        out.append({
            "name": attachment_names[ai] if ai < len(attachment_names) else f"obj_{ai}",
            "object_index": int(ai),
            "height_m": height_m,
            "n_frames_used": int(len(ts)),
        })
    return out


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
    ap.add_argument("--diameter-at-heights", default="",
                    help="Comma-separated heights in metres above the "
                         "pole base, e.g. '1.5,3.0,5.0'. Each is "
                         "measured against the fitted axis and added "
                         "to the output JSON under 'diameters'. "
                         "Requires --gps-scale (need metric scale).")
    ap.add_argument("--attachment-objects", default="",
                    help="Comma-separated object indices in the masks "
                         "NPZ to treat as attachments (crossarm, "
                         "transformer, wire, ...). Each becomes a row "
                         "in the output JSON's 'attachments' list. "
                         "Optional 'name=index' syntax: "
                         "'crossarm=0,transformer=1'.")
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

    attachments = None
    if args.attachment_objects:
        if metric_scale is None:
            print("WARN: --attachment-objects requires --gps-scale "
                  "(need metric scale to report heights in metres).")
        else:
            full = np.load(args.masks, allow_pickle=True)["masks"]
            if full.ndim != 4:
                print("WARN: masks NPZ has no object dimension; "
                      "attachment-objects requires multi-object masks.")
            else:
                names, indices = [], []
                for tok in args.attachment_objects.split(","):
                    tok = tok.strip()
                    if not tok:
                        continue
                    if "=" in tok:
                        nm, ix = tok.split("=", 1)
                        names.append(nm.strip())
                        indices.append(int(ix))
                    else:
                        indices.append(int(tok))
                        names.append(f"obj_{tok}")
                sub = full[indices]
                print(f"\nMeasuring attachments {names} (object indices "
                      f"{indices}) ...")
                attachments = measure_attachments(
                    attachment_masks=sub,
                    attachment_names=names,
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                    native_hw=tuple(int(x) for x in native_hw),
                    axis_point=axis_point,
                    axis_dir=signed_dir,
                    pole_bottom_xyz=pole_bot,
                    scale=metric_scale,
                )
                for a in attachments:
                    if a["height_m"] is None:
                        print(f"  {a['name']}: no usable frame")
                    else:
                        print(f"  {a['name']}: {a['height_m']:.2f} m "
                              f"(n={a['n_frames_used']})")

    diameters = None
    if args.diameter_at_heights:
        if metric_scale is None:
            print("WARN: --diameter-at-heights requires --gps-scale "
                  "(need metric scale to interpret heights).")
        else:
            heights = [float(h) for h in args.diameter_at_heights.split(",")
                       if h.strip()]
            print(f"\nMeasuring diameter at heights {heights} m ...")
            diameters = measure_diameter(
                masks=masks, extrinsics=extrinsics, intrinsics=intrinsics,
                native_hw=tuple(int(x) for x in native_hw),
                axis_point=axis_point, axis_dir=signed_dir,
                pole_bottom_xyz=pole_bot,
                heights_m=heights, scale=metric_scale,
            )
            for d in diameters:
                if d["diameter_m"] is None:
                    print(f"  h={d['height_m']:.1f} m: no usable frame")
                else:
                    print(f"  h={d['height_m']:.1f} m: {d['diameter_m']:.3f} m "
                          f"(n={d['n_frames_used']})")

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
        "diameters": diameters,
        "attachments": attachments,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
