"""Phase 1.3: triangulate a 3D pole axis from SAM masks + lingbot-map poses.

Inputs:
  --masks   path to pole_masks.npz   (from scripts/phase1_sam.py)
  --poses   path to pole_walkaround.poses.npz   (from scripts/phase0_modal.py)
  --output  path to write triangulation result (.json)

For each frame:
  1. Rescale the SAM mask from native (cv2-extracted) resolution to the
     518x518 pad-transformed frame the lingbot-map intrinsics correspond to.
  2. Take the largest connected component (drops second-pole and wire
     fragments that aren't connected to the dominant pole region).
  3. Run PCA on the kept mask pixels. The major axis is the pole's
     image-space direction; project mask pixels onto it and take the
     extreme +/- points as 'top' and 'bottom' image points for that frame.

Then triangulate:
  - For every frame, build two 3D rays from the camera centre through the
    top image point and the bottom image point.
  - Solve a least-squares closest-point-to-all-rays problem for each
    endpoint independently. Result: a 3D pole top and a 3D pole bottom.
  - Pole height = distance between them in lingbot-map's metric-ish
    world units.

Notes / known limits:
  - 'top' vs 'bottom' is determined by which direction along the major axis
    is closer to the camera's camera-up vector (after projection). In
    practice for a vertical pole captured at near-eye level it sorts itself
    out; the script also flips at the end if pole_top is below pole_bottom
    in the y-axis (relative to the world frame's mean camera-up).
  - lingbot-map's world is up-to-rough-scale, so 'pole height' here is in
    model units. We'll calibrate with the EXIF GNSS Sim(3) step in
    Phase 1.4 to convert to metres.
"""

import argparse
import json
import os
import sys

import numpy as np


def fit_largest_component(mask: np.ndarray):
    """Return the (M, 2) [x, y] coords of the largest 4-connected component
    of a boolean mask, or None if the mask is empty."""
    import cv2  # local import — only triangulation script needs cv2

    if not mask.any():
        return None
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8,
    )
    if num <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = 1 + int(np.argmax(areas))
    ys, xs = np.where(labels == biggest)
    if xs.size == 0:
        return None
    return np.stack([xs, ys], axis=1).astype(np.float32)


def axis_endpoints(points_xy: np.ndarray):
    """Run PCA on (M, 2) image points; return (top_xy, bottom_xy) along
    the major axis. 'Top' = the end with smaller y (image y points down,
    so smaller y is upper in the image)."""
    centroid = points_xy.mean(axis=0)
    centered = points_xy - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    major = vt[0]
    proj = centered @ major
    p_min = points_xy[int(np.argmin(proj))]
    p_max = points_xy[int(np.argmax(proj))]
    if p_min[1] > p_max[1]:
        p_min, p_max = p_max, p_min
    return p_min, p_max


def native_to_pad518(p_xy_native, native_hw, target=518, patch=14, pad_mode=True):
    """Map a point in the native cv2 frame (W=native_hw[1], H=native_hw[0])
    into the 518x518 pad-mode image lingbot-map's intrinsics expect.

    Pad mode (when width >= height):
      new_w = 518
      new_h = round(H * 518 / W / patch) * patch
      pad_top = (518 - new_h) // 2,  pad_bottom = 518 - new_h - pad_top
    """
    Hn, Wn = native_hw
    if Wn >= Hn:
        new_w = target
        new_h = round(Hn * (target / Wn) / patch) * patch
        pad_top = (target - new_h) // 2
        u = p_xy_native[0] * (new_w / Wn)
        v = p_xy_native[1] * (new_h / Hn) + pad_top
    else:
        new_h = target
        new_w = round(Wn * (target / Hn) / patch) * patch
        pad_left = (target - new_w) // 2
        u = p_xy_native[0] * (new_w / Wn) + pad_left
        v = p_xy_native[1] * (new_h / Hn)
    return np.array([u, v], dtype=np.float32)


def pixel_to_world_ray(pixel_xy, intrinsic, extrinsic_c2w):
    """Return (origin, direction) of the world-space ray through a pixel.
    intrinsic: (3, 3); extrinsic_c2w: (3, 4) camera-to-world."""
    K_inv = np.linalg.inv(intrinsic)
    pixel_h = np.array([pixel_xy[0], pixel_xy[1], 1.0], dtype=np.float64)
    cam_dir = K_inv @ pixel_h
    R = extrinsic_c2w[:3, :3]
    t = extrinsic_c2w[:3, 3]
    world_dir = R @ cam_dir
    world_dir = world_dir / np.linalg.norm(world_dir)
    return t.astype(np.float64), world_dir


def closest_point_to_rays(origins, dirs):
    """Least-squares closest 3D point to a bundle of rays.
    For each ray r_i = o_i + t * d_i (||d_i||=1), the perpendicular projection
    matrix is P_i = I - d_i d_i^T. Solve sum_i P_i p = sum_i P_i o_i ."""
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    for o, d in zip(origins, dirs):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    return p


def main():
    ap = argparse.ArgumentParser(description="Triangulate pole axis from masks + poses")
    ap.add_argument("--masks", default="pole_masks.npz")
    ap.add_argument("--poses", default="pole_walkaround.poses.npz")
    ap.add_argument("--output", default="pole_triangulation.json")
    args = ap.parse_args()

    if not os.path.exists(args.masks):
        sys.exit(f"masks file not found: {args.masks}")
    if not os.path.exists(args.poses):
        sys.exit(f"poses file not found: {args.poses}")

    masks_data = np.load(args.masks, allow_pickle=True)
    poses_data = np.load(args.poses, allow_pickle=True)

    masks = masks_data["masks"]
    native_hw = masks_data["image_hw"]
    extrinsics = poses_data["extrinsic"]
    intrinsics = poses_data["intrinsic"]
    pad_hw = poses_data["image_hw"]

    S_m = masks.shape[0]
    S_p = extrinsics.shape[0]
    S = min(S_m, S_p)
    if S_m != S_p:
        print(f"WARNING: mask frame count {S_m} != pose frame count {S_p}. "
              f"Using first {S} frames of each.")

    print(f"Triangulating from {S} frame(s)")
    print(f"  Native mask hw:  {tuple(native_hw)}")
    print(f"  Recon image hw:  {tuple(pad_hw)}")

    top_origins, top_dirs = [], []
    bot_origins, bot_dirs = [], []
    used = 0
    for i in range(S):
        pts = fit_largest_component(masks[i])
        if pts is None or pts.shape[0] < 50:
            continue
        top_xy_native, bot_xy_native = axis_endpoints(pts)
        top_uv = native_to_pad518(top_xy_native, native_hw)
        bot_uv = native_to_pad518(bot_xy_native, native_hw)

        K = intrinsics[i]
        E = extrinsics[i]
        o_t, d_t = pixel_to_world_ray(top_uv, K, E)
        o_b, d_b = pixel_to_world_ray(bot_uv, K, E)
        top_origins.append(o_t); top_dirs.append(d_t)
        bot_origins.append(o_b); bot_dirs.append(d_b)
        used += 1

    if used < 5:
        sys.exit(f"Only {used} usable frame(s) — aborting.")

    pole_top = closest_point_to_rays(np.array(top_origins), np.array(top_dirs))
    pole_bot = closest_point_to_rays(np.array(bot_origins), np.array(bot_dirs))
    height = float(np.linalg.norm(pole_top - pole_bot))
    axis = (pole_top - pole_bot) / max(height, 1e-9)

    cam_centers = extrinsics[:S, :3, 3]
    world_up_proxy = -axis  # ensure pole 'top' is in same general direction as scene's vertical
    # if the inferred axis points down relative to the camera-y mean, flip
    cam_up_dir = -extrinsics[:S, :3, 1].mean(axis=0)
    cam_up_dir = cam_up_dir / max(np.linalg.norm(cam_up_dir), 1e-9)
    if np.dot(axis, cam_up_dir) < 0:
        pole_top, pole_bot = pole_bot, pole_top
        axis = -axis

    pole_center = 0.5 * (pole_top + pole_bot)
    cam_centroid = cam_centers.mean(axis=0)
    pole_to_cam = float(np.linalg.norm(pole_center - cam_centroid))

    print(f"\nUsed {used}/{S} frames")
    print(f"Pole top:    {pole_top}")
    print(f"Pole bottom: {pole_bot}")
    print(f"Axis dir:    {axis}")
    print(f"Pole height: {height:.4f}  (model units; ~{height * 11:.2f} m if "
          f"1 unit ~= 11 m from earlier scale check)")
    print(f"Pole-to-camera centroid distance: {pole_to_cam:.4f}")

    out = {
        "pole_top_xyz": pole_top.tolist(),
        "pole_bottom_xyz": pole_bot.tolist(),
        "axis_direction": axis.tolist(),
        "height_model_units": height,
        "frames_used": used,
        "frames_total": int(S),
        "native_hw": [int(x) for x in native_hw],
        "pad_hw": [int(x) for x in pad_hw],
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
