"""Synthetic geometry generator for pole-pipeline tests.

Plants a vertical pole at the world origin (going +Y in world space),
positions `n_cameras` on a horizontal ring around it, and rasterizes
the pole silhouette as a vertical rectangle in each camera's image
plane. Output schema matches what the real pipeline consumes
(`masks`, `extrinsics`, `intrinsics`, `image_hw`).

Camera convention is OpenCV: c2w extrinsic, +X right, +Y down,
+Z forward. Same as `lingbot-map`'s pose-head output, so these
fixtures plug directly into `fit_pole_axis.fit_axis_from_planes` etc.

Use:
    scene = make_pole_scene(n_cameras=8, noise_px=0.5)
    masks = scene["masks"]              # (S, H, W) bool
    extrinsics = scene["extrinsics"]    # (S, 3, 4)
    ...
"""

import numpy as np


def _look_at_c2w(eye: np.ndarray, target: np.ndarray,
                 world_up: np.ndarray) -> np.ndarray:
    """Build a c2w 3x4 extrinsic for a camera at `eye` looking at
    `target`. OpenCV convention: cam +Y points DOWN in world."""
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, world_up)
    right = right / np.linalg.norm(right)
    image_up = np.cross(right, fwd)         # "up in image" in world coords
    cam_x = right
    cam_y = -image_up                       # OpenCV +Y is image-down
    cam_z = fwd
    R_c2w = np.column_stack([cam_x, cam_y, cam_z])
    ext = np.zeros((3, 4), dtype=np.float64)
    ext[:3, :3] = R_c2w
    ext[:3, 3] = eye
    return ext


def _project(P: np.ndarray, R_c2w: np.ndarray, t_c2w: np.ndarray,
             K: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Project a world point P into pixel coordinates. Returns
    (pixel_uv, depth) or (None, depth) if behind the camera."""
    P_cam = R_c2w.T @ (P - t_c2w)
    if P_cam[2] <= 1e-6:
        return None, float(P_cam[2])
    p = K @ P_cam
    return np.array([p[0] / p[2], p[1] / p[2]], dtype=np.float64), float(P_cam[2])


def _rasterize_segment(mask: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                       thickness_px: float) -> None:
    """Rasterize a thick line segment into `mask` (in-place).
    Used to draw horizontal crossarm silhouettes at known pixel
    endpoints. Implementation is a simple oversampled-step rasterizer:
    walk along the segment in 0.5-px increments, set a small disk of
    radius thickness/2 at each step."""
    H, W = mask.shape
    seg = p2 - p1
    L = float(np.linalg.norm(seg))
    if L < 1e-3:
        return
    n_steps = max(2, int(L * 2.0))
    r = max(1, int(round(thickness_px / 2.0)))
    for s in range(n_steps + 1):
        t = s / float(n_steps)
        cu = p1[0] + t * seg[0]
        cv = p1[1] + t * seg[1]
        u_lo = int(np.floor(cu - r))
        u_hi = int(np.ceil(cu + r)) + 1
        v_lo = int(np.floor(cv - r))
        v_hi = int(np.ceil(cv + r)) + 1
        u_lo, u_hi = max(0, u_lo), min(W, u_hi)
        v_lo, v_hi = max(0, v_lo), min(H, v_hi)
        if u_hi > u_lo and v_hi > v_lo:
            mask[v_lo:v_hi, u_lo:u_hi] = True


def make_pole_scene(
    height_m: float = 10.0,
    diameter_m: float = 0.30,
    n_cameras: int = 8,
    radius_m: float = 8.0,
    noise_px: float = 0.0,
    image_size: int = 518,
    focal_px: float = 400.0,
    seed: int = 0,
    attachments: list[dict] | None = None,
) -> dict:
    """Plant a vertical pole + ring of cameras and rasterize silhouette
    masks for each camera. See module docstring for conventions.

    `noise_px` adds Gaussian jitter to the projected pole-top and
    pole-bottom pixel positions (and to the silhouette width); use 0
    for the bit-exact recovery test, larger values for robustness
    tests later in the suite.

    `attachments`, if given, is a list of dicts each describing a
    horizontal crossbar to additionally rasterize:
        {"name": str, "height_m": float (above pole base, along +Y),
         "length_m": float, "thickness_m": float (default 0.10),
         "axis": "x" | "z" (world direction the bar runs along; defaults
         to x)}
    The output dict gains:
        "attachment_masks": (N_attach, S, H, W) bool array
        "attachment_names": list[str]
        "attachment_gt_height_m": list[float]
    """
    rng = np.random.default_rng(seed)
    H = W = int(image_size)
    cx = cy = (image_size - 1) / 2.0
    K = np.array(
        [[focal_px, 0.0, cx], [0.0, focal_px, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    pole_bot = np.array([0.0, 0.0, 0.0])
    pole_top = np.array([0.0, height_m, 0.0])
    pole_mid = 0.5 * (pole_top + pole_bot)
    world_up = np.array([0.0, 1.0, 0.0])

    masks = np.zeros((n_cameras, H, W), dtype=bool)
    extrinsics = np.zeros((n_cameras, 3, 4), dtype=np.float64)
    intrinsics = np.zeros((n_cameras, 3, 3), dtype=np.float64)
    atts = list(attachments or [])
    n_att = len(atts)
    attachment_masks = np.zeros((n_att, n_cameras, H, W), dtype=bool)
    # Per-frame ground-truth pixel projections — useful for tests that
    # need a "reference tap" (two pixel coords on a known 3D segment)
    # without re-deriving the projection math.
    proj_top_uv = np.full((n_cameras, 2), np.nan, dtype=np.float64)
    proj_bot_uv = np.full((n_cameras, 2), np.nan, dtype=np.float64)

    for i in range(n_cameras):
        theta = 2.0 * np.pi * i / n_cameras
        eye = np.array([
            radius_m * np.cos(theta),
            height_m / 2.0,
            radius_m * np.sin(theta),
        ])
        ext = _look_at_c2w(eye, pole_mid, world_up)
        R_c2w = ext[:3, :3]
        t_c2w = ext[:3, 3]

        uv_top, _ = _project(pole_top, R_c2w, t_c2w, K)
        uv_bot, _ = _project(pole_bot, R_c2w, t_c2w, K)
        _, mid_depth = _project(pole_mid, R_c2w, t_c2w, K)
        if uv_top is None or uv_bot is None or mid_depth <= 0:
            # Pole behind the camera — should not happen on a ring of
            # cameras pointing at the pole, but bail out gracefully.
            continue

        if noise_px > 0:
            uv_top = uv_top + rng.normal(0.0, noise_px, size=2)
            uv_bot = uv_bot + rng.normal(0.0, noise_px, size=2)

        pole_pix_width = diameter_m * focal_px / mid_depth

        u_center = 0.5 * (uv_top[0] + uv_bot[0])
        v_lo = int(np.floor(min(uv_top[1], uv_bot[1])))
        v_hi = int(np.ceil(max(uv_top[1], uv_bot[1])))
        u_lo = int(np.floor(u_center - pole_pix_width / 2.0))
        u_hi = int(np.ceil(u_center + pole_pix_width / 2.0))

        v_lo, v_hi = max(v_lo, 0), min(v_hi, H)
        u_lo, u_hi = max(u_lo, 0), min(u_hi, W)
        if v_hi > v_lo and u_hi > u_lo:
            masks[i, v_lo:v_hi, u_lo:u_hi] = True

        extrinsics[i] = ext
        intrinsics[i] = K
        proj_top_uv[i] = uv_top
        proj_bot_uv[i] = uv_bot

        for ai, att in enumerate(atts):
            length = float(att["length_m"])
            thick_m = float(att.get("thickness_m", 0.10))
            axis = att.get("axis", "x")
            base_y = float(att["height_m"])
            if axis == "x":
                A = np.array([-length / 2.0, base_y, 0.0])
                B = np.array([+length / 2.0, base_y, 0.0])
            else:
                A = np.array([0.0, base_y, -length / 2.0])
                B = np.array([0.0, base_y, +length / 2.0])
            uvA, depthA = _project(A, R_c2w, t_c2w, K)
            uvB, depthB = _project(B, R_c2w, t_c2w, K)
            if uvA is None or uvB is None:
                continue
            mid_d = 0.5 * (depthA + depthB)
            thick_px = max(2.0, thick_m * focal_px / max(mid_d, 1e-3))
            _rasterize_segment(attachment_masks[ai, i], uvA, uvB, thick_px)

    return {
        "masks": masks,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "image_hw": (H, W),
        "gt_top_xyz": pole_top,
        "gt_bottom_xyz": pole_bot,
        "gt_axis_dir": np.array([0.0, 1.0, 0.0]),
        "gt_height_m": float(height_m),
        "gt_diameter_m": float(diameter_m),
        "gt_top_uv": proj_top_uv,        # (S, 2) — pole-top pixel per frame
        "gt_bot_uv": proj_bot_uv,        # (S, 2) — pole-bottom pixel per frame
        "attachment_masks": attachment_masks,
        "attachment_names": [a["name"] for a in atts],
        "attachment_gt_height_m": [float(a["height_m"]) for a in atts],
    }
