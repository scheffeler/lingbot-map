"""Phase 1.7: viser viewer for the image-set pole pipeline.

Loads the artifacts produced by Phase 1.4 and renders them together so a
human can sanity-check the result:

  * dense point cloud (.ply from phase0_modal_imageset)
  * per-photo camera frustums textured with the original JPEGs
  * per-object pole axis line + top/bottom spheres labelled with metric
    height (using the GPS Sim(3) scale)
  * SAM mask overlay on each frustum image (toggleable)
  * 1 m floor grid lines as a metric reference

Run:
    python scripts/visualize_pole.py \\
        --ply pole_001.ply \\
        --poses pole_001.poses.npz \\
        --masks pole_001.masks.npz \\
        --triangulation pole_001.triangulation.json \\
        --gps-scale pole_001.gps_scale.json \\
        --image-folder ./pole_001

Open http://localhost:8080 in a browser. Hit Ctrl-C to quit.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import viser
from PIL import Image


def load_ply_binary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse the binary PLY format written by `test_pole.py:write_ply_binary`.
    Returns (points (N,3) float32, colors (N,3) uint8)."""
    with path.open("rb") as f:
        header = b""
        while not header.endswith(b"end_header\n"):
            chunk = f.read(1)
            if not chunk:
                raise ValueError(f"Truncated PLY header in {path}")
            header += chunk
        n = next(
            int(line.split()[2])
            for line in header.decode().splitlines()
            if line.startswith("element vertex")
        )
        dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("r", "u1"), ("g", "u1"), ("b", "u1"),
        ])
        rows = np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype, count=n)
    pts = np.stack([rows["x"], rows["y"], rows["z"]], axis=1)
    cols = np.stack([rows["r"], rows["g"], rows["b"]], axis=1)
    return pts, cols


def fit_ground_plane(
    pts: np.ndarray, cam_centres: np.ndarray, n_iter: int = 200,
    inlier_thresh: float = 0.02,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """RANSAC plane fit through the lowest-quartile points relative to the
    camera centroid (the ground / road region). Returns (normal, offset,
    inlier_mask) where the plane is `normal . x = offset`, or None if
    the fit doesn't find a plausible plane.

    Heuristic: shoot rays from camera centroid downward (lingbot-map's
    +Y is roughly camera-down at capture height) and keep the points in
    the lower 25 % of the cloud's y-range as ground candidates.
    `inlier_thresh` is in model units; 0.02 ~ 30 cm at our test capture's
    scale (15.4 m/unit), enough to absorb grass+pebble noise."""
    if pts.shape[0] < 200:
        return None

    cam_y_med = float(np.median(cam_centres[:, 1]))
    # In lingbot-map's c2w convention, +Y is roughly down for cameras held
    # at chest height; ground points then have *larger* Y than camera y.
    # If our captures inverted this, the heuristic still picks a coherent
    # subset — RANSAC just runs on whatever subset we hand it.
    y_thresh = np.percentile(pts[:, 1], 60) if cam_y_med < pts[:, 1].mean() \
        else np.percentile(pts[:, 1], 40)
    candidates_mask = (pts[:, 1] > y_thresh) if cam_y_med < pts[:, 1].mean() \
        else (pts[:, 1] < y_thresh)
    candidates = pts[candidates_mask]
    if candidates.shape[0] < 200:
        return None

    rng = np.random.default_rng(0)
    best_inliers = None
    best_n = 0
    best_normal = None
    best_offset = 0.0
    for _ in range(n_iter):
        idx = rng.choice(candidates.shape[0], size=3, replace=False)
        p0, p1, p2 = candidates[idx]
        v1, v2 = p1 - p0, p2 - p0
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        offset = float(n @ p0)
        # Distance of every candidate to this plane.
        d = np.abs(candidates @ n - offset)
        inliers = d < inlier_thresh
        n_in = int(inliers.sum())
        if n_in > best_n:
            best_n = n_in
            best_inliers = inliers
            best_normal = n
            best_offset = offset

    if best_inliers is None or best_n < 50:
        return None

    # Refit on inliers (least-squares plane).
    inlier_pts = candidates[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    _, _, vt = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
    refined_normal = vt[2]
    refined_normal = refined_normal / np.linalg.norm(refined_normal)
    # Make the normal point "up" (away from cam centroid majority).
    cam_centroid = cam_centres.mean(axis=0)
    if (cam_centroid - centroid) @ refined_normal < 0:
        refined_normal = -refined_normal
    refined_offset = float(refined_normal @ centroid)
    return refined_normal, refined_offset, best_inliers


def alignment_R_for_normal(plane_normal: np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation matrix whose action on world points moves
    `plane_normal` to (0, 1, 0) (viser's +Y == 'up'). Constructs an
    orthonormal frame around `plane_normal`."""
    up = plane_normal / np.linalg.norm(plane_normal)
    target = np.array([0.0, 1.0, 0.0])
    v = np.cross(up, target)
    s = np.linalg.norm(v)
    c = float(up @ target)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def rotmat_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a (w, x, y, z) quaternion using
    Shepperd's method — numerically stable across all branches."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def make_thumbnail(img_path: Path, max_w: int = 512) -> np.ndarray:
    """Load and downsample a JPEG to a max width for frustum texturing.
    Returns RGB uint8 (H, W, 3)."""
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    if w > max_w:
        new_h = int(round(h * max_w / w))
        im = im.resize((max_w, new_h), Image.BILINEAR)
    return np.asarray(im)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color=(255, 60, 60),
                 alpha: float = 0.45) -> np.ndarray:
    """Blend a colored mask into an RGB thumbnail. mask is bool, may be at
    a different resolution — we resize to match `rgb`."""
    h, w = rgb.shape[:2]
    if mask.shape != (h, w):
        # Nearest-neighbor resize via PIL.
        m = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (w, h), Image.NEAREST,
        )
        mask = np.asarray(m) > 0
    out = rgb.copy().astype(np.float32)
    overlay = np.array(color, dtype=np.float32)
    a = mask.astype(np.float32) * alpha
    out = out * (1 - a[..., None]) + overlay[None, None, :] * a[..., None]
    return out.clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description="Phase 1.7 pole-result viewer")
    ap.add_argument("--ply", required=True)
    ap.add_argument("--poses", required=True)
    ap.add_argument("--masks", default=None,
                    help="multi-object masks NPZ (optional; enables mask "
                         "overlays + per-object axes)")
    ap.add_argument("--triangulation", default=None,
                    help="triangulation JSON (optional; draws pole axis)")
    ap.add_argument("--gps-scale", default=None,
                    help="gps_scale JSON (optional; converts labels to "
                         "metres)")
    ap.add_argument("--image-folder", default=None,
                    help="folder of source JPEGs (optional; textures the "
                         "camera frustums)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--point-size", type=float, default=0.005,
                    help="Point cloud point size (model units). Default "
                         "0.005 ~ 7 cm at our test capture's scale.")
    ap.add_argument("--align-ground", action="store_true",
                    help="RANSAC-fit a ground plane to the cloud and "
                         "rotate the whole scene so the plane is at "
                         "y=0 (viser +Y up). On by default in the GUI; "
                         "this flag just makes it the initial state.")
    args = ap.parse_args()

    # ---------- load ----------
    print(f"Loading PLY: {args.ply}")
    pts, cols = load_ply_binary(Path(args.ply))
    print(f"  {len(pts):,} points")

    poses = np.load(args.poses, allow_pickle=True)
    extrinsics = poses["extrinsic"]                # (S, 3, 4) c2w
    intrinsics = poses["intrinsic"]                # (S, 3, 3)
    pad_h, pad_w = (int(x) for x in poses["image_hw"])
    image_paths = [str(p) for p in poses["image_paths"]]
    S = extrinsics.shape[0]
    print(f"Loaded {S} camera poses; recon image hw={pad_h}x{pad_w}")

    masks_data = None
    n_obj = 0
    obj_ids: list[int] = []
    if args.masks and Path(args.masks).exists():
        masks_data = np.load(args.masks, allow_pickle=True)
        masks_arr = masks_data["masks"]
        if masks_arr.ndim == 4:
            n_obj = masks_arr.shape[0]
            obj_ids = [int(x) for x in masks_data["obj_ids"]] if "obj_ids" in masks_data.files else list(range(n_obj))
        else:
            n_obj = 1
            obj_ids = [0]
            masks_arr = masks_arr[None]            # → (1, S, H, W)
        native_h, native_w = (int(x) for x in masks_data["image_hw"])
        print(f"Masks: {n_obj} object(s); native hw={native_h}x{native_w}; "
              f"obj_ids={obj_ids}")
    else:
        masks_arr = None

    triang = None
    if args.triangulation and Path(args.triangulation).exists():
        triang = json.loads(Path(args.triangulation).read_text())
        print(f"Triangulation: pole_top={triang['pole_top_xyz']}, "
              f"pole_bottom={triang['pole_bottom_xyz']}, "
              f"height_model_units={triang['height_model_units']:.4f}")

    metric_scale = None
    if args.gps_scale and Path(args.gps_scale).exists():
        gs = json.loads(Path(args.gps_scale).read_text())
        metric_scale = float(gs["scale"])
        print(f"GPS scale: {metric_scale:.4f} m / model_unit  "
              f"(residual {gs.get('residual_m'):.2f} m)")

    image_folder = Path(args.image_folder) if args.image_folder else None
    thumbnails: list[np.ndarray | None] = [None] * S
    inspector_imgs: list[np.ndarray | None] = [None] * S
    if image_folder and image_folder.is_dir():
        for i, name in enumerate(image_paths):
            p = image_folder / name
            if p.exists():
                thumbnails[i] = make_thumbnail(p, max_w=512)
                inspector_imgs[i] = make_thumbnail(p, max_w=1024)
        n_loaded = sum(1 for t in thumbnails if t is not None)
        print(f"Frustum textures: {n_loaded}/{S} loaded "
              f"(plus 1024-wide inspector copies)")

    # ---------- ground-plane fit ----------
    cam_centres = extrinsics[:, :3, 3]
    plane = fit_ground_plane(pts.astype(np.float64), cam_centres.astype(np.float64))
    if plane is not None:
        plane_normal, plane_offset, _ = plane
        R_align = alignment_R_for_normal(plane_normal)
        # Translation that puts the plane at y=0 after rotation:
        # for any point p on the plane (p . n == offset), R_align @ p has
        # y-component = R_align[1] @ p = R_align[1, :] . p. We want this
        # to equal 0 globally, so add a y-translation of -plane_offset
        # (since R_align @ n = (0,1,0)).
        align_t = np.array([0.0, -plane_offset, 0.0])
        align_wxyz = rotmat_to_wxyz(R_align)
        print(f"Ground plane fit: normal={plane_normal}, "
              f"offset={plane_offset:.4f}")
    else:
        R_align = np.eye(3)
        align_t = np.zeros(3)
        align_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        print("Ground plane fit: not enough points; alignment disabled.")

    # ---------- viser scene ----------
    server = viser.ViserServer(port=args.port)
    print(f"\nViser server: http://localhost:{args.port}")

    # Parent frame — everything under /scene inherits this transform, so
    # toggling alignment is a single set on this handle.
    scene_frame = server.scene.add_frame(
        name="/scene",
        wxyz=align_wxyz if args.align_ground else (1.0, 0.0, 0.0, 0.0),
        position=tuple(align_t.tolist()) if args.align_ground else (0.0, 0.0, 0.0),
        show_axes=False,
    )

    # ---------- ground-plane visualization ----------
    # Translucent quad showing the RANSAC-fitted plane. Lives under
    # /scene/, so when alignment is on the plane snaps to horizontal at
    # y=0. If the cloud's road points don't appear flat against this
    # plane in the aligned view, the plane fit picked the wrong surface.
    plane_viz_handle = None
    if plane is not None:
        plane_normal, plane_offset, _ = plane
        cloud_centroid = pts.mean(axis=0)
        plane_centroid = cloud_centroid - plane_normal * (
            cloud_centroid @ plane_normal - plane_offset
        )
        # Build an in-plane basis (u, v) orthogonal to the normal.
        helper = np.array([0.0, 1.0, 0.0]) if abs(plane_normal[1]) < 0.9 \
            else np.array([1.0, 0.0, 0.0])
        u = np.cross(plane_normal, helper)
        u = u / np.linalg.norm(u)
        v = np.cross(plane_normal, u)
        size = 1.5                                  # ~23 m at our test scale
        verts = np.array([
            plane_centroid + size * u + size * v,
            plane_centroid - size * u + size * v,
            plane_centroid - size * u - size * v,
            plane_centroid + size * u - size * v,
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        plane_viz_handle = server.scene.add_mesh_simple(
            name="/scene/ground_plane_viz",
            vertices=verts,
            faces=faces,
            color=(255, 220, 100),
            opacity=0.25,
            side="double",
            visible=False,                          # off by default; GUI toggle
        )

    # Random-but-deterministic colors per object.
    def obj_color(k: int) -> tuple[int, int, int]:
        palette = [(220, 50, 50), (50, 200, 80), (60, 100, 230),
                   (230, 180, 50), (200, 80, 200), (60, 200, 200)]
        return palette[k % len(palette)]

    # --- Point cloud ---
    pc_handle = server.scene.add_point_cloud(
        name="/scene/point_cloud",
        points=pts,
        colors=cols,
        point_size=args.point_size,
        point_shape="circle",
    )

    # --- Camera frustums + per-frame texture (no mask overlay) ---
    raw_frustums = []
    for i in range(S):
        ext = extrinsics[i]
        K = intrinsics[i]
        R = ext[:3, :3]
        t = ext[:3, 3]
        wxyz = rotmat_to_wxyz(R)
        # FOV from the recon image dims, not the native photo; intrinsics
        # are in 518-pixel pad space.
        fy = float(K[1, 1])
        fov = 2.0 * math.atan2(pad_h * 0.5, fy)
        aspect = pad_w / pad_h
        # Color frustum by frame index along a viridis-like ramp.
        u = i / max(S - 1, 1)
        c = (int(255 * (0.2 + 0.6 * u)), int(255 * (0.7 - 0.5 * u)),
             int(255 * (0.9 - 0.4 * u)))
        raw_frustums.append(server.scene.add_camera_frustum(
            name=f"/scene/frustums_raw/cam{i:02d}",
            fov=fov, aspect=aspect,
            scale=0.25,
            line_width=2.0,
            color=c,
            wxyz=wxyz,
            position=t,
            image=thumbnails[i],
        ))

    # --- Per-object: separate frustum stack with mask overlays + axis line + endpoints. ---
    obj_groups: dict[int, dict] = {}
    if masks_arr is not None:
        for k in range(n_obj):
            color = obj_color(k)
            # Per-object frustum stack (same pose, but textured with the
            # object's mask overlaid on the thumbnail).
            f_handles = []
            for i in range(S):
                ext = extrinsics[i]
                R = ext[:3, :3]; t = ext[:3, 3]
                wxyz = rotmat_to_wxyz(R)
                fy = float(intrinsics[i, 1, 1])
                fov = 2.0 * math.atan2(pad_h * 0.5, fy)
                aspect = pad_w / pad_h
                tex = thumbnails[i]
                if tex is not None:
                    tex = overlay_mask(tex, masks_arr[k, i], color=color)
                f_handles.append(server.scene.add_camera_frustum(
                    name=f"/scene/obj{k}/frustums/cam{i:02d}",
                    fov=fov, aspect=aspect,
                    scale=0.25,
                    line_width=2.0,
                    color=color,
                    wxyz=wxyz,
                    position=t,
                    image=tex,
                    visible=False,                  # toggled by GUI below
                ))
            obj_groups[k] = {"frustums": f_handles, "color": color}

    # --- Triangulation overlay (active object only) ---
    pole_axis_handle = None
    pole_top_handle = None
    pole_bot_handle = None
    pole_top_label = None
    pole_bot_label = None
    if triang is not None:
        top = np.array(triang["pole_top_xyz"], dtype=np.float64)
        bot = np.array(triang["pole_bottom_xyz"], dtype=np.float64)
        height_units = float(triang["height_model_units"])
        height_m = height_units * metric_scale if metric_scale else None

        seg = np.array([[bot, top]], dtype=np.float32)
        pole_axis_handle = server.scene.add_line_segments(
            name="/scene/pole/axis",
            points=seg,
            colors=(255, 220, 50),
            line_width=8,
        )
        sphere_r = 0.02
        pole_top_handle = server.scene.add_icosphere(
            name="/scene/pole/top",
            radius=sphere_r, color=(255, 80, 80),
            position=tuple(top.tolist()),
        )
        pole_bot_handle = server.scene.add_icosphere(
            name="/scene/pole/bottom",
            radius=sphere_r, color=(80, 200, 255),
            position=tuple(bot.tolist()),
        )
        # Floating labels at the endpoints.
        top_text = (f"top  ({height_m:.2f} m)"
                    if height_m is not None
                    else f"top  ({height_units:.3f} units)")
        bot_text = "bottom (0 m)"
        pole_top_label = server.scene.add_label(
            name="/scene/pole/top_label",
            text=top_text,
            position=tuple((top + np.array([0, 0, 0.05])).tolist()),
        )
        pole_bot_label = server.scene.add_label(
            name="/scene/pole/bottom_label",
            text=bot_text,
            position=tuple((bot + np.array([0, 0, 0.05])).tolist()),
        )

    # ---------- GUI ----------
    gui = server.gui
    gui.add_markdown("**Phase 1.7 viewer** — pole image-set result")

    if metric_scale:
        gui.add_markdown(
            f"GPS scale: **{metric_scale:.3f} m / model_unit**  \n"
            f"Cameras: {S}  •  Points: {len(pts):,}"
        )
    if triang is not None and triang.get("height_m") is not None:
        lean_lines = []
        if triang.get("axis_lean_deg") is not None:
            lean_lines.append(
                f"Lean vs ground-plane normal: **{triang['axis_lean_deg']:.1f}°**"
            )
        if triang.get("axis_lean_cam_up_deg") is not None:
            lean_lines.append(
                f"Lean vs mean camera-up: **{triang['axis_lean_cam_up_deg']:.1f}°**"
            )
        gui.add_markdown(
            f"Pole height: **{triang['height_m']:.2f} m** "
            f"({triang['height_model_units']:.4f} model units)  \n"
            + "  \n".join(lean_lines)
        )

    tabs = gui.add_tab_group()
    scene_tab = tabs.add_tab("Scene")
    masks_tab = tabs.add_tab("Masks")

    with scene_tab:
        align_toggle = gui.add_checkbox(
            "Align to ground plane",
            initial_value=args.align_ground,
            disabled=(plane is None),
        )

        @align_toggle.on_update
        def _(_):
            if align_toggle.value:
                scene_frame.wxyz = align_wxyz
                scene_frame.position = tuple(align_t.tolist())
            else:
                scene_frame.wxyz = (1.0, 0.0, 0.0, 0.0)
                scene_frame.position = (0.0, 0.0, 0.0)

        # Metric ground grid: 5 m x 5 m, 1 m spacing. Inherits the
        # alignment toggle because it lives under /scene/.
        if metric_scale and plane is not None:
            unit_per_m = 1.0 / metric_scale
            grid_n = 5
            lines = []
            for k in range(-grid_n, grid_n + 1):
                lines.append([
                    (-grid_n * unit_per_m, 0.0, k * unit_per_m),
                    (+grid_n * unit_per_m, 0.0, k * unit_per_m),
                ])
                lines.append([
                    (k * unit_per_m, 0.0, -grid_n * unit_per_m),
                    (k * unit_per_m, 0.0, +grid_n * unit_per_m),
                ])
            grid_pts = np.array(lines, dtype=np.float32)
            grid_handle = server.scene.add_line_segments(
                name="/scene/ground_grid",
                points=grid_pts,
                colors=(80, 120, 80),
                line_width=1.5,
            )
            show_grid = gui.add_checkbox("Ground grid (1 m)", initial_value=True)

            @show_grid.on_update
            def _(_):
                grid_handle.visible = show_grid.value

        if plane_viz_handle is not None:
            show_plane = gui.add_checkbox(
                "Show fitted ground plane (translucent quad)",
                initial_value=False,
            )

            @show_plane.on_update
            def _(_):
                plane_viz_handle.visible = show_plane.value

        show_pc = gui.add_checkbox("Point cloud", initial_value=True)

        @show_pc.on_update
        def _(_):
            pc_handle.visible = show_pc.value

        show_raw = gui.add_checkbox("Raw camera frustums", initial_value=True)

        @show_raw.on_update
        def _(_):
            for h in raw_frustums:
                h.visible = show_raw.value

        point_size_slider = gui.add_slider(
            "Point size", min=0.001, max=0.05,
            step=0.001, initial_value=args.point_size,
        )

        @point_size_slider.on_update
        def _(_):
            pc_handle.point_size = point_size_slider.value

        # Per-object axis + frustum-mask controls.
        if obj_groups:
            gui.add_markdown("---")
            gui.add_markdown("**Tracked objects**")
            obj_options = [f"obj {k} (id={obj_ids[k]})" for k in range(n_obj)]
            active_dropdown = gui.add_dropdown(
                "Active object", options=obj_options,
                initial_value=obj_options[min(2, n_obj - 1)],
            )
            show_obj_frustums = gui.add_checkbox(
                "Mask overlay on frustums", initial_value=True,
            )
            show_axis = gui.add_checkbox(
                "Pole axis + endpoints", initial_value=True,
            )

            def _refresh_active():
                active_k = int(active_dropdown.value.split()[1])
                for k, group in obj_groups.items():
                    visible = (k == active_k) and show_obj_frustums.value
                    for h in group["frustums"]:
                        h.visible = visible
                if pole_axis_handle is not None:
                    pole_axis_handle.visible = show_axis.value
                    pole_top_handle.visible = show_axis.value
                    pole_bot_handle.visible = show_axis.value
                    pole_top_label.visible = show_axis.value
                    pole_bot_label.visible = show_axis.value

            active_dropdown.on_update(lambda _: _refresh_active())
            show_obj_frustums.on_update(lambda _: _refresh_active())
            show_axis.on_update(lambda _: _refresh_active())
            _refresh_active()

    # ---------- Masks tab: 2D image inspector embedded in the GUI ----------
    if masks_arr is not None and any(t is not None for t in inspector_imgs):
        with masks_tab:
            gui.add_markdown(
                f"Source photos with SAM 3.1 mask overlays. "
                f"Frame slider, per-object toggles. {S} frames • "
                f"{n_obj} objects."
            )
            inspector_frame = gui.add_slider(
                "Frame", min=0, max=S - 1, step=1, initial_value=0,
            )
            opacity_slider = gui.add_slider(
                "Mask opacity (%)", min=0, max=100, step=5, initial_value=55,
            )
            obj_cbs = []
            for k in range(n_obj):
                color = obj_color(k)
                cb = gui.add_checkbox(
                    f"obj {k} (id={obj_ids[k]})  [{color[0]},{color[1]},{color[2]}]",
                    initial_value=True,
                )
                obj_cbs.append(cb)

            def _render_inspector():
                f_idx = int(inspector_frame.value)
                base = inspector_imgs[f_idx]
                if base is None:
                    return np.zeros((100, 100, 3), dtype=np.uint8)
                composite = base.copy().astype(np.float32)
                a = opacity_slider.value / 100.0
                hh, ww = composite.shape[:2]
                for k, cb in enumerate(obj_cbs):
                    if not cb.value:
                        continue
                    m = masks_arr[k, f_idx]
                    if m.shape != (hh, ww):
                        m = np.asarray(
                            Image.fromarray(m.astype(np.uint8) * 255).resize(
                                (ww, hh), Image.NEAREST,
                            )
                        ) > 0
                    if not m.any():
                        continue
                    color = np.array(obj_color(k), dtype=np.float32)
                    alpha = m.astype(np.float32) * a
                    composite = (
                        composite * (1 - alpha[..., None])
                        + color[None, None, :] * alpha[..., None]
                    )
                return composite.clip(0, 255).astype(np.uint8)

            img_handle = gui.add_image(
                _render_inspector(), label=None,
                format="jpeg", jpeg_quality=85,
            )

            def _refresh_inspector():
                img_handle.image = _render_inspector()

            inspector_frame.on_update(lambda _: _refresh_inspector())
            opacity_slider.on_update(lambda _: _refresh_inspector())
            for cb in obj_cbs:
                cb.on_update(lambda _: _refresh_inspector())

    print("Ready. Ctrl-C to quit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
