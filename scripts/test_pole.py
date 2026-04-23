"""Phase 0 smoke test: reconstruct a utility pole from a video clip.

Runs the lingbot-map streaming pipeline on a walk-around video, confidence-
filters the dense point cloud, writes a binary PLY, and prints sanity stats
to let us decide go/no-go on Phase 1.

Example:
    python scripts/test_pole.py \\
        --model_path ./checkpoints/lingbot-map.pt \\
        --video_path "C:/Users/jdsch/Downloads/IMG_0039.MOV" \\
        --output pole_cloud.ply
"""

import argparse
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

# Allow running as `python scripts/test_pole.py` from repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from demo import load_model, postprocess, prepare_for_visualization
from lingbot_map.utils.load_fn import load_and_preprocess_images
import cv2
import glob as _glob
from tqdm.auto import tqdm


def load_images_pole(
    video_path=None, image_folder=None, fps=10, crop_mode="pad",
    image_size=518, patch_size=14,
):
    """Load frames from video or image folder, preprocess with crop/pad mode.

    Same frame-extraction logic as demo.load_images, but lets us pass
    mode="pad" for portrait captures so the full pole stays in frame.
    """
    if (video_path is None) == (image_folder is None):
        raise ValueError("Provide exactly one of video_path or image_folder")

    if video_path is not None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(os.path.dirname(video_path), f"{video_name}_frames")
        os.makedirs(out_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, round(src_fps / fps))
        idx, saved = 0, []
        pbar = tqdm(total=total_frames, desc="Extracting frames", unit="frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                path = os.path.join(out_dir, f"{len(saved):06d}.jpg")
                cv2.imwrite(path, frame)
                saved.append(path)
            idx += 1
            pbar.update(1)
        pbar.close()
        cap.release()
        print(f"Extracted {len(saved)} frames from video (interval={interval}, mode={crop_mode})")
    else:
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
        saved = sorted(
            p for p in _glob.glob(os.path.join(image_folder, "*"))
            if p.lower().endswith(exts)
        )
        if not saved:
            raise ValueError(f"No images found in {image_folder}")
        print(f"Found {len(saved)} images in {image_folder} (mode={crop_mode})")

    images = load_and_preprocess_images(
        saved, mode=crop_mode, image_size=image_size, patch_size=patch_size,
    )
    h, w = images.shape[-2:]
    print(f"Preprocessed to {w}x{h} using {crop_mode} mode")
    return images, saved


def _build_model_args(cli):
    class _A:
        pass
    a = _A()
    a.mode = cli.mode
    a.image_size = 518
    a.patch_size = 14
    a.enable_3d_rope = True
    a.max_frame_num = 1024
    a.num_scale_frames = 8
    a.kv_cache_sliding_window = 64
    a.camera_num_iterations = 4
    a.use_sdpa = True  # PyTorch SDPA; avoids FlashInfer install
    a.model_path = cli.model_path
    return a


def write_ply_binary(path, vertices, colors):
    """Write a binary little-endian PLY (XYZ + RGB). No trimesh needed."""
    n = len(vertices)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    rows = np.empty(n, dtype=dtype)
    rows["x"] = vertices[:, 0]
    rows["y"] = vertices[:, 1]
    rows["z"] = vertices[:, 2]
    rows["r"] = colors[:, 0]
    rows["g"] = colors[:, 1]
    rows["b"] = colors[:, 2]

    with open(path, "wb") as f:
        f.write(header)
        f.write(rows.tobytes())


def main():
    p = argparse.ArgumentParser(description="Phase 0 pole reconstruction smoke test")
    p.add_argument("--model_path", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video_path", default=None)
    src.add_argument("--image_folder", default=None,
                     help="Folder of image files (alternative to --video_path)")
    p.add_argument("--output", default="pole_cloud.ply")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--conf_percentile", type=float, default=50.0,
                   help="Drop points below this percentile of world_points_conf")
    p.add_argument("--mode", choices=["streaming", "windowed"], default="streaming")
    p.add_argument("--offload_to_cpu", action="store_true",
                   help="Offload per-frame predictions to CPU (use on <=8 GB GPUs)")
    p.add_argument("--first_k", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--window_size", type=int, default=64)
    p.add_argument("--overlap_size", type=int, default=16)
    p.add_argument("--downsample", type=int, default=1,
                   help="Keep every Nth point after confidence filtering (1 = keep all)")
    p.add_argument("--crop_mode", choices=["crop", "pad"], default="pad",
                   help="pad: preserve full aspect ratio (default; best for portrait pole captures). "
                        "crop: match demo.py behavior; may chop off pole top on portrait video.")
    p.add_argument("--mask_sky", action="store_true",
                   help="Apply sky segmentation to zero out sky-region confidence before filtering.")
    p.add_argument("--skyseg_model_path", default="skyseg.onnx",
                   help="Path to the sky segmentation ONNX model (auto-downloaded if missing).")
    cli = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    t0 = time.time()
    images, paths = load_images_pole(
        video_path=cli.video_path,
        image_folder=cli.image_folder,
        fps=cli.fps,
        crop_mode=cli.crop_mode,
    )
    print(f"Loaded {len(paths)} frames in {time.time()-t0:.1f}s")

    model_args = _build_model_args(cli)
    model = load_model(model_args, device)

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        model.aggregator = model.aggregator.to(dtype=dtype)

    images_dev = images.to(device)
    num_frames = images_dev.shape[0]
    print(f"Input: {num_frames} frames, shape {tuple(images_dev.shape)}, mode={cli.mode}")

    output_device = torch.device("cpu") if cli.offload_to_cpu else None

    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        if cli.mode == "streaming":
            keyframe_interval = 1 if num_frames <= 320 else (num_frames + 319) // 320
            predictions = model.inference_streaming(
                images_dev,
                num_scale_frames=8,
                keyframe_interval=keyframe_interval,
                output_device=output_device,
            )
        else:
            predictions = model.inference_windowed(
                images_dev,
                window_size=cli.window_size,
                overlap_size=cli.overlap_size,
                num_scale_frames=8,
                output_device=output_device,
            )
    infer_s = time.time() - t0
    print(f"Inference: {infer_s:.1f}s ({num_frames/infer_s:.1f} FPS)")

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"GPU peak: {peak:.2f} GB")

    # Prefer CPU copy if we offloaded during inference
    if cli.offload_to_cpu:
        del images_dev
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        images_for_post = predictions["images"]
    else:
        images_for_post = images_dev

    predictions, images_cpu = postprocess(predictions, images_for_post)
    vis = prepare_for_visualization(predictions, images_cpu)

    world_points = vis["world_points"]           # (S, H, W, 3)
    conf = vis["world_points_conf"]              # (S, H, W)
    imgs = vis["images"]                         # (S, 3, H, W) or (S, H, W, 3)

    verts = world_points.reshape(-1, 3).astype(np.float32)
    if imgs.ndim == 4 and imgs.shape[1] == 3:
        colors_rgb = np.transpose(imgs, (0, 2, 3, 1))
    else:
        colors_rgb = imgs
    colors = (colors_rgb.reshape(-1, 3) * 255).clip(0, 255).astype(np.uint8)

    if cli.mask_sky:
        from lingbot_map.vis.sky_segmentation import apply_sky_segmentation
        print(f"Applying sky segmentation with model at {cli.skyseg_model_path}...")
        conf = apply_sky_segmentation(
            conf,
            image_paths=paths,
            images=imgs,
            skyseg_model_path=cli.skyseg_model_path,
        )

    conf_flat = conf.reshape(-1)
    thresh = np.percentile(conf_flat, cli.conf_percentile) if cli.conf_percentile > 0 else 0.0
    keep = (conf_flat >= thresh) & (conf_flat > 1e-5)
    verts_k = verts[keep]
    colors_k = colors[keep]

    if cli.downsample > 1:
        rng = np.random.default_rng(0)
        keep_idx = rng.choice(verts_k.shape[0], size=verts_k.shape[0] // cli.downsample, replace=False)
        verts_k = verts_k[keep_idx]
        colors_k = colors_k[keep_idx]
        print(f"Downsampled to {verts_k.shape[0]:,} points (every {cli.downsample}th)")

    total = verts.shape[0]
    kept = verts_k.shape[0]
    print(f"Points: kept {kept:,} / {total:,} "
          f"({100.0 * kept / max(total, 1):.1f}%) at p{cli.conf_percentile:g}, "
          f"conf_threshold={thresh:.4f}")

    if kept == 0:
        print("ERROR: no points passed confidence filter.")
        sys.exit(2)

    mins = verts_k.min(axis=0)
    maxs = verts_k.max(axis=0)
    extent = maxs - mins
    depths = np.linalg.norm(verts_k, axis=1)
    print(f"Scene AABB min:    {mins}")
    print(f"Scene AABB max:    {maxs}")
    print(f"Scene extent XYZ:  {extent}  (arbitrary scale units)")
    print(f"Depth  median:     {np.median(depths):.3f}")
    print(f"Depth  p5 / p95:   {np.percentile(depths, 5):.3f} / {np.percentile(depths, 95):.3f}")

    write_ply_binary(cli.output, verts_k, colors_k)
    size_mb = os.path.getsize(cli.output) / 1e6
    print(f"Wrote {cli.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
