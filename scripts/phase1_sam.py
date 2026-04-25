"""Phase 1.2: SAM 2 pole segmentation, video-propagated from a single click.

Extracts frames at the same fps as the Phase 0 reconstruction, seeds a
single positive click at the center of a reference frame (middle of the
clip), propagates the mask forward and backward, and writes masks.npz
plus a 3x3 preview grid.

Run:
    modal run scripts/phase1_sam.py::main \\
        --video "C:/Users/jdsch/Downloads/IMG_3545.MP4" \\
        --fps 10 \\
        --output pole_masks.npz

Notes on SAM version: this uses SAM 2.1 via facebookresearch/sam2, the
latest release we can rely on at the time of writing. When SAM 3.x is
generally available we should swap the checkpoint + config (API is
likely similar).
"""

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App("polevision-phase1-sam")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.10",
    )
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.9.1",
        "torchvision==0.24.1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "numpy",
        "opencv-python-headless",
        "Pillow",
        "tqdm",
        "huggingface_hub",
        "hydra-core",
        "iopath",
    )
    .run_commands(
        "pip install --no-build-isolation "
        "git+https://github.com/facebookresearch/sam2.git@main"
    )
)

sam_vol = modal.Volume.from_name("sam2-ckpt", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/ckpt": sam_vol},
    timeout=60 * 30,
)
def segment(
    video_bytes: bytes,
    video_name: str,
    fps: int,
) -> dict:
    import io
    import os
    import sys
    from contextlib import redirect_stdout, redirect_stderr

    import cv2
    import numpy as np
    import torch

    log_buf = io.StringIO()

    def log(msg):
        print(msg, flush=True)
        log_buf.write(msg + "\n")

    ckpt_name = "sam2.1_hiera_large.pt"
    ckpt_path = f"/ckpt/{ckpt_name}"
    if not os.path.exists(ckpt_path):
        log(f"Fetching {ckpt_name} (first run only)...")
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            repo_id="facebook/sam2.1-hiera-large",
            filename=ckpt_name,
            local_dir="/ckpt",
        )
        sam_vol.commit()
    log(f"Using SAM 2.1 checkpoint at {ckpt_path}")

    video_path = f"/tmp/{video_name}"
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    frames_dir = "/tmp/frames"
    os.makedirs(frames_dir, exist_ok=True)
    for name in os.listdir(frames_dir):
        os.remove(os.path.join(frames_dir, name))

    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = max(1, round(src_fps / fps))
    idx, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            cv2.imwrite(f"{frames_dir}/{saved:06d}.jpg", frame)
            saved += 1
        idx += 1
    cap.release()

    first = cv2.imread(f"{frames_dir}/000000.jpg")
    H, W = first.shape[:2]
    ref_idx = saved // 2
    click_xy = np.array([W / 2.0, H / 2.0], dtype=np.float32)
    log(f"Extracted {saved} frames at {W}x{H}, ref frame {ref_idx}, "
        f"center click at ({click_xy[0]:.0f}, {click_xy[1]:.0f})")

    from sam2.build_sam import build_sam2_video_predictor
    cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = build_sam2_video_predictor(cfg, ckpt_path, device="cuda")

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=frames_dir)
        predictor.reset_state(state)
        _, _, _ = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=ref_idx,
            obj_id=1,
            points=click_xy[None, :],
            labels=np.array([1], dtype=np.int32),
        )

        masks = np.zeros((saved, H, W), dtype=bool)
        log("Propagating forward...")
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            m = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            masks[frame_idx] = m
        log("Propagating backward...")
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state, reverse=True):
            m = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            masks[frame_idx] = m

    coverage = masks.any(axis=(1, 2))
    pct_hit = 100.0 * coverage.sum() / saved
    mean_area = float(masks.sum(axis=(1, 2)).mean())
    log(f"Mask coverage: {coverage.sum()}/{saved} frames ({pct_hit:.1f}%)")
    log(f"Mean mask area: {mean_area:.0f} px  ({100.0 * mean_area / (H * W):.2f}% of frame)")

    masks_bytes_io = io.BytesIO()
    np.savez_compressed(
        masks_bytes_io,
        masks=masks,
        ref_frame=np.int32(ref_idx),
        click_xy=click_xy,
        image_hw=np.array([H, W], dtype=np.int32),
    )

    preview_idxs = np.linspace(0, saved - 1, 9).astype(int)
    tiles = []
    for i in preview_idxs:
        img = cv2.imread(f"{frames_dir}/{i:06d}.jpg")
        m = masks[i]
        overlay = img.copy()
        red = np.zeros_like(overlay)
        red[..., 2] = 255
        alpha = (m.astype(np.float32) * 0.5)[..., None]
        overlay = (overlay * (1 - alpha) + red * alpha).astype(np.uint8)
        if i == ref_idx:
            cv2.circle(overlay, tuple(click_xy.astype(int)), 10, (0, 255, 255), 2)
        cv2.putText(overlay, f"f{i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        tile = cv2.resize(overlay, (W // 3, H // 3))
        tiles.append(tile)
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, 9, 3)]
    grid = np.vstack(rows)
    ok, preview_bytes = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    preview = preview_bytes.tobytes() if ok else b""

    return {
        "masks": masks_bytes_io.getvalue(),
        "preview": preview,
        "logs": log_buf.getvalue(),
        "returncode": 0,
    }


@app.local_entrypoint()
def main(
    video: str,
    output: str = "pole_masks.npz",
    fps: int = 10,
    preview: str = "pole_masks_preview.jpg",
):
    video_path = Path(video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video}")
    size_mb = video_path.stat().st_size / 1e6
    print(f"Uploading {video_path.name} ({size_mb:.1f} MB) to Modal...")

    result = segment.remote(
        video_bytes=video_path.read_bytes(),
        video_name=video_path.name,
        fps=fps,
    )

    print("\n=== Remote logs ===")
    print(result["logs"])
    print("===================\n")

    if result["returncode"] != 0:
        raise SystemExit(f"Remote run failed (exit {result['returncode']})")

    Path(output).write_bytes(result["masks"])
    print(f"Wrote {output} ({len(result['masks']) / 1e6:.2f} MB)")

    if result.get("preview"):
        Path(preview).write_bytes(result["preview"])
        print(f"Wrote {preview} ({len(result['preview']) / 1e3:.1f} KB)")
