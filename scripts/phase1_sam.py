"""Phase 1.2: SAM 3.1 pole segmentation, text-prompted video propagation.

Uses Meta's SAM 3.1 (released 2026-03-27) via `build_sam3_multiplex_video_predictor`.
Prompts with a single text string (default: "utility pole") rather than
a point click — the concept-level prompt is more robust for thin
upright objects against sky than a center-of-frame click, and needs
no frame-specific UX.

Requires a Modal Secret named "huggingface" containing HF_TOKEN so
the container can download the gated facebook/sam3.1 checkpoint:

    modal secret create huggingface HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

Run:
    modal run scripts/phase1_sam.py::main \\
        --video "C:/Users/jdsch/Downloads/IMG_3545.MP4" \\
        --fps 10 \\
        --output pole_masks.npz

The masks.npz frame count/order matches what phase0_modal.py produces at
the same --fps so poses and masks stay aligned for Phase 1.3 triangulation.
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
        "numpy<2",
        "opencv-python-headless<4.11",
        "Pillow",
        "tqdm",
        "huggingface_hub",
        "matplotlib",
        "scikit-learn",
        "einops",
    )
    .run_commands(
        "pip install 'sam3[notebooks,train] @ "
        "git+https://github.com/facebookresearch/sam3.git@main'"
    )
)

hf_cache_vol = modal.Volume.from_name("hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("HF_TOKEN")


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    secrets=[hf_secret],
    timeout=60 * 30,
)
def segment(
    video_bytes: bytes,
    video_name: str,
    fps: int,
    text_prompt: str,
) -> dict:
    import io
    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import cv2
    import numpy as np
    import torch

    log_buf = io.StringIO()

    def log(msg):
        print(msg, flush=True)
        log_buf.write(msg + "\n")

    if os.environ.get("HF_TOKEN"):
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
    log(f"HF auth configured: {'yes' if os.environ.get('HUGGING_FACE_HUB_TOKEN') else 'NO'}")

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
            cv2.imwrite(f"{frames_dir}/{saved}.jpg", frame)
            saved += 1
        idx += 1
    cap.release()

    first = cv2.imread(f"{frames_dir}/0.jpg")
    H, W = first.shape[:2]
    ref_idx = 0
    log(f"Extracted {saved} frames at {W}x{H}, reference frame index {ref_idx}")
    log(f"Text prompt: {text_prompt!r}")

    from sam3.model_builder import build_sam3_multiplex_video_predictor
    # use_fa3=False routes through PyTorch SDPA. FA3 needs flash_attn_interface
    # (a separate package, Hopper-only fp8 kernels) which we don't have on A10G.
    predictor = build_sam3_multiplex_video_predictor(use_fa3=False)

    # Workaround for sam3 main @ c3a42ff: base predictor's start_session
    # unconditionally passes offload_state_to_cpu to model.init_state, but
    # the multiplex model's init_state doesn't accept that kwarg. Drop it.
    _orig_init_state = predictor.model.init_state

    def _patched_init_state(*args, **kwargs):
        kwargs.pop("offload_state_to_cpu", None)
        return _orig_init_state(*args, **kwargs)

    predictor.model.init_state = _patched_init_state
    log("SAM 3.1 predictor ready (init_state patched)")

    with torch.inference_mode():
        resp = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=frames_dir,
                offload_video_to_cpu=True,
            ),
        )
        session_id = resp["session_id"]
        log(f"session_id = {session_id}")

        resp = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=ref_idx,
                text=text_prompt,
            )
        )
        ref_out = resp["outputs"]
        n_objs_ref = len(ref_out.get("out_obj_ids", []))
        log(f"Reference frame {ref_idx}: SAM 3.1 found {n_objs_ref} matching object(s)")

        masks = np.zeros((saved, H, W), dtype=bool)
        hit_frames = 0
        mask_area_sum = 0.0

        for stream_resp in predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=session_id),
        ):
            frame_idx = stream_resp["frame_index"]
            out = stream_resp["outputs"]
            obj_ids = out.get("out_obj_ids", np.array([]))
            if hasattr(obj_ids, "tolist"):
                obj_ids = obj_ids.tolist()
            binary_masks = out.get("out_binary_masks", None)
            if binary_masks is None or len(obj_ids) == 0:
                continue
            combined = np.zeros((H, W), dtype=bool)
            for i in range(len(obj_ids)):
                m = binary_masks[i]
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                combined |= m
            if combined.any():
                masks[frame_idx] = combined
                hit_frames += 1
                mask_area_sum += float(combined.sum())

    pct_hit = 100.0 * hit_frames / max(saved, 1)
    mean_area = mask_area_sum / max(hit_frames, 1)
    log(f"Mask coverage: {hit_frames}/{saved} frames ({pct_hit:.1f}%)")
    log(f"Mean mask area (over hit frames): {mean_area:.0f} px  "
        f"({100.0 * mean_area / (H * W):.2f}% of frame)")

    masks_bytes_io = io.BytesIO()
    np.savez_compressed(
        masks_bytes_io,
        masks=masks,
        ref_frame=np.int32(ref_idx),
        text_prompt=np.array(text_prompt),
        image_hw=np.array([H, W], dtype=np.int32),
    )

    preview_idxs = np.linspace(0, saved - 1, 9).astype(int)
    tiles = []
    for i in preview_idxs:
        img = cv2.imread(f"{frames_dir}/{i}.jpg")
        m = masks[i]
        overlay = img.copy()
        red = np.zeros_like(overlay)
        red[..., 2] = 255
        alpha = (m.astype(np.float32) * 0.5)[..., None]
        overlay = (overlay * (1 - alpha) + red * alpha).astype(np.uint8)
        cv2.putText(overlay, f"f{i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if i == ref_idx:
            cv2.putText(overlay, "REF", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
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
    text: str = "utility pole",
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
        text_prompt=text,
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
