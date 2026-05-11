"""Phase 1.4: SAM 3.1 multi-pole tracking on a folder of stills (Modal GPU).

Uses SAM 3.1's **video predictor** on a sorted image folder (treating
photos as video frames). One text prompt is added on the reference frame;
propagation tracks each detected object's identity across the rest of the
frames. This is what keeps "pole A" distinct from "pole B" when multiple
utility poles are visible — the per-image image-predictor approach picks
whichever pole has the highest score per frame independently and ends up
flipping between poles, ruining triangulation.

Output schema:
  masks:               (N_obj, S, H, W) bool — one mask track per object
  obj_ids:             (N_obj,) int          — SAM's internal IDs
  obj_present:         (N_obj, S) bool       — was object tracked on frame?
  image_hw:            (2,) int              — H, W of the native frames
  image_paths:         (S,) str              — sorted filenames (=poses order)
  text_prompt:         scalar str
  ref_frame:           scalar int            — which frame got the prompt

Frames are sorted alphabetically; matches `phase0_modal_imageset.py` so
masks and poses align by index.

Run:
    modal run scripts/phase1_sam_imageset.py \\
        --image-folder ./pole_001 \\
        --output pole_001.masks.npz
"""

import io
import os
import shutil
import zipfile
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
            ".heic", ".heif")

def parse_text_prompts(text: str) -> list[str]:
    """Split a comma-separated prompt list into individual SAM prompts.

    Each prompt becomes a separate `add_prompt` call on the reference
    frame, so SAM tracks one object track per matched class. Internal
    spaces are preserved (so "electrical bracket" survives intact);
    only commas delimit prompts.
    """
    parts = [p.strip() for p in text.split(",")]
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError(
            f"No prompts in {text!r}; provide at least one comma-separated "
            f"phrase, e.g. 'utility pole, crossarm'."
        )
    return parts


app = modal.App("polevision-phase1-sam-imageset")

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
        "pillow-heif",
        "tqdm",
        "huggingface_hub",
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
    images_zip: bytes,
    text_prompt: str,
    ref_frame: int,
) -> dict:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from pillow_heif import register_heif_opener

    register_heif_opener()
    log_buf = io.StringIO()

    def log(msg):
        print(msg, flush=True)
        log_buf.write(msg + "\n")

    if os.environ.get("HF_TOKEN"):
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])
    log(f"HF auth: {'yes' if os.environ.get('HUGGING_FACE_HUB_TOKEN') else 'NO'}")

    # Stage 1: unpack and convert to JPEGs the SAM video predictor can read.
    # SAM 3.1's video predictor reads frames from a directory of files named
    # 0.jpg, 1.jpg, ... in order. We re-encode HEIC/PNG/etc to JPEG at full
    # resolution so the predictor doesn't have to know about pillow-heif.
    in_dir = "/tmp/pole_images"
    frames_dir = "/tmp/frames"
    for d in (in_dir, frames_dir):
        os.makedirs(d, exist_ok=True)
        for n in os.listdir(d):
            os.remove(os.path.join(d, n))
    with zipfile.ZipFile(io.BytesIO(images_zip)) as zf:
        zf.extractall(in_dir)
    sources = sorted(
        os.path.join(in_dir, n) for n in os.listdir(in_dir)
        if n.lower().endswith(IMG_EXTS)
    )
    image_basenames = [os.path.basename(p) for p in sources]
    for i, src in enumerate(sources):
        dst = f"{frames_dir}/{i}.jpg"
        if src.lower().endswith((".jpg", ".jpeg")):
            shutil.copyfile(src, dst)
        else:
            Image.open(src).convert("RGB").save(dst, "JPEG", quality=95)
    first = cv2.imread(f"{frames_dir}/0.jpg")
    H, W = first.shape[:2]
    S = len(sources)
    log(f"Prepared {S} frame(s) at {W}x{H}; reference frame = {ref_frame}")
    log(f"Text prompt: {text_prompt!r}")

    from sam3.model_builder import build_sam3_multiplex_video_predictor
    # Same FA3 / init_state workarounds as phase1_sam.py.
    predictor = build_sam3_multiplex_video_predictor(use_fa3=False)
    _orig_init_state = predictor.model.init_state

    def _patched_init_state(*args, **kwargs):
        kwargs.pop("offload_state_to_cpu", None)
        return _orig_init_state(*args, **kwargs)

    predictor.model.init_state = _patched_init_state
    log("SAM 3.1 video predictor ready")

    # Stage 2: start session + add text prompt on reference frame +
    # propagate. Same dance as phase1_sam.py.
    with torch.inference_mode():
        resp = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=frames_dir,
                offload_video_to_cpu=True,
            ),
        )
        session_id = resp["session_id"]

        prompts = parse_text_prompts(text_prompt)
        log(f"Prompt list ({len(prompts)}): {prompts}")

        ref_obj_ids: list = []
        class_names: list[str] = []
        # Per-prompt ref-frame masks, kept as a list of (mask, oid)
        # tuples so we can stamp them after `masks` is allocated below.
        ref_frame_masks: list[tuple] = []
        for prompt in prompts:
            resp = predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=ref_frame,
                    text=prompt,
                )
            )
            ids = resp["outputs"].get("out_obj_ids", [])
            if hasattr(ids, "tolist"):
                ids = ids.tolist()
            log(f"  prompt {prompt!r}: matched {len(ids)} object(s); ids={ids}")
            binary = resp["outputs"].get("out_binary_masks", None)
            for k_in, oid in enumerate(ids):
                if oid in ref_obj_ids:
                    # SAM may return overlapping detections across prompts;
                    # keep the first prompt that named it.
                    continue
                ref_obj_ids.append(oid)
                class_names.append(prompt)
                if binary is not None and k_in < len(binary):
                    ref_frame_masks.append((oid, binary[k_in]))

        # Output: a fixed-size (N_obj, S, H, W) bool tensor where N_obj is
        # frozen at the reference-frame count. If propagation later loses
        # one of the objects on a frame, that frame's slice stays zero
        # and obj_present[i, frame] is False.
        n_obj = len(ref_obj_ids)
        if n_obj == 0:
            raise RuntimeError(
                f"No objects matched any prompt in {prompts!r} on "
                f"reference frame {ref_frame}. Try different prompts or "
                f"a different reference frame."
            )
        masks = np.zeros((n_obj, S, H, W), dtype=bool)
        obj_present = np.zeros((n_obj, S), dtype=bool)
        # Stamp the reference frame's masks now (propagate_in_video may
        # not re-yield them). Each entry is (oid, raw_mask).
        for oid, m in ref_frame_masks:
            k = ref_obj_ids.index(oid)
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            masks[k, ref_frame] = m
            obj_present[k, ref_frame] = bool(m.any())

        for stream_resp in predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=session_id),
        ):
            f = stream_resp["frame_index"]
            out = stream_resp["outputs"]
            obj_ids = out.get("out_obj_ids", [])
            if hasattr(obj_ids, "tolist"):
                obj_ids = obj_ids.tolist()
            binary = out.get("out_binary_masks", None)
            if binary is None or len(obj_ids) == 0:
                continue
            for k_out, oid in enumerate(obj_ids):
                # Map back to the reference-frame ordering.
                if oid not in ref_obj_ids:
                    continue
                k = ref_obj_ids.index(oid)
                m = binary[k_out]
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
                if m.any():
                    masks[k, f] = m
                    obj_present[k, f] = True

    # Coverage report per object.
    log("\nPer-object coverage:")
    for k, oid in enumerate(ref_obj_ids):
        present = obj_present[k]
        areas = masks[k].reshape(S, -1).sum(axis=1)
        mean_area = areas[present].mean() if present.any() else 0.0
        log(f"  object {k} (id={oid}): {present.sum()}/{S} frames, "
            f"mean area on present frames {mean_area:.0f} px "
            f"({100.0 * mean_area / (H * W):.2f}%)")

    masks_buf = io.BytesIO()
    np.savez_compressed(
        masks_buf,
        masks=masks,
        obj_ids=np.array(ref_obj_ids, dtype=np.int64),
        class_names=np.array(class_names),
        obj_present=obj_present,
        ref_frame=np.int32(ref_frame),
        text_prompt=np.array(text_prompt),
        image_hw=np.array([H, W], dtype=np.int32),
        image_paths=np.array(image_basenames),
    )

    # Per-object preview grid: 3x3 sample of frames, one tile each, with
    # all tracked objects colored differently so a human can see which
    # object number maps to which physical pole.
    obj_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
                  (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    preview_idxs = np.linspace(0, S - 1, 9).astype(int)
    tiles = []
    for f in preview_idxs:
        img_bgr = cv2.imread(f"{frames_dir}/{f}.jpg")
        if img_bgr.shape[:2] != (H, W):
            img_bgr = cv2.resize(img_bgr, (W, H))
        overlay = img_bgr.copy()
        for k in range(n_obj):
            if not obj_present[k, f]:
                continue
            color = np.zeros_like(overlay)
            color[..., 0] = obj_colors[k % len(obj_colors)][0]
            color[..., 1] = obj_colors[k % len(obj_colors)][1]
            color[..., 2] = obj_colors[k % len(obj_colors)][2]
            alpha = (masks[k, f].astype(np.float32) * 0.45)[..., None]
            overlay = (overlay * (1 - alpha) + color * alpha).astype(np.uint8)
        cv2.putText(overlay, f"f{f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        if f == ref_frame:
            cv2.putText(overlay, "REF", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        tile = cv2.resize(overlay, (W // 3, H // 3))
        tiles.append(tile)
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, 9, 3)]
    grid = np.vstack(rows)
    # Legend strip at bottom: "obj 0: red | obj 1: green | ..."
    legend_h = 60
    legend = np.full((legend_h, grid.shape[1], 3), 255, dtype=np.uint8)
    x = 20
    for k in range(n_obj):
        col = obj_colors[k % len(obj_colors)]
        cv2.rectangle(legend, (x, 15), (x + 50, 45), col, thickness=-1)
        cv2.putText(legend, f"obj {k} (id={ref_obj_ids[k]})",
                    (x + 60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        x += 320
    grid = np.vstack([grid, legend])
    ok, preview_bytes = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    preview = preview_bytes.tobytes() if ok else b""

    return {
        "masks": masks_buf.getvalue(),
        "preview": preview,
        "logs": log_buf.getvalue(),
        "returncode": 0,
    }


def _zip_image_folder(folder: Path) -> tuple[bytes, list[str]]:
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    if not files:
        raise SystemExit(f"No supported images in {folder}. "
                         f"Looking for {IMG_EXTS}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)
    return buf.getvalue(), [f.name for f in files]


@app.local_entrypoint()
def main(
    image_folder: str,
    output: str = "pole_masks.npz",
    preview: str = "pole_masks_preview.jpg",
    text: str = "utility pole",
    ref_frame: int = 0,
):
    folder = Path(image_folder)
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {image_folder}")

    zip_bytes, names = _zip_image_folder(folder)
    print(f"Uploading {len(names)} image(s) ({len(zip_bytes) / 1e6:.1f} MB) "
          f"from {folder} to Modal...")
    print(f"  ref frame: {ref_frame} ({names[ref_frame]})")

    result = segment.remote(
        images_zip=zip_bytes,
        text_prompt=text,
        ref_frame=ref_frame,
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
