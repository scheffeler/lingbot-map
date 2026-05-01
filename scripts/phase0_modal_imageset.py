"""Phase 1.4: Run lingbot-map pose/depth on a folder of stills (Modal GPU).

Image-set replacement for `phase0_modal.py`. Accepts JPEG / PNG / HEIC.

Run:
    modal run scripts/phase0_modal_imageset.py \\
        --image-folder ./pole_001 \\
        --output pole_001.ply

Writes `pole_001.ply` and `pole_001.poses.npz` (same NPZ schema as the
video entrypoint, so `triangulate_pole.py` consumes it unchanged).
"""

import io
import os
import zipfile
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
            ".heic", ".heif")

app = modal.App("polevision-phase0-imageset")

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
        "pillow-heif",
        "huggingface_hub",
        "einops",
        "safetensors",
        "scipy",
        "tqdm",
        "onnxruntime-gpu",
        "requests",
        # Pulled in transitively by `lingbot_map.vis.__init__` (sky
        # segmentation imports matplotlib via point_cloud_viewer).
        "matplotlib",
        "viser",
        "trimesh",
    )
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/root/lingbot-map",
        copy=True,
        ignore=[
            "checkpoints",
            "*_frames",
            "*.ply",
            ".git",
            "__pycache__",
            "*.pyc",
            ".venv",
            "node_modules",
        ],
    )
    .run_commands(
        "cd /root/lingbot-map && pip install -e . --no-deps",
    )
)

ckpt_vol = modal.Volume.from_name("lingbot-map-ckpt", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/ckpt": ckpt_vol},
    timeout=60 * 30,
)
def reconstruct(
    images_zip: bytes,
    conf_percentile: float,
    mode: str,
    window_size: int,
    overlap_size: int,
    downsample: int,
    crop_mode: str,
    mask_sky: bool,
) -> dict:
    import subprocess
    import sys

    os.chdir("/root/lingbot-map")
    sys.path.insert(0, "/root/lingbot-map")

    # Register HEIC opener so PIL.Image.open(...) handles iPhone exports.
    from pillow_heif import register_heif_opener
    register_heif_opener()

    img_dir = "/tmp/pole_images"
    os.makedirs(img_dir, exist_ok=True)
    for name in os.listdir(img_dir):
        os.remove(os.path.join(img_dir, name))
    with zipfile.ZipFile(io.BytesIO(images_zip)) as zf:
        zf.extractall(img_dir)
    n_imgs = sum(1 for n in os.listdir(img_dir)
                 if n.lower().endswith(IMG_EXTS))
    print(f"Unpacked {n_imgs} image(s) into {img_dir}")

    ckpt_path = "/ckpt/lingbot-map.pt"
    if not os.path.exists(ckpt_path):
        from huggingface_hub import snapshot_download
        print("Fetching checkpoint from HF (first run only)...")
        snapshot_download(repo_id="robbyant/lingbot-map", local_dir="/ckpt")
        ckpt_vol.commit()

    skyseg_path = "/ckpt/skyseg.onnx"
    if mask_sky and not os.path.exists(skyseg_path):
        from lingbot_map.vis.sky_segmentation import download_skyseg_model
        download_skyseg_model(skyseg_path)
        ckpt_vol.commit()

    out_path = "/tmp/pole_cloud.ply"
    cmd = [
        "python", "-u", "scripts/test_pole.py",
        "--model_path", ckpt_path,
        "--output", out_path,
        "--image_folder", img_dir,
        "--conf_percentile", str(conf_percentile),
        "--mode", mode,
        "--window_size", str(window_size),
        "--overlap_size", str(overlap_size),
        "--downsample", str(downsample),
        "--crop_mode", crop_mode,
        "--skyseg_model_path", skyseg_path,
    ]
    if mask_sky:
        cmd.append("--mask_sky")

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    logs = result.stdout + (
        "\n--- STDERR ---\n" + result.stderr if result.stderr else ""
    )

    ply_bytes = Path(out_path).read_bytes() if os.path.exists(out_path) else b""
    poses_path = os.path.splitext(out_path)[0] + ".poses.npz"
    poses_bytes = Path(poses_path).read_bytes() if os.path.exists(poses_path) else b""

    return {
        "ply": ply_bytes,
        "poses": poses_bytes,
        "logs": logs,
        "returncode": result.returncode,
    }


def _zip_image_folder(folder: Path) -> tuple[bytes, list[str]]:
    """Zip every supported image in `folder` (non-recursive) into a bytes blob.
    Returns (zip_bytes, sorted_filenames). Sort order matches what
    test_pole.py / triangulate_pole.py see, so masks and poses align."""
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
    output: str = "pole_cloud.ply",
    conf_percentile: float = 50.0,
    mode: str = "streaming",
    window_size: int = 64,
    overlap_size: int = 16,
    downsample: int = 1,
    crop_mode: str = "pad",
    mask_sky: bool = False,
):
    folder = Path(image_folder)
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {image_folder}")

    zip_bytes, names = _zip_image_folder(folder)
    print(f"Uploading {len(names)} image(s) ({len(zip_bytes) / 1e6:.1f} MB) "
          f"from {folder} to Modal...")
    print(f"  first: {names[0]}, last: {names[-1]}")

    result = reconstruct.remote(
        images_zip=zip_bytes,
        conf_percentile=conf_percentile,
        mode=mode,
        window_size=window_size,
        overlap_size=overlap_size,
        downsample=downsample,
        crop_mode=crop_mode,
        mask_sky=mask_sky,
    )

    print("\n=== Remote logs ===")
    print(result["logs"])
    print("===================\n")

    if result["returncode"] != 0:
        raise SystemExit(f"Remote run failed (exit {result['returncode']})")

    if not result["ply"]:
        raise SystemExit("Remote run returned no PLY bytes.")

    out = Path(output)
    out.write_bytes(result["ply"])
    print(f"Wrote {out} ({len(result['ply']) / 1e6:.1f} MB)")

    poses_bytes = result.get("poses") or b""
    if poses_bytes:
        poses_out = out.with_suffix("").with_suffix(".poses.npz")
        poses_out.write_bytes(poses_bytes)
        print(f"Wrote {poses_out} ({len(poses_bytes) / 1e3:.1f} KB)")
