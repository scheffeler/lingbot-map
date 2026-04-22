"""Run the Phase 0 pole-reconstruction smoke test on Modal (cloud GPU).

One-time:
    pip install modal
    modal setup

Run:
    modal run scripts/phase0_modal.py \\
        --video "C:/Users/jdsch/Downloads/IMG_0039.MOV" \\
        --output pole_cloud.ply

The function streams stdout back to your terminal and writes the PLY
locally when it finishes. The lingbot-map checkpoint is cached in a
Modal Volume after the first run, so subsequent runs start in seconds.
"""

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent

app = modal.App("polevision-phase0")

# CUDA 12.8 + Python 3.10 + torch 2.9.1-cu128, then repo code installed in place.
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
        "huggingface_hub",
        "einops",
        "safetensors",
        "scipy",
        "tqdm",
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
    video_bytes: bytes,
    video_name: str,
    fps: int,
    conf_percentile: float,
    mode: str,
    window_size: int,
    overlap_size: int,
    downsample: int,
) -> dict:
    import os
    import subprocess
    import sys

    os.chdir("/root/lingbot-map")
    sys.path.insert(0, "/root/lingbot-map")

    ckpt_path = "/ckpt/lingbot-map.pt"
    if not os.path.exists(ckpt_path):
        from huggingface_hub import snapshot_download

        print("Fetching checkpoint from HF (first run only)...")
        snapshot_download(repo_id="robbyant/lingbot-map", local_dir="/ckpt")
        ckpt_vol.commit()
        print("Checkpoint cached in volume. Contents:")
        for name in sorted(os.listdir("/ckpt")):
            size = os.path.getsize(os.path.join("/ckpt", name))
            print(f"  {name}  ({size / 1e9:.2f} GB)")

    if not os.path.exists(ckpt_path):
        candidates = [
            f for f in os.listdir("/ckpt")
            if f.endswith(".pt") or f.endswith(".safetensors")
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No .pt or .safetensors file in /ckpt — found: {os.listdir('/ckpt')}"
            )
        ckpt_path = os.path.join("/ckpt", candidates[0])
        print(f"Using checkpoint: {ckpt_path}")

    video_path = f"/tmp/{video_name}"
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    out_path = "/tmp/pole_cloud.ply"
    cmd = [
        "python", "-u", "scripts/test_pole.py",
        "--model_path", ckpt_path,
        "--video_path", video_path,
        "--output", out_path,
        "--fps", str(fps),
        "--conf_percentile", str(conf_percentile),
        "--mode", mode,
        "--window_size", str(window_size),
        "--overlap_size", str(overlap_size),
        "--downsample", str(downsample),
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    logs = result.stdout + ("\n--- STDERR ---\n" + result.stderr if result.stderr else "")

    ply_bytes = b""
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            ply_bytes = f.read()

    return {
        "ply": ply_bytes,
        "logs": logs,
        "returncode": result.returncode,
    }


@app.local_entrypoint()
def main(
    video: str,
    output: str = "pole_cloud.ply",
    fps: int = 10,
    conf_percentile: float = 50.0,
    mode: str = "streaming",
    window_size: int = 64,
    overlap_size: int = 16,
    downsample: int = 1,
):
    video_path = Path(video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video}")

    size_mb = video_path.stat().st_size / 1e6
    print(f"Uploading {video_path.name} ({size_mb:.1f} MB) to Modal...")

    result = reconstruct.remote(
        video_bytes=video_path.read_bytes(),
        video_name=video_path.name,
        fps=fps,
        conf_percentile=conf_percentile,
        mode=mode,
        window_size=window_size,
        overlap_size=overlap_size,
        downsample=downsample,
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
