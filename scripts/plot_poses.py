"""Plot camera trajectory from a saved poses.npz.

Use this to verify lingbot-map's pose head produced a sensible
trajectory before building multi-view triangulation on top.

    python scripts/plot_poses.py pole_walkaround.poses.npz

Expected for a ~180 deg walkaround of a pole: cameras form a smooth
arc of roughly-constant radius in the ground plane, all pointing
inward. Expected for a drive-by: roughly-straight line of cameras
all pointing forward along the same axis.

Fails loud (non-zero exit) if no file is passed. Requires matplotlib.
"""

import argparse
import os
import sys

import numpy as np


def _camera_positions(extrinsic: np.ndarray) -> np.ndarray:
    """Extract camera centers from c2w extrinsics (S, 3, 4)."""
    return extrinsic[:, :3, 3]


def _camera_forwards(extrinsic: np.ndarray) -> np.ndarray:
    """Extract camera +Z (viewing direction) from c2w rotation columns."""
    return extrinsic[:, :3, 2]


def _arc_stats(positions: np.ndarray) -> dict:
    center = positions.mean(axis=0)
    radii = np.linalg.norm(positions - center, axis=1)
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    total_path = float(steps.sum())

    if len(positions) >= 3:
        v0 = positions[0] - center
        vN = positions[-1] - center
        cos = np.clip(
            np.dot(v0, vN) / (np.linalg.norm(v0) * np.linalg.norm(vN) + 1e-9),
            -1.0, 1.0,
        )
        sweep_deg = float(np.degrees(np.arccos(cos)))
    else:
        sweep_deg = 0.0

    return {
        "center": center,
        "radius_mean": float(radii.mean()),
        "radius_std": float(radii.std()),
        "path_length": total_path,
        "step_mean": float(steps.mean()) if len(steps) else 0.0,
        "step_max": float(steps.max()) if len(steps) else 0.0,
        "sweep_deg": sweep_deg,
    }


def main():
    ap = argparse.ArgumentParser(description="Plot camera trajectory from poses.npz")
    ap.add_argument("poses_path", help="Path to <run>.poses.npz")
    ap.add_argument("--save", default=None,
                    help="Save plot to this path instead of showing interactively.")
    args = ap.parse_args()

    if not os.path.exists(args.poses_path):
        sys.exit(f"File not found: {args.poses_path}")

    data = np.load(args.poses_path, allow_pickle=True)
    extrinsic = data["extrinsic"]
    positions = _camera_positions(extrinsic)
    forwards = _camera_forwards(extrinsic)

    stats = _arc_stats(positions)
    print(f"Frames: {len(positions)}")
    print(f"Centroid: {stats['center']}")
    print(f"Mean radius from centroid: {stats['radius_mean']:.3f} "
          f"(std {stats['radius_std']:.3f})")
    print(f"Path length: {stats['path_length']:.3f}")
    print(f"Step mean/max: {stats['step_mean']:.3f} / {stats['step_max']:.3f}")
    print(f"Approx sweep angle (first-to-last about centroid): "
          f"{stats['sweep_deg']:.1f} deg")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], "-o", markersize=2)
    arrow_len = max(stats["radius_mean"] * 0.15, 0.1)
    ax1.quiver(
        positions[:, 0], positions[:, 1], positions[:, 2],
        forwards[:, 0], forwards[:, 1], forwards[:, 2],
        length=arrow_len, normalize=True, color="tab:red", alpha=0.5,
    )
    ax1.scatter(*stats["center"], color="tab:green", s=60, label="centroid")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.set_title(f"Camera trajectory (3D)\n{len(positions)} frames, "
                  f"sweep ~{stats['sweep_deg']:.0f} deg")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(positions[:, 0], positions[:, 2], "-o", markersize=3)
    ax2.scatter(stats["center"][0], stats["center"][2], color="tab:green", s=60,
                label="centroid")
    ax2.set_xlabel("X"); ax2.set_ylabel("Z (forward)")
    ax2.set_title(f"Top-down (X-Z)\nmean radius {stats['radius_mean']:.2f} "
                  f"(+/- {stats['radius_std']:.2f})")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=120)
        print(f"Wrote {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
