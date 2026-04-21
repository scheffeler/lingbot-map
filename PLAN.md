# Utility-Pole Vision Tool — Plan

## Product thesis

Phone video walk-around → dense point cloud → segmented pole → measurements
(height, attachment heights, lean, diameter) → georeferenced export.
Zero-hardware capture. Competes with Katapult Pro, IKE GeoSpatial,
Pointivo, SPIDA, and Osmose on labor and cost — their workflows assume
LiDAR, RTK GNSS, or a measuring stick in frame; ours assumes a phone.

## Pipeline

```
[1] lingbot-map streaming recon          ← exists (this repo)
[2] Metric scale (EXIF GNSS / AR / tag)  ← NEW (Phase 1)
[3] SAM 2 segmentation (pole + attachments)
[4] Mask → 3D fusion (lift 2D mask into world_points)
[5] RANSAC pole axis + ground plane
[6] Measurements + uncertainty
[7] DINOv2-head classifier (reuse aggregator features)
[8] Export (GeoJSON / KML / Katapult / SPIDAcalc)
[9] Measurement UI (viser extension)
```

## What the base model gives us

Feed-forward 3D reconstruction (DINOv2 backbone + geometric-context
transformer). Per frame, a single forward pass yields `extrinsic` (c2w),
`intrinsic`, `depth`, `depth_conf`, `world_points`, `world_points_conf`
— dense per-pixel 3D points in a shared world frame. ~20 FPS at
518×378 streaming; windowed mode for long sequences. Scale is
ambiguous by default — Phase 1 has to solve it.

## Metric-scale options (support all, graceful fallback)

1. **ARKit / ARCore VIO** — best UX, needs a companion app.
2. **EXIF GNSS Sim(3) fit** — default; zero user effort. Requires ≥1–2 m
   of user motion during capture.
   *Note:* MOV containers usually carry GPS in QuickTime metadata, not
   per-frame EXIF. Probe via `ffprobe -show_format` on the container,
   not on extracted frames.
3. **Known reference tap** — meter stick, AprilTag, or known crossarm
   length tapped in the UI.
4. **Learned pole-diameter prior** — weakest; flag output as "estimated".

## Meta-model fit (what plugs in naturally)

- **SAM 2** — video-propagated segmentation from one click per object.
  The unlock for labor-free 2D→3D object isolation.
- **DINOv2 features** (already computed in the aggregator) — reuse for
  pole class + attachment type via a small MLP head. Big compute win.
- **CoTracker3** (optional) — cross-frame mask consistency; lean/sway
  detection.
- **Depth-Anything-v2** (non-Meta complement) — fallback for single-
  still uploads or to pair with the aggregator on thin verticals.

## Phased build

| Phase | Scope | Duration |
|-------|-------|----------|
| **0** | Validate lingbot-map reconstructs a pole cleanly (smoke test). | now |
| **1** | Measurement MVP: EXIF Sim(3) solver, SAM 2 pole mask, RANSAC axis + ground, height measurement, viser "measure" panel. | 2–3 wk |
| **2** | Classification + multi-attachment: DINOv2-head classifier, SAM 2 + Grounding-DINO for bulk attachments, uncertainty. | 3–4 wk |
| **3** | Georef + export: WGS84 anchoring, GeoJSON/KML, PDF report, Katapult-compatible JSON, SPIDAcalc XML. | 2 wk |
| **4** | Capture app + cloud: iOS/Android capture (video + ARKit/ARCore pose + GNSS); cloud worker runs pipeline; web app for interactive 3D + measurements. | ongoing |

## Phase 0 — Validate the base

```bash
conda activate lingbot-map
python scripts/test_pole.py \
    --model_path ./checkpoints/lingbot-map.pt \
    --video_path "C:/Users/jdsch/Downloads/IMG_0039.MOV" \
    --output pole_cloud.ply
```

Open `pole_cloud.ply` in CloudCompare / MeshLab.

**Gate criteria:**

- Pole reconstructs as a coherent vertical structure (not sparse noise).
- Camera trajectory is stable across the orbit.
- Attachments (crossarm, transformer, insulators) have enough point
  density to segment.
- Ground plane is visible and roughly planar.

**If Phase 0 fails:** try `lingbot-map-long` checkpoint, lower `--fps`,
reduce `--conf_percentile`, or switch to `--mode windowed`. If still
broken, pair with Depth-Anything-v2 before the aggregator.

## Risks

- **Thin-vertical reconstruction.** Known weak spot of feed-forward
  MVS. This is the Phase 0 gate.
- **Short baseline captures.** <1 m of motion kills both recon and
  Sim(3) scale recovery. Capture-app UX must enforce motion.
- **Scale without GNSS/AR.** Diameter prior is a last resort; we
  should flag low-confidence measurements instead of silently guessing.
- **Regulatory.** Make-ready engineering deliverables (SPIDAcalc) have
  tight accuracy requirements. Phase 3 has to characterize our error
  bars honestly before we sell to utilities.

## Key files in this repo

- `demo.py:45` — `load_images`, video → frames → preprocessing.
- `demo.py:107` — `load_model`, checkpoint loader.
- `demo.py:170` — `postprocess`, pose → c2w, tensors → CPU.
- `demo.py:375` — streaming inference call.
- `lingbot_map/utils/geometry.py` — `unproject_depth_map_to_point_map`
  for measurement code (Phase 1).
- `lingbot_map/vis/point_cloud_viewer.py` — viser viewer; extend for
  the measurement UI in Phase 1.
- `lingbot_map/vis/glb_export.py` — reference for confidence-filtered
  cloud building. Our `test_pole.py` mirrors the filtering logic
  without the `trimesh` dependency.
- `scripts/test_pole.py` — Phase 0 entrypoint.
