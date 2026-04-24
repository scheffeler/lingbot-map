# Utility-Pole Vision Tool — Plan

## Product thesis

Phone video walk-around → dense point cloud → segmented pole → measurements
(height, attachment heights, lean, diameter) → georeferenced export.
Zero-hardware capture. Competes with Katapult Pro, IKE GeoSpatial,
Pointivo, SPIDA, and Osmose on labor and cost — their workflows assume
LiDAR, RTK GNSS, or a measuring stick in frame; ours assumes a phone.

## Pipeline (revised 2026-04-23 after Phase 0 findings)

```
[1] lingbot-map streaming recon       → camera poses (reliable)
                                      → dense cloud for SCENE CONTEXT only
                                        (ground, trees, buildings, chunky
                                         attachments like transformer cans)
[2] Metric scale (EXIF GNSS / AR / tag)
[3] SAM 3.x (3.1 if available, else 3.0) per-frame pole mask (single click, video-propagated)
[4] Multi-view ray triangulation of mask centerline + boundary
    → pole geometry at sub-pixel precision
      (NOT dense-depth-derived; dense depth fails on thin poles)
[5] RANSAC pole axis + ground plane from triangulated points
[6] Measurements + uncertainty
[7] DINOv2-head classifier for pole class / attachment type
    (reuse aggregator features from step 1)
[8] Export (GeoJSON / KML / Katapult / SPIDAcalc)
[9] Measurement UI (viser extension)
```

**Why step 4 changed.** Phase 0 proved the base model's dense depth
does not reconstruct utility poles cleanly from either walkaround or
drive-by capture. Pole is typically ~20 cm against sky — low photometric
texture, thin silhouette. Scale also collapses on shorter captures.
Camera poses from the pose head are unaffected, so we keep those and
move pole geometry onto mask triangulation, which is a classical,
texture-independent method that works sub-pixel.

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

- **SAM 3.x (3.1 if available, else 3.0)** — video-propagated segmentation from one click per object.
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
| **1** | Measurement MVP: EXIF Sim(3) solver, SAM 3.x (3.1 if available, else 3.0) pole mask, RANSAC axis + ground, height measurement, viser "measure" panel. | 2–3 wk |
| **2** | Classification + multi-attachment: DINOv2-head classifier, SAM 3.x (3.1 if available, else 3.0) + Grounding-DINO for bulk attachments, uncertainty. | 3–4 wk |
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

## Phase 0 result (2026-04-23)

**Gate: cleared.** See `Phase_0_learnings.md` for the full write-up.

Short version: after fixing a shipped-checkpoint bug (lingbot-map.pt
is missing all 62 `point_head.*` weights, so we now unproject from the
depth head instead), the base model produces clean reconstructions on
the bundled `example/church` scene and a metric-ish scene around
captured pole walkarounds. Thin pole shafts still don't resolve as
crisp cylinders via dense depth alone — expected architectural
limitation, addressed by the Phase 1 mask-triangulation approach.

## Phase 1 — mask triangulation (next)

Architecture:

1. **Reconstruction primitive** = lingbot-map (depth head unprojection,
   poses from pose head). Already validated.
2. **Pole segmentation** = SAM 3.x (3.1 preferred if released at
   implementation time, otherwise 3.0). Single click on one frame,
   video-propagated to all frames.
3. **Pole geometry** = multi-view ray triangulation of the SAM mask
   centerline using the lingbot-map camera poses. Sub-pixel, texture-
   independent, handles thin verticals that dense MVS can't.
4. **Axis + ground plane** = RANSAC on the triangulated points.
5. **Measurements** = height, lean, attachment heights, approximate
   diameter at defined points.

First task for Phase 1: pose-verification script that exports
lingbot-map extrinsics from a walkaround run and plots the camera
trajectory to confirm the arc shape is usable before building
triangulation on top.

## Risks

- **Thin-vertical reconstruction.** Confirmed in Phase 0 — base model's
  dense depth does not cleanly resolve a ~20 cm pole against sky from
  realistic capture distances. Mitigation: mask triangulation (pipeline
  step 4 above).
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
- `scripts/phase0_modal.py` — cloud GPU runner (A100-40GB on Modal).
- `Phase_0_learnings.md` — full Phase 0 post-mortem.
