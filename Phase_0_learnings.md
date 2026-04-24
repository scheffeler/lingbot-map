# Phase 0 — Learnings

Phase 0 goal was to determine whether lingbot-map can serve as the base
reconstruction primitive for a phone-video utility-pole measurement
pipeline. After several false starts, the answer is **yes for scene
geometry and camera poses; no for thin-pole geometry via dense depth
alone** — which tees up Phase 1 exactly as designed.

## Pipeline verdicts

| Subject | Result |
|---|---|
| Driving drive-by capture (45 mph, 92 s) | Too much motion per frame + low parallax. Unusable. |
| Walkaround capture (180°, ~6 m radius, 20 s) | Scene reconstructs cleanly. Pole resolves as a fuzzy vertical but not a crisp cylinder. |
| Bundled `example/church` scene | Reconstructs cleanly after fixes. This is what validated the pipeline end-to-end. |

## Bugs we hit (in order)

1. **FlashInfer dependency** blocks inference out of the box. Fix: default
   `--use_sdpa` in `scripts/test_pole.py`. PyTorch SDPA is a supported
   path per the README.
2. **A10G (24 GB) too small** for 64-frame windowed global attention at
   518×518. Fix: A100-40GB on Modal. Still tight at ~22 GB for streaming
   and ~37 GB for windowed on long clips.
3. **Preprocessing `mode="crop"`** center-crops height on portrait video,
   chopping the crossarm out of frame entirely. Fix: `--crop_mode pad`
   default, preserves full aspect ratio. Note: for landscape input
   (church scene) `crop` is still the right choice — it keeps native
   aspect, while `pad` adds white bars that are out-of-distribution.
4. **`point_head` weights missing from `lingbot-map.pt`** (the
   root-cause bug that wasted most of the session). The shipped
   checkpoint has 0 of 62 `point_head.*` tensors. Reading from
   `predictions["world_points"]` gives random output that collapses to
   a ~flat disc at uniform depth. Fix: use
   `lingbot_map.utils.geometry.unproject_depth_map_to_point_map` to
   compute world points from the (properly loaded) depth head +
   extrinsic + intrinsic. The handoff brief flagged this utility —
   should have wired it in from the start.
5. **Sky masking over-aggressive on thin verticals.** The skyseg ONNX
   model classifies pole-edge pixels as sky, so `--mask_sky` drops
   pole-shaft points. Fine for scene-level visualization, wrong primitive
   for pole pixels. Phase 1 needs a pole-aware segmenter (SAM 3.x).

## Lessons

- **Read the README end-to-end before running.** We missed `--mask_sky`
  and the bundled example scenes on the first pass. Validating against
  known-good captures would have caught the `point_head` bug the same day.
- **A "Missing keys: N" warning is not always benign.** Print the key
  names before dismissing; 62 missing keys here were *all* the
  pointmap head.
- **Feed-forward dense MVS cannot resolve ~20 cm verticals against
  untextured sky** at realistic capture distances, even with ideal
  walk-around motion. This is a known limitation of the architecture
  family (lingbot-map, VGGT, DUSt3R/MASt3R), not a bug.

## What's reliable going forward

- **Camera poses** from lingbot-map's pose head are trustworthy enough
  to build mask-triangulation on top of. The church reconstruction
  proves pose + depth are self-consistent for chunky scenes.
- **Scene context** (ground plane, foliage, buildings, transformer cans
  and similar chunky attachments) reconstructs fine. Useful for
  visualization, ground-plane fit, and attachment segmentation.

## What's not reliable

- **`predictions["world_points"]`** and **`predictions["world_points_conf"]`**
  are garbage with the currently-shipped checkpoint. Always unproject
  from depth instead.
- **Sky masking** for the pole-pixel filter. Works for visualization
  cleanup only.
- **Thin vertical structures** via dense depth alone — pole shafts,
  primary/secondary wires, guy wires will all need mask triangulation.

## Cost / time signal

- Modal A100-40GB: ~$2/hr. A church run: ~2 min compute, <$0.10. A
  walkaround pole run: similar.
- First-run overhead: image build (~3-5 min) + checkpoint download
  (~3 min) + skyseg download (~30 s). All cached in a Modal Volume
  afterwards.
- Local workstation has no CUDA GPU (Intel UHD 730) — all inference
  must go through Modal.

## Phase 1 entry conditions (all met)

- Phase 0 produces a scene reconstruction with meaningful metric-ish
  scale and real depth variation.
- Camera poses from the walkaround are plausible (180° arc shape
  visible in the cloud's camera trail).
- The failure mode on the pole is *thin-structure-specific* rather than
  pipeline-wide, matching the architectural bet that a pole-aware
  mask + multi-view triangulation will fix it.
