"""Phase 1.7 (companion): standalone HTML mask viewer.

Generates a single self-contained HTML file with the source JPEGs and
the SAM 3.1 masks embedded as base64. Opens directly in a browser; no
server, no extra installs. Lets you flip through frames and toggle each
tracked object's mask on/off independently — useful for sanity-checking
which SAM object id corresponds to which physical pole before invoking
`triangulate_pole.py --object-id N`.

Run:
    python scripts/visualize_masks.py \\
        --masks pole_001.masks.npz \\
        --image-folder ./pole_001 \\
        --output pole_001_masks.html

Then open `pole_001_masks.html` in any browser.
"""

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def encode_jpeg(arr: np.ndarray, quality: int = 80) -> str:
    """Encode an RGB uint8 array to a base64 data: URL (JPEG)."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def encode_png_alpha(mask: np.ndarray, color: tuple[int, int, int]) -> str:
    """Encode a bool mask as a transparent PNG: opaque-coloured where the
    mask is True, fully transparent elsewhere. Tiny on disk because the
    PNG compression handles the constant transparent regions cheaply."""
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask, 0] = color[0]
    rgba[mask, 1] = color[1]
    rgba[mask, 2] = color[2]
    rgba[mask, 3] = 255  # opaque on the mask, alpha handled in CSS
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


OBJECT_PALETTE = [
    (220, 50, 50), (50, 200, 80), (60, 100, 230),
    (230, 180, 50), (200, 80, 200), (60, 200, 200),
]


def generate_html(
    masks_path: str,
    image_folder: str,
    max_width: int = 1024,
    jpeg_quality: int = 80,
    gps_scale_path: str | None = None,
    quiet: bool = False,
) -> str:
    """Build the self-contained mask-viewer HTML and return it as a
    string. Called both from the CLI `main()` and from the FastAPI
    `/api/captures/{id}/mask-viewer` endpoint."""
    masks_data = np.load(masks_path, allow_pickle=True)
    masks_arr = masks_data["masks"]
    if masks_arr.ndim == 3:
        masks_arr = masks_arr[None]                # legacy single-object
    n_obj, S, native_h, native_w = masks_arr.shape
    obj_ids = ([int(x) for x in masks_data["obj_ids"]]
               if "obj_ids" in masks_data.files else list(range(n_obj)))
    image_basenames = ([str(p) for p in masks_data["image_paths"]]
                       if "image_paths" in masks_data.files else None)
    text_prompt = (str(masks_data["text_prompt"])
                   if "text_prompt" in masks_data.files else "?")
    ref_frame = int(masks_data["ref_frame"]) if "ref_frame" in masks_data.files else 0

    folder = Path(image_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")

    if image_basenames is None:
        # Fall back to alphabetical order — matches phase1_sam_imageset.
        IMG_EXTS = (".jpg", ".jpeg", ".png", ".heic", ".heif")
        image_basenames = sorted(
            p.name for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )
    if len(image_basenames) != S:
        raise ValueError(
            f"Mask frame count {S} != image count {len(image_basenames)}"
        )

    # Optional HEIC support.
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass

    if not quiet:
        print(f"Embedding {S} frames x {n_obj} object(s)")

    frames = []
    for f, name in enumerate(image_basenames):
        path = folder / name
        if not path.exists():
            raise FileNotFoundError(f"Missing image: {path}")
        im = Image.open(path).convert("RGB")
        w, h = im.size
        if w > max_width:
            new_h = int(round(h * max_width / w))
            im = im.resize((max_width, new_h), Image.BILINEAR)
        rgb = np.asarray(im)
        out_h, out_w = rgb.shape[:2]
        img_url = encode_jpeg(rgb, quality=jpeg_quality)

        # Resize each object mask to match the displayed photo.
        mask_urls = []
        mask_areas = []
        for k in range(n_obj):
            m = masks_arr[k, f]
            if m.shape != (out_h, out_w):
                m = np.asarray(
                    Image.fromarray(m.astype(np.uint8) * 255).resize(
                        (out_w, out_h), Image.NEAREST,
                    )
                ) > 0
            mask_areas.append(int(m.sum()))
            mask_urls.append(encode_png_alpha(m, OBJECT_PALETTE[k % len(OBJECT_PALETTE)]))

        frames.append({
            "name": name,
            "image": img_url,
            "masks": mask_urls,
            "areas": mask_areas,
            "w": out_w, "h": out_h,
        })
        if not quiet:
            print(f"  frame {f}: {name}  {out_w}x{out_h}  "
                  f"areas={mask_areas} px")

    gps_block = ""
    if gps_scale_path:
        try:
            gs = json.loads(Path(gps_scale_path).read_text())
            gps_block = (f"GPS scale {gs.get('scale', 0):.3f} m/unit"
                         + (f" (residual {gs['residual_m']:.2f} m)"
                            if 'residual_m' in gs else ""))
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    obj_legend = []
    for k in range(n_obj):
        c = OBJECT_PALETTE[k % len(OBJECT_PALETTE)]
        obj_legend.append({
            "k": k,
            "id": obj_ids[k],
            "css_color": f"rgb({c[0]},{c[1]},{c[2]})",
        })

    payload = {
        "frames": frames,
        "objects": obj_legend,
        "ref_frame": ref_frame,
        "text_prompt": text_prompt,
        "gps_block": gps_block,
        "n_obj": n_obj,
        "S": S,
        # Native (mask-space) dimensions; clicks in displayed image
        # space get rescaled to these so scale_solver receives the
        # correct coordinates.
        "native_w": int(native_w),
        "native_h": int(native_h),
    }

    html_str = TEMPLATE.replace(
        "/*PAYLOAD*/",
        json.dumps(payload),
    ).replace("/*PROMPT*/", html.escape(text_prompt))
    return html_str


def main():
    ap = argparse.ArgumentParser(description="Standalone HTML mask viewer")
    ap.add_argument("--masks", required=True,
                    help="multi-object masks NPZ from phase1_sam_imageset")
    ap.add_argument("--image-folder", required=True,
                    help="folder of source JPEG/HEIC photos")
    ap.add_argument("--output", default="masks_viewer.html")
    ap.add_argument("--max-width", type=int, default=1024,
                    help="downscale photos and masks to this max width "
                         "before embedding (keeps the HTML small)")
    ap.add_argument("--jpeg-quality", type=int, default=80)
    ap.add_argument("--gps-scale", default=None,
                    help="optional gps_scale.json — included in the header")
    args = ap.parse_args()
    html_str = generate_html(
        masks_path=args.masks,
        image_folder=args.image_folder,
        max_width=args.max_width,
        jpeg_quality=args.jpeg_quality,
        gps_scale_path=args.gps_scale,
    )
    Path(args.output).write_text(html_str, encoding="utf-8")
    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"\nWrote {args.output} ({size_mb:.1f} MB)")
    print(f"Open in any browser: file:///{Path(args.output).resolve()}")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SAM mask viewer</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
         Roboto, sans-serif; margin: 0; background: #1a1a1a; color: #ddd; }
  header { padding: 12px 18px; background: #111; border-bottom: 1px solid #333;
           display: flex; align-items: center; gap: 18px; flex-wrap: wrap; }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  header .meta { font-size: 13px; color: #aaa; }
  main { display: flex; min-height: calc(100vh - 60px); }
  aside { width: 240px; padding: 14px; background: #222; border-right: 1px solid #333;
          font-size: 13px; }
  aside h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .05em;
             color: #999; margin: 0 0 8px 0; }
  aside .group { margin-bottom: 18px; }
  aside label { display: flex; align-items: center; gap: 8px;
                padding: 4px 0; cursor: pointer; }
  aside .swatch { width: 14px; height: 14px; border-radius: 2px;
                  display: inline-block; }
  aside .area { color: #888; font-size: 11px; margin-left: auto; font-variant-numeric: tabular-nums; }
  aside input[type=range] { width: 100%; }
  .frame-strip { display: flex; gap: 6px; overflow-x: auto;
                 padding: 8px 14px; background: #222; }
  .frame-strip button { width: 80px; padding: 6px 4px; flex: 0 0 auto;
                       background: #333; border: 1px solid #444; color: #ccc;
                       cursor: pointer; font-size: 12px; border-radius: 3px; }
  .frame-strip button.active { background: #557; border-color: #88a; color: white; }
  .frame-strip button.ref { border-color: #cc0; }
  section.viewer { flex: 1; display: flex; align-items: center;
                   justify-content: center; padding: 18px; overflow: hidden; }
  .stage { position: relative; max-width: 100%; max-height: calc(100vh - 180px); }
  .stage img.base, .stage img.mask { display: block; max-width: 100%;
                                      max-height: calc(100vh - 180px); }
  .stage img.mask { position: absolute; top: 0; left: 0; pointer-events: none; }
  .grid-mode .stage { display: grid; grid-template-columns: repeat(3, 1fr);
                      gap: 8px; max-width: none; }
  .grid-mode .tile { position: relative; }
  .grid-mode .tile img.base, .grid-mode .tile img.mask {
                      width: 100%; height: auto; max-height: none; }
  .grid-mode .tile img.mask { position: absolute; top: 0; left: 0;
                               width: 100%; height: 100%; pointer-events: none; }
  .grid-mode .tile .label { position: absolute; top: 4px; left: 4px;
                             background: rgba(0,0,0,0.6); padding: 2px 6px;
                             font-size: 11px; border-radius: 2px; }
</style>
</head>
<body>
<header>
  <h1>SAM 3.1 mask viewer</h1>
  <span class="meta">prompt: <code id="prompt">/*PROMPT*/</code></span>
  <span class="meta" id="meta-line"></span>
  <label style="margin-left:auto"><input type="checkbox" id="grid"> Grid view</label>
</header>
<main id="main">
  <aside>
    <div class="group">
      <h2>Layers</h2>
      <label><input type="checkbox" id="show-base" checked> Base photo</label>
      <div id="object-toggles"></div>
    </div>
    <div class="group">
      <h2>Mask opacity</h2>
      <input type="range" id="opacity" min="0" max="100" value="55">
      <div style="font-size:11px;color:#888" id="opacity-label">55%</div>
    </div>
    <div class="group">
      <h2>Reference tap</h2>
      <div style="font-size:12px;color:#bbb;line-height:1.4">
        Click two points on the photo: <b>1st click = top end</b>,
        <b>2nd click = bottom end</b> of the reference object.
        Repeat on a 2nd or 3rd frame for triangulation. Then copy the
        scale_solver command below.
      </div>
      <div style="margin-top:8px;font-size:12px">
        Length (m): <input type="number" id="ref-length-m"
          value="2.44" step="0.01" min="0.01"
          style="width:80px;background:#333;color:#fff;border:1px solid #555;padding:2px 4px">
      </div>
      <div id="ref-picks" style="margin-top:8px;font-size:11px;color:#bbb;
                                  font-family:monospace;line-height:1.4"></div>
      <button id="ref-clear" style="margin-top:6px;padding:4px 8px;
              background:#444;color:#ddd;border:1px solid #555;
              cursor:pointer;font-size:11px">Clear picks</button>
      <textarea id="ref-cmd" readonly rows="6"
        style="margin-top:8px;width:100%;background:#0d0d0d;color:#9c9;
               border:1px solid #555;padding:6px;font-family:monospace;
               font-size:10px;resize:vertical;display:none"></textarea>
    </div>
  </aside>
  <section class="viewer" id="viewer">
    <div class="stage" id="stage"></div>
  </section>
</main>
<div class="frame-strip" id="strip"></div>

<script>
const data = /*PAYLOAD*/;
let curFrame = data.ref_frame;
let visibleObjects = data.objects.map(() => true);
let showBase = true;
let opacity = 0.55;
let gridMode = false;

const stage = document.getElementById('stage');
const strip = document.getElementById('strip');
const main = document.getElementById('main');
const meta = document.getElementById('meta-line');
meta.textContent =
  `${data.S} frames • ${data.n_obj} object(s)` +
  (data.gps_block ? ` • ${data.gps_block}` : '');

function buildObjectToggles() {
  const root = document.getElementById('object-toggles');
  root.innerHTML = '';
  data.objects.forEach((o, i) => {
    const lbl = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = visibleObjects[i];
    cb.onchange = () => { visibleObjects[i] = cb.checked; render(); };
    const sw = document.createElement('span');
    sw.className = 'swatch';
    sw.style.background = o.css_color;
    const txt = document.createElement('span');
    txt.textContent = `obj ${o.k} (id=${o.id})`;
    const area = document.createElement('span');
    area.className = 'area';
    area.id = `area-${i}`;
    lbl.appendChild(cb); lbl.appendChild(sw); lbl.appendChild(txt); lbl.appendChild(area);
    root.appendChild(lbl);
  });
}

function buildStrip() {
  strip.innerHTML = '';
  data.frames.forEach((fr, i) => {
    const b = document.createElement('button');
    b.textContent = `f${i} ${fr.name.slice(-12)}`;
    if (i === data.ref_frame) b.classList.add('ref');
    if (i === curFrame) b.classList.add('active');
    b.onclick = () => { curFrame = i; render(); };
    strip.appendChild(b);
  });
}

function buildSingleFrame() {
  const fr = data.frames[curFrame];
  stage.innerHTML = '';
  if (showBase) {
    const img = document.createElement('img');
    img.src = fr.image; img.className = 'base';
    stage.appendChild(img);
  }
  data.objects.forEach((o, i) => {
    if (!visibleObjects[i]) return;
    const m = document.createElement('img');
    m.src = fr.masks[i]; m.className = 'mask';
    m.style.opacity = opacity;
    stage.appendChild(m);
  });
  // update areas
  data.objects.forEach((o, i) => {
    document.getElementById(`area-${i}`).textContent =
      fr.areas[i] > 0 ? `${(fr.areas[i] / 1000).toFixed(1)}k` : '—';
  });
}

function buildGrid() {
  stage.innerHTML = '';
  data.frames.forEach((fr, i) => {
    const tile = document.createElement('div');
    tile.className = 'tile';
    if (showBase) {
      const img = document.createElement('img');
      img.src = fr.image; img.className = 'base';
      tile.appendChild(img);
    }
    data.objects.forEach((o, k) => {
      if (!visibleObjects[k]) return;
      const m = document.createElement('img');
      m.src = fr.masks[k]; m.className = 'mask';
      m.style.opacity = opacity;
      tile.appendChild(m);
    });
    const lab = document.createElement('div');
    lab.className = 'label';
    lab.textContent = `f${i}` + (i === data.ref_frame ? ' REF' : '');
    tile.appendChild(lab);
    stage.appendChild(tile);
  });
}

function render() {
  // refresh strip active class
  Array.from(strip.children).forEach((b, i) => {
    b.classList.toggle('active', i === curFrame);
  });
  if (gridMode) buildGrid(); else buildSingleFrame();
}

document.getElementById('grid').onchange = (e) => {
  gridMode = e.target.checked;
  main.classList.toggle('grid-mode', gridMode);
  render();
};
document.getElementById('show-base').onchange = (e) => {
  showBase = e.target.checked; render();
};
document.getElementById('opacity').oninput = (e) => {
  opacity = e.target.value / 100;
  document.getElementById('opacity-label').textContent = `${e.target.value}%`;
  render();
};
window.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowRight') {
    curFrame = (curFrame + 1) % data.S; render();
  } else if (e.key === 'ArrowLeft') {
    curFrame = (curFrame - 1 + data.S) % data.S; render();
  }
});

// ---- Reference-tap pick accumulator ----
// Per frame: collected picks in NATIVE pixel coords. After two picks
// per frame, the frame is "complete". Picks across >=2 frames yield
// a runnable scale_solver command.
const refPicks = {};   // frame_idx -> [[u,v], [u,v]]
function refRender() {
  const lines = [];
  Object.keys(refPicks).sort((a, b) => +a - +b).forEach(f => {
    const p = refPicks[f];
    const tag = p.length === 2 ? '' : ' (need 2nd click)';
    lines.push(`f${f}: ${p.map(xy =>
      `(${Math.round(xy[0])},${Math.round(xy[1])})`).join(' → ')}${tag}`);
  });
  document.getElementById('ref-picks').innerHTML =
    lines.length ? lines.join('<br>') : '<i>no picks yet</i>';

  // Build CLI if at least 2 frames have 2 picks each.
  const complete = Object.entries(refPicks)
    .filter(([_, p]) => p.length === 2)
    .map(([f, p]) =>
      `${f},${Math.round(p[0][0])},${Math.round(p[0][1])},` +
      `${Math.round(p[1][0])},${Math.round(p[1][1])}`);
  const cmdBox = document.getElementById('ref-cmd');
  if (complete.length >= 2) {
    const lengthM = document.getElementById('ref-length-m').value || '2.44';
    cmdBox.value =
      `python scripts/scale_solver.py \\\n` +
      `    --gps pole_001.gps.json \\\n` +
      `    --poses pole_001.poses.npz \\\n` +
      `    --ref-pixels "${complete.join(';')}" \\\n` +
      `    --ref-length-m ${lengthM} \\\n` +
      `    --out pole_001.scale.json`;
    cmdBox.style.display = 'block';
  } else {
    cmdBox.style.display = 'none';
  }
}
function refOnImageClick(e) {
  if (gridMode) return;
  const baseImg = stage.querySelector('img.base');
  if (!baseImg) return;
  const rect = baseImg.getBoundingClientRect();
  const dispX = e.clientX - rect.left;
  const dispY = e.clientY - rect.top;
  // Rescale to native mask coords.
  const fr = data.frames[curFrame];
  const u = dispX * (data.native_w / fr.w);
  const v = dispY * (data.native_h / fr.h);
  if (!refPicks[curFrame]) refPicks[curFrame] = [];
  if (refPicks[curFrame].length >= 2) refPicks[curFrame] = [];
  refPicks[curFrame].push([u, v]);
  refRender();
  // Visual marker (transient circle).
  const ring = document.createElement('div');
  const r = 8;
  ring.style.cssText =
    `position:absolute;left:${dispX - r}px;top:${dispY - r}px;` +
    `width:${2*r}px;height:${2*r}px;border:2px solid yellow;` +
    `border-radius:50%;pointer-events:none;` +
    `box-shadow:0 0 4px black,inset 0 0 4px black`;
  stage.appendChild(ring);
}
stage.addEventListener('click', refOnImageClick);
document.getElementById('ref-clear').onclick = () => {
  Object.keys(refPicks).forEach(k => delete refPicks[k]);
  refRender();
  render();   // wipes any visible markers
};
document.getElementById('ref-length-m').addEventListener('input', refRender);
refRender();

buildObjectToggles();
buildStrip();
render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
