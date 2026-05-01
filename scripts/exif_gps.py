"""Phase 1.4: extract per-image EXIF GPS from a folder of stills.

Reads JPEG / PNG / HEIC, pulls GPSLatitude / GPSLongitude / GPSAltitude
plus DateTimeOriginal, and writes one JSON file describing each image's
GPS fix. Errors loudly if any image is missing GPS — partial GPS
coverage silently breaks the Sim(3) fit downstream, so we'd rather fail
fast than guess.

Run:
    python scripts/exif_gps.py --folder ./pole_001 \\
        --out pole_001.gps.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from PIL import ExifTags, Image

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass  # HEIC not supported in this env; JPEG/PNG still work.

IMG_EXTS = (".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff")

# EXIF tag IDs we care about, resolved from PIL's reverse mapping.
_GPS_TAG_ID = next(k for k, v in ExifTags.TAGS.items() if v == "GPSInfo")
_DT_TAG_ID = next(k for k, v in ExifTags.TAGS.items() if v == "DateTimeOriginal")
_GPS_NAMES = ExifTags.GPSTAGS  # id (int) -> name (str)


def _to_float(rational) -> float:
    """EXIF rationals come as IFDRational, tuple, or float depending on PIL
    version + image source. Normalize all of them to a Python float."""
    if isinstance(rational, tuple):
        num, den = rational
        return float(num) / float(den) if den else 0.0
    return float(rational)


def _dms_to_decimal(dms, ref: str) -> float:
    """Convert (deg, min, sec) tuple to signed decimal degrees."""
    d, m, s = (_to_float(x) for x in dms)
    val = d + m / 60.0 + s / 3600.0
    if ref in ("S", "W"):
        val = -val
    return val


def extract_gps(path: Path) -> dict | None:
    """Return {'lat', 'lon', 'alt_m', 'ts'} for one image, or None if
    the image has no GPS block at all. Raises ValueError if the GPS
    block is present but malformed."""
    img = Image.open(path)
    exif = img.getexif()
    gps_ifd = exif.get_ifd(_GPS_TAG_ID) if hasattr(exif, "get_ifd") else None
    if not gps_ifd:
        # Some PIL paths put GPSInfo directly under the top-level dict.
        raw_gps = exif.get(_GPS_TAG_ID)
        if not raw_gps:
            return None
        gps_ifd = raw_gps

    gps = {_GPS_NAMES.get(k, k): v for k, v in gps_ifd.items()}

    if "GPSLatitude" not in gps or "GPSLongitude" not in gps:
        return None
    lat = _dms_to_decimal(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
    lon = _dms_to_decimal(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
    alt_m = None
    if "GPSAltitude" in gps:
        alt_m = _to_float(gps["GPSAltitude"])
        # Ref 1 = below sea level; flip sign if so.
        if gps.get("GPSAltitudeRef") in (1, b"\x01"):
            alt_m = -alt_m

    ts = exif.get(_DT_TAG_ID)
    if ts:
        try:
            ts = datetime.strptime(ts, "%Y:%m:%d %H:%M:%S").isoformat()
        except (ValueError, TypeError):
            ts = str(ts)

    return {"lat": lat, "lon": lon, "alt_m": alt_m, "ts": ts}


def main():
    ap = argparse.ArgumentParser(description="Extract per-image EXIF GPS to JSON")
    ap.add_argument("--folder", required=True, help="Folder of images")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        sys.exit(f"Not a directory: {folder}")

    paths = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )
    if not paths:
        sys.exit(f"No supported images in {folder}")

    entries = []
    missing = []
    for p in paths:
        gps = extract_gps(p)
        if gps is None:
            missing.append(p.name)
            continue
        entries.append({"path": p.name, **gps})

    if missing:
        msg = (f"GPS missing on {len(missing)}/{len(paths)} image(s): "
               f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
        sys.exit(msg + "\nFix: re-shoot with Location enabled, or remove "
                 "these images from the capture folder.")

    # Quick stats: baseline = max pairwise distance in metres (small-angle).
    R_EARTH = 6_378_137.0
    import math
    lats = [e["lat"] for e in entries]
    lons = [e["lon"] for e in entries]
    lat0 = sum(lats) / len(lats)
    coslat = math.cos(math.radians(lat0))
    xs = [(lon - lons[0]) * coslat * math.radians(1) * R_EARTH for lon in lons]
    ys = [(lat - lats[0]) * math.radians(1) * R_EARTH for lat in lats]
    max_d = 0.0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            d = ((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2) ** 0.5
            if d > max_d:
                max_d = d

    out = {
        "folder": str(folder),
        "n_images": len(entries),
        "max_pairwise_baseline_m": round(max_d, 2),
        "centroid_latlon": [lat0, sum(lons) / len(lons)],
        "images": entries,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(f"  {len(entries)} image(s), baseline {max_d:.1f} m, "
          f"centroid {lat0:.5f},{out['centroid_latlon'][1]:.5f}")
    if max_d < 3.0:
        print("WARNING: baseline < 3 m. Sim(3) scale fit will be dominated "
              "by GPS noise. Re-shoot with more lateral motion.")


if __name__ == "__main__":
    main()
