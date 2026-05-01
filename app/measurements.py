"""Measurement extraction from on-disk pipeline artifacts.

We don't currently persist measurements to the DB — they're derived
on demand from `{name}.triangulation.json` and `{name}.scale.json`.
This keeps the source of truth in the artifact files (which are
what `fit_pole_axis.py` already produces) and avoids stale rows
when artifacts are regenerated.
"""

from __future__ import annotations

import json
from pathlib import Path

from app import captures


def measurements_for(capture_name: str) -> list[dict]:
    """Return a list of `{kind, value, sigma, unit, source}` records
    for `capture_name` based on whichever artifacts exist."""
    ws = captures.workspace_path()
    out: list[dict] = []

    triang_path = ws / f"{capture_name}.triangulation.json"
    if triang_path.exists():
        try:
            t = json.loads(triang_path.read_text())
        except json.JSONDecodeError:
            t = {}
        ci = t.get("ci") or {}
        h_ci = ci.get("height_m_ci") or ci.get("height_units_ci")
        height_sigma = None
        if h_ci and len(h_ci) == 2:
            # Treat half-width of the 90% CI as a 1σ-ish indicator. For
            # a normally distributed quantity this slightly under-states
            # σ but is the right feel for the dashboard.
            height_sigma = (float(h_ci[1]) - float(h_ci[0])) / 2.0
        if t.get("height_m") is not None:
            out.append({
                "kind": "height_m",
                "value": float(t["height_m"]),
                "sigma": height_sigma,
                "unit": "m",
                "source": "fit / triangulation.json",
                "info": {
                    "frames_used": t.get("frames_used"),
                    "method": t.get("method"),
                    "object_id": t.get("object_id"),
                    "ci_90": h_ci,
                    "ci_n_iter": ci.get("n_iter"),
                },
            })
        lean_ci = ci.get("lean_deg_ci")
        lean_sigma = ((lean_ci[1] - lean_ci[0]) / 2.0
                      if lean_ci and len(lean_ci) == 2 else None)
        if t.get("axis_lean_deg") is not None:
            out.append({
                "kind": "lean_deg",
                "value": float(t["axis_lean_deg"]),
                "sigma": lean_sigma,
                "unit": "deg",
                "source": "lean vs ground-plane normal",
                "info": {"ci_90": lean_ci},
            })
        if t.get("axis_lean_cam_up_deg") is not None:
            out.append({
                "kind": "lean_cam_up_deg",
                "value": float(t["axis_lean_cam_up_deg"]),
                "sigma": lean_sigma,
                "unit": "deg",
                "source": "lean vs mean camera-up",
            })

    scale_path = ws / f"{capture_name}.scale.json"
    if scale_path.exists():
        try:
            s = json.loads(scale_path.read_text())
        except json.JSONDecodeError:
            s = {}
        if "scale" in s:
            out.append({
                "kind": "metric_scale",
                "value": float(s["scale"]),
                "sigma": float(s["sigma"]) if s.get("sigma") is not None else None,
                "unit": "m/model_unit",
                "source": s.get("method", "scale_solver"),
            })

    return out
