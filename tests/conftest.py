"""Pytest config: make `scripts/` importable as a package-ish path.

The polevision scripts live under `scripts/` (flat layout, no setup.py
for them) and import each other via `sys.path.insert` at runtime. For
tests we replicate that path setup once here so test files can do
`from scripts.fit_pole_axis import ...` without per-test boilerplate.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts")
for p in (_REPO_ROOT, _SCRIPTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
