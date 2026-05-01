"""Pytest runner endpoint helper. Spawns pytest, parses its summary
line, returns structured counts for the dashboard's Run-tests modal.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Match summary lines like:
#   "============== 21 passed in 1.10s =============="
#   "==== 18 passed, 3 failed, 1 skipped, 2 errors in 1.40s ===="
_COUNT_RE = re.compile(r"(\d+)\s+(passed|failed|skipped|errors?|warnings?)")
_DURATION_RE = re.compile(r"in\s+([\d.]+)s")


def parse_pytest_summary(text: str) -> dict:
    """Extract counts + duration from the last pytest summary line in
    `text`. Returns zeros for any category not present."""
    out = {"passed": 0, "failed": 0, "skipped": 0,
           "errors": 0, "duration_s": 0.0}
    # Use the last non-empty line that looks like a summary banner.
    summary_line = ""
    for line in reversed(text.splitlines()):
        if "=" in line and re.search(r"\b(passed|failed|skipped|error)\b", line):
            summary_line = line
            break
    if not summary_line:
        # Fall back to scanning the whole text.
        summary_line = text
    for m in _COUNT_RE.finditer(summary_line):
        n = int(m.group(1))
        kind = m.group(2)
        if kind.startswith("error"):
            out["errors"] = n
        elif kind in out:
            out[kind] = n
    dm = _DURATION_RE.search(summary_line)
    if dm:
        out["duration_s"] = float(dm.group(1))
    return out


def _run_pytest_subprocess(args: list[str] | None = None) -> dict:
    """Run pytest as a subprocess; return parsed summary + raw output.

    Tests monkeypatch this function; production code calls it
    directly from the FastAPI endpoint."""
    cmd = [sys.executable, "-m", "pytest", "tests/", "--tb=short", "-q"]
    if args:
        cmd.extend(args)
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    output = proc.stdout + ("\n--- STDERR ---\n" + proc.stderr
                            if proc.stderr else "")
    parsed = parse_pytest_summary(output)
    parsed["exit_code"] = proc.returncode
    parsed["output"] = output
    return parsed


def run_tests(args: list[str] | None = None) -> dict:
    """Public entry called by the FastAPI endpoint."""
    return _run_pytest_subprocess(args)
