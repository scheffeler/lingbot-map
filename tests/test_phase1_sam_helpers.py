"""Phase 1.10 T7.1 — pure-Python helpers extracted from
phase1_sam_imageset.py so the prompt-parsing logic can be unit tested
without spinning up a Modal job."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from phase1_sam_imageset import parse_text_prompts  # noqa: E402


def test_parse_simple_csv():
    assert parse_text_prompts("utility pole, crossarm, wire") == [
        "utility pole", "crossarm", "wire",
    ]


def test_parse_strips_whitespace_and_drops_empties():
    assert parse_text_prompts("utility pole, , transformer ,") == [
        "utility pole", "transformer",
    ]


def test_parse_single_prompt():
    assert parse_text_prompts("utility pole") == ["utility pole"]


def test_parse_preserves_internal_spaces():
    """Multi-word phrases must survive intact; only the comma separators
    delimit prompts."""
    assert parse_text_prompts("electrical bracket, fire hydrant") == [
        "electrical bracket", "fire hydrant",
    ]


def test_parse_empty_string_raises():
    with pytest.raises(ValueError):
        parse_text_prompts("")


def test_parse_only_separators_raises():
    """A string of nothing-but-separators should also fail loud rather
    than silently running SAM with zero prompts."""
    with pytest.raises(ValueError):
        parse_text_prompts(",, ,")
