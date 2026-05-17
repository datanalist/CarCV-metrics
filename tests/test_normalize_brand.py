"""Regression tests for normalize_brand (Goal 4.1)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "deploy" / "evaluation"))

from evaluate import NGC_MAKES_LOWER, normalize_brand  # noqa: E402


@pytest.mark.parametrize("inp", ["", " ", "  ", "ab", "  ab "])
def test_empty_or_short_does_not_collapse_to_acura(inp):
    # Pre-fix bug: "" in "acura" was True, so every short/missing brand
    # routed to "acura". Must NOT match any NGC make now.
    out = normalize_brand(inp)
    assert out not in NGC_MAKES_LOWER, f"{inp!r} → {out!r} (should not match NGC)"


@pytest.mark.parametrize(
    "inp,expected",
    [
        ("audi", "audi"),
        ("Audi", "audi"),
        ("AUDI", "audi"),
        ("  bmw  ", "bmw"),
        ("toyota", "toyota"),
        ("volkswagen", "volkswagen"),
        ("vw", "vw"),  # not in NGC's 20, stays raw
    ],
)
def test_exact_and_normalized_matches(inp, expected):
    assert normalize_brand(inp) == expected


def test_substring_match_for_compound_names():
    # `mercedes-benz` should map to `mercedes` (NGC class)
    assert normalize_brand("mercedes-benz") == "mercedes"


def test_unknown_brand_returns_raw():
    # mad-cars has RU brands like `vaz`, `gaz`, `uaz` — must NOT collapse
    # to any NGC make (caller will filter as out-of-distribution).
    for ru in ["vaz", "gaz", "uaz", "lada"]:
        out = normalize_brand(ru)
        assert out not in NGC_MAKES_LOWER, f"{ru} → {out} (NGC false-positive)"
