"""Tests for Stanford Cars → VehicleTypeNet body-type derivation (Goal 4.3)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "deploy" / "evaluation"))

from evaluate import derive_typenet_label  # noqa: E402


@pytest.mark.parametrize(
    "stanford_class,expected",
    [
        ("Acura RL Sedan 2012", "sedan"),
        ("Audi A5 Coupe 2012", "coupe"),
        ("BMW M3 Coupe 2012", "coupe"),
        ("AM General Hummer SUV 2000", "suv"),
        ("Chevrolet Tahoe Hybrid SUV 2012", "suv"),
        ("Ford F-150 Regular Cab 2007", "truck"),
        ("Ford F-150 Crew Cab 2012", "truck"),
        ("GMC Savana Cargo Van 2012", "van"),
        ("Dodge Caravan Minivan 1997", "van"),
        ("Volkswagen Golf Hatchback 2012", "sedan"),
        ("Audi A6 Wagon 2012", "sedan"),
        ("BMW Z4 Convertible 2012", "coupe"),
        # Unknown body — empty string, caller must skip
        ("Spyker C8 Coupe 2009", "coupe"),
        ("Eagle Talon Hatchback 1998", "sedan"),
    ],
)
def test_known_stanford_classes(stanford_class, expected):
    assert derive_typenet_label(stanford_class) == expected


def test_no_body_keyword_returns_empty():
    assert derive_typenet_label("Foobar Mystery 2099") == ""
    assert derive_typenet_label("") == ""


def test_case_insensitive():
    assert derive_typenet_label("BMW M3 COUPE 2012") == "coupe"
    assert derive_typenet_label("bmw m3 coupe 2012") == "coupe"


def test_priority_long_keyword_first():
    # "Crew Cab" must NOT be misread as plain "cab" (both → truck here, but
    # priority ordering still matters for future ambiguous cases).
    assert derive_typenet_label("Ford F-150 Crew Cab 2012") == "truck"
    # Minivan must beat plain "van"
    assert derive_typenet_label("Dodge Grand Caravan Minivan 2007") == "van"
