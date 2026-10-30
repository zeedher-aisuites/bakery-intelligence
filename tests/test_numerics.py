"""Offline tests for safe numeric normalization.

Covers the Experiment 010 anomaly where display "515" matched while
numeric "515" (string) mismatched 515.0: type confusion must never
decide agreement. All fixtures synthetic.
"""

from __future__ import annotations

import math

import pytest

from bakery_intelligence.verification.numerics import (
    numbers_match,
    normalize_number,
)


@pytest.mark.parametrize("raw, value, status", [
    (515, 515.0, "ok"),
    (515.0, 515.0, "ok"),
    (-750, -750.0, "ok"),
    ("515", 515.0, "ok"),
    ("  515  ", 515.0, "ok"),
    ("-750.5", -750.5, "ok"),
    ("1,234", 1234.0, "ok"),
    ("-1,234,567.5", -1234567.5, "ok"),
    ("5 15", None, "ambiguous"),
    ("1 234", None, "ambiguous"),
    ("12,5", None, "ambiguous"),
    ("", None, "invalid"),
    ("C00", None, "invalid"),
    ("515g", None, "invalid"),
    ("1.234.567", None, "invalid"),
    (None, None, "invalid"),
    (True, None, "invalid"),
    (["515"], None, "invalid"),
])
def test_normalize_number_cases(raw: object, value: float | None,
                                status: str) -> None:
    result = normalize_number(raw)
    assert result.original == raw
    assert result.value == value
    assert result.status == status


def test_non_finite_rejected() -> None:
    assert normalize_number(math.inf).status == "invalid"
    assert normalize_number(math.nan).status == "invalid"


def test_thousands_note_recorded() -> None:
    result = normalize_number("1,234")
    assert result.note == "thousands_separator_removed"


def test_ambiguous_notes_recorded() -> None:
    assert normalize_number("5 15").note == "ambiguous_internal_spacing"
    assert normalize_number("12,5").note == "ambiguous_decimal_comma"


def test_string_515_matches_float_reference() -> None:
    """Regression test for the Experiment 010 numeric mismatch."""
    match, note = numbers_match("515", 515.0)
    assert match is True
    assert note == "equal_normalized_values"


@pytest.mark.parametrize("first, second, match", [
    (515, 515.0, True),
    ("515", 515, True),
    ("1,234", 1234.0, True),
    ("515", 514.0, False),
    ("5 15", 515.0, False),
    ("C00", 515.0, False),
    (None, 515.0, False),
    ("515", None, False),
])
def test_numbers_match_cases(first: object, second: object,
                             match: bool) -> None:
    result, note = numbers_match(first, second)
    assert result is match
    assert isinstance(note, str) and note


def test_mismatch_explains_distinct_values() -> None:
    result, note = numbers_match("515", 514.0)
    assert result is False
    assert "515.0" in note and "514.0" in note
