"""Safe numeric normalization for reported display values.

Compares model-reported numerics against references without type
confusion: JSON numbers, numeric strings, and strict thousands forms
normalize to floats, while ambiguous formats (internal spacing,
decimal commas) and non-numeric input never silently convert. The
original display text is always preserved by the caller; this module
only derives a comparable value plus an explicit status.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

STATUS_OK = "ok"
STATUS_INVALID = "invalid"
STATUS_AMBIGUOUS = "ambiguous"

_PLAIN = re.compile(r"^-?\d+(\.\d+)?$")
_THOUSANDS_COMMA = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
_DECIMAL_COMMA = re.compile(r"^-?\d+,\d+$")


@dataclass(frozen=True)
class NormalizedNumber:
    """A normalization outcome with its evidence trail."""

    original: Any
    value: float | None
    status: str  # ok | invalid | ambiguous
    note: str | None = None


def normalize_number(raw: Any) -> NormalizedNumber:
    """Normalize a reported value to float when unambiguous."""
    if isinstance(raw, bool):
        return NormalizedNumber(raw, None, STATUS_INVALID,
                                "boolean_not_numeric")
    if isinstance(raw, (int, float)):
        if not math.isfinite(raw):
            return NormalizedNumber(raw, None, STATUS_INVALID,
                                    "non_finite")
        return NormalizedNumber(raw, float(raw), STATUS_OK)
    if not isinstance(raw, str):
        if raw is None:
            return NormalizedNumber(raw, None, STATUS_INVALID,
                                    "missing_value")
        return NormalizedNumber(raw, None, STATUS_INVALID,
                                "unsupported_type")
    text = raw.strip()
    if not text:
        return NormalizedNumber(raw, None, STATUS_INVALID,
                                "empty_string")
    if _PLAIN.match(text):
        return NormalizedNumber(raw, float(text), STATUS_OK)
    if _THOUSANDS_COMMA.match(text):
        return NormalizedNumber(raw, float(text.replace(",", "")),
                                STATUS_OK, "thousands_separator_removed")
    compact = "".join(text.split())
    if compact != text and (
            _PLAIN.match(compact) or _THOUSANDS_COMMA.match(compact)):
        return NormalizedNumber(raw, None, STATUS_AMBIGUOUS,
                                "ambiguous_internal_spacing")
    if _DECIMAL_COMMA.match(text):
        return NormalizedNumber(raw, None, STATUS_AMBIGUOUS,
                                "ambiguous_decimal_comma")
    return NormalizedNumber(raw, None, STATUS_INVALID,
                            "non_numeric_characters")


def numbers_match(first: Any, second: Any) -> tuple[bool, str]:
    """Compare two reported values after safe normalization.

    Returns (match, explanation). Anything less than two unambiguous
    equal numerics is a non-match with an explicit reason; units are
    never inferred here.
    """
    left = normalize_number(first)
    right = normalize_number(second)
    if left.status != STATUS_OK or right.status != STATUS_OK:
        return False, (
            f"uncomparable: left={left.status}({left.note}) "
            f"right={right.status}({right.note})"
        )
    assert left.value is not None and right.value is not None
    if left.value == right.value:
        return True, "equal_normalized_values"
    return False, (
        f"distinct_values: {left.value} != {right.value}"
    )
