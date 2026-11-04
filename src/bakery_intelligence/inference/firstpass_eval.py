"""Offline attribution-accuracy evaluation for batch FirstPass.

Compares predicted scale readings against a verified
reference table at positional sequence offsets. Pure
functions only; no network access and no model calls.

Only reference rows marked ``readable`` with an exact
``reading`` enter the accuracy denominator. ``unreadable``
rows (verified no legible display) and ``unverified`` rows
are excluded from accuracy but still surface fabrication
signals when a prediction claims a value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ReferenceRow:
    """One verified reference entry in sequence order."""

    frame_index: int
    status: str  # "readable" | "unreadable" | "unverified"
    reading: str | None = None


@dataclass(frozen=True)
class Prediction:
    """One predicted reading in the same sequence order."""

    reading: str | None
    confidence: str = "low"


@dataclass(frozen=True)
class OffsetScore:
    """Accuracy of predictions shifted against the reference."""

    offset: int
    compared: int
    exact: int
    missing: int
    wrong: int
    unreadable_with_value: int
    high_conf_wrong: int
    high_conf_fabricated: int
    high_conf_missing: int

    @property
    def false_confident(self) -> int:
        """High-confidence errors of every category."""
        return (self.high_conf_wrong
                + self.high_conf_fabricated
                + self.high_conf_missing)


def score_offset(predictions: Sequence[Prediction],
                 references: Sequence[ReferenceRow],
                 offset: int) -> OffsetScore:
    """Score predictions at one positional offset.

    Prediction at sequence position ``i`` is compared with
    the reference at ``i + offset``. A negative-offset peak
    means predictions lag (each label carries an earlier
    image's content); a positive-offset peak means they
    lead. A zero-offset peak with no rival is correct
    attribution.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references "
            f"({len(references)}) must align by position.")
    compared = exact = missing = wrong = 0
    unreadable_with_value = 0
    high_conf_wrong = high_conf_fabricated = 0
    high_conf_missing = 0
    for position, prediction in enumerate(predictions):
        other = position + offset
        if other < 0 or other >= len(references):
            continue
        reference = references[other]
        if reference.status == "readable":
            if reference.reading is None:
                raise ValueError(
                    f"readable reference at position {other} "
                    f"has no reading.")
            compared += 1
            if prediction.reading is None:
                missing += 1
                if prediction.confidence == "high":
                    high_conf_missing += 1
            elif prediction.reading == reference.reading:
                exact += 1
            else:
                wrong += 1
                if prediction.confidence == "high":
                    high_conf_wrong += 1
        elif (reference.status == "unreadable"
                and prediction.reading is not None):
            unreadable_with_value += 1
            if prediction.confidence == "high":
                high_conf_fabricated += 1
    return OffsetScore(
        offset=offset, compared=compared, exact=exact,
        missing=missing, wrong=wrong,
        unreadable_with_value=unreadable_with_value,
        high_conf_wrong=high_conf_wrong,
        high_conf_fabricated=high_conf_fabricated,
        high_conf_missing=high_conf_missing)


def score_offsets(
        predictions: Sequence[Prediction],
        references: Sequence[ReferenceRow],
        offsets: Sequence[int] = (-2, -1, 0, 1, 2)
        ) -> dict[int, OffsetScore]:
    """Score several positional offsets in one pass."""
    return {offset: score_offset(predictions, references,
                                 offset)
            for offset in offsets}
