"""Attribution-offset evaluation regression tests.

Offline coverage for positional reading accuracy, shift
detection, unverified-row exclusion, fabrication signals,
and high-confidence error accounting.
"""

from __future__ import annotations

import pytest

from bakery_intelligence.inference.firstpass_eval import (
    Prediction,
    ReferenceRow,
    score_offset,
    score_offsets,
)


def _ref(index: int, status: str = "readable",
         reading: str | None = None) -> ReferenceRow:
    return ReferenceRow(frame_index=index, status=status,
                        reading=reading)


def test_zero_offset_perfect_attribution() -> None:
    refs = [_ref(0, reading="11"), _ref(30, reading="22"),
            _ref(60, reading="29")]
    preds = [Prediction("11", "high"), Prediction("22", "high"),
             Prediction("29", "medium")]
    score = score_offset(preds, refs, 0)
    assert score.compared == 3
    assert score.exact == 3
    assert score.missing == 0 and score.wrong == 0
    assert score.false_confident == 0


def test_negative_offset_peak_detects_lag() -> None:
    refs = [_ref(0, reading="3"), _ref(30, reading="11"),
            _ref(60, reading="22"), _ref(90, reading="29")]
    lagged = [Prediction("3", "medium"),
              Prediction("3", "medium"),
              Prediction("11", "medium"),
              Prediction("22", "medium")]
    scores = score_offsets(lagged, refs)
    assert scores[0].exact == 1
    assert scores[-1].exact == 3
    assert scores[1].exact <= 1


def test_positive_offset_peak_detects_lead() -> None:
    refs = [_ref(0, reading="3"), _ref(30, reading="11"),
            _ref(60, reading="22")]
    leading = [Prediction("11", "medium"),
               Prediction("22", "medium"),
               Prediction("99", "medium")]
    scores = score_offsets(leading, refs)
    assert scores[1].exact == 2
    assert scores[0].exact == 0


def test_unverified_and_unreadable_excluded() -> None:
    refs = [_ref(0, reading="11"), _ref(30, "unverified"),
            _ref(60, "unreadable"), _ref(90, reading="22")]
    preds = [Prediction("11", "high"),
             Prediction("999", "low"),
             Prediction(None, "low"),
             Prediction(None, "low")]
    score = score_offset(preds, refs, 0)
    assert score.compared == 2
    assert score.exact == 1
    assert score.missing == 1
    assert score.unreadable_with_value == 0


def test_fabrication_and_false_confidence() -> None:
    refs = [_ref(0, "unreadable"), _ref(30, reading="11"),
            _ref(60, reading="22")]
    preds = [Prediction("45", "high"),
             Prediction("12", "high"),
             Prediction(None, "high")]
    score = score_offset(preds, refs, 0)
    assert score.compared == 2
    assert score.unreadable_with_value == 1
    assert score.high_conf_fabricated == 1
    assert score.high_conf_wrong == 1
    assert score.high_conf_missing == 1
    assert score.false_confident == 3


def test_length_mismatch_and_bad_reference() -> None:
    refs = [_ref(0, reading="11")]
    with pytest.raises(ValueError):
        score_offset([Prediction("11")], refs + refs, 0)
    with pytest.raises(ValueError):
        score_offset([Prediction("11")],
                     [_ref(0, "readable", None)], 0)
