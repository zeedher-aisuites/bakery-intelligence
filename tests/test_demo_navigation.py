"""Offline tests for demo selection and navigation logic."""

from __future__ import annotations

from bakery_intelligence.app.navigation import (
    VIEW_DRAFT,
    VIEW_WORKSPACE,
    available_timestamps,
    default_timestamp,
    events_near,
    initial_state,
    nearest_timestamp,
    row_at_timestamp,
    validate_view,
    video_slot_index,
)

ROWS = [
    {"measurement_id": "m1", "timestamp_seconds": 142.0,
     "has_disagreement": False},
    {"measurement_id": "m2", "timestamp_seconds": 150.0,
     "has_disagreement": True},
    {"measurement_id": "m3", "timestamp_seconds": 153.0,
     "has_disagreement": False},
]

ACTIONS = [
    {"action_id": "a1", "start_seconds": 144.0, "end_seconds": 145.0},
    {"action_id": "a2", "start_seconds": 151.0, "end_seconds": 152.0},
]


def test_validate_view_accepts_known_views() -> None:
    assert validate_view(VIEW_DRAFT) == VIEW_DRAFT


def test_validate_view_defaults_unknown_values() -> None:
    assert validate_view("nope") == VIEW_WORKSPACE
    assert validate_view(None) == VIEW_WORKSPACE
    assert validate_view(42) == VIEW_WORKSPACE


def test_available_timestamps_sorted_numeric_only() -> None:
    rows = [*ROWS, {"timestamp_seconds": "bad"}, {}]
    assert available_timestamps(rows) == [142.0, 150.0, 153.0]


def test_default_timestamp_prefers_disagreement() -> None:
    assert default_timestamp(ROWS) == 150.0


def test_default_timestamp_falls_back_to_first() -> None:
    rows = [{**row, "has_disagreement": False} for row in ROWS]
    assert default_timestamp(rows) == 142.0
    assert default_timestamp([]) is None


def test_row_at_timestamp_exact_match() -> None:
    assert row_at_timestamp(ROWS, 150.0)["measurement_id"] == "m2"
    assert row_at_timestamp(ROWS, 149.0) is None
    assert row_at_timestamp(ROWS, "150") is None


def test_nearest_timestamp_snaps() -> None:
    assert nearest_timestamp(ROWS, 149.0) == 150.0
    assert nearest_timestamp(ROWS, 142.1) == 142.0
    assert nearest_timestamp([], 150.0) is None
    assert nearest_timestamp(ROWS, None) == 150.0


def test_events_near_covers_interval() -> None:
    assert [a["action_id"] for a in events_near(ACTIONS, 144.5)] == ["a1"]
    assert [a["action_id"] for a in events_near(ACTIONS, 151.0)] == ["a2"]
    assert events_near(ACTIONS, 148.0) == []
    assert events_near(ACTIONS, None) == []


def test_initial_state_selects_disagreement_row() -> None:
    state = initial_state(ROWS)
    assert state == {
        "view": VIEW_WORKSPACE,
        "timestamp": 150.0,
        "measurement_id": "m2",
        "seek_nonce": 0,
    }


def test_initial_state_empty_rows() -> None:
    state = initial_state([])
    assert state["view"] == VIEW_WORKSPACE
    assert state["timestamp"] is None
    assert state["measurement_id"] is None
    assert state["seek_nonce"] == 0


def test_video_slot_alternates_with_nonce() -> None:
    assert video_slot_index(0) == 0
    assert video_slot_index(1) == 1
    assert video_slot_index(2) == 0
    assert video_slot_index(3) == 1


def test_video_slot_rejects_invalid_inputs() -> None:
    assert video_slot_index(None) == 0
    assert video_slot_index("bad") == 0
    assert video_slot_index(-1) == 1
