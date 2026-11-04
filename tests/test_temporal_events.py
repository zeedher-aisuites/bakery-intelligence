"""Deterministic tests for observation review, consolidation, events.

Synthetic fixtures only: no provider calls, no real video
validation. Human corrections win at their exact field and
temporal scope and never expand to adjacent frames.
"""

from __future__ import annotations

from typing import Any

from bakery_intelligence.adversarial.consolidation import consolidate
from bakery_intelligence.adversarial.observations import (
    build_review_tasks,
    evaluate_frames,
)
from bakery_intelligence.adversarial.temporal import reconstruct_events


def _frame(index: int, value: float | None, *, readable: bool = True,
           action: str = "none") -> dict[str, Any]:
    return {
        "frame_index": index,
        "timestamp_seconds": index / 50.0,
        "measurement": {
            "measurement_id": f"m_{index}", "value": value,
            "value_status": "ok" if value is not None else "invalid",
            "unit_observed": "g",
        },
        "scale": {
            "readable": readable,
            "addition_or_removal_visible": action,
            "tare_action_visible": False,
        },
    }


def _package(frames: list[dict[str, Any]], **overrides: Any,
             ) -> dict[str, Any]:
    package: dict[str, Any] = {
        "schema_version": "evidence-package/1",
        "run_id": "run_0123456789ab",
        "video": {"duration_seconds": 10.0},
        "frames": frames,
        "action_evidence": {"addition_frames": [], "tare_frames": []},
        "weighing_event": None,
        "unknowns": [],
    }
    package.update(overrides)
    return package


def test_accepted_hidden_and_irrelevant_abstentions_excluded() -> None:
    frames = [_frame(0, 0.0), _frame(50, 10.0),
              _frame(100, None, readable=False)]
    queue = build_review_tasks(_package(frames))
    assert queue["counts"]["accepted"] == 2
    assert queue["tasks"] == []
    states = {item["frame_index"]: item["state"]
              for item in evaluate_frames(_package(frames))}
    assert states[100] == "not_observable"


def test_relevant_missing_grouped_and_tare_splits() -> None:
    frames = [
        _frame(100, None, readable=False, action="addition"),
        _frame(150, None, readable=False, action="addition"),
        _frame(200, None, readable=False, action="addition"),
        _frame(250, None, readable=False, action="addition"),
    ]
    package = _package(
        frames,
        action_evidence={"addition_frames": [100, 150, 200, 250],
                         "tare_frames": [200]})
    queue = build_review_tasks(package)
    assert len(queue["tasks"]) == 2
    assert queue["tasks"][0]["frame_refs"] == [100, 150]


def test_human_correction_exact_scope_and_hash_stable() -> None:
    frames = [_frame(1000, 20.0), _frame(1050, 20.0)]
    package = _package(frames)
    corrections = {"task_x": {
        "task_id": "task_x", "field": "scale_reading",
        "action": "correct", "human_value": 21.0,
        "frame_refs": [1050], "revision": 1,
        "explicit_unknown": False,
    }}
    merged = consolidate(package, corrections)
    first, second = merged["frames"]
    assert "effective" not in first
    assert second["effective"]["value"] == 21.0
    assert second["effective"]["original_value"] == 20.0
    assert merged["content_hash"] == consolidate(
        package, corrections)["content_hash"]
    assert package["frames"][1]["measurement"]["value"] == 20.0


def test_valid_net_quantity_with_plateau_and_baseline() -> None:
    frames = [_frame(0, 0.0), _frame(50, 20.0), _frame(100, 21.0),
              _frame(150, 21.0)]
    events = reconstruct_events(consolidate(_package(frames), {}))
    addition = events["events"][0]
    assert addition["net_quantity"]["value"] == 21.0
    assert addition["net_quantity"]["unit"] == "g"
    assert addition["net_quantity"]["formula"] == "21.0 - 0.0"


def test_unsupported_net_quantity_without_unit_or_baseline() -> None:
    frame = _frame(0, 20.0)
    frame["measurement"]["unit_observed"] = None
    events = reconstruct_events(consolidate(_package([frame]), {}))
    net = events["events"][0]["net_quantity"]
    assert net["value"] is None and net["derivation"] == "unknown"
    assert "unit_compatible" in events["events"][0]["uncertainty"]


def test_tare_and_nonmonotonic_split_events() -> None:
    frames = [_frame(0, 0.0), _frame(50, 30.0), _frame(950, 0.0),
              _frame(1000, 10.0), _frame(1050, 5.0)]
    package = _package(
        frames,
        action_evidence={"addition_frames": [], "tare_frames": [950]})
    result = reconstruct_events(consolidate(package, {}))
    types = [event["event_type"] for event in result["events"]]
    assert "tare_or_reset" in types
    assert len(result["events"]) >= 3


def test_missing_frames_and_invalid_boundaries() -> None:
    events = reconstruct_events(consolidate(_package([]), {}))
    assert events["events"] == []
    frames = [_frame(100, 10.0), _frame(50, 5.0)]
    result = reconstruct_events(consolidate(_package(frames), {}))
    assert result["events"][0]["frame_refs"] == [50, 100]
