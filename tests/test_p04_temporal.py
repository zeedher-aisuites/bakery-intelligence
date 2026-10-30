"""Offline tests for deterministic temporal analysis (Exp. 006).

All fixtures are synthetic and tmp-local; no API credentials, no
network, no real P04 data. Oracles are the documented grouping,
parsing, and association rules in temporal.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "006_p04_multimodal_inference"
sys.path.insert(0, str(EXPERIMENT_DIR))

import temporal  # noqa: E402

MANIFEST_SHA = "abc123"


def _obs(frame_index: int, verbatim: str | None = "0",
         readable: bool = True, action: str = "none",
         tare: bool = False, confidence: str = "high",
         state: str = "stable") -> dict[str, Any]:
    stamp = frame_index / 50.0
    return {
        "frame_index": frame_index,
        "image_path": f"frame_{frame_index:08d}.jpg",
        "timestamp_seconds": stamp,
        "source_timestamp_seconds": stamp,
        "requested_timestamp_seconds": stamp,
        "timestamp_basis": "frame_index_over_fps",
        "manifest_ref": {"manifest_dir": "synthetic",
                         "manifest_sha256": MANIFEST_SHA},
        "batch_id": "batch_test",
        "pass_id": "baseline_1fps",
        "model_fields": {
            "scale_display_verbatim": verbatim,
            "scale_readable": readable,
            "scale_state": state,
            "tare_action_visible": tare,
            "addition_or_removal_visible": action,
            "container_description": None,
            "container_changed_hint": False,
            "confidence": confidence,
        },
        "parse_error": None,
    }


def _write_obs(obs_dir: Path, records: list[dict[str, Any]]) -> None:
    obs_dir.mkdir(parents=True, exist_ok=True)
    (obs_dir / "batch_test.json").write_text(
        json.dumps(records), encoding="utf-8"
    )


def test_consecutive_grouping_forms_one_interval(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [_obs(0), _obs(50), _obs(100)])
    artifact = temporal.analyze(obs_dir, tmp_path / "out.json")
    visibility = [event for event in artifact["events"]
                  if event["candidate_type"] == "scale_visibility"]
    assert len(visibility) == 1
    assert visibility[0]["frame_indices"] == [0, 50, 100]
    assert visibility[0]["verification_status"] == "candidate"


def test_temporal_gap_splits_intervals(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [_obs(0), _obs(50), _obs(250)])
    artifact = temporal.analyze(obs_dir, tmp_path / "out.json")
    assert artifact["gaps"] == [{
        "after_frame_index": 50, "before_frame_index": 250,
        "gap_seconds": 4.0,
    }]
    visibility = [event for event in artifact["events"]
                  if event["candidate_type"] == "scale_visibility"]
    assert [event["frame_indices"] for event in visibility] == [
        [0, 50], [250]]


def test_repeated_readings_form_stable_event(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [_obs(index, verbatim="120")
                         for index in (0, 50, 100, 150)])
    artifact = temporal.analyze(obs_dir, tmp_path / "out.json")
    stable = [event for event in artifact["events"]
              if event["candidate_type"] == "stable_reading"]
    assert len(stable) == 1
    assert stable[0]["parsed_values"] == [
        {"value": 120.0, "status": "ok", "note": None}] * 4
    assert stable[0]["consistency"] == {
        "run_length": 4, "distinct_values": 1, "value": 120.0,
        "enclosing_intervals": [{
            "event_id": "scale_visibility_00000000_00000150",
            "verification_status": "candidate",
        }],
    }
    assert stable[0]["verification_status"] == "candidate"
    assert stable[0]["review_reasons"] == []


def test_short_run_forms_no_stable_event(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [_obs(0, verbatim="120"), _obs(50, verbatim="120")])
    artifact = temporal.analyze(obs_dir, tmp_path / "out.json")
    kinds = [event["candidate_type"] for event in artifact["events"]]
    assert "stable_reading" not in kinds
    assert "scale_visibility" in kinds


def test_abrupt_change_flagged(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [
        _obs(0, verbatim="100"), _obs(50, verbatim="500"),
        _obs(100, verbatim="500"), _obs(150, verbatim="500"),
    ])
    artifact = temporal.analyze(obs_dir, tmp_path / "out.json")
    visibility = [event for event in artifact["events"]
                  if event["candidate_type"] == "scale_visibility"][0]
    assert visibility["verification_status"] == "needs_review"
    assert visibility["consistency"]["max_step"] == 400.0
    assert any("abrupt" in note for note in visibility["uncertainty_notes"])


def test_spike_detected_as_contradiction(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [
        _obs(0, verbatim="100"), _obs(50, verbatim="9000"),
        _obs(100, verbatim="100"),
    ])
    artifact = temporal.analyze(obs_dir, tmp_path / "out.json")
    visibility = [event for event in artifact["events"]
                  if event["candidate_type"] == "scale_visibility"][0]
    assert visibility["consistency"]["spikes"] == [50]
    assert visibility["contradicting_evidence"] == ["frame 50: spike"]
    assert visibility["verification_status"] == "needs_review"


def test_malformed_reading_flagged(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [_obs(0, verbatim="C00"), _obs(50, verbatim="120")])
    artifact = temporal.analyze(obs_dir, tmp_path / "out.json")
    visibility = [event for event in artifact["events"]
                  if event["candidate_type"] == "scale_visibility"][0]
    assert visibility["parsed_values"][0]["status"] == "unparseable"
    assert visibility["verification_status"] == "needs_review"


@pytest.mark.parametrize("raw, value, status, note", [
    ("100", 100.0, "ok", None),
    ("100.0", 100.0, "ok", None),
    ("- 750", -750.0, "normalized", "sign_space_removed"),
    ("- 39", -39.0, "normalized", "sign_space_removed"),
    ("5 15", None, "unparseable", "ambiguous_internal_spacing"),
    ("1 2.5", None, "unparseable", "ambiguous_internal_spacing"),
    (None, None, "null_input", None),
    ("", None, "unparseable", "empty_string"),
    ("C00", None, "unparseable", "non_numeric_characters"),
    ("Lo", None, "unparseable", "non_numeric_characters"),
])
def test_parse_reading_cases(raw: str | None, value: float | None,
                             status: str, note: str | None) -> None:
    reading = temporal.parse_reading(raw)
    assert reading.raw == raw
    assert reading.value == value
    assert reading.status == status
    assert reading.note == note


def test_addition_removal_grouped_once(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [
        _obs(0, action="addition"), _obs(50, action="addition"),
        _obs(100, action="none"), _obs(150, action="removal"),
    ])
    artifact = temporal.analyze(obs_dir, tmp_path / "out.json")
    additions = [event for event in artifact["events"]
                 if event["candidate_type"] == "action_addition"]
    removals = [event for event in artifact["events"]
                if event["candidate_type"] == "action_removal"]
    assert len(additions) == 1
    assert additions[0]["frame_indices"] == [0, 50]
    assert len(removals) == 1
    assert removals[0]["frame_indices"] == [150]


def test_tare_candidates_grouped(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [
        _obs(0), _obs(50, tare=True), _obs(100, tare=True), _obs(150),
    ])
    artifact = temporal.analyze(obs_dir, tmp_path / "out.json")
    tares = [event for event in artifact["events"]
             if event["candidate_type"] == "action_tare"]
    assert len(tares) == 1
    assert tares[0]["frame_indices"] == [50, 100]


def test_action_associated_with_nearby_scale(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [_obs(0, action="addition"), _obs(50)])
    artifact = temporal.analyze(obs_dir, tmp_path / "out.json")
    action = [event for event in artifact["events"]
              if event["candidate_type"] == "action_addition"][0]
    assert len(action["associated_scale_events"]) == 1
    assert action["verification_status"] == "candidate"


def test_unassociated_action_needs_review(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [
        _obs(0, action="addition", readable=False, verbatim=None),
        _obs(50, readable=False, verbatim=None),
    ])
    artifact = temporal.analyze(obs_dir, tmp_path / "out.json")
    action = [event for event in artifact["events"]
              if event["candidate_type"] == "action_addition"][0]
    assert action["associated_scale_events"] == []
    assert action["verification_status"] == "needs_review"
    assert any("no nearby scale" in note
               for note in action["uncertainty_notes"])


def test_low_confidence_needs_review(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [_obs(0, confidence="low")])
    artifact = temporal.analyze(obs_dir, tmp_path / "out.json")
    visibility = [event for event in artifact["events"]
                  if event["candidate_type"] == "scale_visibility"][0]
    assert visibility["verification_status"] == "needs_review"


def test_deterministic_output_and_traceability(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    records = [_obs(0, verbatim="120"), _obs(50, verbatim="121")]
    _write_obs(obs_dir, records)
    first = temporal.analyze(obs_dir, tmp_path / "a.json")
    second = temporal.analyze(obs_dir, tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == \
        (tmp_path / "b.json").read_bytes()
    assert first == second
    known = {item["frame_index"] for item in records}
    for event in first["events"]:
        assert set(event["frame_indices"]) <= known
        for ref in event["observation_refs"]:
            assert ref["frame_index"] in known


def test_empty_input_yields_empty_artifact(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    obs_dir.mkdir()
    artifact = temporal.analyze(obs_dir, tmp_path / "out.json")
    assert artifact["events"] == []
    assert artifact["gaps"] == []
    assert artifact["summary"]["event_count"] == 0


def test_invalid_json_rejected(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    obs_dir.mkdir()
    (obs_dir / "batch_bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(temporal.TemporalError) as exc_info:
        temporal.analyze(obs_dir, tmp_path / "out.json")
    assert exc_info.value.code == "OBS_INVALID"


def test_missing_key_rejected(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    record = _obs(0)
    del record["model_fields"]
    _write_obs(obs_dir, [record])
    with pytest.raises(temporal.TemporalError) as exc_info:
        temporal.analyze(obs_dir, tmp_path / "out.json")
    assert exc_info.value.code == "OBS_INVALID"


def test_duplicate_index_rejected(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [_obs(0), _obs(0)])
    with pytest.raises(temporal.TemporalError) as exc_info:
        temporal.analyze(obs_dir, tmp_path / "out.json")
    assert exc_info.value.code == "OBS_INVALID"


def test_mixed_manifest_rejected(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    first, second = _obs(0), _obs(50)
    second["manifest_ref"]["manifest_sha256"] = "other"
    _write_obs(obs_dir, [first, second])
    with pytest.raises(temporal.TemporalError) as exc_info:
        temporal.analyze(obs_dir, tmp_path / "out.json")
    assert exc_info.value.code == "OBS_INVALID"


def test_visibility_review_reasons_are_machine_readable(
        tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [
        _obs(0, verbatim="100"), _obs(50, verbatim="C00"),
        _obs(100, verbatim="900", confidence="low"),
        _obs(150, verbatim="100"),
    ])
    artifact = temporal.analyze(obs_dir, tmp_path / "out.json")
    visibility = [event for event in artifact["events"]
                  if event["candidate_type"] == "scale_visibility"][0]
    assert visibility["verification_status"] == "needs_review"
    assert visibility["review_reasons"] == [
        "ambiguous_parsing", "low_confidence_observation",
        "temporal_inconsistency",
    ]
    assert any("unparseable display" in item
               for item in visibility["contradicting_evidence"])
    assert any("low-confidence observation" in item
               for item in visibility["contradicting_evidence"])


def test_candidate_events_carry_empty_review_reasons(
        tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [_obs(0), _obs(50), _obs(100)])
    artifact = temporal.analyze(obs_dir, tmp_path / "out.json")
    assert artifact["events"]
    for event in artifact["events"]:
        assert event["verification_status"] == "candidate"
        assert event["review_reasons"] == []


def test_unassociated_action_reason_is_insufficient_evidence(
        tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [
        _obs(0, action="addition", readable=False, verbatim=None),
        _obs(50, readable=False, verbatim=None),
    ])
    artifact = temporal.analyze(obs_dir, tmp_path / "out.json")
    action = [event for event in artifact["events"]
              if event["candidate_type"] == "action_addition"][0]
    assert action["verification_status"] == "needs_review"
    assert action["review_reasons"] == ["insufficient_evidence"]


def test_stable_run_with_abrupt_borders_needs_review(
        tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [
        _obs(0, verbatim="1358"), _obs(50, verbatim="1756"),
        _obs(100, verbatim="1756"), _obs(150, verbatim="1756"),
        _obs(200, verbatim="0"),
    ])
    artifact = temporal.analyze(obs_dir, tmp_path / "out.json")
    stable = [event for event in artifact["events"]
              if event["candidate_type"] == "stable_reading"][0]
    assert stable["verification_status"] == "needs_review"
    assert stable["review_reasons"] == ["temporal_inconsistency"]
    assert any("abrupt preceding step" in item
               for item in stable["contradicting_evidence"])
    assert any("abrupt following step" in item
               for item in stable["contradicting_evidence"])


def test_stable_run_with_clean_borders_records_enclosing_context(
        tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [
        _obs(0, verbatim="100"), _obs(50, verbatim="900"),
        _obs(100, verbatim="905"), _obs(150, verbatim="910"),
        _obs(200, verbatim="910"), _obs(250, verbatim="910"),
        _obs(300, verbatim="912"),
    ])
    artifact = temporal.analyze(obs_dir, tmp_path / "out.json")
    stable = [event for event in artifact["events"]
              if event["candidate_type"] == "stable_reading"][0]
    assert stable["frame_indices"] == [150, 200, 250]
    assert stable["verification_status"] == "candidate"
    assert stable["review_reasons"] == []
    assert stable["consistency"]["enclosing_intervals"] == [{
        "event_id": "scale_visibility_00000000_00000300",
        "verification_status": "needs_review",
    }]
    assert any("locally consistent" in note
               for note in stable["uncertainty_notes"])


def test_unparseable_border_flags_stable_run(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [
        _obs(0, verbatim="C00"), _obs(50, verbatim="69"),
        _obs(100, verbatim="69"), _obs(150, verbatim="69"),
    ])
    artifact = temporal.analyze(obs_dir, tmp_path / "out.json")
    stable = [event for event in artifact["events"]
              if event["candidate_type"] == "stable_reading"][0]
    assert stable["verification_status"] == "needs_review"
    assert stable["review_reasons"] == ["ambiguous_parsing"]
    assert any("unparseable preceding display" in item
               for item in stable["contradicting_evidence"])


def test_adjacent_distinct_values_not_merged(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [
        _obs(0, verbatim="69"), _obs(50, verbatim="69"),
        _obs(100, verbatim="69"), _obs(150, verbatim="68"),
        _obs(200, verbatim="69"), _obs(250, verbatim="69"),
        _obs(300, verbatim="69"),
    ])
    artifact = temporal.analyze(obs_dir, tmp_path / "out.json")
    stable = [event for event in artifact["events"]
              if event["candidate_type"] == "stable_reading"]
    assert len(stable) == 2
    assert [event["frame_indices"] for event in stable] == [
        [0, 50, 100], [200, 250, 300]]
    assert [event["consistency"]["value"] for event in stable] == [
        69.0, 69.0]


def test_equal_values_across_gap_not_merged(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [
        _obs(0, verbatim="69"), _obs(50, verbatim="69"),
        _obs(100, verbatim="69"), _obs(300, verbatim="69"),
        _obs(350, verbatim="69"), _obs(400, verbatim="69"),
    ])
    artifact = temporal.analyze(obs_dir, tmp_path / "out.json")
    assert len(artifact["gaps"]) == 1
    stable = [event for event in artifact["events"]
              if event["candidate_type"] == "stable_reading"]
    assert len(stable) == 2
    assert [event["frame_indices"] for event in stable] == [
        [0, 50, 100], [300, 350, 400]]


def test_stable_run_spans_brief_unreadable_frame(
        tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [
        _obs(0, verbatim="69"),
        _obs(1, readable=False, verbatim=None),
        _obs(2, verbatim="69"), _obs(3, verbatim="69"),
        _obs(4, verbatim="69"),
    ])
    artifact = temporal.analyze(obs_dir, tmp_path / "out.json")
    stable = [event for event in artifact["events"]
              if event["candidate_type"] == "stable_reading"]
    assert len(stable) == 1
    assert stable[0]["frame_indices"] == [0, 2, 3, 4]
    assert stable[0]["verification_status"] == "candidate"
    assert len(stable[0]["consistency"]["enclosing_intervals"]) == 1


def test_summary_counts_review_reasons(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [
        _obs(0, verbatim="100"), _obs(50, verbatim="900"),
        _obs(1000, action="addition", readable=False, verbatim=None),
    ])
    artifact = temporal.analyze(obs_dir, tmp_path / "out.json")
    assert artifact["summary"]["by_review_reason"] == {
        "temporal_inconsistency": 1, "insufficient_evidence": 1,
    }
