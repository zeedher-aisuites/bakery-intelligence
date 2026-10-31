"""Offline tests for deterministic reconstruction (M2-04).

All fixtures are synthetic and tmp-local. Tests assert assembly,
unit sidecar semantics, draft validity, determinism, and input
preservation. No network, no inference, no historical data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from bakery_intelligence.reconstruction import cli as recon_cli
from bakery_intelligence.reconstruction import units as unit_state
from bakery_intelligence.reconstruction.measurements import (
    ReconstructionError,
    assemble_measurements,
    verified_by_timestamp,
)

FRAMES = (7400, 7450, 7500, 7550, 7600)
VERBATIMS = ("353", "421", "5 15", "575", "643")
EVENT_ID = "scale_visibility_test"


def _obs_record(frame: int, verbatim: str | None) -> dict[str, Any]:
    stamp = frame / 50.0
    return {
        "frame_index": frame,
        "image_path": f"frame_{frame:08d}.jpg",
        "timestamp_seconds": stamp,
        "batch_id": "batch_test",
        "model_fields": {
            "scale_display_verbatim": verbatim,
            "scale_readable": True,
            "scale_state": "stable",
            "tare_action_visible": False,
            "addition_or_removal_visible": "addition",
            "confidence": "high",
        },
    }


def _write_inputs(tmp_path: Path,
                  verbatims: tuple = VERBATIMS) -> dict[str, Path]:
    obs_dir = tmp_path / "obs"
    obs_dir.mkdir()
    records = [_obs_record(frame, verbatim)
               for frame, verbatim in zip(FRAMES, verbatims)]
    (obs_dir / "batch_test.json").write_text(
        json.dumps(records), encoding="utf-8")
    artifact = {
        "artifact_version": "1.1.0",
        "events": [
            {"event_id": EVENT_ID,
             "candidate_type": "scale_visibility",
             "start_timestamp_seconds": 148.0,
             "end_timestamp_seconds": 152.0,
             "verification_status": "needs_review",
             "frame_indices": list(FRAMES),
             "review_reasons": ["temporal_inconsistency"],
             "uncertainty_notes": []},
            {"event_id": "action_addition_test",
             "candidate_type": "action_addition",
             "start_timestamp_seconds": 148.0,
             "end_timestamp_seconds": 152.0,
             "verification_status": "candidate",
             "frame_indices": list(FRAMES),
             "review_reasons": [],
             "uncertainty_notes": []},
        ],
    }
    artifact_path = tmp_path / "temporal.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    exec_dir = tmp_path / "executions" / "run_test"
    exec_dir.mkdir(parents=True)
    readings = []
    for frame, display in zip(FRAMES, ("353", "421", "515", "575",
                                       "643")):
        readings.append({
            "timestamp_seconds": frame / 50.0,
            "display_text": display,
            "numeric_value": float(display),
            "readable": True,
            "confidence": "high",
            "uncertainty_notes": None,
        })
    (exec_dir / "execution_report.json").write_text(json.dumps({
        "event_id": EVENT_ID,
        "terminal_state": "valid_prediction",
        "selected_readings": readings,
    }), encoding="utf-8")
    return {"obs": obs_dir, "temporal": artifact_path,
            "exec": tmp_path / "executions"}


def _run_reconstruct(tmp_path: Path, paths: dict[str, Path],
                     extra: list[str] | None = None,
                     run_id: str = "run_01") -> dict[str, Any]:
    args = ["reconstruct", "--obs-dir", str(paths["obs"]),
            "--temporal", str(paths["temporal"]),
            "--event-id", EVENT_ID,
            "--out", str(tmp_path / "recon"), "--run-id", run_id,
            "--executions", str(paths["exec"])]
    if extra:
        args.extend(extra)
    code = recon_cli.main(args)
    assert code == 0
    report = tmp_path / "recon" / run_id / "recipe_draft.json"
    return json.loads(report.read_text(encoding="utf-8"))


def test_verified_preferred_original_preserved(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    _run_reconstruct(tmp_path, paths)
    bundle = json.loads((tmp_path / "recon" / "run_01"
                         / "measurements.json").read_text(
                             encoding="utf-8"))
    target = [item for item in bundle["measurements"]
              if item["frame_index"] == 7500][0]
    assert target["value_source"] == "verification"
    assert target["value"] == 515.0
    assert target["original"]["verbatim"] == "5 15"
    assert target["verified"]["display_text"] == "515"


def test_ambiguous_firstpass_without_verification(
        tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    args = ["reconstruct", "--obs-dir", str(paths["obs"]),
            "--temporal", str(paths["temporal"]),
            "--event-id", EVENT_ID,
            "--out", str(tmp_path / "recon"), "--run-id", "run_01"]
    assert recon_cli.main(args) == 0
    bundle = json.loads((tmp_path / "recon" / "run_01"
                         / "measurements.json").read_text(
                             encoding="utf-8"))
    target = [item for item in bundle["measurements"]
              if item["frame_index"] == 7500][0]
    assert target["value_source"] == "firstpass_unusable"
    assert target["value"] is None
    assert target["value_status"] == "ambiguous"


def test_human_grams_assignment(tmp_path: Path) -> None:
    sidecar = unit_state.new_sidecar()
    unit_state.assign_unit(
        sidecar, "g", "event_window", EVENT_ID, 142.0, 153.0,
        "reviewer-1")
    resolution = unit_state.resolve_unit(150.0, EVENT_ID, sidecar)
    assert resolution == {"unit": "g",
                          "provenance": "human_assigned",
                          "assignment_id": "ua_0001"}
    assert unit_state.resolve_unit(10.0, EVENT_ID, sidecar) == {
        "unit": None, "provenance": "undetermined",
        "assignment_id": None}


def test_session_scope_requires_justification(tmp_path: Path) -> None:
    sidecar = unit_state.new_sidecar()
    with pytest.raises(ReconstructionError) as exc_info:
        unit_state.assign_unit(
            sidecar, "g", "session", None, 0.0, 684.0,
            "reviewer-1")
    assert exc_info.value.code == "SCOPE_INVALID"
    unit_state.assign_unit(
        sidecar, "g", "session", None, 0.0, 684.0, "reviewer-1",
        justification="single scale visible throughout")
    assert unit_state.resolve_unit(
        150.0, "other_event", sidecar)["unit"] == "g"


def test_conflicting_assignments_rejected(tmp_path: Path) -> None:
    sidecar = unit_state.new_sidecar()
    unit_state.assign_unit(
        sidecar, "g", "event_window", EVENT_ID, 142.0, 153.0,
        "reviewer-1")
    with pytest.raises(ReconstructionError) as exc_info:
        unit_state.assign_unit(
            sidecar, "kg", "event_window", EVENT_ID, 150.0, 160.0,
            "reviewer-2")
    assert exc_info.value.code == "UNIT_CONFLICT"
    with pytest.raises(ReconstructionError) as exc_info:
        unit_state.assign_unit(
            sidecar, "ml", "event_window", EVENT_ID, 142.0, 153.0,
            "reviewer-2")
    assert exc_info.value.code == "VOLUME_FORBIDDEN"


def test_append_only_corrections(tmp_path: Path) -> None:
    sidecar = unit_state.new_sidecar()
    unit_state.assign_unit(
        sidecar, "g", "event_window", EVENT_ID, 142.0, 153.0,
        "reviewer-1")
    unit_state.assign_unit(
        sidecar, "g", "event_window", EVENT_ID, 142.0, 153.0,
        "reviewer-2", note="reconfirmed")
    assert len(sidecar["assignments"]) == 2
    first, second = sidecar["assignments"]
    assert first["status"] == "superseded"
    assert second["status"] == "active"
    assert second["revision"] == 2
    assert second["previous_assignment_id"] == "ua_0001"


def test_unknown_ingredients_and_quantities(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    draft = _run_reconstruct(tmp_path, paths)
    assert draft["status"] == "pending_review"
    assert draft["ingredients"] == []
    weighing = json.loads((tmp_path / "recon" / "run_01"
                           / "weighing_event.json").read_text(
                               encoding="utf-8"))
    assert weighing["net_ingredient_quantity"] == {
        "value": None, "unit": None, "derivation": "unknown"}
    assert weighing["final_display_reading"]["unit"] is None
    codes = {question["code"]
             for question in draft["unresolved_questions"]}
    assert {"INGREDIENT_UNKNOWN", "NET_QUANTITY_UNSUPPORTED",
            "UNIT_VISUAL_ABSENT", "UNKNOWN_TARE_HISTORY",
            "TEMPORAL_NOT_EVALUATED"} <= codes


def test_tare_evidence_observed(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    payload = json.loads(
        (paths["obs"] / "batch_test.json").read_text(encoding="utf-8"))
    payload[0]["model_fields"]["tare_action_visible"] = True
    (paths["obs"] / "batch_test.json").write_text(
        json.dumps(payload), encoding="utf-8")
    _run_reconstruct(tmp_path, paths)
    weighing = json.loads((tmp_path / "recon" / "run_01"
                           / "weighing_event.json").read_text(
                               encoding="utf-8"))
    assert weighing["tare_observed"] is True
    draft = json.loads((tmp_path / "recon" / "run_01"
                        / "recipe_draft.json").read_text(
                            encoding="utf-8"))
    codes = {question["code"]
             for question in draft["unresolved_questions"]}
    assert "UNKNOWN_TARE_HISTORY" not in codes


def test_deterministic_output(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    _run_reconstruct(tmp_path, paths, run_id="run_a")
    _run_reconstruct(tmp_path, paths, run_id="run_b")
    for name in ("measurements.json", "weighing_event.json",
                 "action_events.json", "validation.json"):
        first = (tmp_path / "recon" / "run_a" / name).read_bytes()
        second = (tmp_path / "recon" / "run_b" / name).read_bytes()
        assert first == second
    first = json.loads((tmp_path / "recon" / "run_a"
                        / "recipe_draft.json").read_text(
                            encoding="utf-8"))
    second = json.loads((tmp_path / "recon" / "run_b"
                         / "recipe_draft.json").read_text(
                             encoding="utf-8"))
    assert first.pop("run_id") == "run_a"
    assert second.pop("run_id") == "run_b"
    assert first == second


def test_validation_clean_on_fixture(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    _run_reconstruct(tmp_path, paths)
    validation = json.loads((tmp_path / "recon" / "run_01"
                             / "validation.json").read_text(
                                 encoding="utf-8"))
    assert validation["issue_count"] == 0


def test_historical_inputs_preserved(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    before = {}
    for root in (paths["obs"], paths["exec"]):
        for path in sorted(root.rglob("*.json")):
            before[str(path)] = hashlib.sha256(
                path.read_bytes()).hexdigest()
    before[str(paths["temporal"])] = hashlib.sha256(
        paths["temporal"].read_bytes()).hexdigest()
    _run_reconstruct(tmp_path, paths)
    for text, digest in before.items():
        assert hashlib.sha256(
            Path(text).read_bytes()).hexdigest() == digest


def test_deltas_are_not_quantities(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    _run_reconstruct(tmp_path, paths)
    bundle = json.loads((tmp_path / "recon" / "run_01"
                         / "measurements.json").read_text(
                             encoding="utf-8"))
    by_frame = {item["frame_index"]: item
                for item in bundle["measurements"]}
    assert by_frame[7450]["delta_from_previous"] == 68.0
    assert by_frame[7500]["delta_from_previous"] == 94.0
    draft = json.loads((tmp_path / "recon" / "run_01"
                        / "recipe_draft.json").read_text(
                            encoding="utf-8"))
    assert draft["ingredients"] == []


# FPS-aware missing-timestamp fallback (M2-06C.5/S0) -----------------


def _fallback_event(frames: tuple[int, ...] = (100, 200)
                    ) -> dict[str, Any]:
    return {
        "event_id": "fallback_test",
        "candidate_type": "scale_visibility",
        "start_timestamp_seconds": 0.0,
        "end_timestamp_seconds": 10.0,
        "verification_status": "candidate",
        "frame_indices": list(frames),
        "review_reasons": [],
        "uncertainty_notes": [],
    }


def _measurement_by_frame(tmp_path: Path, run_id: str,
                          frame: int) -> dict[str, Any]:
    bundle = json.loads((tmp_path / "recon" / run_id
                         / "measurements.json").read_text(
                             encoding="utf-8"))
    return [item for item in bundle["measurements"]
            if item["frame_index"] == frame][0]


def test_missing_observation_uses_source_fps() -> None:
    measurements = assemble_measurements(
        _fallback_event(), {}, {}, source_fps=25.0)
    by_frame = {item["frame_index"]: item
                for item in measurements}
    assert by_frame[100]["timestamp_seconds"] == pytest.approx(4.0)
    assert by_frame[200]["timestamp_seconds"] == pytest.approx(8.0)
    for item in measurements:
        assert item["original"] is None
        assert item["value"] is None
        assert item["value_source"] == "none"
        assert item["value_status"] == "missing"


def test_missing_observation_legacy_fallback_without_fps(
        ) -> None:
    measurements = assemble_measurements(_fallback_event(), {}, {})
    by_frame = {item["frame_index"]: item
                for item in measurements}
    assert by_frame[100]["timestamp_seconds"] == pytest.approx(2.0)
    assert by_frame[200]["timestamp_seconds"] == pytest.approx(4.0)
    for item in measurements:
        assert item["value_source"] == "none"
        assert item["value_status"] == "missing"


def test_source_fps_accepts_integer_rates() -> None:
    measurements = assemble_measurements(
        _fallback_event((100,)), {}, {}, source_fps=25)
    assert measurements[0]["timestamp_seconds"] == pytest.approx(4.0)


def test_valid_observation_timestamp_preserved_with_source_fps(
        ) -> None:
    record = _obs_record(100, "353")
    record["timestamp_seconds"] = 148.0
    measurements = assemble_measurements(
        _fallback_event((100,)), {100: record}, {},
        source_fps=25.0)
    assert len(measurements) == 1
    assert measurements[0]["timestamp_seconds"] == 148.0
    assert measurements[0]["value"] == 353.0
    assert measurements[0]["value_source"] == "firstpass"


@pytest.mark.parametrize("bad_fps", [
    0, 0.0, -25.0, float("nan"), float("inf"),
    float("-inf"), "25", True, [25.0],
])
def test_invalid_source_fps_rejected(bad_fps: Any) -> None:
    with pytest.raises(ReconstructionError) as exc_info:
        assemble_measurements(
            _fallback_event(), {}, {}, source_fps=bad_fps)
    assert exc_info.value.code == "FPS_INVALID"


def test_fallback_timestamp_window_filtering() -> None:
    # Frame 100 lands at 4.0 s under 25 FPS but at 2.0 s
    # under the legacy 50 FPS divisor.
    included = assemble_measurements(
        _fallback_event((100,)), {}, {}, start=3.0, end=5.0,
        source_fps=25.0)
    assert [item["timestamp_seconds"] for item in included] == [4.0]
    excluded = assemble_measurements(
        _fallback_event((100,)), {}, {}, start=3.0, end=5.0)
    assert excluded == []


def test_verified_lookup_matches_fallback_timestamp() -> None:
    verified = {4.0: {
        "display_text": "120", "numeric_value": 120.0,
        "readable": True, "confidence": "high",
        "uncertainty_notes": None,
    }}
    measurements = assemble_measurements(
        _fallback_event((100,)), {}, verified, source_fps=25.0)
    assert measurements[0]["timestamp_seconds"] == pytest.approx(4.0)
    assert measurements[0]["value_source"] == "verification"
    assert measurements[0]["value"] == 120.0
    legacy = assemble_measurements(
        _fallback_event((100,)), {}, verified)
    assert legacy[0]["value_source"] == "none"
    assert legacy[0]["value"] is None


def test_reconstruct_source_fps_flag(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    batch_path = paths["obs"] / "batch_test.json"
    records = json.loads(batch_path.read_text(encoding="utf-8"))
    kept = [record for record in records
            if record["frame_index"] != 7500]
    assert len(kept) == len(records) - 1
    batch_path.write_text(json.dumps(kept), encoding="utf-8")
    args = ["reconstruct", "--obs-dir", str(paths["obs"]),
            "--temporal", str(paths["temporal"]),
            "--event-id", EVENT_ID,
            "--out", str(tmp_path / "recon")]
    assert recon_cli.main([*args, "--run-id", "legacy"]) == 0
    assert recon_cli.main(
        [*args, "--run-id", "fps25",
         "--source-fps", "25.0"]) == 0
    legacy = _measurement_by_frame(tmp_path, "legacy", 7500)
    assert legacy["timestamp_seconds"] == pytest.approx(150.0)
    assert legacy["value_source"] == "none"
    assert legacy["original"] is None
    fps25 = _measurement_by_frame(tmp_path, "fps25", 7500)
    assert fps25["timestamp_seconds"] == pytest.approx(300.0)
    assert fps25["value_source"] == "none"
    assert fps25["original"] is None
