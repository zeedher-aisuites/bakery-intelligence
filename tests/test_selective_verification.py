"""Offline tests for selective verification (Exp. 007).

All fixtures are synthetic and tmp-local; no API credentials, no
network, no real P04 data. Oracles are the documented selection,
annotation, and metric rules. Agreement scores here exercise the
metric arithmetic only and are not empirical accuracy claims.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "007_selective_verification"
sys.path.insert(0, str(EXPERIMENT_DIR))

import manifest  # noqa: E402
import metrics  # noqa: E402
import review  # noqa: E402

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


def _event(event_id: str, start: float, end: float,
           kind: str = "scale_visibility",
           status: str = "candidate") -> dict[str, Any]:
    return {
        "event_id": event_id,
        "candidate_type": kind,
        "start_timestamp_seconds": start,
        "end_timestamp_seconds": end,
        "verification_status": status,
    }


def _write_obs(obs_dir: Path, records: list[dict[str, Any]]) -> None:
    obs_dir.mkdir(parents=True, exist_ok=True)
    (obs_dir / "batch_test.json").write_text(
        json.dumps(records), encoding="utf-8"
    )


def _write_artifact(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "artifact_version": "1.1.0", "events": events,
    }), encoding="utf-8")


def _write_frames(frames_dir: Path,
                  indices: list[int]) -> Path:
    frames_dir.mkdir(parents=True, exist_ok=True)
    rows = [["frame_index", "image_path"]]
    for index in indices:
        name = f"frame_{index:08d}.jpg"
        (frames_dir / name).write_text("fake", encoding="utf-8")
        rows.append([str(index), name])
    csv_path = frames_dir / "manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    return csv_path


def _ann(frame_index: int, window_id: str = "V01",
         outcome: str = "pending",
         verbatim: str | None = "100",
         display: str | None = None,
         human_numeric: float | None = None,
         model_numeric: float | None = 100.0,
         model_status: str = "ok",
         unit: str | None = None,
         human_action: str | None = None,
         model_action: str = "none",
         model_tare: bool = False,
         reviewer: str | None = None) -> dict[str, Any]:
    human: dict[str, Any] = {
        "display_text": display, "numeric_value": human_numeric,
        "unit": unit, "action": human_action, "tare_visible": None,
        "visibility": None, "uncertainty_notes": None,
    }
    return {
        "annotation_id": review.annotation_id(window_id, frame_index),
        "window_id": window_id,
        "frame_index": frame_index,
        "timestamp_seconds": frame_index / 50.0,
        "model_prediction": {
            "scale_display_verbatim": verbatim,
            "model_numeric": model_numeric,
            "model_numeric_status": model_status,
            "scale_readable": True,
            "scale_state": "stable",
            "tare_action_visible": model_tare,
            "addition_or_removal_visible": model_action,
            "confidence": "high",
            "ambiguity_notes": None,
            "batch_id": "batch_test",
        },
        "human": human,
        "support": {"frame_index": frame_index,
                    "timestamp_seconds": frame_index / 50.0},
        "outcome": outcome,
        "reviewer": reviewer,
        "reviewed_at": "2026-10-30" if reviewer else None,
    }


def test_selection_is_deterministic(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [_obs(7100), _obs(7200)])
    artifact = tmp_path / "temporal.json"
    _write_artifact(artifact, [_event("e1", 142.0, 144.0)])
    csv_path = _write_frames(tmp_path / "frames", [7100, 7200])
    first = manifest.generate(
        artifact, obs_dir, csv_path, tmp_path / "a.json")
    second = manifest.generate(
        artifact, obs_dir, csv_path, tmp_path / "b.json")
    assert first == second
    assert (tmp_path / "a.json").read_bytes() == (
        tmp_path / "b.json").read_bytes()


def test_window_overlap_selects_events(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [_obs(7100)])
    artifact = tmp_path / "temporal.json"
    _write_artifact(artifact, [
        _event("inside", 143.0, 144.0),
        _event("partial_start", 140.0, 142.0),
        _event("partial_end", 153.0, 160.0),
        _event("outside", 200.0, 210.0),
    ])
    csv_path = _write_frames(tmp_path / "frames", [7100])
    result = manifest.generate(
        artifact, obs_dir, csv_path, tmp_path / "out.json")
    window = [entry for entry in result["windows"]
              if entry["window_id"] == "V01"][0]
    refs = {ref["event_id"]: ref["fully_inside"]
            for ref in window["event_refs"]}
    assert refs == {"inside": True, "partial_start": False,
                    "partial_end": False}


def test_observations_bounded_and_ordered(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [_obs(7300), _obs(0), _obs(7100), _obs(7200)])
    artifact = tmp_path / "temporal.json"
    _write_artifact(artifact, [])
    csv_path = _write_frames(tmp_path / "frames", [7100, 7200, 7300])
    result = manifest.generate(
        artifact, obs_dir, csv_path, tmp_path / "out.json")
    window = [entry for entry in result["windows"]
              if entry["window_id"] == "V01"][0]
    frames = [obs["frame_index"] for obs in window["observations"]]
    assert frames == [7100, 7200, 7300]
    for obs in window["observations"]:
        assert 142.0 <= obs["timestamp_seconds"] <= 153.0
        assert obs["frame_accessible"] is True
    assert window["clip_range"] == {
        "start_frame_index": 7100, "end_frame_index": 7300}


def test_missing_extraction_manifest_recorded(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [_obs(7100)])
    artifact = tmp_path / "temporal.json"
    _write_artifact(artifact, [])
    result = manifest.generate(
        artifact, obs_dir, tmp_path / "absent.csv",
        tmp_path / "out.json")
    assert result["source"]["extraction_manifest"] is None
    assert result["source"]["frames_accessible"] is False
    assert result["source"]["missing"]
    window = [entry for entry in result["windows"]
              if entry["window_id"] == "V01"][0]
    assert window["observations"][0]["frame_path"] is None
    assert window["observations"][0]["frame_accessible"] is False


def test_missing_frame_files_recorded(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [_obs(7100)])
    artifact = tmp_path / "temporal.json"
    _write_artifact(artifact, [])
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    csv_path = frames_dir / "manifest.csv"
    csv_path.write_text("frame_index,image_path\n7100,frame_00007100.jpg\n",
                        encoding="utf-8")
    result = manifest.generate(
        artifact, obs_dir, csv_path, tmp_path / "out.json")
    assert result["source"]["frames_accessible"] is False
    assert any("not found" in note
               for note in result["source"]["missing"])
    assert result["summary"]["frames_missing"] == 1


def test_window_without_events_stays_honest(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [_obs(18400)])
    artifact = tmp_path / "temporal.json"
    _write_artifact(artifact, [_event("far", 10.0, 11.0)])
    csv_path = _write_frames(tmp_path / "frames", [18400])
    result = manifest.generate(
        artifact, obs_dir, csv_path, tmp_path / "out.json")
    window = [entry for entry in result["windows"]
              if entry["window_id"] == "V02"][0]
    assert window["event_refs"] == []
    assert len(window["observations"]) == 1


def test_templates_blank_with_frozen_prediction(tmp_path: Path) -> None:
    bundle = review.build_templates({
        "windows": [{
            "window_id": "V01",
            "observations": [
                {"frame_index": 7500, "timestamp_seconds": 150.0,
                 "batch_id": "b", "model_fields": {
                     "scale_display_verbatim": "5 15",
                     "scale_readable": True, "scale_state": "stable",
                     "tare_action_visible": False,
                     "addition_or_removal_visible": "none",
                     "confidence": "high", "ambiguity_notes": None}},
                {"frame_index": 7550, "timestamp_seconds": 151.0,
                 "batch_id": "b", "model_fields": {
                     "scale_display_verbatim": "- 750",
                     "scale_readable": True, "scale_state": "stable",
                     "tare_action_visible": False,
                     "addition_or_removal_visible": "none",
                     "confidence": "high", "ambiguity_notes": None}},
            ],
        }],
    })
    assert bundle["annotation_count"] == 2
    first, second = bundle["annotations"]
    assert first["annotation_id"] == "V01-00007500"
    assert first["outcome"] == "pending"
    assert first["reviewer"] is None
    assert set(first["human"].values()) == {None}
    assert first["model_prediction"]["model_numeric"] is None
    assert first["model_prediction"][
        "model_numeric_status"] == "unparseable"
    assert second["model_prediction"]["model_numeric"] == -750.0
    assert second["model_prediction"][
        "model_numeric_status"] == "normalized"


def test_pending_with_content_rejected() -> None:
    item = _ann(7100, display="100")
    codes = {issue["code"]
             for issue in review.validate_annotation(item)}
    assert "PENDING_NOT_BLANK" in codes


def test_confirmed_requires_exact_match() -> None:
    good = _ann(7100, outcome="confirmed", verbatim="100",
                display="100", reviewer="r1")
    assert review.validate_annotation(good) == []
    bad = _ann(7100, outcome="confirmed", verbatim="100",
               display="101", reviewer="r1")
    codes = {issue["code"]
             for issue in review.validate_annotation(bad)}
    assert "MISMATCH_CONFIRMED" in codes


def test_corrected_rules() -> None:
    good = _ann(7100, outcome="corrected", verbatim="5 15",
                display="515", reviewer="r1")
    assert review.validate_annotation(good) == []
    null_model = _ann(7100, outcome="corrected", verbatim=None,
                      display="515", reviewer="r1",
                      model_numeric=None, model_status="null_input")
    assert review.validate_annotation(null_model) == []
    bad = _ann(7100, outcome="corrected", verbatim="100",
               display="100", reviewer="r1")
    codes = {issue["code"]
             for issue in review.validate_annotation(bad)}
    assert "INVALID_CORRECTION" in codes


def test_unreadable_and_insufficient_rules() -> None:
    readable_bad = _ann(7100, outcome="unreadable", display="100",
                        reviewer="r1")
    codes = {issue["code"]
             for issue in review.validate_annotation(readable_bad)}
    assert "NONBLANK_UNREVIEWABLE" in codes
    readable_good = _ann(7100, outcome="unreadable", reviewer="r1")
    assert review.validate_annotation(readable_good) == []
    insufficient_bad = _ann(7100, outcome="insufficient_evidence",
                            human_action="addition", reviewer="r1")
    codes = {issue["code"]
             for issue in review.validate_annotation(insufficient_bad)}
    assert "NONBLANK_UNREVIEWABLE" in codes
    insufficient_good = _ann(7100, outcome="insufficient_evidence",
                             reviewer="r1")
    assert review.validate_annotation(insufficient_good) == []


def test_field_constraints() -> None:
    assert {issue["code"] for issue in review.validate_annotation(
        _ann(7100, outcome="done", reviewer="r1"))} == {"INVALID_OUTCOME"}
    numeric = _ann(7100, outcome="confirmed", verbatim="100",
                   display="100", reviewer="r1")
    numeric["human"]["numeric_value"] = "heavy"
    assert "INVALID_NUMERIC" in {
        issue["code"] for issue in review.validate_annotation(numeric)}
    unit = _ann(7100, outcome="confirmed", verbatim="100",
                display="100", reviewer="r1", unit="g")
    assert "UNIT_WITHOUT_NUMERIC" in {
        issue["code"] for issue in review.validate_annotation(unit)}
    unit["human"]["numeric_value"] = 100.0
    assert review.validate_annotation(unit) == []
    unit["human"]["unit"] = "cups"
    assert "INVALID_UNIT" in {
        issue["code"] for issue in review.validate_annotation(unit)}
    action = _ann(7100, outcome="confirmed", verbatim="100",
                  display="100", reviewer="r1",
                  human_action="stirring")
    assert "INVALID_ACTION" in {
        issue["code"] for issue in review.validate_annotation(action)}


def test_duplicate_id_detected() -> None:
    first = _ann(7100, outcome="confirmed", verbatim="100",
                 display="100", reviewer="r1")
    assert "DUPLICATE_ID" in {
        issue["code"] for issue in review.validate_all([first, first])}


def test_display_exact_match_metric() -> None:
    annotations = [
        _ann(7100, outcome="confirmed", verbatim="100", display="100",
             reviewer="r1"),
        _ann(7200, outcome="corrected", verbatim="101", display="102",
             reviewer="r1", human_numeric=102.0, model_numeric=101.0),
        _ann(7300),
        _ann(7400, outcome="corrected", verbatim=None, display="50",
             reviewer="r1", model_numeric=None,
             model_status="null_input"),
    ]
    report = metrics.compare(annotations)
    display = report["display_exact_match"]
    assert display["matches"] == 1
    assert display["denominator"] == 2
    assert display["value"] == 0.5
    assert display["excluded"] == {"not_reviewed_display": 1,
                                   "model_verbatim_missing": 1}


def test_numeric_agreement_metric() -> None:
    annotations = [
        _ann(7100, outcome="confirmed", verbatim="100", display="100",
             human_numeric=100.0, reviewer="r1"),
        _ann(7200, outcome="corrected", verbatim="101", display="102",
             human_numeric=103.0, model_numeric=101.0,
             reviewer="r1"),
        _ann(7300, outcome="corrected", verbatim="5 15", display="515",
             human_numeric=515.0, model_numeric=None,
             model_status="unparseable", reviewer="r1"),
        _ann(7400, outcome="confirmed", verbatim="50", display="50",
             model_numeric=50.0, reviewer="r1"),
    ]
    report = metrics.compare(annotations)
    numeric = report["numeric_agreement"]
    assert numeric["matches"] == 1
    assert numeric["denominator"] == 2
    assert numeric["value"] == 0.5
    assert numeric["excluded"] == {"model_numeric_ambiguous": 1,
                                   "human_numeric_missing": 1}


def test_action_precision_recall() -> None:
    annotations = [
        _ann(7100, outcome="confirmed", verbatim="10", display="10",
             human_action="addition", model_action="addition",
             reviewer="r1"),
        _ann(7200, outcome="corrected", verbatim="10", display="11",
             human_action="none", model_action="addition",
             model_numeric=10.0, reviewer="r1"),
        _ann(7300, outcome="corrected", verbatim="10", display="11",
             human_action="removal", model_action="none",
             model_numeric=10.0, reviewer="r1"),
        _ann(7400, outcome="confirmed", verbatim="10", display="10",
             human_action="tare", model_tare=True, reviewer="r1"),
    ]
    report = metrics.compare(annotations)
    assert report["action_agreement"]["matches"] == 2
    assert report["action_agreement"]["denominator"] == 4
    classes = report["action_classes"]
    assert classes["addition"]["precision"] == 0.5
    assert classes["addition"]["recall"] == 1.0
    assert classes["removal"]["precision"] is None
    assert classes["removal"]["recall"] == 0.0
    assert classes["tare"]["precision"] == 1.0
    assert classes["tare"]["recall"] == 1.0
    assert classes["none"]["precision"] == 0.0
    assert classes["none"]["recall"] == 0.0


def test_coverage_and_abstention() -> None:
    annotations = [
        _ann(7100, outcome="confirmed", verbatim="100", display="100",
             reviewer="r1"),
        _ann(7200, outcome="unreadable", reviewer="r1"),
        _ann(7300),
    ]
    report = metrics.compare(annotations)
    assert report["coverage"]["value"] == pytest.approx(2 / 3)
    assert report["abstention"]["value"] == 0.5


def test_empty_annotations_yield_nulls() -> None:
    report = metrics.compare([])
    assert report["annotation_count"] == 0
    assert report["coverage"]["value"] is None
    assert report["abstention"]["value"] is None
    assert report["display_exact_match"]["value"] is None
    assert report["display_exact_match"]["denominator"] == 0


def test_invalid_annotations_block_metrics() -> None:
    with pytest.raises(metrics.MetricsError) as exc_info:
        metrics.compare([_ann(7100, display="100")])
    assert exc_info.value.code == "INVALID_ANNOTATIONS"


def test_cli_entry_points(tmp_path: Path) -> None:
    obs_dir = tmp_path / "obs"
    _write_obs(obs_dir, [_obs(7100)])
    artifact = tmp_path / "temporal.json"
    _write_artifact(artifact, [_event("e1", 142.0, 143.0)])
    csv_path = _write_frames(tmp_path / "frames", [7100])
    assert manifest.main([str(artifact), str(obs_dir), str(csv_path),
                          str(tmp_path / "manifest.json")]) == 0
    assert review.main(["template", str(tmp_path / "manifest.json"),
                        str(tmp_path / "templates.json")]) == 0
    assert review.main(["validate",
                        str(tmp_path / "templates.json")]) == 0
    assert metrics.main([str(tmp_path / "templates.json"),
                         str(tmp_path / "metrics.json")]) == 0
    assert manifest.main([str(tmp_path / "absent.json"), str(obs_dir),
                          str(csv_path),
                          str(tmp_path / "x.json")]) == 1
