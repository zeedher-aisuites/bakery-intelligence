"""Offline tests for M2-05A read-only reconstruction adapters.

Fixtures are synthetic structures mirroring the real v01_full JSON
layout. They validate adapter behavior only and are never presented
as experimental evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bakery_intelligence.app.adapters import (
    event_window,
    find_repo_root,
    load_bundle,
    measurement_rows,
    run_directory,
    unit_for_measurement,
    unresolved_questions,
    video_path,
)

MEASUREMENTS_FIXTURE = {
    "event_id": "scale_visibility_test",
    "window": {"start_seconds": 142.0, "end_seconds": 144.0},
    "measurements": [
        {
            "delta_from_previous": None,
            "frame_index": 7100,
            "measurement_id": "m_test_1",
            "original": {
                "confidence": "medium",
                "readable": True,
                "timestamp_seconds": 142.0,
                "verbatim": "61",
            },
            "timestamp_seconds": 142.0,
            "unit_observed": None,
            "value": 61.0,
            "value_source": "firstpass",
            "value_status": "ok",
            "verified": None,
        },
        {
            "delta_from_previous": 454.0,
            "frame_index": 7500,
            "measurement_id": "m_test_2",
            "original": {
                "confidence": "medium",
                "readable": True,
                "timestamp_seconds": 150.0,
                "verbatim": "5 15",
            },
            "timestamp_seconds": 150.0,
            "unit_observed": None,
            "value": 515.0,
            "value_source": "verification",
            "value_status": "ok",
            "verified": {
                "confidence": "high",
                "display_text": "515",
                "numeric_value": 515,
                "readable": True,
                "uncertainty_notes": None,
            },
        },
    ],
    "unit_resolutions": [
        {
            "assignment_id": "ua_test",
            "measurement_id": "m_test_1",
            "provenance": "human_assigned",
            "unit": "g",
        },
    ],
}

DRAFT_FIXTURE = {
    "draft_id": "draft_test",
    "status": "pending_review",
    "version": 1,
    "video_id": "P04_104",
    "steps": [],
    "unresolved_questions": [
        {
            "code": "INGREDIENT_UNKNOWN",
            "message": "No observation identifies the ingredient.",
            "related_entity_ids": ["scale_visibility_test"],
        },
    ],
}


def _write_run(root: Path, run_id: str = "v01_full") -> Path:
    run_dir = root / "data/processed/experiment_008/reconstruction" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "measurements.json").write_text(
        json.dumps(MEASUREMENTS_FIXTURE), encoding="utf-8")
    (run_dir / "recipe_draft.json").write_text(
        json.dumps(DRAFT_FIXTURE), encoding="utf-8")
    (run_dir / "weighing_event.json").write_text("{}", encoding="utf-8")
    (run_dir / "action_events.json").write_text("[]", encoding="utf-8")
    (run_dir / "units_applied.json").write_text("{}", encoding="utf-8")
    (run_dir / "validation.json").write_text(
        json.dumps({"issue_count": 0, "issues": []}), encoding="utf-8")
    return run_dir


def test_load_bundle_reads_all_artifacts(tmp_path: Path) -> None:
    _write_run(tmp_path)
    bundle = load_bundle(tmp_path)
    assert bundle.run_id == "v01_full"
    assert bundle.issues == ()
    assert bundle.measurements["event_id"] == "scale_visibility_test"
    assert bundle.recipe_draft["status"] == "pending_review"
    assert bundle.validation["issue_count"] == 0


def test_load_bundle_reports_missing_files(tmp_path: Path) -> None:
    bundle = load_bundle(tmp_path)
    assert len(bundle.issues) == 6
    assert {issue.code for issue in bundle.issues} == {"missing_file"}
    assert bundle.measurements == {}
    assert bundle.action_events == ()


def test_load_bundle_reports_corrupt_file(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    (run_dir / "measurements.json").write_text("{invalid", encoding="utf-8")
    bundle = load_bundle(tmp_path)
    codes = [issue.code for issue in bundle.issues]
    assert codes == ["corrupt_file"]
    assert bundle.measurements == {}
    # Other artifacts still load independently.
    assert bundle.recipe_draft["status"] == "pending_review"


def test_load_does_not_modify_source_files(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    before = {
        path.name: path.read_bytes() for path in run_dir.iterdir()
    }
    load_bundle(tmp_path)
    after = {path.name: path.read_bytes() for path in run_dir.iterdir()}
    assert before == after


def test_measurement_rows_preserve_provenance(tmp_path: Path) -> None:
    _write_run(tmp_path)
    rows = measurement_rows(load_bundle(tmp_path))
    assert len(rows) == 2
    first, second = rows
    assert first["original_verbatim"] == "61"
    assert first["value_source"] == "firstpass"
    assert first["has_disagreement"] is False
    assert second["original_verbatim"] == "5 15"
    assert second["verified_display"] == "515"
    assert second["value_source"] == "verification"
    assert second["has_disagreement"] is True


def test_unit_resolution_distinguishes_sources(tmp_path: Path) -> None:
    _write_run(tmp_path)
    bundle = load_bundle(tmp_path)
    resolved = unit_for_measurement(bundle, "m_test_1")
    assert resolved is not None
    assert resolved["unit"] == "g"
    assert resolved["provenance"] == "human_assigned"
    assert unit_for_measurement(bundle, "m_test_2") is None
    rows = measurement_rows(bundle)
    assert rows[0]["unit_observed"] is None
    assert rows[0]["unit_assigned"] == "g"


def test_unresolved_questions_passthrough(tmp_path: Path) -> None:
    _write_run(tmp_path)
    questions = unresolved_questions(load_bundle(tmp_path))
    assert len(questions) == 1
    assert questions[0]["code"] == "INGREDIENT_UNKNOWN"


def test_event_window_reports_bounds(tmp_path: Path) -> None:
    _write_run(tmp_path)
    window = event_window(load_bundle(tmp_path))
    assert window == {
        "event_id": "scale_visibility_test",
        "start_seconds": 142.0,
        "end_seconds": 144.0,
    }


def test_path_helpers_use_relative_layout(tmp_path: Path) -> None:
    assert run_directory(tmp_path).name == "v01_full"
    assert str(video_path(tmp_path)).replace("\\", "/").endswith(
        "data/raw/epic_kitchens/P04_104.MP4")


def test_find_repo_root_from_package() -> None:
    root = find_repo_root()
    assert (root / "pyproject.toml").is_file()


def test_find_repo_root_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_repo_root(tmp_path)
