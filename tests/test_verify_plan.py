"""Offline tests for the selective verification planner.

All fixtures are synthetic and tmp-local. Planning never performs
model calls, extraction, or network access; tests assert the plan
contents, honesty about missing evidence, and determinism.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "007_selective_verification"
sys.path.insert(0, str(EXPERIMENT_DIR))

import pytest  # noqa: E402

import verify  # noqa: E402


def _selection() -> dict[str, Any]:
    return {
        "windows": [
            {"window_id": "V01", "start_seconds": 142.0,
             "end_seconds": 153.0},
            {"window_id": "V02", "start_seconds": 368.0,
             "end_seconds": 378.0},
        ],
    }


def _event(event_id: str, start: float, end: float,
           status: str, frames: list[int],
           reasons: list[str] | None = None) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "candidate_type": "scale_visibility",
        "start_timestamp_seconds": start,
        "end_timestamp_seconds": end,
        "verification_status": status,
        "frame_indices": frames,
        "review_reasons": reasons or [],
        "uncertainty_notes": ["note"],
    }


def _artifact() -> dict[str, Any]:
    return {
        "artifact_version": "1.1.0",
        "events": [
            _event("needs_a", 142.0, 144.0, "needs_review",
                   [7100, 7150], ["ambiguous_parsing"]),
            _event("ok_b", 145.0, 146.0, "candidate", [7250]),
            _event("far_c", 10.0, 11.0, "needs_review", [500],
                   ["temporal_inconsistency"]),
            _event("needs_d", 368.0, 369.0, "needs_review",
                   [18400], ["temporal_inconsistency"]),
        ],
    }


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(_selection()), encoding="utf-8")
    artifact = tmp_path / "temporal.json"
    artifact.write_text(json.dumps(_artifact()), encoding="utf-8")
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "frame_00007100.jpg").write_bytes(b"fake")
    (frames / "frame_00007150.jpg").write_bytes(b"fake")
    csv_path = frames / "manifest.csv"
    csv_path.write_text(
        "frame_index,image_path\n"
        "7100,frame_00007100.jpg\n"
        "7150,frame_00007150.jpg\n"
        "18400,frame_00018400.jpg\n",
        encoding="utf-8",
    )
    return selection, artifact, csv_path


def test_plan_selects_in_scope_review_events(tmp_path: Path) -> None:
    selection, artifact, csv_path = _write_inputs(tmp_path)
    bundle = verify.write_plan(
        selection, artifact, csv_path, tmp_path / "plan.json")
    ids = [entry["event_id"] for entry in bundle["entries"]]
    assert ids == ["needs_a", "needs_d"]
    first = bundle["entries"][0]
    assert first["window_ids"] == ["V01"]
    assert first["review_reasons"] == ["ambiguous_parsing"]
    assert first["extraction_proposed"] == []
    assert first["token_budgets"] == [1024, 2048, 3072]
    assert first["max_requests"] == 3
    assert first["cost_estimate_usd"] is None
    assert "separately" in first["separation_rule"]


def test_plan_marks_frame_availability(tmp_path: Path) -> None:
    selection, artifact, csv_path = _write_inputs(tmp_path)
    bundle = verify.write_plan(
        selection, artifact, csv_path, tmp_path / "plan.json")
    by_id = {entry["event_id"]: entry
             for entry in bundle["entries"]}
    assert [frame["frame_available"]
            for frame in by_id["needs_a"]["reuse_frames"]] == [True, True]
    assert [frame["frame_available"]
            for frame in by_id["needs_d"]["reuse_frames"]] == [False]
    assert bundle["summary"] == {
        "event_count": 2,
        "reuse_frame_count": 3,
        "reuse_frames_available": 2,
        "extraction_proposed_count": 0,
        "max_total_requests": 6,
    }


def test_plan_without_extraction_manifest_is_honest(
        tmp_path: Path) -> None:
    selection, artifact, _ = _write_inputs(tmp_path)
    bundle = verify.write_plan(
        selection, artifact, None, tmp_path / "plan.json")
    assert bundle["summary"]["reuse_frames_available"] == 0
    assert all(
        frame["frame_available"] is False
        for entry in bundle["entries"]
        for frame in entry["reuse_frames"]
    )


def test_plan_is_deterministic(tmp_path: Path) -> None:
    selection, artifact, csv_path = _write_inputs(tmp_path)
    first = verify.write_plan(
        selection, artifact, csv_path, tmp_path / "a.json")
    second = verify.write_plan(
        selection, artifact, csv_path, tmp_path / "b.json")
    assert first == second
    assert (tmp_path / "a.json").read_bytes() == (
        tmp_path / "b.json").read_bytes()


def test_plan_cli_entry_points(tmp_path: Path) -> None:
    selection, artifact, csv_path = _write_inputs(tmp_path)
    assert verify.main([
        str(selection), str(artifact), str(tmp_path / "plan.json"),
        "--extraction-manifest", str(csv_path),
    ]) == 0
    assert verify.main([
        str(tmp_path / "absent.json"), str(artifact),
        str(tmp_path / "x.json"),
    ]) == 1


def test_plan_never_proposes_extraction(tmp_path: Path) -> None:
    selection, artifact, csv_path = _write_inputs(tmp_path)
    bundle = verify.write_plan(
        selection, artifact, csv_path, tmp_path / "plan.json")
    assert all(entry["extraction_proposed"] == []
               for entry in bundle["entries"])
    assert bundle["summary"]["extraction_proposed_count"] == 0
