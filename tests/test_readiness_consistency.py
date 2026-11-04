"""Readiness/queue consistency regression tests (P0).

Review Findings and Generate Recipe must derive their status from the
same persisted corrections. These tests pin the authoritative
calculation in ``build_review_tasks`` (used by both the
observation-review queue and the recipe readiness endpoints):

- resolving every finding leaves zero pending and no block;
- a resolved conflict no longer blocks;
- an unresolved (or deferred) conflict still blocks;
- deferred findings are counted consistently;
- ``cannot_determine`` stays a resolved, uncertainty-preserving
  decision and never fabricates a measurement.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from bakery_intelligence.adversarial.consolidation import (  # noqa: E402
    consolidate,
)
from bakery_intelligence.adversarial.observations import (  # noqa: E402
    build_review_tasks,
)
from bakery_intelligence.api.app import build_app  # noqa: E402
from bakery_intelligence.api.jobs import (  # noqa: E402
    JobService,
    WORKFLOW_STORAGE_SMOKE,
)
from bakery_intelligence.api.storage import LocalStorageBackend  # noqa: E402
from bakery_intelligence.api.store import SqliteJobStore  # noqa: E402


def _frame(index: int, value, readable: bool = True) -> dict:
    return {
        "frame_index": index,
        "timestamp_seconds": float(index * 10),
        "measurement": {"value": value},
        "scale": {"readable": readable},
    }


def _package(frames: list[dict], addition_frames: list[int]) -> dict:
    return {
        "run_id": "run_consistency",
        "frames": frames,
        "action_evidence": {
            "addition_frames": addition_frames,
            "tare_frames": [],
        },
    }


def _eight_task_package() -> dict:
    """Eight alternating missing/uncertain frames -> eight tasks."""
    frames = []
    for index in range(8):
        if index % 2 == 0:
            frames.append(_frame(index, None))
        else:
            frames.append(_frame(index, 100.0 + index, readable=False))
    return _package(frames, list(range(8)))


def _resolve(task_id: str, action: str,
             human_value=None) -> dict:
    return {
        "action": action,
        "human_value": human_value,
        "reviewer": "chef-test",
        "explicit_unknown": action in (
            "not_identifiable", "skip_unknown"),
    }


def test_eight_resolved_findings_report_zero_pending() -> None:
    package = _eight_task_package()
    queue = build_review_tasks(package, {})
    assert len(queue["tasks"]) == 8
    corrections = {}
    actions = ("supply", "correct", "confirm", "not_identifiable")
    for position, task in enumerate(queue["tasks"]):
        action = actions[position % len(actions)]
        value = 120.0 if action in ("supply", "correct") else None
        corrections[task["task_id"]] = _resolve(
            task["task_id"], action, value)
    resolved = build_review_tasks(package, corrections)
    assert resolved["counts"]["open"] == 0
    assert resolved["counts"]["deferred"] == 0
    assert resolved["counts"]["resolved"] == 8
    assert resolved["counts"]["blocking_errors"] == []


def test_resolved_conflict_no_longer_blocks() -> None:
    package = _package(
        [_frame(0, 10.0), _frame(1, 5.0)], addition_frames=[])
    queue = build_review_tasks(package, {})
    assert len(queue["tasks"]) == 1
    assert queue["tasks"][0]["state"] == "contradictory"
    assert queue["counts"]["blocking_errors"] == [
        queue["tasks"][0]["task_id"]]
    task_id = queue["tasks"][0]["task_id"]
    resolved = build_review_tasks(
        package, {task_id: _resolve(task_id, "correct", 5.0)})
    assert resolved["counts"]["blocking_errors"] == []
    assert resolved["counts"]["open"] == 0


def test_unresolved_conflict_still_blocks() -> None:
    package = _package(
        [_frame(0, 10.0), _frame(1, 5.0)], addition_frames=[])
    queue = build_review_tasks(package, {})
    task_id = queue["tasks"][0]["task_id"]
    assert queue["counts"]["blocking_errors"] == [task_id]
    deferred = build_review_tasks(
        package, {task_id: _resolve(task_id, "review_later")})
    assert deferred["counts"]["deferred"] == 1
    assert deferred["counts"]["blocking_errors"] == [task_id]


def test_deferred_findings_counted_consistently() -> None:
    package = _package([_frame(0, None)], addition_frames=[0])
    queue = build_review_tasks(package, {})
    assert len(queue["tasks"]) == 1
    task_id = queue["tasks"][0]["task_id"]
    deferred = build_review_tasks(
        package, {task_id: _resolve(task_id, "review_later")})
    assert deferred["counts"]["open"] == 0
    assert deferred["counts"]["deferred"] == 1
    assert deferred["counts"]["blocking_errors"] == []


def test_cannot_determine_keeps_uncertainty_without_value() -> None:
    package = _package([_frame(0, None)], addition_frames=[0])
    task_id = build_review_tasks(package, {})["tasks"][0]["task_id"]
    corrections = {task_id: _resolve(task_id, "not_identifiable")}
    resolved = build_review_tasks(package, corrections)
    assert resolved["counts"]["open"] == 0
    assert resolved["counts"]["blocking_errors"] == []
    consolidated = consolidate(package, corrections)
    assert consolidated["review_status"]["explicit_unknowns"] == 1
    unknown = consolidated["unknowns"][-1]
    assert unknown["code"] == "HUMAN_MARKED_UNKNOWN"
    assert unknown["task_id"] == task_id
    assert "effective" not in consolidated["frames"][0]


def _stack(tmp_path: Path, measurements: list[dict],
           addition_frames: list[int]) -> tuple[TestClient, str]:
    storage = LocalStorageBackend(tmp_path / "storage")
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    service = JobService(store)
    record = service.submit(
        WORKFLOW_STORAGE_SMOKE, source_object_id="clip.mp4")
    storage.store_source("clip.mp4", io.BytesIO(b"fake"))
    store.claim_next("worker-1", "2026-11-01T00:00:00+00:00")
    artifacts = {
        "measurements.json": {
            "window": {"start_seconds": 0.0, "end_seconds": 3.0},
            "measurements": measurements,
            "action_evidence": {
                "addition_frames": addition_frames, "tare_frames": [],
                "removal_frames": [], "temporal_actions": []},
        },
        "pipeline_summary.json": {
            "video": {"file_name": "clip.mp4",
                      "duration_seconds": 3.0},
            "source_sha256": "a" * 64},
        "firstpass_report.json": {"model": "muse-spark-1.3"},
        "recipe_draft.json": {"draft_id": "draft_test",
                              "unresolved_questions": []},
        "weighing_event.json": {"ingredient_id": None,
                                "verification_status": "ambiguous"},
    }
    digests = {}
    for artifact_id, payload in artifacts.items():
        digests[artifact_id] = storage.store_artifact(
            record.run_id, artifact_id, json.dumps(payload).encode())
    store.complete(record.run_id, "worker-1",
                   "2026-11-01T00:01:00+00:00", "a" * 64, digests)
    client = TestClient(build_app(
        storage=storage, db_path=tmp_path / "jobs.sqlite3"))
    return client, record.run_id


def test_api_queue_and_readiness_agree_after_resolve(
        tmp_path: Path) -> None:
    measurements = [
        {"measurement_id": "m_0", "frame_index": 0,
         "timestamp_seconds": 0.0, "value": 10.0,
         "value_status": "ok", "value_source": "firstpass",
         "unit_observed": None},
        {"measurement_id": "m_1", "frame_index": 1,
         "timestamp_seconds": 1.0, "value": 5.0,
         "value_status": "ok", "value_source": "firstpass",
         "unit_observed": None},
        {"measurement_id": "m_2", "frame_index": 2,
         "timestamp_seconds": 2.0, "value": None,
         "value_status": "invalid",
         "value_source": "firstpass_unusable",
         "unit_observed": None},
    ]
    client, run_id = _stack(tmp_path, measurements, [2])
    queue = client.get(
        f"/api/analyses/{run_id}/observation-review").json()
    assert len(queue["tasks"]) == 2
    by_state = {task["state"]: task for task in queue["tasks"]}
    blocked_before = client.get(
        f"/api/analyses/{run_id}/recipe/readiness").json()
    assert blocked_before["blocking_errors"] == [
        by_state["contradictory"]["task_id"]]
    for task in queue["tasks"]:
        if task["state"] == "contradictory":
            body = {"task_id": task["task_id"], "action": "confirm",
                    "human_value": None, "reviewer": "chef-test"}
        else:
            body = {"task_id": task["task_id"], "action": "supply",
                    "human_value": 12.0, "reviewer": "chef-test"}
        saved = client.post(
            f"/api/analyses/{run_id}/observation-review/corrections",
            json=body)
        assert saved.status_code == 200
    queue_after = client.get(
        f"/api/analyses/{run_id}/observation-review").json()
    readiness = client.get(
        f"/api/analyses/{run_id}/recipe/readiness").json()
    assert queue_after["counts"]["open"] == 0
    assert queue_after["counts"]["deferred"] == 0
    assert readiness["open_tasks"] == 0
    assert readiness["deferred"] == 0
    assert readiness["blocking_errors"] == []
    assert readiness["ready"] is True
    reloaded_queue = client.get(
        f"/api/analyses/{run_id}/observation-review").json()
    assert reloaded_queue["counts"]["resolved"] == 2
