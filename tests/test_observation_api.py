"""API tests for observation review and simulated generation."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from bakery_intelligence.api.app import build_app  # noqa: E402
from bakery_intelligence.api.jobs import JobService, WORKFLOW_STORAGE_SMOKE  # noqa: E402
from bakery_intelligence.api.storage import LocalStorageBackend  # noqa: E402
from bakery_intelligence.api.store import SqliteJobStore  # noqa: E402


def _stack(tmp_path: Path) -> tuple[TestClient, str]:
    storage = LocalStorageBackend(tmp_path / "storage")
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    service = JobService(store)
    record = service.submit(
        WORKFLOW_STORAGE_SMOKE, source_object_id="clip.mp4")
    storage.store_source("clip.mp4", io.BytesIO(b"fake"))
    store.claim_next("worker-1", "2026-11-01T00:00:00+00:00")
    artifacts = {
        "measurements.json": {
            "window": {"start_seconds": 0.0, "end_seconds": 2.0},
            "measurements": [
                {"measurement_id": "m_0", "frame_index": 0,
                 "timestamp_seconds": 0.0, "value": 0.0,
                 "value_status": "ok", "value_source": "firstpass",
                 "unit_observed": None},
                {"measurement_id": "m_50", "frame_index": 50,
                 "timestamp_seconds": 1.0, "value": None,
                 "value_status": "invalid",
                 "value_source": "firstpass_unusable",
                 "unit_observed": None},
            ],
            "action_evidence": {
                "addition_frames": [50], "tare_frames": [],
                "removal_frames": [], "temporal_actions": []},
        },
        "pipeline_summary.json": {
            "video": {"file_name": "clip.mp4",
                      "duration_seconds": 2.0},
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


def test_observation_queue_correction_and_generate(
        tmp_path: Path) -> None:
    client, run_id = _stack(tmp_path)
    queue = client.get(f"/api/analyses/{run_id}/observation-review")
    assert queue.status_code == 200
    body = queue.json()
    assert body["counts"]["accepted"] == 1
    assert body["counts"]["tasks"] == 1
    task_id = body["tasks"][0]["task_id"]
    corrected = client.post(
        f"/api/analyses/{run_id}/observation-review/corrections",
        json={"task_id": task_id, "action": "correct",
              "human_value": 12.0, "reviewer": "chef-maria"})
    assert corrected.status_code == 200
    assert corrected.json()["provenance"] == "human-corrected"
    consolidated = client.get(
        f"/api/analyses/{run_id}/consolidated-evidence").json()
    assert consolidated["content_hash"]
    assert consolidated["frames"][1]["effective"]["value"] == 12.0
    assert consolidated["frames"][1]["effective"][
        "original_value"] is None
    events = client.get(
        f"/api/analyses/{run_id}/temporal-events").json()
    assert events["schema_version"] == "temporal-events/1"
    readiness = client.get(
        f"/api/analyses/{run_id}/recipe/readiness").json()
    assert readiness["ready"] is True
    assert readiness["blocking_errors"] == []
    assert readiness["live_generation"] is False
    generated = client.post(f"/api/analyses/{run_id}/recipe/generate")
    assert generated.status_code == 200
    assert generated.json()["generation"]["simulated"] is True
    again = client.post(f"/api/analyses/{run_id}/recipe/generate")
    assert again.json()["version"] == generated.json()["version"]


def test_review_later_defers_without_resolving(
        tmp_path: Path) -> None:
    client, run_id = _stack(tmp_path)
    task_id = client.get(
        f"/api/analyses/{run_id}/observation-review").json()["tasks"][0][
            "task_id"]
    deferred = client.post(
        f"/api/analyses/{run_id}/observation-review/corrections",
        json={"task_id": task_id, "action": "review_later",
              "human_value": None, "reviewer": "chef-maria"})
    assert deferred.status_code == 200
    assert deferred.json()["action"] == "review_later"
    queue = client.get(
        f"/api/analyses/{run_id}/observation-review").json()
    task = next(item for item in queue["tasks"]
                if item["task_id"] == task_id)
    assert task["status"] == "deferred"
    assert queue["counts"]["deferred"] == 1
    assert queue["counts"]["resolved"] == 0


def test_viewer_classification_preserves_original(
        tmp_path: Path) -> None:
    client, run_id = _stack(tmp_path)
    classified = client.post(
        f"/api/analyses/{run_id}/observation-review/classifications",
        json={"frame_index": 0, "action": "flag",
              "human_value": None, "reviewer": "chef-maria"})
    assert classified.status_code == 200
    assert classified.json()["provenance"] == "human-corrected"
    queue = client.get(
        f"/api/analyses/{run_id}/observation-review").json()
    assert any(task.get("from_viewer") for task in queue["tasks"])


def _task_id(client: TestClient, run_id: str) -> str:
    queue = client.get(
        f"/api/analyses/{run_id}/observation-review")
    assert queue.status_code == 200
    return queue.json()["tasks"][0]["task_id"]


def _task_status(client: TestClient, run_id: str,
                 task_id: str) -> dict:
    queue = client.get(
        f"/api/analyses/{run_id}/observation-review").json()
    return next(item for item in queue["tasks"]
                if item["task_id"] == task_id)


def test_correction_rejects_empty_and_non_finite_values(
        tmp_path: Path) -> None:
    client, run_id = _stack(tmp_path)
    task_id = _task_id(client, run_id)
    for bad in (None, "abc", True):
        denied = client.post(
            f"/api/analyses/{run_id}/observation-review/corrections",
            json={"task_id": task_id, "action": "correct",
                  "human_value": bad, "reviewer": "chef-maria"})
        assert denied.status_code == 422, bad
    for bad in (float("nan"), float("inf")):
        raw = client.post(
            f"/api/analyses/{run_id}/observation-review/corrections",
            content=json.dumps({"task_id": task_id, "action": "correct",
                                "human_value": bad, "reviewer": "chef-maria"},
                               allow_nan=True),
            headers={"Content-Type": "application/json"})
        assert raw.status_code == 422, bad
    assert _task_status(client, run_id, task_id)["status"] == "open"
    counts = client.get(
        f"/api/analyses/{run_id}/observation-review").json()["counts"]
    assert counts["resolved"] == 0


def test_confirm_needs_no_value_and_resolves(
        tmp_path: Path) -> None:
    client, run_id = _stack(tmp_path)
    task_id = _task_id(client, run_id)
    confirmed = client.post(
        f"/api/analyses/{run_id}/observation-review/corrections",
        json={"task_id": task_id, "action": "confirm",
              "human_value": None, "reviewer": "chef-maria"})
    assert confirmed.status_code == 200
    stored = confirmed.json()
    assert stored["action"] == "confirm"
    assert stored["human_value"] is None
    assert _task_status(client, run_id, task_id)["status"] == "resolved"
    assert _task_status(
        client, run_id, task_id)["correction"]["action"] == "confirm"


def test_edit_correction_replaces_effective_value(
        tmp_path: Path) -> None:
    client, run_id = _stack(tmp_path)
    task_id = _task_id(client, run_id)
    first = client.post(
        f"/api/analyses/{run_id}/observation-review/corrections",
        json={"task_id": task_id, "action": "correct",
              "human_value": 12.0, "reviewer": "chef-maria"})
    assert first.status_code == 200
    assert first.json()["revision"] == 1
    second = client.post(
        f"/api/analyses/{run_id}/observation-review/corrections",
        json={"task_id": task_id, "action": "correct",
              "human_value": 15.5, "reviewer": "chef-ana"})
    assert second.status_code == 200
    assert second.json()["revision"] == 2
    assert second.json()["reviewer"] == "chef-ana"
    consolidated = client.get(
        f"/api/analyses/{run_id}/consolidated-evidence").json()
    assert consolidated["frames"][1]["effective"]["value"] == 15.5
    assert consolidated["frames"][1]["effective"]["revision"] == 2
    assert _task_status(client, run_id, task_id)["status"] == "resolved"


def test_undo_restores_previous_decision_without_losing_audit(
        tmp_path: Path) -> None:
    client, run_id = _stack(tmp_path)
    task_id = _task_id(client, run_id)
    assert client.post(
        f"/api/analyses/{run_id}/observation-review/corrections",
        json={"task_id": task_id, "action": "correct",
              "human_value": 12.0, "reviewer": "chef-maria"}
    ).status_code == 200
    undone = client.post(
        f"/api/analyses/{run_id}/observation-review/corrections/undo",
        json={"task_id": task_id, "reviewer": "chef-maria"})
    assert undone.status_code == 200
    assert undone.json()["correction"] is None
    assert undone.json()["status"] == "open"
    assert undone.json()["undone"]["human_value"] == 12.0
    assert _task_status(client, run_id, task_id)["status"] == "open"
    assert _task_status(
        client, run_id, task_id).get("correction") is None
    consolidated = client.get(
        f"/api/analyses/{run_id}/consolidated-evidence").json()
    assert "effective" not in consolidated["frames"][1]
    repeat = client.post(
        f"/api/analyses/{run_id}/observation-review/corrections/undo",
        json={"task_id": task_id, "reviewer": "chef-maria"})
    assert repeat.status_code == 422


def test_undo_restores_edited_value_not_first_value(
        tmp_path: Path) -> None:
    client, run_id = _stack(tmp_path)
    task_id = _task_id(client, run_id)
    for value in (12.0, 15.5):
        assert client.post(
            f"/api/analyses/{run_id}/observation-review/corrections",
            json={"task_id": task_id, "action": "correct",
                  "human_value": value, "reviewer": "chef-maria"}
        ).status_code == 200
    undone = client.post(
        f"/api/analyses/{run_id}/observation-review/corrections/undo",
        json={"task_id": task_id, "reviewer": "chef-ana"})
    assert undone.status_code == 200
    assert undone.json()["correction"]["human_value"] == 12.0
    assert undone.json()["status"] == "resolved"
    consolidated = client.get(
        f"/api/analyses/{run_id}/consolidated-evidence").json()
    assert consolidated["frames"][1]["effective"]["value"] == 12.0
