"""Offline tests for the human review workspace API (demo-first).

Exercises run listing, the read-only review bundle, retained
frame serving, and durable human annotations through the real
FastAPI app over tmp-local storage. Original inference
artifacts must stay byte-identical across review writes. No
network, no live models, no paid services.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from bakery_intelligence.api.app import build_app
from bakery_intelligence.api.jobs import (
    STATUS_COMPLETED,
    JobService,
    WORKFLOW_VIDEO_PIPELINE_OFFLINE,
)
from bakery_intelligence.api.review import (
    ReviewError,
    validate_frame_name,
)
from bakery_intelligence.api.storage import (
    LocalStorageBackend,
    sha256_hex,
)
from bakery_intelligence.api.store import SqliteJobStore
from bakery_intelligence.api.worker import run_once


def _make_video(path: Path, fps: float = 10.0,
                frame_count: int = 30) -> Path:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    if not writer.isOpened():
        pytest.skip("MJPG video encoding is unavailable.")
    for index in range(frame_count):
        shade = (index * 7) % 256
        writer.write(np.full((48, 64, 3), shade, dtype=np.uint8))
    writer.release()
    return path


def _stack(tmp_path: Path):
    storage = LocalStorageBackend(tmp_path / "storage")
    db_path = tmp_path / "jobs.sqlite3"
    return storage, db_path, TestClient(build_app(
        storage=storage, db_path=db_path))


def _complete_run(tmp_path: Path, name: str = "clip.avi") -> str:
    storage = LocalStorageBackend(tmp_path / "storage")
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    service = JobService(store)
    raw = _make_video(tmp_path / name).read_bytes()
    storage.store_source(name, io.BytesIO(raw))
    record = service.submit(WORKFLOW_VIDEO_PIPELINE_OFFLINE,
                            source_object_id=name)
    assert run_once(store, storage, "worker-1") == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_COMPLETED
    return record.run_id


def _queued_run(tmp_path: Path) -> str:
    storage = LocalStorageBackend(tmp_path / "storage")
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    service = JobService(store)
    raw = _make_video(tmp_path / "pending.avi").read_bytes()
    storage.store_source("pending.avi", io.BytesIO(raw))
    record = service.submit(WORKFLOW_VIDEO_PIPELINE_OFFLINE,
                            source_object_id="pending.avi")
    return record.run_id


def test_list_analyses_returns_completed_newest_first(
        tmp_path: Path) -> None:
    _, _, client = _stack(tmp_path)
    assert client.get("/api/analyses").json() == []
    first = _complete_run(tmp_path, "a.avi")
    second = _complete_run(tmp_path, "b.avi")
    payload = client.get("/api/analyses").json()
    assert [item["run_id"] for item in payload] == [second, first]
    assert payload[0]["status"] == STATUS_COMPLETED
    assert client.get("/api/analyses?limit=1").json()[0][
        "run_id"] == second
    assert client.get("/api/analyses?limit=0").status_code == 422
    assert client.get("/api/analyses?limit=500").status_code == 422


def test_review_bundle_maps_frames_to_observations(
        tmp_path: Path) -> None:
    _, _, client = _stack(tmp_path)
    run_id = _complete_run(tmp_path)
    response = client.get(f"/api/analyses/{run_id}/review")
    assert response.status_code == 200
    bundle = response.json()
    assert bundle["run_id"] == run_id
    assert bundle["workflow"] == WORKFLOW_VIDEO_PIPELINE_OFFLINE
    indices = [frame["frame_index"] for frame in bundle["frames"]]
    assert indices == sorted(indices)
    assert indices == [0, 10, 20]
    for frame in bundle["frames"]:
        assert frame["observation"] is not None
        assert frame["observation"]["frame_index"] == (
            frame["frame_index"])
        assert frame["image_available"] is True
        assert frame["image_matches_record"] is True
        assert frame["measurement"] is not None
        assert frame["review"] is None
    assert bundle["reviews"] == {}
    assert bundle["missing"] == []


def test_review_bundle_rejects_unknown_and_pending(
        tmp_path: Path) -> None:
    _, _, client = _stack(tmp_path)
    assert client.get(
        "/api/analyses/run_0123456789ab/review").status_code == 404
    assert client.get("/api/analyses/not-a-run/review").status_code == 422
    assert client.get("/api/analyses/../review").status_code in (404, 422)
    pending = _queued_run(tmp_path)
    assert client.get(
        f"/api/analyses/{pending}/review").status_code == 409


def test_frame_serving_returns_bytes_and_blocks_traversal(
        tmp_path: Path) -> None:
    storage, _, client = _stack(tmp_path)
    run_id = _complete_run(tmp_path)
    bundle = client.get(f"/api/analyses/{run_id}/review").json()
    name = bundle["frames"][0]["image_name"]
    response = client.get(f"/api/analyses/{run_id}/frames/{name}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == (
        storage.base_dir / "runs" / run_id / "work" / "frames" / name
    ).read_bytes()
    for bad in ("....jpg", "frame_1.jpg", "frame_00000001.png",
                "frame_XXXXXXXX.jpg", "manifest.csv"):
        assert client.get(
            f"/api/analyses/{run_id}/frames/{bad}").status_code == 422
    for dotty in ("..", "../secret", "..%2Fsecret"):
        with pytest.raises(ReviewError):
            validate_frame_name(dotty)
    assert client.get(
        f"/api/analyses/{run_id}/frames/frame_99999999.jpg"
    ).status_code == 404
    assert client.get(
        "/api/analyses/run_0123456789ab/frames/frame_00000000.jpg"
    ).status_code == 404


def test_reviews_crud_and_restart_persistence(
        tmp_path: Path) -> None:
    _, _, client = _stack(tmp_path)
    run_id = _complete_run(tmp_path)
    assert client.get(
        f"/api/analyses/{run_id}/reviews").json()["annotations"] == []
    saved = client.post(
        f"/api/analyses/{run_id}/reviews",
        json={"frame_index": 0, "decision": "corrected",
              "value": 29.5, "notes": "digit check",
              "reviewer": "op"}).json()
    assert saved["frame_index"] == 0
    assert saved["decision"] == "corrected"
    assert saved["value"] == 29.5
    assert saved["reviewer"] == "op"
    assert "updated_at" in saved
    client.post(f"/api/analyses/{run_id}/reviews",
                json={"frame_index": 10, "decision": "uncertain"})
    payload = client.get(f"/api/analyses/{run_id}/reviews").json()
    assert [item["frame_index"] for item in payload["annotations"]] == [0, 10]
    bundle = client.get(f"/api/analyses/{run_id}/review").json()
    assert bundle["frames"][0]["review"]["decision"] == "corrected"
    rebuilt = TestClient(build_app(
        storage=LocalStorageBackend(tmp_path / "storage"),
        db_path=tmp_path / "jobs.sqlite3"))
    kept = rebuilt.get(f"/api/analyses/{run_id}/reviews").json()
    assert [item["decision"] for item in kept["annotations"]] == [
        "corrected", "uncertain"]


def test_review_upsert_replaces_without_duplicates(
        tmp_path: Path) -> None:
    _, _, client = _stack(tmp_path)
    run_id = _complete_run(tmp_path)
    client.post(f"/api/analyses/{run_id}/reviews",
                json={"frame_index": 0, "decision": "uncertain"})
    client.post(f"/api/analyses/{run_id}/reviews",
                json={"frame_index": 0, "decision": "rejected",
                      "notes": "wrong object"})
    payload = client.get(f"/api/analyses/{run_id}/reviews").json()
    assert len(payload["annotations"]) == 1
    assert payload["annotations"][0]["decision"] == "rejected"


def test_review_rejects_invalid_submissions(
        tmp_path: Path) -> None:
    _, _, client = _stack(tmp_path)
    run_id = _complete_run(tmp_path)
    base = f"/api/analyses/{run_id}/reviews"
    assert client.post(base, json={
        "frame_index": 0, "decision": "approved"}).status_code == 422
    assert client.post(base, json={
        "frame_index": 0, "decision": "corrected"}).status_code == 422
    assert client.post(base, json={
        "frame_index": 0, "decision": "corrected",
        "value": "lots"}).status_code == 422
    assert client.post(base, json={
        "frame_index": 0, "decision": "confirmed",
        "value": 5}).status_code == 422
    assert client.post(base, json={
        "frame_index": 0, "decision": "uncertain",
        "notes": "x" * 1001}).status_code == 422
    assert client.post(base, json={
        "frame_index": 999, "decision": "uncertain"}).status_code == 404
    assert client.post(
        "/api/analyses/run_0123456789ab/reviews",
        json={"frame_index": 0,
              "decision": "uncertain"}).status_code == 404
    pending = _queued_run(tmp_path)
    assert client.post(
        f"/api/analyses/{pending}/reviews",
        json={"frame_index": 0,
              "decision": "uncertain"}).status_code == 409


def test_reviews_leave_original_predictions_untouched(
        tmp_path: Path) -> None:
    storage, _, client = _stack(tmp_path)
    run_id = _complete_run(tmp_path)
    run_dir = storage.base_dir / "runs" / run_id
    before = {
        str(path.relative_to(run_dir)): path.read_bytes()
        for path in sorted(run_dir.rglob("*")) if path.is_file()
    }
    client.post(f"/api/analyses/{run_id}/reviews",
                json={"frame_index": 0, "decision": "corrected",
                      "value": 31})
    client.post(f"/api/analyses/{run_id}/reviews",
                json={"frame_index": 20, "decision": "confirmed"})
    after = {
        str(path.relative_to(run_dir)): path.read_bytes()
        for path in sorted(run_dir.rglob("*")) if path.is_file()
    }
    assert before == after
    reviews_path = (storage.base_dir / "reviews" / f"{run_id}.json")
    assert reviews_path.is_file()
    stored = json.loads(reviews_path.read_text(encoding="utf-8"))
    assert sorted(stored["annotations"]) == ["0", "20"]
