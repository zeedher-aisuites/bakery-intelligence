"""Offline tests for the processing queue, size coherence, retry.

Covers persistent job visibility across all statuses, upload and
pipeline size agreement, lifecycle transitions, failure display
data, worker absence, and billable retry safety. Isolated SQLite
databases and synthetic media only; no live inference.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from bakery_intelligence.api.app import build_app  # noqa: E402
from bakery_intelligence.api.jobs import (  # noqa: E402
    JobRecord,
    new_run_id,
)
from bakery_intelligence.api.storage import LocalStorageBackend  # noqa: E402
from bakery_intelligence.api.store import SqliteJobStore  # noqa: E402
from bakery_intelligence.api.worker import run_once  # noqa: E402

META = "video_pipeline_meta"
OFFLINE = "video_pipeline_offline"


def _stack(tmp_path: Path) -> tuple[TestClient, SqliteJobStore,
                                    LocalStorageBackend]:
    storage = LocalStorageBackend(tmp_path / "storage")
    db_path = tmp_path / "jobs.sqlite3"
    client = TestClient(build_app(storage=storage, db_path=db_path))
    return client, SqliteJobStore(db_path), storage


def _make_mp4(path: Path, frames: int = 20, fps: float = 10.0) -> bytes:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    if not writer.isOpened():
        pytest.skip("MP4 video encoding is unavailable.")
    for index in range(frames):
        writer.write(np.full((48, 64, 3), index * 7 % 256,
                             dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def _queued_offline(client: TestClient, tmp_path: Path) -> str:
    raw = _make_mp4(tmp_path / "clip.mp4")
    upload = client.post(
        "/api/uploads", files={"file": ("clip.mp4", raw)}).json()
    created = client.post(
        f"/api/uploads/{upload['upload_id']}/analyses",
        json={"workflow": OFFLINE})
    assert created.status_code == 202
    return created.json()["run_id"]


def test_queue_lists_queued_job_without_worker(tmp_path: Path) -> None:
    client, _, _ = _stack(tmp_path)
    run_id = _queued_offline(client, tmp_path)
    rows = client.get("/api/jobs").json()
    assert [row["run_id"] for row in rows] == [run_id]
    row = rows[0]
    assert row["status"] == "queued"
    assert row["workflow"] == OFFLINE
    assert row["claimed_by"] is None
    assert row["error"] is None
    assert row["created_at"] and row["updated_at"]
    detail = client.get(f"/api/analyses/{run_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "queued"


def test_queue_survives_app_rebuild(tmp_path: Path) -> None:
    client, _, _ = _stack(tmp_path)
    run_id = _queued_offline(client, tmp_path)
    rebuilt, _, _ = _stack(tmp_path)
    rows = rebuilt.get("/api/jobs").json()
    assert run_id in [row["run_id"] for row in rows]
    assert rebuilt.get(f"/api/analyses/{run_id}").json()["status"] == "queued"


def test_queue_status_filter_and_bounds(tmp_path: Path) -> None:
    client, _, _ = _stack(tmp_path)
    run_id = _queued_offline(client, tmp_path)
    queued = client.get("/api/jobs?status=queued").json()
    assert [row["run_id"] for row in queued] == [run_id]
    assert client.get("/api/jobs?status=completed").json() == []
    assert client.get("/api/jobs?status=bogus").status_code == 422
    assert client.get("/api/jobs?limit=0").status_code == 422
    assert client.get("/api/jobs?limit=201").status_code == 422


def test_queued_running_completed_transition(tmp_path: Path) -> None:
    client, store, storage = _stack(tmp_path)
    run_id = _queued_offline(client, tmp_path)
    claimed = store.claim_next("worker-1", "2026-11-01T00:00:00+00:00")
    assert claimed is not None and claimed.run_id == run_id
    running = client.get("/api/jobs?status=running").json()
    assert running[0]["run_id"] == run_id
    assert running[0]["claimed_by"] == "worker-1"
    assert running[0]["attempts"] == 1
    # Release the manual claim by failing it, then let the worker
    # path complete a fresh job end to end.
    assert store.fail(run_id, "worker-1",
                      "2026-11-01T00:01:00+00:00", "probe") is True
    second = _queued_offline(client, tmp_path)
    assert run_once(store, storage, "worker-1") == second
    done = client.get("/api/jobs?status=completed").json()
    assert [row["run_id"] for row in done] == [second]
    assert client.get("/api/analyses").json()[0]["run_id"] == second


def test_failed_job_exposes_structured_error(tmp_path: Path) -> None:
    client, store, storage = _stack(tmp_path)
    storage.store_source("bad.mp4", __import__("io").BytesIO(b"\x00" * 4096))
    from bakery_intelligence.api.jobs import JobService
    record = JobService(store).submit(OFFLINE, source_object_id="bad.mp4")
    assert run_once(store, storage, "worker-1") == record.run_id
    row = client.get(f"/api/analyses/{record.run_id}").json()
    assert row["status"] == "failed"
    assert "Cannot open video container" in row["error"]
    assert row["error_category"] == "validation"
    failed = client.get("/api/jobs?status=failed").json()
    assert failed[0]["run_id"] == record.run_id


def test_oversized_upload_rejected_before_storage(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BAKERY_PIPELINE_MAX_VIDEO_BYTES", "1024")
    client, _, storage = _stack(tmp_path)
    raw = _make_mp4(tmp_path / "clip.mp4")
    assert len(raw) > 1024
    response = client.post(
        "/api/uploads", files={"file": ("clip.mp4", raw)})
    assert response.status_code == 413
    assert client.get("/api/jobs").json() == []


def test_size_gate_at_job_creation_creates_no_job(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = _stack(tmp_path)
    raw = _make_mp4(tmp_path / "clip.mp4")
    upload = client.post(
        "/api/uploads", files={"file": ("clip.mp4", raw)}).json()
    monkeypatch.setenv("BAKERY_PIPELINE_MAX_VIDEO_BYTES", "1024")
    refused = client.post(
        f"/api/uploads/{upload['upload_id']}/analyses",
        json={"workflow": OFFLINE})
    assert refused.status_code == 413
    assert "No job was created" in refused.json()["detail"]
    assert client.get("/api/jobs").json() == []
    assert hashlib.sha256(raw).hexdigest() == upload["sha256"]


def test_retry_requeues_failed_offline_job(tmp_path: Path) -> None:
    client, store, storage = _stack(tmp_path)
    run_id = _queued_offline(client, tmp_path)
    claimed = store.claim_next("worker-1", "2026-11-01T00:00:00+00:00")
    assert claimed is not None
    assert store.fail(run_id, "worker-1",
                      "2026-11-01T00:01:00+00:00", "boom") is True
    retried = client.post(f"/api/jobs/{run_id}/retry")
    assert retried.status_code == 200
    assert retried.json()["status"] == "queued"
    assert retried.json()["error"] is None
    assert run_once(store, storage, "worker-1") == run_id
    assert client.get(f"/api/analyses/{run_id}").json()["status"] == "completed"


def test_retry_refuses_billable_and_preserves_record(
        tmp_path: Path) -> None:
    client, store, _ = _stack(tmp_path)
    probe = JobRecord(run_id=new_run_id(), workflow=META,
                      source_object_id="clip.mp4", attempts=1)
    store.create(probe)
    assert store.fail(probe.run_id, None,
                      "2026-11-01T00:00:00+00:00",
                      "Pipeline video too large") is True
    refused = client.post(f"/api/jobs/{probe.run_id}/retry")
    assert refused.status_code == 409
    assert "Billable" in refused.json()["detail"]
    row = client.get(f"/api/analyses/{probe.run_id}").json()
    assert row["status"] == "failed"
    assert row["error"] == "Pipeline video too large"
    assert row["attempts"] == 1


def test_retry_refuses_completed_and_exhausted(tmp_path: Path) -> None:
    client, store, storage = _stack(tmp_path)
    run_id = _queued_offline(client, tmp_path)
    assert client.post(f"/api/jobs/{run_id}/retry").status_code == 409
    assert run_once(store, storage, "worker-1") == run_id
    assert client.post(f"/api/jobs/{run_id}/retry").status_code == 409
    exhausted = JobRecord(run_id=new_run_id(), workflow=OFFLINE,
                          source_object_id="clip.mp4", attempts=3)
    store.create(exhausted)
    assert store.fail(exhausted.run_id, None,
                      "2026-11-01T00:00:00+00:00", "boom") is True
    refused = client.post(f"/api/jobs/{exhausted.run_id}/retry")
    assert refused.status_code == 409
    assert "Retry limit" in refused.json()["detail"]
    assert client.post(
        "/api/jobs/run_0123456789ab/retry").status_code == 404
    assert client.post("/api/jobs/not-a-run/retry").status_code == 422
