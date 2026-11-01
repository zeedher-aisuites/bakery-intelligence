"""Phase 3 tests: real-video upload, authorization gate, job flow.

Offline only. Videos are tiny synthetic MP4s written with OpenCV;
the Meta workflow is submitted with a grant but never executed.
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
from bakery_intelligence.api.authorization import (  # noqa: E402
    confirmation_sentence,
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


def _upload(client: TestClient, raw: bytes,
            name: str = "clip.mp4") -> object:
    return client.post("/api/uploads", files={"file": (name, raw)})


def _grant(frames: int) -> dict:
    return {
        "grant_format": "bakery-billable-grant/1",
        "workflow": META,
        "provider": "meta",
        "model": "muse-spark-1.3",
        "max_requests": frames,
        "token_budget": 4096,
        "max_frames": frames,
        "operator": "test-operator",
        "created_at": "2026-11-01T00:00:00+00:00",
        "confirmation": confirmation_sentence(
            META, "muse-spark-1.3", frames, 4096, frames,
            "test-operator"),
        "timeout_seconds": 120,
        "store": True,
        "cost_ceiling_usd": 20.0,
        "rates_usd_per_m": [1.25, 4.25],
        "strict_cost": True,
    }


def test_valid_upload_stages_without_creating_job(
        tmp_path: Path) -> None:
    client, _, storage = _stack(tmp_path)
    raw = _make_mp4(tmp_path / "clip.mp4")
    response = _upload(client, raw)
    assert response.status_code == 201
    payload = response.json()
    assert payload["file_name"] == "clip.mp4"
    assert payload["bytes"] == len(raw)
    assert payload["sha256"] == hashlib.sha256(raw).hexdigest()
    assert payload["width"] == 64 and payload["height"] == 48
    assert payload["duration_seconds"] == pytest.approx(2.0, abs=0.3)
    assert payload["estimated_frames"] == 2
    assert payload["status"] == "ready"
    assert payload["deduplicated"] is False
    assert storage.source_exists(payload["upload_id"])
    # Uploading must never create or queue an analysis job.
    assert client.get("/api/analyses").json() == []


def test_upload_dedupes_identical_content(tmp_path: Path) -> None:
    client, _, _ = _stack(tmp_path)
    raw = _make_mp4(tmp_path / "clip.mp4")
    first = _upload(client, raw).json()
    second = _upload(client, raw).json()
    assert second["upload_id"] == first["upload_id"]
    assert second["deduplicated"] is True
    fetched = client.get(f"/api/uploads/{first['upload_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["sha256"] == first["sha256"]


def test_upload_rejects_unsupported_and_unreadable(
        tmp_path: Path) -> None:
    client, _, _ = _stack(tmp_path)
    text = client.post(
        "/api/uploads", files={"file": ("notes.txt", b"hello")})
    assert text.status_code == 422
    garbage = _upload(client, b"not a video at all" * 64)
    assert garbage.status_code == 422
    raw = _make_mp4(tmp_path / "clip.mp4")
    truncated = _upload(client, raw[:200])
    assert truncated.status_code == 422


def test_upload_rejects_oversized(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BAKERY_MAX_VIDEO_UPLOAD_BYTES", "512")
    client, _, _ = _stack(tmp_path)
    raw = _make_mp4(tmp_path / "clip.mp4")
    assert len(raw) > 512
    response = _upload(client, raw)
    assert response.status_code == 413


def test_meta_requires_grant_and_queues_without_executing(
        tmp_path: Path) -> None:
    client, _, _ = _stack(tmp_path)
    raw = _make_mp4(tmp_path / "clip.mp4")
    upload = _upload(client, raw).json()
    refused = client.post(
        f"/api/uploads/{upload['upload_id']}/analyses",
        json={"workflow": META})
    assert refused.status_code == 422
    created = client.post(
        f"/api/uploads/{upload['upload_id']}/analyses",
        json={"workflow": META, "grant": _grant(2)})
    assert created.status_code == 202
    body = created.json()
    assert body["status"] == "queued"
    assert "grant" not in body
    status = client.get(f"/api/analyses/{body['run_id']}").json()
    assert status["status"] == "queued"
    assert "grant" not in status
    assert client.get(
        f"/api/analyses/{body['run_id']}/progress"
    ).json()["frames_processed"] is None


def test_generic_worker_cannot_claim_billable_job(
        tmp_path: Path) -> None:
    client, store, storage = _stack(tmp_path)
    raw = _make_mp4(tmp_path / "clip.mp4")
    upload = _upload(client, raw).json()
    created = client.post(
        f"/api/uploads/{upload['upload_id']}/analyses",
        json={"workflow": META, "grant": _grant(2)})
    assert created.status_code == 202
    run_id = created.json()["run_id"]
    # A generic worker pass must leave the paid job untouched.
    assert run_once(store, storage, "generic",
                    allow_billable=False) is None
    status = client.get(f"/api/analyses/{run_id}").json()
    assert status["status"] == "queued"
    assert status["attempts"] == 0


def test_offline_job_completes_and_opens_in_review(
        tmp_path: Path) -> None:
    client, store, storage = _stack(tmp_path)
    raw = _make_mp4(tmp_path / "clip.mp4")
    upload = _upload(client, raw).json()
    created = client.post(
        f"/api/uploads/{upload['upload_id']}/analyses",
        json={"workflow": OFFLINE})
    assert created.status_code == 202
    run_id = created.json()["run_id"]
    assert run_once(store, storage, "worker-1") == run_id
    status = client.get(f"/api/analyses/{run_id}").json()
    assert status["status"] == "completed"
    progress = client.get(f"/api/analyses/{run_id}/progress").json()
    assert progress["status"] == "completed"
    assert progress["frames_total"] == 2
    assert progress["frames_processed"] == 2
    assert progress["percent"] == 100.0
    bundle = client.get(f"/api/analyses/{run_id}/review")
    assert bundle.status_code == 200
    assert len(bundle.json()["frames"]) == 2
    listed = client.get("/api/analyses").json()
    assert run_id in [row["run_id"] for row in listed]


def test_pipeline_env_bound_supports_large_video(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bakery_intelligence.api.pipeline import run_pipeline_offline
    from bakery_intelligence.api.storage import sha256_hex
    from bakery_intelligence.video.intake import VideoError

    monkeypatch.setenv(
        "BAKERY_PIPELINE_MAX_VIDEO_BYTES", str(64 * 1024 * 1024))
    raw = b"x" * (17 * 1024 * 1024)
    with pytest.raises(VideoError) as caught:
        run_pipeline_offline(
            video_bytes=raw, video_name="big.mp4",
            run_id="testrun-large", source_object_id="big.mp4",
            source_sha256=sha256_hex(raw),
            work_dir=tmp_path / "work")
    # The size gate passed; the failure is the unreadable
    # container, proving the configured bound replaced 16 MB.
    assert caught.value.code != "VIDEO_TOO_LARGE"


def test_upload_and_job_survive_rebuild(tmp_path: Path) -> None:
    client, store, storage = _stack(tmp_path)
    raw = _make_mp4(tmp_path / "clip.mp4")
    upload = _upload(client, raw).json()
    created = client.post(
        f"/api/uploads/{upload['upload_id']}/analyses",
        json={"workflow": OFFLINE}).json()
    run_once(store, storage, "worker-1")
    rebuilt, _, _ = _stack(tmp_path)
    assert rebuilt.get(
        f"/api/uploads/{upload['upload_id']}").status_code == 200
    status = rebuilt.get(f"/api/analyses/{created['run_id']}")
    assert status.json()["status"] == "completed"
