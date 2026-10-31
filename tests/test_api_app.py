"""Offline HTTP tests for the FastAPI spike application."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from bakery_intelligence.api.app import build_app  # noqa: E402
from bakery_intelligence.api.storage import LocalStorageBackend  # noqa: E402
from bakery_intelligence.api.store import SqliteJobStore  # noqa: E402
from bakery_intelligence.api.worker import run_once  # noqa: E402


def _stack(tmp_path: Path) -> tuple[TestClient, SqliteJobStore,
                                    LocalStorageBackend]:
    storage = LocalStorageBackend(tmp_path)
    db_path = tmp_path / "jobs.sqlite3"
    client = TestClient(build_app(storage=storage, db_path=db_path))
    return client, SqliteJobStore(db_path), storage


def test_health_reports_backend_identity(tmp_path: Path) -> None:
    client, _, _ = _stack(tmp_path)
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "bakery-intelligence-api"
    assert payload["storage_backend"] == "LocalStorageBackend"
    assert payload["storage_ready"] is True


def test_create_queues_and_worker_completes(tmp_path: Path) -> None:
    client, store, storage = _stack(tmp_path)
    response = client.post(
        "/api/analyses",
        files={"file": ("clip01.mp4", b"fake-video-bytes")},
    )
    assert response.status_code == 202
    created = response.json()
    assert created["workflow"] == "storage_smoke"
    assert created["status"] == "queued"
    assert run_once(store, storage, "worker-1") == created["run_id"]
    status = client.get(f"/api/analyses/{created['run_id']}")
    assert status.status_code == 200
    final = status.json()
    assert final["status"] == "completed"
    assert final["source_sha256"] is not None
    assert final["artifact_ids"] == ["smoke_result.json"]
    artifact = client.get(
        f"/api/analyses/{created['run_id']}/artifacts/smoke_result.json")
    assert artifact.status_code == 200
    assert artifact.json()["video_analysis_performed"] is False


def test_status_survives_app_rebuild(tmp_path: Path) -> None:
    client, store, storage = _stack(tmp_path)
    created = client.post(
        "/api/analyses", files={"file": ("c.mp4", b"fake-video-bytes")})
    run_id = created.json()["run_id"]
    run_once(store, storage, "worker-1")
    rebuilt, _, _ = _stack(tmp_path)
    status = rebuilt.get(f"/api/analyses/{run_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"


def test_create_rejects_oversized_upload(tmp_path: Path) -> None:
    client, _, _ = _stack(tmp_path)
    response = client.post(
        "/api/analyses",
        files={"file": ("big.mp4", b"x" * (16 * 1024 * 1024 + 1))},
    )
    assert response.status_code == 413


def test_create_sanitizes_malicious_filename(tmp_path: Path) -> None:
    client, store, storage = _stack(tmp_path)
    response = client.post(
        "/api/analyses",
        files={"file": ("../../evil.mp4", b"fake-video-bytes")},
    )
    assert response.status_code == 202
    object_id = response.json()["source_object_id"]
    assert ".." not in object_id and "/" not in object_id
    run_once(store, storage, "worker-1")
    final = client.get(f"/api/analyses/{response.json()['run_id']}")
    assert final.json()["status"] == "completed"


def test_get_unknown_run_returns_404(tmp_path: Path) -> None:
    client, _, _ = _stack(tmp_path)
    response = client.get("/api/analyses/run_0123456789ab")
    assert response.status_code == 404


def test_get_malformed_run_id_returns_422(tmp_path: Path) -> None:
    client, _, _ = _stack(tmp_path)
    response = client.get("/api/analyses/not-a-run")
    assert response.status_code == 422


def test_get_missing_artifact_returns_404(tmp_path: Path) -> None:
    client, store, storage = _stack(tmp_path)
    created = client.post(
        "/api/analyses", files={"file": ("c.mp4", b"fake-video-bytes")})
    run_id = created.json()["run_id"]
    run_once(store, storage, "worker-1")
    response = client.get(f"/api/analyses/{run_id}/artifacts/absent.json")
    assert response.status_code == 404
