"""Offline tests for the independent worker process."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pytest

import bakery_intelligence.api.worker as worker_module
from bakery_intelligence.api.jobs import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    JobRecord,
    JobService,
    WORKFLOW_STORAGE_SMOKE,
)
from bakery_intelligence.api.storage import (
    LocalStorageBackend,
    sha256_hex,
)
from bakery_intelligence.api.store import SqliteJobStore
from bakery_intelligence.api.worker import run_once


def _stack(tmp_path: Path):
    storage = LocalStorageBackend(tmp_path / "storage")
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    return storage, store, JobService(store)


def test_worker_completes_smoke_with_verified_artifact(
        tmp_path: Path) -> None:
    storage, store, service = _stack(tmp_path)
    storage.store_source("clip01.mp4", io.BytesIO(b"fake-video"))
    record = service.submit(
        WORKFLOW_STORAGE_SMOKE, source_object_id="clip01.mp4")
    assert run_once(store, storage, "worker-1") == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_COMPLETED
    assert final.source_sha256 == sha256_hex(b"fake-video")
    assert final.artifact_ids == ["smoke_result.json"]
    raw = storage.read_artifact(record.run_id, "smoke_result.json")
    payload = json.loads(raw)
    assert payload["video_analysis_performed"] is False
    assert payload["source_sha256"] == sha256_hex(b"fake-video")
    assert final.artifact_sha256["smoke_result.json"] == sha256_hex(raw)


def test_worker_is_idle_without_jobs(tmp_path: Path) -> None:
    storage, store, _ = _stack(tmp_path)
    assert run_once(store, storage, "worker-1") is None


def test_worker_persists_failure_for_missing_source(
        tmp_path: Path) -> None:
    storage, store, service = _stack(tmp_path)
    record = service.submit(
        WORKFLOW_STORAGE_SMOKE, source_object_id="absent.mp4")
    assert run_once(store, storage, "worker-1") == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_FAILED
    assert final.error is not None
    assert final.artifact_ids == []


def test_worker_does_not_claim_twice(tmp_path: Path) -> None:
    storage, store, service = _stack(tmp_path)
    storage.store_source("clip01.mp4", io.BytesIO(b"fake-video"))
    record = service.submit(
        WORKFLOW_STORAGE_SMOKE, source_object_id="clip01.mp4")
    assert run_once(store, storage, "worker-1") == record.run_id
    assert run_once(store, storage, "worker-2") is None
    assert service.get(record.run_id).status == STATUS_COMPLETED


def test_crashed_job_recovers_on_next_worker_pass(
        tmp_path: Path) -> None:
    storage, store, service = _stack(tmp_path)
    storage.store_source("clip01.mp4", io.BytesIO(b"fake-video"))
    record = service.submit(
        WORKFLOW_STORAGE_SMOKE, source_object_id="clip01.mp4")
    claimed = store.claim_next(
        "crashed-worker", "2020-01-01T00:00:00+00:00")
    assert claimed is not None
    # The crashed worker never heartbeats or finishes; the next pass
    # treats its ancient heartbeat as stale, requeues, and completes.
    assert run_once(store, storage, "worker-2") == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_COMPLETED
    assert final.attempts == 2


def test_completed_state_survives_restart(tmp_path: Path) -> None:
    storage, store, service = _stack(tmp_path)
    storage.store_source("clip01.mp4", io.BytesIO(b"fake-video"))
    record = service.submit(
        WORKFLOW_STORAGE_SMOKE, source_object_id="clip01.mp4")
    run_once(store, storage, "worker-1")
    reopened = JobService(SqliteJobStore(tmp_path / "jobs.sqlite3"))
    reread = reopened.get(record.run_id)
    assert reread is not None
    assert reread.status == STATUS_COMPLETED
    assert reread.artifact_ids == ["smoke_result.json"]


def test_worker_heartbeats_during_long_execution(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage, store, service = _stack(tmp_path)
    storage.store_source("clip01.mp4", io.BytesIO(b"fake-video"))
    record = service.submit(
        WORKFLOW_STORAGE_SMOKE, source_object_id="clip01.mp4")
    beats: list[str] = []
    real_heartbeat = store.heartbeat

    def counting_heartbeat(run_id: str, worker_id: str,
                           now_iso: str) -> bool:
        beats.append(now_iso)
        return real_heartbeat(run_id, worker_id, now_iso)

    monkeypatch.setattr(store, "heartbeat", counting_heartbeat)

    def slow_executor(storage: LocalStorageBackend,
                      record: JobRecord) -> dict[str, bytes]:
        time.sleep(0.2)
        return {}

    monkeypatch.setitem(
        worker_module._EXECUTORS, WORKFLOW_STORAGE_SMOKE,
        slow_executor)
    assert run_once(store, storage, "worker-1",
                    heartbeat_seconds=0.01) == record.run_id
    assert len(beats) >= 2
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_COMPLETED


def test_worker_billable_stale_job_not_requeued(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage, store, service = _stack(tmp_path)
    storage.store_source("clip01.mp4", io.BytesIO(b"fake-video"))
    record = service.submit(
        WORKFLOW_STORAGE_SMOKE, source_object_id="clip01.mp4")
    claimed = store.claim_next(
        "crashed-worker", "2020-01-01T00:00:00+00:00")
    assert claimed is not None
    # Test scope only: mark the smoke workflow billable so the
    # worker-level refusal path is exercised end to end. The
    # override replaces the production registry for this
    # test and is reverted afterwards.
    monkeypatch.setattr(
        worker_module, "BILLABLE_WORKFLOWS",
        frozenset({WORKFLOW_STORAGE_SMOKE}))
    # Recovery refuses the stale billable job, so the pass goes
    # idle: no requeue, no execution, no second attempt.
    assert run_once(store, storage, "worker-2") is None
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_FAILED
    assert final.attempts == 1
    assert final.artifact_ids == []
    assert "Billable workflow" in (final.error or "")
    assert "automatic retry refused" in (final.error or "")


def test_worker_entry_loads_local_env(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        worker_module, "_load_local_env",
        lambda: calls.append("loaded"))
    assert worker_module.main(
        ["--once", "--storage-dir", str(tmp_path)]) == 0
    assert calls == ["loaded"]


def test_recover_run_completes_failed_smoke_job(
        tmp_path: Path) -> None:
    from bakery_intelligence.api.jobs import utc_now_iso
    from bakery_intelligence.api.worker import recover_run

    storage, store, service = _stack(tmp_path)
    storage.store_source("clip01.mp4", io.BytesIO(b"fake-video"))
    record = service.submit(
        WORKFLOW_STORAGE_SMOKE, source_object_id="clip01.mp4")
    now = utc_now_iso()
    store.claim_next("worker-a", now)
    assert store.fail(
        record.run_id, "worker-a", now, "boom") is True
    status = recover_run(
        store, storage, record.run_id, "recovery-1",
        "operator-verified retry")
    assert status == STATUS_COMPLETED
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_COMPLETED
    assert final.attempts == 3


def test_recover_run_refuses_nonfailed_job(
        tmp_path: Path) -> None:
    from bakery_intelligence.api.worker import (
        RecoveryError,
        recover_run,
    )

    storage, store, service = _stack(tmp_path)
    record = service.submit(WORKFLOW_STORAGE_SMOKE)
    with pytest.raises(RecoveryError, match="not a failed job"):
        recover_run(store, storage, record.run_id,
                    "recovery-1", "note")


def test_recover_cli_requires_note(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        worker_module.main(
            ["--recover", "run_0123456789ab",
             "--storage-dir", str(tmp_path)])
    assert exc.value.code == 2


def _failed_at_ceiling(tmp_path: Path):
    """Smoke job failed twice: attempts == 3 (default ceiling)."""
    from bakery_intelligence.api.jobs import utc_now_iso

    storage, store, service = _stack(tmp_path)
    storage.store_source("clip01.mp4", io.BytesIO(b"fake-video"))
    record = service.submit(
        WORKFLOW_STORAGE_SMOKE, source_object_id="clip01.mp4")
    now = utc_now_iso()
    store.claim_next("worker-a", now)
    assert store.fail(record.run_id, "worker-a", now, "boom")
    assert store.requeue_billable_failed(
        record.run_id, now, "first retry") is True
    store.claim_next("worker-b", now)
    assert store.fail(
        record.run_id, "worker-b", now, "boom again")
    final = service.get(record.run_id)
    assert final is not None
    assert final.attempts == 3
    return storage, store, service, record


def test_recover_run_refuses_ceiling_job_by_default(
        tmp_path: Path) -> None:
    """The default ceiling still refuses a twice-failed job."""
    from bakery_intelligence.api.worker import (
        RecoveryError,
        recover_run,
    )

    storage, store, _, record = _failed_at_ceiling(tmp_path)
    with pytest.raises(RecoveryError, match="attempt ceiling"):
        recover_run(store, storage, record.run_id,
                    "recovery-2", "second retry")


def test_recover_run_accepts_operator_ceiling_override(
        tmp_path: Path) -> None:
    """An explicit higher ceiling allows one more recovery."""
    from bakery_intelligence.api.worker import recover_run

    storage, store, service, record = _failed_at_ceiling(
        tmp_path)
    status = recover_run(
        store, storage, record.run_id, "recovery-2",
        "second retry", max_attempts=5)
    assert status == STATUS_COMPLETED
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_COMPLETED
    # Requeue 3 -> 4, claim 4 -> 5: each recovery cycle
    # consumes two attempts.
    assert final.attempts == 5


def test_recover_cli_plumbs_max_attempts(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_recover(store, storage, run_id, worker_id,
                     note, **kwargs):
        captured.update(kwargs)
        captured["run_id"] = run_id
        return STATUS_COMPLETED

    monkeypatch.setattr(
        worker_module, "recover_run", fake_recover)
    assert worker_module.main(
        ["--recover", "run_0123456789ab",
         "--recover-note", "second retry",
         "--recover-max-attempts", "5",
         "--storage-dir", str(tmp_path)]) == 0
    assert captured["run_id"] == "run_0123456789ab"
    assert captured["max_attempts"] == 5


def test_recover_cli_rejects_nonpositive_ceiling(
        tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        worker_module.main(
            ["--recover", "run_0123456789ab",
             "--recover-note", "note",
             "--recover-max-attempts", "0",
             "--storage-dir", str(tmp_path)])
    assert exc.value.code == 2
