"""Offline tests for the persistent job service and registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from bakery_intelligence.api.jobs import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    JobRecord,
    JobService,
    UnknownWorkflowError,
    WORKFLOW_STORAGE_SMOKE,
    new_run_id,
    utc_now_iso,
)
from bakery_intelligence.api.storage import validate_run_id
from bakery_intelligence.api.store import SqliteJobStore
from bakery_intelligence.api.worker import WORKFLOW_MAX_ATTEMPTS


def _service(tmp_path: Path) -> JobService:
    return JobService(SqliteJobStore(tmp_path / "jobs.sqlite3"))


def test_new_run_id_shape() -> None:
    assert validate_run_id(new_run_id()) is not None


def test_submit_persists_queued_job(tmp_path: Path) -> None:
    service = _service(tmp_path)
    record = service.submit(
        WORKFLOW_STORAGE_SMOKE, source_object_id="clip01.mp4")
    assert record.status == STATUS_QUEUED
    stored = service.get(record.run_id)
    assert stored is not None
    assert stored.workflow == WORKFLOW_STORAGE_SMOKE
    assert stored.source_object_id == "clip01.mp4"


def test_submit_rejects_unknown_workflow(tmp_path: Path) -> None:
    service = _service(tmp_path)
    with pytest.raises(UnknownWorkflowError):
        service.submit("full_analysis")


def test_get_unknown_run_returns_none(tmp_path: Path) -> None:
    service = _service(tmp_path)
    assert service.get("run_0123456789ab") is None


def test_records_survive_service_restart(tmp_path: Path) -> None:
    first = _service(tmp_path)
    record = first.submit(WORKFLOW_STORAGE_SMOKE)
    second = _service(tmp_path)
    reread = second.get(record.run_id)
    assert reread is not None
    assert reread.run_id == record.run_id
    assert reread.status == STATUS_QUEUED


def test_claim_is_atomic_single_winner(tmp_path: Path) -> None:
    service = _service(tmp_path)
    record = service.submit(WORKFLOW_STORAGE_SMOKE)
    store = service.store
    now = utc_now_iso()
    winner = store.claim_next("worker-a", now)
    assert winner is not None
    assert winner.run_id == record.run_id
    assert winner.status == STATUS_RUNNING
    assert store.claim_next("worker-b", now) is None
    assert service.get(record.run_id).status == STATUS_RUNNING


def test_claim_returns_none_when_idle(tmp_path: Path) -> None:
    service = _service(tmp_path)
    assert service.store.claim_next("worker-a", utc_now_iso()) is None


def test_stale_running_job_requeues_within_budget(tmp_path: Path) -> None:
    service = _service(tmp_path)
    record = service.submit(WORKFLOW_STORAGE_SMOKE)
    store = service.store
    claimed = store.claim_next("worker-a", "2026-01-01T00:00:00+00:00")
    assert claimed is not None
    recovered = store.requeue_stale(
        "2026-06-01T00:00:00+00:00", "2026-02-01T00:00:00+00:00",
        WORKFLOW_MAX_ATTEMPTS)
    assert recovered == [record.run_id]
    reread = service.get(record.run_id)
    assert reread is not None
    assert reread.status == STATUS_QUEUED
    assert reread.attempts == 1


def test_stale_job_fails_when_budget_spent(tmp_path: Path) -> None:
    service = _service(tmp_path)
    record = service.submit(WORKFLOW_STORAGE_SMOKE)
    store = service.store
    for worker, day in (("worker-a", "01"), ("worker-b", "02")):
        store.claim_next(worker, f"2026-01-{day}T00:00:00+00:00")
        store.requeue_stale(f"2026-01-{day}T12:00:00+00:00",
                            f"2026-01-{day}T06:00:00+00:00",
                            WORKFLOW_MAX_ATTEMPTS)
    assert service.get(record.run_id).status == STATUS_QUEUED
    store.claim_next("worker-c", "2026-01-03T00:00:00+00:00")
    recovered = store.requeue_stale(
        "2026-01-03T12:00:00+00:00", "2026-01-03T06:00:00+00:00",
        WORKFLOW_MAX_ATTEMPTS)
    assert recovered == [record.run_id]
    reread = service.get(record.run_id)
    assert reread is not None
    assert reread.status == STATUS_FAILED
    assert "retry budget" in (reread.error or "")


def test_unknown_workflow_never_requeues(tmp_path: Path) -> None:
    service = _service(tmp_path)
    record = service.submit(WORKFLOW_STORAGE_SMOKE)
    store = service.store
    store.claim_next("worker-a", "2026-01-01T00:00:00+00:00")
    recovered = store.requeue_stale(
        "2026-06-01T00:00:00+00:00", "2026-02-01T00:00:00+00:00", {})
    assert recovered == [record.run_id]
    assert service.get(record.run_id).status == STATUS_FAILED


def test_fail_and_complete_require_ownership(tmp_path: Path) -> None:
    service = _service(tmp_path)
    record = service.submit(WORKFLOW_STORAGE_SMOKE)
    store = service.store
    now = utc_now_iso()
    store.claim_next("worker-a", now)
    assert store.complete(
        record.run_id, "worker-b", now, None, {}) is False
    assert store.fail(record.run_id, "worker-b", now, "x") is False
    assert service.get(record.run_id).status == STATUS_RUNNING
    assert store.fail(record.run_id, "worker-a", now, "boom") is True
    assert service.get(record.run_id).status == STATUS_FAILED


def test_billable_stale_job_fails_without_requeue(
        tmp_path: Path) -> None:
    service = _service(tmp_path)
    store = service.store
    probe = JobRecord(run_id=new_run_id(), workflow="billable_probe")
    store.create(probe)
    claimed = store.claim_next(
        "worker-a", "2026-01-01T00:00:00+00:00")
    assert claimed is not None
    # The retry budget is generous on purpose: the billable
    # registry overrides it and refuses any automatic requeue.
    recovered = store.requeue_stale(
        "2026-06-01T00:00:00+00:00", "2026-02-01T00:00:00+00:00",
        {"billable_probe": 5},
        billable=frozenset({"billable_probe"}))
    assert recovered == [probe.run_id]
    reread = service.get(probe.run_id)
    assert reread is not None
    assert reread.status == STATUS_FAILED
    assert reread.attempts == 1
    assert "Billable workflow" in (reread.error or "")
    assert "automatic retry refused" in (reread.error or "")
