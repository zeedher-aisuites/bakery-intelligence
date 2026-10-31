"""Analysis job lifecycle over a durable registry (M2-06A.1).

``JobService`` creates and reads job records through ``SqliteJobStore``,
so API restarts never lose execution state. Execution itself belongs
to the independent worker process (``api.worker``); nothing here runs
work on HTTP handler threads.

Two workflows are supported. ``storage_smoke`` covers source
storage plus SHA-256 verification; it performs no video analysis
and its outputs say so explicitly. ``offline_reconstruct``
consumes an uploaded bundle of existing analysis artifacts and
runs deterministic reconstruction (``api.offline``); it performs
no inference and its summary artifact states exactly that.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bakery_intelligence.api.store import SqliteJobStore

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

WORKFLOW_STORAGE_SMOKE = "storage_smoke"
SMOKE_RESULT_ARTIFACT = "smoke_result.json"
WORKFLOW_OFFLINE_RECONSTRUCT = "offline_reconstruct"

SUPPORTED_WORKFLOWS = (
    WORKFLOW_STORAGE_SMOKE,
    WORKFLOW_OFFLINE_RECONSTRUCT,
)


class UnknownWorkflowError(ValueError):
    """The requested workflow is not supported by this service."""


def new_run_id() -> str:
    """Generate a stable-shaped run identifier."""
    return f"run_{secrets.token_hex(6)}"


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobRecord:
    """Durable record of one submitted job (mirrors the jobs table)."""

    run_id: str
    workflow: str
    status: str = STATUS_QUEUED
    source_object_id: str | None = None
    source_sha256: str | None = None
    artifact_ids: list[str] = field(default_factory=list)
    artifact_sha256: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    attempts: int = 0
    claimed_by: str | None = None
    claimed_at: str | None = None
    heartbeat_at: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        """Return the public JSON representation of this job."""
        return {
            "run_id": self.run_id,
            "workflow": self.workflow,
            "status": self.status,
            "source_object_id": self.source_object_id,
            "source_sha256": self.source_sha256,
            "artifact_ids": list(self.artifact_ids),
            "artifact_sha256": dict(self.artifact_sha256),
            "error": self.error,
            "attempts": self.attempts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class JobService:
    """Job lifecycle owner: submit to the store, read from the store."""

    def __init__(self, store: SqliteJobStore) -> None:
        self._store = store

    @property
    def store(self) -> SqliteJobStore:
        """Return the underlying durable registry."""
        return self._store

    def submit(self, workflow: str,
               source_object_id: str | None = None) -> JobRecord:
        """Persist a queued job and return immediately (never blocks)."""
        if workflow not in SUPPORTED_WORKFLOWS:
            raise UnknownWorkflowError(
                f"Unsupported workflow: {workflow!r}.")
        record = JobRecord(
            run_id=new_run_id(), workflow=workflow,
            source_object_id=source_object_id,
        )
        self._store.create(record)
        return record

    def get(self, run_id: str) -> JobRecord | None:
        """Return a job record or None when unknown."""
        return self._store.get(run_id)
