"""Independent analysis worker for durable job execution.

Runs as a separate process from the API
(``uv run python -m bakery_intelligence.api.worker``), claims queued
jobs atomically from ``SqliteJobStore``, executes exactly one
workflow attempt per claim, and records completion or structured
failure. Artifacts are read back and re-hashed before a job may
complete: an artifact is never assumed valid from a write call.

Retry policy: only explicitly idempotent workflows may be requeued
after a worker crash (see ``WORKFLOW_MAX_ATTEMPTS``). Explicit
failures never retry automatically. Unknown future workflows —
including any paid inference — default to zero requeue attempts,
so duplicate paid operations require a deliberate policy entry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

from bakery_intelligence.api.jobs import (
    SMOKE_RESULT_ARTIFACT,
    WORKFLOW_OFFLINE_RECONSTRUCT,
    WORKFLOW_STORAGE_SMOKE,
    WORKFLOW_VIDEO_RECONSTRUCT,
    JobRecord,
    utc_now_iso,
)
from bakery_intelligence.api.offline import (
    execute_offline_reconstruct,
)
from bakery_intelligence.api.video import (
    execute_video_reconstruct,
)
from bakery_intelligence.api.storage import (
    LocalStorageBackend,
    StorageBackend,
)
from bakery_intelligence.api.store import SqliteJobStore

DEFAULT_POLL_SECONDS = 2.0
DEFAULT_STALE_SECONDS = 300.0

# Attempts are counted at claim time (first claim => attempts == 1).
# A workflow listed here tolerates that many total attempts; anything
# unlisted (including future paid inference) is never requeued.
WORKFLOW_MAX_ATTEMPTS = {
    WORKFLOW_STORAGE_SMOKE: 2,
    # Deterministic and idempotent (artifact writes overwrite via
    # read-back verification), so a crashed attempt may be requeued
    # under the same bound as the smoke workflow.
    WORKFLOW_OFFLINE_RECONSTRUCT: 2,
    # C.1 only: recorded-evidence assembly plus deterministic
    # reconstruction, idempotent like offline_reconstruct.
    # Re-evaluate before any paid stage joins this workflow name:
    # paid operations must never inherit an offline retry budget.
    WORKFLOW_VIDEO_RECONSTRUCT: 2,
}


def _storage_smoke_artifact(record: JobRecord,
                            source_sha256: str) -> dict[str, Any]:
    return {
        "workflow": WORKFLOW_STORAGE_SMOKE,
        "run_id": record.run_id,
        "source_object_id": record.source_object_id,
        "source_sha256": source_sha256,
        "video_analysis_performed": False,
        "note": ("Infrastructure smoke test only: source stored "
                 "and integrity-verified. No video was analyzed."),
    }


def execute_storage_smoke(storage: StorageBackend,
                          record: JobRecord) -> dict[str, Any]:
    """Verify a stored source; return artifact id -> bytes mapping."""
    if record.source_object_id is None:
        raise ValueError("storage_smoke requires a source object.")
    if not storage.source_exists(record.source_object_id):
        raise ValueError(f"Unknown source: {record.source_object_id!r}.")
    digest = hashlib.sha256()
    with storage.open_source(record.source_object_id) as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    payload = _storage_smoke_artifact(record, digest.hexdigest())
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
    return {SMOKE_RESULT_ARTIFACT: data}


_EXECUTORS = {
    WORKFLOW_STORAGE_SMOKE: execute_storage_smoke,
    WORKFLOW_OFFLINE_RECONSTRUCT: execute_offline_reconstruct,
}


def _hash_source(storage: StorageBackend,
                 object_id: str | None) -> str | None:
    """Stream the SHA-256 of a source object (None when absent)."""
    if object_id is None or not storage.source_exists(object_id):
        return None
    digest = hashlib.sha256()
    with storage.open_source(object_id) as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _store_verified(storage: StorageBackend, run_id: str,
                    artifacts: dict[str, bytes]) -> dict[str, str]:
    """Store artifacts and verify each by read-back before returning.

    Returns artifact id -> verified SHA-256 hex. Raises when any
    read-back digest differs from the write digest.
    """
    digests: dict[str, str] = {}
    for artifact_id, data in artifacts.items():
        written = storage.store_artifact(run_id, artifact_id, data)
        read_back = storage.read_artifact(run_id, artifact_id)
        if read_back != data:
            raise IOError(
                f"Artifact read-back mismatch: {artifact_id!r}.")
        digest = hashlib.sha256(read_back).hexdigest()
        if digest != written:
            raise IOError(
                f"Artifact digest mismatch: {artifact_id!r}.")
        digests[artifact_id] = digest
    return digests


def run_once(store: SqliteJobStore, storage: StorageBackend,
             worker_id: str,
             evidence_dir: Path | None = None) -> str | None:
    """Claim and execute one job; return its run id or None if idle.

    Recovery of stale jobs runs first so a restarted worker picks up
    work orphaned by a crashed predecessor before taking new jobs.
    ``evidence_dir`` roots recorded-evidence resolution for the
    video workflow; other workflows ignore it.
    """
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(seconds=DEFAULT_STALE_SECONDS)).isoformat()
    store.requeue_stale(now.isoformat(), cutoff, WORKFLOW_MAX_ATTEMPTS)
    record = store.claim_next(worker_id, now.isoformat())
    if record is None:
        return None
    try:
        if (record.workflow != WORKFLOW_VIDEO_RECONSTRUCT
                and record.workflow not in _EXECUTORS):
            raise ValueError(
                f"Unsupported workflow: {record.workflow!r}.")
        store.heartbeat(record.run_id, worker_id, utc_now_iso())
        if record.workflow == WORKFLOW_VIDEO_RECONSTRUCT:
            # Video resolution needs the evidence root, so it
            # dispatches explicitly instead of via _EXECUTORS.
            artifacts = execute_video_reconstruct(
                storage, record, evidence_dir)
        else:
            artifacts = _EXECUTORS[record.workflow](storage, record)
        digests = _store_verified(storage, record.run_id, artifacts)
        source_sha = _hash_source(storage, record.source_object_id)
        store.complete(record.run_id, worker_id, utc_now_iso(),
                       source_sha, digests)
    except Exception as exc:
        store.fail(record.run_id, worker_id, utc_now_iso(), str(exc))
    return record.run_id


def serve(store: SqliteJobStore, storage: StorageBackend,
          worker_id: str, poll_seconds: float = DEFAULT_POLL_SECONDS,
          stop: threading.Event | None = None,
          evidence_dir: Path | None = None) -> None:
    """Claim and execute jobs until the stop event is set."""
    stop_event = stop or threading.Event()
    while not stop_event.is_set():
        try:
            claimed = run_once(store, storage, worker_id,
                               evidence_dir=evidence_dir)
        except Exception:
            claimed = None
        if claimed is None:
            stop_event.wait(poll_seconds)


def main(argv: list[str] | None = None) -> int:
    """Worker entry point: ``python -m bakery_intelligence.api.worker``."""
    parser = argparse.ArgumentParser(
        description="Durable Bakery Intelligence analysis worker.")
    parser.add_argument("--storage-dir", type=Path, default=None,
                        help="API storage root (default: ./runs_api).")
    parser.add_argument("--db-path", type=Path, default=None,
                        help="Job database (default: <root>/jobs.sqlite3).")
    parser.add_argument("--evidence-dir", type=Path, default=None,
                        help="Recorded-evidence root for video jobs"
                        " (default: ./data/processed).")
    parser.add_argument("--worker-id", default="worker-1")
    parser.add_argument("--poll-seconds", type=float,
                        default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--once", action="store_true",
                        help="Execute at most one job, then exit.")
    args = parser.parse_args(argv)

    root = args.storage_dir or (Path.cwd() / "runs_api")
    storage = LocalStorageBackend(root)
    db_path = args.db_path or (storage.base_dir / "jobs.sqlite3")
    store = SqliteJobStore(db_path)
    if args.once:
        run_id = run_once(store, storage, args.worker_id,
                          evidence_dir=args.evidence_dir)
        print(f"worker done: {run_id or 'idle'}")
        return 0
    print(f"worker {args.worker_id} serving {storage.base_dir} ...")
    try:
        serve(store, storage, args.worker_id,
              poll_seconds=args.poll_seconds,
              evidence_dir=args.evidence_dir)
    except KeyboardInterrupt:
        print("worker stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
