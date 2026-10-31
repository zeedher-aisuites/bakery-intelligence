"""SQLite-backed job registry for durable local execution.

Persists every job record so API restarts never lose execution
state. Claiming is a single conditional UPDATE, which makes it
atomic: at most one worker can move a job from queued to running.
Worker crashes surface as stale running rows that explicit
recovery requeues (bounded per workflow) or fails.

SQLite serves the single-host local MVP only. It must not be
treated as a shared transactional database across independent
Modal workers; that move needs a networked store.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from bakery_intelligence.api.jobs import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    JobRecord,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  run_id TEXT PRIMARY KEY,
  workflow TEXT NOT NULL,
  status TEXT NOT NULL,
  source_object_id TEXT,
  source_sha256 TEXT,
  artifact_ids TEXT NOT NULL DEFAULT '[]',
  artifact_sha256 TEXT NOT NULL DEFAULT '{}',
  error TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  claimed_by TEXT,
  claimed_at TEXT,
  heartbeat_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status);
"""

_COLUMNS = (
    "run_id", "workflow", "status", "source_object_id",
    "source_sha256", "artifact_ids", "artifact_sha256", "error",
    "attempts", "claimed_by", "claimed_at", "heartbeat_at",
    "created_at", "updated_at",
)


def _row_to_record(row: sqlite3.Row) -> JobRecord:
    payload: dict[str, Any] = dict(row)
    payload["artifact_ids"] = json.loads(payload["artifact_ids"] or "[]")
    payload["artifact_sha256"] = json.loads(
        payload["artifact_sha256"] or "{}")
    known = {field for field in JobRecord.__dataclass_fields__}
    return JobRecord(**{key: payload[key] for key in known})


class SqliteJobStore:
    """Thread-safe durable registry; one instance per process."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    @property
    def db_path(self) -> Path:
        """Return the database file path."""
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self._db_path), timeout=10.0,
            isolation_level=None, check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA busy_timeout=10000;")
        connection.execute("PRAGMA foreign_keys=ON;")
        return connection

    def _cursor(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = self._connect()
            self._local.connection = connection
        return connection

    def create(self, record: JobRecord) -> None:
        """Insert a new queued job record."""
        with self._write_lock:
            self._cursor().execute(
                "INSERT INTO jobs (run_id, workflow, status,"
                " source_object_id, source_sha256, artifact_ids,"
                " artifact_sha256, error, attempts, claimed_by,"
                " claimed_at, heartbeat_at, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (record.run_id, record.workflow, record.status,
                 record.source_object_id, record.source_sha256,
                 json.dumps(record.artifact_ids),
                 json.dumps(record.artifact_sha256),
                 record.error, record.attempts, record.claimed_by,
                 record.claimed_at, record.heartbeat_at,
                 record.created_at, record.updated_at),
            )

    def get(self, run_id: str) -> JobRecord | None:
        """Return one job record or None when unknown."""
        row = self._cursor().execute(
            "SELECT * FROM jobs WHERE run_id = ?", (run_id,),
        ).fetchone()
        return _row_to_record(row) if row is not None else None

    def claim_next(self, worker_id: str, now_iso: str) -> JobRecord | None:
        """Atomically move the oldest queued job to running.

        Returns the claimed record, or None when no job is queued.
        At most one worker can win each job: the conditional UPDATE
        matches only rows still queued.
        """
        with self._write_lock:
            connection = self._cursor()
            connection.execute("BEGIN IMMEDIATE;")
            try:
                row = connection.execute(
                    "SELECT run_id FROM jobs WHERE status = ?"
                    " ORDER BY created_at LIMIT 1;",
                    (STATUS_QUEUED,),
                ).fetchone()
                if row is None:
                    connection.execute("ROLLBACK;")
                    return None
                cursor = connection.execute(
                    "UPDATE jobs SET status = ?, claimed_by = ?,"
                    " claimed_at = ?, heartbeat_at = ?,"
                    " attempts = attempts + 1, updated_at = ?"
                    " WHERE run_id = ? AND status = ?;",
                    (STATUS_RUNNING, worker_id, now_iso, now_iso,
                     now_iso, row["run_id"], STATUS_QUEUED),
                )
                if cursor.rowcount != 1:
                    connection.execute("ROLLBACK;")
                    return None
                connection.execute("COMMIT;")
            except Exception:
                connection.execute("ROLLBACK;")
                raise
            return self.get(row["run_id"])

    def heartbeat(self, run_id: str, worker_id: str,
                  now_iso: str) -> bool:
        """Refresh a running job's heartbeat when owned by worker."""
        with self._write_lock:
            cursor = self._cursor().execute(
                "UPDATE jobs SET heartbeat_at = ?, updated_at = ?"
                " WHERE run_id = ? AND status = ? AND claimed_by = ?;",
                (now_iso, now_iso, run_id, STATUS_RUNNING, worker_id),
            )
            return cursor.rowcount == 1

    def complete(self, run_id: str, worker_id: str, now_iso: str,
                 source_sha256: str | None,
                 artifacts: dict[str, str]) -> bool:
        """Mark a worker-owned job completed with artifact digests."""
        with self._write_lock:
            cursor = self._cursor().execute(
                "UPDATE jobs SET status = ?, source_sha256 = ?,"
                " artifact_ids = ?, artifact_sha256 = ?, error = NULL,"
                " updated_at = ?"
                " WHERE run_id = ? AND status = ? AND claimed_by = ?;",
                (STATUS_COMPLETED, source_sha256,
                 json.dumps(sorted(artifacts)),
                 json.dumps(artifacts), now_iso,
                 run_id, STATUS_RUNNING, worker_id),
            )
            return cursor.rowcount == 1

    def fail(self, run_id: str, worker_id: str | None, now_iso: str,
             error: str) -> bool:
        """Mark a job failed with a structured error message.

        A None worker id bypasses ownership (used by recovery for
        stale jobs no live worker owns).
        """
        with self._write_lock:
            if worker_id is None:
                cursor = self._cursor().execute(
                    "UPDATE jobs SET status = ?, error = ?,"
                    " updated_at = ? WHERE run_id = ?;",
                    (STATUS_FAILED, error[:2000], now_iso, run_id),
                )
            else:
                cursor = self._cursor().execute(
                    "UPDATE jobs SET status = ?, error = ?,"
                    " updated_at = ? WHERE run_id = ? AND status = ?"
                    " AND claimed_by = ?;",
                    (STATUS_FAILED, error[:2000], now_iso,
                     run_id, STATUS_RUNNING, worker_id),
                )
            return cursor.rowcount == 1

    def requeue_stale(self, now_iso: str, stale_before_iso: str,
                      max_attempts: dict[str, int]) -> list[str]:
        """Recover running jobs whose heartbeat predates the cutoff.

        Each stale job is requeued while its attempt count allows
        another run, else failed explicitly. Returns recovered ids.
        """
        recovered: list[str] = []
        with self._write_lock:
            connection = self._cursor()
            rows = connection.execute(
                "SELECT run_id, workflow, attempts FROM jobs"
                " WHERE status = ? AND heartbeat_at < ?;",
                (STATUS_RUNNING, stale_before_iso),
            ).fetchall()
            for row in rows:
                allowed = max_attempts.get(row["workflow"], 0)
                if row["attempts"] <= allowed:
                    connection.execute(
                        "UPDATE jobs SET status = ?, claimed_by = NULL,"
                        " claimed_at = NULL, heartbeat_at = NULL,"
                        " error = NULL, updated_at = ? WHERE run_id = ?;",
                        (STATUS_QUEUED, now_iso, row["run_id"]),
                    )
                else:
                    connection.execute(
                        "UPDATE jobs SET status = ?, error = ?,"
                        " updated_at = ? WHERE run_id = ?;",
                        (STATUS_FAILED,
                         "Stale worker: no heartbeat; retry budget spent.",
                         now_iso, row["run_id"]),
                    )
                recovered.append(row["run_id"])
        return recovered
