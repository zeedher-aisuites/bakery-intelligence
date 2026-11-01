# Local Backup System

## Purpose

The backup system preserves Bakery Intelligence application data —
paid inference results, original videos, evidence artifacts, SQLite
state, and human review records — in a sibling directory outside
the Git repository:

```text
C:\Users\zeedher\Documents\github_portafolio\bakery-intelligence-backups
```

It is implemented in `scripts/backup_data.py` using only the
Python 3.12 standard library. Backups never modify, move, or
delete original data, never call the Meta API, and never expose
credentials.

## Architecture

The utility snapshots the API storage tree (`runs_api` by
default). Two storage layers are written:

- **Content-addressed blobs** under
  `objects/<sha256[:2]>/<sha256>`. Every file is stored once per
  content hash. The multi-gigabyte source video and its pipeline
  work copies share a single blob, and unchanged files are never
  recopied by later snapshots.
- **Snapshots** under `snapshots/<snapshot_id>/manifest.json`.
  The manifest is written atomically as the final step of a
  backup and records each file's path, size, SHA-256, blob
  reference, kind, and status, plus totals, the backup reason,
  any jobs active at backup time, and the completed run ids.

A snapshot directory without a manifest is an interrupted
attempt. It is listed as `incomplete` and cannot be verified or
restored. Mutable records (review files, metadata sidecars,
pipeline state) are versioned naturally: changed content produces
a new blob, and each manifest pins the version it captured.

### Data included

- Original uploaded videos (`sources/`).
- Upload metadata and integrity sidecars (`uploads/`).
- Completed and historical analysis runs (`runs/`, artifacts and
  retained work trees: extracted frames, raw responses,
  observations, checkpoints, pipeline state, reports).
- Human review records (`reviews/`, when present).
- SQLite application state (`jobs.sqlite3`), captured through
  the SQLite backup API so a snapshot is transactionally
  consistent even while the API is running. The live `-wal` and
  `-shm` files are never copied directly.

### Exclusions

`.env` files and other credential material, temporary upload
parts (`uploads_tmp/`, `*.part`), Python caches, and build
artifacts are excluded. Grants and documentation live in the
repository and are outside this system's scope.

## Commands

Run from the repository root:

```powershell
# Manual backup
.venv\Scripts\python.exe scripts/backup_data.py run

# Pre-merge / pre-commit backup (same command, labeled reason)
.venv\Scripts\python.exe scripts/backup_data.py run --reason pre-merge

# List snapshots with status
.venv\Scripts\python.exe scripts/backup_data.py list

# Rehash every blob referenced by the latest snapshot
.venv\Scripts\python.exe scripts/backup_data.py verify

# Post-analysis trigger: backs up only when a new run completed
# since the last completed snapshot; otherwise exits quickly.
.venv\Scripts\python.exe scripts/backup_data.py trigger
```

`--source` and `--dest` override the storage root and backup
destination (used by tests). A `run` always ends with a full
hash verification of the new snapshot; failure is reported and
the original data remains untouched.

## Automation

- **After a completed analysis:** invoke
  `scripts/backup_data.py trigger` from any external hook (for
  example a wrapper that runs the one-shot worker). It is a
  separate short-lived process: it cannot block or corrupt the
  inference pipeline, and it does nothing when no new run has
  completed. It is intentionally not wired into the worker.
- **Pre-merge / pre-commit:** run
  `scripts/backup_data.py run --reason pre-merge` before merging
  branches or creating commits that touch application data.
- **Periodic (optional, not installed):** Windows Task Scheduler
  can run the trigger hourly without any code change:

  ```powershell
  schtasks /Create /TN "BakeryIntelligenceBackup" /SC HOURLY `
    /TR "C:\Users\zeedher\Documents\github_portafolio\bakery-intelligence\.venv\Scripts\python.exe C:\Users\zeedher\Documents\github_portafolio\bakery-intelligence\scripts\backup_data.py trigger"
  ```

  Installing this task requires explicit operator approval and
  has not been done.

Backups taken while jobs are active are allowed (SQLite state is
captured consistently) and the active job ids are recorded in the
manifest as a warning. For a perfectly static file set, back up
while no worker is running.

## Retention policy

There is no automatic deletion. Snapshots and blobs are removed
only manually by the operator. Because blobs are shared between
snapshots, deleting one snapshot's manifest does not reclaim
blob space; reclaiming space requires deleting blobs that no
remaining manifest references, which the utility deliberately
does not automate.

## Restore procedure

Restore into a separate directory for inspection first:

```powershell
.venv\Scripts\python.exe scripts/backup_data.py restore `
  --snapshot 20261101T200910Z --target C:\restore-check
```

Rules enforced by the utility:

- Only `completed` snapshots can be restored; incomplete
  snapshots are refused.
- Every restored file is hash-verified after writing.
- Restoring into a non-empty directory requires both
  `--overwrite` and `--yes` (explicit confirmation). The utility
  never silently replaces the live database or storage tree.
- Restoration is a local file operation; it makes no Meta API
  calls.

To return a restored tree to service, stop the API and worker,
replace `runs_api` with the restored directory manually, and
restart. That step is deliberately manual.

## Verified results

Initial backup performed on 2026-11-01 (snapshot
`20261101T200910Z`): 1,273 files, 6.67 GB logical size, 2.49 GB
actual disk usage after content-addressed deduplication,
including 1,113 files of the completed Meta run
`run_446813e33927` (685 requests). All blob hashes verified
against the manifest. No jobs were active during the backup.

## Known limitations

- Deduplication keys on full-file SHA-256; a one-byte change in
  a large video stores a new full blob.
- File copies are read-only but not transactional: a file being
  actively written by a running pipeline may be captured
  mid-write. The manifest records active jobs so such snapshots
  are identifiable.
- The SQLite snapshot and file copies are taken sequentially,
  not at a single instant; cross-consistency between the
  database and very recent artifacts is best-effort.
- Restore materializes full copies; it does not hard-link blobs.
- Scheduling is external (Task Scheduler); the utility does not
  install or manage scheduled tasks.
