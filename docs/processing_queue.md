# Processing Queue, Upload Limits, and Job Lifecycle

## Architecture

- Job state is persisted in SQLite (`SqliteJobStore`) and remains
  the single source of truth. The frontend never duplicates it.
- `GET /api/jobs?status=&limit=` returns persisted jobs across
  queued, running, completed, and failed states, newest first
  (limit 1-200). `GET /api/analyses` keeps its completed-only
  contract for the review library.
- Job payloads include `error_category` (`validation` or
  `processing`) and `claimed_by`, so the UI can distinguish a job
  waiting for a worker from a job a worker has claimed.
- The Processing Queue view polls every 3 seconds while mounted,
  cleans up its timer on unmount, keeps the last known state when
  the backend is temporarily unavailable, and links completed
  jobs to the evidence review workspace.

## Size limits

- One effective video limit is resolved in
  `bakery_intelligence.api.limits`: `BAKERY_PIPELINE_MAX_VIDEO_BYTES`,
  then `BAKERY_MAX_VIDEO_UPLOAD_BYTES`, then 2 GiB.
- The upload endpoint enforces the effective (stricter) cap while
  streaming to disk, and job creation from a staged upload
  re-checks the pipeline cap before any job row exists. Oversized
  videos receive HTTP 413 with the actual and limit byte counts;
  no job is created.
- Pipeline executors run from disk (`run_pipeline_from_path`):
  size comes from file metadata and SHA-256 is streamed, so the
  full video is no longer held in memory by the worker path. The
  bytes-based `run_pipeline_offline` entry point remains for
  compatibility with existing callers and tests.

## Retries and billable safety

- `POST /api/jobs/{run_id}/retry` requeues one failed job in
  place through a single conditional UPDATE. The original record,
  source, and evidence are preserved.
- Billable workflows (`video_pipeline_meta`,
  `video_pipeline_multimodal`) are refused with HTTP 409 at
  both the service and store layers. Retrying over HTTP never
  invokes inference and never replays a grant. The only
  recovery path for a failed paid run is the explicit worker
  CLI (`--recover <run_id> --recover-note ...`), which
  requeues conditionally, reuses persisted stage outputs
  without rebilling them, and records the operator note.
- Retries stop at 3 total attempts. Stale-job recovery and atomic
  claiming semantics are unchanged.

## Known limitations

- Progress remains truthful but coarse: stage and frame counts
  come only from the existing progress endpoint and durable
  pipeline state. No percentages are invented when frame totals
  are unknown.
- Frame-count limits (1,000 sampled frames) still apply and are
  reported separately from byte-size limits.
- Very large videos still require enough local disk for staging,
  extracted frames, and inference checkpoints.

## Verification

Offline only: isolated SQLite databases, synthetic videos, and
deterministic fake inference. No Meta credits are consumed by
the test suites covering this behavior.
