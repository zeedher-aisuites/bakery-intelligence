# M2-06C.4 offline video pipeline

Status: implemented, host-verified through M2-06C.5/S1; S2
durable recovery implemented, awaiting host-side verification.
Entirely offline: no model is queried and no network is used.

## 1. Workflow

`video_pipeline_offline` accepts one small uploaded video
(max 16 MB, enforced by both the API cap and the worker) and
runs, inside one worker claim over a durable run-scoped
work directory (`runs/<run_id>/work`, outside the public
artifacts namespace):

1. Integrity-checked staging (`video.intake`, content hash).
2. Frame extraction at 1 FPS over the full window
   (`video.extract`, max 1000 frames).
3. First-pass inference with the deterministic fake client
   (`inference.fake`, batch size 5).
4. Evidence transformation into `pipeline-evidence/1`:
   observations pass through, one window-spanning temporal
   candidate is synthesized and marked
   `pipeline_synthetic_stub`, no verification reports, no
   units.
5. Deterministic reconstruction through the shared
   `offline.run_bundle_reconstruction` core.

Intermediates (staged copy, frames, first-pass files,
`pipeline_state.json`) persist in the work directory across
worker restarts. Only completed runs publish artifacts: the
five reconstruction files plus `firstpass_report.json` and
`pipeline_summary.json`.

## 2. Durable recovery (S2)

Stage transitions record atomically in `pipeline_state.json`
(`pipeline-workdir/1`): source identity, video metadata,
extraction fingerprint (sampling, frame list, manifest and
per-frame hashes, sidecar hash), the effective first-pass
configuration, and the stage (`extracted`,
`firstpass_running`, `firstpass_done`).

A restarted worker verifies the state fingerprint against
its live inputs and resumes instead of restarting:
verified extraction outputs are reused, the first-pass
runner resumes from its own checkpoints (completed requests
are never re-submitted; uncertain batches follow the
offline rerun policy), and reconstruction re-runs
deterministically. Any fingerprint mismatch, missing or
changed frame, altered report, or unknown stage fails the
job explicitly (`PipelineResumeError`, never silently
retried) instead of restarting onto doubt. The summary
records whether the run resumed (`resumed`); first-pass
counts in the summary are lifetime totals derived from
durable files, while `firstpass_report.json` keeps the
per-attempt report with its prior/current budget split.

## 3. Honesty contract

- The workflow name, the summary (`synthetic_inference:
  true`, fake model/provider ids, three false flags), the
  first-pass report, and the console radio label all state
  that outputs are synthetic test vectors.
- Missing or failed frames become explicit `missing` /
  `none` measurement entries; total inference failure fails
  the job instead of reconstructing from nothing.
- The temporal candidate is a marked placeholder, not an
  analysis result; units, ingredients, actions, and tare
  stay unknown unless evidence supports them.
- Same video bytes yield byte-identical semantic artifacts
  (measurements, validation, weighing, actions, draft modulo
  run_id). Provenance legitimately differs per upload: the
  staged source name embeds the upload filename, so
  `firstpass_report.source_name` and the bundle fingerprint
  vary across upload names while all reconstructed values
  stay identical (asserted exactly by test).

## 4. Lifecycle and limits

- Synchronous single-claim execution; artifacts store only
  on success, so no partial sets exist. A daemon thread
  refreshes the claim heartbeat while the executor runs,
  so slow jobs are not mistaken for crashed workers.
  Crashes requeue under bound 2 and the next attempt
  resumes verified progress; explicit failures never retry.
  The bound must be re-evaluated before any paid stage
  joins this workflow name.
- Fixed worker settings in C.4: 1 FPS, full window,
  batch size 5, 16 MB video cap, 1000-frame cap. The
  underlying functions accept configuration; worker-level
  configurability arrives with later intake work.
- No schema change: the four job states plus heartbeat
  suffice. The HTTP API needs no change (generic workflow
  field). The console gained a radio option and renders
  pipeline results through the existing review section.

## 5. Verification status

Implemented and statically reviewed; Python is unavailable in
the agent sandbox, so the Python suite was authored but not
executed here. The frontend type-check passed in-sandbox
(`tsc --noEmit`, exit 0). Host-side validation:

```powershell
uv run python -m pytest tests/test_api_pipeline.py -v
uv run python -m pytest -q
uv run python scripts/sync_contract_schemas.py --check
cd frontend; npm test; npm run build
git diff --check
git status --short
```

## 6. Limitations and crash windows

- Missing-observation timestamps use the measured source
  frame rate carried through the bundle (S0); the legacy
  50 FPS divisor applies only when the source rate is
  genuinely unknown (bundles without `source_fps`).
- JPEG bytes depend on the OpenCV build; selection,
  metadata, and fake outputs do not.
- No multi-gigabyte intake, no live client, no
  authorization boundary, no cloud execution (C.5+).
- Review UI shows pipeline drafts through the offline
  results view; pipeline-specific sections await later
  milestones.
- Crash windows (all fail safe, none silent): a crash
  before the state file exists restarts the free
  stage/extract stages; a crash mid-batch leaves that
  batch uncertain (offline reruns it, bounded by one
  batch; a paid future must refuse and resolve
  explicitly); files written without their checkpoint
  are superseded on rerun, with the replayed frames
  named in the first-pass report. Durability covers
  process crashes; OS-level crashes may need fsync
  hardening (not implemented).
- Work directories are retained after completion for
  audit; no automatic cleanup exists yet. A second
  worker claiming the same job past the stale
  threshold while the first still runs would share
  one work directory; heartbeats make that require a
  genuinely stalled worker, and single-worker
  deployments never hit it.
