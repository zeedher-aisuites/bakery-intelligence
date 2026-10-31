# M2-06C.1 video reconstruction from recorded evidence

Status: implemented, awaiting host-side verification.
Offline only: no inference, network, or paid service is used.

## 1. Scope

`video_reconstruct` resolves a small uploaded manifest against an
explicit allowlist of recorded evidence sets, reads the recorded
files strictly read-only, assembles the equivalent offline bundle,
and runs the shared reconstruction core
(`offline.run_bundle_reconstruction`, also used by
`offline_reconstruct`). Historical experiment code (003, 006,
007) is untouched, including every P04-specific constraint.

Out of scope for C.1: generalized first-pass inference (C.2),
video intake and staging (C.3), verification wiring beyond
recorded reports (C.4), paid authorization (C.5), and any UI
change (C.6). The 16 MB upload cap is unchanged: the manifest is
a few hundred bytes and recorded evidence is never uploaded.

## 2. Manifest format (`bakery-video-manifest/1`)

```json
{
  "manifest_format": "bakery-video-manifest/1",
  "evidence_set": "p04_full_4096",
  "event_id": "scale_visibility_00007100_00007650",
  "video_id": "P04_104",
  "draft_version": 1,
  "source_dataset": null,
  "window": null
}
```

Required: `manifest_format` (exact match), `evidence_set`
(allowlisted name, never a path), `event_id`, `video_id`.
Optional: `draft_version` (integer >= 1, default 1),
`source_dataset` (string or null), `window` (numeric
start/end object or null). Unknown sets, malformed JSON, and
missing evidence fail the job with a message; nothing is
retried silently.

## 3. Evidence allowlist

`api.video.EVIDENCE_SETS` maps set names to paths relative to
the evidence root (worker `--evidence-dir`, default
`<cwd>/data/processed`). C.1 ships exactly one set:

| Role | Path |
|---|---|
| observations | `experiment_006/full_run/p04_full_4096/observations` (`batch_*.json`) |
| temporal | `experiment_006/temporal/p04_full_4096/temporal_events_1.1.0.json` |
| executions | `experiment_007/selective_verification/executions` (optional) |
| units | `experiment_008/reconstruction/units.json` (optional) |

Observations and temporal are required; executions and units
resolve to absent when missing and are recorded as absent (the
same tolerance the reconstruction loaders apply). New sets join
only through an explicit code change to the allowlist. User
input can never supply a path, so traversal is impossible by
construction; resolution additionally asserts containment.

## 4. Provenance

`video_summary.json` records, per consumed file: role,
repository-relative path, SHA-256, and byte size (or explicit
absence). It also records the manifest SHA-256, the evidence
set name, and the same three honesty flags as the offline
summary (`inference_performed`, `network_accessed`,
`paid_services_used`, all false for C.1: this run queries no
model even though the recorded evidence was model-produced).
Recorded files are hashed but never written, moved, renamed, or
deleted; the recorded-evidence test re-hashes every consumed
file after the run to prove it.

## 5. Equivalence with offline_reconstruct

For equivalent evidence, both workflows stage byte-identical
inputs (observations concatenate in sorted batch order;
measurement assembly sorts by frame regardless) through the
same shared core, so `measurements.json`, `validation.json`,
`weighing_event.json`, `action_events.json`, and
`units_applied.json` are byte-identical. Exactly two
run-specific differences exist by design and are asserted as
such: `recipe_draft.json` carries the job's own `run_id`, and
each workflow writes its own summary receipt
(`offline_summary.json` vs `video_summary.json`).

## 6. Lifecycle notes

- No schema change: C.1 executes synchronously inside one
  claim, so the existing queued/running/completed/failed
  states plus heartbeat suffice. Additive progress fields
  arrive with multi-stage C.2+ work, when they can be
  justified by real stages.
- Retry bound 2, same as the offline workflow: C.1 execution
  is deterministic with read-only inputs and idempotent
  artifact writes. This bound must be re-evaluated before any
  paid stage joins the `video_reconstruct` name.
- The HTTP API needs no change: `POST /api/analyses` already
  accepts an explicit `workflow` field, so `video_reconstruct`
  submits today with a manifest file as the source.
- The React console is unchanged in C.1: video runs submit at
  the HTTP API level, like offline bundles before M2-06B.2.

## 7. Verification status

Implemented and statically reviewed; Python is unavailable in
the agent sandbox, so the suite was authored but not executed
here. Host-side validation:

```powershell
uv run python -m pytest tests/test_api_video.py -v
uv run python -m pytest -q
uv run python scripts/sync_contract_schemas.py --check
git diff --check
```

Report whether `test_recorded_p04_evidence_reconstructs`
ran or skipped: it runs only when the recorded P04 set is
present and skips explicitly otherwise. (Frontend contracts
are unaffected, so no React gate is required for C.1.)

## 8. Limitations

- One hard-coded evidence set; arbitrary videos need C.2/C.3.
- Recorded frames (JPEGs) are not consumed or re-verified;
  only observation records, the temporal artifact, reports,
  and the sidecar flow into the run.
- Two temporal files exist side by side
  (`temporal_events.json`, `temporal_events_1.1.0.json`); the
  allowlist pins the explicitly versioned one.
- Worker and API share one evidence-root default
  (`./data/processed`); multi-host or cloud-backed evidence
  is out of scope.
