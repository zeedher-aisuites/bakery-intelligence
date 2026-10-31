# M2-06B.1 offline pipeline integration

Status: implemented, awaiting host-side verification.
All behavior below is deterministic and offline unless a section
says otherwise.

## 1. Scope

`offline_reconstruct` is the second API workflow. A queued job
consumes one uploaded JSON bundle of existing analysis artifacts
and executes the deterministic reconstruction stage
(`bakery_intelligence.reconstruction.cli.reconstruct`). The stage
performs no inference, touches no network, and uses no paid
service; the run summary artifact states all three explicitly.

Explicitly out of scope: frame extraction (needs an `ffmpeg`
adapter, scoped but not built), any model inference (Baseten,
Modal, selective verification), and server-side references to
repository or `data/` artifact sets by name. The bundle travels
with the job, so every offline run reproduces from its stored
source bytes alone.

## 2. Workflow selection

`POST /api/analyses` accepts a multipart `workflow` form field:

| `workflow`            | Source content      | Result                                    |
|-----------------------|---------------------|-------------------------------------------|
| `storage_smoke`       | any small file      | integrity receipt, no analysis            |
| `offline_reconstruct` | JSON offline bundle | recipe draft plus evidence artifacts      |

The field defaults to `storage_smoke`, so existing smoke clients
are unchanged. Unknown workflows return HTTP 422 and are never
executed. The worker dispatches strictly on the stored
`record.workflow`; a smoke result can never be relabeled as a
reconstruction because each workflow writes its own artifact set
with its own workflow name embedded.

## 3. Offline bundle format (`bakery-offline-bundle/1`)

```json
{
  "bundle_format": "bakery-offline-bundle/1",
  "video_id": "P04_104",
  "event_id": "scale_visibility_test",
  "draft_version": 1,
  "source_dataset": null,
  "window": null,
  "observations": [
    {"frame_index": 7400, "timestamp_seconds": 148.0,
     "model_fields": {"scale_display_verbatim": "353", "...": "..."}}
  ],
  "temporal": {"artifact_version": "1.1.0", "events": ["..."]},
  "execution_reports": [],
  "units": null
}
```

Required: `bundle_format` (exact match), `video_id`,
`event_id`, non-empty `observations`, `temporal` object.
Optional: `draft_version` (integer >= 1, default 1),
`source_dataset` (string or null), `window` (object with
numeric `start`/`end` or null), `execution_reports` (list of
report objects, each with an `event_id` string),
`units` (sidecar object with an `assignments` list or null).

Validation is two-layered. The API checks structure and scalar
types (`OfflineBundleError`); the reconstruction loaders check
domain rules and their structured `ReconstructionError` codes
(`UNKNOWN_EVENT`, `NO_MEASUREMENTS`, `WINDOW_INVALID`,
`OBS_INVALID`, `ARTIFACT_INVALID`, `UNITS_INVALID`) surface
unchanged as the job error. Either failure marks the job
`failed` with a message; no partial artifact set is recorded.

## 4. Artifacts

A completed run stores the reconstruction CLI outputs unchanged
(`recipe_draft.json`, `weighing_event.json`,
`action_events.json`, `measurements.json`, `validation.json`,
plus `units_applied.json` only when the bundle carried a
sidecar) and one API-level receipt, `offline_summary.json`:

- workflow and run identity, bundle SHA-256, event and video
- `inference_performed: false`, `network_accessed: false`,
  `paid_services_used: false`
- draft id, measurement/action/unresolved/issue counts
- an explicit note that the draft needs human review

Every artifact is verified by read-back hash before the job may
complete (existing `_store_verified` path, shared by all
workflows).

## 5. Failure and retry semantics

- Malformed bundles, unknown events, missing sources, and
  reconstruction domain errors fail the job explicitly with a
  message (truncated to 2000 characters by the store).
- Explicit failures never retry automatically.
- A crashed worker's `running` job is recovered by the next
  pass under the same bound as the smoke workflow
  (`WORKFLOW_MAX_ATTEMPTS`), which is safe because offline
  execution is deterministic and artifact writes are
  idempotent.
- Paid inference keeps its default of zero requeue attempts;
  this workflow adds no paid path.

## 6. Verification status

Implemented and reviewed against existing code; Python is
unavailable in the agent sandbox, so the suite below was
authored but not executed here. Host-side validation:

```powershell
uv run python -m pytest tests/test_api_offline.py -v
uv run python -m pytest tests/test_api_app.py tests/test_api_jobs.py tests/test_api_storage.py tests/test_api_worker.py -q
uv run python -m pytest -q
uv run python scripts/sync_contract_schemas.py --check
git diff --check
```

End-to-end HTTP validation of the new workflow (M2-06B.1b)
uses the example bundle
`fixtures/offline/offline_bundle_http_example.json`
(synthetic evidence, 5 frames, verified readings, one human
unit assignment). With the backend and worker running (see
`docs/m206/spike-setup.md`):

```powershell
$created = curl.exe -s -F "file=@fixtures/offline/offline_bundle_http_example.json" -F "workflow=offline_reconstruct" http://localhost:8000/api/analyses | ConvertFrom-Json
$runId = $created.run_id
do { Start-Sleep -Seconds 2; $status = Invoke-RestMethod "http://localhost:8000/api/analyses/$runId" } while ($status.status -in @('queued', 'running'))
$status | Format-List run_id, workflow, status, source_sha256, artifact_ids, error
Invoke-RestMethod "http://localhost:8000/api/analyses/$runId/artifacts/offline_summary.json"
```

then fetch the remaining artifacts and compare
`summary.bundle_sha256` against the local file hash. Host-side
execution of this flow is the M2-06B.1b acceptance gate.

## 7. Limitations

- The React console submits smoke tests only; offline bundles
  are submitted at the HTTP API level in this milestone.
- Bundle size is bounded by the existing 16 MB upload cap.
- No provenance check links bundle contents to canonical
  Observation or FrameEvidence stores; reconstruction treats
  them as experiment-scoped inputs, as documented in
  `validate_bundle`.
- A `validation.json` issue count above zero still completes
  the job; issues are reported, not treated as failures.
