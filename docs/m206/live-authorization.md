# S5-A live authorization for Meta inference (prepared, never executed)

S5-A introduces the authorization machinery for a future
first paid call without performing one: a per-run grant
document, submit-time and execution-time enforcement, a
registered but grant-gated `video_pipeline_meta`
workflow, and safe credential handling. No live request
was made during S5-A, and none can occur without an
explicit operator-authored grant plus a worker-side key.

## 1. Authorization design

A grant (`bakery-billable-grant/1`,
`src/bakery_intelligence/api/authorization.py`) binds one
job submission to an exact billable configuration:
workflow, provider (`meta`), model (allowlist:
`muse-spark-1.3` only), `max_requests` (hard cap,
required), `token_budget`, `max_frames`, optional
timeout/store/cost-ceiling/rates/strict-cost controls,
operator identity, timestamp, and a confirmation
sentence. The sentence must exactly equal the text
derived from the grant's own fields:

```text
authorize video_pipeline_meta model muse-spark-1.3 up to 3 requests, 512 output tokens, 3 frames, operator op
```

(a single line; `docs/m206/smoke-grant.json` carries the
canonical text), so the operator demonstrably reviewed
the parameters.
Unknown keys are rejected: a mistyped constraint name
can never silently drop a billing bound. Grants never
carry credentials.

Enforcement points:

- Submit (`JobService.submit`, API `POST
  /api/analyses`): the meta workflow requires a
  structurally valid grant whose workflow matches
  (else HTTP 422, no job row); fake workflows reject
  any grant. The canonical grant persists in the job
  row (`grant_json`, added by idempotent migration)
  and is never served back (`to_dict` excludes it).
- Execution (`execute_video_pipeline_meta`): the
  stored grant is re-parsed and re-bound to the job
  (a tampered row fails), the key is loaded from the
  worker environment (`META_API_KEY`, else
  `MODEL_API_KEY`; missing means explicit failure
  before any client exists), forwarded memory-only
  through the pipeline into the runner
  configuration, and handed to the SDK factory at
  first submit. It is never written to state,
  checkpoints, reports, artifacts, or errors; job
  errors are key-scrubbed before persistence, and
  the runner config hides the key from `repr()`.
- Recovery: the meta workflow is listed in
  `BILLABLE_WORKFLOWS`, so stale jobs fail
  explicitly and are never requeued; S3
  single-submit invariants hold structurally.

Request caps vs cost ceilings: `max_requests` is a
hard pre-dispatch limit with lifetime accounting
across checkpoints and restarts (S1/S2 machinery).
`cost_ceiling_usd` with operator rates is an
estimated-spend control (reserve + overrun
detection), never a guaranteed maximum charge:
actual usage is unknown before dispatch, and
failed attempts may still bill per provider
policy (U12).

## 2. First live smoke-test procedure (future, authorized)

Prerequisites (all required before the first paid call):

1. S5-A host validation is green (full suite plus
   contract check) on the exact commit being run.
2. A human reviewer has approved this design and the
   specific smoke grant below.
3. `META_API_KEY` is exported in the worker process
   environment only; it appears in no file, log, or
   artifact. No other process shares the worker DB.
4. Exactly one worker instance serves the database
   (multi-worker fencing does not exist).
5. A tiny smoke clip is prepared (about 3 seconds,
   yielding at most 3 extracted frames), not the
   full 30-second demo. `max_frames` is a hard
   extraction bound, not truncation: a schedule
   larger than the grant fails with
   `TOO_MANY_FRAMES` before any inference, so the
   clip must fit the grant.
6. The provider account is funded and a spend alert
   is configured (operator duty, outside this repo).
7. U11–U20 below are acknowledged, with the stop
   conditions understood.

Smoke grant (3 requests maximum):

```json
{
  "grant_format": "bakery-billable-grant/1",
  "workflow": "video_pipeline_meta",
  "provider": "meta",
  "model": "muse-spark-1.3",
  "max_requests": 3,
  "token_budget": 512,
  "max_frames": 3,
  "operator": "op",
  "created_at": "2026-10-31T12:00:00+00:00",
  "confirmation": "authorize video_pipeline_meta model muse-spark-1.3 up to 3 requests, 512 output tokens, 3 frames, operator op"
}
```

Submit (from the repository root, API running):

```powershell
$grant = Get-Content -Raw docs/m206/smoke-grant.json
curl.exe -X POST http://127.0.0.1:8000/api/analyses `
  -F "file=@smoke_3s.mp4" `
  -F "workflow=video_pipeline_meta" `
  -F "grant=$grant"
```

Expected: HTTP 202 with a `run_id`; the worker
dispatches at most 3 frame requests, then the run
reaches `completed` with 7 artifacts
(`recipe_draft.json`, `measurements.json`,
`validation.json`, `weighing_event.json`,
`action_events.json`, `pipeline_summary.json`,
`firstpass_report.json`). Audit
`firstpass_report.json` (`billable: true`,
`requests_submitted`, usage totals, stopped
reason) and confirm the provider-side request
count matches before any larger run.

Failure handling:

- 422 at submit: grant malformed or mismatched; fix
  the document, no job exists.
- `TOO_MANY_FRAMES`: the clip schedules more frames
  than the grant allows; extraction refuses with
  zero submits. Trim the clip or re-authorize.
- `CREDENTIAL_MISSING`: no key in worker env; the
  job fails with zero submits.
- `AUTH_FAILED`: bad key or permissions; the run
  aborts on the first 401 with no further spend.
- Failed frames (e.g. HTTP 400): recorded
  explicitly as `missing` measurements; no retry.
- Crash mid-run: restart refuses uncertain batches
  (`UNCERTAIN_REQUESTS`); completed work is
  preserved, nothing replays.
- Request-cap stop: explicit `request_limit_reached`
  with remaining frames listed.

Stop / rollback: stop the worker process to halt
further claims (an in-flight HTTP request may still
complete; no retries or requeues follow). Revoke or
rotate the key if exposure is suspected. Failed or
stale meta jobs stay failed by design; resume only
via a new authorized submission, never by editing
the database.

## 3. Unresolved live assumptions (U11–U20)

Offline tests prove the adapter, authorization, and
accounting logic against doubles; they do not
establish live API compatibility. Verify before
real execution:

- U11: live acceptance of base64 image data URLs at
  pipeline frame sizes (proven only on experiment
  frames).
- U12: whether failed or rejected attempts are
  billed (provider policy; affects cost reading).
- U13: rate limits and burst tolerance for
  sequential frame requests (429 handling exists;
  limits unknown).
- U14: usage-field presence on incomplete or
  error responses (absence degrades to unknown
  cost by design).
- U15: `incomplete_details` reason vocabulary
  beyond token limits (other reasons end frames
  explicitly as unsupported).
- U16: per-request latency distribution for the
  120 s timeout at pipeline scale.
- U17: `store=true` server-side retention and
  data handling.
- U18: continued availability of the
  `muse-spark-1.3` model id.
- U19: content-policy refusal shape for kitchen
  video frames.
- U20: single-worker assumption for the meta work
  directory (the S2 multi-worker race persists).

## 4. Verification status

Implemented and statically reviewed; Python is
unavailable in the agent sandbox, so the suites
were authored but not executed here. Host-side
validation:

```powershell
uv run python -m pytest tests/test_api_authorization.py tests/test_pipeline_meta.py tests/test_inference_meta_spark.py -v
uv run python -m pytest -q
uv run python scripts/sync_contract_schemas.py --check
git diff --check
```

Expected: all authorization, integration, and
adapter suites pass with zero network access, the
full suite passes, and all 9 contract schemas
still match. No live call was made or is
scheduled by any suite.
