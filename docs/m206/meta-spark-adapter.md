# S4-A Meta Muse Spark 1.3 adapter (offline-tested, no live execution)

`src/bakery_intelligence/inference/meta_spark.py` adapts the
generalized `FirstPassRunner` seam to Meta Model API's
Responses interface for `muse-spark-1.3`. Every behavior is
exercised through an injected SDK double in
`tests/test_inference_meta_spark.py`; no test contacts the
network, and S4-A wires no live path.

At S4-A time paid execution was unavailable: no
billable API workflow existed, the worker billable
registry was empty, and key provisioning plus
authorization grants remained S5 duties. S4-A
validates the adapter contract and the S3 safety
posture a future integration must satisfy. (S5-A
adds the grant mechanism and the authorized
`video_pipeline_meta` workflow; see
`docs/m206/live-authorization.md`.)

## 1. Verified contract elements

Each element below is triangulated from read-only
repository evidence. Paths are relative to the repo root.

| # | Element | Value | Evidence |
|---|---|---|---|
| V1 | SDK base URL | `https://api.meta.ai/v1` | `experiments/001_meta_api/check_connection.py`, `experiments/004_baseten_api/check_meta_vision.py`, `experiments/006_p04_multimodal_inference/meta_client.py` |
| V2 | Model id | `muse-spark-1.3` | Same three files, plus recorded `model` echo in `data/processed/experiment_006/full_run/p04_full_4096/raw_responses/` |
| V3 | Auth | Bearer API key (`OpenAI(api_key=...)`; raw `Authorization: Bearer` for `/v1/models`) | Exp. 001 (both forms), `meta_client.py` (`list_models`) |
| V4 | Request shape | `responses.create(model, input=[{"role": "user", "content": [{"type": "input_text", ...}, {"type": "input_image", "image_url": "data:<mime>;base64,..."}]}])` | `check_meta_vision.py`, `meta_client.py` |
| V5 | Token budget | Per-request `max_output_tokens` (1024 locally proven in 001/004; 4096 used by the 006 full run) | Exp. 001, `meta_client.py` |
| V6 | No SDK retries | `max_retries=0` on every SDK construction | Exp. 001, `check_meta_vision.py`, `meta_client.py` |
| V7 | Response surface | `status`, `output_text`, `model`, `usage`, `id`, `incomplete_details`, `output` attributes | Exp. 001 diagnostics, `meta_client.py` normalization |
| V8 | Usage fields | Integer `input_tokens` / `output_tokens` / `total_tokens` | Recorded P04 raw responses (e.g. 2913/1240/4153) |
| V9 | Status-to-code map | 401 auth, 429 rate-limit (+`retry-after`), 400 bad request, 413 payload, 5xx server, timeout/connection names | `meta_client.py` `_map_exception` (code-verified; live behavior inherited) |

The adapter reproduces V4-V6 and V9 call-for-call and
normalizes V7-V8 into `InferenceResponse`. The 50-image
cap and `store` flag are passed through from 006 (see
U3/U4).

## 2. Deliberate deviations from Exp. 006

- MIME is sniffed from magic bytes (JPEG/PNG) instead of
  hard-coding `image/jpeg`; anything else raises
  `UNSUPPORTED_IMAGE` before dispatch. This fixes the
  PNG-as-JPEG mislabeling in the historical client.
- The token budget travels per request (the generalized
  seam) instead of mutable client state.
- Non-integer token counts degrade to unknown (`None`)
  instead of propagating into cost accounting; values
  are never invented or string-coerced.
- A missing or empty model echo falls back to the
  configured model, keeping the required `model: str`
  field truthful.
- `raw_status_details` is not carried: the generalized
  `InferenceResponse` schema has no such field, and the
  runner consumes only `incomplete_reason`. The adapter
  does not extend the shared schema.

## 3. Unresolved uncertainties

- U1: Raw HTTP wire format (endpoint path, method,
  envelope) is unverified; the adapter uses the SDK
  call shape only and never raw HTTP.
- U2: `store` server-side semantics and retention are
  unknown; the flag is recorded as provenance.
- U3: The 50-image cap is inherited from a 006 comment
  claiming verified docs; enforced client-side either
  way (a wrong cap either under-utilizes or meets an
  explicit server 400).
- U4: Usage detail field names (`cached_tokens`,
  `reasoning_tokens`) are inherited from 006
  normalization and read defensively.
- U5: No provider pricing is shipped; spend estimates
  need operator-supplied rates (S1 semantics).
- U6: `incomplete_details` beyond `.reason` is not
  consumed.
- U7: Billing-specific statuses (e.g. 402) are
  unobserved and map to `PROVIDER_ERROR` (explicit
  frame failure, no retry); revisit if documented.
- U8: The 120 s timeout default comes from the 006
  full run (001 used 30 s, 004-meta 60 s);
  configurable per client.
- U9: Multi-image requests are supported by the seam
  but the runner always sends single-frame requests.
- U10: SDK exception names beyond 001's verified
  `APIStatusError`/`APIConnectionError` imports are
  inherited from 006's mapper.

## 4. Safety assessment for S4-B integration

Holds today (offline-proven):

- `billable_runner_config` fixes `billable=True`,
  `max_attempts=1`, one token budget, and
  `uncertain_policy="refuse"` structurally; the S3
  runner validation rejects anything else before any
  dispatch.
- One logical frame submits at most once per run:
  no transport retry, no escalation replay, no
  uncertain replay.
- The default SDK factory sets `max_retries=0`; the
  adapter itself never loops over a dispatch.
- Auth failures abort the run (`AUTH_FAILED`) instead
  of burning further frames; unknown costs never read
  as zero.
- Stale billable jobs fail explicitly at the job
  layer (S3 registry), never requeue.

Required before any live call (not in S4-A):

- S5 authorization grants and key provisioning; the
  factory takes `api_key` as an explicit parameter
  and performs no environment lookup.
- A registered billable workflow plus worker
  registry entry; the offline pipeline still uses
  the deterministic fake exclusively.
- Operator-supplied rates and a cost ceiling if
  spend bounds are wanted; no hard USD ceiling is
  claimed while usage is unknown pre-dispatch.
- Confirmation of U1-U10 against current provider
  behavior, especially billing statuses (U7) and
  the image cap (U3).

## 5. Verification status

Implemented and statically reviewed; Python is
unavailable in the agent sandbox, so the suite was
authored but not executed here. Host-side validation:

```powershell
uv run python -m pytest tests/test_inference_meta_spark.py -v
uv run python -m pytest -q
uv run python scripts/sync_contract_schemas.py --check
git diff --check
```

Expected: all new adapter tests pass with zero
network access, the full suite passes, and all 9
contract schemas still match.

## 6. Pipeline integration (S4-B, offline only)

`run_pipeline_offline` accepts an explicit
caller-supplied `billable_client` (S4-B:
MetaSparkClient with a test-double SDK only). The
seam runs the runner through `billable_runner_config`
with the pipeline pass id, batch size, and a single
caller-chosen token budget, so S3 single-submit
invariants hold structurally. At most one client
override (`fake` or `billable_client`) is allowed;
no registered workflow supplies the seam, and the
default fake path is behaviorally unchanged.

The durable fingerprint distinguishes the full
execution policy: provider, model, pass id, batch
size, token budgets, prompt hash, request/cost
guards, `max_attempts`, `billable`, and
`uncertain_policy`. Cross-path resume (fake state
under a billable client or vice versa) refuses with
a fingerprint diff, as do pre-S4-B states without
the `max_attempts` entry. The summary keeps its
shape and flags (`synthetic_inference: true`, no
model queried, no network, no paid services — all
true while only test doubles run here); identity
keys carry the actual client values and the note
states the caller-supplied test-double posture.

Verified through `tests/test_pipeline_meta.py`
(fake SDK exclusively): end-to-end artifacts with
usage totals, provenance, timestamps, and
request-id linkage; custom budget plumbing;
ambiguous-client and bad-budget rejection before
any state write; exact fingerprint contents;
cross-path refusal both directions; uncertain
post-crash refusal with zero new submits;
certain-failure preservation without replay (a
completed first pass reuses its verified outputs,
so failed frames stay explicitly failed);
unknown-cost accounting without fabrication; and
a registry-enumerating safety test proving no
registered workflow can trigger MetaSparkClient.

Remaining blockers for first authorized live
inference: S5 authorization grants and key
provisioning; a registered billable workflow plus
worker registry entry; operator rates and cost
ceiling; U1–U10 confirmation against current
provider behavior; an explicit authorized retry
mechanism for certainly-failed frames (none
exists; resume never re-dispatches them);
summary-flag and identity-key contract revision
for live runs (the S4-B flags assert
synthetic/test-double execution and must become
transport-aware); and frontend handling for
billable summaries.
