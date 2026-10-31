# M2-06C.2 generalized first-pass inference

Status: implemented, awaiting host-side verification.
Offline by construction: the module performs no network
access and wires no live client.

## 1. Component

`src/bakery_intelligence/inference/firstpass.py` runs
first-pass multimodal analysis over an explicit frame list.
Configuration supplies frames (`frame_index`, `image_path`,
timestamps), a source integrity anchor, the model name, prompt
text, batching, token budgets, retry policy, an optional rate
pair, optional request/cost guardrails (S1), and a client
factory. The client seam is the
`InferenceClient` protocol — `submit(prompt, images, key,
max_output_tokens)` returning a normalized response — modeled
on `MetaClient.submit`, except the token budget travels as an
explicit per-request parameter instead of mutable client
state, so escalation is observable at the seam and a future
live adapter stays mechanical.
Live provider wiring is out of scope until the C.5
authorization boundary; a missing factory fails closed.

## 2. Behavior contract

- Frames sort by `(timestamp_seconds, frame_index)`; duplicate
  indices fail closed before any request.
- Every image is validated upfront (readable file, JPEG or PNG
  magic bytes) and hashed; validation failures make zero
  requests.
- One image per request (the proven 006 default); contiguous
  batches named `batch_<first08d>-<last08d>`; per-batch
  `raw_responses/` and `observations/` files.
- Transport retries honor `retry_after_seconds`, then
  exponential backoff with jitter; abort codes
  (`CREDENTIAL_MISSING`, `AUTH_FAILED`) propagate, preserving
  already-checkpointed batches.
- Token budgets escalate only on explicit token exhaustion
  (`decide_attempt` semantics); each frame ends in one of the
  seven lifecycle terminal states.
- Failed frames get raw error records plus terminal counts but
  no observation; the batch is marked `needs_human_review`.
  Empty, invalid, and abstained outcomes keep explicit
  records with null fields, never manufactured successes.
- Checkpoints append JSONL per batch (with exact frame lists);
  run state and the final report write atomically. Resume skips
  `completed` batches; `needs_human_review` batches re-run as a
  unit (re-submitting their known-good requests, as in 006).
  Budget-halted batches checkpoint as `budget_stopped` with
  their partial outputs and resume on the remainder only, so
  completed frames are never re-submitted. Reusing a directory
  for a different frame set fails closed
  (`CHECKPOINT_MISMATCH`) instead of mixing outputs.
- Request/cost guardrails (S1, all optional, off by default):
  `max_requests` caps submitted requests exactly (retries and
  escalations included); `cost_ceiling_usd` halts further
  spend once known spend exceeds it; `strict_cost` additionally
  halts on the first unknown-cost submit. Every submit without
  usable usage counts as unknown cost, never as zero, and
  unknown totals poison the reported estimate while the known
  part stays visible. Halts record a stable reason, the
  remaining frame indices, and split this-run/prior counters;
  nothing is silently discarded. Resume seeds its accounting
  from checkpoints (recorded blocks preferred, pre-S1 token
  totals reconstructed and labeled as such), so limits apply
  to lifetime totals. The pre-dispatch cost reserve is a
  heuristic (largest observed single-request estimate):
  input tokens are unknowable before dispatch, so overruns
  are detected after the fact, never prevented outright;
  the report states this gap explicitly.
- Observation records carry the 006 key set (frame identity,
  three timestamps, basis, source anchor, batch/pass ids,
  call provenance, model fields, parse diagnostics, lifecycle
  evidence, image hash), consumable by the existing temporal
  and reconstruction loaders unchanged.

## 3. Deliberate deviations from Exp. 006

| Area | 006 (frozen) | This module |
|---|---|---|
| Manifest | exact 685-row P04 manifest | caller frame list, any count/order |
| Images | JPEG magic only | JPEG or PNG magic, encoding recorded |
| Requests | 1/request default, gated multi-image | 1/request only |
| Rates | hard-coded Spark/Wonder prices | caller-supplied or unknown |
| Budgets | `max_output_tokens` interplay | `token_budgets` only |
| Client | `MetaClient` + SDK factory | injected `InferenceClient` protocol |
| Passes | primary + review | first pass only |

`manifest_ref` keeps its historical key name (with
`manifest_sha256`) because downstream loaders require it;
for non-manifest sources the caller anchors an equivalent
source hash. The mirrored policy blocks cite their 006
reference at each site; if 006 changes, this module does not
follow automatically.

## 4. Tested claims (and only these)

The suite (`tests/test_inference_firstpass.py`, scripted fake
client, zero network) covers variable counts, non-contiguous
indices, deterministic ordering, batch edges, upfront
validation with zero-call proofs, retry/backoff/`Retry-After`,
abort propagation with partial durability, budget escalation
and recovery, exhausted budgets, all terminal classes,
partial-batch reporting, resume without re-calling completed
batches (asserted on the fake call log), refusal to resume a
different frame set, provenance fields,
cost known/unknown reporting, PNG acceptance, and the S1
guardrails: request-cap halts (including zero, retries, and
exact completion), reserve and overrun ceiling halts,
strict unknown-cost halts, poisoned-vs-known spend splits,
zero-ceiling overrun semantics, superseded-usage spend
inclusion, resume accounting with partial-batch merge,
pre-S1 checkpoint reconstruction, invalid-limit rejection,
and vacuous strict-without-ceiling behavior. Crash
safety beyond tested resume is not claimed.

## 5. P04 compatibility

`test_recorded_raw_replay_matches_observation_shape`
replays the five recorded model replies (plus usage) from
`experiment_006/.../raw_responses/batch_00000000-00000200.json`
through the new runner and compares against the recorded
observations: record keys are a superset, model fields are
equal, identity/timestamp/batch/pass/image fields match, and
terminals are valid or abstained per the mirrored policy.
This proves structural and parse-path compatibility. It does
not prove a live model would emit the same text today, and it
exercises one batch, not the full 685-frame run. The test
skips explicitly when recorded data is absent.

## 6. Verification status

Implemented and statically reviewed; Python is unavailable in
the agent sandbox, so the suite was authored but not executed
here. Host-side validation:

```powershell
uv run python -m pytest tests/test_inference_firstpass.py -v
uv run python -m pytest -q
uv run python scripts/sync_contract_schemas.py --check
git diff --check
```

Report whether the recorded-replay test ran or skipped.

## 7. Limitations

- No live client, ingestion, extraction, worker wiring, UI,
  or cloud support (later milestones).
- `needs_human_review` batches re-submit known-good requests
  on resume; acceptable offline, cost-relevant when live.
  Their prior submits do not seed the S1 ledger (S1 seeds
  only batches that will not re-run), so a resumed review
  batch spends fresh budget on already-billed frames; the
  re-spend is counted, never hidden.
- A crash between submit and batch checkpoint leaves that
  batch's spend outside every checkpoint, so a later resume
  cannot account for it; strict cross-process durability is
  S2 scope, explicitly not claimed here.
- Pre-S1 checkpoints seed request counts exactly but cost
  only from batch token totals, flagged
  `reconstructed_from_token_totals`.
- Strict mode cannot prevent a first-request overrun: with
  rates known it dispatches (input cost unknowable
  pre-request) and halts after detection; only unknown
  costs halt before further spend.
- Mirrored 006 policy can drift from the frozen experiment;
  the replay test guards shapes, not future 006 edits.
