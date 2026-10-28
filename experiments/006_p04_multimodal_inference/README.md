# Experiment 006 — P04 Full-Video Multimodal Inference

## Objective

Reproducible, resumable multimodal inference over all 685 extracted
frames of P04_104 (`global_1fps_rerun`, 1 FPS, indices 0–34200).
Primary pass: Meta Muse Spark 1.3 Standard. Selective second pass:
Meta Muse Wonder 1.0 over difficult intervals. Predictions are
experiment JSON with timestamp provenance; they are not canonical
ledger records and never claim ground truth.

Status vocabulary: IMPLEMENTED (code written), LOCAL-VERIFIED (ran on
this host), API-VERIFIED (confirmed by a provider response),
MEASURED (recorded numbers), ESTIMATED (rate-derived), UNTESTED.

## Verified API contract

Researched 2026-10-28 against official Meta Model API documentation
(`https://dev.meta.ai/docs/...`) and the repository's own Exp. 001/004
scripts (which ran successfully against the live API).

| # | Claim | Status |
|---|---|---|
| 1 | Base URL `https://api.meta.ai/v1`, `POST /v1/responses` | LOCAL + DOCS |
| 2 | Bearer auth; key passed explicitly (SDK does not read `MODEL_API_KEY`) | LOCAL + DOCS |
| 3 | Model IDs `muse-spark-1.3`, `muse-spark-1.3-contributor`, `muse-wonder-1.0` (Standard only) | LOCAL + DOCS |
| 4 | Spark 1.3 recommended; text/image/video/audio/PDF in, text out | DOCS |
| 5 | 1,048,576-token context, both models | DOCS |
| 6 | Standard $1.25 in / $4.25 out (Spark), $4.00 in / $20.00 out (Wonder) per 1M; Standard content not used for training | DOCS |
| 7 | Rate limits per team: 3000 RPM / 4,000,000 TPM (Standard) | DOCS |
| 8 | Responses `input_image` with plain-string `image_url` (data URL or public URL) | LOCAL + DOCS |
| 9 | Multi-image supported, hard max 50/request (HTTP 400 above) | DOCS ONLY |
| 10 | ~50 MB inline-media bound; `store=true` large payloads can 413 | DOCS ONLY |
| 11 | Usage object `{input_tokens, output_tokens, total_tokens, details}` | LOCAL + DOCS |
| 12 | Per-image tokens NOT fixed; measure via `POST /v1/responses/input_tokens` | DOCS |
| 13 | `input + max_output` must fit context; `max_output_tokens` min 16 | DOCS |
| 14 | 429/5xx retryable with backoff + `Retry-After`; 400/401/413 never retried unchanged | DOCS |
| 15 | Completion signal `status == "completed"` | LOCAL + DOCS |

Explicit unknowns: numeric `max_output_tokens` maximum (1024/4096
locally proven; default here is 1024); P04 per-image token cost
(measure in calibration); exact sync body cap; server time limit;
multi-image requests never sent from this repo (default stays
1 image/request until a probe lands).

## Architecture

- `meta_client.py`: Responses client, env keys (`META_API_KEY`, then
  `MODEL_API_KEY`), base64 data-URL images, normalized results, named
  error codes, injectable client factory (mock seam), zero-generation
  `list_models` preflight.
- `prompts.py`: frozen primary/review prompts with SHA-256 pinning,
  defensive JSON parsing (unparseable replies preserved, never
  invented), observation builder with per-pass field schemas, review
  prompt formatter (primary JSON attached), deterministic selection.
- `runner.py`: manifest validation (exact header, 685 rows anchored at
  0..34200, step-50 indices, `timestamp == index/50`, JPEG magic
  bytes), contiguous batches, bounded retries (max 4, base 0.5 s ×2 +
  jitter, `Retry-After` honored; 401/400/413 never retried),
  multi-image requests gated behind an explicit opt-in flag,
  append-only JSONL checkpoints + atomic `run_state.json`, per-model
  cost aggregation (Spark $1.25/$4.25, Wonder $4.00/$20.00).
- `run_inference.py`: `preflight`, `calibrate`, `run`, `select-review`,
  `review`, `report` subcommands.

Batch completion requires raw responses, parsed observations, and the
checkpoint record all durable. Resume replays checkpoints and never
re-submits completed batches. Uncertain outcomes are recorded as
`needs_human_review` gaps, never silently completed.

## Observation contract

Every prediction carries `frame_index`, `image_path`, the three
manifest timestamps verbatim, `timestamp_basis: frame_index_over_fps`,
manifest hash, batch/range/pass IDs, and full call provenance. Model
fields stay visible-evidence-only with explicit nulls; display strings
are never converted to quantities; contradictions are preserved.

## Review pass

Selection is deterministic: a frame is selected when any trigger
holds — unparseable reply, `scale_readable == false`, visible tare
action, container-change hint, or low confidence. Output: sorted
unique indices. Each review request carries the frame image plus its
primary observation JSON (hashed as `review_of_sha256`), uses exactly
1 image per request, and parses the review schema (`agree`,
`corrected_fields`, `missed_evidence`, `contradiction_flag`,
`confidence`). Primary and review outputs live in separate run
directories and are never merged silently.

## Runtime layout (all gitignored)

- `data/processed/experiment_006/calibration/<name>/`
- `data/processed/experiment_006/full_run/<name>/`
  (`raw_responses/`, `observations/`, `checkpoints/`, `reports/`)
- `data/processed/experiment_006/review/<name>/`

## Local procedures (PowerShell, from repo root)

```powershell
# 1. Key (session-scoped, never on disk; 5.1-compatible)
$sec = Read-Host 'META_API_KEY' -AsSecureString
$env:META_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))

# 2. Offline tests (no API)
uv run python -m pytest tests/test_p04_inference.py -v
uv run python -m pytest -q

# 3. Preflight (zero generation spend)
uv run python experiments/006_p04_multimodal_inference/run_inference.py preflight

# 4. Calibration (5 frames, records real usage)
uv run python experiments/006_p04_multimodal_inference/run_inference.py calibrate

# 5. Full run (one command; resumable)
uv run python experiments/006_p04_multimodal_inference/run_inference.py run

# 6. Resume (re-run the same command; completed batches are skipped)
uv run python experiments/006_p04_multimodal_inference/run_inference.py run --name default

# 7. Report + review
uv run python experiments/006_p04_multimodal_inference/run_inference.py report --run-dir data/processed/experiment_006/full_run/default
uv run python experiments/006_p04_multimodal_inference/run_inference.py select-review --primary-run-dir data/processed/experiment_006/full_run/default --output data/processed/experiment_006/review_selection.json
uv run python experiments/006_p04_multimodal_inference/run_inference.py review --model muse-wonder-1.0 --selection data/processed/experiment_006/review_selection.json --primary-run-dir data/processed/experiment_006/full_run/default
```

Compare measured tokens/cost against
`experiments/005_full_video_coverage/cost_report.json` without editing
its historical projections.

## Limitations

- Implementation is UNTESTED here (sandbox has no interpreter); the
  26-test mocked suite (30 cases) must run on the host first.
- Run reports reflect the current execution (a no-op resume reports
  zero new tokens); cumulative history lives in `checkpoints.jsonl`.
- Resume granularity is per batch: a partially failed batch re-runs as
  a unit (its known-good requests are re-submitted), but observations
  stay duplicate-free because batch artifacts are keyed by index
  range; completed batches are never re-submitted.
- No real inference has occurred; all cost figures remain estimates.
- Multi-image batching, `input_tokens` probing, and Wonder availability
  are verified-docs-or-unknown until the first live probe.
- Calibration frames are excluded from nothing; their outputs join the
  full run by `frame_index` only if the same run directory is reused
  (default calibration uses a separate directory and is re-inferred in
  the full run; costed accordingly).

## Data basis

P04 media comes from EPIC-KITCHENS-100 (CC BY-NC-ND 4.0). This
experiment is non-commercial research; frames are submitted to the
inference API for processing, not redistributed. Only Standard billing
is used (verified: Standard content is not used for training);
Contributor/data-sharing modes are never selected. Dataset media,
frames, credentials, and raw model outputs stay outside version
control. Verified M1/M2 production modules are unchanged.
