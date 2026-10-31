# M2-06C.4 offline video pipeline

Status: implemented, awaiting host-side verification.
Entirely offline: no model is queried and no network is used.

## 1. Workflow

`video_pipeline_offline` accepts one small uploaded video
(max 16 MB, enforced by both the API cap and the worker) and
runs, inside one worker claim over a temporary directory:

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

Intermediates (staged copy, frames, first-pass files) live in
the temp directory and vanish with it, including on failure.
Only completed runs persist artifacts: the five
reconstruction files plus `firstpass_report.json` and
`pipeline_summary.json`.

## 2. Honesty contract

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

## 3. Lifecycle and limits

- Synchronous single-claim execution; artifacts store only
  on success, so no partial sets exist. Crashes requeue and
  re-run idempotently under bound 2; explicit failures
  never retry. The bound must be re-evaluated before any
  paid stage joins this workflow name.
- Fixed worker settings in C.4: 1 FPS, full window,
  batch size 5, 16 MB video cap, 1000-frame cap. The
  underlying functions accept configuration; worker-level
  configurability arrives with later intake work.
- No schema change: the four job states plus heartbeat
  suffice. The HTTP API needs no change (generic workflow
  field). The console gained a radio option and renders
  pipeline results through the existing review section.

## 4. Verification status

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

## 5. Limitations

- Failed-frame placeholder timestamps inherit the
  reconstruction `frame/50` fallback, which mislabels
  non-50 FPS videos (existing semantics, unchanged).
- JPEG bytes depend on the OpenCV build; selection,
  metadata, and fake outputs do not.
- No multi-gigabyte intake, no live client, no
  authorization boundary, no cloud execution (C.5+).
- Review UI shows pipeline drafts through the offline
  results view; pipeline-specific sections await later
  milestones.
