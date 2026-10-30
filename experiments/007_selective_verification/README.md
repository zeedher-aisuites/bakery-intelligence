# Experiment 007 — Selective Multimodal Verification

Offline-first workflow for reviewing selected P04 video intervals
against existing `muse-spark-1.3` observations and temporal events
(artifact version 1.1.0). No new model inference is performed here.

## Workflow

1. Generate the selection manifest for the four fixed review
   windows (V01–V04).
2. Export frame references from the manifest for local inspection.
3. Produce blank annotation templates.
4. A human reviewer fills in templates (display text, numeric value
   only when unambiguous, unit only when visually established,
   visible action, support, outcome).
5. Validate completed annotations structurally.
6. Compare predictions against reviewed observations with explicit
   denominators and exclusions.

```powershell
uv run python experiments/007_selective_verification/manifest.py data/processed/experiment_006/temporal/p04_full_4096/temporal_events_1.1.0.json data/processed/experiment_006/full_run/p04_full_4096/observations data/processed/experiment_003/global_1fps_rerun/manifest.csv data/processed/experiment_007/selective_verification/selection_manifest.json
uv run python experiments/007_selective_verification/review.py template data/processed/experiment_007/selective_verification/selection_manifest.json data/processed/experiment_007/selective_verification/annotation_templates.json
uv run python experiments/007_selective_verification/review.py validate data/processed/experiment_007/selective_verification/annotations_completed.json
uv run python experiments/007_selective_verification/metrics.py data/processed/experiment_007/selective_verification/annotations_completed.json data/processed/experiment_007/selective_verification/metrics.json
uv run python -m pytest tests/test_selective_verification.py -v
```

All generated artifacts live under gitignored `data/processed/`.
Inputs (video, observations, temporal artifacts) are never modified.

## Review windows

| Window | Seconds | Purpose |
|---|---|---|
| V01 | 142–153 | Ambiguous `"5 15"` reading and increasing values |
| V02 | 368–378 | Stable `1756` reading and subsequent zero |
| V03 | 396–420 | Stable `238` and `315` readings with abrupt changes |
| V04 | 495–512 | Highly variable readings and possible button interactions |

Events overlapping a window are attached by computation (partial
overlap counts; `fully_inside` records containment). Windows with no
overlapping events keep an empty reference list rather than an
invented one. Each window stores its rationale, event references,
in-window observations with resolved frame paths, a clip range, and
`pending` initial status.

## Annotation contract

One annotation per selected observation. `model_prediction` freezes
the recorded fields plus the deterministically parsed model numeric
(same rule as temporal analysis 1.1.0); human fields start blank and
stay blank until a human reviewer acts. Outcomes: `pending`,
`confirmed` (exact display-text agreement), `corrected` (recorded
difference), `unreadable`, `insufficient_evidence`. Structural rules:
pending records must be fully blank; reviewed records require a
reviewer; a unit requires a numeric value; unreadable records carry no
display, numeric, or unit; insufficient-evidence records carry no
human observation besides an optional note.

## Metrics

Reported with numerator, denominator, and exclusions; empty
denominators yield null:

- Display exact-match accuracy over reviewed readable pairs.
- Numeric agreement over unambiguous comparable pairs.
- Action agreement plus per-class precision/recall over labeled
  pairs (`addition`, `removal`, `tare`, `none`).
- Coverage (reviewed / selected) and abstention
  (`unreadable` + `insufficient_evidence` / reviewed).

Scores measure prediction–review agreement, not ground-truth
accuracy of either side.

## Experiment 010 integration

Verified findings from the adaptive single-timestamp probe
(`run_adaptive_01`, 12 requests, USD 0.269947 estimated): all five
conditions (A–E) recovered to valid `515` predictions within three
budgets; the initial 1,024-token budget was insufficient in every
tested condition; `store=false` allowed the seven-PNG request to
complete; PNG showed no accuracy gain over JPG; denser frames showed
no accuracy gain at this timestamp. These are single-timestamp
engineering observations, not general accuracy measurements.

Integrated consequences: the shared inference lifecycle
(`experiments/006/.../lifecycle.py`) carries terminal outcomes and
the exercised 1024→2048→3072 escalation with configurable initial
budget; the probe evaluator and future comparisons use shared
numeric normalization
(`src/bakery_intelligence/verification/numerics.py`), which fixed
the string-`"515"`-vs-float mismatch without touching saved
outputs; `verify.py` plans bounded verification from temporal
`needs_review` events and selection manifests (reuse existing
frames, propose no extraction automatically, store verified
observations separately, never overwrite originals).

Deferred: PNG default, global 3 FPS, ROI localization, OCR
integration, image enhancement, model comparisons, large-scale
benchmarking. JPEG remains the default; global 1 FPS first pass is
unchanged.

## Experiment 011 — single-event execution

`execute.py` connects planning to a controlled second-pass request
for exactly one planned event. Inputs: verification plan, event ID,
first-pass observations directory, extraction manifest. Outputs
under `data/processed/experiment_007/selective_verification/executions/<run_id>/`:
per-attempt records plus `execution_report.json` with originals,
inputs, prompt, attempts, reconciliation, tokens, cost, and terminal
state. First-pass observations and prior runs are never modified.

```powershell
uv run python experiments/007_selective_verification/execute.py --plan data/processed/experiment_007/selective_verification/verification_plan.json --event-id <ACTUAL_EVENT_ID> --obs-dir data/processed/experiment_006/full_run/p04_full_4096/observations --extraction-manifest data/processed/experiment_003/global_1fps_rerun/manifest.csv --dry-run
```

Dry-run is the default: it validates the event, resolves frames,
reports eligibility, prompt hash, budgets, and destinations, makes
zero API requests, needs no API key, and writes nothing. Live
inference requires `--execute` plus a key; `--dry-run` and
`--execute` together are rejected. Safeguards: one event only, no
batch continuation, no extraction, request cap (`--max-requests`,
default 12), optional paired pricing with post-attempt ceiling
enforcement, hard-ceiling refusal without a pre-estimate, run-ID
overwrite protection, and partial-attempt preservation.

Execution reuses the shared Meta client (configurable `store`),
lifecycle budgets (default 1024,2048,3072; retry only on explicit
token exhaustion), transport retry policy, terminal vocabulary, and
numeric normalization. The frozen prompt labels images by timestamp
and frame only; first-pass values and human references never appear
in it, and the second pass is independent (agreement is never
requested). Reconciliation classifies each frame as `agreement`,
`disagreement`, `verification_abstained`, `original_ambiguous`,
`insufficient_evidence`, or `not_comparable`, compares numbers only
after normalization, never promotes results to ground truth, and
flags any non-agreement for human review.

Verified by construction and mocked tests only; no live second-pass
request has been executed from this workflow. Cost figures are
estimates from configured rates, never confirmed billing.

## Execution report semantics

Cost: `spent_cost_usd_estimate` is the summed estimate only when
every executed attempt has a known cost; any unknown attempt cost
yields `null` with the count in
`spent_cost_usd_unknown_attempts`, never a false zero. Zero requests
yield zero spend. A configured monetary ceiling halts escalation
once known spend exceeds it, but unknown-cost attempts cannot trip
it, so a post-attempt ceiling is not a guaranteed maximum charge; a
hard ceiling refuses to start when no pre-execution estimate exists.

Review: numerical reconciliation (`agreement`, `disagreement`,
`verification_abstained`, `original_ambiguous`,
`insufficient_evidence`, `not_comparable`) is independent of
temporal review. `temporal_review` stays `not_evaluated` whenever
the event carries temporal reasons, because this executor performs
no temporal evaluation; full numerical agreement never clears it.
`requires_human_review` is true when any frame disagrees, abstains,
or lacks evidence, or when temporal review is unevaluated. Neither
agreement nor execution promotes results to ground truth.

## Temporal window selection

Without window flags the executor verifies the complete event.
`--window-start` and `--window-end` (seconds, both required
together) restrict verification to existing event frames whose
timestamps fall in the inclusive interval, preserving order,
indices, paths, MIME types, and hashes. No frames are extracted,
interpolated, or duplicated, and the plan event is never modified.
Bounds must be finite, ordered, inside the event's original
boundaries, and select at least one frame; violations fail with
`INVALID_WINDOW`, `WINDOW_OUT_OF_BOUNDS`, or `WINDOW_EMPTY`
without clamping. The prompt states the requested sub-interval and
lists only selected images; reconciliation covers only selected
frames against their originals. Reports carry a `selection` block
(original bounds, requested bounds, selected indices,
`temporal_subset` flag) with the parent event ID and review
reasons intact. A localized verification never claims to resolve
the full event's temporal inconsistency.

```powershell
uv run python experiments/007_selective_verification/execute.py --plan data/processed/experiment_007/selective_verification/verification_plan.json --event-id scale_visibility_00007100_00007650 --obs-dir data/processed/experiment_006/full_run/p04_full_4096/observations --extraction-manifest data/processed/experiment_003/global_1fps_rerun/manifest.csv --window-start 148 --window-end 152 --dry-run
```

## Recipe reconstruction (M2-04)

`src/bakery_intelligence/reconstruction/` deterministically assembles
measurements, unit resolutions, and canonical drafts from existing
artifacts. `reconstruct` loads first-pass observations, one temporal
event (or sub-window), matching verification reports, and an optional
unit sidecar; it prefers valid verification readings while preserving
original links, computes plain consecutive differences (never
quantities), and emits `recipe_draft.json`, `weighing_event.json`,
`action_events.json`, `measurements.json`, and `validation.json`.
Net quantities and ingredients stay unknown unless supported;
unresolved questions name every gap. `assign-unit` appends
human unit revisions (event-window scope, or session scope with
justification); overlapping conflicts and volume units for scale
data are rejected; observed units stay null. Canonical schemas are
unchanged; inputs are read-only; outputs are deterministic apart
from the run directory name.

```powershell
uv run python -m bakery_intelligence.reconstruction.cli assign-unit --units-file data/processed/experiment_008/reconstruction/units.json --unit g --scope-type event_window --event-id scale_visibility_00007100_00007650 --start 142 --end 153 --reviewer operator-1
uv run python -m bakery_intelligence.reconstruction.cli reconstruct --obs-dir data/processed/experiment_006/full_run/p04_full_4096/observations --temporal data/processed/experiment_006/temporal/p04_full_4096/temporal_events_1.1.0.json --event-id scale_visibility_00007100_00007650 --executions data/processed/experiment_007/selective_verification/executions --units data/processed/experiment_008/reconstruction/units.json --out data/processed/experiment_008/reconstruction --run-id v01_full
```

Verified: mocked unit behavior only (see `tests/test_reconstruction.py`).
Limitation: no canonical Observation/FrameEvidence store is ingested,
so full referential integrity is out of scope; measurement IDs are
experiment-scoped links. Hypothesis: one weighing event per temporal
event suffices until multi-addition intervals require splitting.

## Verified, assumed, limited

Verified (by inspection in this environment): temporal artifact 1.1.0
structure (98 events, 17 `needs_review`, machine-readable reasons);
observation record schema (137 batch files, 685 records); extraction
manifest columns (`image_path`, `frame_index`, timestamps, 685 rows);
frame files present; window-to-event overlap mapping above.

Assumed: the 1 FPS frames and manifest remain the authoritative
visual reference; reviewer identity and timestamps are supplied by
the human workflow, not generated.

Limited: no reviewer has filled in annotations, so no empirical
agreement scores exist; synthetic tests exercise arithmetic only.
Model numerics reuse the temporal 1.1.0 parser, inheriting its
conservative ambiguity rule. Action labels reduce tare visibility
and addition/removal flags to one label per frame.
