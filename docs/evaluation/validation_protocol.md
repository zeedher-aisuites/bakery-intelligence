# Bakery Intelligence — Validation Protocol

**Status:** Proposed experimental protocol; final holdout evaluation not yet executed.

## Questions

1. Does structured event memory improve reconstruction over a visual baseline?
2. Does aligned audio improve identification and quantity interpretation?
3. Can the system reconstruct a usable, evidence-linked draft within a bounded processing budget?
4. Does human review of the draft save active work time and total cost relative to manual documentation?

## Dataset and independence

Use **at least ten independent holdout videos** with varied durations, preparations, operators, and recording conditions where feasible. Development material—including synthetic scale examples and the previously inspected real video—must not be reused as final holdout data. Freeze selection criteria, prompts, model settings, scoring rubric, and budget limits **before** evaluation. Record exclusions and reasons.

Annotate relevant actions, ingredients, scale readings, tare events, quantities where independently recoverable, and unresolved ambiguity. Record timestamped references and annotator provenance. Do **not** manufacture ground truth when the source does not establish an answer. Ten videos constitute an exploratory benchmark, not proof of generalization.

## Comparison

| Configuration | Visual evidence | Event memory | Audio | Human clarification during inference |
|---|---|---|---|---|
| **A — Baseline** | Yes | Limited | No | No |
| **B — Temporal** | Yes | Yes | No | No |
| **C — Multimodal target** | Yes | Yes | Yes, when present | No |

Evaluate configurations on the **same eligible holdouts**, with the same output schema, comparable budgets, and identical adjudication rules. If C is not implemented, report the A/B results and explicitly mark the audio hypothesis **not evaluated**. Optional Telegram HITL belongs to a separate assisted experiment. Final human approval remains required for operational use but is not assistance during autonomous inference.

## Acceptance and metrics

A draft is **satisfactorily reconstructed for research evaluation** only when it passes a preregistered rubric covering meaningful action order, supported ingredient identities, supported quantities, honest unknowns, and retrievable evidence. Define critical failures and numerical tolerances before testing. Research acceptance **does not mean operational human approval**.

| Metric | Definition / proposed target |
|---|---|
| Satisfactory drafts | Initial exploratory target **≥6/10**, if ten eligible videos and recipe-level evaluation is supported |
| Autonomous terminal completion | Target **100%** of eligible jobs ending in a draft or honest partial result; not equivalent to success |
| Tare recall | Target **≥80%** of independently annotated evaluable tare events |
| Ingredient precision | Target **≥85%** on evaluable predictions; also report recall and abstention coverage |
| Quantity error | Proposed **≤10%** relative error on evaluable nonzero quantities; freeze aggregation method |
| Traceability | Target **100%** of material claims linked to evidence or explicitly labeled uncertain; uncertainty is not correctness |
| Audio contribution | Change from B to C; report alignment mistakes and contradictory claims |
| Processing cost | Total variable inference USD / total source-video minutes; also report USD per satisfactory draft |
| Runtime | Wall-clock processing time / source-video duration; no real-time requirement |

Targets are **hypotheses and design goals**, not achieved results or official hackathon thresholds. Report per-video failures, unknowns, and denominators.

## Human productivity and economic evaluation

Technical reconstruction quality and business value are **separate tests**. To evaluate productivity, compare active time spent manually documenting a preparation with active time spent reviewing/correcting an AI draft, using equivalent acceptance criteria and representative participants. Include review, corrections, retries, API expenditure, and attributable infrastructure cost.

The primary economic comparison is **total cost per human-approved recipe** versus **total cost of equivalent-quality manual documentation**. A small API bill alone does not establish savings. Mental workload and downstream profit improvements require separate evidence.

## Reproducibility and reporting

For every run preserve video ID/duration, source/license, model ID, prompts/settings, frame/audio sampling, observation and event outputs, reference annotations, model calls, tokens when available, retries, elapsed time, errors, and costs. Record the code commit and date. Keep calibration separate from holdouts; preserve failures and abstentions.

**Previously reported, limited evidence:** controlled synthetic scale reading (10/10 across two runs), preliminary real-video observations, and eight passing frame-extraction tests. None verifies full recipe reconstruction, audio fusion, or labor savings.

Publish conclusions as **Verified**, **Inconclusive**, or **Not evaluated**, supported by versioned artifacts. If the system only completes a narrower task, evaluate and name that task rather than claiming full recipe reconstruction.
