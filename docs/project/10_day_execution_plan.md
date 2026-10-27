# Bakery Intelligence — 10-Day Execution Plan

**Status:** Preliminary execution plan (v0.1)  
**Planning baseline:** End of Day 1  
**Scope:** Meta AI Developer Hackathon prototype  
**Strategy:** Six-day MVP implementation followed by four days of evaluation, refinement, and submission preparation.

## Objective

Develop and evaluate an evidence-grounded video-to-recipe prototype that transforms real cooking footage into structured observations, candidate ingredient measurements, and a recipe or procedural representation. The system must preserve uncertainty, distinguish scale-display readings from net ingredient quantities, and avoid inventing unsupported information.

The plan defines intended work and acceptance criteria; future milestones are not claims of implemented or verified functionality.

## Execution schedule

| Day | Focus | Planned deliverable | Acceptance criterion |
| --- | --- | --- | --- |
| 1 | Foundations, dataset, and exploratory review | API connectivity experiment, synthetic scale-reading experiment, real-video selection, frame extractor, preliminary manual observations | Relevant experiments documented; extractor tests pass; preliminary observations are clearly distinguished from verified reference annotations |
| 2 | Structured annotations and visual evidence | Machine-readable preliminary annotations, validation checks, selected source frames | At least three candidate events can be traced to source timestamps and reviewed visually; ambiguous readings remain explicitly uncertain |
| 3 | Real-video multimodal baseline | Muse Spark predictions on selected real-video evidence, with execution metadata | Reproducible predictions and recorded errors, latency, and token usage; no requirement for perfect accuracy |
| 4 | Temporal reconstruction and critical verification | Logic for tare boundaries, unstable readings, and an evidence-request or abstention mechanism | At least one problematic case is identified or explicitly abstained from, with traceable evidence |
| 5 | End-to-end integration | Video segment to structured observations and recipe/procedure representation | At least one real segment can be processed without silently assigning unsupported ingredient identities or masses |
| 6 | MVP stabilization | Reproducible MVP v0.1 and working demonstration | A documented execution path and tests; a recoverable version is retained before further experimentation |
| 7 | Comparative evaluation | Baseline-versus-verification results and error analysis | Evaluation uses independently reviewed reference evidence and reports the limits of the sample |
| 8 | Targeted improvements | Focused corrections to the highest-impact verified failure modes | Changes are tested against existing examples and do not compromise the stable demonstration |
| 9 | Submission materials | Demo video, technical documentation, presentation assets, and submission draft | Materials are internally consistent, evidence-backed, and ready for final review |
| 10 | Final review and submission | Validated submission package | Submission requirements are checked and deliverables submitted before the deadline |

## Current evidence at the planning baseline

- The repository includes documented API connectivity and synthetic scale-reading experiments.
- The real-video candidate is EPIC-KITCHENS-100 `P04_104`.
- A deterministic frame extractor and automated tests have been committed on `experiment/003-video-to-recipe`.
- Eight extractor tests passed in the reported local run.
- Human observations of candidate weighing events are recorded as **preliminary annotations**, not verified ground truth.
- Real-video model accuracy, temporal reconstruction quality, agent effectiveness, and end-to-end recipe quality have **not yet been established**.

## Scope priorities

**P0 — Required for a defensible MVP**
- Real-video evidence ingestion and traceable observations.
- Structured output that separates observed readings, inferred quantities, and unknowns.
- Reproducible execution, evaluation, and demonstration.

**P1 — Important differentiator**
- Critical verification that requests more evidence or abstains when observations are unreliable.
- Comparative evaluation against a direct multimodal baseline.

**P2 — Optional extensions**
- Elaborate graphical interface.
- Additional videos, broader recipe coverage, or custom model training.

## Decision gates

- **End of Day 3:** If real-video reading quality is inadequate, narrow the evaluated event set and document the failure modes rather than inventing measurements.
- **End of Day 4:** If critical verification is not reliable, retain a simpler deterministic validation/abstention mechanism for the MVP.
- **End of Day 6:** Preserve a reproducible MVP v0.1 before introducing improvements.
- **End of Day 9:** Complete submission materials; reserve Day 10 for final checks, contingencies, and submission.

## Daily review protocol

At the end of each day, record:

1. **Completed:** Implemented artifacts and relevant commit identifiers.
2. **Verified:** Tests, observations, metrics, and their evidence.
3. **Uncertain or blocked:** Open experimental limitations and dependencies.
4. **Scope decision:** Work retained, reduced, deferred, or removed for the next day.

Adjust the remaining schedule based on observed progress. Do not retroactively describe planned milestones as completed results.

## Data and evaluation boundaries

- Preserve the distinction between manual preliminary observations, independently verified reference annotations, and model predictions.
- Record timestamps, source offsets, frame provenance, and uncertainty when available.
- A scale-display value is not automatically an ingredient's net mass; tare operations, hand pressure, occlusion, and battery interruptions may invalidate direct interpretation.
- Do not commit source videos, extracted frames, credentials, or other restricted data to the repository.
- Respect EPIC-KITCHENS-100 dataset licensing and attribution requirements in any public demonstration or distribution.

## Versioning and integration

Implement work on `experiment/003-video-to-recipe` using focused commits. Review repository status and preserve uncommitted work before staging, switching branches, or merging. Integrate into the default branch only after the relevant changes have been reviewed and validated. This plan may be revised as evidence accumulates; revisions should identify their scope and rationale.
