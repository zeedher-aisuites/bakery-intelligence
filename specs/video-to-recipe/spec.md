# Video-to-Recipe — Functional Specification

**Status:** Proposed requirements for the first executable vertical slice  
**Specification ID:** VTR-001  
**Primary users:** Bakery operator documenting a process; human reviewer verifying a recipe draft  
**Implementation state:** Not asserted by this document

## 1. Goal and boundaries

Transform a supported interval of a recorded bakery/pastry preparation into a structured, chronological, evidence-linked **recipe or procedure draft** that makes unsupported information explicit and requires human approval. Processing is asynchronous. The first slice prioritizes visual evidence and weighing-event reasoning; audio is a subsequent multimodal evaluation target, not a hidden prerequisite for P0.

A partial recipe, or a procedural representation with explicit unknowns, is a valid outcome when evidence is insufficient. A plausibly complete but unsupported recipe is not.

### In scope — P0

- Accept a local recorded video or a reproducible, pre-extracted evidence fixture with video identity.
- Recover or ingest time-indexed frame observations and preserve source references.
- Convert observations into ordered, evidence-linked candidate events.
- Preserve tare uncertainty, measurement mode, ingredient identity uncertainty, and mass derivation.
- Generate a structured draft with ingredients/quantities only when supported, procedural steps where supported, evidence references, and unresolved questions.
- Apply a separate deterministic verification gate before exposing a draft for human review.
- Produce a versioned run manifest and a clear nonapproved review state.
- Demonstrate a reproducible offline vertical slice with fixtures and automated tests.

### In scope — P1 after P0

- Real provider-backed visual inference with recorded request/response metadata.
- Targeted evidence requests or frame reinspection, bounded by execution limits.
- Comparison of baseline visual interpretation against event-memory reconstruction.
- Basic processing-time and provider-usage accounting.

### Conditional — P2

- Audio transcription with timestamped claims and explicit cross-modal alignment.
- Human clarification during processing; separate this assisted setting from autonomous evaluation.

### Out of scope for this slice

Production SaaS, real-time streaming, cross-tenant access, inventory synchronization, recipe commercialization, automated approval, distributed worker orchestration, unrestricted video ingestion, and general claims of model accuracy.

## 2. Stakeholders and success scenario

A baker supplies a supported recording. The application captures a bounded interval, generates timestamped observations, reconstructs events conservatively, and offers an inspectable draft. A reviewer can see which source evidence supports each material item, correct or reject items, and explicitly approve a version when adequate. No generated draft is approved automatically.

## 3. Functional requirements

| ID | Requirement | Acceptance evidence |
| --- | --- | --- |
| VTR-FR-01 | Identify input video, duration or bounded interval, timestamp basis, and analysis configuration. | Run metadata stores input identity and basis; invalid intervals are rejected. |
| VTR-FR-02 | Produce or ingest frame observations with stable identifiers and timestamped source references. | Fixture provides resolvable observation references and valid timestamps. |
| VTR-FR-03 | Distinguish extracted visual readings, preliminary human notes, model outputs, and adjudicated references. | Provenance and verification states survive normalization. |
| VTR-FR-04 | Reconstruct ordered candidate actions and weighing events without converting all display readings into net masses. | Tests cover cumulative and tare examples with correct abstention. |
| VTR-FR-05 | Link each material event and draft claim to source observation IDs. | Referential integrity validator rejects dangling references. |
| VTR-FR-06 | Preserve unknown ingredient identity, unreadable digits, uncertain units, missing tare history, and conflicting readings. | No unknown is filled with an invented concrete value in fixtures. |
| VTR-FR-07 | Generate a recipe/procedure draft with ordered steps, supported quantities, unresolved items, and draft status. | Offline end-to-end fixture yields inspectable structured output. |
| VTR-FR-08 | Run independent structural and measurement checks before draft submission. | Invalid fixture is rejected with specific issues; generator cannot bypass check. |
| VTR-FR-09 | Require affirmative human approval of a particular version. | Tests prove default `pending_review` and prohibit automatic approval. |
| VTR-FR-10 | Persist run identity, inputs, provenance, parameters, errors, elapsed time, and provider usage when available. | Run manifest schema and fixture output are inspectable. |
| VTR-FR-11 | Respect bounded frame/model budgets and terminate as completed, partial, failed, or awaiting review. | Test demonstrates honest partial output at a configured limit. |
| VTR-FR-12 | Permit independent offline execution without provider keys or restricted media. | Documented command runs fixture path with network disabled. |

## 4. Proposed logical records

These are **design-level entities**, not a claim that matching Python classes exist. Reconcile with `docs/schemas/weighing_event_schema.md` before writing code; prefer evolution over a second, conflicting event contract.

| Entity | Minimum fields / invariant |
| --- | --- |
| `VideoSource` | `video_id`, `source_ref`, `timestamp_basis`, duration or bounded interval, optional license/source metadata |
| `FrameEvidence` | Stable ID, video ID, frame index or source time, media reference, provenance, extraction settings |
| `Observation` | Stable ID, modality, timestamp/interval, content or structured values, source evidence IDs, origin, status |
| `ActionEvent` | Stable ID, ordered interval, type, optional ingredient, evidence observation IDs, interpretation status |
| `WeighingEvent` | Existing contract fields: final displayed reading, net quantity with derivation, mode, tare state, evidence IDs |
| `RecipeDraft` | Stable ID/version, ordered steps, supported ingredients/quantities, unresolved questions, claim-to-evidence links, `pending_review` |
| `VerificationReport` | Verdict, list of machine-checkable issues, checked entity IDs, verifier version/config |
| `RunManifest` | Run ID, code revision, input snapshot, configuration, outputs, budgets, attempts, timings, provider usage/errors |
| `ReviewDecision` | Draft ID/version, reviewer identity, decision, notes/corrections, decision time |

**Minimum status semantics:** preliminary/unverified evidence must not become verified during mapping; `unknown` values must remain explicit; reference annotations and predictions must be stored separately; `approved` is a human-origin decision rather than an output of inference.

## 5. Domain rules

1. `scale_value` is the displayed value; it is **not** the derived `net_ingredient_quantity`.
2. `direct_after_tare` requires credible tare history and relevant unit evidence, not a single image.
3. `difference_between_readings` requires compatible measurement states and an explicit derivation trace; it is not the default.
4. A zero reading after tare does not imply an empty container.
5. One sharp frame is insufficient to assert temporal display stability.
6. When frames and claimed timestamps conflict, retain both signals and mark the mismatch for review.
7. Audio statements, when later introduced, are claims to align, not substitutes for verified physical measurements.
8. Model disagreement is a review signal, not a vote that automatically resolves truth.
9. Reference labels may be corrected only by a documented independent reference-adjudication process.
10. Independent checks must enforce a rejection or an explicit unresolved status; repeated model attempts cannot override hard invariants.

## 6. External inference boundary

Provider-backed extraction must sit behind an adapter that accepts a normalized evidence request and returns a typed result plus metadata or an explicit error. Use model and provider identifiers as data; do not embed provider-specific assumptions in event reconstruction. Raw responses must be retained or referenced in a way consistent with privacy and dataset rules. No paid calls without consent and configured limits.

## 7. Verification and human review

Verification is a distinct stage applied to the output of interpretation and draft assembly. It must check at least:

- Required fields, valid timestamps, ID uniqueness, and source references.
- Event-to-observation and draft-to-event integrity.
- No approved status without human action.
- Numeric quantity derivations supported by a known unit and compatible measurement states.
- Unknown and contradictory claims are not silently dropped.
- Partial/failure states accurately represent unmet preconditions and exhausted budgets.

A verifier report may recommend additional evidence; it must not fabricate it. Human reviewers may edit/correct claims and approve a specific version. Corrections preserve the original machine draft and the correction provenance.

## 8. Required acceptance scenarios

| Case | Input / condition | Expected result |
| --- | --- | --- |
| VTR-AC-01 | Synthetic video fixture and deterministic observations | End-to-end draft or partial draft, with ordered events and traceable evidence. |
| VTR-AC-02 | Visible reading with unknown preceding tare | Display value preserved; net ingredient mass remains unknown. |
| VTR-AC-03 | Plausible but unverified preliminary note | Preliminary status retained; not promoted to adjudicated ground truth. |
| VTR-AC-04 | Observation references a nonexistent source frame | Verification failure names the unresolved reference. |
| VTR-AC-05 | Contradictory scale readings from the same interval | Conflict recorded and surfaced; no silent consensus. |
| VTR-AC-06 | Budget exhausted mid-analysis | Honest `partial` result with a reason and available evidence. |
| VTR-AC-07 | Model provider unavailable | Structured error and recoverable run state; no invented observations. |
| VTR-AC-08 | Draft produced without reviewer action | Status remains `pending_review` or another explicitly nonapproved state. |
| VTR-AC-09 | Human review approves version N | Approval recorded for version N only, retaining original evidence links. |
| VTR-AC-10 | Offline execution without media/API credentials | Fixtures run without live API calls or private dataset redistribution. |

## 9. Evaluation constraints

Follow `docs/evaluation/validation_protocol.md`. Development case `P04_104` is **not** a sealed holdout. The currently committed `P04_104.json` is preliminary and not frame-verified. Its structural validation is not a claim of factual accuracy. Evaluate at least ten independent holdouts for the intended later recipe-level comparison, with criteria frozen before the holdout run. Preserve failed and abstained cases. If audio is absent, explicitly report the multimodal hypothesis as not evaluated.

## 10. Definition of done — first vertical slice

P0 is done only when all of the following are inspectable:

- A documented offline entry point accepts a fixture representing a supported interval.
- Output contains observations, ordered events, a recipe/procedure draft, explicit unknowns, and provenance.
- Independent validation produces a persisted verdict and blocks invalid transitions.
- The draft remains pending human review unless a separate reviewer explicitly approves it.
- At least the specified failure/uncertainty cases have automated tests.
- The run metadata records the code version and reproducibility parameters.
- Documentation identifies which parts were implemented and actually exercised versus those still proposed.

Working from a real media interval is the next integration gate; P0 offline success alone must not be reported as verified real-video recipe reconstruction.
