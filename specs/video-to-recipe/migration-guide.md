# Video-to-Recipe — Architecture Migration and Assessment Guide

**Status:** Proposed migration instructions for engineering assessment  
**Audience:** Engineering agent performing a controlled repository audit and plan  
**Starting point:** `experiment/003-video-to-recipe` at `548230b`; verify the actual checkout  
**Target:** Smallest defensible, evidence-linked video-to-recipe vertical slice

## 1. Mandate

Assess the existing codebase and design an incremental path toward `spec.md`. **Do not begin by rewriting the repository.** Verify which components already exist and use them as the starting point. Separate current implementations from recorded prior results and proposed features. A migration item is not complete merely because it appears in this document.

This guide describes the desired engineering properties of Bakery Intelligence; it does not require adoption of any external reference application's domain, folder hierarchy, or agent framework.

**Initial execution mode:** Read-only analysis. Do not modify project files, write assessment artifacts, install dependencies, invoke paid model APIs, create branches, or commit until separately authorized under `AGENTS.md`.

## 2. Repository-observed starting inventory

The following is a **source-inspection inventory**, not a claim that every component was executed in the current environment. Recheck it against the active checkout.

| Existing artifact | What source inspection supports | Do not infer |
| --- | --- | --- |
| `experiments/003_video_to_recipe/extract_frames.py` | Frame extraction from a local video; selected interval; CSV manifest containing frame index, derived time, requested time, and offset-adjusted source time. | Robust variable-FPS timing or full multimodal ingestion. |
| `tests/test_extract_frames.py` | Offline synthetic-video tests for timestamps, sampling, invalid inputs, and missing video. | Coverage of a complete recipe pipeline. |
| `experiments/003_video_to_recipe/annotations/preliminary/P04_104.json` | Preliminary manual-review record with four segments, 52 observations, and 11 candidate events (count reported by its validator). | Verified scale values, independently adjudicated event labels, or ingredient masses. |
| `experiments/003_video_to_recipe/validate_preliminary.py` | Checks preliminary status, select identifiers, timestamp bounds, and unverified frame-reference restrictions. | Full schema validation, semantic correctness, or approval. |
| `docs/schemas/weighing_event_schema.md` | Proposed domain contract for frame observations, actions, and weighing events. | Existing production model classes or complete automatic reconciliation. |
| `experiments/002_ocr_scales/` | Synthetic-scale evaluation scripts, reference CSV, recorded results. | General real-world scale reading accuracy. |
| `experiments/004_baseten_api/` | Provider experiment scripts and persisted Baseten request results; preliminary single-frame comparison. | Provider-agnostic production inference or a controlled model benchmark. |
| `scripts/muse-meta.ps1`, `scripts/muse-baseten.ps1` | Local development launchers for two inference environments. | Production application adapters or exhaustive agent compatibility. |
| `docs/architecture/system_proposal.md`, `docs/project/hackathon_mvp_scope.md`, `docs/evaluation/validation_protocol.md` | Proposed architecture, MVP priorities, and planned experimental controls. | Implemented runtime, passing integration tests, or evaluated economic benefit. |
| `src/bakery_intelligence/__init__.py` | Minimal Python package scaffold. | A reusable video-to-recipe application already exists in the package. |

**Important:** The local `P04_104.MP4` and extracted frames are not tracked in Git. Their availability must be checked before proposing any real-media execution. The video has dataset-licensing constraints.

## 3. Target components and responsibilities

The architecture is a *candidate decomposition*, not a mandate to create one class or microservice per box. Reuse existing code and select the fewest modules that satisfy the contracts.

| Boundary | Responsibility | Testable contract |
| --- | --- | --- |
| Input and run coordinator | Input metadata, interval bounds, configuration, run lifecycle, bounded execution | Validated input and run manifest or structured rejection |
| Evidence acquisition | Reuse frame extraction; preserve manifest and source references; later audio input | Frames/observations with stable IDs and time basis |
| Provider adapter | Optional visual extraction through configured provider; normalized outputs and usage/error metadata | Provider-free fixture adapter and one pluggable live adapter |
| Observation ledger | Append or version source observations without erasing discrepancies; provenance/status | Typed, inspectable observation records and referential integrity |
| Temporal event reconstruction | Assemble actions/weighing events with conservative tare and quantity rules | Ordered events; unknowns remain unknown; derivations auditable |
| Independent verification | Deterministic checks separate from generators; issue feedback and blocking verdict | Pass/fail/review-needed with precise issues |
| Draft projection | Construct ingredients and procedural steps only from supported events | Evidence-linked, nonapproved `RecipeDraft` |
| Review and decision boundary | Explicit human corrections, approval per draft version | Cannot approve automatically; history retained |
| Run diagnostics | Configuration, cost/usage, latency, errors, task status | Reproducible audit trail, bounded retries |

For the first pass, a deterministic orchestrator with typed module boundaries is preferred to an open-ended multi-agent loop. Add model-driven reanalysis only when a controlled failure case proves its value.

## 4. Migration decisions, in required order

### M0 — Baseline audit and discrepancy report (no code changes)

**Audit:** Confirm branch/commit/worktree, inspect tree and dependencies, inspect representative test cases and existing schema. Map every required entity in `spec.md` to an existing schema, implementation, or absence. Flag incompatible assumptions (including frame-derived timestamps versus presentation timestamps; preliminary records versus normalized events).

**Output:** Capability matrix and prioritized risks with exact file references. Explicitly state which tests are only historically reported and which were executed in this session.

**Exit condition:** A human can see a trustworthy implemented/proposed/gap matrix, with unresolved design decisions listed. No code modifications.

### M1 — Reconcile and version the core contracts

**Design:** Reuse the weighing-event schema as the domain anchor. Define the minimum typed structures needed for video source, evidence, observation, event, draft, verification, and run manifest. Document whether preliminary annotation JSON needs an import/adapter layer; do not mutate the source annotations to force compliance. Identify versioning and backward-compatibility rules.

**Exit condition:** One authoritative core event contract, a preliminary-record adapter design, and tests planned for invalid references, timestamps, unknowns, and measurement provenance.

### M2 — Build an offline fixture-first end-to-end slice

**Design:** Use synthetic fixtures and existing extractor tests; create only the necessary coordinator and transformation interfaces. The first path should move from normalized observations to ordered events, a conservative procedure draft, and verification **without network or model credentials**. Distinguish a fixture-only vertical slice from raw-video-to-recipe success.

**Exit condition:** A reproducible command and test scenario exercising all boundaries, with a nonapproved evidence-linked output.

### M3 — Event memory and conservative mass reasoning

**Design:** Maintain a chronological record of observations, candidate actions, tare cues, and weighing-event derivations. Never use a scale number as net mass by default. Handle zero after tare, negative display, direct-versus-cumulative readings, missing intermediate states, and conflicting model outputs. Emit explicit unresolved cases.

**Exit condition:** Deterministic tests show correct abstention and derivation decisions; evidence references survive all transitions.

### M4 — Independent verification gates and bounded correction

**Design:** Separate generating modules from validation. Run structural and domain checks at normalization, event assembly, and draft submission. Return actionable, machine-readable feedback. A failed gate cannot be bypassed by a generator. If reanalysis is introduced, bound retries/calls and preserve previous failed versions. Human escalation/review is the terminal safe path when evidence is insufficient.

**Exit condition:** Invalid or unsupported quantities fail or remain unresolved; validation rules cannot be weakened automatically after retries; review cannot be auto-approved.

### M5 — Durable and reproducible runs

**Design:** Persist run manifest, normalized outputs, verification reports, attempt history, model/provider metadata, failure codes, and partial states. Establish a run ID and idempotent output naming strategy. Consider checkpoint/resume only if demonstrably needed and safe with the first implementation scope.

**Exit condition:** A fixture run can be inspected after execution; failed/partial runs preserve provenance and do not falsely report success.

### M6 — Connect to real visual evidence

**Design:** Reuse `extract_frames.py`; supply manifest-backed frames to a provider adapter with explicit rate/cost/time budgets. Resolve how to associate source and requested times, ensure frame references point to locally available evidence, and record prompt/model versions. Begin with a supported interval rather than promising whole-video reconstruction.

**Exit condition:** A real-media run has persisted evidence and an honest draft/partial outcome; if local licensed media or API permission is absent, report the blocked gate rather than simulating verification.

### M7 — Controlled evaluation and expansion

**Design:** Follow the existing validation protocol. Compare a visual baseline with event memory under common budgets; preserve a distinct sealed holdout; introduce audio only if it can be timestamped, aligned, and evaluated separately. Measure unknown rate, coverage, traceability, errors, runtime, and variable cost. Human productivity remains a separate study.

**Exit condition:** Results have immutable inputs, clear denominators, failure cases, and no preliminary-reference leakage. If not executed, report "not evaluated."

## 5. Cross-cutting architectural rules

### Worklist before inference

Before each processing phase, define which evidence questions it can answer: ingredient identity, scale display, tare occurrence, action ordering, or missing steps. Collect supporting **and contradictory** evidence. Do not waste calls on already sufficient items without a concrete reason.

### Minimal evidence ledger

Represent each claim with source IDs, time range, content, confidence/uncertainty status, and the origin of the interpretation. Preserve raw evidence references and avoid destructive overwrites. Include a reason when an item is dismissed or changed.

### Independent enforcement

Implement checks as pure/testable functions where possible. Run them outside model-authored content paths. Rules such as “the ingredient mass must have a derivation and a unit” are invariants, not preferences for a prompt.

### Bounded review/reanalysis

Verification returns a verdict and targeted issue list. One unresolved issue may trigger a bounded retrieval or crop request, but retries never justify filling gaps. If still unresolved, produce a partial result and request human review. Human actions are logged and attributable.

### Restart and provenance

Make run state explicit. A resumed or repeated run must not turn previous model guesses into human reference labels, hide failures, or silently change the approved draft version. A checkpoint system is optional until justified by real failure-recovery needs.

### Testing without outside services

Use deterministic video fixtures, fake provider results, and isolated validators to cover success and failure cases offline. A provider connectivity script is not an application integration test. Live provider calls are separate, user-authorized experiments with budgets and stored metadata.

## 6. Decisions the engineering agent must make explicit

Return reasoned alternatives for each issue; do not silently lock in a solution:

1. **Contract evolution:** Extend `weighing_event_schema.md` directly or introduce an additive application model plus explicit mapper? Document compatibility cost.
2. **Event ledger representation:** Append-only JSONL, versioned JSON documents, or another simple local format? Prioritize inspection and determinism.
3. **Verifier boundary:** What are the smallest hard invariants required at normalization, event assembly, and draft submission?
4. **Recipe granularity:** How should a fragmentary cooking interval yield a partial procedure without claiming a complete ingredient list?
5. **Provider abstraction:** Which response envelope is sufficient to normalize Baseten and Meta API observations without claiming their behavior is identical?
6. **Temporal basis:** How to represent frame index/FPS estimates, requested sampling times, offsets, and genuine presentation timestamps?
7. **Run persistence:** What minimal records enable reproducibility before checkpoint/resume complexity is warranted?
8. **Human review interface:** What is the simplest explicit approval mechanism for a demo without inventing a production UI?
9. **Licensed media:** Which tests can be distributed as synthetic fixtures versus requiring local-only EPIC-KITCHENS footage?
10. **Verification of source evidence:** What qualifies as a verified reference, and how will preliminary annotations remain isolated?

## 7. Requested output format — Phase A assessment

Respond in chat only; no file writes:

1. **Baseline:** checked-out branch, revision, worktree condition, inspection limitations.
2. **Capability inventory:** per component, `Implemented`, `Verified`, `Proposed`, `Unknown`, or `Not evaluated`, with source paths.
3. **Gap and contradiction table:** requirement ID, observed gap, evidence, severity, dependency.
4. **Recommended minimal architecture:** modules, data flow, existing-code reuse, explicit non-goals.
5. **Contract migration matrix:** existing field/type -> proposed field/type -> compatibility impact -> required tests.
6. **Verification design:** gates, failure feedback, uncertainty behavior, human escalation and approval.
7. **Milestones M0–M7:** priority, estimated effort as a range with assumptions, acceptance evidence, blocked dependencies.
8. **Risk register:** dataset licensing, temporal alignment, calibration/reference leakage, model variability, cost, key/config safety.
9. **Questions requiring approval:** no more than the critical unresolved decisions; propose defaults and alternatives.
10. **Recommended first implementation task:** one small unit with exact files likely affected and its offline acceptance test.

**Do not claim code exists merely because it is present in this migration guide.** The output must be useful even if no code is subsequently generated.

## 8. Phase B artifacts (only after explicit authorization)

- `assessment.md`: evidence-backed repository audit with file references and dated baseline.
- `plan.md`: one proposed implementation architecture, rejected alternatives, assumptions, and migration gates.
- `tasks.md`: ordered small tasks; each maps to `VTR-FR-*` and `VTR-AC-*`, includes files, verification commands, pass/fail criteria, and prerequisites.

No PR or merge is implied by producing these documents. The user reviews and approves implementation separately.

## 9. Completion and reporting rule

Migration is complete only for the individual gates with inspected, reproducible acceptance evidence. A successful offline scaffold is not automatically a real-video demonstration, and one real-video demonstration is not a validated recipe-level benchmark.

All reports must preserve differences between **source-inspected facts**, **locally executed tests**, **prior recorded experimental observations**, **proposals**, and **unverified assumptions**.
