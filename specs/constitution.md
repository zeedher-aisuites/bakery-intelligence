# Bakery Intelligence — Engineering Constitution

**Status:** Normative design principles  
**Applies to:** Video-to-Recipe implementation, evaluation, and documentation  
**Baseline:** Repository revision `548230b` on `experiment/003-video-to-recipe`; confirm active revision during assessment.

## Purpose

This constitution establishes the non-negotiable engineering rules for turning video evidence into reviewable bakery recipe documentation. It governs technical tradeoffs, specifications, plans, tests, and change reviews. It does not assert that any proposed pipeline component is already operational.

## I. Evidence precedes interpretation

Every material assertion in a generated recipe draft must be traceable to one or more source observations, a documented derivation, or an explicit human confirmation. A missing source, illegible display, or ambiguous ingredient must be represented as unknown. Plausibility is not evidence.

**Acceptance gate:** A reviewer can navigate from a material draft claim to observation identifiers and the relevant video timestamp or source reference, or sees the claim marked unresolved.

## II. Observation, inference, reference, and approval are different records

Keep at least these roles distinct:

- **Observation:** What an identifiable source frame, interval, audio segment, or human report records.
- **Inference:** A proposed interpretation or quantity derived from observations, with derivation and confidence/uncertainty information.
- **Reference annotation:** An independently adjudicated label used for evaluation; it must not be generated or silently corrected from model predictions.
- **Human approval:** An explicit decision to accept a reviewable draft for use, with reviewer identity and version.

Preliminary human review is not independently adjudicated ground truth. Preserve its status throughout transformations.

## III. Preserve chronological and measurement semantics

A scale display is not automatically an ingredient's net mass. Tare, container changes, negative readings, cumulative measurements, unstable displays, and missing intermediate actions must be modeled explicitly or remain unresolved. Temporal proximity alone cannot prove ingredient attribution or audio-video alignment.

**Acceptance gate:** Deterministic fixtures for tare, cumulative readings, conflicting observations, and missing evidence never create unsupported net quantities.

## IV. Verify independently of generation

Draft generation must not be allowed to waive validation. Independent checks should run at defined boundaries: evidence normalization, event construction, and draft submission. Use deterministic rules for structural integrity, chronology, referenced evidence, numeric derivations, and required review fields. Any model-assisted semantic critique is advisory unless independently evaluated and never a substitute for these checks.

Verification can **pass**, **fail with actionable feedback**, or **request human evidence/review**. Bounded retries must not silently weaken validation criteria.

## V. Human review is mandatory and explicit

Every reconstructed recipe begins in a nonapproved state. The system may produce a partial draft, request clarification, or abstain. Approval requires an affirmative human action on a specific draft version. A verified JSON schema or successful model response is not recipe approval.

**Acceptance gate:** No automatic processing path can transition a draft directly to `approved`.

## VI. Bounded, reproducible execution

Every run should record its input identity, code revision, configuration, sampling policy, model/provider information, relevant prompts, timestamps, retries, failures, and variable cost when available. Retries, tool calls, and inference spending must have limits. A stopped or failed run retains a useful partial record where feasible and truthful status, not fabricated success.

**Acceptance gate:** An offline fixture run is reproducible without provider credentials; external calls are opt-in and instrumented.

## VII. Evaluation isolation and honest reporting

Development examples and final holdouts are separate. Do not tune prompts or derive truth labels using the same sealed evaluation examples. Compare configurations on eligible common inputs with preregistered scoring rules. Report unknowns, abstentions, mismatches, exclusions, and denominators, not just headline accuracy. Human productivity and economic savings require separate comparative evidence.

**Acceptance gate:** Every reported measured result references a persisted artifact or a reproducible evaluation and states its limits.

## VIII. Modular simplicity before orchestration complexity

The default initial implementation is a modular Python application with explicit data contracts and a single processing worker. Provider adapters should be replaceable; domain logic should not depend on any one model or provider. Do not add distributed infrastructure, autonomous agent swarms, or elaborate queues unless a measured requirement justifies them.

**Acceptance gate:** An offline vertical slice can process one supported example through the same public contracts as the eventual provider-backed path.

## IX. Non-destructive, incremental change

Existing experiments, provenance, schemas, user changes, and source-media constraints are assets to preserve. Change contracts explicitly and version them if compatibility breaks. Narrow tasks, focused commits, and regression tests are preferred to broad rewrites. Never assume uncommitted work can be discarded.

## X. Scientific and operational integrity

Use the vocabulary **Implemented**, **Verified**, **Proposed**, **Observed but not independently verified**, **Unknown**, and **Not evaluated** consistently. Technical reports should be usable as final documentation. Do not conflate a model's assertion with a visual measurement or a validation of structure with factual accuracy.

## Decision policy

If a proposed solution violates any principle, select the smallest alternative that preserves the rule, or request an explicit scope decision. A written decision can alter implementation priorities but cannot relabel missing evidence as verified. Deviations from this constitution require a documented rationale, consequences, and the user's authorization.

## Relationship to other documents

- `specs/video-to-recipe/spec.md` defines the behavior to build.
- `specs/video-to-recipe/migration-guide.md` defines how to assess and evolve existing code toward that behavior.
- `AGENTS.md` governs agent actions and repository safety.
- Existing design and experiment records remain the evidence base; this constitution does not replace them.
