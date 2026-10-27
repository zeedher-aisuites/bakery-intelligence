# Video-to-Recipe — Implementation Tasks (M1–M2)

**Status:** Proposed task breakdown for planning approval · Not implemented  
**Specification ID:** VTR-001 (`specs/video-to-recipe/spec.md`)  
**Constitution:** `specs/constitution.md` I–X (normative)  
**Migration framework:** `specs/video-to-recipe/migration-guide.md` M0–M7  
**Depends on:** `specs/video-to-recipe/assessment.md` (2026-10-27) and `specs/video-to-recipe/plan.md` (target architecture)  
**Scope:** M1 (Reconcile and version core contracts) + M2 (Offline fixture-first end-to-end slice) only. M3–M7 are out of scope for this task list.  
**Authorization:** No implementation is authorized by this document. `AGENTS.md` Phase C requires a separate human go-ahead before code, dependency, or branch changes.

---

## 1. How to Read This Document

- Each task is **one small, independently reviewable change** with its own verification command and acceptance criteria.
- **Requirement citations** map to `spec.md` §3 `VTR-FR-01`–`VTR-FR-12` and §8 `VTR-AC-01`–`VTR-AC-10`, plus `docs/schemas/weighing_event_schema.md` validation invariants and `docs/evaluation/validation_protocol.md` isolation rules.
- **Contracts** reference `plan.md` §4 versioned records. Preliminary data is `preliminary_schema_version 0.1.0` (`experiments/003_video_to_recipe/annotations/preliminary/P04_104.json`); canonical records are `schema_version 1.0.0` per `docs/schemas/weighing_event_schema.md`.
- **Verification** is by inspection or offline deterministic command. No task requires a live model, API key, or network access — see §5.
- **Failure behavior** is explicit: every task states what happens on invalid input, budget exhaustion, or verifier rejection.
- Tasks are **ordered by dependency**; a task may start only when its stated prerequisites have passed.

---

## 2. Execution Constraints (Normative for M1–M2)

1. **Do not mutate** `experiments/003_video_to_recipe/annotations/preliminary/P04_104.json` or `P04_104_manual_observations.md` — translate via adapter (Constitution IX; requirement in this brief).
2. Preserve **timestamp provenance** — `frame_index / FPS` is allowed only under a documented constant-FPS assumption carried in `timestamp_basis` (plan.md §4.2). Do not use it as the sole basis for domain contracts.
3. Unknown ingredient quantities remain **unknown, never zero** — `net_ingredient_quantity {value:null, unit:null, derivation:"unknown"}` when tare/unit/stability preconditions fail (Constitution III, `spec.md` §5 Rules 1–5, schema validation invariants).
4. Do **not** automatically promote model-generated observations to verified ground truth — `verification_status` ceiling is `ambiguous` for model/adapter/preliminary sources (plan.md §5.1 state 2). Only the deterministic verifier may assign `confirmed` to pipeline records after G1–G3 plus derivation preconditions (state 3); human-adjudicated reference `confirmed` (state 4) is a separate, isolated process. Deterministic verification establishes structural and semantic consistency, not factual correctness of visual interpretation (`spec.md` §7, plan.md §5.1/§7/§12).
5. Keep the slice **offline and deterministic** — no inference calls, no `.env` secrets read, no network egress in M1–M2 (Constitution VI, migration guide §M2, `VTR-AC-10`).
6. No distributed services or multi-agent orchestration (Constitution VIII).

---

## 3. Milestone Overview

| Milestone | Goal | Entry Criteria | Exit Gate (from `plan.md` §10) |
|---|---|---|---|
| **M1** | One authoritative `schema_version 1.0.0` contract + `0.1.0 → 1.0.0` adapter design, with contract-level tests for invalid refs, timestamps, unknowns, and measurement provenance. | `assessment.md` and `plan.md` approved; Git HEAD re-verified against declared `548230b` (assessment §1 limitation to be closed). | `pytest tests/test_contracts.py` passes (see §4 acceptance rows); `P04_104.json` adapter emits valid canonical records without mutating source. |
| **M2** | Reproducible minimal `[fixture → ledger → events → draft(pending_review) → verification → manifest]` without network/credentials. Introduces each capability in minimal form; M3–M5 later extend and harden it (plan.md §10). | M1 tasks accepted (M1-01 through M1-03 green). | `bakery-intelligence run --fixture fixtures/synthetic/... --offline` yields ordered events, evidence-linked `RecipeDraft{pending_review}`, persisted `VerificationReport` (structural/semantic consistency), and an inspectable minimal `RunManifest` seed (full fields in M5); `VTR-AC-01/04/08/10` pass offline. No draft can be `approved` without a `ReviewDecision`. |

---

## 4. Tasks — M1: Reconcile and Version the Core Contracts

### M1-01 — Versioned Domain Models (`schema_version 1.0.0`)

| Field | Content |
|---|---|
| **Objective** | Implement the authoritative typed domain models for `VideoSource`, `FrameEvidence`, `Observation`, `ActionEvent`, `WeighingEvent`, `RecipeDraft`, `VerificationReport`, `RunManifest`, `ReviewDecision` in Python. Export JSON Schemas for each record. |
| **Requirement** | `VTR-FR-01`, `VTR-FR-02`, `VTR-FR-03`, `VTR-FR-05`, `VTR-FR-07`, `VTR-FR-08`, `VTR-FR-09`, `VTR-FR-10`; schema validation invariants (`docs/schemas/weighing_event_schema.md` §Validation invariants); plan.md §4.1 versioning policy. |
| **Files to create** | `src/bakery_intelligence/models/video_source.py`, `frame_evidence.py`, `observation.py`, `action_event.py`, `weighing_event.py`, `recipe_draft.py`, `verification.py`, `run_manifest.py`, `review_decision.py`; `src/bakery_intelligence/models/__init__.py`; `src/bakery_intelligence/models/schemas/*.json` (exported JSON Schemas); `docs/schemas/migration_notes.md` (compatibility log). |
| **Files to modify** | `pyproject.toml` only if a strictly necessary Pydantic-related pin is missing (prefer existing `pydantic>=2.14.1`); otherwise none. |
| **Input contract** | Native Python construction of records; JSON round-trip for each record type. |
| **Output contract** | Pydantic models with `schema_version`, typed enums (`timestamp_basis ∈ {frame_index_over_fps, presentation_timestamp, approximate_human_video_review, synthetic_fixture}`, `verification_status ∈ {confirmed, ambiguous, unobservable}`, `derivation ∈ {direct_after_tare, difference_between_readings, independently_verified, unknown}`, `measurement_mode ∈ {individual, cumulative, unknown}`, `verdict ∈ {pass, fail, review_needed}`, `review_decision ∈ {approved, rejected, needs_revision}`), and JSON Schema artifacts. |
| **Dependencies** | None (first task). Git HEAD verification to be recorded in `migration_notes.md`. |
| **Tests / verification** | `pytest tests/test_contracts.py -k "test_schema_version or test_model_roundtrip"` — constructs valid/invalid bundles; asserts valid bundles parse, invalid bundles raise `ValidationError` with field-specific messages. |
| **Acceptance criteria** | (1) Every record type is constructable and JSON-serializable. (2) `schema_version` is required and rejected when unknown major. (3) Hard invariants enforced at model layer: `0 <= start < end <= duration` on `VideoSource.interval` and on `ActionEvent`/`WeighingEvent`; timestamp `>=0` and within interval; `derivation=="unknown"` ⇒ `net_ingredient_quantity.value==null && unit==null`. (4) Exported JSON Schemas validate canonical examples. |
| **Expected failure** | Missing required field / invalid timestamp / unknown `schema_version` → `ValidationError` with `loc` pointing to offending field; no file written. |
| **External API / credentials** | None. |

### M1-02 — Preliminary Annotation Adapter (`0.1.0 → 1.0.0`)

| Field | Content |
|---|---|
| **Objective** | Translate `preliminary_schema_version 0.1.0` sources (exemplified by `P04_104.json`) into canonical `1.0.0` `Observation` + `FrameEvidence` records without mutating the source file, never inventing evidence or promoting hypothesis to fact. |
| **Requirement** | `VTR-FR-03`, `VTR-AC-03`; `spec.md` §5 Rule 9; Constitution II + IX; assessment §5 contract conflict map. |
| **Files to create** | `src/bakery_intelligence/adapters/preliminary_adapter.py` (function `adapt_preliminary(path: Path) -> AdapterResult{observations, frame_evidence, issues, source_hash}`), `tests/test_preliminary_adapter.py`. |
| **Files to modify** | None (adapter reads `experiments/003_video_to_recipe/annotations/preliminary/P04_104.json` read-only). |
| **Input contract** | `P04_104.json` shape: `{preliminary_schema_version:"0.1.0", annotation_status:"preliminary_not_frame_verified", segments[].observations[{observation_id, approximate_timestamp_seconds, reported_action, reported_display_or_state, evidence{frame_index:null,image_ref:null,verification_status:not_frame_verified}, provenance}]}`. |
| **Output contract** | Canonical `Observation[]` with `timestamp_basis:"approximate_human_video_review"`, `provenance:{source_type:adapter, source_id:"P04_104.json@<sha256>"}`; `frame_evidence: []` for all `not_frame_verified` items; `scale_unit_hypothesis` never copied to `scale_unit`; `verification_status` capped at `ambiguous`/`unobservable`; `raw_reported_value` preserved when parser is uncertain (e.g. `"54–55"`, `"-750"`, `"Lo"` cases). |
| **Dependencies** | M1-01 (canonical models must exist). |
| **Tests / verification** | `pytest tests/test_preliminary_adapter.py` — (a) golden-file comparison: adapted `P04_104.json` yields `canonical/P04_104.canonical.json` snapshot checked into `tests/data/`; (b) `assert adapted[0].verification_status != "confirmed"`; (c) `assert source_file_hash unchanged after adapt`; (d) ambiguous display (`"54–55"`) → `scale_value is None, raw_reported_value == "54–55"`. |
| **Acceptance criteria** | Source file byte-identical before/after; output validates under `1.0.0` schemas; zero `confirmed` observations produced from preliminary inputs (only the M2-03 verifier may later assign `confirmed` per plan.md §5.1 state 3); `scale_unit_hypothesis:"g"` never becomes `scale_unit:"g"`. |
| **Expected failure** | Malformed/missing `segments` → `AdapterError{code:"invalid_preliminary", entity_id, message}`; no canonical record emitted for that observation. |
| **External API / credentials** | None. |

### M1-03 — Evidence Reference and Measurement Validation (Contract-Level)

| Field | Content |
|---|---|
| **Objective** | Enforce structural and measurement invariants at the contract layer: ID uniqueness, interval bounds, timestamp validity, referential integrity of `evidence_observation_ids`/`observation_ids`/`action_ids`, and the `derivation → value/unit` coherence rule. |
| **Requirement** | `VTR-FR-05`, `VTR-FR-08`, `VTR-AC-04`, `VTR-AC-05`; schema validation invariants; plan.md §7 G1–G2. |
| **Files to create** | `src/bakery_intelligence/verification/reference_checks.py` (pure functions: `check_unique_ids`, `check_interval_bounds`, `check_timestamps`, `check_referential_integrity`, `check_measurement_derivation`), `tests/test_reference_validation.py`. |
| **Input contract** | Canonical bundles from M1-01/M1-02 or hand-crafted fixtures. |
| **Output contract** | `Issue[]` with `{code, entity_id, gate∈{G1,G2}, severity, message}`; codes include `DUPLICATE_ID`, `INVALID_INTERVAL`, `TIMESTAMP_OUT_OF_BOUNDS`, `DANGLING_REFERENCE`, `DERIVATION_WITHOUT_UNIT`, `UNKNOWN_DERIVATION_WITH_VALUE`. |
| **Dependencies** | M1-01, M1-02. |
| **Tests / verification** | `pytest tests/test_reference_validation.py` — fixtures: dangling `observation_ids`, `0<=start<end` violations, `derivation:"unknown"` with non-null `value`, duplicate `observation_id`. |
| **Acceptance criteria** | Each invariant has a dedicated test; every failing fixture returns the expected `code` with the offending `entity_id`; valid fixtures return `[]`. |
| **Expected failure** | Validates — does not throw on domain errors; returns `Issue[]` for caller to surface as `VerificationReport`. |
| **External API / credentials** | None. |

---

## 5. Tasks — M2: Offline Fixture-First End-to-End Slice

### M2-01 — Offline Synthetic Fixtures

| Field | Content |
|---|---|
| **Objective** | Commit deterministic synthetic fixtures that exercise the hard cases from `P04_104.json` without requiring real media, to serve as the offline test bed for the entire slice. |
| **Requirement** | `VTR-FR-01`, `VTR-FR-02`, `VTR-FR-04`, `VTR-FR-06`, `VTR-FR-12`, `VTR-AC-01`, `VTR-AC-02`, `VTR-AC-05`, `VTR-AC-10`; migration guide §M2. |
| **Files to create** | `fixtures/synthetic/synthetic_interval_nominal.json` (happy-path: stable tare + readable mass), `fixtures/synthetic/synthetic_interval_unknown_tare.json` (VTR-AC-02), `fixtures/synthetic/synthetic_interval_conflicting.json` (VTR-AC-05), `fixtures/synthetic/synthetic_interval_dangling_ref.json` (VTR-AC-04), `fixtures/synthetic/synthetic_interval_partial_budget.json` (VTR-AC-06); `fixtures/README.md`. |
| **Input contract** | Each fixture is a self-contained `VideoSource + FrameEvidence[] + Observation[]` bundle with `timestamp_basis` documented and `interval` bounds. Fixtures fall into three classes (plan.md §9.3): (a) structurally invalid — `synthetic_interval_dangling_ref.json`; (b) valid structure, uncertain semantics — `synthetic_interval_unknown_tare.json`, `synthetic_interval_conflicting.json`; (c) valid structure, budget-exhausted execution — `synthetic_interval_partial_budget.json`; plus nominal `synthetic_interval_nominal.json` (expected to complete). |
| **Output contract** | All fixtures are schema-parseable JSON with `schema_version:"1.0.0"` and `timestamp_basis: synthetic_fixture`. Classes (b), (c), and nominal additionally pass M1-03 structural checks (IDs, bounds, referential integrity); class (a) parses but must fail M1-03 with `DANGLING_REFERENCE`. |
| **Dependencies** | M1-01 (schema to validate fixtures). |
| **Tests / verification** | `pytest tests/test_fixtures.py` — every fixture `schema_version==1.0.0`; nominal + classes (b)+(c) pass M1-03 structural checks; class (a) `synthetic_interval_dangling_ref` fails M1-03 with `DANGLING_REFERENCE`; classes (b)+(c) diverge only at later stages (M2-03/M2-04/M2-06), not at structural validation. `fixtures/README.md` presence checked and documents the three classes with expected outcomes. |
| **Acceptance criteria** | 5 fixtures committed with class and expected outcome tagged in each `description`: nominal → `completed`/`pass`; class (a) dangling_ref → M1-03/G1/G2 `fail` with `DANGLING_REFERENCE`, no draft; class (b) unknown_tare → `unknown` quantity + `UNKNOWN_TARE_HISTORY`, conflicting → `review_needed` with contradictions preserved; class (c) partial_budget → structurally valid, yields `partial` + `BUDGET_EXHAUSTED` only under budget exhaustion (M2-06). No `scale_value` conflated with `net_ingredient_quantity`. |
| **Expected failure** | Fixture loader reports `FIXTURE_INVALID` with `issue.code` if fixture drifts from schema. |
| **External API / credentials** | None. |

### M2-02 — Append-Only Observation Ledger

| Field | Content |
|---|---|
| **Objective** | Implement the durable, append-only ledger that records every observation with `provenance`, `verification_status`, `supersedes/superseded_by`, and `contradiction_of` links, and derives a snapshot view without mutating history. |
| **Requirement** | `VTR-FR-03`, `VTR-FR-05`; Constitution I + II; plan.md §5.2. |
| **Files to create** | `src/bakery_intelligence/ledger/ledger.py` (`append(entry)`, `read_ledger()`, `derive_snapshot()`), `src/bakery_intelligence/ledger/__init__.py`, `tests/test_ledger.py`. |
| **Input contract** | Canonical `Observation` objects from fixtures, adapter, or future provider adapter. Physical form: `runs/<run_id>/ledger.jsonl` (one JSON object per line, chronological), plus derived `observations.json` snapshot. |
| **Output contract** | `LedgerEntry{observation, ledger_seq, appended_at, supersedes?, superseded_by?, contradiction_of?}`; `derive_snapshot()` never drops contradicted entries — both sides of a conflict remain visible (VTR-AC-05). |
| **Dependencies** | M1-01, M2-01. |
| **Tests / verification** | `pytest tests/test_ledger.py` — append 3 observations, assert `ledger.jsonl` line count == 3, `snapshot` mirrors them; supersede one, assert both retained and `supersedes` back-pointer present; contradictory readings on same interval retained (no silent resolution). |
| **Acceptance criteria** | Ledger is append-only (no in-place update helper exists); `provenance` and `verification_status` never promoted from `adapter`/`model`/`human`-preliminary to `confirmed` by ledger code — only the M2-03 verifier assigns `confirmed` (plan.md §5.1 state 3); contradictions are preserved not deduplicated. |
| **Expected failure** | Append with duplicate `observation_id` → `LedgerError{code:"duplicate_id"}`; read of missing/corrupt line → `LedgerError{code:"corrupt_line", line_no}`. |
| **External API / credentials** | None (local filesystem only). |

### M2-03 — Independent Verifier (G1/G2/G3)

| Field | Content |
|---|---|
| **Objective** | Implement the three minimal deterministic verification gates (M2 scope; M4 hardens enforcement and bounded retry) as pure functions that cannot be bypassed by any generator, with machine-readable feedback. Gates establish structural and semantic consistency, not factual correctness of visual interpretation (plan.md §5.1/§7/§12). |
| **Requirement** | `VTR-FR-08`, `VTR-AC-04`, `VTR-AC-05`, `VTR-AC-08`; `spec.md` §7 checklist; plan.md §7. |
| **Files to create** | `src/bakery_intelligence/verification/gate_g1_normalization.py`, `gate_g2_assembly.py`, `gate_g3_submission.py`, `src/bakery_intelligence/verification/report.py` (`VerificationReport` builder), `tests/test_verifier.py`. |
| **Input contract** | G1: `VideoSource + FrameEvidence[] + Observation[] + Ledger snapshot`. G2: `ActionEvent[] + WeighingEvent[]`. G3: `RecipeDraft + RunManifest draft + prior VerificationReports`. |
| **Output contract** | `VerificationReport{report_id, verdict∈{pass, fail, review_needed}, issues[{code, entity_id, gate, severity, message, suggested_action}], checked_ids[], verifier_version, checked_at}`. Hard invariants (plan.md §7.1 table) always enforced; retries never weaken them (Domain Rule 10). A `pass` verdict attests structural/semantic consistency only, never pixel-level reading correctness. |
| **Dependencies** | M1-01, M1-03, M2-02. |
| **Tests / verification** | `pytest tests/test_verifier.py` — (a) `synthetic_interval_nominal` → `verdict:pass` (consistency, not visual correctness); (b) class (a) `synthetic_interval_dangling_ref` → `fail` with `DANGLING_REFERENCE`; (c) class (b) `synthetic_interval_conflicting` → `review_needed` with contradiction issue, both readings preserved; (d) class (b) `synthetic_interval_unknown_tare` → `pass` or `review_needed` (never `fail` on structure) so M2-04 can project an `unknown`-quantity draft; (e) crafted `RecipeDraft{status:approved}` without `ReviewDecision` → G3 `fail` with `AUTO_APPROVAL_REJECTED` (VTR-AC-08). |
| **Acceptance criteria** | All three minimal gates block on hard violations; `review_needed` preserves both sides of a conflict; no generator path can skip a gate (verified by call-graph inspection — orchestrator always delegates to verifier before advancing). `pass` is documented and tested as structural/semantic consistency, not factual visual correctness; bounded-retry hardening is deferred to M4. |
| **Expected failure** | Invalid gate payload → `Report{verdict:fail, issues:[...]}` (never a throw for domain errors); internal bug → exception with `stage:verification` logged to manifest. |
| **External API / credentials** | None. |

### M2-04 — Evidence-Linked Recipe Draft (`pending_review`)

| Field | Content |
|---|---|
| **Objective** | Project supported `ActionEvent`/`WeighingEvent` into an ordered, evidence-linked `RecipeDraft` that invents no quantities, marks every unsupported claim as an `unresolved_question`, and is always `pending_review`. |
| **Requirement** | `VTR-FR-07`, `VTR-FR-06`, `VTR-FR-09`, `VTR-AC-01`, `VTR-AC-02`, `VTR-AC-08`; plan.md §3.2 Draft Projection. |
| **Files to create** | `src/bakery_intelligence/draft/projector.py` (`project_draft(events, verification_ref, run_id) -> RecipeDraft`), `tests/test_draft_projection.py`. |
| **Input contract** | `ActionEvent[] + WeighingEvent[]` that have passed G2 (or `review_needed` with explicit tolerance), plus the authoritative `VerificationReport` ID. |
| **Output contract** | `RecipeDraft{draft_id, version:1, video_id, steps[{index, description, action_ids, observation_ids, evidence_summary}], ingredients[{ingredient_id, quantity{value,unit,derivation,source_event_id}, status}], unresolved_questions[{code, message, related_entity_ids}], verification_ref, run_id, status:"pending_review"}`. Quantities only when `derivation != "unknown"` and verification passed; verification pass attests consistency, not pixel-level reading correctness (plan.md §12). |
| **Dependencies** | M1-01, M2-03 (G2 must exist so draft never projects from failed events). |
| **Tests / verification** | `pytest tests/test_draft_projection.py` — nominal fixture produces steps with `observation_ids` and `evidence_summary`; class (b) unknown-tare fixture (`synthetic_interval_unknown_tare`) produces draft with `ingredients[].quantity == null` + `unresolved_questions` containing `UNKNOWN_TARE_HISTORY`; class (b) conflicting fixture projects only under explicit `review_needed` tolerance with contradictions preserved; any crafted `status:"approved"` input rejected. |
| **Acceptance criteria** | Draft is `pending_review` iff G2/G3 allow it; unsupported quantities absent (not invented as zero); every material claim traces to `observation_ids` or is flagged unresolved per `spec.md` §7. |
| **Expected failure** | Empty event set → draft with single `unresolved_question: NO_SUPPORTED_EVENTS` but still `pending_review` (honest, not fabricated); upstream G2 `fail` → projector refuses and returns `DraftProjectionError{code:"upstream_verification_failed"}`. |
| **External API / credentials** | None. |

### M2-05 — Pending-Review State and Approval Boundary

| Field | Content |
|---|---|
| **Objective** | Enforce that no approval happens automatically: the draft leaves M2 only as `pending_review`, and approval requires an explicit, version-pinned `ReviewDecision` that never auto-upgrades. |
| **Requirement** | `VTR-FR-09`, `VTR-AC-08`, `VTR-AC-09`; Constitution V; plan.md §8. |
| **Files to create** | `src/bakery_intelligence/review/state.py` (`assert_not_approved(draft)`, `record_decision(draft, reviewer_id, decision) -> ReviewDecision`), `src/bakery_intelligence/review/__init__.py`, `tests/test_review_state.py`. |
| **Input contract** | `RecipeDraft` (any version) + reviewer identity string + decision enum. |
| **Output contract** | `ReviewDecision{decision_id, draft_id, draft_version, reviewer_id, decision∈{approved, rejected, needs_revision}, notes?, corrections_ref?, decided_at}`; approval is version-pinned: approving `V_N` does not approve `V_{N+1}`; `record_decision` never mutates the draft. |
| **Dependencies** | M1-01, M2-04. |
| **Tests / verification** | `pytest tests/test_review_state.py` — initial draft `assert_not_approved` passes; `record_decision(draft, "reviewer_01", "approved")` succeeds with version pin; second draft `V2` without new decision still `pending_review`; `assert not_approved` fails on any smuggled `approved` status with message `AUTO_APPROVAL_REJECTED`. |
| **Acceptance criteria** | No automatic approval path exists (verified by absence of any code path that sets `status:"approved"` without `ReviewDecision`); `approved` requires a `ReviewDecision` record with matching `draft_version`. |
| **Expected failure** | Draft with `status:"approved"` but no matching `ReviewDecision` → `ReviewStateError{code:"approval_without_decision"}` and G3 `fail`. |
| **External API / credentials** | None. |

### M2-06 — Offline Command-Line Execution

| Field | Content |
|---|---|
| **Objective** | Expose the offline slice as a single, documented CLI command that runs without network or credentials and fails gracefully on invalid input or budget exhaustion. Integrates the M2-07 minimal manifest writer via its lifecycle hooks; does not implement manifest persistence itself. |
| **Requirement** | `VTR-FR-11`, `VTR-FR-12`, `VTR-AC-01`, `VTR-AC-06`, `VTR-AC-10`; `spec.md` §10 Definition of done; plan.md §9.3 layout. |
| **Files to create** | `src/bakery_intelligence/cli.py` (entry point `bakery-intelligence`), `src/bakery_intelligence/orchestrator.py` (`run_fixture(fixture_path, out_dir, budgets) -> RunResult`), `tests/test_offline_cli.py`. |
| **Files to modify** | `pyproject.toml` (`[project.scripts] bakery-intelligence = "bakery_intelligence.cli:main"` — already declared as `bakery_intelligence:main`; adjust to `cli:main` if present entry point is a stub). |
| **Input contract** | `bakery-intelligence run --fixture <path> [--out <dir>] [--frame-budget N] [--offline]`; fixture validated before processing; `--offline` asserts no network call is attempted (guarded by a missing adapter injection). |
| **Output contract** | On success: `runs/<run_id>/{manifest.json, ledger.jsonl, observations.json, frame_evidence.json, events.json, draft.json, verification.json}` plus console summary, where `manifest.json` is the M2-07 minimal seed (identity, input snapshot, `started_at`/`budget_state`/`status`, output paths). On budget exhaustion (`--frame-budget` exceeded, class (c)): `status:partial` with `status_reason: BUDGET_EXHAUSTED` and best-effort artifacts preserved (VTR-FR-11 / VTR-AC-06). Class (a) yields `failed` with `verification.json{verdict:fail}` preserved. |
| **Dependencies** | M1-01–M1-03, M2-01–M2-05, and M2-07 (minimal manifest writer must exist before the orchestrator lifecycle that calls it; see §6). |
| **Tests / verification** | `pytest tests/test_offline_cli.py -k test_offline_happy or test_budget_exhaustion or test_invalid_fixture` — nominal via `python -m bakery_intelligence.cli run --fixture fixtures/synthetic/synthetic_interval_nominal.json --out /tmp/... --offline` asserts exit 0 and artifact tree with minimal-seed `manifest.json`; class (c) `--frame-budget 1` on `synthetic_interval_partial_budget.json` (or multi-frame nominal) → exit 0 but `manifest.status == "partial"` with `BUDGET_EXHAUSTED`; class (a) `synthetic_interval_dangling_ref.json` → exit 1 with `verification.json{verdict:fail}` preserved. |
| **Acceptance criteria** | Documented command runs with no `.env` / network; `manifest.status` truthfully reflects `completed|partial|failed`; no invented observations on exhaustion. |
| **Expected failure** | Invalid fixture path → `CLIError{code:"fixture_not_found", hint}` exit 2; validation failure → exit 1 with `verification.json{verdict:fail}` preserved for inspection. |
| **External API / credentials** | None required; fails closed if any provider adapter is injected without `--offline` and credentials. |

### M2-07 — Reproducible Run Manifest

| Field | Content |
|---|---|
| **Objective** | Persist a minimal, auditable `RunManifest` seed for every run (including `partial`/`failed`) that supports reproducibility without provider secrets or licensed media in artifacts. Full §9.1 fields (attempts, provider calls, cost, output hashes, `finished_at`/`elapsed_ms`) are deferred to M5 (plan.md §9.1, §10). This task precedes M2-06 despite its number (see §6). |
| **Requirement** | `VTR-FR-10`, `VTR-FR-11`, `VTR-AC-06`; Constitution VI; plan.md §9. |
| **Files to create** | `src/bakery_intelligence/run_manifest/writer.py` (`begin_run`, `record_stage`, `finalize_manifest`, plus offline stubs `record_provider_call`, `record_error`), `tests/test_run_manifest.py`. |
| **Input contract** | Defines the orchestrator lifecycle hook signatures (`begin_run`, `record_stage`, `finalize_manifest`) that M2-06 calls; minimal hooks for provider/error recording are stubs returning empty offline defaults. |
| **Output contract** | `runs/<run_id>/manifest.json` minimal seed containing `{run_id, code_revision (git HEAD or "unknown-dirty"), started_at, video_source_snapshot, run_config_snapshot, timestamp_assumption, budget_state, outputs{paths}, status, status_reason}`; `run_id` is UUID or deterministic hash of `fixture_hash + config + revision`. M5 adds `finished_at`, `elapsed_ms`, output `sha256` hashes, `attempts[]`, `provider_calls[]`, `errors[]`, `cost_summary` (offline defaults: empty/zero, no secrets). |
| **Dependencies** | M1-01 (RunManifest model) only. Must be complete before M2-06; M2-06 is the consumer, not the prerequisite. |
| **Tests / verification** | `pytest tests/test_run_manifest.py` — happy run minimal seed has `status:completed` with `run_id`, `code_revision`, `started_at`, input snapshots, `budget_state`, output paths, and `status_reason`; failed/budget-exhausted runs still have `manifest.json` with truthful `status:partial`/`failed`; rerun of same fixture+config is deterministic or cleanly namespaced (no silent overwrite). `elapsed_ms`, output hashes, `attempts[]`, and `cost_summary` assertions are deferred to M5. |
| **Acceptance criteria** | Every CLI invocation writes a minimal-seed `manifest.json` regardless of outcome; `run_id`, `code_revision`, `started_at`, input snapshots, `budget_state`, output paths, and `status`/`status_reason` are always present; offline runs carry no provider secrets. Full-field assertions (`elapsed_ms`, hashes, `attempts[]`, `provider_calls[]`, `cost_summary`) belong to M5. |
| **Expected failure** | Manifest write failure (e.g. output dir not writable) → `ManifestError{code:"write_failed"}` surfaced to CLI as exit 2; never a silent no-op. |
| **External API / credentials** | None. |

---

## 6. Dependency Graph and Rollout Order

```
M1-01  (domain models — no deps)
  ├─► M1-02 (adapter)
  ├─► M1-03 (reference/measurement checks)
  ├─► M2-01 (synthetic fixtures)
  └─► M2-07 (minimal manifest seed — models only, no orchestrator dep)

M1-02 + M1-03
  └─► M2-02 (ledger) ─┐
                       ├─► M2-03 (minimal verifier G1/G2/G3) ─► M2-04 (draft) ─► M2-05 (approval boundary) ─┐
M2-01 ─────────────────┘                                                                                    │
                                                                                                            ├─► M2-06 (CLI/orchestrator, integrates all; runs last)
M2-07 (minimal manifest) ───────────────────────────────────────────────────────────────────────────────────┘
```

- M1-01 is the **sole root**.
- M2-07 (minimal manifest seed) must be complete before M2-06; task numbers are preserved for review stability, so build M2-07 before M2-06 despite the order. M2-06 integrates the full slice and runs last before final sign-off; M2-07 is a separate review unit so manifest discipline can be reviewed in isolation.
- M1-02, M1-03, and M2-07 may proceed in parallel once M1-01 is green (M2-07 needs only the models).
- M2-02 and M2-01 may proceed in parallel once their respective parents are green.

---

## 7. Verification Matrix — Requirements to Tasks

| Requirement | Tasks | Acceptance Signal |
|---|---|---|
| `VTR-FR-01` — bounded interval & timestamp basis | M1-01, M2-01, M2-06 | Invalid intervals rejected with `ValidationError`; `timestamp_basis` present on every record. |
| `VTR-FR-02` — frame observations with stable IDs | M1-01, M2-01, M2-02 | Ledger snapshot has resolvable IDs; fixture exercises happy path. |
| `VTR-FR-03` — provenance & status preserved | M1-02, M2-02 | Adapter preserves `not_frame_verified` as `unobservable`; ledger never promotes `adapter→confirmed`. |
| `VTR-FR-04` — tare-aware events, no default mass | M1-03, M2-01 (unknown-tare fixture), M2-04 | Unknown-tare draft has null quantity + unresolved question; `600` force-artifact fixture never becomes net mass (extends via M3). |
| `VTR-FR-05` — evidence links | M1-03, M2-03 (G2), M2-04 | Dangling ref → `DANGLING_REFERENCE` → G2 `fail` → no draft with dangling links. |
| `VTR-FR-06` — preserve unknown | M1-01, M2-01, M2-04 | `derivation:unknown` ⇒ null value/unit enforced; unresolved_questions surfaced. |
| `VTR-FR-07` — evidence-linked draft | M2-04 | Draft has `observation_ids` per step or `unresolved_questions`; traceable claim→evidence. |
| `VTR-FR-08` — independent verification | M1-03, M2-03 | Minimal gates G1/G2/G3 exist as pure functions; generators cannot bypass; `VerificationReport` persisted. `pass` attests structural/semantic consistency, not visual correctness; bounded-retry hardening is M4. |
| `VTR-FR-09` — mandatory approval | M2-05, M2-03 G3 | Initial draft `pending_review`; auto-approval rejected with `AUTO_APPROVAL_REJECTED`. |
| `VTR-FR-10` — run manifest | M2-07 (before M2-06) | Every run writes a minimal-seed `manifest.json` with `run_id`, revision, `started_at`, input snapshots, budgets, output paths, and truthful `status`; hashes, `elapsed_ms`, `attempts[]`, and cost are M5 extensions. |
| `VTR-FR-11` — bounded budgets → partial | M2-07, M2-06 | Class (c) budget exhaustion → `partial` with reason, not invented data; minimal seed records truthful `budget_state`/`status`. |
| `VTR-FR-12` + `VTR-AC-10` — offline fixture path | M2-01, M2-06 | `pytest` and CLI run with `uv run --no-network` (or equivalent sandbox) and no `.env`. |
| `VTR-AC-01` — synthetic fixture E2E | M2-06 | CLI on `synthetic_interval_nominal` → `completed`. |
| `VTR-AC-02` — unknown tare → unknown mass | M2-01, M2-03, M2-04 | Class (b) `synthetic_interval_unknown_tare` (structurally valid) → verifier `pass`/`review_needed`, draft quantity null + `UNKNOWN_TARE_HISTORY`. |
| `VTR-AC-03` — preliminary status retained | M1-02, M2-03 | Adapted `P04_104` never yields `confirmed` as a direct adapter output; only the verifier may later assign `confirmed` (plan.md §5.1 state 3). |
| `VTR-AC-04` — dangling evidence ref → fail | M1-03, M2-03, M2-06 | Class (a) `synthetic_interval_dangling_ref` → M1-03/G1/G2 `fail` with `DANGLING_REFERENCE`; no draft. |
| `VTR-AC-05` — contradictory readings surfaced | M1-03, M2-02, M2-03 | Class (b) conflicting fixture (structurally valid) → ledger preserves both → `review_needed` with contradiction issue (no silent consensus). |
| `VTR-AC-06` — budget exhausted → partial | M2-07, M2-06 | Class (c) low-budget CLI on valid fixture → `partial` + `BUDGET_EXHAUSTED` with artifacts preserved (never `fail` on structure). |
| `VTR-AC-08` — no auto-approval | M2-05, M2-03 G3 | Draft without `ReviewDecision` rejected if smuggled as `approved`. |
| `VTR-AC-09` — approval per version | M2-05 | `record_decision(V1)` does not approve `V2`. |

Deferred to M3+ (not in this milestone): `VTR-AC-07` (provider unavailable — no live adapter in M2), `VTR-AC-05` force-artifact subtlety at full semantics (M3), bounded-retry hardening (M4), full `RunManifest` fields (M5), and all holdout evaluation (`validation_protocol.md`).

---

## 8. Non-Destructive Verification Commands (Per Task)

All commands assume `uv`/`pytest` toolchain per `pyproject.toml` / `AGENTS.md` (Python 3.12).

```bash
# M1
pytest tests/test_contracts.py -v
pytest tests/test_preliminary_adapter.py -v
pytest tests/test_reference_validation.py -v

# M2
pytest tests/test_fixtures.py -v
pytest tests/test_ledger.py -v
pytest tests/test_verifier.py -v
pytest tests/test_draft_projection.py -v
pytest tests/test_review_state.py -v
pytest tests/test_offline_cli.py -k "test_offline_happy or test_budget_exhaustion or test_invalid_fixture" -v
pytest tests/test_run_manifest.py -v

# Slice gate (all offline, no network, no secrets)
pytest -v

bakery-intelligence run --fixture fixtures/synthetic/synthetic_interval_nominal.json --offline --out runs/out
# inspect:
jq . runs/out/manifest.json
cat runs/out/verification.json
cat runs/out/draft.json
```

No command installs packages, fetches media, or calls a provider. Licensed `data/raw/epic_kitchens/P04_104.MP4` is never required for M1–M2.

---

## 9. Rollback and Compatibility

- Every breaking `schema_version` bump (e.g. `1.0.0 → 2.0.0`) must be recorded in `docs/schemas/migration_notes.md` with affected records and a migration script or adapter note.
- `preliminary_schema_version 0.1.0` artifacts remain readable indefinitely — the adapter is maintained, source files are never rewritten.
- No Git-destructive operations per `AGENTS.md` (no `reset --hard`, `clean`, `checkout -- <path>`, force-push, or branch deletion).

---

## 10. What Is Explicitly Not In This Milestone

- M3–M5 hardening of M2 capabilities — extended derivation matrix and force-artifact guards (M3), bounded-retry enforcement and verifier versioning (M4), full `RunManifest` fields with hashes and cost accounting (M5). M2 introduces each capability in minimal form per `plan.md` §10; M3–M5 extend rather than reintroduce.
- Live provider adapter, frame extraction from real video, PTS timing (M6); audio/P2, checkpoint/resume, distributed deployment, UI, and holdout evaluation (M6–M7).
- Exhaustive recipe reconstruction from a full bakery video — the slice deliberately targets **supported intervals** with synthetic fixtures; a successful M2 must not be reported as verified real-video reconstruction (`spec.md` §10, `plan.md` §17).

---

*This task list is planning evidence only. Execution follows `AGENTS.md` Phase C: one narrowly scoped task at a time, with tests, migration notes, and explicit verification results, after separate human authorization.*
