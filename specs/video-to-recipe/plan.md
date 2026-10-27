# Video-to-Recipe — Architecture Plan

**Status:** Proposed architecture for planning approval · Not implemented  
**Specification ID:** VTR-001 (see `specs/video-to-recipe/spec.md`)  
**Constitution:** `specs/constitution.md` Principles I–X (normative)  
**Migration framework:** `specs/video-to-recipe/migration-guide.md` M0–M7  
**Assessment baseline:** `specs/video-to-recipe/assessment.md` (2026-10-27)  
**Target:** Minimal modular-monolith with a single worker producing an offline, deterministic, evidence-linked `RecipeDraft` in `pending_review`  
**Implementation state:** None of the components below are claimed to exist. Approval of this plan does not authorize implementation.

---

## 1. Goals and Non-Goals

### Goals (P0 per `spec.md` §1 and `docs/project/hackathon_mvp_scope.md`)
- Transform a **supported interval** of a recorded bakery/pastry preparation into a structured, chronological, evidence-linked **recipe/procedure draft** that makes unsupported information explicit.
- Preserve **tare uncertainty, measurement mode, ingredient-identity uncertainty, and mass derivation** per `docs/schemas/weighing_event_schema.md` v1.0.0.
- Enforce **deterministic verification gates** before any draft is exposed for review (Constitution IV) and **mandatory human approval** before operational use (Constitution V).
- Demonstrate a **reproducible offline vertical slice** from fixture to draft with inspectable provenance and a versioned `RunManifest` (Constitution VI, `spec.md` VTR-FR-10/12, VTR-AC-10).
- Keep inference **bounded** by explicit frame/model budgets and surface honest `partial`/`failed` outcomes (VTR-FR-11, VTR-AC-06).

### Non-Goals for the first slice (explicitly deferred)
- Production SaaS, multi-tenancy, billing, inventory sync, real-time streaming, recipe commercialization, and unrestricted video ingestion (`spec.md` §1 "Out of scope"; migration guide §3 preamble).
- Distributed queues, job schedulers, or multi-agent orchestration (Constitution VIII; `migration-guide.md` M2 design note: deterministic orchestrator preferred over open-ended multi-agent loop).
- Audio transcription and cross-modal alignment — **Conditional P2** per `spec.md` §1; to be introduced only with timestamped, independently evaluable audio claims (H2 remains `Not evaluated` until then, per `docs/evaluation/validation_protocol.md`).
- Human-in-the-loop clarification during inference and Telegram integration — **Conditional** per `docs/project/hackathon_mvp_scope.md`.
- Checkpoint/resume orchestration — optional until a measured failure-recovery need justifies it (migration guide M5 design).

---

## 2. Design Principles (from Constitution)

1. **Evidence precedes interpretation** (I): every material draft claim must resolve to observation IDs + timestamp/source reference, or be marked unresolved. Missing/illegible/ambiguous → `null`/`unknown`/`unresolved`, never a plausible default.
2. **Distinct records** (II): `Observation` (what a source recorded), `Inference` (derived interpretation with derivation + uncertainty), `ReferenceAnnotation` (independently adjudicated evaluation label), `HumanApproval` (explicit `ReviewDecision` on a draft version). Preliminary human review (`P04_104.json: preliminary_not_frame_verified`) is never ground truth.
3. **Chronological & measurement semantics** (III): `scale_value ≠ net_ingredient_quantity`; tare, container changes, negative readings, cumulative measurements, unstable displays, missing history are modeled explicitly or remain unresolved.
4. **Independent verification** (IV): deterministic checks at normalization, event construction, and draft submission; generator cannot waive or weaken them; verdict is `pass` / `fail with actionable feedback` / `request human review`.
5. **Mandatory explicit approval** (V): no automatic `approved` transition; a valid schema or model success is not approval.
6. **Bounded, reproducible execution** (VI): every run records input identity, code revision, config, sampling policy, provider/prompt metadata, retries, errors, timings, and variable cost where available.
7. **Evaluation isolation** (VII): development fixtures (including P04_104) are not holdouts; ≥10 independent holdouts with frozen rubric per `docs/evaluation/validation_protocol.md`.
8. **Modular simplicity** (VIII): typed module boundaries + one worker; provider adapters are replaceable; domain logic has no provider-specific assumptions.
9. **Non-destructive, incremental change** (IX): existing experiments, schemas, and `P04_104.json` are preserved via adapter; contract changes are versioned.
10. **Scientific integrity** (X): consistent use of **Implemented / Verified / Proposed / Observed but not independently verified / Unknown / Not evaluated**.

---

## 3. Target Architecture

### 3.1 Component diagram

```
                          ┌─────────────────────────────────────────────┐
                          │          Input & Run Coordinator            │
                          │  VideoSource + interval + RunConfig         │
                          │  validates bounds, timestamp_basis, budgets │
                          └──────────────┬──────────────────────────────┘
                                         │ typed VideoSource / RunManifest seed
                ┌────────────────────────┼─────────────────────────┐
                ▼                        ▼                         ▼
     ┌───────────────────┐   ┌──────────────────────┐   ┌────────────────────┐
     │ Evidence          │   │ Provider Adapter     │   │ Preliminary        │
     │ Acquisition       │   │ (offline fake →      │   │ Annotation Adapter │
     │ (reuses          │   │  live pluggable)     │   │ (0.1.0 → v1.0.0)   │
     │ extract_frames.py│   │  EvidenceRequest →   │   │ preserves raw,     │
     │ + manifest.csv)  │   │  typed observations  │   │ never mutates      │
     └─────────┬─────────┘   └──────────┬───────────┘   │ source             │
               │                        │               └─────────┬──────────┘
               └────────────┬───────────┘                         │
                            ▼                                     │
                 ┌──────────────────────┐                          │
                 │  Observation Ledger  │◄─────────────────────────┘
                 │  append-only,        │
                 │  provenance + status │
                 └──────────┬───────────┘
                            │ ordered observations (stable IDs, refs)
                 ┌──────────▼───────────┐
                 │ Temporal Event       │
                 │ Reconstruction       │  tare-aware, abstaining
                 │ (ActionEvent +      │  derivation: direct_after_tare /
                 │  WeighingEvent)     │  difference_between_readings / unknown
                 └──────────┬───────────┘
                            │ candidate events
                 ┌──────────▼───────────┐      ┌──────────────────┐
                 │ Independent          │─────▶│ Verification     │
                 │ Verification Gates   │◀─────│ Report           │
                 │ (pure functions)     │      │ (verdict+issues) │
                 └──────────┬───────────┘      └──────────────────┘
                            │ pass / review-needed
                 ┌──────────▼───────────┐      ┌──────────────────┐
                 │ Draft Projection     │─────▶│ RecipeDraft      │
                 │ (supported claims    │      │ pending_review   │
                 │  + unresolved Qs)    │      │ evidence-linked  │
                 └──────────┬───────────┘      └──────────────────┘
                            │ draft version
                 ┌──────────▼───────────┐      ┌──────────────────┐
                 │ Review & Decision    │─────▶│ ReviewDecision   │
                 │ Boundary             │      │ (human approval) │
                 └──────────────────────┘      └──────────────────┘

     ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
     │  Run Diagnostics (cross-cutting): RunManifest {run_id, code_rev,   │
     │  input snapshot, config, budgets, attempts, timings, provider      │
     │  usage/errors, output refs, status ∈ {completed, partial, failed, │
     │  awaiting_review}}, checkpoint/resume deferred                      │
     ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
```

The first slice is a **deterministic orchestrator** with typed boundaries, not a multi-agent loop. Model-driven reanalysis is added only when a controlled failure case proves its value (migration guide §3 final note, §5 Bounded review/reanalysis).

### 3.2 Component responsibilities and boundaries

| Component | Responsibility | Inputs → Outputs | Invariants it enforces |
|---|---|---|---|
| **Input & Run Coordinator** | Validates `VideoSource{video_id, source_ref, timestamp_basis, duration/interval, license}`, `RunConfig{sample_fps, interval, frame_budget, model_budget, retry_limit}`, and budget exhaustion policy. Seeds `RunManifest`. | `VideoSource + RunConfig → validated RunContext` or structured rejection (VTR-FR-01, VTR-FR-11) | Invalid intervals rejected; `timestamp_basis` propagated; budgets have defaults and hard caps |
| **Evidence Acquisition** | Reuses `experiments/003_video_to_recipe/extract_frames.py` + `manifest.csv`. Preserves `image_path, frame_index, timestamp_seconds, source_timestamp_seconds, requested_timestamp_seconds`. Does not interpret displays. | `VideoSource + interval → FrameEvidence[] + Manifest` | Every emitted frame has `image_ref` resolvable under a configured evidence root; no silent duplicate indices |
| **Provider Adapter** | Pluggable behind `EvidenceRequest → TypedResult + ProviderMetadata|Error`. Offline slice uses a **fake adapter** returning fixture observations; live adapter (M6) wraps rate/cost/time-bounded calls and records `provider, model_id, prompt_hash, settings, tokens, latency, retries, error`. Domain logic never branches on provider identity. | `EvidenceRequest{frame_refs, questions} → Observation[] + ProviderCallRecord` | Provider-free fixture path is first-class; no domain rule depends on a provider string; no invented observations on error (VTR-AC-07) |
| **Preliminary Annotation Adapter** | Translates `preliminary_schema_version 0.1.0` → canonical v1.0.0 records **without mutating source**. Preserves `approximate_timestamp_seconds`, `reported_display_or_state`, and `evidence{not_frame_verified}` as provenance metadata; parses structured values only when unambiguous. | `P04_104.json (0.1.0) → Canonical Observation[]` (preliminary-sourced) | Source file never rewritten; unverified status never promoted to `confirmed`; `scale_unit_hypothesis` → `unit_hypothesis` annotation, not `scale_unit` |
| **Observation Ledger** | Append-only, versioned history of all observations with `provenance{source_type, source_id}`, `verification_status`, `supersedes/superseded_by`, and `contradiction_of` links. Supports `ambiguous`/`unobservable` retention and contradiction surfacing. | `Observation[] (any origin) → LedgerEntry[] (ordered, immutable)` | No destructive overwrite; status never `confirmed → ambiguous` via adapter; contradiction preserved not resolved by voting (VTR-AC-05, Domain Rule 8) |
| **Temporal Event Reconstruction** | Orders candidate `ActionEvent`/`WeighingEvent` from ledger; applies conservative tare/mode/derivation rules; emits explicit `unknown`/`unresolved` when preconditions fail. | `LedgerEntry[] + VideoSource → ActionEvent[] + WeighingEvent[]` (with `derivation` + `review_notes`) | Zero-after-tare ≠ empty; negative/cumulative/single-frame never become `confirmed` mass; `direct_after_tare`/`difference_between_readings` require credible history + unit + compatible states |
| **Independent Verification** | Pure, deterministic checks at three gates (see §7). Returns `VerificationReport{verdict∈{pass,fail,review_needed}, issues[{code, entity_id, message, gate}], checked_ids[], verifier_version}`. Cannot be bypassed by any generator. | `Gate payload → VerificationReport` | Hard invariants (ID uniqueness, interval validity, referential integrity, derivation→unit/value coherence, no auto-approval) always enforced; retries never weaken them (Domain Rule 10) |
| **Draft Projection** | Constructs `RecipeDraft{version, steps[], ingredients[], quantities[], evidence_links[], unresolved_questions[], status=pending_review, verification_ref}` **only** from supported events; invents no quantities. | `ActionEvent[] + WeighingEvent[] + VerificationReport(pass/review_needed) → RecipeDraft` | `pending_review` is the only initial status; no draft without evidence links for material claims; unresolved items are explicit questions |
| **Review & Decision Boundary** | Records `ReviewDecision{draft_id, version, reviewer_id, decision∈{approved, rejected, needs_revision}, notes, decided_at}`. Approval is version-pinned and attributable. | `RecipeDraft + reviewer action → ReviewDecision` + new `RecipeDraft` version if corrected | No automatic approval path (VTR-AC-08); corrections preserve original draft + correction provenance (`spec.md §7`); approved version ≠ auto-approved next version |
| **Run Diagnostics** | Captures and persists `RunManifest` across the run lifecycle, including partial/failed runs (Constitution VI). | Cross-cutting: every stage appends timing/error/usage | **Two stages:** M2 seeds minimal `RunManifest` (identity, input snapshot, `started_at`/`budget_state`/`status`); M5 extends to full `§9.1` fields (attempts, provider calls, cost, output hashes, `finished_at`/`elapsed_ms`). Failures never fabricate success; offline runs have no provider secrets in artifacts; `run_id` is deterministic or UUID with idempotent output naming |

### 3.3 Technology choices and non-goals

- **Language & toolchain:** Python 3.12, `uv` + `pytest` per `pyproject.toml` / `AGENTS.md` Implementation constraints.
- **Modeling:** `pydantic>=2.14.1` already declared — used for all versioned domain models.
- **Single worker:** One-process orchestration for M1–M5; no Celery/RQ/Cloud Tasks.
- **Storage:** Local filesystem (JSON + JSONL); no database, object store, or queue required for P0.
- **Distribution:** If ever introduced, must be justified by a measured requirement (Constitution VIII).

---

## 4. Data Contracts (Versioned)

### 4.1 Contract versioning policy

- Domain contract version is `schema_version` on each record bundle (e.g. `"1.0.0"` per `docs/schemas/weighing_event_schema.md`). The preliminary contract remains `preliminary_schema_version: "0.1.0"` solely as source provenance.
- Breaking changes (field rename/removal/type narrowing, new required field, enum value removal) bump **major**; additive optional fields bump **minor**; documentation-only bump **patch**.
- No source mutation on version change: an adapter/mapper translates between versions; the old artifact remains readable.
- Every persisted bundle records its own `schema_version`; the verifier rejects unknown majors.
- Forward reference: `preliminary_schema_version 0.1.0` → canonical `schema_version 1.0.0` is handled exclusively by the Preliminary Annotation Adapter (§3.2).

### 4.2 Contract catalog (normative for M1–M2; P2/Audio deferred)

#### VideoSource (§4 per `spec.md` §4; reconciled with `weighing_event_schema.md` Video metadata)

| Field | Type | Meaning | Constraint |
|---|---|---|---|
| `schema_version` | `str` | Contract version | `"1.0.0"` for M1–M2 |
| `video_id` | `str` | Stable dataset-local ID, e.g. `P04_104` or `SYNTH.FIXTURE.001` | Non-empty |
| `source_ref` | `str` | Original filename or local relative path, never an absolute secret path | Required |
| `source_dataset` | `str` | Dataset name, e.g. `EPIC-KITCHENS-100` or `synthetic-fixture` | Optional but recommended |
| `timestamp_basis` | `str` | Enum below | Required |
| `fps` | `float` | Source FPS when known | `>0` |
| `frame_count` | `int` | Source frame count when known | `>=0` |
| `duration_seconds` | `float` | Analysis interval duration (M2: interval length; M6: video duration) | `>0`, `end <= duration` enforced at coordinator |
| `interval` | `{start_seconds, end_seconds}` | Bounded interval under analysis | `0 <= start < end <= duration` when duration known |
| `license` | `str\|null` | License tag, e.g. `CC-BY-NC-4.0` | Carried from source docs |

#### FrameEvidence (M2: fixture-backed; M6: manifest-backed)

| Field | Type | Notes |
|---|---|---|
| `evidence_id` | `str` | Stable, e.g. `frm_{frame_index:08d}` |
| `video_id` | `str` | FK to VideoSource |
| `frame_index` | `int|null` | `null` for preliminary-sourced or non-frame fixtures |
| `timestamp_seconds` | `float\|null` | Under explicit `timestamp_basis` |
| `source_timestamp_seconds` | `float\|null` | `timestamp_seconds + source_offset` from manifest |
| `requested_timestamp_seconds` | `float\|null` | Caller's intended sample time |
| `image_ref` | `str\|null` | Local relative path / evidence ID; required for any `verified` observation |
| `region` | `{x,y,width,height}\|null` | Optional crop |
| `extraction` | `{sample_fps, source_offset, manifest_sha256}\|null` | M6 provenance |

#### Timestamp provenance (§3.1-mandated representation)

| Enum value | Meaning | Provenance carried with every timestamp |
|---|---|---|
| `frame_index_over_fps` | `frame_index / FPS` (constant-FPS assumption) | `fps`, `frame_index`, assumption note: **M1–M2 default** |
| `presentation_timestamp` | Container PTS | `pts_seconds` + container metadata ref — **M6+ only** |
| `approximate_human_video_review` | Human time label from `P04_104.json: approximate_timestamp_seconds` | Raw label + note: **never** a measurement timestamp; verifier downgrades any `confirmed` claim sourced only from this basis |
| `synthetic_fixture` | Deterministic fixture time | Fixture ID + generator version |

Assumption discipline: `frame_index_over_fps` may be used **only** when the analysis configuration documents the constant-FPS assumption and the coordinator has recorded it in `RunManifest.timestamp_assumption`. Variable-FPS video must be rejected or routed to a `presentation_timestamp` path (deferred to M6). This satisfies the plan requirement that frame-index/FPS division is only under a documented constant-FPS assumption.

#### Observation (canonical; `spec.md` §4)

| Field | Type | Constraint |
|---|---|---|
| `observation_id` | `str` | Unique within bundle |
| `modality` | `str` | `visual` for M2; `audio` deferred |
| `timestamp_seconds` / `interval{start,end}` | `float` / object | `>=0`, within interval; single timestamp vs interval is explicit |
| `timestamp_basis` | `str` | FK to taxonomy above; copied from source evidence |
| `observation_type` | `str` | `scale_display, ingredient, action, container, other` (from `weighing_event_schema.md`) |
| `scale_value` | `float\|null` | Displayed value if legible; **not** net mass |
| `scale_unit` | `str\|null` | `g, kg, oz, lb…`; `null` unless explicitly observed |
| `scale_unit_hypothesis` | `str\|null` | Hypothesis annotation (e.g. `"g"`) — never promoted to `scale_unit` |
| `display_state` | `str\|null` | `changing, stable, zero, occluded, unknown`; `stable` requires multi-frame evidence (schema rule) |
| `evidence_ids` | `str[]` | `FrameEvidence` IDs supporting this observation |
| `raw_reported_value` | `str\|null` | Original free-text (`"-750"`, `"54–55"`) preserved when parser cannot unambiguously produce `scale_value` |
| `description` | `str\|null` | Concise visual description |
| `provenance` | `{source_type, source_id}` | `human | model | reviewer | adapter` + source file/run ID |
| `verification_status` | `str` | `confirmed | ambiguous | unobservable`; adapter/model/preliminary sources are capped at `ambiguous`/`unobservable` (state 2); only the deterministic verifier may assign `confirmed` for pipeline records (state 3, §5.1/§7); reference `confirmed` (state 4) is a separate human-adjudication process (§5.3) |

#### ActionEvent

Per `docs/schemas/weighing_event_schema.md §Action events`: `action_id, start_seconds, end_seconds, action_type∈{add, remove, tare, place_container, remove_container, mix, other, unknown}, ingredient_id|null, evidence_observation_ids[], provenance, verification_status`.

#### WeighingEvent

Per `docs/schemas/weighing_event_schema.md §Weighing events`: `event_id, start_seconds, end_seconds, ingredient_id|null, action_ids[], observation_ids[], final_display_reading{value, unit, observation_id}, net_ingredient_quantity{value, unit, derivation}, measurement_mode∈{individual, cumulative, unknown}, tare_observed∈{bool|null}, verification_status, review_notes`. The `derivation` enum is `{direct_after_tare, difference_between_readings, independently_verified, unknown}` — a cumulative reading requires `derivation !== unknown` to yield a net quantity.

#### RecipeDraft (`spec.md` §4)

| Field | Type | Meaning |
|---|---|---|
| `draft_id` | `str` | Stable |
| `version` | `int` | Monotonic per draft lineage; approval is version-pinned |
| `video_id` | `str` | FK |
| `steps[]` | `[{index, description, action_ids[], observation_ids[], evidence_summary}]` | Ordered; every material step has evidence links or is explicitly marked unresolved |
| `ingredients[]` | `[{ingredient_id, quantity{value,unit,derivation, source_event_id}, status∈{supported, unknown, conflicting}}]` | Quantities only when `derivation !== unknown` |
| `unresolved_questions[]` | `[{code, message, related_entity_ids[]}]` | Required when any material claim lacks support |
| `evidence_refs[]` | `Observation[]` + `FrameEvidence[]` indices | Navigable claim→evidence |
| `verification_ref` | `str` | `VerificationReport` ID |
| `run_id` | `str` | `RunManifest` ID |
| `status` | `str` | `pending_review` (only initial status); `approved` only via `ReviewDecision` (never set by generator) |

#### VerificationReport (`spec.md` §4)

`{ report_id, verdict∈{pass, fail, review_needed}, issues[{code, entity_id, gate, message, severity}], checked_ids[], verifier_version, checked_at }`.

#### RunManifest (`spec.md` §4; Constitution VI)

`{ run_id, code_revision, started_at, finished_at, elapsed_ms, video_source_snapshot, run_config_snapshot, timestamp_assumption, budget_state, outputs{ledger_path, events_path, draft_path, verification_path}, attempts[], provider_calls[], errors[], status∈{completed, partial, failed, awaiting_review}, cost_summary }`.

#### ReviewDecision (`spec.md` §4; Constitution V)

`{ decision_id, draft_id, draft_version, reviewer_id, decision∈{approved, rejected, needs_revision}, notes, corrections_ref, decided_at }`.

---

## 5. Evidence Provenance and Append-Only Observation History

### 5.1 Provenance and verification-status taxonomy

Four evidence/interpretation states are distinct; each has its own promotion rule (§7 gates). A valid evidence reference never implies verified status.

| State | Representation | Example | `verification_status` / promotion |
|---|---|---|---|
| **(1) Frame-linked evidence** | `FrameEvidence` exists and `Observation.evidence_ids[]` resolves to it; `image_ref` is resolvable under the configured evidence root | `FrameEvidence{evidence_id: frm_00000120, image_ref: runs/<id>/frames/frame_00000120.jpg}` cited by an `Observation` | **Not verified** — existence only proves a frame was extracted, not that a scale reading is correct, stable, or tare-aware. Verifier G1 checks resolvability; G2/G3 check semantic preconditions before any `confirmed` quantity. |
| **(2) Model-generated interpretation** | `Observation` with `provenance{source_type: model, source_id: "<provider>/<model_id>@<prompt_hash>"}` or `adapter` | Live adapter output `"scale_value: 54"`; preliminary-adapter output from `P04_104.json` | `ambiguous` ceiling. May never be promoted to `confirmed` by the adapter or ledger alone. Requires G2 display-state / tare / unit checks and, where applicable, `measurement_mode` / `derivation` preconditions (§6). |
| **(3) Independently verified observation** | Same `Observation` after passing deterministic gates G1–G3 **and** satisfying §6 derivation preconditions; `verification_status` upgraded by verifier, not by generator | Synthetic fixture ground truth `Observation{scale_value:120, scale_unit:"g", display_state:stable}` with two compatible `stable` evidences and credible tare history | `confirmed` is a **verifier-assigned** state: G1 (structural + evidence resolvability), G2 (measurement derivation + stable-display support), G3 (draft-level coherence). No generator, adapter, or ledger code may assign `confirmed`. |
| **(4) Human-adjudicated reference label** | Separate bundle at `experiments/003_video_to_recipe/annotations/reference/`; never read by pipeline stages; used only for evaluation scoring | Independent adjudicator marks `WeighingEvent.net_ingredient_quantity: 120 g, derivation: direct_after_tare` with `verification_status: confirmed` | `confirmed` by **human adjudication process** with provenance `reviewer`; isolated from predictions per §5.3. |

**Promotion rules (normative):**

- Frame existence (`FrameEvidence.image_ref` resolvable) ⇒ no status change.
- Model/adapter provenance (`source_type: model | adapter | human` preliminary) ⇒ capped at `ambiguous`/`unobservable` until verifier upgrades.
- Verifier promotion to `confirmed` ⇒ requires all gate checks for that record type plus §6 derivation preconditions; recorded in `VerificationReport.checked_ids[]`.
- Reference labels ⇒ never read by `Evidence Acquisition`, `Observation Ledger`, `Temporal Reconstruction`, `Draft Projection`, or `Run Diagnostics` (isolation guarantee §5.3); reference corrections require a separately documented revision.

Preliminary observations carry `provenance: {source_type: "human", source_id: "P04_104_manual_observations.md"}` + `timestamp_basis: approximate_human_video_review` + `evidence: {frame_index: null, image_ref: null, verification_status: not_frame_verified}` (per `P04_104.json`). The adapter preserves all three as first-class provenance; it never synthesizes `frame_index`/`image_ref` and never assigns `confirmed`.

### 5.2 Ledger design

- **Physical form:** One **JSONL ledger** per run: `runs/<run_id>/ledger.jsonl` (one JSON object per line, ordered chronologically). A **snapshot** `observations.json` aggregates the current view for convenience. JSONL is the source of truth; the snapshot is a derived projection. This satisfies the migration guide M1 decision point 2 (append-only, inspectable, deterministic) without a database.
- **Append-only invariant:** Ledger entries are never updated in place. A correction appends a new entry with `supersedes: <prior_observation_id>` and a `reason` string; the prior entry gains a back-pointer `superseded_by` in the derived snapshot but remains in the log.
- **Contradiction handling:** Conflicting readings on the same interval append separate entries; the reconstructor and verifier surface them as `issues` (VTR-AC-05) — no silent consensus or voting (Domain Rule 8).
- **Referential posture:** Every `observation_id`, `action_id`, `event_id` is unique within the bundle (schema Validation invariants). Ledger indices are secondary.

### 5.3 Reference annotations — isolation guarantee

- Human **reference** annotations live at `experiments/003_video_to_recipe/annotations/reference/` (or future `data/reference/` local-only) and are read-only inputs to evaluation scoring.
- **Predictions** live at `predictions/<run_id>/` and **evaluations** at `evaluations/<run_id>/metrics.json` (per `docs/schemas/weighing_event_schema.md` recommended layout).
- No pipeline stage reads reference data; no evaluation result writes back to reference (Constitution II, VII). Reference corrections require a separately documented revision.

---

## 6. Tare-Aware Quantity Reasoning

This section enforces `spec.md §5` Domain Rules and Constitution III with **explicit unknown** discipline. The reconstructor never converts a `scale_value` into a `net_ingredient_quantity` by default. Satisfying these preconditions establishes semantic consistency for a quantity claim; it does not certify pixel-level reading correctness (§5.1, §7).

### 6.1 Core invariant

```
net_ingredient_quantity ∈ {
  value: number|null,
  unit: string|null,
  derivation: "direct_after_tare" | "difference_between_readings"
              | "independently_verified" | "unknown"
}
// derivation == "unknown"  ⇒  value==null && unit==null (enforced by verifier G3)
```

### 6.2 Derivation preconditions (all must hold to leave "unknown")

| `derivation` | Precondition (all required) | Counter-example from P04_104.json where it would be wrong |
|---|---|---|
| `direct_after_tare` | Credible tare history on same video + relevant unit evidence + post-tare `scale_value` with `verification_status: confirmed` + no intervening container loss | `pre_s1_02: 100` before tare — tare not yet observed, so cannot be direct |
| `difference_between_readings` | Two compatible measurement states (same unit, same container context, both `stable`/`confirmed` or explicitly verified) + explicit `derivation_trace{from_observation_id, to_observation_id, delta}` | `pre_s3_05: 600` during stirring vs `60` before — states incompatible (force artifact), so difference is invalid |
| `independently_verified` | External independent measurement referenced in `review_notes` + human review provenance | No case in P04_104 supports this yet |
| `unknown` (default) | Fallback when any precondition missing | `pre_s1_03: -750` (negative after removal), `pre_s3_03: 54–55` (ambiguous), `pre_s3_05: Lo` (not a mass), any `not_frame_verified` reading |

### 6.3 Hard rules (must be tested — see `tasks.md`)

1. **Zero after tare is not emptiness.** `display_state: zero` after a tare records `tare_observed: true` and `review_notes` but does not assert container mass.
2. **Negative readings are not ingredients.** Negative `scale_value` is preserved as `final_display_reading` but `net_ingredient_quantity` remains `unknown` unless an independent verification explains it.
3. **One sharp frame ≠ stable.** `display_state: stable` requires multi-frame evidence per schema design principle; a single legible crop cannot produce `stable`.
4. **Cumulative vs individual mode.** Unless `measurement_mode: individual` is established by evidence (e.g. verified tare between additions), `measurement_mode` is `unknown`/`cumulative` and readings are not net quantities.
5. **Units are explicit.** `42` with no visible unit is `scale_value: 42, scale_unit: null` and cannot yield `net_ingredient_quantity{value:42, unit:"g"}`.

### 6.4 Observability of abstention

Every abstention emits `WeighingEvent{ net_ingredient_quantity{derivation:"unknown"}, review_notes: <why> }` — never a silently missing event. The draft projection surfaces these as `unresolved_questions`.

---

## 7. Independent Verification Gates

Constitution IV: generators never waive validation. Verification is **pure, deterministic, and outside model-authored content paths** (migration guide §5 Independent enforcement). Deterministic verification establishes structural and semantic consistency (IDs, references, intervals, derivation preconditions, status discipline) — not factual correctness of visual interpretation (whether a scale display was read correctly from pixels). Factual correctness requires independent frame inspection or human adjudication (§5.1 states 3–4).

### 7.1 Three gates

| Gate | When | Payload | Hard invariants checked |
|---|---|---|---|
| **G1 — Normalization** | After Evidence Acquisition + Adapter → canonical records | `VideoSource + FrameEvidence[] + Observation[] + LedgerEntry[]` | ID uniqueness; `0 <= start < end <= duration`; timestamps `>=0` and within interval; `frame_index`↔`timestamp_seconds` consistency under declared `timestamp_basis`; evidence refs resolve or are explicitly absent-not-verified; no `confirmed` status from adapter-only provenance |
| **G2 — Event Assembly** | After Temporal Reconstruction → candidate events | `ActionEvent[] + WeighingEvent[]` | All `evidence_observation_ids`/`observation_ids`/`action_ids` resolve within bundle; event intervals within source interval; `WeighingEvent.final_display_reading` points to a real observation; `net_ingredient_quantity` obeys §6.1; `stable` display has multi-evidence support per observation it cites |
| **G3 — Draft Submission** | After Draft Projection → reviewable draft | `RecipeDraft + ReviewDecision attempt + RunManifest draft` | Every material claim has `observation_ids` or is in `unresolved_questions`; `status != approved` (no auto-approval); `verification_ref` points to a passing/review_needed report (not `fail`); no unknown silently dropped; `RunManifest.status` truthfully reflects `completed|partial|failed|awaiting_review` |

### 7.2 Feedback shape

- On `fail`/`review_needed`, the gate returns a `VerificationReport` with **machine-readable** `issues[{code, entity_id, gate, severity, message, suggested_action}]` — never a free-form string alone.
- A single unresolved issue may trigger **one bounded retrieval/retry** (M4 design), but retries **never weaken invariants** (Domain Rule 10) and preserve all prior failed versions.
- Human escalation is the terminal safe path when evidence remains insufficient (migration guide §5 Bounded review/reanalysis).

---

## 8. Human Approval Boundary

### 8.1 State machine (VTR-FR-09, Constitution V)

```
  [Draft Projection] ──►  RecipeDraft{ status: "pending_review", version: N }
                                     │
         ┌───────────────────────────┼──────────────────────────┐
         │                           │                          │
  ReviewDecision{              ReviewDecision{            ReviewDecision{
   decision:"approved",        decision:"rejected",       decision:"needs_revision",
   version:N,                  version:N }               version:N,
   reviewer_id, decided_at }   (+ notes) }               corrections_ref }
         │                           │                          │
         ▼                           ▼                          ▼
   RecipeDraft V_N            (terminal for this         RecipeDraft V_{N+1}
   approved (historical)       version; new analysis      (new version,
                               may be started)             pending_review)
         │                                                  │
         └──────────── audit trail ────────────────────────┘
         Original draft + every ReviewDecision retained (correction provenance).
```

- **No automatic path** to `approved` exists (Constitution V acceptance gate). The verifier at G3 rejects any `status: approved` not paired with a valid `ReviewDecision`.
- Approval is **per-version** (`spec.md` VTR-AC-09): approving `V_N` does not approve `V_{N+1}`.
- Corrections create a new draft version while preserving the original + `corrections_ref` provenance (`spec.md` §7).

### 8.2 Demo-grade interface (no production UI)

For M2–M5, approval is a local CLI operation that appends a `ReviewDecision` JSON file and updates the draft index:

```bash
bakery-intelligence review approve <draft_id> --version <N> --reviewer <id> --notes "<text>"
bakery-intelligence review reject  <draft_id> --version <N> --reviewer <id>
```

This satisfies migration guide decision 8 without inventing a service.

---

## 9. Reproducible Runs, Bounded Costs, and Persistence

### 9.1 RunManifest (Constitution VI; `spec.md` VTR-FR-10/11; `validation_protocol.md` Reproducibility)

Every run — including `partial`/`failed` — persists at `runs/<run_id>/manifest.json`. The manifest is staged: M2 seeds a minimal manifest before the offline CLI requires it; M5 extends that seed to the full field set below (see §3.2 Run Diagnostics and §10).

| Field group | Contents | Stage |
|---|---|---|
| Identity | `run_id` (UUID or deterministic hash of input+config+revision), `code_revision` (`git rev-parse HEAD` or `unknown-dirty` with warning), `started_at` | M2 minimal seed |
| Identity (finalization) | `finished_at`, `elapsed_ms` | M5 extension |
| Input snapshot | Frozen `VideoSource`, `RunConfig`, `timestamp_assumption`, input file hashes (`sha256`) | M2 minimal seed |
| Outputs | Paths for `ledger.jsonl`, `events.json`, `draft.json`, `verification.json` | M2 minimal seed (paths); M5 adds `sha256` hashes |
| Budgets | `{frame_budget, model_budget, cost_budget_usd, retry_limit}` + `budget_state{consumed, remaining, exhausted_at}` | M2 minimal seed |
| Provider usage | `provider_calls[{provider, model_id, prompt_hash, settings, started_at, latency_ms, tokens{prompt, completion}, cost_usd, error_code}]` — **no raw media in manifest** if dataset license forbids | M5 extension (empty offline in M2) |
| Attempts & errors | Ordered `attempts[{stage, verdict, issues_ref}]` + `errors[{stage, code, message}]` | M5 extension |
| Final status | `status ∈ {completed, partial, failed, awaiting_review}` + `status_reason` (e.g. budget exhausted per VTR-AC-06) | M2 minimal seed |
| Cost | `cost_summary` | M5 extension (zero offline in M2) |

A stopped/failed run retains a useful partial record with truthful status (Constitution VI acceptance gate). The offline CLI (M2) depends on the minimal seed writer, not the reverse: the writer exists before the orchestrator lifecycle that calls it (`tasks.md` M2-07 precedes M2-06).

### 9.2 Bounded execution (VTR-FR-11, VTR-AC-06/07)

- The coordinator enforces `frame_budget` (max frames/observations processed), `model_budget` (max provider calls — zero in M2 offline), and `cost_budget_usd` (M6+). Exhaustion yields `status: partial` with `status_reason` and available evidence preserved — never invented details.
- Retries per stage are capped by `retry_limit`; the manifest records every retry and never silently loosens a verification gate.
- Provider-unavailable (VTR-AC-07) yields a structured `error_code` and recoverable run state — no invented observations.

### 9.3 Persistence layout

```
runs/<run_id>/
├── manifest.json          # RunManifest
├── source.json            # frozen VideoSource + RunConfig snapshot
├── ledger.jsonl           # append-only Observation history
├── observations.json      # derived snapshot (convenience)
├── frame_evidence.json    # FrameEvidence[]
├── events.json            # ActionEvent[] + WeighingEvent[]
├── draft.json             # RecipeDraft (pending_review)
├── verification.json      # VerificationReport
└── review/                # populated post-M2
    └── <decision_id>.json
fixtures/synthetic/        # committed, offline fixtures
└── synthetic_interval_*.json
data/raw/epic_kitchens/    # git-ignored, local-only (CC-BY-NC-4.0)
predictions/<run_id>/      # isolated from reference per schema recommended layout
evaluations/<run_id>/      # metrics, per validation_protocol
```

Naming is deterministic / idempotent per bundle; reruns with identical inputs do not overwrite prior `run_id` directories.

Synthetic fixtures fall into three classes with different expected outcomes (`tasks.md` M2-01): (a) **structurally invalid** (`synthetic_interval_dangling_ref.json`) — referentially broken, must yield G1/G2 `fail` with `DANGLING_REFERENCE` and no draft; (b) **valid structure, uncertain semantics** (`synthetic_interval_unknown_tare.json`, `synthetic_interval_conflicting.json`) — pass structural checks but yield `unknown` quantities with `unresolved_questions` or `review_needed` with contradictions preserved; (c) **valid structure, budget-exhausted execution** (`synthetic_interval_partial_budget.json`, or nominal under `--frame-budget 1`) — structurally valid but yields CLI `partial` with `BUDGET_EXHAUSTED` and preserved artifacts. Only the nominal fixture is expected to complete cleanly.

---

## 10. Migration Stages M0–M7 — Acceptance Gates

Stages follow `specs/video-to-recipe/migration-guide.md` §4; exit conditions below are **gated** (stage is not complete without them).

| Stage | Objective | Reuse | New Components | Acceptance Gate |
|---|---|---|---|---|
| **M0 — Baseline audit** | Dated, evidence-backed capability/gap/contradiction inventory | All inspected docs + code | `specs/video-to-recipe/assessment.md` | Human can see a trustworthy Implemented/Proposed/Gap matrix with file citations and explicit Unknowns; no code changes. ✅ This plan's prerequisite. |
| **M1 — Reconcile & version core contracts** | One authoritative v1.0.0 contract + preliminary adapter design + contract tests planned | `weighing_event_schema.md` as domain anchor; `P04_104.json` as adapter test case; `pyproject.toml` Pydantic | Versioned Pydantic models, JSON Schemas, adapter spec; tests for invalid refs, timestamps, unknowns, measurement provenance | `pytest tests/test_contracts.py` passes: valid bundles pass, invalid bundles fail with named issues; preliminary→canonical adapter never mutates source, never promotes `not_frame_verified` to `confirmed`. |
| **M2 — Offline fixture-first end-to-end slice** | Reproducible minimal `[fixture → ledger → events → draft(pending_review) → verification → manifest]` without network/credentials. Introduces each pipeline capability in minimal form. | Fixture design, existing extractor docs (not its execution), M1 contracts | Minimal coordinator, ledger, reconstructor (core §6.1 + §6.2 happy-path + `unknown` fallback), draft projector, minimal structural G1–G3 gates, offline CLI, synthetic fixtures, minimal RunManifest seed (§9.1) | `bakery-intelligence run --fixture fixtures/synthetic/... --offline` produces ordered events, evidence-linked `RecipeDraft{pending_review}`, persisted `VerificationReport` (structural/semantic consistency), and an inspectable minimal `RunManifest` seed (full §9.1 fields deferred to M5); VTR-AC-01/04/08/10 pass offline. No draft can be `approved` without a `ReviewDecision`. |
| **M3 — Event memory & conservative mass** | Harden the M2 reconstructor with full chronological weighing history and abstention semantics; extends M2, does not reintroduce the pipeline. | Synthetic fixtures, M2 ledger + minimal reconstructor | Extended derivation matrix (§6.2 full cases), cumulative-vs-individual disambiguation, hardened zero/negative/single-frame/conflicting/force-artifact guards | Beyond the M2 happy-path: deterministic fixtures for tare, cumulative, zero-after-tare, negative, conflicting, and force-artifact readings all yield correct `unknown`/derivation; evidence refs survive transitions (`spec.md` §10, VTR-AC-02/05). |
| **M4 — Independent verification & bounded correction** | Harden the M2 minimal gates into enforced blocking verification with bounded correction; human escalation is terminal safe path. | M2 minimal G1–G3 | Blocking-gate enforcement audit, bounded retry/retrieval envelope, partial-result path, verifier versioning (extends M2 minimal gates) | Invalid/unsupported quantities are `fail` or `review_needed` with actionable issues; auto-approval is rejected even under retries; verification version is recorded. |
| **M5 — Durable & reproducible runs** | Extend the M2 minimal manifest seed to durable, fully reproducible runs with stable naming. | M1–M4 artifacts, manifest design (§9) | Full RunManifest writer (§9.1: attempts, provider calls, cost, output hashes, `finished_at`/`elapsed_ms`), checkpoint convention (resume deferred); extends the M2 seed, does not reintroduce it | Fixture run is fully inspectable post-execution; failed/partial runs preserve provenance and never falsely report `completed`. |
| **M6 — Connect to real visual evidence** | Manifest-backed frames through a cost/time-bounded provider adapter for a supported interval | `extract_frames.py` + `manifest.csv`, M2 pipeline | Provider adapter, PTS or validated FPS path if needed, prompt/model recording | Real-media run has persisted evidence + honest draft/partial outcome; if `P04_104.MP4` or credentials absent, the blocked gate is reported — not simulated. |
| **M7 — Controlled evaluation & expansion** | Frozen-rubric comparison of configs A/B/(C) on ≥10 independent holdouts; audio introduced only as evaluable claim-aligned signal | `validation_protocol.md` protocol | Holdout registry, scoring harness, cost/runtime accounting | Results have immutable inputs, stated denominators, preserved failures/abstentions, and no preliminary-reference leakage; if not executed, reported as `Not evaluated` (Constitution VII). |

### Effort guidance

M0–M2 are the **smallest independently verifiable vertical slice** (2–4 small PRs). M2 introduces each pipeline capability in minimal form; M3–M5 extend and harden those existing capabilities rather than reintroducing them. M6–M7 are integration/evaluation stages that depend on licensed media and optional provider credentials. No time estimate is asserted as a commitment; ordering is by dependency, not a schedule promise.

---

## 11. Contract Migration Matrix

Existing `preliminary_schema_version 0.1.0` → proposed canonical `schema_version 1.0.0`.

| Existing field (0.1.0) | Proposed field (1.0.0) | Compatibility | Handling | Test required |
|---|---|---|---|---|
| `preliminary_schema_version` + `record_type: preliminary_human_review` | `schema_version: "1.0.0"` on every bundle | **Breaking — type boundary** | Source file immutable; adapter emits canonical bundle with `provenance: {adapter, source_id: P04_104.json}` and preserves original version as `source_schema_version` provenance | Round-trip adapter test: source unchanged, output validates as 1.0.0 |
| `annotation_status: preliminary_not_frame_verified` | `verification_status: confirmed｜ambiguous｜unobservable` **per observation** | **Semantic refinement** | `preliminary_not_frame_verified` → `observation.verification_status: unobservable` for evidence, `ambiguous` for interpretations; never `confirmed` | Adapter test asserts no `confirmed` produced from preliminary inputs |
| `segments[].observations[].approximate_timestamp_seconds` | `Observation.timestamp_seconds` + `timestamp_basis` | **Semantic narrowing** | Copied to `timestamp_seconds` only with `timestamp_basis: approximate_human_video_review`; G1 verifier penalizes any `confirmed` mass derived solely from this basis | Timestamp-basis propagation test; G1 downgrade test |
| `evidence{ frame_index: null, image_ref: null, verification_status: not_frame_verified }` | `FrameEvidence{ frame_index, image_ref, … }` + `Observation.evidence_ids[]` | **Structural — absent evidence** | Adapter emits zero `FrameEvidence` for null entries; any downstream `WeighingEvent` lacking resolvable `image_ref` must be `unknown`/G2 `review_needed` | Dangling-evidence verifier test |
| `reported_display_or_state: "`-750`"` (free text) | `scale_value: number\|null, scale_unit, display_state, raw_reported_value` | **Structured replacement** | Adapter parses only unambiguous numeric displays; ambiguous (`"54–55"`, `"partially occluded 54"`) → `scale_value: null, raw_reported_value: original text`; never invents digits | Parser-no-invention test with P04_104 hard cases |
| `reported_action` + `interpretation_and_verification_need` | `ActionEvent{ action_type, ingredient_id, review_notes }` + ledger entry `description` | **Lossy if flattened** | `reported_action` → `description`; `interpretation_and_verification_need` → `review_notes` on derived event, not a `confirmed` fact | Notes-preservation test |
| `scale_unit: null` + `scale_unit_hypothesis: "g"` | `Observation.scale_unit: null` + `scale_unit_hypothesis: "g"` (annotation) | **Policy — hypothesis isolation** | Hypothesis stored in `scale_unit_hypothesis` annotation field; `scale_unit` remains `null` until independent frame evidence confirms it | Hypothesis-not-promoted test |

---

## 12. Verification Design Summary

Gates, feedback, uncertainty, and escalation are detailed in §7–§8. In brief:

- **Gates:** G1 (Normalization), G2 (Event Assembly), G3 (Draft Submission) — all blocking, all pure, all outside generator control.
- **Failure feedback:** `VerificationReport{ verdict, issues[{code, entity_id, gate, severity, message, suggested_action}] }` — actionable and machine-readable.
- **Uncertainty:** `null` values, `derivation: unknown`, `measurement_mode: unknown`, `unresolved_questions[]` — never dropped, always surfaced to reviewer.
- **Human escalation:** `review_needed` verdict is the safe terminal path when evidence is insufficient; a reviewer decision is the only way to reach `approved`.
- **Approval:** No automatic transition (Constitution V acceptance gate). Every approval is version-pinned and audited.
- **Semantics:** a `pass` verdict attests structural and semantic consistency only. It does not certify that a scale display, ingredient, or action was interpreted correctly from pixels; frame-linked evidence (state 1) and model-generated interpretation (state 2) remain distinct from verifier-checked consistency (state 3) and human-adjudicated reference labels (state 4) per §5.1.

---

## 13. Rejected Alternatives and Why

| Alternative | Why rejected for P0 |
|---|---|
| Evolve `preliminary_schema_version 0.1.0` in place as the domain contract | Couples the authoritative contract to a single atypical, preliminary, `not_frame_verified` video; risks baking `approximate_human_video_review` timestamps and free-text fields into the normative model. |
| Use P04_104 observations as the sole basis for domain records | Contradicts plan requirement and Constitution II/IX; the four segments (low-battery `Lo`, freezer transition, battery gap, S04 scale interruption) are insufficient to define a general contract; v1.0.0 schema must remain the anchor. |
| Single `frame_index / FPS` timestamp with no basis tagging | Loses provenance needed to distinguish an estimate from a PTS or a human label — violates preservation requirement. |
| Treat `0` after tare as empty-container evidence | Violates Domain Rule 4 — zero indicates a tared baseline, not container state. |
| Infer `scale_unit: "g"` from hypothesis or numeric plausibility | Violates Constitution III and plan requirement: unit is `null` until explicitly observed. |
| Promote model observations to `confirmed` to improve pass rates | Violates Domain Rules 8/10 and plan requirement: model agreement is a signal, not truth; hard invariants cannot be overridden by retries. |
| Distributed queue / multi-agent orchestrator for M2 | Violates Constitution VIII — no measured requirement justifies it; a deterministic orchestrator suffices for offline P0. |
| Auto-approve `RecipeDraft` when verification passes | Violates Constitution V — schema validation is not human approval. |

---

## 14. Assumptions

1. **Baseline commit:** `experiment/003-video-to-recipe` at declared `548230b`. Could not be re-verified in this session due to filesystem-ownership sandbox blocking `git status/log`; must be confirmed before any M1 commit (`assessment.md §1`).
2. **Synthetic fixtures can carry P0.** Deterministic fixtures exercising tare, cumulative, unstable, and conflicting cases are adequate to exercise §6/§7 logic without real media.
3. **Pydantic v2 available** with the declared `pydantic>=2.14.1` dependency.
4. **`extract_frames.py` reuse** for M6 will not require changing its half-open `[start, end)` interval or 4-field manifest — only wrapping it with the typed `FrameEvidence` + `RunManifest` provenance.
5. **No audio in P0.** Audio remains Conditional P2 (`spec.md` §1) and is exercised only as a timestamped claim aligned by meaning (H2 stays `Not evaluated`).
6. **`P04_104.MP4` is local-only** (git-ignored, CC-BY-NC-4.0) per `docs/experiments/003_video_to_recipe.md` and migration guide §2 — all CI-safe tests use synthetics.

---

## 15. Decisions Requiring Human Approval

Only the critical unresolved decisions are listed (per migration guide §6). Each item names alternatives, a recommendation, and required approval before implementation.

1. **Contract evolution.** *Alternatives:* (a) extend `weighing_event_schema.md` in place, (b) introduce additive Pydantic models + explicit mapper preserving both (recommended — §11, preserves provenance, avoids overfitting to P04_104). *Needs approval:* versioning policy and where JSON Schemas export.
2. **Event ledger representation.** *Alternatives:* append-only JSONL (recommended — §5.2), versioned JSON documents, or a DB. *Rationale:* JSONL is inspectable, deterministic, git-friendly; no infra justified for P0. *Needs approval:* ledger vs snapshot file naming.
3. **Verifier invariants at each gate.** *Alternatives:* minimal structural only vs §7's hard invariants (recommended — ID uniqueness, referential integrity, derivation→unit/value coherence, no auto-approval). *Needs approval:* whether to freeze these as blocking for M2.
4. **Recipe granularity for partial intervals.** *Alternatives:* always claim full recipe; emit partial procedure with explicit gaps (recommended — `spec.md §10` Definition of done allows partial per-interval procedures). *Needs approval:* minimum step schema for fragmentary intervals.
5. **Provider abstraction envelope.** *Alternatives:* provider-specific paths vs uniform `EvidenceRequest → TypedResult + ProviderMetadata|Error` (recommended — §3.2). *Needs approval:* exact metadata fields (prompt hash, token accounting) and media-retention policy under license constraints.
6. **Temporal basis.** *Alternatives:* raw `frame/FPS` everywhere, vs typed `timestamp_basis` per record with documented constant-FPS assumption (recommended — §4.2 taxonomy). *Needs approval:* that `frame_index_over_fps` is the M1–M2 default and PTS is deferred to M6.
7. **Run persistence scope.** *Alternatives:* minimal log vs full `RunManifest` with budgets/costs/retry history (recommended per §9.1). *Needs approval:* manifest fields required for M2 Definition of done.
8. **Human review interface.** *Alternatives:* web UI, messaging bot, vs CLI + JSON decision file (recommended — §8.2). *Needs approval:* reviewer identity source and approval authority.
9. **Licensed media boundary.** *Alternatives:* commit crops/media, vs synthetics in repo + real media local-only (recommended — §9.3 layout). *Needs approval:* CI policy for `data/raw/epic_kitchens/`.
10. **Verification of source evidence.** *Alternatives:* treat P04_104 as ground truth, vs keep preliminary isolated and require independent frame-verified reference for evaluation (recommended — `assessment.md §6` + `spec.md §5` Rule 9). *Needs approval:* who qualifies as independent reference adjudicator.

Migration decisions 2–10 above map 1:1 to `specs/video-to-recipe/migration-guide.md` §6 questions 1–10.

---

## 16. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Dataset licensing (CC-BY-NC-4.0) blocks redistribution of media/crops | High | CI breakage / legal | Synthetics in repo; real media local-only with `.gitignore`; manifests carry hashes not pixels |
| Timestamp drift (constant-FPS assumption violated) | Medium | Event-order / tare errors | Explicit `timestamp_basis` on every record; G1 consistency check; reject variable-FPS or defer to PTS |
| Reference leakage (P04_104 used as test oracle) | Medium | Invalid evaluation | Isolation guarantee (§5.3); ≥10 independent holdouts per `validation_protocol.md`; preliminary never treated as adjudicated ground truth |
| Model variability / cost | Medium | Nondeterministic, expensive runs | Offline fake adapter for M2; bounded budgets + deterministic fixtures; provider metadata fully recorded when live |
| Hard invariants perceived as too strict | Medium | Pressure to weaken verifier | Approval-gated invariant freeze (§7); retries never weaken rules (Domain Rule 10) |

---

## 17. First Implementation Milestone

The first implementation milestone is **M1 + the offline fixture-first vertical slice (M2)**. The ordered, dependency-aware breakdown with contracts, verification commands, acceptance criteria, and API requirements is in `specs/video-to-recipe/tasks.md`.

---

## 18. Traceability

- **Normative:** `specs/constitution.md`, `specs/video-to-recipe/spec.md`, `docs/schemas/weighing_event_schema.md`
- **Process:** `specs/video-to-recipe/migration-guide.md`, `specs/video-to-recipe/assessment.md`
- **Product scope:** `docs/architecture/system_proposal.md`, `docs/project/hackathon_mvp_scope.md`
- **Evaluation:** `docs/evaluation/validation_protocol.md`, `docs/experiments/003_video_to_recipe.md`
- **Raw evidence:** `experiments/003_video_to_recipe/annotations/preliminary/P04_104.json`, `experiments/003_video_to_recipe/extract_frames.py`, `experiments/003_video_to_recipe/validate_preliminary.py`

---

*This plan is the first half of the Phase B handoff. Read it together with `assessment.md` and `tasks.md`. Implementation requires a separate, explicit human go-ahead per `AGENTS.md` Phase C.*
