# Video-to-Recipe — Repository Assessment

**Status:** Phase A assessment · Planning evidence only  
**Specification ID:** VTR-001  
**Baseline:** `experiment/003-video-to-recipe` at `548230b` (declared in `AGENTS.md`, `specs/constitution.md`, `specs/video-to-recipe/migration-guide.md`)  
**Assessment date:** 2026-10-27  
**Worktree verification:** Not independently verified in this session (git inspection blocked by filesystem ownership sandbox: `dubious ownership` error on `git status`/`git log`/`git branch --show-current`). The checked-out commit and cleanliness are **Unknown** by direct inspection.  
**Mode:** Read-only document inspection. No dependencies installed, no tests executed, no inference or external API calls made. All claims below distinguish inspected facts from unverified assumptions.

Related documents: `specs/constitution.md` (normative principles), `specs/video-to-recipe/spec.md` (VTR-001 P0/P1/P2 scope), `specs/video-to-recipe/migration-guide.md` (M0–M7 framework), `docs/schemas/weighing_event_schema.md` (v1.0.0 domain contract), `docs/evaluation/validation_protocol.md` (holdout protocol), `docs/architecture/system_proposal.md` and `docs/project/hackathon_mvp_scope.md` (proposed architecture and MVP priorities), `docs/experiments/003_video_to_recipe.md` (experiment 003 design).

---

## 1. Inspection Scope and Limitations

### Documents inspected in full
- `AGENTS.md`
- `specs/constitution.md`
- `specs/video-to-recipe/spec.md`
- `specs/video-to-recipe/migration-guide.md`
- `docs/architecture/system_proposal.md`
- `docs/project/hackathon_mvp_scope.md`
- `docs/schemas/weighing_event_schema.md`
- `docs/evaluation/validation_protocol.md`
- `docs/experiments/003_video_to_recipe.md`
- `experiments/003_video_to_recipe/annotations/preliminary/P04_104.json` (complete JSON, v0.1.0)
- `pyproject.toml` + `.python-version` (project metadata only)
- `README.md`, `docs/project/*`, `docs/datasets/*` (titles)

### Source code inspected (sampled, not exhaustive)
Source bodies opened for: `experiments/003_video_to_recipe/extract_frames.py`, `experiments/003_video_to_recipe/validate_preliminary.py`, `experiments/003_video_to_recipe/annotations/preliminary/P04_104.json`. Package scaffold `src/bakery_intelligence/__init__.py` was referenced via migration guide inventory but not re-opened in this Phase B pass. Test files `tests/test_extract_frames.py`, experiment folders `experiments/002_ocr_scales/`, `experiments/004_baseten_api/`, and `data/` contents were **not** opened — status for those components is **Unknown / Not evaluated**.

### What was NOT done
No `pytest` run, no `extract_frames.py` execution, no frame decoding, no model/API invocation, no retrieval of `P04_104.MP4` or extracted frames, no dependency installation. Therefore no capability is labeled **Verified** via execution in this assessment; **Verified** below means corroborated by inspectable implementation artifact, not by a fresh test run.

### Baseline uncertainty carried forward
The declared baseline commit `548230b` is taken from governing documents, not from `git rev-parse HEAD` (which failed). Any drift between the declared baseline and the actual worktree is **Unknown**. The assessment records this gap rather than assuming alignment, per Constitution IX and migration guide M0 exit condition.

---

## 2. Repository Structure Observed

```
bakery-intelligence/
├── AGENTS.md, specs/constitution.md, specs/video-to-recipe/{spec.md, migration-guide.md}
├── docs/{architecture/system_proposal.md, project/hackathon_mvp_scope.md,
│         schemas/weighing_event_schema.md, evaluation/validation_protocol.md,
│         experiments/003_video_to_recipe.md, datasets/}
├── experiments/003_video_to_recipe/{extract_frames.py, validate_preliminary.py,
│         annotations/preliminary/P04_104.json, annotations/preliminary/P04_104_manual_observations.md}
├── experiments/{002_ocr_scales/, 004_baseten_api/}   # not inspected
├── scripts/{muse-meta.ps1, muse-baseten.ps1}     # not inspected
├── src/bakery_intelligence/__init__.py               # scaffold per migration guide §2
├── tests/test_extract_frames.py                      # not opened
├── pyproject.toml (Python >=3.12, uv_build, deps: openai, opencv-python, pydantic, httpx, numpy, pillow, python-dotenv; dev: pytest)
├── .python-version (3.12)
└── data/raw/epic_kitchens/P04_104.MP4                # git-ignored, not inspected; licensed CC-BY-NC-4.0
```

---

## 3. Capability Inventory

Classification vocabulary per Constitution X and `AGENTS.md` evidence rules. **Proposed** = documented as intended behavior. **Unknown** = insufficient direct inspection in this session. **Not evaluated** = protocol exists but no executed result inspected.

| ID | Capability (maps to VTR-FR-* / VTR-AC-*) | Classification | Evidence / File References | Gap / Risk |
|---|---|---|---|---|
| C-01 | Video ingestion with bounded interval & timestamp basis (VTR-FR-01, VTR-AC-10) | **Partially Implemented** | `extract_frames.py: extract_frames()` validates `0 <= start < end`, `sample_fps>0`, `end <= duration`, `source_offset>=0`; half-open `[start, end)` sampling; `pyproject.toml` declares `opencv-python>=5.0.0.93`. No `VideoSource` model or `RunManifest` persists `video_id`/`timestamp_basis`. | No `VideoSource` record; source media itself is git-ignored and unverified locally. |
| C-02 | Time-indexed frame evidence with stable IDs & manifest (VTR-FR-02, VTR-AC-01) | **Partially Implemented** | `extract_frames.py` derives `frame_index=int(req*FPS+0.5)`, records `image_path, frame_index, timestamp_seconds (=frame/FPS), source_timestamp_seconds (+offset), requested_timestamp_seconds`; emits `manifest.csv` with `MANIFEST_FIELDS`. Supports offsets and duplicate-index guard. | Timestamps assume constant FPS; no presentation-timestamp (PTS) path; variable-FPS video unverified. No `FrameEvidence` typed contract enforces this. |
| C-03 | Distinguish human / model / reviewer / adjudicated provenance (VTR-FR-03, VTR-AC-03) | **Proposed** | `docs/schemas/weighing_event_schema.md §Frame observations` defines `provenance{source_type, source_id}` and `verification_status{confirmed, ambiguous, unobservable}`; `P04_104.json` sets `provenance` consistently. Constitution II and `spec.md §4` require separation of Observation/Inference/Reference/Approval. | `P04_104.json` field names diverge (`approximate_timestamp_seconds`, `reported_action`, `reported_display_or_state`, `evidence{verification_status:not_frame_verified}`) — not conforming to the normative schema; no enforcement layer prevents promotion. |
| C-04 | Preliminary human observations for P04_104 (development reference) | **Observed but not independently verified** | `experiments/003_video_to_recipe/annotations/preliminary/P04_104.json`: `preliminary_schema_version 0.1.0`, `record_type: preliminary_human_review`, `annotation_status: preliminary_not_frame_verified`, `timestamp_basis: approximate_human_video_review`, `scale_unit:null` with `scale_unit_hypothesis:"g"`, 4 segments, dozens of `pre_s*_XX` observations with `frame_index:null, image_ref:null`, explicit `interpretation_and_verification_need` per item. `source_annotation_commit: e00dbd1`; source annotations at `.../P04_104_manual_observations.md`. | Every observation is `not_frame_verified`; no frame-linked evidence; `interpretation_policy` correctly encodes `reported_display_values_are_not_net_ingredient_quantities`. Must NOT be used as ground truth. |
| C-05 | Structural validation of preliminary records | **Partially Implemented** | `validate_preliminary.py: validate()` checks `record_type`, `annotation_status`, `scale_unit is None`, `duration_seconds>0`, unique `observation_id`/`candidate_id`, `0<=timestamp<=duration`, and that `not_frame_verified` observations carry no `frame_index`/`image_ref`. No execution in this session. | Validates only the `0.1.0` preliminary contract, not `weighing_event_schema.md v1.0.0`; does not cover `ActionEvent`/`WeighingEvent`/`RecipeDraft`/`VerificationReport` rules. |
| C-06 | Weighing-event domain contract v1.0.0 | **Proposed — documented** | `docs/schemas/weighing_event_schema.md` defines `Video metadata`, `FrameObservation`, `ActionEvent{start_seconds,end_seconds,action_type,ingredient_id,evidence_observation_ids}`, `WeighingEvent{final_display_reading, net_ingredient_quantity{value,unit,derivation}, measurement_mode{individual,cumulative,unknown}, tare_observed, verification_status, review_notes}`, derivation enum `{direct_after_tare, difference_between_readings, independently_verified, unknown}`, and validation invariants (ID uniqueness, interval bounds, `confirmed` requires value+unit+derivation, never conflate `scale_value` with quantity). | No Python implementation of these records inspected; `plan.md` will define versioned models. Contract evolution from P04_104's `preliminary_schema_version 0.1.0` is required. |
| C-07 | Chronological event reconstruction with tare-aware quantity reasoning (VTR-FR-04, VTR-FR-06, VTR-AC-02, VTR-AC-05) | **Proposed** | `spec.md §5 Domain Rules` (10 rules) and Constitution III define semantics: `scale_value != net_ingredient_quantity`, `direct_after_tare`/`difference_between_readings` preconditions, zero-after-tare ≠ empty, single-frame ≠ stable, etc. Example in schema §Illustrative example correctly preserves `derivation:unknown` for an isolated `150 g` reading. | No reconstructor inspected; missing engine is the primary P0 gap. Negative reading `-750`, ambiguous `54–55`, force artifact `600` during stirring, `Lo` battery indicator are unresolved in data — system has no logic to mark these unknown yet. |
| C-08 | Evidence-linked generation & referential integrity (VTR-FR-05, VTR-FR-07) | **Proposed** | `spec.md §4 RecipeDraft, §7 Verification` requires `evidence_observation_ids`, `observation_ids`/`action_ids` links, `claim→evidence` traceability; `validation_protocol.md` targets `Traceability: 100%`. Migration guide §3 lists `Draft projection` as a distinct boundary. | No `RecipeDraft` generator, no referential-integrity validator inspected. |
| C-09 | Independent deterministic verification gates (VTR-FR-08) | **Proposed** | `spec.md §7` enumerates checks: required fields, ID uniqueness, timestamp validity, referential integrity, `approved` requires human action, quantity derivations require unit+derivation+compatible states. Constitution IV requires gates at normalization, event construction, draft submission. | Only `validate_preliminary.py` exists, which is not the P0 verifier. No blocking gate on draft generation inspected. |
| C-10 | Explicit uncertainty & abstention (VTR-FR-06) | **Proposed (contract exists)** | Schema requires `value:null, unit:null, derivation:"unknown"` and `measurement_mode:"unknown"` for unsupported cases; `spec.md §3 VTR-FR-06` and Domain Rule 10 require null/unresolved over invented values. | No code path that enforces abstention inspected. |
| C-11 | Human review & approval boundary (VTR-FR-09, VTR-AC-08, VTR-AC-09) | **Proposed** | Constitution V + `spec.md §7` + `specs/video-to-recipe/spec.md §4 ReviewDecision{draft_id, reviewer, decision, notes, time}`; proposal diagram shows `D[Draft] → H[Mandatory human review] → P[Approved version]`. | No review store, decision record, or approval CLI inspected. |
| C-12 | Reproducible runs & bounded inference costs (VTR-FR-10, VTR-FR-11) | **Proposed** | `spec.md §4 RunManifest{run_id, code_rev, input snapshot, config, outputs, budgets, attempts, timings, provider usage/errors}` + §6 External inference boundary (adapter returning typed result + metadata/error, provider/model as data, limits required). Constitution VI. | No manifest writer, budget enforcer, or provider adapter implementation inspected. Offline fixture path (`VTR-FR-12`) has no documented command. |
| C-13 | Offline deterministic execution path (VTR-FR-12, VTR-AC-10) | **Proposed** | `spec.md §10 Definition of done` and `migration-guide §M2` describe a fixture-first offline slice with no network/credentials; `spec.md VTR-AC-10` is the acceptance case. | No offline entry point inspected beyond `extract_frames.py` (which needs a real video). |
| C-14 | Controlled evaluation & holdout isolation (validation protocol) | **Proposed (protocol documented, not executed)** | `docs/evaluation/validation_protocol.md` specifies ≥10 independent holdouts, configs A/B/C evaluated on same inputs, frozen rubric before run, failure/abstention preservation, productivity evaluated separately. `docs/experiments/003_video_to_recipe.md §Evaluation measures` lists OCR exact-match, attribution accuracy, mass error, temporal alignment, cost/runtime. | Inspected as **Not evaluated** — no evaluation results, no holdout annotations, no scoring code inspected. P04_104 is explicitly development-only, not a holdout. |
| C-15 | Audio / multimodal extension (P2) | **Proposed** | `system_proposal.md §Proposed system` and `spec.md §1 Conditional-P2` describe `U[Audio transcription and timed claims] → F[Temporal evidence alignment]` with meaning-aware alignment (not timestamp overlap alone). | No audio component inspected; hypothesis H2 per proposal is unevaluated. |
| C-16 | Application package scaffold | **Implemented (minimal)** | `src/bakery_intelligence/__init__.py` per migration guide §2; `pyproject.toml` project `bakery-intelligence 0.1.0`, entry point `bakery-intelligence = "bakery_intelligence:main"` (target `main` not verified to exist); deps declared. | Scaffold only — not evidence of pipeline implementation. |

Unknown / not re-inspected in this session: `tests/test_extract_frames.py` body, `experiments/002_ocr_scales/`, `experiments/004_baseten_api/`, `scripts/muse-meta.ps1`/`muse-baseten.ps1`, `src/` depth, `data/` contents, actual Git HEAD and worktree diff, and live `P04_104.MP4` availability.

---

## 4. Architectural Gaps and Priorities

### P0 — Blocks the first executable vertical slice

| # | Gap | Evidence | Severity | Blocks |
|---|---|---|---|---|
| G-01 | **No versioned domain models in code.** `weighing_event_schema.md v1.0.0` and `spec.md §4` entities exist only as Markdown/logical records; preliminary data uses `0.1.0` with incompatible field names (`approximate_timestamp_seconds` vs `timestamp_seconds`, `reported_display_or_state` vs `scale_value/scale_unit/display_state`, `evidence.verification_status:not_frame_verified` vs `confirmed/ambiguous/unobservable`). Without typed models, every downstream boundary is uncontracted. | `docs/schemas/weighing_event_schema.md` vs `P04_104.json` top-level keys; `validate_preliminary.py` validates only 0.1.0. | High | All VTR-FR-*; M1 |
| G-02 | **No preliminary→canonical adapter.** Raw preliminary fields cannot be consumed by the v1.0.0 pipeline without lossy mutation of source data (prohibited by Constitution IX). | `P04_104.json interpretation_policy.missing_frame_evidence_remains_null`, vs missing adapter. | High | M1–M2 |
| G-03 | **No independent verification gates.** The three required gates (normalization, event assembly, draft submission) do not exist as code; `validate_preliminary.py` is a narrow structural check for one legacy file. Generators could bypass validation — violates Constitution IV. | `spec.md §7` checklist vs single `validate(data)` function in `validate_preliminary.py:7`. | High | VTR-FR-08; M4 |
| G-04 | **No tare-aware quantity engine with explicit unknown.** Schema defines `derivation` and `measurement_mode` but no logic enforces it. The repo cannot honor Domain Rules 1–5: `scale_value → net_ingredient_quantity` must abstain on isolated readings, zero-after-tare, negative values, cumulative context, single-frame stability. | `docs/schemas/weighing_event_schema.md §Weighing events, §Validation invariants`; `P04_104.json` S01–S04 contains the canonical hard cases (`-750`, `0` after tare, `29`, `54–55`, `600` force artifact, `Lo`). | High | VTR-FR-04, VTR-FR-06, VTR-AC-02, VTR-AC-05; M3 |
| G-05 | **No evidence-linked draft or approval boundary.** `RecipeDraft`/`VerificationReport`/`ReviewDecision`/`RunManifest` are unspecified in code; no mechanism enforces `pending_review` default or mandatory human approval. | `spec.md §4` table + `spec.md §7` + Constitution V vs absent code. | High | VTR-FR-07, VTR-FR-09, VTR-AC-08, VTR-AC-09; M2/M4 |

### P1 — Required for credible evidence handling and real media

| # | Gap | Evidence |
|---|---|---|
| G-06 | **Timestamp provenance is ambiguous.** 5 notions coexist: `frame_index_over_fps`, `presentation_timestamp`, `approximate_human_video_review`, and within `extract_frames.py` the triple `requested_timestamp_seconds` / `source_timestamp_seconds` (= actual + offset) / `timestamp_seconds` (= frame/FPS). No single contract carries `timestamp_basis` through the pipeline. | `docs/schemas/weighing_event_schema.md §Video metadata.timestamp_basis` + `P04_104.json timestamp_basis` + `extract_frames.py:82-84` |
| G-07 | **Frame extractor assumes constant FPS.** `frame_index = int(t*FPS+0.5)` presumes `CAP_PROP_FPS` is constant; no PTS/variable-FPS path, no post-extraction timing validation. Violates schema note that frame-index timestamps are estimates unless PTS-validated. | `extract_frames.py:28-33,62-63` docstring admits "not presentation timestamps"; schema adds disclaimer. |
| G-08 | **No append-only observation ledger.** Constitution I requires traceability from draft claim → observation → source frame/interval. No durable ledger preserves provenance, status transitions, or contradiction metadata. | `spec.md VTR-FR-05` + migration guide §3 Observation ledger vs absent ledger. |
| G-09 | **No offline synthetic fixtures or offline CLI.** `VTR-FR-12`/`VTR-AC-10` require a fixture path that runs without video or credentials; `docs/evaluation` requires reproducible fixtures. No fixture inspected. | Migration guide §M2 "offline fixture-first slice" vs no fixture file found. |
| G-10 | **Run provenance & bounded-cost enforcement absent.** No `RunManifest` writer, no frame/model budget enforcer, no provider metadata capture (Constitution VI, `spec.md §6`, `VTR-FR-10`/`VTR-FR-11`, `VTR-AC-06`/`VTR-AC-07`). | `spec.md §10 Definition of done` final bullet vs absent manifest code. |

### P2 — Required for evaluation credibility

| # | Gap | Evidence |
|---|---|---|
| G-11 | **Evaluation not executed; holdout isolation unenforced.** ≥10 holdouts, A/B/C configurations, frozen rubric, and isolation rules are documented but have no implementation artifact. P04_104 is development-only and must not be reused as a holdout (validation_protocol §Dataset and independence). | `docs/evaluation/validation_protocol.md` + `docs/experiments/003_video_to_recipe.md §Result status` ("No ... performance results are asserted") |

---

## 5. Contract and Timestamp Analysis

### Contract conflict map

| Existing (0.1.0 preliminary) | Normative (v1.0.0 schema + spec §4) | Compatibility | Required handling |
|---|---|---|---|
| `preliminary_schema_version`, `record_type: preliminary_human_review` | `schema_version: 1.0.0` + `weighing_event_schema` logical records | **Breaking** — different type system | Adapter must translate, not mutate source; version field preserved as provenance. |
| `annotation_status: preliminary_not_frame_verified` | `verification_status: confirmed/ambiguous/unobservable` per observation | **Narrowing** — status vocab mismatch | Map to `unobservable` for evidence and `ambiguous` for interpretations; never to `confirmed`. |
| `segments[].observations[].approximate_timestamp_seconds` | `FrameObservation.timestamp_seconds` | **Loosening** — approximate vs measured | Adapter records `timestamp_basis: approximate_human_video_review` and preserves raw value as `source_timestamp_hint`; rejects use as `confirmed` evidence. |
| `evidence{ frame_index:null, image_ref:null }` | `FrameObservation{ frame_index, image_ref, region }` | **Lossy** — evidence absent | Adapter emits `FrameEvidence`-absent observations; verifier must flag any downstream `confirmed` quantity that lacks resolvable `image_ref`. |
| `reported_display_or_state` (free text, e.g. "`-750`", "`54–55`") | `scale_value:number|null`, `scale_unit:string|null`, `display_state:string|null` | **Structural+semantic** | Adapter parses where unambiguous, otherwise preserves raw text in `raw_reported_value` and sets structured fields to `null` — never invents digits. |
| `scale_unit:null` + `scale_unit_hypothesis:"g"` | `scale_unit:string|null` (explicit unit) | **Policy** — hypothesis must remain hypothesis | Adapter stores hypothesis in `unit_hypothesis` annotation, not in `scale_unit`. |
| `candidate_events` (if present, 11 per migration guide) | `ActionEvent` + `WeighingEvent` with `derivation` | **Incomplete** | Candidates are not authoritative events; they must be re-derived by the reconstruction engine. |

### Timestamp provenance taxonomy (proposed for `plan.md`)

| Signal | Source | Meaning | Safe use |
|---|---|---|---|
| `requested_timestamp_seconds` | Extractor caller (`start + n/fps`) | Intended sample time | Scheduling only |
| `timestamp_seconds` (= `frame_index/FPS`) | Extractor manifest | Frame-derived estimate | Ordered indexing under **documented constant-FPS assumption**; must carry `timestamp_basis: frame_index_over_fps` |
| `source_timestamp_seconds` (= actual + offset) | Extractor manifest + user offset | Offset-adjusted view for sync | Display / alignment hint; not evidence of measurement time |
| `approximate_human_video_review` | `P04_104.json` `approximate_timestamp_seconds` | Human time label | Scheduling/segmentation hint; **never** a measurement timestamp — must be flagged `not_frame_verified` |
| `presentation_timestamp` (PTS) | Container metadata | True display time | Authoritative when available; not yet implemented — future P1 |

---

## 6. Evidence vs Claim Discipline

Per Constitution X, every reference in this assessment uses the evidence vocabulary:

- **Implemented** — `extract_frames.py` sampling logic and `validate_preliminary.py` checks (inspected source present, not execution-verified).
- **Verified** — **none claimed** in this session. Prior documents note 8 passing frame-extraction tests and 10/10 synthetic scale readings — these are **Reported observations from prior work, not re-verified here**. Assessment §4 gaps remain.
- **Proposed** — Domain contract v1.0.0, all `spec.md` functional requirements, RunManifest, verification gates, evaluation protocol, multimodal audio path, deployment shapes (`system_proposal.md` clearly labels async model vs "not current implementation claims").
- **Observed but not independently verified** — `P04_104.json` and `P04_104_manual_observations.md` contents (preliminary, not frame-verified).
- **Unknown** — Actual worktree commit, test outcomes on this machine, availability/contents of `data/raw/epic_kitchens/P04_104.MP4`, and uninspected modules listed in §1.
- **Not evaluated** — Holdout benchmark, audio contribution, productivity/economic outcomes, and reliability at scale (explicitly marked incomplete in governing docs).

---

## 7. Key Limitations and Uncertainties Carried Forward

1. **Worktree trust gap.** Because `git status/log` were blocked, the assessment cannot assert that the on-disk code matches `548230b`; downstream plans must re-verify `git rev-parse HEAD` before M1 implementation.
2. **Uninspected test and experiment surface.** Conclusions about frame-extractor quality, OCR accuracy, and Baseten/Meta provider paths rest on secondary descriptions in `system_proposal.md`/`migration-guide.md`, not on opened source.
3. **P04_104 is a single, atypical, preliminary record.** Any domain model derived solely from its four segments risks overfitting bakery-specific quirks (low-battery `Lo`, freezer transition, battery swap across 3 min gap). The v1.0.0 contract must be the normative anchor, with P04_104 as one validation case.
4. **Licensing constraint on real media.** `data/raw/epic_kitchens/P04_104.MP4` is CC-BY-NC-4.0 and git-ignored; redistribution constraints may limit CI fixtures to synthetics — plan must keep real-media paths local-only.
5. **Offline-first assumption.** The first slice (M2) will depend on synthetic fixtures whose fidelity to egocentric scale-reading challenges is limited; results cannot be claimed as real-video reconstruction until M6.

---

## 8. References

- `AGENTS.md` — Engineering Agent Instructions (specification-first, evidence-grounded)
- `specs/constitution.md` — Principles I–X and Decision policy
- `specs/video-to-recipe/spec.md` — VTR-001, VTR-FR-01–12, VTR-AC-01–10, §10 Definition of done
- `specs/video-to-recipe/migration-guide.md` — Starting inventory (§2), target components (§3), M0–M7 (§4), cross-cutting rules (§5)
- `docs/schemas/weighing_event_schema.md` — v1.0.0 logical records, derivation enum, validation invariants, synthetic example
- `docs/evaluation/validation_protocol.md` — Holdout and productivity study design
- `docs/architecture/system_proposal.md` — H1–H4 hypotheses, evidence/limitations statement, async flow with mandatory human review
- `docs/project/hackathon_mvp_scope.md` — P0/P1/P2/Conditional/Deferred priorities
- `docs/experiments/003_video_to_recipe.md` — EPIC-KITCHENS-100 P04_104 metadata (1920×1080, 50 FPS, 34 235 frames, 684.7 s, `ea8b40457a400c3fffa1c7f406ef3dc169cc2522`)
- `experiments/003_video_to_recipe/annotations/preliminary/P04_104.json` — 0.1.0 preliminary record
- `experiments/003_video_to_recipe/validate_preliminary.py` — Narrow structural validator
- `experiments/003_video_to_recipe/extract_frames.py` — Sampling + manifest writer
- `pyproject.toml` / `.python-version` — Python 3.12, `uv`/`pytest` toolchain

---

*This assessment is planning evidence only. No implementation is implied by its approval. A successful M2 offline slice must not be reported as verified real-video recipe reconstruction (per Constitution VII and `spec.md §10`).*
