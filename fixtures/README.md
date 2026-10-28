# Synthetic Fixtures (`fixtures/synthetic/`)

Status labels used in this document: **IMPLEMENTED**, **VERIFIED**, **PROPOSED**,
**UNKNOWN**, **NOT EVALUATED**.

## 1. Purpose

This directory holds the M2-01 offline synthetic test bed for the video-to-recipe
slice (see `specs/video-to-recipe/tasks.md` M2-01). The fixtures exercise the hard
cases observed in preliminary annotations — missing tare history, conflicting
readings, broken evidence links, and budget-exhausted execution — without
requiring real video media, licensed datasets, credentials, or network access.

Each fixture is a self-contained `VideoSource + FrameEvidence[] + Observation[]`
bundle serialized in a test-harness envelope. The envelope (six top-level keys)
is a serialization convention only; it is **NOT** a new canonical domain model.
No fixture contains `ActionEvent` or `WeighingEvent` records: bundles carry
evidence and observations, never derived events or draft quantities.

## 2. Synthetic evidence

All evidence in these fixtures is invented for structural testing. Concretely:

- `timestamp_basis` is `synthetic_fixture` on every `VideoSource` and every
  `Observation` (**IMPLEMENTED** in all five fixtures).
- `image_ref` is `null` on every `FrameEvidence`: no images are referenced or
  implied (**IMPLEMENTED**).
- `source_ref` is a relative synthetic path of the form
  `synthetic/<fixture_stem>.mp4`. No absolute path, no `data/raw/` path, and no
  EPIC-KITCHENS or other licensed media is referenced (**IMPLEMENTED**).
- All `VideoSource` records share one interval: `start_seconds 0.0`,
  `end_seconds 10.0`, `duration_seconds 10.0`, `fps 25.0`, `frame_count 250`,
  `source_dataset "synthetic-fixture"`, and a `SYNTH.<FIX>.001` video id
  (**IMPLEMENTED**).
- Every `Observation` carries provenance `source_type "human"` with `source_id`
  equal to the fixture filename, and `verification_status "ambiguous"`.
  Promotion to `confirmed` belongs to the M2-03 verifier, never to a fixture
  (**IMPLEMENTED**).
- Frame indexes are consistent with `fps 25.0` (index = timestamp x 25), and all
  timestamps are closed-bound-safe interior points of the 0-10 s interval
  (**IMPLEMENTED**).
- Scale readings (`scale_value`) are displayed values only. No fixture conflates
  a display reading with a net ingredient mass (**IMPLEMENTED**).

## 3. Envelope contract

Each fixture file carries exactly these six top-level keys, in stable order:

1. `schema_version` — the string `"1.0.0"` (pinned exactly; the loader rejects
   anything else).
2. `fixture_id` — deterministic, equal to the filename stem.
3. `description` — non-empty prose carrying machine-readable bracket tags:
   `[class: ...]`, `[expected-m1-03: ...]`, `[expected-m2: ...]`.
4. `video_source` — one canonical `VideoSource`.
5. `frame_evidence` — array of canonical `FrameEvidence`.
6. `observations` — array of canonical `Observation`.

Every nested canonical record carries its own `schema_version "1.0.0"` and
conforms to its M1-01 Pydantic model with unknown fields rejected. The fixture
loader (a test helper in `tests/test_fixtures.py`; no new `src` module)
validates this envelope and raises `FixtureError(code="FIXTURE_INVALID")` on
any load-time failure: malformed JSON, unknown envelope key, non-`"1.0.0"`
envelope `schema_version`, missing required keys, or a wrapped nested
`ValidationError`. The structurally invalid fixture parses successfully and
fails only at check time, never at load time.

## 4. Fixture inventory and structural expectations

### 4.1 `synthetic_interval_nominal.json` — nominal happy path

- Class tag: `[class: nominal]`.
- Records: `SYNTH.NOM.001`; 3 frames (`frm_nom_001` t=2.0 idx 50,
  `frm_nom_002` t=4.8 idx 120, `frm_nom_003` t=5.0 idx 125); 2 observations
  (`obs_nom_tare`: 0.0 g, `zero`, evidence `[frm_nom_001]`; `obs_nom_mass`:
  120.0 g, `stable`, evidence `[frm_nom_002, frm_nom_003]`).
- Structural expectation: passes M1-03 structural checks (unique ids,
  timestamps within bounds, referential integrity). Status: **NOT EVALUATED**
  by this fixture commit (asserted by `tests/test_fixtures.py`, owned outside
  this directory).
- Intended downstream M2 behavior: full run completes; verifier consistency
  `pass`; draft carries evidence-linked steps. Status: **NOT EVALUATED**.

### 4.2 `synthetic_interval_unknown_tare.json` — class (b), uncertain semantics

- Class tag: `[class: b]`.
- Records: `SYNTH.UNK.001`; 2 frames (`frm_unk_001` t=4.8 idx 120,
  `frm_unk_002` t=5.0 idx 125); 1 observation (`obs_unk_001`: 150.0 g,
  `stable`, evidence `[frm_unk_001, frm_unk_002]`).
- Honesty property: the bundle contains no tare history — no observation with
  `display_state "zero"` and no `0.0` scale value — and the description states
  this absence explicitly ("No tare observation; tare history unknown").
  Status: **IMPLEMENTED**.
- Structural expectation: passes M1-03 structural checks; uncertainty is
  semantic, not structural. Status: **NOT EVALUATED** by this fixture commit.
- Intended downstream M2 behavior: verifier `pass` or `review_needed` (never a
  structural `fail`); draft quantity null with an `UNKNOWN_TARE_HISTORY`
  unresolved question. Status: **NOT EVALUATED**.

### 4.3 `synthetic_interval_conflicting.json` — class (b), uncertain semantics

- Class tag: `[class: b]`.
- Records: `SYNTH.CON.001`; 2 frames at the same moment (`frm_con_001` and
  `frm_con_002`, both t=4.8 idx 120, distinct ids); 2 observations
  (`obs_con_001`: 54.0 g, `stable`, evidence `[frm_con_001]`; `obs_con_002`:
  58.0 g, `stable`, evidence `[frm_con_002]`).
- Structural expectation: passes M1-03 structural checks; the conflict is
  semantic and both readings must remain visible. Status: **NOT EVALUATED** by
  this fixture commit.
- Intended downstream M2 behavior: verifier `review_needed` with the
  contradiction preserved (no silent resolution); draft projects only under
  explicit `review_needed` tolerance. Status: **NOT EVALUATED**.

### 4.4 `synthetic_interval_dangling_ref.json` — class (a), structurally invalid

- Class tag: `[class: a]`.
- Records: `SYNTH.DAN.001`; 1 frame (`frm_dan_001` t=4.8 idx 120);
  1 observation (`obs_dan_001`: 120.0 g, `stable`, evidence
  `["frm_missing_001"]`).
- Precision property: exactly one holder (`obs_dan_001`) cites exactly one
  missing target (`frm_missing_001`), and both ids appear in the description.
  This dangling edge is the only violation in the file. Status:
  **IMPLEMENTED**.
- Structural expectation: parses under the loader, then fails M1-03 structural
  checks with `DANGLING_REFERENCE` against `obs_dan_001`; no draft is
  produced. Status: **NOT EVALUATED** by this fixture commit.
- Intended downstream M2 behavior: verifier `fail`; CLI reports a failed run
  with the verification report preserved. Status: **NOT EVALUATED**.

### 4.5 `synthetic_interval_partial_budget.json` — class (c), budget-exhausted execution

- Class tag: `[class: c]`.
- Records: `SYNTH.BUD.001`; 3 frames (`frm_bud_001` t=1.0 idx 25,
  `frm_bud_002` t=4.8 idx 120, `frm_bud_003` t=7.0 idx 175); 3 observations
  (`obs_bud_001`: 0.0 g, `zero`, `[frm_bud_001]`; `obs_bud_002`: 120.0 g,
  `stable`, `[frm_bud_002]`; `obs_bud_003`: 120.0 g, `stable`,
  `[frm_bud_003]`).
- Structural expectation: passes M1-03 structural checks unconditionally;
  partiality arises only from execution budgets, never from structure.
  Status: **NOT EVALUATED** by this fixture commit.
- Intended downstream M2 behavior: under a frame budget the run yields
  `partial` with `BUDGET_EXHAUSTED` and best-effort artifacts preserved;
  without a budget it completes. Status: **NOT EVALUATED**.

## 5. Expected M1-03 behavior (structural; assertable by tests)

| Fixture | M1-03 structural outcome |
|---|---|
| nominal | `pass` (empty issue list) |
| unknown_tare | `pass` — valid structure, uncertain semantics |
| conflicting | `pass` — valid structure, uncertain semantics |
| dangling_ref | `fail` with `DANGLING_REFERENCE` on `obs_dan_001` |
| partial_budget | `pass` — valid structure regardless of budget |

These outcomes are asserted by `tests/test_fixtures.py`, which exercises the
M1-03 checks `check_unique_ids`, `check_timestamps`, and
`check_referential_integrity` over the bundles. The event-scoped checks
(`check_interval_bounds`, `check_measurement_derivation`) receive empty event
lists and return no issues, since M2-01 bundles contain no events. From the
perspective of this fixture commit, all M1-03 outcomes are **NOT EVALUATED**
(they are predictions the test suite checks, not results observed here).

## 6. Intended downstream M2 behavior (semantic; not asserted here)

| Fixture | Intended M2-03 / M2-04 / M2-06 outcome |
|---|---|
| nominal | completed run; consistency `pass`; evidence-linked draft |
| unknown_tare | `pass` or `review_needed`; null quantity + `UNKNOWN_TARE_HISTORY` |
| conflicting | `review_needed`; contradiction preserved, both readings kept |
| dangling_ref | `fail`; no draft |
| partial_budget | `partial` + `BUDGET_EXHAUSTED` only under budget exhaustion |

All downstream semantic outcomes are **NOT EVALUATED**: M2-03, M2-04, and M2-06
are unimplemented at the time of writing, and a structural fixture cannot
evidence the behavior of components that do not exist yet. The `[expected-m2:
...]` description tags record intent for those future stages.

## 7. Verification notes for this commit

- **IMPLEMENTED**: five fixture files plus this README; envelope shape; shared
  `VideoSource` parameters; deterministic ids; `synthetic_fixture` timestamp
  basis; `ambiguous` verification status; null `image_ref`; relative synthetic
  `source_ref`; bracket tags in every description; absence-of-tare statement in
  the unknown-tare description; single dangling edge in the dangling fixture.
- **VERIFIED**: static review only — field names and enum literals were read
  from the committed M1-01 models (`src/bakery_intelligence/models/`) before
  writing; fixture text was scanned for forbidden runtime-result tokens
  (`net_ingredient_quantity`, `derivation`, `measurement_mode`, `quantity`,
  `action_ids`, `consensus`), none present; no absolute paths, secrets, or
  real-media references present.
- **NOT EVALUATED**: loader acceptance, M1-03 check outcomes, byte-stability
  across test runs, and every M2-03/M2-04/M2-06 semantic prediction. No test or
  model call was executed as part of this fixture commit.
