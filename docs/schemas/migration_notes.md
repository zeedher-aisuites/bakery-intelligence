# Domain Contract Migration Notes

**Scope:** Compatibility log for the canonical `schema_version 1.0.0` domain
models (`src/bakery_intelligence/models/`), starting with M1-01.
**Normative sources:** `specs/video-to-recipe/plan.md` §4,
`specs/video-to-recipe/tasks.md` M1-01,
`docs/schemas/weighing_event_schema.md`.

## M1-01 baseline (2026-10-28)

- Branch: `experiment/005-muse-code-audit`
- HEAD at implementation time: `8200e2e7248c12028adff645179a5e12f1930ac2`
  (`docs: finalize Phase B.1 architecture and implementation tasks`)
- Working tree at start: clean, no uncommitted changes.
- Toolchain declared: Python 3.12, `pydantic>=2.14.1` (installed: 2.14.1),
  `pytest>=9.1.1` (installed: 9.1.1). The initial model implementation
  needed no dependency change; schema finalization added
  `jsonschema>=4.26.0` to the development dependency group (with `uv.lock`
  updated) to execute behavioral schema tests.
- Test suite: `tests/test_contracts.py` (contract-level, offline).

## Verification status

| Check | Result | Evidence |
|---|---|---|
| `pytest tests/test_contracts.py -v` | 159 passed in 1.44 s | Executed by the user on Python 3.12.10; includes model, extension-presence, drift, and behavioral schema tests. |
| `pytest -v` (complete project suite) | 167 passed in 3.23 s | Same user session; includes the pre-existing frame-extraction tests. |
| `python scripts/sync_contract_schemas.py --check` | All 9 schemas match (exit 0) | Executed by the user after regenerating with the extension-aware script. |
| Behavioral schema accept/reject tests | Executed with `jsonschema` installed | 9 positive fixture cases and 6 negative cases, same user session. |
| `git diff --check` | No whitespace errors | Reported from the user session. |
| Static implementation review | Completed | Muse reviewed models, extensions, and tests by inspection; the sandbox interpreter remained unavailable, so no Python was executed by Muse. |

Passing tests establish that the implemented contract cases and the
representative JSON Schema acceptance/rejection cases behave as specified.
They do not establish universal Pydantic/JSON-Schema equivalence and do
not validate every possible payload.

## Versioning policy (plan.md §4.1)

- `schema_version` is a required field on every persisted top-level record;
  nested records (`TimeInterval`, `Provenance`, quantities, steps, issues,
  config snapshots) carry no version of their own.
- The model layer accepts any `1.x.y` value so additive minor revisions stay
  readable, and rejects malformed values and unknown majors (`0.1.0`,
  `2.0.0`, …) with a field-specific `ValidationError`.
- Models use `extra="forbid"`: M1 is the exact `1.0.0` contract. Tolerating
  future additive fields is a deliberate later migration decision, not a
  silent default.
- Breaking changes bump major; additive optional fields bump minor;
  documentation-only changes bump patch. The preliminary
  `preliminary_schema_version 0.1.0` contract stays a read-only source; its
  translation to canonical records is M1-02 adapter work, out of scope here.

## Interpretation decisions

Where the specification fixes a closed vocabulary, the models use typed
enums; where it leaves a set open, the models use plain strings:

| Decision | Rationale |
|---|---|
| `Modality` holds `visual` only | Plan §4.2: `visual` for M2, audio deferred to P2. Adding `audio` later is an additive minor change. |
| `scale_unit` and quantity `unit` are free strings | The weighing schema allows `g, kg, oz, lb, or another explicitly observed unit`: an open set. |
| Issue `severity` is a required free string | No severity vocabulary is specified; M1-03/M2-03 define conventions. |
| Issue `gate` is the `G1/G2/G3` enum | Plan §7.1 names exactly three gates; M1-03 uses the G1/G2 subset. |
| `EvidenceRefs` is `{observation_ids[], frame_evidence_ids[]}` | Typed form of plan §4.2 `evidence_refs[]` (claim-to-evidence indices). |
| `BudgetState` is `{frames_used, model_calls_used, cost_usd_spent, exhausted, exhausted_at}` | Typed form of plan §9.1 `budget_state{consumed, remaining, exhausted_at}`. |
| `RunOutputs` holds optional paths plus optional SHA-256 hashes | Covers plan §4.2 paths and plan §9.1 path+hash shape; M2 seeds paths, M5 adds hashes. |
| `CostSummary` defaults to `{total_cost_usd: 0.0, currency: "USD"}` | Matches the offline zero-cost expectation. |
| `Observation` requires exactly one of `timestamp_seconds` / `interval` | Plan §4.2: single timestamp versus interval is explicit. |
| `RecipeDraft.status` defaults to `pending_review` | Only initial status (Constitution V). `approved` is representable for history but creatable only via the review boundary and gate G3. |
| Models never change `verification_status` | Status ceilings live in the M1-02 adapter; promotion to `confirmed` belongs to the M2-03 verifier or isolated human adjudication (plan.md §5.1). |
| `VideoSource.source_ref` rejects absolute paths | Plan §4.2 constraint; checked for POSIX and Windows forms on any platform. |
| `manifest_sha256` / output hashes require 64 hex characters | Field names promise SHA-256 digests. |
| `derivation == "unknown"` requires null value and unit | Plan §6.1, on both `NetIngredientQuantity` and `DraftQuantity`. Presence of value/unit for non-unknown derivations is additionally checked at bundle level (M1-03 `DERIVATION_WITHOUT_UNIT`). |

## Bounds enforced above the model layer

Single-record models cannot see bundle context, so these specified bounds
are M1-03 reference-check responsibilities, not model validators:

- `end <= duration` on `ActionEvent` / `WeighingEvent` intervals
  (`VideoSource` enforces it because it carries its own duration).
- Observation/event timestamps contained within the source interval.
- Identifier uniqueness and `*_ids[]` referential integrity.
- `stable` display support across multiple evidences (schema rule).

## JSON Schema export

`src/bakery_intelligence/models/schemas/*.json` (JSON Schema draft 2020-12,
one self-contained file per record).

Revision history: the first committed revision was written by hand
against the models because no runnable interpreter was available in that
session. A user `--check` run then confirmed substantial drift versus the
Pydantic-generated base. The current artifacts are produced by the
generator plus the explicit extensions below; the hand revision was
superseded, not patched. The synchronized artifacts were generated and
verified in the user session recorded above.

### Authoritative sources

- The Pydantic models in `src/bakery_intelligence/models/` are the
  implementation source of truth for shapes, required fields, enums,
  defaults, and numeric bounds.
- The approved domain rules come from `plan.md` §4–§7, `tasks.md` M1-01,
  `spec.md`, and `docs/schemas/weighing_event_schema.md`.
- The committed `.json` artifacts are derived: `model_json_schema()`
  output plus the explicit extensions listed here. They must byte-match
  the generator; `test_json_schemas_match_model_exports` enforces this.

### Generation procedure

`scripts/sync_contract_schemas.py` renders each record with
`Model.model_json_schema()`, applies the extensions, injects `$schema`
(draft 2020-12) and `$id`
(`https://bakery-intelligence.local/schemas/1.0.0/<file>`), and serializes
canonically (indent 2, sorted keys, LF). Generation is idempotent for a
fixed pydantic version and has no network or optional dependencies.
`--check` compares without writing and exits nonzero on drift. Nested
models (including the `VideoSource` snapshot inside `RunManifest`) are
inlined into per-file `$defs`; no cross-file references are emitted.

### Explicit extensions and justification

| # | Target | Rule | Justification |
|---|---|---|---|
| E1 | All files: `properties.schema_version.pattern = ^1\.\d+\.\d+$` | Accept canonical major 1 only. | tasks.md M1-01 acceptance criterion 2; plan.md §4.1. Pydantic exports a plain string; the version gate is runtime-only. |
| E2 | `observation.json`: top-level `oneOf` on time forms | Exactly one of `timestamp_seconds` / `interval` populated; omitted and null both count as unset. | plan.md §4.2 explicit-representation rule. Branch logic mirrors the model validator's None-as-unset semantics. |
| E3 | `NetIngredientQuantity` / `DraftQuantity` `$defs`: `if`/`then` | Value/unit must be null unless a supported derivation is explicitly present; an omitted derivation behaves as `unknown`, mirroring the model default. | plan.md §6.1; tasks.md required behavior 4. |
| E4 | `video_source.json`: `source_ref` `not` pattern `^([/\\]|[A-Za-z]:[/\\])` | Reject absolute paths. | plan.md §4.2 constraint. The pattern exactly mirrors the runtime check (`posixpath.isabs` or `ntpath.isabs`); drive-relative `C:foo` stays allowed, as at runtime. |

Each extension anchors on a node the base generator must produce and
raises an error when the anchor is missing, so model refactors cannot
silently drop a constraint.

### Difference classification (superseded hand revision vs generated base)

- Cosmetic: key ordering, per-property titles, docstring descriptions,
  `default` annotations (inert in Draft 2020-12), formatting.
- Equivalent structural: shared `$defs` with local `$ref` (including enum
  definitions) versus hand-inlined enums; `anyOf` nullable unions versus
  hand-written type arrays; inlined `VideoSource` snapshot versus the hand
  revision's cross-file `$ref` (the generated form also satisfies the
  self-containment requirement).
- Material, preserved via E1–E4: schema_version major gate, timestamp
  exclusivity, unknown-derivation nulls, relative-path restriction.
- Material, intentionally Pydantic-only (see below): interval ordering,
  duration containment, whitespace stripping, strict datetime parsing.

### Rules enforced by Pydantic only

- `start < end` on every interval and `end <= duration` on `VideoSource`:
  cross-property numeric comparison is not expressible in Draft 2020-12.
- Identifier whitespace stripping (`NonBlankStr`): schemas enforce
  `minLength: 1`; a whitespace-only string passes standalone schema
  validation but fails model parsing.
- Strict datetime parsing: schemas carry `format: date-time`, which most
  validators treat as an annotation. Malformed datetime strings may pass
  standalone validation but fail model parsing.
- Lax scalar coercion works in the opposite direction: canonical JSON
  numbers validate identically in both representations, but a numeric
  string such as `"120"` for a float field is accepted by Pydantic lax
  parsing and rejected by schema `type: number`. Canonical payloads use
  JSON numbers.

### Rules requiring cross-record or execution-context validation

Same set as "Bounds enforced above the model layer" (identifier
uniqueness, referential integrity, timestamp containment, `stable`
display support), plus verification-status promotion policy and approval
gating (M1-02/M2-03/runtime concerns). None of these are single-record
schema constraints.

### Artifact hashes

SHA-256 of the validated local schema artifacts (paths relative to the
repository root). These hashes identify the validated files, not an
immutable release; any contract change regenerates them via the sync
script.

| File | SHA-256 |
|---|---|
| `src/bakery_intelligence/models/schemas/action_event.json` | `AC0901846D2BD3CFFE7396B64ECF97CBE9BED88CB29F52BBA4AFC9D834F8D1ED` |
| `src/bakery_intelligence/models/schemas/frame_evidence.json` | `7DCF599D7F89FB922D6462EC4C2EA83E7199368C0F18533AAD754332904B16E5` |
| `src/bakery_intelligence/models/schemas/observation.json` | `C8EA6D174CB64273E8D69421C8676E08FB5965780B79FE13CB5F49A9CD95CE61` |
| `src/bakery_intelligence/models/schemas/recipe_draft.json` | `99ACFECCB632C7891ECEE73A31D45F5A14654CF2CDA069F5D62007EDD3978FBD` |
| `src/bakery_intelligence/models/schemas/review_decision.json` | `56D9FF0DADD64328F31FAD41939874BE1DCD2CF589917BC75A495CDFC512A2FF` |
| `src/bakery_intelligence/models/schemas/run_manifest.json` | `22D2B28F75B9E7830BF38B7F284A51C03989F9CE25BF329518EBE0FC103E9385` |
| `src/bakery_intelligence/models/schemas/verification.json` | `E2B844F7709AFCC338DB8C300C72A3C6FFB2F23D232C4A7BE2CD8CD244F1D6C1` |
| `src/bakery_intelligence/models/schemas/video_source.json` | `975189F8DAE0F622C552D122DAF821BB5DD63B3D923B8EBF72A255E302F7F621` |
| `src/bakery_intelligence/models/schemas/weighing_event.json` | `A9F71B4A6237A4EA376A406A7906B538348804D929AC36B2F85BE8FC23BCB458` |

## Execution environment note

Muse implementation work in this repository was done by static inspection:
the project interpreter (CPython 3.12.10 per `.venv/pyvenv.cfg`) is not
accessible from the Muse sandbox, no other local interpreter exists there,
and network access for toolchain recovery was denied. All Python execution
reported in the verification table above was performed by the user on
Windows with Python 3.12.10, using `jsonschema>=4.26.0` from the
development dependency group for the behavioral schema tests. The
canonical re-verification procedure is:

```bash
python scripts/sync_contract_schemas.py --check
pytest tests/test_contracts.py -v
pytest -v
```

A clean `--check` (exit 0) confirms the committed artifacts still match
deterministic generation. The behavioral schema tests skip when no JSON
Schema validator library is installed; absence is reported as skipped,
never as passed.
