# Evidence-Grounded Adversarial Recipe Reconstruction

## Implemented (non-billable)

- `adversarial/evidence.py` builds a normalized package from a
  completed run's artifacts: timestamped frames, scale readings,
  visible objects, action evidence, human reviews, unknowns, and
  artifact provenance. Ingredient, unit, and net quantity stay
  explicit unknowns.
- `adversarial/claims.py` defines the claim unit with statuses
  `supported`, `uncertain`, `contradicted`, and
  `requires_human_review`, and provenance `ai-extracted`,
  `human-corrected`, `chef-supplied`.
- `adversarial/providers.py` defines proposer/challenger
  interfaces. The deterministic evidence proposer and fake
  challenger are the only providers exercised locally and are
  labeled simulated.
- `adversarial/resolution.py` debates claims individually.
  One round is one challenger critique over still-disputed
  claims plus deterministic verification and a status decision.
  Maximum three rounds; no video or frame reprocessing occurs.
- `adversarial/verification.py` emits a canonical
  `VerificationReport`: dangling frame references, timestamp
  bounds, unknown derivations with values, quantities without
  units, ambiguous tare, and conflicting quantities.
  Agreement without evidence never produces `supported`.
- `adversarial/store.py` persists chef review at
  `recipe_reviews/<run_id>.json`. Corrections create a new
  draft version. Approval validates a canonical
  `ReviewDecision`, requires a reviewer identity and a
  non-failing verification verdict, and appends an immutable
  snapshot with SHA-256. Original run artifacts are untouched.
- API: `GET /recipe`, `GET /recipe/evidence`,
  `POST /recipe/corrections`, `POST /recipe/decisions` under
  `/api/analyses/{run_id}`. The frontend Recipe review tab
  exposes editable ingredients and steps, claim evidence links,
  disagreements, decisions, and version history.

## Provider findings (verified locally, no live call)

| Role | Provider | Model | Interface | Status |
|---|---|---|---|---|
| Proposer | Meta | `muse-spark-1.3` | Responses API via existing client | Implemented for frame observation; recipe-claim prompting not live-validated |
| Challenger | Baseten | `meta-models/muse-spark-open-1026` | Chat Completions, `https://inference.baseten.co/v1` | Connectivity and vision/text verified in Experiment 004; recipe critique not live-validated |

Experiment 004 recorded Chat Completions usage as prompt,
completion, and total tokens. It did not measure USD pricing.
The hackathon grant terms (US$100 credits, up to 2M tokens per
minute) are account terms, not a per-token price.

The Baseten adapter (`adversarial/baseten_challenger.py`)
sends no reasoning parameter: `medium` reasoning is a
requested configuration recorded for provenance, not a
verified capability. Spark's existing client likewise sends
no reasoning parameter. Structured JSON output from the
Baseten challenger is prompted and parsed, not schema-enforced.

## Remaining gaps requiring live validation

- Whether the Baseten challenger reliably returns parseable
  JSON objections for recipe-scale prompts.
- Whether either provider accepts a reasoning-effort parameter,
  and under what name and values.
- Actual challenger latency, usage, and failure modes under
  recipe prompts.
- Quality of Spark recipe claims against a chef-validated
  cinnamon-roll reference.

Simulated adversarial tests are not live model validation.
No billable inference is authorized by this implementation.
