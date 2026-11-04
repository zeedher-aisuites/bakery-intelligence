# Prompt Architecture: Current State and Gaps

Status: inventory of the prompts that produced the P04 evidence fused in
Phase 2A. No inference behavior is changed by this document; revised
prompts for future experiments are proposed separately at the end.

## 1. What exists today

| Component | Prompt location | Versioning | Role used |
|---|---|---|---|
| Frame first-pass (P35 run) | prompt text in run config; canonical text in `experiments/006_p04_multimodal_inference/prompts.py` | SHA-256 recorded as `prompt_sha256` in run records (`firstpass.py`) | user content only |
| Culinary enrichment | `CULINARY_PROMPT` in `adversarial/culinary.py` | none | user content only |
| Live proposer/challenger | inline strings in `adversarial/live.py`, persisted as `prompt_text` | none (raw text persisted) | user content only |
| Baseten challenger | JSON-via-prompt in `adversarial/baseten_challenger.py` | none | user content only |
| Native video (Spark/Wonder/Baseten) | `build_prompt()` in each `experiments/native_video_*.py` | none | user content only |

Verified facts:

- Provenance is partial: first-pass records carry `prompt_sha256`;
  live requests persist raw `prompt_text`; experiment runs persist the
  prompt text in `request.json`. None carry a human-readable version id.
- No component uses system/developer instructions. Every request sends
  task text as user-role content, including text that embeds prior
  model outputs (segmented Baseten context digests) and reviewer
  notes. Untrusted evidence and instructions share one channel.
- Epistemic vocabulary (`observed` vs `inferred` vs `unknown`) is
  enforced downstream by schema normalization, not by a shared
  instruction all models receive.

## 2. Gaps against the role model

1. No versioned prompt registry: the same analysis cannot name the
   exact instruction revision it ran.
2. No system role: nothing separates behavioral constraints (never
   invent quantities) from the task input.
3. No shared grounding contract across the three analyst roles; each
   prompt restates similar rules in different words.
4. No reconciler prompt exists at all; Phase 2A reconciliation is
   deterministic by design.

## 3. Proposed versioned specifications (future work, not implemented)

- `prompts/roles/global-culinary-analyst.v1.md` (Spark 1.3)
- `prompts/roles/local-evidence-analyst.v1.md` (Spark Open)
- `prompts/roles/measurement-analyst.v1.md` (frame pipeline)
- `prompts/roles/evidence-reconciler.v1.md` (deterministic contract;
  an LLM fallback would require the anti-fabrication clauses)

Each file must state: model id, endpoint, system instruction text,
user template, structured-output schema name and version, grounding
rules, and a changelog. New prompts are validated against the
existing P04 evidence (offline replay where possible) before any
billable call, and every future run records the prompt version id
alongside the existing SHA-256 digest.
