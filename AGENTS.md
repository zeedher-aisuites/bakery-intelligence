# Bakery Intelligence — Engineering Agent Instructions

**Scope:** Repository-wide instructions for AI-assisted engineering, auditing, and planning.  
**Primary workflow:** Specification-first, evidence-grounded development.  
**Current audit baseline:** `experiment/003-video-to-recipe` at `548230b` (verify the checked-out revision before using this baseline).  
**Document status:** Prescriptive operating policy; not evidence that the target architecture is implemented.

## Mission

Bakery Intelligence converts recorded bakery and pastry preparation into **evidence-linked, time-indexed recipe drafts**. It is an asynchronous system. A qualified human must explicitly review and approve a draft before operational use. The engineering priority is a reproducible vertical slice with honest uncertainty, not a polished demonstration that hides missing information.

This repository contains both experiments and proposed designs. Do not treat a proposal, a preliminary annotation, a successful API call, or a passing structural validator as proof of end-to-end recipe reconstruction.

## Required reading order

Before recommending architecture or changing implementation:

1. `docs/architecture/system_proposal.md`
2. `docs/project/hackathon_mvp_scope.md`
3. `docs/schemas/weighing_event_schema.md`
4. `docs/evaluation/validation_protocol.md`
5. `docs/experiments/003_video_to_recipe.md`
6. `experiments/003_video_to_recipe/annotations/preliminary/P04_104.json`
7. `experiments/003_video_to_recipe/validate_preliminary.py`
8. `experiments/003_video_to_recipe/extract_frames.py` and `tests/test_extract_frames.py`
9. `experiments/004_baseten_api/README.md`
10. `specs/constitution.md`, `specs/video-to-recipe/spec.md`, and `specs/video-to-recipe/migration-guide.md`

Read relevant code and tests rather than relying on document summaries. When a document and implementation disagree, describe the discrepancy; do not silently choose one.

## Evidence and reporting rules

- Label claims **Implemented**, **Verified**, **Proposed**, **Observed but not independently verified**, **Unknown**, or **Not evaluated** as appropriate.
- Cite repository paths and, when possible, functions, test names, recorded artifacts, and commit IDs for every substantive audit finding.
- Distinguish human preliminary observations, independently adjudicated references, model predictions, deterministic inferences, and reviewer corrections.
- Never promote a preliminary scale reading to ground truth, nor a visible scale display to a net ingredient mass without validated tare and measurement context.
- Missing or contradictory evidence must remain visible. Use `null`, explicit unknown status, or a request for review rather than plausible defaults.
- Do not claim tests or model calls were executed unless their actual outcome is captured. Clearly distinguish reading an earlier result file from rerunning an experiment.
- No invented benchmarks, model rankings, business savings, quality scores, or hackathon judging criteria.

## Agent operating modes

### Phase A — Repository assessment (default)

**Read-only.** Inspect project files, working-tree state, architecture contracts, tests, and experimental artifacts. You may recommend commands, but do not run commands that mutate files, create caches, install packages, fetch new dependencies, use paid APIs, or change repository state without explicit approval.

Return an assessment in the chat containing:

1. Current revision and cleanliness of the worktree, or an explicit inability to verify either.
2. Capability matrix: existing implementation, recorded evidence, gaps, and conflicting documentation.
3. Interface and dependency map for the proposed vertical slice.
4. Risk register with severity, likelihood, evidence, and proposed mitigation.
5. Ranked implementation options, with a small default and reasons for rejecting unnecessary complexity.
6. Unresolved design decisions requiring human approval.

Do **not** create `assessment.md`, `plan.md`, `tasks.md`, source code, or commits in Phase A. Ask permission to persist the approved assessment.

### Phase B — Design artifacts (approval required)

After the user approves the assessment and authorizes writing, create or update only the specifically authorized design files, normally:

- `specs/video-to-recipe/assessment.md`
- `specs/video-to-recipe/plan.md`
- `specs/video-to-recipe/tasks.md`

Each task must cite a requirement in `spec.md`, list dependency and acceptance evidence, and identify a non-destructive verification command or inspection procedure. Never present a plan as an implemented feature.

### Phase C — Implementation (separate approval required)

Implement one narrowly scoped task at a time, accompanied by tests, migration notes, and explicit verification results. Obtain confirmation before introducing new dependencies, large refactors, paid inference calls, external network operations, or changes to reference annotations.

## Git and workspace safety

- Before any file replacement, branch operation, restore, stash, merge, rebase, clean, or commit, inspect `git status --short --untracked-files=all` and current branch. Never assume work is saved or committed.
- Preserve and report untracked and modified files. Explicitly ask how to protect existing work before a destructive or potentially overwriting action.
- Do not invoke `git reset --hard`, `git clean`, `git checkout -- <path>`, force-push, or branch deletion without specific prior authorization and a verified recovery plan.
- Do not create, switch, publish, or delete branches or open/merge pull requests autonomously.
- Stage explicit paths only; avoid `git add .`.
- Before recommending a commit, show the complete English commit title and body. For GitHub PRs, show every applicable field in English, ready to copy.
- Keep `main` unchanged unless the user explicitly authorizes integration into it.

## Implementation constraints

- Prefer Python 3.12, existing `uv` and `pytest` tooling, and an inspectable modular application with one worker for the initial vertical slice.
- Reuse the existing frame extractor and weighing-event contract where appropriate. Propose compatibility changes before implementing them; do not silently rewrite experimental artifacts.
- Separate interfaces for ingestion, evidence acquisition, temporal event reconstruction, validation, draft generation, and human approval.
- Use deterministic, testable checks for schema integrity, IDs, references, units, and derivation invariants. LLM-based interpretation is a separately recorded component, not the verifier of its own claims.
- Preserve provider name, model ID, prompt or prompt hash, settings, token usage (if available), elapsed time, retries, errors, and relevant costs in run records.
- Minimize inference cost. Prefer fixtures, fakes, and offline tests while designing the backbone.
- Do not implement speculative multi-agent orchestration, distributed queues, SaaS billing, or real-time streaming to satisfy the MVP.
- Never auto-approve a recipe. Human review is a required state transition.

## Privacy, datasets, and credentials

- Never disclose, print, stage, or commit API keys, `.env` contents, local private paths with secrets, or provider configuration containing credentials.
- Videos and extracted frames from restricted or externally licensed datasets remain local unless redistribution is explicitly permitted.
- Honor the declared noncommercial constraints of the EPIC-KITCHENS-100 sample; do not assume a public repository grants rights to redistribute its media.
- Scripts that temporarily change the user's Muse Code configuration require special care: do not run concurrently with other sessions using the same configuration; forced-exit restoration is not established.
- API requests require explicit approval when they consume credits or transmit local media.

## Communication and quality bar

All repository content, source comments, test names, CLI messages, Git messages, and PR descriptions must be in English. Explanations to the user may be in Spanish.

Documentation must read as finished technical documentation, not as a development chat. Express experimental limitations scientifically, never as personal unfinished-work notes. Separate verified results from hypotheses and prospective functionality.

The default answer to an uncertain engineering claim is: **identify the missing evidence and propose a bounded way to obtain it**.
