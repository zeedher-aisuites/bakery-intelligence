# Bakery Intelligence
**Turning everyday kitchen work into knowledge that lasts.**

Across professional kitchens, valuable operational knowledge is
created every day. Recipes are adjusted, techniques are refined,
and production decisions are made in real time — yet much of
this knowledge is never formally documented. For small and
medium-sized food businesses, that gap makes it harder to keep
processes consistent, train new staff, and preserve expertise
as teams and practices evolve.

Bakery Intelligence explores a different approach. Instead of
requiring chefs and kitchen staff to manually document every
action, it uses multimodal AI to analyze real production videos
and reconstruct structured, evidence-backed recipe drafts. By
combining visual observations, digital scale readings, spoken
instructions, timestamps, and human review, the system helps
turn practical kitchen experience into documentation that can
be inspected, corrected, and reused.

Bakery Intelligence is an evidence-first system for capturing
culinary operational knowledge. The current MVP focuses on
bakery production workflows; broader kitchen coverage is a
future direction, not a completed capability.

Every claim in a draft carries provenance: what was seen in
video, what was mentioned in audio, what the model estimated,
and what a chef confirmed. The system never auto-approves a
recipe.

## Sample Input Video

The original bakery production video used for real-world
workflow validation is available as a GitHub Release asset:

[bakery_chef_full_production.mp4](https://github.com/zeedher-aisuites/bakery-intelligence/releases/download/v1.0.0-hackathon/bakery_chef_full_production.mp4)

Download it and upload it through the New Recipe workflow to
reproduce the end-to-end analysis. The repository
intentionally does not track the MP4, and the application
ships without precomputed results. Executing the analysis
requires supported provider credentials and may incur
inference costs.

## Problem

In restaurants, bakeries, and other small food businesses,
production knowledge lives in recordings and in the memory of
experienced staff. Recipes evolve, quantities change, and
techniques improve — often without any written record. When
that knowledge is not captured, businesses struggle with
consistency, training, standardization, and continuity.

Reconstructing a reliable recipe from footage means watching
hours of video, transcribing spoken quantities, reading scale
displays frame by frame, and reconciling contradictions.
Bakery Intelligence automates the evidence extraction and
drafts the recipe, while keeping a human in charge of every
consequential decision.

## Core capabilities

- Video upload with integrity validation (readability, duration,
  resolution, SHA-256) and content-addressed local storage.
- Four selectable FirstPass frame-analysis engines, including a
  parallel scale-reading engine (10 frames per request, 5
  concurrent requests).
- Global native-video understanding, audio transcription with
  structured speech-claim extraction, and segmented video
  analysis.
- Deterministic evidence merge, model-based consistency
  validation, and Baseten fusion into a structured recipe draft.
- Chef review workspace: finding-by-finding correction,
  confirmation, cannot-determine, undo with audit history.
- Recipe versioning with branching-aware history, per-version
  display names, and AI-assisted revision that preserves the
  parent version.
- PDF and DOCX export from the saved recipe state.
- Explicit paid-inference authorization with server-validated
  grants, per-request cost ledger, and strict spending ceilings.

## Why multimodal AI

Transcription alone cannot solve this problem: essential
information is often visible but never spoken — a scale
display reading 238, an unlabeled white powder, a tare action.
Visual analysis alone is equally insufficient: it cannot
always explain ingredient identity, intent, or preparation
decisions that the cook states out loud.

Meta Muse Spark provides the visual understanding at the core
of the MVP: per-frame and batched scale-display observations,
whole-video culinary context, structured speech claims, and
evidence consistency checks. Baseten-hosted Muse Spark Open
contributes segment-level event analysis, recipe fusion, and
bounded recipe revision; diarized transcription comes from
Meta's voice model. Deterministic local stages handle merging,
accounting, validation gates, and exports. The design goal is
reconstruction from observable evidence — never plausible
invention.

## Architecture overview

```text
Browser (React) ──► FastAPI backend ──► SQLite job registry
        │                  │                     │
        │                  ▼                     ▼
        │             runs_api/ storage    background worker
        │             (git-ignored)               │
        │                                         ▼
        │              ┌──────────────────────────────────┐
        │              │ Multimodal pipeline                │
        │              │  1. Global video (Meta Spark)      │
        │              │  2. FirstPass frames (Meta Spark)  │
        │              │  3. Audio + speech claims (Meta)   │
        │              │  4. Segments (Baseten Spark Open)  │
        │              │  5. Merge (deterministic)          │
        │              │  6. Validation (Meta Spark)        │
        │              │  7. Fusion → recipe (Baseten)      │
        │              └──────────────────────────────────┘
        │                                         │
        └──────── Chef Review ◄── recipe build ───┘
                       (edit, revise, approve, export)
```

### Model responsibilities

| Role | Provider | Model | Task |
|---|---|---|---|
| Frame analysis | Meta Model API | `muse-spark-1.3` | Scale readings, containers, actions per frame or batch |
| Global video | Meta Model API | `muse-spark-1.3` | Whole-video culinary context, operations, timestamps |
| Speech claims | Meta Model API | `muse-spark-1.3` | Structured culinary claims from transcripts |
| Transcription | Meta Model API | `muse-voice-transcribe-1.0` | Diarized transcription in ≤330 s chunks |
| Evidence validation | Meta Model API | `muse-spark-1.3` | Contradiction and consistency checks |
| Segment analysis | Baseten | `meta-models/muse-spark-open-1026` | Native-video culinary events per 30 s window |
| Recipe fusion | Baseten | `meta-models/muse-spark-open-1026` | Structured recipe draft from merged evidence |
| Recipe revision | Baseten | `meta-models/muse-spark-open-1026` | One bounded chef-requested rewrite |

Deterministic local stages (no model calls): video intake, frame
extraction, evidence merge, cost accounting, recipe assembly,
validation gates, exports.

### FirstPass engines

Selected in the upload UI and pinned in the authorization grant:

| Engine | Frames/request | Concurrency | Output ceiling |
|---|---|---|---|
| `single_frame` (default) | 1 | 1 | grant-derived |
| `contextual_batch` | 10 | 1 | 32,768 |
| `focused_batch` | 50 | 1 | 65,536 |
| `parallel_scale` | 10 | 5 | 8,192 |

`parallel_scale` extracts verbatim scale-display readings with
anchored frame IDs and no shared context between requests. The
worker may also select an engine via `FIRSTPASS_ENGINE`; a grant
value conflicting with the worker environment fails closed.

## Repository structure

```text
src/bakery_intelligence/
  api/            FastAPI app, jobs, worker, pipeline, multimodal stages
  inference/      Meta client, FirstPass engines, scale reader, eval
  video/          Intake, probing, frame extraction
  recipe_export/  Proposal assembly, labels, PDF, DOCX, revision
  phase2c/        Prompts, agents, budget ledger
  models/         Pydantic contracts + generated JSON schemas
  app/            Legacy demo navigation helpers
frontend/         React 18 + TypeScript + Vite application
tests/            Backend pytest suite (isolated tmp storage)
experiments/      Historical experiment scripts and notes
scripts/          Ops utilities (backup, launcher, benchmarks)
docs/             Architecture, schemas, protocols, session notes
specs/            Product spec and migration guide
fixtures/         Shared test data layout notes
```

## Prerequisites

- Python 3.12 with [`uv`](https://docs.astral.sh/uv/).
- Node.js 20+ with `npm`.
- MP4 codecs sufficient for OpenCV decoding.
- Meta API key for billable inference; Baseten API key for
  segments, fusion, and revision.

## Local installation

```powershell
uv sync
npm install --prefix frontend
```

This creates `.venv/` from `uv.lock` and installs frontend
dependencies. `.venv/`, `frontend/node_modules/`, and
`frontend/dist/` are git-ignored build artifacts.

## Environment configuration

Copy the template and fill in values (never commit the result):

```powershell
Copy-Item .env.example .env
```

| Variable | Purpose | Default |
|---|---|---|
| `META_API_KEY` | Meta Model API key (wins over `MODEL_API_KEY`) | — |
| `MODEL_API_KEY` | Fallback Meta key name | — |
| `BASETEN_API_KEY` | Baseten inference key | — |
| `BAKERY_BASETEN_MODEL` | Baseten deployment override | `meta-models/muse-spark-open-1026` |
| `BAKERY_MAX_VIDEO_UPLOAD_BYTES` | Upload endpoint size gate | 2 GiB |
| `BAKERY_PIPELINE_MAX_VIDEO_BYTES` | Worker-side size gate | 16 MB |
| `FIRSTPASS_ENGINE` | Worker engine fallback | `single_frame` |

For videos larger than 16 MB, raise the pipeline bound in the
worker shell before launching, e.g.
`$env:BAKERY_PIPELINE_MAX_VIDEO_BYTES = "140000000"`.

## Backend startup

From the repository root:

```powershell
.venv\Scripts\python.exe -m uvicorn bakery_intelligence.api.app:build_app --factory --port 8000
```

Health check: `http://127.0.0.1:8000/api/health`.

## Frontend startup

From the repository root:

```powershell
npm run dev --prefix frontend
```

Open the printed URL (default `http://localhost:5173`). To point
at another backend, set `VITE_API_BASE` in `frontend/.env`.

## Worker startup and execution

Free worker (offline previews and smoke workflows; never claims
paid jobs unless `--allow-billable` is passed):

```powershell
.venv\Scripts\python.exe -m bakery_intelligence.api.worker
```

Billable one-shot launcher (validates credentials with `--check`
and executes exactly one authorized pass with `--run`):

```powershell
.venv\Scripts\python.exe scripts/run_billable_worker_once.py --check
.venv\Scripts\python.exe scripts/run_billable_worker_once.py --run
```

Explicit recovery of one failed billable job (reuses persisted
stage outputs; never rebills completed stages):

```powershell
.venv\Scripts\python.exe -m bakery_intelligence.api.worker --recover <run_id> --recover-note "<why the retry is safe>"
```

## First-time user workflow

1. Open the frontend and choose **New Recipe**.
2. Enter the recipe name and language, drop an MP4, and pick the
   FirstPass engine. The panel shows authorized Meta requests and
   the worst-case frame-analysis bound.
3. Check the explicit paid-processing authorization and click
   **Generate my recipe**. This only enqueues the job with a
   server-validated grant; nothing billable runs yet.
4. Run the billable launcher (`--check`, then `--run`) to execute
   the queued job. Track progress in the queue: frame completion,
   pipeline stage, and recipe-artifact readiness are reported
   separately.
5. Open the result in **Review Findings**: resolve uncertain
   measurements one finding at a time (confirm, correct with an
   explicit value, cannot-determine, or defer).
6. Choose **Generate Recipe** to build the draft from reviewed
   evidence, then refine it in **Review Recipe**.
7. Request an AI revision with natural-language instructions when
   needed (creates a new immutable version; the parent is
   preserved), approve, and export PDF or DOCX.

## Human-in-the-loop validation

- Findings show the source frame, timestamp, AI prediction, and
  effective value side by side; corrections never overwrite the
  original model output.
- Empty numeric corrections are rejected; confirmation and
  correction are distinct persisted decisions with reviewer
  identity and timestamps.
- Undo restores the previous effective decision without deleting
  audit history.
- Generation gates block recipe building while mandatory
  findings are unresolved; the backend revalidates independently
  of the UI.
- Review states are `ai-extracted`, `chef-supplied`,
  `human-corrected`, `human-verified`, and `unknown`. A human
  correction is never relabeled as independent verification.

## Evidence provenance and uncertainty

- Scale readings keep verbatim display text, unit only when
  visible, confidence, readability, and source frame reference.
- Spoken quantities are evidence of utterance, not proof of
  measurement; audio-identified ingredients are never presented
  as visually confirmed.
- Net ingredient quantities are derived only with sufficient
  tare, container-continuity, and ingredient-association
  evidence; otherwise they stay unresolved with a recorded
  reason.
- Contradictions (audio vs visual, segment vs frame, spoken
  self-corrections) are preserved in validation records and
  recipe limitations, never silently resolved.
- The evidence bundle caps FirstPass readings at the first 40
  chronological observations; the truncation is recorded in
  `readings_usable_total` / `readings_truncated` rather than
  hidden.

## Automated testing

```powershell
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.venv\Scripts\python.exe scripts/sync_contract_schemas.py --check
npm test --prefix frontend
npm run build --prefix frontend
```

Backend tests use isolated temporary storage and fake provider
clients; they never touch `runs_api/` or billable services.
Frontend tests run under `node --test` with jsdom rendering for
interaction coverage. The schema check enforces byte-identical
Pydantic-to-JSON-Schema exports.

## Cost and authorization safeguards

- Paid inference requires an explicit user action plus a
  server-validated grant bound to workflow, model, request cap,
  token budget, frame bound, operator, timeout, and ceiling.
- The USD 20.00 `cost_ceiling_usd` with `strict_cost: true`
  bounds FirstPass Meta spend: sending stops at the cap, and
  requests with unknown cost halt the run instead of executing.
- One submit per request; no automatic billable retries and no
  automatic resume of failed runs — recovery is an explicit,
  audited operator action.
- The worst-case frame-analysis bound is shown before
  authorization; in-flight usage is estimated from
  provider-published per-token rates, never confirmed billing.
- Baseten (segments, fusion, revision) has no application-level
  spending ceiling; usage is recorded with explicit unknown-cost
  labeling. The USD 20 ceiling must not be read as a total
  multi-provider cap.

## Future direction

The MVP establishes bakery production as the first domain for
evidence-backed culinary knowledge capture. Natural next steps
include additional food-production environments, richer
ingredient identification from packaging and labels, tighter
audio-visual timestamp alignment, and longitudinal tracking of
how recipes evolve across batches. Each extension must meet
the same bar as the MVP: observable evidence, explicit
uncertainty, and human approval before anything is treated as
confirmed.

## Known limitations

- Only the first 40 chronological FirstPass readings enter the
  evidence bundle; later stable readings are preserved in
  artifacts but do not reach fusion or validation inputs.
- Readable scale displays are not net ingredient quantities:
  tare history, container changes, and ingredient association
  are frequently unverified.
- Ingredient identities are often audio-dependent; packaging
  and labels are rarely visible.
- Spoken and measured quantities conflict in practice (e.g.
  self-corrections, 1 g rounding differences); the recipe
  preserves both sides and asks the chef.
- Segment timestamps are clip-local estimates; edited footage
  can misalign absolute timelines.
- Baseten spend is uncapped; Meta spend beyond FirstPass is
  estimated, not ceiling-enforced.
- Recovery reuses completed stages only when fingerprints
  match; ambiguous prior requests fail closed rather than
  rebill.
- Passing offline tests validate contracts and fixtures, not
  live provider behavior, pricing, or rate limits.
- Single-operator local application: no authentication or
  multi-user model.

## Security and privacy considerations

- API keys live in the process environment or local `.env`
  (git-ignored); they are never printed, persisted in artifacts,
  or passed via argv by the provided tooling.
- Videos, frames, transcripts, and inference outputs stay local
  under git-ignored `runs_api/`; the repository ships no user
  media or generated results.
- The EPIC-KITCHENS sample footage referenced by early
  experiments carries noncommercial constraints; do not
  redistribute dataset media.
- Review records persist locally with reviewer identity and
  timestamps for auditability.

## Scripts

| Script | Purpose | Billable |
|---|---|---|
| `scripts/run_billable_worker_once.py` | Credential-checked one-shot billable worker launcher | Only with `--run` |
| `scripts/backup_data.py` | Content-addressed snapshots of `runs_api/` with verified restore | No |
| `scripts/parallel_scale_experiment.py` | Isolated 5×10 scale-reading decision experiment (dry-run default) | Only with `--execute` |
| `scripts/benchmark_firstpass.py` | Engine wall-clock comparison harness (dry-run default) | Only with `--execute` |
| `scripts/sync_contract_schemas.py` | Regenerate/verify Pydantic JSON schemas | No |
| `scripts/epic_dataset_discovery.py` | Reproducible EPIC-KITCHENS selection pipeline | No |
| `scripts/muse-meta.ps1`, `scripts/muse-baseten.ps1` | Provider smoke-test helpers | See script headers |

## Further reading

- `specs/video-to-recipe/spec.md` — product requirements and
  contracts.
- `docs/architecture/system_proposal.md` — original system
  proposal (historical; implementation has evolved).
- `docs/schemas/weighing_event_schema.md` — weighing contract.
- `docs/evaluation/validation_protocol.md` — validation
  methodology.
- `docs/backup-system.md` — backup and restore procedures.
- `experiments/` — per-experiment notes; each states whether
  its approach is integrated, superseded, or standalone.
