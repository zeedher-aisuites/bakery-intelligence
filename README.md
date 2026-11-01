# Bakery Intelligence

Bakery Intelligence converts recorded bakery and pastry preparation
into **evidence-linked, time-indexed recipe drafts**. A real video
analysis produces retained frames, timestamped model observations
with abstentions, provenance records, and an explicit human review
step. A qualified human must review a draft before any operational
use; the system never auto-approves a recipe.

Current MVP scope: upload a real preparation video (up to about
12 minutes), validate and stage it, run the existing Meta
Muse Spark 1.3 inference pipeline only after explicit paid
authorization, track background execution, and inspect, correct,
and export the evidence in a review workspace. What the pipeline
produces is an **evidence-backed draft**, not a fully validated
recipe: ingredient identities, units, tare history, and
preparation steps remain unknown unless a human verifies them.

## Architecture

- **FastAPI backend** (`src/bakery_intelligence/api/`): upload
  staging, durable SQLite job registry, background workers,
  review persistence, and artifact serving. Storage root is
  `runs_api/` (git-ignored by design).
- **Video intake and extraction** (`src/bakery_intelligence/video/`):
  OpenCV probing, SHA-256 integrity, content-addressed storage,
  and 1 FPS sampled frame extraction.
- **Inference pipeline** (`src/bakery_intelligence/api/pipeline.py`,
  `src/bakery_intelligence/inference/`): offline preview path
  plus the live Meta Muse Spark path with schema-constrained
  JSON output, attempt/timeout/byte bounds, a per-request cost
  ledger, and strict ceiling enforcement.
- **React + TypeScript frontend** (`frontend/`): landing page,
  product upload workflow, evidence review workspace,
  preparation record, and CSV/JSON export.
- **Local backup utility** (`scripts/backup_data.py`):
  content-addressed snapshots with manifests and verified
  restore. See `docs/backup-system.md`.

## Prerequisites

- Python 3.12 (verified with 3.12.10) and
  [`uv`](https://docs.astral.sh/uv/) for environment management.
- Node.js with `npm` for the frontend.
- OpenCV-system video codecs sufficient to decode MP4 (used by
  upload validation and frame extraction).

## Installation

```powershell
uv sync
npm install --prefix frontend
```

This creates `.venv/` (Python dependencies from `uv.lock`) and
installs frontend dependencies. `.venv/`, `frontend/node_modules/`,
and `frontend/dist/` are git-ignored build artifacts.

## Starting the application

Backend API (from the repository root):

```powershell
.venv\Scripts\python.exe -m uvicorn bakery_intelligence.api.app:build_app --factory --port 8000
```

Health check: `http://127.0.0.1:8000/api/health`.

Frontend (from the repository root):

```powershell
npm run dev --prefix frontend
```

Open [http://localhost:5173/](http://localhost:5173/) in a browser.

Free background worker (offline previews and smoke workflows;
**never claims paid jobs** unless `--allow-billable` is passed):

```powershell
.venv\Scripts\python.exe -m bakery_intelligence.api.worker
```

## Offline-preview workflow (free)

1. Open **+ New analysis**, drop an MP4 (validated locally:
   container readability, duration, resolution, SHA-256).
2. Review projected frames at 1 FPS and the nonbinding cost
   estimate (or the explicit "cost uncertain" notice).
3. Click **Run free offline preview**. Observations are
   simulated, clearly labeled, and never sent to Meta.
4. When the job completes, open it in the Evidence Review
   Workspace to inspect frames, save human reviews, and export.

## Paid Meta analysis workflow (billable)

1. Upload and configure as above. The panel shows the model
   (`muse-spark-1.3`), the projected frame/request count, the
   nonbinding estimate, and the **authorized maximum of
   USD 20.00**.
2. Check the authorization box and click **Authorize and start
   analysis**. This only enqueues the job with a server-validated
   grant; uploading or previewing never triggers paid inference.
3. Execute exactly one authorized pass with the billable launcher:

```powershell
.venv\Scripts\python.exe scripts/run_billable_worker_once.py --check
.venv\Scripts\python.exe scripts/run_billable_worker_once.py --run
```

`--check` validates credentials and exits without launching. For
videos larger than 16 MB, set the pipeline size bound in the
same shell first (see [Large-video resource behavior](#large-video-resource-behavior)).

## Authorization and cost enforcement

- Paid inference starts only from an explicit user action and a
  server-validated grant bound to the exact workflow, model,
  frame/request limits, operator, and spending ceiling.
- Maximum authorized cost: **USD 20.00** (`cost_ceiling_usd`),
  enforced by a per-request ledger with `strict_cost: true`:
  sending stops when known spend reaches the cap, and any
  grant with an unknown cost is rejected rather than executed.
- `max_attempts=1`, no automatic billable retries, no resume of
  failed billable runs, and an `uncertain → refuse` policy: a
  replacement run refactures from scratch and is never started
  automatically.
- The generic worker excludes billable workflows by default; a
  generic pass leaves a queued paid job untouched (`attempts`
  stays 0). Only the one-shot launcher passes `--allow-billable`.

Credential variables, by name only (never commit values):
`META_API_KEY` (wins) or `MODEL_API_KEY`, read from the process
environment or the local `.env` file.

## Evidence review, correction, and export

Completed analyses open in the Evidence Review Workspace: a
filmstrip timeline, the retained frame viewer with hash-integrity
notes, the original AI prediction or abstention, and review
controls (confirm, correct, reject, uncertain) with optional
notes. Original predictions are immutable; annotations persist
separately and survive restarts. The preparation record links
each reading to its evidence frame and marks AI-extracted,
human-verified, human-corrected, rejected, uncertain, and
unknown states honestly. CSV/JSON exports preserve model values
alongside human decisions and never modify source evidence.

## Local data persistence

All application state lives under `runs_api/`: staged uploads
and metadata, per-run work and artifact trees, review records,
and `jobs.sqlite3`. This directory is git-ignored and is the
local backup source. The sibling directory
`bakery-intelligence-backups` holds verified snapshots; see
`docs/backup-system.md` for backup, trigger, retention, and
restore procedures.

## Safe demonstration procedure

Without spending anything: start the API and frontend, open the
completed analysis `run_387662c3eaab` (30 frames, 6 numeric
predictions, 24 abstentions) from the Analyses library, inspect
an AI prediction against its frame, and save a review decision
to see it reflected in the preparation record.

## Tests and checks

```powershell
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.venv\Scripts\python.exe scripts/sync_contract_schemas.py --check
npm test --prefix frontend
npm run build --prefix frontend
```

Python tests use isolated temporary storage; they never touch
`runs_api/` or billable services.

## Known limitations and future work

- Pipeline executors load the whole video into memory; see
  [Large-video resource behavior](#large-video-resource-behavior).
  Memory optimization is future work.
- The cost estimate is calibrated on a 640×360 reference run
  and is nonbinding; 1080p frames may cost more input tokens.
  The USD 20 cap bounds exposure regardless.
- The pipeline emits an evidence-backed draft (observations,
  deltas, honest unknowns), not a validated recipe: no
  ingredient identities, units, net weights, tare events, or
  preparation steps are invented.
- No authentication or multi-user model; the local API is a
  single-operator demo backend.
- Landing statistics fetch one bundle per listed run, which is
  slow with many runs.

## Large-video resource behavior

- **In-memory loading (current behavior):** pipeline executors
  read the entire staged video into RAM (`handle.read()` into
  `video_bytes`) before writing the run input copy. For the
  2,074,288,282-byte (≈1.93 GiB) P04_104 analysis, worker
  process memory peaks around 2.5–3.5 GB on top of the normal
  interpreter footprint. This behavior is unchanged by this
  task; streaming execution is future work.
- **`BAKERY_PIPELINE_MAX_VIDEO_BYTES`:** the pipeline-side byte
  gate, defaulting to 16 MB. It must be raised in the worker's
  environment for larger videos, e.g.
  `$env:BAKERY_PIPELINE_MAX_VIDEO_BYTES = "2074288282"`. Below
  the gate the job fails explicitly *before* any provider call.
- **Upload limit (`BAKERY_MAX_VIDEO_UPLOAD_BYTES`):** a separate
  gate on the upload endpoint (default 2 GiB). Raising the
  pipeline bound does not change the upload bound or vice versa.
- **Observed P04_104 characteristics (measured):** 684.7 s,
  34,235 source frames at 50 FPS, 1920×1080; 685 extracted
  frames in ~229 s; ~390 MB of retained JPEGs; 685 sequential
  requests in batches of 5 (tens of minutes of inference);
  ~6.5 GB total local footprint including input, staging, and
  frame copies. Machine: 15.4 GB RAM, ample free disk.
- **Operational recommendations:** run large videos with no
  other heavy processes, set the pipeline bound in the same
  shell that launches the worker, prefer the one-shot launcher
  for paid runs, and back up `runs_api/` before long executions
  (`scripts/backup_data.py run`).
