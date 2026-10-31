# M2-06A spike setup and execution (Windows PowerShell)

All commands run from the repository root unless stated otherwise.
No cloud execution is part of setup.

## 1. Backend dependencies

```powershell
uv sync
```

This installs the new `fastapi`, `uvicorn`, and `python-multipart`
declarations and refreshes `uv.lock`. Verify:

```powershell
uv run python -m pytest tests/test_api_storage.py tests/test_api_jobs.py tests/test_api_app.py -v
uv run python -m pytest -q
uv run python scripts/sync_contract_schemas.py --check
```

Expected: full suite green (708 existing tests plus the new API
tests), 9/9 schemas synchronized.

## 2. Start the backend

```powershell
uv run uvicorn bakery_intelligence.api.app:build_app --factory --port 8000
```

Health check (second terminal or browser):

```powershell
Invoke-RestMethod http://localhost:8000/api/health | Format-List
```

API run storage lands in the gitignored `runs_api/` directory,
beside the durable job registry `runs_api/jobs.sqlite3`.

## 3. Start the worker (required for execution)

Submitted runs stay `queued` until a worker claims them. In a second
terminal:

```powershell
uv run python -m bakery_intelligence.api.worker
```

For a single deterministic pass (one job or idle exit):

```powershell
uv run python -m bakery_intelligence.api.worker --once
```

A crashed worker's running jobs are recovered automatically: the
next worker pass requeues stale jobs within their retry budget or
fails them explicitly. Restarting the API never loses job state.

## 4. Frontend

Prerequisites: Node.js 20+.

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. Select a small local file (backend
cap: 16 MB), run the storage smoke test, and watch the run id,
truthful status transitions, SHA-256, and result artifact link.

Type-check and production build:

```powershell
npm run build
```

## 5. Modal experiment (separate authorization required)

Do not run until the user explicitly authorizes cloud execution:

1. Complete Step 0 of
   `experiments/008_modal_volumes_spike/PROCEDURE.md` (verify the
   official Volumes v2 API; correct the draft script).
2. Follow Steps 1-6 with small files only, monitoring spend.
3. Record observations and clean up the test volume.

## 6. Stop

Stop the frontend, the backend, and the worker (`Ctrl+C` in each
terminal). Job state persists in `runs_api/jobs.sqlite3`; delete
that file only for a deliberate local reset. No background
services, databases, or cloud resources remain.
