# Bakery Intelligence frontend (M2-06A spike)

Minimal React + TypeScript console for the FastAPI spike backend.
It checks backend health, uploads a small test file, polls a
storage smoke-test run, and displays truthful execution states.

## Prerequisites

- Node.js 20 or later (`node --version`).
- The FastAPI backend running on `http://localhost:8000` (see
  `docs/m206/spike-setup.md`).

## Install and run (Windows PowerShell)

```powershell
cd frontend
npm install
npm run dev
```

Open the printed URL (default `http://localhost:5173`) in a browser.

To point at a different backend, create `frontend/.env` (gitignored):

```text
VITE_API_BASE=http://localhost:8000
```

## Workflows and results (M2-06B.2)

The workflow radio group selects the queued workflow explicitly:

- **Storage smoke test**: upload any small file; the worker stores
  it and verifies its SHA-256. No analysis is performed.
- **Offline reconstruction**: upload a JSON offline bundle
  (`bakery-offline-bundle/1`, see
  `docs/m206/offline-integration.md` and
  `fixtures/offline/offline_bundle_http_example.json`). No
  inference, network, or paid services are involved.
- **Offline video pipeline**: upload a small video (max
  16 MB). The worker stages it, extracts frames at 1 FPS,
  runs deterministic fake inference, and reconstructs a
  draft. All model outputs are synthetic test vectors,
  labeled as such in the run summary; see
  `docs/m206/pipeline-offline.md`.

Completed offline runs render a review section below the status
cards: run summary with honesty flags, the recipe draft with its
unresolved questions, a measurement table with values, units,
provenance, and verbatim readings, the weighing event, temporal
evidence, and validation results. Absent ingredients, net
quantities, units, and tare history are shown as unknown or
undetermined, never guessed. Original JSON artifacts stay
available through the artifact links in the run status card.

## Scope limits

- Smoke-test workflow only: no video analysis is performed.
- Uploads are capped at 16 MB by the backend; large-file browser
  uploads are explicitly out of scope for this spike.
- No routing, state library, or component framework beyond React.
