# ADR M2-06A: Modular backend instead of independent microservices

Status: Accepted for the M2-06A spike.
Date: 2026-10-31.

## Context

Bakery Intelligence needs an HTTP backend and a browser frontend on
top of a validated, evidence-preserving Python analysis pipeline.
The team is one developer, the timeline is a hackathon, and the
pipeline's scientific contracts (canonical schemas, provenance,
append-only history) must not be weakened by the new serving layer.

## Decision

Build a modular monolith: one Python service composed of explicit
internal modules (`api.storage`, `api.jobs`, `api.store`,
`api.worker`, `api.modal_storage`, `api.app`) behind a small HTTP
contract, with a separately deployable React frontend. Jobs persist
in SQLite and execute in an independent worker process, so a future
Modal worker can replace the local one without changing the API.

## Rationale

- The validated pipeline (`models`, `reconstruction`, `ledger`,
  `verification`, `adapters`) is imported directly; no serialization
  boundary, version skew, or duplicated validation logic between
  services.
- One deployment unit keeps local reproduction (`uv run uvicorn`)
  identical to the reviewed spike, which matters more than
  independent scaling at this stage.
- Module boundaries (`StorageBackend`, durable job store,
  worker entry point) preserve the option to extract services
  later, once load or team size demands it.
- Microservices would add network contracts, distributed failure
  modes, and provenance-tracking complexity with no corresponding
  benefit for a single-developer MVP.

## Consequences

- The API process must never block request handlers on analysis
  work; submission only inserts a queued row, and the separate
  worker process claims and executes it.
- SQLite is the single-host local job registry; it must not serve
  as a shared transactional database across Modal workers.
- Storage stays behind `StorageBackend` so local paths never leak
  into API responses or the future Modal adapter.
- If a module needs independent scaling or a different language
  runtime, it must be extracted behind the already-stable HTTP
  contract, not by reaching into process internals.

## Reuse map (Phase 1 findings)

Directly reusable without changes:

- `models/*`: canonical Pydantic contracts and JSON schemas for
  observations, events, drafts, and review decisions.
- `verification/*`: deterministic reference and numeric checks.
- `reconstruction/*`: offline draft assembly from existing artifacts.
- `ledger/*`: append-only observation history.
- `app/adapters.py`: read-only loaders for reconstruction outputs.

Require thin adapters before HTTP exposure:

- `experiments/003_video_to_recipe/extract_frames.py`: frame
  extraction works on local paths; needs a storage-aware wrapper
  that stages uploads and records provenance (not built in M2-06A).
- `experiments/006_p04_multimodal_inference/*`: inference client
  and runner are experiment scripts with paid-API side effects;
  reuse requires a job boundary, credential handling, and cost
  accounting (explicitly out of spike scope).
- `experiments/007_selective_verification/*`: verification planner
  and executor are CLI-driven; same job-boundary requirement.
- `app/demo.py`: Streamlit reference UI stays as-is; the React
  frontend reuses its evidence vocabulary, not its code.

Not reused in the spike:

- Historical experiment outputs under `data/` (read-only evidence).
- The preliminary adapter (one-shot migration utility).
