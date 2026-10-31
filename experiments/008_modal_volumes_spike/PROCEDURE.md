# Experiment 008 — Modal Volumes v2 Storage and Worker Smoke Test

Status: PROCEDURE ONLY. No cloud execution has been performed.
Cloud execution requires explicit user authorization per run.

## Objective

Verify, with a small file and a dedicated test volume, that Modal
Volumes v2 can persist a source object, serve it to an isolated
worker, and return a generated artifact with intact SHA-256
integrity. Success here unblocks implementing
`src/bakery_intelligence/api/modal_storage.py`.

## Prerequisites

- Modal account with the documented promotional credits.
- `modal` CLI installed and authenticated on the host
  (`modal token new`), verified with `modal volume list`.
- A configured US$0 personal spend limit, with the explicit
  understanding that a limit does not eliminate all delayed-billing
  or operational risk. Monitor usage after every run.
- The official references below, read before execution:
  - https://modal.com/docs/guide/volumes
  - https://modal.com/pricing

## Step 0 — Verify the API (mandatory, no cloud cost)

The draft script `modal_smoke.py` in this directory is marked
UNVERIFIED: it was written without access to the official
documentation. Before any execution:

1. Open https://modal.com/docs/guide/volumes.
2. Confirm the current Volume v2 creation, lookup, file-write,
   file-read, reload/commit, and deletion calls.
3. Confirm worker file access semantics (mount vs. network reload).
4. Confirm whether Volume v1 and v2 calls differ for the operations
   used here; do not assume interchangeability.
5. Correct `modal_smoke.py` to match the verified API and record the
   documentation revision or retrieval date in the run notes.

Do not execute the draft until Step 0 is complete.

## Step 1 — Create a dedicated test volume

Create a volume used only by this experiment, for example
`bakery-spike-008`. Record its exact name and creation output.

## Step 2 — Upload a small test file

Upload a file under 1 MB (for example a generated 256 KB random
blob). Record the local SHA-256 before upload.

## Step 3 — Verify integrity from an isolated worker

Run a worker function that reads the file from the volume,
recomputes SHA-256, and returns the digest. It must match Step 2
exactly. Record the worker logs.

## Step 4 — Generate and retrieve a test artifact

From the same worker, write a small JSON artifact (source name,
digest, worker id, UTC timestamp) back to the volume, download it
locally, and validate its contents and hash.

## Step 5 — Record observations

Capture into the run notes:

- Volume name, region, and observed latency per operation.
- Exact CLI/SDK calls used (post-verification).
- Credit or usage delta reported by Modal, if visible.
- Any deviation from this procedure.

## Step 6 — Clean up

Delete the test file, the artifact, and the test volume. Verify
deletion with a fresh `modal volume list`. Record confirmation.

## Spend guardrails

- Small files only; one volume; one worker invocation per step.
- Stop immediately on unexpected charges, quota prompts, or API
  behavior that contradicts the verified documentation.
- Never store EPIC-KITCHENS video, extracted frames, or licensed
  media on cloud volumes without a separate redistribution review.
