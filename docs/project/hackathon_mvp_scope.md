# Bakery Intelligence — Hackathon MVP Scope

**Status:** Planned implementation scope; verified behavior must be supported by executable evidence.

## Delivery goal

Demonstrate an **asynchronous, evidence-grounded workflow** that accepts a recorded bakery or pastry preparation, extracts time-indexed observations, builds a chronological execution history, and produces a **recipe draft for mandatory human review**. A visible digital scale is an initial eligibility constraint for scale-focused experiments, not a permanent product requirement.

The goal is a **working, inspectable vertical slice**, not a production SaaS or real-time assistant.

## Priorities

| Priority | Deliverable | Acceptance evidence |
|---|---|---|
| **P0 — Essential** | Input video, timestamped visual observations, linked chronological events, reviewable draft with unknowns | Reproducible end-to-end demonstration on a supported input |
| **P1 — Core reasoning** | Tare-aware measurements, event memory, evidence retrieval, uncertainty handling, basic cost logs | Tests and inspectable outputs showing how evidence supports events |
| **P2 — Multimodal validation target** | Timestamped audio transcription, ingredient/quantity claims, audio-video alignment | Working comparison of visual-only and multimodal configurations |
| **Conditional** | Targeted reanalysis, Telegram HITL | Only if core pipeline and evaluation are stable |
| **Deferred** | Multi-tenant SaaS, billing, inventory integrations, real-time alerts | Outside hackathon delivery |

**Audio is part of the intended final validation**, but cannot be represented as implemented until demonstrated. If schedule constraints prevent it, deliver and evaluate the supported visual system, and explicitly mark multimodal validation incomplete. Do not compromise P0 to present unfinished P2 functionality.

## Minimal data and workflow contract

Each source video has an ID and duration. Each observation has a modality, timestamp/interval, source reference, content, and uncertainty. Each event references supporting observations and preserves action order, tare state, unresolved identities, and quantity provenance. The output is an **execution-derived recipe draft**, not an automatically approved recipe.

Jobs may perform multiple analysis passes, but must respect limits on calls, retries, and spending. Stopping at a budget limit produces a clearly labeled partial result rather than invented details.

```mermaid
flowchart LR
    A[Video] --> B[Batch analysis]
    B --> C[Evidence and events]
    C --> D[Recipe draft]
    D --> E[Human review]
```

## Decision gates

1. **Schema:** freeze a minimal observation/event contract before expanding modalities.
2. **Vertical slice:** demonstrate one supported interval from input to evidence-linked draft.
3. **Evaluation:** preserve outputs, errors, costs, and independent reference labels; do not tune on holdout videos.
4. **Submission:** publish executable instructions, demo evidence, results, limitations, and repository references.

If full recipe reconstruction is not working, report the narrower verified capability (for example, tare-aware scale-event extraction). **Do not call a component-level success an end-to-end recipe success.**

## Evidence and completion criteria

Previously reported work includes API connectivity, controlled synthetic scale-reading tests, a real video with preliminary observations, and frame-extraction tests. These do not establish a working recipe-reconstruction pipeline.

A delivered feature is **verified** only when its code, input, output, and test or experiment record can be inspected. Repository communication must distinguish **Implemented / Verified / Proposed / Not included**. No official judging criteria or model performance should be assumed without a source.
