# Weighing Event Data Contract

## Purpose and scope

This document defines the data contract for representing observable scale readings, kitchen actions, and ingredient weighing events in Bakery Intelligence Experiment 003. The contract supports human reference annotations, model-generated observations, deterministic temporal inference, and evidence-based review. It does not imply that the EPIC-KITCHENS-100 annotations provide ground-truth ingredient masses.

## Design principles

1. **Separate observation from inference.** A displayed scale value is not automatically an ingredient's net mass.
2. **Preserve temporal evidence.** Every visual observation is linked to a video timestamp and source frame.
3. **Represent uncertainty explicitly.** Unknown values are `null`; uncertainty must not be filled with plausible estimates.
4. **Retain provenance.** Human annotations, model outputs, and reviewer decisions remain distinguishable.
5. **Avoid evaluation leakage.** Model predictions must not be used to establish human reference values.
6. **Use consistent units.** Store numeric measurements with explicit units; do not assume grams from an unreadable display.

## Logical records

The dataset consists of four logically distinct records. JSON files may group these records under a single video, but each record retains its own identifier.

### Video metadata

| Field | Type | Meaning |
| --- | --- | --- |
| `schema_version` | string | Version of this data contract, initially `1.0.0` |
| `video_id` | string | Dataset video identifier, e.g. `P04_104` |
| `source_dataset` | string | Dataset name |
| `source_video_filename` | string | Original filename, not a repository-hosted video |
| `annotation_repository_commit` | string | Git commit of the source annotation snapshot |
| `fps` | number | Reported frame rate |
| `frame_count` | integer | Reported number of frames |
| `duration_seconds` | number | Duration derived from frame count and frame rate, unless explicitly labeled otherwise |
| `timestamp_basis` | string | `frame_index_over_fps` or `presentation_timestamp` |

### Frame observations

Each observation captures a single claim about a visible frame or crop. Multiple observations may refer to the same frame.

| Field | Type | Meaning |
| --- | --- | --- |
| `observation_id` | string | Unique identifier |
| `frame_index` | integer | Zero-based frame index |
| `timestamp_seconds` | number | Video-relative timestamp |
| `image_ref` | string or null | Local relative path or stable evidence identifier |
| `region` | object or null | Optional image crop coordinates (`x`, `y`, `width`, `height`) in pixels |
| `observation_type` | string | `scale_display`, `ingredient`, `action`, `container`, or `other` |
| `scale_value` | number or null | Value displayed on the scale, if legible |
| `scale_unit` | string or null | `g`, `kg`, `oz`, `lb`, or another explicitly observed unit |
| `display_state` | string or null | `changing`, `stable`, `zero`, `occluded`, or `unknown` |
| `description` | string or null | Concise description of visual evidence |
| `provenance` | object | `source_type` (`human`, `model`, or `reviewer`) and `source_id` |
| `verification_status` | string | `confirmed`, `ambiguous`, or `unobservable` |

A `stable` display state must be supported by repeated or otherwise sufficient temporal evidence. It must not be assigned solely because one frame appears sharp.

### Action events

| Field | Type | Meaning |
| --- | --- | --- |
| `action_id` | string | Unique identifier |
| `start_seconds` / `end_seconds` | number | Inclusive start and exclusive end of the annotated interval |
| `action_type` | string | `add`, `remove`, `tare`, `place_container`, `remove_container`, `mix`, `other`, or `unknown` |
| `ingredient_id` | string or null | Ingredient identity, when supported by evidence |
| `evidence_observation_ids` | array of strings | Observations supporting the action |
| `provenance` | object | Source information |
| `verification_status` | string | `confirmed`, `ambiguous`, or `unobservable` |

### Weighing events

| Field | Type | Meaning |
| --- | --- | --- |
| `event_id` | string | Unique weighing-event identifier |
| `start_seconds` / `end_seconds` | number | Event interval |
| `ingredient_id` | string or null | Ingredient being weighed, if identifiable |
| `action_ids` | array of strings | Related actions |
| `observation_ids` | array of strings | Related observations |
| `final_display_reading` | object | `value`, `unit`, and `observation_id`; values may be `null` |
| `net_ingredient_quantity` | object | `value`, `unit`, and `derivation` |
| `measurement_mode` | string | `individual`, `cumulative`, or `unknown` |
| `tare_observed` | boolean or null | Whether a tare action was visibly supported |
| `verification_status` | string | `confirmed`, `ambiguous`, or `unobservable` |
| `review_notes` | string or null | Concise evidence-focused explanation |

The `net_ingredient_quantity.derivation` field is one of `direct_after_tare`, `difference_between_readings`, `independently_verified`, or `unknown`. A cumulative reading must not be reported as a net ingredient quantity without a supported derivation.

## Reference annotations and predictions

Human reference annotations and model predictions use the same conceptual contract but must be stored separately. A recommended local layout is:

```text
data/
  raw/epic_kitchens/P04_104.MP4
  processed/experiment_003/...
experiments/003_video_to_recipe/
  annotations/reference/P04_104.json
  predictions/<run_id>/P04_104.json
  evaluations/<run_id>/metrics.json
```

Whether a reference annotation can be committed to a public repository depends on its content and the dataset's redistribution conditions. The source video and generated image crops remain local by default.

## Illustrative example (synthetic; not P04_104 ground truth)

```json
{
  "schema_version": "1.0.0",
  "video_id": "SYNTHETIC_EXAMPLE",
  "frame_observations": [
    {
      "observation_id": "obs_001",
      "frame_index": 500,
      "timestamp_seconds": 10.0,
      "image_ref": null,
      "region": null,
      "observation_type": "scale_display",
      "scale_value": 150,
      "scale_unit": "g",
      "display_state": "stable",
      "description": "Display remains at 150 g across a verified interval.",
      "provenance": {"source_type": "human", "source_id": "annotator_01"},
      "verification_status": "confirmed"
    }
  ],
  "action_events": [],
  "weighing_events": [
    {
      "event_id": "weigh_001",
      "start_seconds": 5.0,
      "end_seconds": 11.0,
      "ingredient_id": "flour",
      "action_ids": [],
      "observation_ids": ["obs_001"],
      "final_display_reading": {"value": 150, "unit": "g", "observation_id": "obs_001"},
      "net_ingredient_quantity": {"value": null, "unit": null, "derivation": "unknown"},
      "measurement_mode": "unknown",
      "tare_observed": null,
      "verification_status": "ambiguous",
      "review_notes": "The display is legible, but prior scale state is not established."
    }
  ]
}
```

## Validation invariants

- Identifiers are unique within each record collection.
- All event intervals satisfy `0 <= start_seconds < end_seconds <= video_duration` when duration is known.
- Observation timestamps are nonnegative and within the video duration.
- Referenced observation and action identifiers resolve within the same annotation dataset.
- A `confirmed` net quantity requires a non-null value, an explicit unit, and an evidence-supported derivation.
- `unobservable` quantities must not be silently converted into numeric zero.
- Scale values and ingredient quantities are never conflated solely because they share a numeric value.
- Frame-index timestamps are estimates unless validated against actual presentation timestamps.

## Evaluation implications

OCR accuracy is evaluated against human-verified readable display values. Ingredient attribution and net-mass accuracy are evaluated only where the reference evidence supports a determinate answer. Ambiguous or unobservable cases are reported separately; they are not treated as automatically correct predictions. Temporal event matching must specify its tolerance and overlap rule before scoring.

## Known limitations

The source dataset provides action annotations, not verified digital-scale readings or recipe quantities. Video occlusion, reflections, display refresh behavior, and cumulative weighing may prevent unambiguous measurement recovery. This contract represents those cases explicitly rather than imputing unsupported values.
