# Experiment 003: Evidence-Based Video-to-Recipe Extraction

## Research objective

Evaluate whether a multimodal pipeline can reconstruct ingredient weighing events from an egocentric cooking video, including displayed scale readings, ingredient associations, and temporal evidence. The experiment separates observed measurements from inferred ingredient quantities and supports explicit abstention when evidence is insufficient.

## Evaluation video

- Dataset: EPIC-KITCHENS-100
- Video: `P04_104.MP4`
- Source annotation repository: https://github.com/epic-kitchens/epic-kitchens-100-annotations
- Source annotation commit: `ea8b40457a400c3fffa1c7f406ef3dc169cc2522`
- Observed video metadata: 1920 × 1080, 50 FPS, 34,235 frames; estimated duration 684.7 seconds
- Local video storage: `data/raw/epic_kitchens/P04_104.MP4` (excluded from Git)
- Dataset selection methodology: [`../datasets/epic_kitchens_100_selection.md`](../datasets/epic_kitchens_100_selection.md)

The dataset's declared license is Creative Commons Attribution-NonCommercial 4.0. Public access does not imply unrestricted commercial reuse. Original video files and extracted frames are not distributed in this repository.

## Data contract

The observation, action, and weighing-event records follow [`../schemas/weighing_event_schema.md`](../schemas/weighing_event_schema.md). Human reference annotations, model predictions, and evaluation outputs remain separate.

## Experimental design

1. **Reference annotation:** Identify weighing intervals, ingredients, scale states, legible readings, tare actions, and evidence timestamps through independent human inspection.
2. **Frame extraction:** Extract frames from selected intervals using OpenCV and preserve frame indices and timestamps in a manifest. Validate time alignment before interpreting event boundaries.
3. **Multimodal baseline:** Evaluate Muse Spark on frames and optional scale-display crops. Record raw outputs, normalized observations, latency, and token usage where available.
4. **Temporal inference:** Associate consecutive scale observations with ingredient actions, detect stable readings, and distinguish direct and cumulative measurements using explicit rules.
5. **Critical review:** Evaluate a constrained reviewer that can request neighboring frames or crops, compare conflicting observations, and confirm or abstain. Review decisions retain their evidence references.
6. **Comparative evaluation:** Compare baseline, deterministic correlation, and reviewer-assisted configurations against the independently established reference annotations.

## Evaluation measures

- Exact-match accuracy for human-readable scale displays, with unreadable cases reported separately.
- Ingredient attribution accuracy on independently verified events.
- Absolute net-mass error where a reference quantity is determinable.
- Temporal alignment error and event precision/recall under a documented matching rule.
- Abstention and unresolved-case rates.
- Runtime, model token consumption, and estimated cost per event or analyzed interval.
- Change in event-level performance and resource consumption attributable to critical review.

## Evidence and reproducibility

Each result must identify the input video, annotation snapshot, extraction parameters, model configuration, relevant timestamps, and evidence identifiers. The reference labels must not be revised using model predictions during the same evaluation. Any subsequent reference corrections require a separately documented revision.

## Scope and limitations

This experiment evaluates weighing-event reconstruction on one selected cooking video. It does not establish cross-video generalization, complete recipe correctness, or production reliability. EPIC-KITCHENS-100 action annotations are not a substitute for independently verified mass readings. Unreadable displays, occlusion, ambiguous tare behavior, and missing ingredient evidence may limit event-level conclusions.

## Result status

Video decoding was verified locally with OpenCV 5.0.0: successful video open, first-frame decode, 50 FPS, 34,235 frames, and a 1920 × 1080 first frame. No frame-extraction, weighing-event, multimodal-model, or reviewer performance results are asserted in this document.
