# Dataset and Video Selection — EPIC-KITCHENS-100

**Project:** Bakery Intelligence  
**Experiment:** 003 — Video-to-Recipe / Temporal Weighing Analysis  
**Recorded:** 2026-10-26

## 1. Research objective

Select realistic kitchen footage for evaluating digital kitchen-scale OCR, temporal correlation of weight readings, ingredient/action recognition, and evidence-backed recipe reconstruction. The goal is to identify the final stable weight associated with each weighing event, not simply the last visible numeric reading.

## 2. Dataset discovery and selection rationale

YouCook2 was considered as a cooking-instruction video dataset. During manual exploration, its videos were judged less suitable for this particular experiment because visible digital kitchen scales and readable displays were not readily found. This is an exploratory observation, not a quantitative evaluation or a claim that YouCook2 contains no scales.

EPIC-KITCHENS-100 was selected for closer examination because it contains first-person kitchen footage, timestamped action annotations, and machine-readable noun and verb classes. Its official annotations explicitly include the noun class `scale` (class 108, including `scale:kitchen` and `reading:scale`) and the verb class `measure` (class 69, including `weigh`). Relevant food classes include `dough` (25), `flour` (75), `butter` (78), and `sugar` (102).

Official resources:

- Dataset: https://epic-kitchens.github.io/
- Annotations: https://github.com/epic-kitchens/epic-kitchens-100-annotations
- Download scripts: https://github.com/epic-kitchens/epic-kitchens-download-scripts
- Academic Torrents: https://academictorrents.com/details/c92b4a3cd3834e9af9666ac82379ff15ca289a83

## 3. Repository accessibility, video availability, and licensing

| Resource | Observed status | Important distinction |
|---|---|---|
| Annotation GitHub repository | Public GitHub repository | Public access does not mean unrestricted reuse. |
| Download-scripts GitHub repository | Public GitHub repository | Script source is separately hosted from the video files. |
| EPIC-KITCHENS-100 video distribution | Publicly discoverable via the official download instructions and Academic Torrents | Availability is not the same as public-domain status. |
| Dataset licensing | README states Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) | Requires attribution; prohibits commercial use without separate authorization. |
| `P04_104.MP4` | Listed in the Academic Torrents distribution; downloaded and inspected locally by the experimenter | Do not assume the video is public domain or approved for unrestricted redistribution. |

Official licensing statement: https://github.com/epic-kitchens/epic-kitchens-100-annotations#license  
License text: https://creativecommons.org/licenses/by-nc/4.0/

**Project data-use policy:** The source video is treated as third-party material licensed for non-commercial use with attribution. The original MP4 is not redistributed in the Bakery Intelligence repository. Any publication of excerpts or frames must comply with the license and applicable rights; commercial uses require separate authorization.

## 4. Reproducible annotation search

The selection procedure uses the following public annotation files:

- `EPIC_100_noun_classes.csv`
- `EPIC_100_verb_classes.csv`
- `EPIC_100_train.csv`
- `EPIC_100_validation.csv`

**First-pass filter:** Case-insensitive substring search for `scale`, `weigh`, or `measur` across narration, verb, noun, and all-nouns fields.

**Observed result:** 408 matching annotated segments across 70 distinct video IDs. This broad pass includes measuring cups/jugs and other irrelevant matches.

**Second-pass filter:** Case-insensitive search for `scale` or `balance` in noun labels, `scale` in narration, or `weigh` in narration/verb.

**Observed result:** 165 matching annotated segments across 22 distinct video IDs. This count still includes false positives such as `P10_04`, where `scale` means fish scales. These are candidate counts, not a measured precision or recall score.

The search uses the publicly available EPIC-KITCHENS-100 training and validation annotation files. Results describe the inspected dataset snapshot; the reproduction script below provides the filtering logic.

### Reproduction script

Run in the root of the annotation repository:

```python
import csv
from collections import defaultdict
from pathlib import Path

matches = defaultdict(list)
for filename in ("EPIC_100_train.csv", "EPIC_100_validation.csv"):
    with Path(filename).open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            narration = row.get("narration", "").lower()
            noun = row.get("noun", "").lower()
            verb = row.get("verb", "").lower()
            if ("scale" in noun or "scale" in narration
                    or "balance" in noun or "weigh" in narration
                    or "weigh" in verb):
                matches[row["video_id"]].append(row)

print("Matching segments:", sum(map(len, matches.values())))
print("Unique videos:", len(matches))
for video_id, segments in sorted(matches.items(), key=lambda item: -len(item[1])):
    print(video_id, len(segments))
    for row in segments:
        print(" ", row["start_timestamp"], row["stop_timestamp"], row["narration"])
```

## 5. Candidate ranking and manual screening

| Candidate | Matching segments | Relevant annotation evidence | Assessment |
|---|---:|---|---|
| `P04_104` | 30 | Turning on scale, opening battery compartment, inserting battery, pressing button, placing bowl | Selected: strong indication of electronic scale and repeated interaction. |
| `P04_103` | 22 | Turning on scale and repeated button presses | Candidate; ingredient weighing not established by listed examples. |
| `P03_120` | 15 | Zeroing scale, pouring sugar, adding butter | Candidate; digital display not established by annotations alone. |
| `P27_103` | 12 | Explicit `weigh butter` events around 03:59–04:06 | Strong alternative for ingredient-weight association. |
| `P04_24` | 4 | Baking tray placed on scale around 00:52–00:59 | Relevant to baking workflow. |
| `P10_04` | 14 | Removing scales from fish | Excluded: semantic false positive. |

**Selection decision:** `P04_104` was prioritized because annotations indicate an electronic scale with buttons and batteries, followed by real-world visual inspection that confirmed a blue-lit digital display during food preparation. Visual inspection identified pancake preparation. The inspected video duration is approximately 11 minutes and 24 seconds. The footage contains occlusions, changing ingredient loads, oblique display views, and multiple numeric readings.

Example relevant annotations for `P04_104`:

- `00:03:45.61–00:03:46.07`: `turn on scales`
- `00:05:12.66–00:05:13.81`: `open battery compartment on scales`
- `00:05:19.02–00:05:21.25`: `slide battery into scales`
- `00:06:06.20–00:06:06.87`: `press button on scales`
- `00:06:07.12–00:06:08.27`: `put bowl on scales`

**Caution:** These are action annotations, not authoritative labels of displayed grams, ingredient quantities, or recipe steps. Screen readability varies by frame.

## 6. Video acquisition and visual validation

`P04/videos/P04_104.MP4` is distributed through the official dataset download infrastructure and the EPIC-KITCHENS-100 Academic Torrents collection. The latter supports selecting individual files rather than downloading the entire dataset. The selected file is approximately 1.93 GiB.

Visual inspection confirmed a blue-backlit digital kitchen scale used during pancake preparation. Representative frames show changing readings, partial occlusion by hands and transparent bowls, reflections, and varying viewing angles. These properties make the footage suitable for testing OCR and temporal association under realistic conditions.

The source MP4 is retained outside the project repository. The dataset reference, selection procedure, and derived experimental results can be documented independently of the original video.

## 7. Ground-truth definition and evaluation protocol

EPIC-KITCHENS supplies **action-level ground truth**, including timestamps and action labels. It does **not** automatically supply verified per-frame digital scale values or the complete recipe ground truth required for Bakery Intelligence.

The evaluation protocol defines independent human annotations for:

- Ingredient identity and the evidence supporting it.
- Start/end timestamps of each weighing event.
- OCR-visible readings with timestamps and units.
- Stable reading candidates and their confidence.
- Tara/reset events, removal of containers, and cumulative-versus-incremental weights.
- Final verified ingredient quantity, or `unknown` if the evidence is insufficient.

Model predictions are not treated as reference labels. Raw OCR observations and timestamps are retained to support auditability and repeatable evaluation.

## 8. Citation

Damen, D., Doughty, H., Farinella, G. M., Furnari, A., Ma, J., Kazakos, E., Moltisanti, D., Munro, J., Perrett, T., Price, W., & Wray, M. (2022). *Rescaling Egocentric Vision*. International Journal of Computer Vision, 130, 33–55. https://arxiv.org/abs/2006.13256

Dataset and annotations: https://github.com/epic-kitchens/epic-kitchens-100-annotations  
License: https://creativecommons.org/licenses/by-nc/4.0/

### Dataset Version and Reproducibility

The video selection analysis was conducted using the
EPIC-KITCHENS-100 annotations repository at the following
Git commit:

- Repository: https://github.com/epic-kitchens/epic-kitchens-100-annotations
- Commit: `ea8b40457a400c3fffa1c7f406ef3dc169cc2522`
- Annotation files: `EPIC_100_train.csv`, `EPIC_100_validation.csv`,
  `EPIC_100_noun_classes.csv`, and `EPIC_100_verb_classes.csv`

This commit identifies the annotation version used for
candidate discovery and video selection.