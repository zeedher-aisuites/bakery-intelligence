# Experiment 003 — P04_104 Manual Video Observation Log

## Record metadata

| Field | Value |
|---|---|
| Video ID | `P04_104` |
| Source | EPIC-KITCHENS-100 |
| Coverage | Approximately 00:15 to video end (approximately 11:25) |
| Annotation method | Human review of the original video, reported by the reviewer |
| Record status | Preliminary human observations; not independently frame-verified |
| Timestamp convention | Approximate `MM:SS` positions in the original video |
| Mass unit | Grams hypothesized from context; unit not visible on the scale |
| Intended use | Candidate-event localization and independent reference-annotation preparation |

## Interpretation rules

1. **Observed** means the reviewer reported a visible action, display state, or number. It does not mean an independent second annotator has verified the frame.
2. **Approximate** indicates that the reviewer supplied an estimated timestamp or reading.
3. **Inferred** indicates a contextual interpretation (such as ingredient identity, grams, cause of a reading change, or the purpose of a battery).
4. **Unknown** is retained where the source does not establish a fact. A displayed value is not automatically the net mass of an ingredient.
5. Scale readings during stirring, physical contact, vessel removal, or absent vessel placement are candidates for *measurement artifacts*, not valid ingredient masses without additional evidence.
6. This document is a transcription and structured organization of the manual review, **not** ground truth for model scoring. Verified reference annotations must subsequently cite individual frames and independently adjudicate uncertain values.
7. No model-generated readings or classifications were used to prepare this log.

## Segment 1 — Initial pancake-mixture weighing and tare (00:15–01:28)

| Time | Reported observation | Reading / state | Interpretation and verification need |
|---|---|---|---|
| 00:15 | The person begins adding pancake mixture to a glass bowl. | Shortly afterward the scale displays `0`. | Identify the exact moment of zeroing and whether the bowl is on the scale. |
| ~01:18 | The bowl is removed from the scale. | Last reported maximum reading: `100`. | Reading is reported, but its relation to net ingredient mass is not established. |
| After ~01:18 | The scale is visible after the bowl is removed. | `-750`. | Negative reading may reflect an earlier tare/reference; the mechanism is unverified. |
| ~01:26 | The person places a vessel containing pancake batter on the scale and operates the tare control. | Tare action reported. | Determine action order and confirm the button press from frames. |
| ~01:28 | The vessel with batter remains on the scale. | `0`. | Candidate new measurement baseline. |

**Segment-level uncertainty:** Whether the initial `100` represents net pancake mixture, a transient display value, or a different reference is unresolved. The displayed `-750` is not evidence of ingredient mass by itself.

## Segment 2 — Addition of unidentified dry ingredient (01:35–02:04)

| Time | Reported observation | Reading / state | Interpretation and verification need |
|---|---|---|---|
| 01:35 | A substance resembling flour is added to the vessel containing batter. | Not reported. | Ingredient identity is inferred; flour is not confirmed. |
| ~02:04 | Addition stops and the vessel is removed from the scale. | `29`. | Confirm whether the reading occurs before, during, or after vessel removal and whether it is stable. |

**Segment-level uncertainty:** The net quantity of the added substance cannot be assigned solely from the reported reading without verifying the tare state and the timing of the display.

## Segment 3 — Repeated water addition, stirring, and force artifacts (02:04–03:23)

| Time | Reported observation | Reading / state | Interpretation and verification need |
|---|---|---|---|
| 02:04 | Water is added to the vessel while it is off the scale. | No scale reading. | Confirm visible liquid and vessel position. |
| 02:09 | Vessel returns to the scale after water addition, then is removed for more water. | `53`. | Candidate snapshot after first water addition. |
| 02:13 | Vessel returns after additional water, then is removed again. | `54–55`. | Ambiguous reported reading; do not force a single value. |
| 02:18 | Vessel returns after additional water, then is removed again. | `56`. | Candidate snapshot after additional water. |
| 02:24 | Vessel returns after additional water and contents are stirred while on the scale. | `60` before stirring; peaks near `600` during stirring. | High readings during hand contact are suspected force artifacts, not added mass. |
| 02:33 | Vessel is removed; more water is added while stirring continues. | Negative numbers visible on the scale. | Negative readings with vessel removed are suspected reference/tare artifacts. |
| 02:50 | More water is added while stirring continues. | Negative numbers visible on the scale. | Treat as potential noise; inspect vessel placement and display. |
| 03:00 | Vessel is placed on the scale again. | Partially occluded `54`, inferred by the reviewer from context. | **Do not label `54` as a directly readable number** until verified; retain partial-visibility status. |
| 03:06 | After a camera movement, the scale and vessel are visible again; vessel is then removed for cleaning. | `69`. | Candidate reading; inspect stability and timing. |
| 03:18 | Vessel returns to the scale. Plastic film is placed over it. | `68` before covering; spurious readings during contact. | Verify baseline and identify film-handling force artifacts. |
| 03:23 | The film-covered vessel is removed and taken to the freezer. | Not reported. | End of candidate weighing sequence. |

**Segment-level uncertainty:** The numerical progression is not a direct water-only mass series unless the scale reference and changes in vessel contents are established. The reported `54` at 03:00 is a contextual human inference from a partial display.

## Segment 4 — Powder addition, low battery, replacement, water additions, mixing, and cleanup (03:30–video end)

| Time | Reported observation | Reading / state | Interpretation and verification need |
|---|---|---|---|
| 03:30 | Glass bowl is placed on the scale and the scale is tared. | `0` around 03:34. | Candidate tare and new baseline. |
| 03:45 | Person returns after retrieving an ingredient; scale display appears off. Scale is turned on and tared again. | Display off, then reinitialized. | Confirm whether the same tare baseline is restored. |
| 03:48 | Scale is seen through the glass bowl. | Noisy/partially obscured `0`. | Reading visibility reduced by glass. |
| 03:51 | Powder resembling flour or sugar begins to be added. | Not reported. | Ingredient identity unknown. |
| 04:08 | Powder is being added. | `448` through glass, followed by `Lo`. | `448` is a reported noisy reading, not yet a verified final mass; `Lo` is consistent with low battery but requires display confirmation. |
| 04:09 | Ingredient addition stops; display turns off. | Display off. | Loss of measurement continuity. |
| 04:13 | Bowl is removed; person takes the scale elsewhere. | Display off. | Start of scale interruption. |
| 04:55 | Coin-cell-type batteries are found in drawers. | Not applicable. | Battery purpose inferred from subsequent actions. |
| 05:13 | First battery of the scale is changed. | Not applicable. | Physical maintenance event. |
| 05:40 | Second battery is changed; scale display subsequently turns on. | Display on. | New power session; reference continuity not assumed. |
| 06:07 | Glass bowl containing the powder is placed on the scale and scale is tared. | `0`. | **New measurement reference**; pre-battery readings cannot simply be added to post-tare readings. |
| 06:19 | Water is poured into a second vessel resembling a ladle. | Not reported. | Water identity reported by reviewer; check visual evidence. |
| 06:22 | Water is poured from the second vessel into the glass bowl. | Not reported. | The reviewer proposes ~`448` as the last powder reading before power loss; this remains an inference, not verified net powder mass. |
| 06:24 | More water is placed in the second vessel. | `38`. | Candidate cumulative post-tare reading. |
| 06:27 | Water is poured into the glass bowl; more water is prepared. | `108`. | Candidate cumulative post-tare reading. |
| 06:32 | More water is poured; additional water is prepared. | `167`. | Candidate cumulative post-tare reading. |
| 06:35 | More water is poured; additional water is prepared. | `238`. | Candidate cumulative post-tare reading. |
| 06:40 | More water is poured. | `286`. | Candidate cumulative post-tare reading. |
| 06:45 | Water is poured; the person attempts to remove some water. | Maximum reported `319`. | Maximum is not necessarily the final water quantity. |
| 06:48 | Some contents are removed, but removal itself is not clearly visible. | `315`. | Candidate corrected reading; removed substance/amount not directly verified. |
| 06:51 | A new ingredient is added to the bowl. | `315` before addition. | Ingredient identity unknown; transition to a new addition. |
| 06:57 | Addition of the new ingredient ends. | Brief `323`. | Difference from `315` is a candidate net change of 8 display units **only if** reference and stability are confirmed. |
| 07:05 | Person retrieves a knife and begins mixing contents on the scale. | `323` visible again immediately before mixing. | Readings during subsequent mixing are suspected contact-force artifacts. |
| 08:03 | Mixing stops; knife is removed. | `320`. | Reviewer hypothesizes that material adhered to the knife; cause of decrease not independently established. |
| 08:09 | Bowl is covered with a yellow cloth and removed from the scale. | Not reported. | End of mixing/weighing sequence. |
| 08:16 | While holding the covered bowl, person operates scale control. | No confirmed tare. | Bowl position is partly inferred because hands are not visible. |
| 08:33 | After repeated/long button presses, the scale turns off. | Display off. | Reviewer interprets the operation as power-off rather than successful tare. |
| 08:48 | Scale is placed in a drawer. | Not applicable. | Scale no longer in use. |
| 08:58 | Pan and flexible plastic spatula are removed from an induction-style stovetop and stored elsewhere. | Not applicable. | Induction surface identification inferred from appearance. |
| 09:05–~10:40 | Utensils are collected and rearranged, including placement in a dishwasher. | Not applicable. | Approximate ~95-second cleanup sequence. |
| 10:43 | A tablet resembling detergent is added to the dishwasher. | Not applicable. | Detergent identity inferred from appearance. |
| 10:50 | With dishwasher door ajar, person turns the appliance on and then closes it. | Not applicable. | Appliance operation reported. |
| 10:54 | Digital controls of the cooktop are operated; a display increases to what appears to be `0:30:00` over at least 10 seconds. | `0:30:00` (apparent timer). | **Not a scale reading**; useful negative example for numeric-display classification. |
| Video end | Person picks something up from the floor and washes hands. | Not applicable. | Final cleanup activity. |

## Candidate scale-reading sequence (reported, not independently verified)

| Timestamp | Display | Context | Preliminary quality flag |
|---|---|---|---|
| ~01:18 | `100` | Bowl before removal | `needs_frame_review` |
| After ~01:18 | `-750` | Bowl removed | `negative_reference_or_tare` |
| ~01:28 | `0` | Tare with vessel present | `candidate_tare` |
| ~02:04 | `29` | End of dry-ingredient addition | `needs_frame_review` |
| 02:09 | `53` | Vessel returned after water | `candidate_reading` |
| 02:13 | `54–55` | Vessel returned after water | `ambiguous_digits` |
| 02:18 | `56` | Vessel returned after water | `candidate_reading` |
| 02:24 | `60` | Before stirring | `candidate_reading` |
| 02:24 onward | ~`600` | Stirring/hand pressure | `force_artifact` |
| 02:33, 02:50 | Negative (unspecified) | Vessel removed | `negative_reference_or_tare` |
| 03:00 | `54` (partly visible) | Vessel returned | `contextually_inferred_digits` |
| 03:06 | `69` | Vessel on scale | `candidate_reading` |
| 03:18 | `68` | Vessel on scale before covering | `candidate_reading` |
| ~03:34 | `0` | Tare | `candidate_tare` |
| 03:48 | `0` | Through glass | `obscured_display` |
| 04:08 | `448` | Powder addition | `obscured_display` |
| Immediately after 04:08 | `Lo` | Battery warning candidate | `display_status_not_mass` |
| 06:07 | `0` | Tare after battery replacement | `candidate_tare` |
| 06:24 | `38` | Water addition sequence | `candidate_reading` |
| 06:27 | `108` | Water addition sequence | `candidate_reading` |
| 06:32 | `167` | Water addition sequence | `candidate_reading` |
| 06:35 | `238` | Water addition sequence | `candidate_reading` |
| 06:40 | `286` | Water addition sequence | `candidate_reading` |
| 06:45 | `319` | Before attempted removal | `transient_maximum` |
| 06:48 | `315` | After attempted removal | `candidate_reading` |
| 06:51 | `315` | Before new ingredient | `candidate_baseline` |
| 06:57 | `323` | After new ingredient | `brief_reading` |
| 07:05 | `323` | Immediately before mixing | `candidate_reading` |
| 08:03 | `320` | After mixing | `candidate_reading` |
| 10:54 onward | `0:30:00` | Cooktop timer | `non_scale_display` |

## Preliminary candidate events for later adjudication

| Candidate | Approximate interval | Event type | Critical question |
|---|---|---|---|
| `C01` | 00:15–01:18 | Initial pancake-mixture addition | Does `100` represent an ingredient net mass? |
| `C02` | 01:18–01:28 | Vessel removal and tare | What does `-750` mean relative to the prior tare? |
| `C03` | 01:35–02:04 | Unidentified dry-ingredient addition | Is `29` stable and attributable to the added substance? |
| `C04` | 02:04–03:23 | Repeated water addition and mixing | Which readings are stable, which are force artifacts, and is net water mass recoverable? |
| `C05` | 03:30–04:13 | Powder addition interrupted by low battery | Is `448` stable and can a final powder mass be established? |
| `C06` | 04:13–06:07 | Scale battery replacement and reset | Where is measurement continuity lost and re-established? |
| `C07` | 06:19–06:48 | Water addition with apparent correction | Is `315` a stable corrected cumulative reading? |
| `C08` | 06:51–07:05 | New unidentified ingredient | Is the `315 → 323` change attributable to one ingredient? |
| `C09` | 07:05–08:09 | Mixing and material removal | Can transient forces be separated from true mass change? |
| `C10` | 08:16–08:48 | Scale shutdown and storage | Is the control action power-off rather than tare? |
| `C11` | 08:58–video end | Cleanup and appliance operations | Can non-recipe actions and non-scale numeric displays be excluded? |

## Evidence requirements for verified reference annotations

For each candidate event, capture the original-video frame index and timestamp, image reference, display crop where applicable, observed digits and uncertainty, vessel-on-scale state, action interval, ingredient identity confidence, tare/power-session state, and whether the reading is stable, transient, occluded, negative, or affected by contact. A second human review or explicit adjudication is needed before using uncertain labels for quantitative scoring.

## Experimental limitations

- All timestamps and readings in this record originate from one human review and are approximate unless later frame-verified.
- The scale does not visibly provide a unit in the reported observations; grams remain a contextual hypothesis.
- Ingredient identity is often ambiguous from appearance alone.
- Scale readings may reflect vessel tare, cumulative contents, hand pressure, power interruption, and removal of material rather than a single ingredient's net mass.
- This is a purposively selected single-video case study; results will not establish performance across kitchens, scales, lighting, or users.
- EPIC-KITCHENS-100 licensing and attribution requirements apply to any redistributed source-derived materials.
