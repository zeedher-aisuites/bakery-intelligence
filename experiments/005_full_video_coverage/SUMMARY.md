# Experiment 005 — Full-Video Coverage and Inference-Cost Projection

## Scope

Cover the full P04_104.MP4 timeline with a deterministic 1 FPS baseline,
design an optional denser second pass, and project offline Meta Model API
(`muse-spark-1.3`, Standard) inference cost at 0.5 / 1 / 2 FPS. No API
calls, uploads, remote token counting, or spending occurred. No M1/M2
files were modified. All media and derived frames stay under gitignored
`data/`.

## Video properties (container-measured this session)

Direct MP4 box parsing (file head + exact 583,974-byte `moov` tail)
established, without a decoder:

| Property | Value | Evidence |
|---|---|---|
| File | `data/raw/epic_kitchens/P04_104.MP4`, 2,074,288,282 bytes | Measured |
| Codec / tracks | hvc1 (H.265); vide + soun + meta + tmcd | `stsd`, `hdlr` |
| Resolution | 1920x1080 | Video `tkhd` |
| Frame count | 34,235 samples | `stsz` |
| Frame rate | Constant 50.0 FPS | `stts`: single entry, 34235 x 1800/90000 s |
| Duration | 684.700 s video (`mdhd`); 684.715 s movie (`mvhd`) | Exact rational reads |
| Keyframes | 685 (`stss`) | ~1 per 50 samples |
| Muxer tag | lavf58.12.100 (FFmpeg) in `udta` | File was remuxed after capture |

## Timestamp model (validated, not assumed)

- No `ctts` box exists, so presentation timestamps equal decoding
  timestamps: PTS(n) = n x 0.02 s exactly for sample n.
- The edit list is a single entry with `media_time = 0`, rate 1.0:
  no timeline offset, presentation starts at 0.
- The single-entry `stts` table confirms constant frame rate; there is
  no variable-frame-rate ambiguity to resolve.
- Consequence: the manifest's index-derived timestamps
  (`frame_index / 50`) are **exact**, and the extractor's `--end 684.7`
  bound matches the true video duration.

## Baseline extraction (already complete)

`data/processed/experiment_003/global_1fps/` holds **685 JPEG frames**
(step 50, indices 0–34200) plus a 685-row `manifest.csv` carrying
`image_path`, `frame_index`, `timestamp_seconds`,
`source_timestamp_seconds`, `requested_timestamp_seconds` (all equal;
offset 0). Verified this session: 685/685 files present, zero zero-byte
files, 390,341,105 bytes total (~372 MB). Mean frame size 569,798 bytes
(range 257,727–778,907).

Timed rerun (user-measured locally): `global_1fps_rerun/` reproduces
the extraction — 686 entries, 390,341,105 bytes, 685 manifest rows — in
**206.5691046 s** (~3.32 frames/s), confirming deterministic output.
Remaining gap: `cv2.__version__`, source SHA-256, and UTC timestamps
were not recorded. Capture them in a sidecar JSON on the next run:

```powershell
Measure-Command { uv run python experiments/003_video_to_recipe/extract_frames.py --video data/raw/epic_kitchens/P04_104.MP4 --output data/processed/experiment_003/global_1fps_rerun --start 0 --end 684.7 --fps 1 }
```

## Second-pass design (proposed, not executed)

Dense 5 FPS windows over complex operations from preliminary candidate
events: W1 [75,91], W2 [121,206], W3 [207,256], W4 [376,492], W5
[493,531] seconds — 1,520 targeted frames, of which 304 coincide with
baseline indices and are reused, leaving **1,216 unique new frames**.
Battery-dead (249–367 s) and cleanup (538–end) spans stay at baseline
cadence deliberately. Passes live in separate directories with a
`passes.json` registry; frame identity is the source `frame_index`, so
duplicates are never counted twice. Per-window extraction uses the same
CLI with `--start/--end/--fps 5` and distinct `--output` directories.

## Cost projection (central case)

Model `muse-spark-1.3` Standard at $1.25 / $4.25 per 1M input / output
tokens (rates as published in Meta developer documentation via multiple
secondary sources; official page not directly retrieved offline).
Per-request tokens assumed: 100 text + 2,800 image in, 550 out —
anchored to the single Baseten vision measurement (2,901 / 554), which
covers a different provider, model, and API.

| Rate | Images | Input tokens | Output tokens | USD |
|---|---|---|---|---|
| 0.5 FPS | 343 | 994,700 | 188,650 | **2.05** |
| 1 FPS | 685 | 1,986,500 | 376,750 | **4.08** |
| 2 FPS | 1,370 | 3,973,000 | 753,500 | **8.17** |
| Second pass (new) | 1,216 | 3,526,400 | 668,800 | **7.25** |

Sensitivity at 1 FPS: $1.81 (low: 1,500/150 tokens) to $8.31 (high:
4,500/1,500 tokens). Bounds are arbitrary planning margins, not
measurements.

## Reproduce locally

```powershell
python experiments/005_full_video_coverage/estimate_cost.py --json
python experiments/005_full_video_coverage/estimate_cost.py --rate 2 --with-second-pass --json
```

Figures in `cost_report.json` were verified by executed PowerShell
arithmetic (the projector script itself needs a local interpreter for
its first run). Calibrate with 3–5 recorded `muse-spark-1.3` calls
(separate API authorization required) before budgeting a full-video run.
