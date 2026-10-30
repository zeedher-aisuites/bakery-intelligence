"""Offline inference-cost projector for full-video frame analysis (Exp. 005).

Estimates Meta Model API token consumption and USD cost for submitting
extracted P04_104 frames to `muse-spark-1.3` (Standard tier) at
configurable sampling rates. Deterministic, stdlib-only, no network
access, no API calls, no media reads: all inputs are explicit CLI
arguments with documented defaults.

Frame counts replicate the sampling loop of
`experiments/003_video_to_recipe/extract_frames.py`: sample `n` covers
`start + n / rate` while `< end - 1e-9`, so counts are exact for the
half-open interval, not `duration * rate` rounded.

Measured inputs (defaults, see cost_report.json for provenance):
  duration 684.7 s, source 50 FPS / 34,235 frames / 1920x1080.
Assumed inputs (defaults, override via CLI):
  100 text tokens, 2800 image tokens, 550 output tokens per request;
  $1.25 / $4.25 per 1M input / output tokens (Standard tier).
The per-image token figure is inferred from the single Baseten vision
measurement (2901 prompt tokens incl. ~100 text tokens); Meta-side
image tokenization was never measured. Treat every USD figure as a
projection, not a measurement.

Example:
  python experiments/005_full_video_coverage/estimate_cost.py --json
  python experiments/005_full_video_coverage/estimate_cost.py --rate 2
"""

from __future__ import annotations

import argparse
import json

DURATION_SECONDS = 684.7
SOURCE_FPS = 50.0
DEFAULT_TEXT_TOKENS = 100
DEFAULT_IMAGE_TOKENS = 2800
DEFAULT_OUTPUT_TOKENS = 550
DEFAULT_INPUT_PRICE_PER_M = 1.25
DEFAULT_OUTPUT_PRICE_PER_M = 4.25

# Optional second pass: dense 5 FPS windows over complex operations,
# derived from preliminary candidate events (approximate timestamps).
SECOND_PASS_RATE = 5.0
SECOND_PASS_WINDOWS = (
    {"window_id": "W1", "start": 75.0, "end": 91.0,
     "rationale": "S01 tail: bowl removal, -750 reading, tare, re-zero"},
    {"window_id": "W2", "start": 121.0, "end": 206.0,
     "rationale": "Dry-addition end plus full water/stirring span (C04)"},
    {"window_id": "W3", "start": 207.0, "end": 256.0,
     "rationale": "Powder through glass, 448/Lo, display-off (C05)"},
    {"window_id": "W4", "start": 376.0, "end": 492.0,
     "rationale": "Post-tare water series, correction, mixing (C07-C09)"},
    {"window_id": "W5", "start": 493.0, "end": 531.0,
     "rationale": "Mixing end, shutdown presses, scale storage (C10)"},
)


def count_samples(start: float, end: float, rate: float) -> int:
    """Replicate the extractor sampling loop for [start, end)."""
    count = 0
    while start + count / rate < end - 1e-9:
        count += 1
    return count


def frame_indices(start: float, end: float, rate: float) -> set[int]:
    """Source frame indices sampled by one pass (dedup identity)."""
    return {
        int((start + n / rate) * SOURCE_FPS + 0.5)
        for n in range(count_samples(start, end, rate))
    }


def project(
    image_count: int,
    text_tokens: int,
    image_tokens: int,
    output_tokens: int,
    input_price_per_m: float,
    output_price_per_m: float,
) -> dict[str, float | int]:
    input_tokens = image_count * (text_tokens + image_tokens)
    completion_tokens = image_count * output_tokens
    cost = (
        input_tokens * input_price_per_m / 1_000_000
        + completion_tokens * output_price_per_m / 1_000_000
    )
    return {
        "image_count": image_count,
        "input_tokens": input_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": input_tokens + completion_tokens,
        "cost_usd": round(cost, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--text-tokens", type=int, default=DEFAULT_TEXT_TOKENS)
    parser.add_argument("--image-tokens", type=int, default=DEFAULT_IMAGE_TOKENS)
    parser.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    parser.add_argument("--input-price", type=float,
                        default=DEFAULT_INPUT_PRICE_PER_M)
    parser.add_argument("--output-price", type=float,
                        default=DEFAULT_OUTPUT_PRICE_PER_M)
    parser.add_argument("--with-second-pass", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    baseline = frame_indices(0.0, DURATION_SECONDS, args.rate)
    result: dict[str, object] = {
        "baseline": project(
            len(baseline), args.text_tokens, args.image_tokens,
            args.output_tokens, args.input_price, args.output_price,
        ),
    }
    if args.with_second_pass:
        targeted: set[int] = set()
        windows = []
        for window in SECOND_PASS_WINDOWS:
            indices = frame_indices(
                window["start"], window["end"], SECOND_PASS_RATE
            )
            targeted |= indices
            windows.append({
                "window_id": window["window_id"],
                "frame_count": len(indices),
            })
        new_indices = targeted - baseline
        result["second_pass"] = {
            "windows": windows,
            "targeted_frame_count": len(targeted),
            "shared_with_baseline": len(targeted & baseline),
            "unique_new_frames": project(
                len(new_indices), args.text_tokens, args.image_tokens,
                args.output_tokens, args.input_price, args.output_price,
            ),
        }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
