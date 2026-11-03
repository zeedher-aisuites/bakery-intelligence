"""Isolated native video understanding experiment (Meta Muse Spark 1.3).

Standalone evaluation of native video understanding on one local MP4:
the whole file is uploaded through the Files API (purpose="user_data")
and analyzed with a single Responses API request using an
``input_video`` block that references the uploaded ``file_id``.
No frames are extracted and no individual images are submitted.

This script is intentionally isolated from the Bakery Intelligence
production pipeline: it imports nothing from ``bakery_intelligence``
and touches no run storage, reviews, or recipe records. All outputs
land under ``experiments/native_video_understanding/<run>/``.

Safety posture:
  * Dry-run is the default. Nothing touches the network unless
    ``--execute`` is passed explicitly.
  * Exactly one inference request is made per ``--execute`` run;
    SDK retries are disabled (``max_retries=0``).
  * The API key is read from the environment (``MODEL_API_KEY``,
    falling back to ``META_API_KEY``) and is never printed,
    logged, or persisted.

Reference: https://dev.meta.ai/docs/video-understanding
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://api.meta.ai/v1"
MODEL = "muse-spark-1.3"
KEY_VARS = ("MODEL_API_KEY", "META_API_KEY")
EXPERIMENT_DIR = Path(__file__).resolve().parent / "native_video_understanding"
SCHEMA_NAME = "NativeVideoCookingAnalysis"
DEFAULT_MAX_OUTPUT_TOKENS = 16384
DEFAULT_TIMEOUT_SECONDS = 1800.0

# Structured-output contract for the whole-video analysis. Flat object
# without $ref: recursive schemas return HTTP 400 on this API family.
ANALYSIS_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "overall_activity": {"type": ["string", "null"]},
        "ingredients": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "visible_label": {"type": ["string", "null"]},
                    "normalized_name": {"type": ["string", "null"]},
                    "supporting_timestamps": {
                        "type": "array",
                        "items": {"type": ["number", "null"]},
                    },
                    "evidence": {"type": ["string", "null"]},
                    "ambiguities": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "used_or_visible": {"type": ["string", "null"]},
                },
                "required": ["visible_label", "normalized_name",
                             "supporting_timestamps", "evidence",
                             "ambiguities", "used_or_visible"],
                "additionalProperties": False,
            },
        },
        "containers_utensils": {
            "type": "array", "items": {"type": "string"},
        },
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event_id": {"type": ["string", "null"]},
                    "action_type": {"type": ["string", "null"]},
                    "action_description": {"type": ["string", "null"]},
                    "start_seconds": {"type": ["number", "null"]},
                    "end_seconds": {"type": ["number", "null"]},
                    "ingredients": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "containers_utensils": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "transfers": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "scale_readings": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "evidence": {"type": ["string", "null"]},
                    "confidence": {"type": ["string", "null"]},
                    "uncertainty": {"type": ["string", "null"]},
                    "preceding_event_id": {"type": ["string", "null"]},
                    "following_event_id": {"type": ["string", "null"]},
                },
                "required": [
                    "event_id", "action_type", "action_description",
                    "start_seconds", "end_seconds", "ingredients",
                    "containers_utensils", "transfers",
                    "scale_readings", "evidence", "confidence",
                    "uncertainty", "preceding_event_id",
                    "following_event_id",
                ],
                "additionalProperties": False,
            },
        },
        "scale_readings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp_seconds": {"type": ["number", "null"]},
                    "displayed_value": {"type": ["string", "null"]},
                    "visible_unit": {"type": ["string", "null"]},
                    "evidence": {"type": ["string", "null"]},
                    "tare_observable": {"type": ["boolean", "null"]},
                    "uncertain": {"type": ["boolean", "null"]},
                },
                "required": [
                    "timestamp_seconds", "displayed_value",
                    "visible_unit", "evidence", "tare_observable",
                    "uncertain",
                ],
                "additionalProperties": False,
            },
        },
        "additions_transfers": {
            "type": "array", "items": {"type": "string"},
        },
        "tare_events": {
            "type": "array", "items": {"type": "string"},
        },
        "uncertainties": {
            "type": "array", "items": {"type": "string"},
        },
        "temporal_coverage": {
            "type": "object",
            "properties": {
                "observed_duration_seconds": {
                    "type": ["number", "null"],
                },
                "first_timestamp_analyzed": {
                    "type": ["number", "null"],
                },
                "last_timestamp_analyzed": {
                    "type": ["number", "null"],
                },
                "covers_full_video": {"type": ["boolean", "null"]},
                "insufficient_evidence_intervals": {
                    "type": "array", "items": {"type": "string"},
                },
                "notes": {"type": ["string", "null"]},
            },
            "required": [
                "observed_duration_seconds",
                "first_timestamp_analyzed",
                "last_timestamp_analyzed", "covers_full_video",
                "insufficient_evidence_intervals", "notes",
            ],
            "additionalProperties": False,
        },
        "limitations": {
            "type": "array", "items": {"type": "string"},
        },
    },
    "required": [
        "overall_activity", "ingredients", "containers_utensils",
        "operations", "scale_readings", "additions_transfers",
        "tare_events", "uncertainties", "temporal_coverage",
        "limitations",
    ],
    "additionalProperties": False,
}


def build_prompt() -> str:
    """Task prompt: evidence reconstruction, never storytelling."""
    return (
        "Analyze this complete cooking video as one continuous culinary "
        "process and return a structured JSON description. Do NOT "
        "generate a polished recipe. Report: (1) video coverage — "
        "observed duration, first and last timestamps analyzed, "
        "whether the entire timeline appears covered, intervals with "
        "insufficient visual evidence, and explicit limitations; "
        "(2) culinary events — each with an event id, start and end "
        "timestamps in seconds, action type and description, "
        "ingredient candidates, containers and utensils, visible "
        "ingredient transfers, relevant scale readings, evidence "
        "description, justified confidence, uncertainties, and links "
        "to preceding and following events; (3) ingredients — visible "
        "name or label, normalized name when supported, supporting "
        "timestamps, visual evidence, ambiguities, and whether each "
        "was actually used or merely visible; (4) measurements — "
        "timestamp, displayed value, visible unit if any, evidence, "
        "whether a tare operation is observable, and whether the "
        "reading is uncertain; (5) unresolved information — unknown "
        "identities, quantities, units, unclear actions, missing "
        "evidence, and incomplete temporal coverage. Grounding rules: "
        "never invent ingredient quantities, units, actions, or "
        "timestamps; never assume grams when the unit is not visible; "
        "never infer net quantities from isolated scale readings — a "
        "displayed scale value is not a net ingredient quantity unless "
        "the evidence supports that interpretation; never claim to "
        "have observed something outside the video; never convert "
        "uncertainty into high-confidence evidence. Leave any field "
        "null (or empty) when the video does not support it."
    )


def fail(message: str) -> "NoReturn":
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def validate_video(path: Path) -> dict:
    """Check existence, MP4 format, and size without network access."""
    if not path.is_file():
        fail(f"video not found: {path}")
    if path.suffix.lower() != ".mp4":
        fail(f"expected an .mp4 file, got: {path.name}")
    size_bytes = path.stat().st_size
    if size_bytes <= 0:
        fail(f"video file is empty: {path}")
    with open(path, "rb") as handle:
        magic = handle.read(12)
    # MP4 files carry an 'ftyp' box at offset 4.
    if len(magic) < 12 or magic[4:8] != b"ftyp":
        fail(f"file does not look like MP4 (missing ftyp box): {path}")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "filename": path.name,
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
    }


def probe_duration_seconds(path: Path) -> dict:
    """Best-effort duration probe; unknown stays explicit null."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return {"duration_seconds": None,
                "duration_note": "ffprobe not available"}
    try:
        completed = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries",
             "format=duration", "-of",
             "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"duration_seconds": None,
                "duration_note": f"ffprobe failed: {exc}"}
    try:
        return {"duration_seconds": float(completed.stdout.strip()),
                "duration_note": "ffprobe format=duration"}
    except ValueError:
        return {"duration_seconds": None,
                "duration_note": "ffprobe returned no duration"}


def read_api_key() -> str:
    for name in KEY_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    fail(f"no API key in environment "
         f"({' or '.join(KEY_VARS)}). Key is required only for --execute.")


def new_run_dir(out_root: Path, stem: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_root / f"{stem}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n",
                    encoding="utf-8")


def as_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def summarize_usage(response: object) -> dict:
    """Extract token usage when the provider reports it, else nulls."""
    usage = getattr(response, "usage", None)
    dumped: dict = {}
    if usage is not None:
        if hasattr(usage, "model_dump"):
            dumped = usage.model_dump()
        elif isinstance(usage, dict):
            dumped = usage
    if not isinstance(dumped, dict):
        dumped = {}
    details_in = dumped.get("input_tokens_details") or {}
    details_out = dumped.get("output_tokens_details") or {}
    return {
        "input_tokens": as_int(dumped.get("input_tokens")),
        "output_tokens": as_int(dumped.get("output_tokens")),
        "total_tokens": as_int(dumped.get("total_tokens")),
        "cached_input_tokens": as_int(details_in.get("cached_tokens"))
        if isinstance(details_in, dict) else None,
        "reasoning_output_tokens": as_int(
            details_out.get("reasoning_tokens"))
        if isinstance(details_out, dict) else None,
    }


def evaluate_output(text: str, duration: float | None) -> dict:
    """Structural checks on the model JSON; quality stays human."""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        return {"output_parses_as_json": False,
                "parse_error": str(exc),
                "operations_identified": None,
                "timestamp_consistent": None,
                "coverage_fraction": None}
    operations = parsed.get("operations")
    if not isinstance(operations, list):
        return {"output_parses_as_json": True,
                "parse_error": None,
                "operations_identified": None,
                "timestamp_consistent": False,
                "coverage_fraction": None}
    stamps: list[tuple] = []
    consistent = True
    previous_end: float | None = None
    for item in operations:
        if not isinstance(item, dict):
            consistent = False
            continue
        start, end = item.get("start_seconds"), item.get("end_seconds")
        if not isinstance(start, (int, float)) or not isinstance(
                end, (int, float)):
            consistent = False
            continue
        start, end = float(start), float(end)
        if not (start >= 0 and end >= start):
            consistent = False
        if duration is not None and end > duration:
            consistent = False
        if previous_end is not None and start < previous_end:
            consistent = False
        previous_end = end
        stamps.append((start, end))
    coverage = None
    if duration and duration > 0 and stamps:
        coverage = max(end for _, end in stamps) / duration
    temporal = parsed.get("temporal_coverage") or {}
    return {
        "output_parses_as_json": True,
        "parse_error": None,
        "operations_identified": len(operations),
        "timestamp_consistent": consistent,
        "coverage_fraction": coverage,
        "model_claims_full_coverage": temporal.get("covers_full_video")
        if isinstance(temporal, dict) else None,
    }


def execute(video: dict, duration: dict, run_dir: Path,
            max_output_tokens: int, timeout_seconds: float,
            rates: tuple[float | None, float | None]) -> dict:
    """Upload once, infer exactly once, persist everything."""
    # Lazy import: dry-run works without the SDK installed.
    from openai import OpenAI

    api_key = read_api_key()
    client = OpenAI(base_url=BASE_URL, api_key=api_key,
                    timeout=timeout_seconds, max_retries=0)

    upload_started = time.perf_counter()
    with open(video["path"], "rb") as handle:
        uploaded = client.files.create(file=handle,
                                       purpose="user_data")
    upload_latency = time.perf_counter() - upload_started
    file_id = getattr(uploaded, "id", None)
    if not isinstance(file_id, str) or not file_id:
        fail("Files API response carried no file id.")
    (run_dir / "file_id.txt").write_text(file_id + "\n",
                                         encoding="utf-8")

    prompt = build_prompt()
    request_record = {
        "base_url": BASE_URL,
        "model": MODEL,
        "file_id": file_id,
        "video_sha256": video["sha256"],
        "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_video", "file_id": file_id},
        ],
        "max_output_tokens": max_output_tokens,
        "structured_output_schema": SCHEMA_NAME,
        "max_retries": 0,
        "timeout_seconds": timeout_seconds,
    }
    write_json(run_dir / "request.json", request_record)

    # Exactly one billable inference request; no retry loop.
    inference_started = time.perf_counter()
    response = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": request_record["content"]}],
        max_output_tokens=max_output_tokens,
        text={"format": {"type": "json_schema",
                         "name": SCHEMA_NAME,
                         "schema": ANALYSIS_JSON_SCHEMA}},
    )
    inference_latency = time.perf_counter() - inference_started

    response_record = (response.model_dump()
                       if hasattr(response, "model_dump")
                       else {"output_text": getattr(
                           response, "output_text", None)})
    write_json(run_dir / "response.json", response_record)

    text = getattr(response, "output_text", "") or ""
    usage = summarize_usage(response)
    rate_in, rate_out = rates
    estimated_cost_usd: float | None = None
    cost_note = "no pricing configured"
    if (rate_in is not None and rate_out is not None
            and usage["input_tokens"] is not None
            and usage["output_tokens"] is not None):
        estimated_cost_usd = (
            usage["input_tokens"] * rate_in
            + usage["output_tokens"] * rate_out) / 1_000_000
        cost_note = "estimate from supplied per-million-token rates"
    details = getattr(response, "incomplete_details", None)
    incomplete_reason = None
    if isinstance(details, dict):
        incomplete_reason = details.get("reason")
    elif details is not None:
        incomplete_reason = getattr(details, "reason", None)

    evaluation = {
        "video": {**video, **duration},
        "upload_latency_seconds": upload_latency,
        "inference_latency_seconds": inference_latency,
        "provider": {
            "model": getattr(response, "model", MODEL),
            "request_id": getattr(response, "id", None),
            "status": getattr(response, "status", None),
            "incomplete_reason": (str(incomplete_reason)
                                  if incomplete_reason is not None
                                  else None),
        },
        "usage": usage,
        "estimated_cost_usd": estimated_cost_usd,
        "cost_note": cost_note,
        "output_length": len(text),
        **evaluate_output(text, duration.get("duration_seconds")),
    }
    write_json(run_dir / "evaluation.json", evaluation)
    return evaluation


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolated Meta Muse Spark 1.3 native video "
                    "understanding experiment. Dry-run by default; "
                    "--execute performs one upload and one billable "
                    "inference request.")
    parser.add_argument("video", type=Path,
                        help="Local MP4 file to analyze.")
    parser.add_argument("--execute", action="store_true",
                        help="Upload the video and run one billable "
                             "inference request. Without it, only a "
                             "local dry-run plan is written.")
    parser.add_argument("--out-dir", type=Path, default=EXPERIMENT_DIR,
                        help="Experiment output root "
                             f"(default: {EXPERIMENT_DIR}).")
    parser.add_argument("--max-output-tokens", type=int,
                        default=DEFAULT_MAX_OUTPUT_TOKENS,
                        help="Output token budget for the single "
                             "inference request "
                             f"(default: {DEFAULT_MAX_OUTPUT_TOKENS}).")
    parser.add_argument("--timeout-seconds", type=float,
                        default=DEFAULT_TIMEOUT_SECONDS,
                        help="SDK timeout in seconds "
                             f"(default: {DEFAULT_TIMEOUT_SECONDS}).")
    parser.add_argument("--rate-input-per-m", type=float, default=None,
                        help="Optional input USD per 1M tokens for a "
                             "cost estimate.")
    parser.add_argument("--rate-output-per-m", type=float, default=None,
                        help="Optional output USD per 1M tokens for a "
                             "cost estimate.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    video = validate_video(args.video)
    duration = probe_duration_seconds(args.video)
    if (not isinstance(args.max_output_tokens, bool)
            and isinstance(args.max_output_tokens, int)
            and args.max_output_tokens > 0):
        pass
    else:
        fail("--max-output-tokens must be a positive int.")
    run_dir = new_run_dir(args.out_dir, args.video.stem)

    if not args.execute:
        plan = {
            "mode": "dry-run",
            "video": {**video, **duration},
            "model": MODEL,
            "base_url": BASE_URL,
            "steps_if_executed": [
                "upload video via Files API with purpose=user_data",
                "persist returned file id to file_id.txt",
                "exactly one responses.create with input_video+file_id",
                "save request.json, response.json, evaluation.json",
            ],
            "billable": False,
            "note": "Re-run with --execute to perform the upload "
                    "and the single billable inference request.",
        }
        write_json(run_dir / "plan.json", plan)
        print(f"dry-run ok: {args.video.name} "
              f"({video['size_bytes']} bytes) validated, no network used.")
        print(f"plan: {run_dir / 'plan.json'}")
        return 0

    evaluation = execute(
        video, duration, run_dir,
        args.max_output_tokens, args.timeout_seconds,
        (args.rate_input_per_m, args.rate_output_per_m))
    print(f"executed: {evaluation['operations_identified']} operations, "
          f"output {evaluation['output_length']} chars, "
          f"request {evaluation['provider']['request_id']}.")
    print(f"results: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
