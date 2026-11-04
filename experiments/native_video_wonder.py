"""Isolated native video understanding experiment (Meta Muse Wonder 1.0).

Same protocol as ``experiments/native_video_understanding.py``
(Muse Spark 1.3): the whole MP4 is uploaded through the Meta Files
API (purpose="user_data") and analyzed with a single Responses API
request using an ``input_video`` block that references the uploaded
``file_id``. No frames are extracted.

Model identity ``muse-wonder-1.0`` and video input support are
verified against the official Meta Model API documentation
(https://dev.meta.ai/docs/models): Wonder takes text, image, video,
audio, and PDF input with a 1,048,576-token context window, and
``muse-wonder-1.0`` is available on the Model API in Standard mode.
Documented Standard pricing: USD 4.00 per 1M input tokens,
USD 20.00 per 1M output tokens.

Reuse: prompt, JSON schema, MP4 validation, duration probing, and
structural evaluation are imported from the Spark 1.3 experiment so
both runs are directly comparable. This file defines only the
Wonder-specific request execution. The Spark 1.3 script is not
modified; outputs land under ``experiments/native_video_wonder/``.

Safety posture: dry-run default, ``--execute`` required for upload
and the single billable request, SDK retries disabled
(``max_retries=0``), API key from ``MODEL_API_KEY`` (fallback
``META_API_KEY``) never printed, logged, or persisted.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import native_video_understanding as base

MODEL = "muse-wonder-1.0"
EXPERIMENT_DIR = Path(__file__).resolve().parent / "native_video_wonder"
SCHEMA_NAME = "NativeVideoCookingAnalysis"
DEFAULT_MAX_OUTPUT_TOKENS = 16384
DEFAULT_TIMEOUT_SECONDS = 1800.0
# Documented Standard-mode Muse Wonder pricing (USD per 1M tokens).
DEFAULT_RATE_INPUT_PER_M = 4.00
DEFAULT_RATE_OUTPUT_PER_M = 20.00


def execute(video: dict, duration: dict, run_dir: Path,
            file_id: str | None,
            max_output_tokens: int, timeout_seconds: float,
            rates: tuple[float | None, float | None]) -> dict:
    """Upload once (unless a file_id is reused), infer exactly once."""
    # Lazy import: dry-run works without the SDK installed.
    from openai import OpenAI

    api_key = base.read_api_key()
    client = OpenAI(base_url=base.BASE_URL, api_key=api_key,
                    timeout=timeout_seconds, max_retries=0)

    if file_id is None:
        upload_started = time.perf_counter()
        with open(video["path"], "rb") as handle:
            uploaded = client.files.create(file=handle,
                                           purpose="user_data")
        upload_latency = time.perf_counter() - upload_started
        file_id = getattr(uploaded, "id", None)
        if not isinstance(file_id, str) or not file_id:
            base.fail("Files API response carried no file id.")
        upload_note = "fresh upload"
    else:
        upload_latency = 0.0
        upload_note = "reused existing file_id"
    (run_dir / "file_id.txt").write_text(file_id + "\n",
                                         encoding="utf-8")

    prompt = base.build_prompt()
    request_record = {
        "base_url": base.BASE_URL,
        "model": MODEL,
        "file_id": file_id,
        "file_id_source": upload_note,
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
    base.write_json(run_dir / "request.json", request_record)

    # Exactly one billable inference request; no retry loop.
    inference_started = time.perf_counter()
    response = client.responses.create(
        model=MODEL,
        input=[{"role": "user", "content": request_record["content"]}],
        max_output_tokens=max_output_tokens,
        text={"format": {"type": "json_schema",
                         "name": SCHEMA_NAME,
                         "schema": base.ANALYSIS_JSON_SCHEMA}},
    )
    inference_latency = time.perf_counter() - inference_started

    response_record = (response.model_dump()
                       if hasattr(response, "model_dump")
                       else {"output_text": getattr(
                           response, "output_text", None)})
    base.write_json(run_dir / "response.json", response_record)

    text = getattr(response, "output_text", "") or ""
    usage = base.summarize_usage(response)
    rate_in, rate_out = rates
    estimated_cost_usd: float | None = None
    cost_note = "no pricing configured"
    if (rate_in is not None and rate_out is not None
            and usage["input_tokens"] is not None
            and usage["output_tokens"] is not None):
        estimated_cost_usd = (
            usage["input_tokens"] * rate_in
            + usage["output_tokens"] * rate_out) / 1_000_000
        cost_note = ("estimate from supplied per-million-token rates "
                     "(Muse Wonder Standard)")
    details = getattr(response, "incomplete_details", None)
    incomplete_reason = None
    if isinstance(details, dict):
        incomplete_reason = details.get("reason")
    elif details is not None:
        incomplete_reason = getattr(details, "reason", None)

    evaluation = {
        "video": {**video, **duration},
        "upload_latency_seconds": upload_latency,
        "upload_note": upload_note,
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
        **base.evaluate_output(text, duration.get("duration_seconds")),
    }
    base.write_json(run_dir / "evaluation.json", evaluation)
    return evaluation


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolated Meta Muse Wonder 1.0 native video "
                    "understanding experiment. Dry-run by default; "
                    "--execute performs one upload and one billable "
                    "inference request.")
    parser.add_argument("video", type=Path,
                        help="Local MP4 file to analyze.")
    parser.add_argument("--execute", action="store_true",
                        help="Upload the video and run one billable "
                             "inference request. Without it, only a "
                             "local dry-run plan is written.")
    parser.add_argument("--file-id", default=None,
                        help="Reuse an existing Meta file_id instead of "
                             "uploading the video again.")
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
    parser.add_argument("--rate-input-per-m", type=float,
                        default=DEFAULT_RATE_INPUT_PER_M,
                        help="Input USD per 1M tokens for a cost "
                             "estimate.")
    parser.add_argument("--rate-output-per-m", type=float,
                        default=DEFAULT_RATE_OUTPUT_PER_M,
                        help="Output USD per 1M tokens for a cost "
                             "estimate.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    video = base.validate_video(args.video)
    duration = base.probe_duration_seconds(args.video)
    if (not isinstance(args.max_output_tokens, bool)
            and isinstance(args.max_output_tokens, int)
            and args.max_output_tokens > 0):
        pass
    else:
        base.fail("--max-output-tokens must be a positive int.")
    if args.file_id is not None and not args.file_id.strip():
        base.fail("--file-id must be a non-empty file id.")
    run_dir = base.new_run_dir(args.out_dir, args.video.stem)

    if not args.execute:
        plan = {
            "mode": "dry-run",
            "video": {**video, **duration},
            "model": MODEL,
            "base_url": base.BASE_URL,
            "file_id_source": ("reuse " + args.file_id
                               if args.file_id else "fresh upload"),
            "steps_if_executed": [
                "upload video via Files API with purpose=user_data "
                "(skipped when --file-id is given)",
                "persist returned file id to file_id.txt",
                "exactly one responses.create with input_video+file_id",
                "save request.json, response.json, evaluation.json",
            ],
            "billable": False,
            "note": "Re-run with --execute to perform the upload "
                    "and the single billable inference request.",
        }
        base.write_json(run_dir / "plan.json", plan)
        print(f"dry-run ok: {args.video.name} "
              f"({video['size_bytes']} bytes) validated, no network used.")
        print(f"plan: {run_dir / 'plan.json'}")
        return 0

    evaluation = execute(
        video, duration, run_dir,
        args.file_id.strip() if args.file_id else None,
        args.max_output_tokens, args.timeout_seconds,
        (args.rate_input_per_m, args.rate_output_per_m))
    print(f"executed: {evaluation['operations_identified']} operations, "
          f"output {evaluation['output_length']} chars, "
          f"request {evaluation['provider']['request_id']}.")
    print(f"results: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
