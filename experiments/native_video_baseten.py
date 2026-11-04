"""Isolated native video understanding experiment (Baseten Muse Spark Open).

Attempts one native-video Chat Completions request against the
authorized Baseten deployment ``meta-models/muse-spark-open-1026``
(the repository API key is restricted to this exact id; never
substitute ``meta-models/muse-spark-open``).

Compatibility status (verified before writing this script):
  * Model family: Baseten's Muse Spark Open library page lists
    Text/Image/Video/Audio input modalities.
  * Deployment: this dated ``-1026`` deployment has only text and
    single-image (``image_url``) usage verified in this repository
    (experiments/004_baseten_api, ``baseten_challenger.py``).
    A ``video_url`` content part on THIS deployment is UNVERIFIED.
  * No Files API exists on the Baseten inference route, so the
    video is sent inline as a base64 data URL in a ``video_url``
    content part (the OpenAI-conformant shape mirroring the
    verified ``image_url`` usage). There is no Meta-style
    ``file_id`` reference across providers.
  * Structured output via ``response_format`` is unverified on
    this deployment; like the production challenger, this script
    prompts for JSON and parses it, recording the outcome.

If the provider rejects the video part, the exact rejection is
persisted as the compatibility verdict. No frame-based fallback
is implemented or described as native video understanding.

Reuse: MP4 validation, duration probing, culinary prompt, and
structural evaluation are imported from the Spark 1.3 experiment
so all runs stay comparable. The Spark 1.3 script is not modified;
outputs land under ``experiments/native_video_baseten/``.

Safety posture: dry-run default, ``--execute`` required for the
single billable request, SDK retries disabled (``max_retries=0``),
``BASETEN_API_KEY`` (and optional ``BAKERY_BASETEN_MODEL``, which
must equal the authorized id) never printed, logged, persisted,
or embedded in saved artifacts. The inline video bytes are never
written to disk; ``request.json`` records only part types and
byte sizes.
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import native_video_understanding as base

BASE_URL = "https://inference.baseten.co/v1"
AUTHORIZED_MODEL = "meta-models/muse-spark-open-1026"
KEY_VAR = "BASETEN_API_KEY"
MODEL_VAR = "BAKERY_BASETEN_MODEL"
EXPERIMENT_DIR = Path(__file__).resolve().parent / "native_video_baseten"
DEFAULT_MAX_TOKENS = 16384
DEFAULT_TIMEOUT_SECONDS = 1800.0
# Baseten Model API family rate for Muse Spark Open (USD per 1M
# tokens); the dated -1026 deployment may differ, so estimates
# are labeled accordingly.
DEFAULT_RATE_INPUT_PER_M = 1.25
DEFAULT_RATE_OUTPUT_PER_M = 4.25


def read_config() -> tuple[str, str]:
    """Return (api key, model id), enforcing the authorized id."""
    import os
    api_key = os.environ.get(KEY_VAR, "").strip()
    if not api_key:
        base.fail(f"no API key in environment ({KEY_VAR}). "
                  f"Key is required only for --execute.")
    model = os.environ.get(MODEL_VAR, AUTHORIZED_MODEL).strip()
    if model != AUTHORIZED_MODEL:
        base.fail(f"{MODEL_VAR} must be exactly {AUTHORIZED_MODEL}.")
    return api_key, model


def execute(video: dict, duration: dict, run_dir: Path,
            max_tokens: int, timeout_seconds: float,
            rates: tuple[float | None, float | None]) -> dict:
    """Send exactly one native-video Chat Completions request."""
    # Lazy import: dry-run works without the SDK installed.
    from openai import OpenAI

    api_key, model = read_config()
    client = OpenAI(base_url=BASE_URL, api_key=api_key,
                    timeout=timeout_seconds, max_retries=0)

    raw = Path(video["path"]).read_bytes()
    data_url = ("data:video/mp4;base64,"
                + base64.b64encode(raw).decode("ascii"))
    del raw

    prompt = base.build_prompt()
    content = [
        {"type": "text", "text": prompt},
        {"type": "video_url", "video_url": {"url": data_url}},
    ]
    # The payload bytes are never persisted; only their size is.
    request_record = {
        "base_url": BASE_URL,
        "model": model,
        "video_sha256": video["sha256"],
        "video_size_bytes": video["size_bytes"],
        "content_parts": [
            {"type": "text", "length": len(prompt)},
            {"type": "video_url", "url_length": len(data_url),
             "note": "base64 data URL bytes excluded from artifacts"},
        ],
        "max_tokens": max_tokens,
        "response_format": None,
        "response_format_note": "json_schema unverified on this "
                               "deployment; plain JSON prompt used",
        "max_retries": 0,
        "timeout_seconds": timeout_seconds,
    }
    base.write_json(run_dir / "request.json", request_record)
    del data_url

    # Exactly one billable inference request; no retry loop.
    inference_started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
        )
    except Exception as exc:  # Recorded, never retried.
        latency = time.perf_counter() - inference_started
        status = getattr(exc, "status_code", None)
        evaluation = {
            "video": {**video, **duration},
            "upload_latency_seconds": 0.0,
            "upload_note": "no upload step; inline data URL",
            "inference_latency_seconds": latency,
            "provider": {
                "model": model,
                "request_id": None,
                "status": None,
                "finish_reason": None,
                "error_type": type(exc).__name__,
                "error_status": status,
                "error_detail": str(exc)[:1000],
            },
            "usage": {"input_tokens": None, "output_tokens": None,
                      "total_tokens": None,
                      "usage_note": "request failed before usage"},
            "estimated_cost_usd": None,
            "cost_note": "request failed; no usage reported",
            "output_length": 0,
            "native_video_verdict": (
                "REJECTED: provider refused the video_url request; "
                "see provider.error_detail"),
            **base.evaluate_output("", duration.get("duration_seconds")),
        }
        # evaluate_output("") reports parse failure; keep it honest.
        base.write_json(run_dir / "evaluation.json", evaluation)
        print(f"provider refused the request: "
              f"{type(exc).__name__} (status {status}).")
        print(f"results: {run_dir}")
        raise SystemExit(3)
    inference_latency = time.perf_counter() - inference_started

    response_record = (response.model_dump()
                       if hasattr(response, "model_dump")
                       else {"id": getattr(response, "id", None)})
    base.write_json(run_dir / "response.json", response_record)

    choice = response.choices[0]
    text = getattr(choice.message, "content", None) or ""
    raw_usage = getattr(response, "usage", None)
    dumped = (raw_usage.model_dump() if raw_usage is not None
              and hasattr(raw_usage, "model_dump") else {})
    if not isinstance(dumped, dict):
        dumped = {}
    usage = {
        "input_tokens": base.as_int(dumped.get("prompt_tokens")),
        "output_tokens": base.as_int(dumped.get("completion_tokens")),
        "total_tokens": base.as_int(dumped.get("total_tokens")),
        "usage_note": "chat completions usage mapped to "
                      "input/output tokens",
    }
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
                     "(Baseten Muse Spark Open family rate; "
                     "-1026 deployment may differ)")
    finish_reason = getattr(choice, "finish_reason", None)

    struct = base.evaluate_output(text, duration.get("duration_seconds"))
    evaluation = {
        "video": {**video, **duration},
        "upload_latency_seconds": 0.0,
        "upload_note": "no upload step; inline data URL",
        "inference_latency_seconds": inference_latency,
        "provider": {
            "model": getattr(response, "model", model),
            "request_id": getattr(response, "id", None),
            "status": None,
            "finish_reason": finish_reason,
            "incomplete_reason": None,
        },
        "usage": usage,
        "estimated_cost_usd": estimated_cost_usd,
        "cost_note": cost_note,
        "output_length": len(text),
        "native_video_verdict": (
            "ACCEPTED" if text and struct["output_parses_as_json"]
            else "UNCLEAR: provider responded but output is empty "
                 "or not valid JSON"),
        **struct,
    }
    base.write_json(run_dir / "evaluation.json", evaluation)
    return evaluation


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolated Baseten Muse Spark Open native video "
                    "experiment. Dry-run by default; --execute sends "
                    "exactly one billable Chat Completions request "
                    "with an inline video_url part.")
    parser.add_argument("video", type=Path,
                        help="Local MP4 file to analyze.")
    parser.add_argument("--execute", action="store_true",
                        help="Send one billable inference request. "
                             "Without it, only a local dry-run plan "
                             "is written.")
    parser.add_argument("--out-dir", type=Path, default=EXPERIMENT_DIR,
                        help="Experiment output root "
                             f"(default: {EXPERIMENT_DIR}).")
    parser.add_argument("--max-tokens", type=int,
                        default=DEFAULT_MAX_TOKENS,
                        help="Completion token budget for the single "
                             "inference request "
                             f"(default: {DEFAULT_MAX_TOKENS}).")
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
    if (not isinstance(args.max_tokens, bool)
            and isinstance(args.max_tokens, int)
            and args.max_tokens > 0):
        pass
    else:
        base.fail("--max-tokens must be a positive int.")
    run_dir = base.new_run_dir(args.out_dir, args.video.stem)

    if not args.execute:
        plan = {
            "mode": "dry-run",
            "video": {**video, **duration},
            "model": AUTHORIZED_MODEL,
            "base_url": BASE_URL,
            "compatibility": {
                "family_video_support": "yes, per Baseten Muse "
                "Spark Open library page (Text/Image/Video/Audio)",
                "deployment_video_url_support": "UNVERIFIED on "
                "meta-models/muse-spark-open-1026; no files API "
                "on this route; determined by provider response "
                "at execute time",
                "structured_output": "response_format unverified; "
                "plain JSON prompt with parse check",
            },
            "steps_if_executed": [
                "build one chat.completions request with an inline "
                "video_url base64 data URL (no upload step)",
                "save request.json WITHOUT payload bytes, "
                "response.json, evaluation.json",
            ],
            "size_warning": (
                f"{video['size_bytes']} bytes inline as base64 "
                f"(~{video['size_bytes'] * 4 // 3} chars); large "
                f"payloads may be rejected: rejection is the "
                f"recorded verdict"),
            "billable": False,
            "note": "Re-run with --execute to send the single "
                    "billable inference request.",
        }
        base.write_json(run_dir / "plan.json", plan)
        print(f"dry-run ok: {args.video.name} "
              f"({video['size_bytes']} bytes) validated, no network used.")
        print(f"compatibility: deployment video_url support UNVERIFIED.")
        print(f"plan: {run_dir / 'plan.json'}")
        return 0

    evaluation = execute(
        video, duration, run_dir,
        args.max_tokens, args.timeout_seconds,
        (args.rate_input_per_m, args.rate_output_per_m))
    print(f"executed: {evaluation['operations_identified']} operations, "
          f"output {evaluation['output_length']} chars, "
          f"request {evaluation['provider']['request_id']}.")
    print(f"verdict: {evaluation['native_video_verdict']}")
    print(f"results: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
