"""FirstPass engine benchmark harness (experiment branch).

Compares single_frame vs contextual_batch wall-clock time,
request counts, tokens, and evidence coverage on the same
video input. Dry-run by default; billable execution
requires --execute plus explicit bounds. Only the
FirstPass stage is billable here (extraction and
reconstruction are offline and deterministic).

Never prints credentials.
"""

from __future__ import annotations

import argparse
import json
import math
import secrets
import sys
import time
from pathlib import Path

from bakery_intelligence.api.authorization import resolve_api_key
from bakery_intelligence.api.pipeline import (
    FIRSTPASS_BATCH_GROUP_SIZE,
    FIRSTPASS_ENGINE_BATCH,
    FIRSTPASS_ENGINE_SINGLE,
    _validate_firstpass_engine,
    run_pipeline_from_path,
)
from bakery_intelligence.inference.meta_spark import (
    MODEL_SPARK,
    MetaSparkClient,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark one FirstPass engine.")
    parser.add_argument("--video", type=Path, required=True,
                        help="Source video file.")
    parser.add_argument(
        "--engine",
        default=FIRSTPASS_ENGINE_SINGLE,
        choices=[FIRSTPASS_ENGINE_SINGLE,
                 FIRSTPASS_ENGINE_BATCH],
        help="FirstPass engine under test.")
    parser.add_argument("--limit-frames", type=int,
                        default=None,
                        help="Cap extracted frames "
                             "(benchmark slice).")
    parser.add_argument("--execute", action="store_true",
                        help="Authorize billable execution. "
                             "Without it, only a plan prints.")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Run directory (default: "
                             "runs_api/benchmarks/<run_id>/).")
    parser.add_argument("--token-budget", type=int,
                        default=None,
                        help="Per-request output budget "
                             "(default 1024 single, "
                             "16384 batch).")
    parser.add_argument("--max-requests", type=int,
                        default=None,
                        help="Request cap (default: frames "
                             "for single, groups+1 for batch).")
    parser.add_argument("--rates-usd-per-m", type=float,
                        nargs=2, default=[1.25, 4.25],
                        metavar=("INPUT", "OUTPUT"))
    parser.add_argument("--cost-ceiling-usd", type=float,
                        default=2.0)
    parser.add_argument("--model", default=MODEL_SPARK)
    parser.add_argument("--timeout-seconds", type=float,
                        default=1200.0)
    return parser


def plan(args: argparse.Namespace) -> dict:
    """Resolve effective bounds without side effects."""
    _validate_firstpass_engine(args.engine)
    batch = args.engine == FIRSTPASS_ENGINE_BATCH
    token_budget = args.token_budget
    if token_budget is None:
        token_budget = 16384 if batch else 1024
    frames = args.limit_frames
    max_requests = args.max_requests
    if max_requests is None and frames is not None:
        max_requests = (
            math.ceil(frames / FIRSTPASS_BATCH_GROUP_SIZE)
            + 1 if batch else frames)
    run_id = "run_" + secrets.token_hex(6)
    work_dir = args.work_dir or (
        Path("runs_api") / "benchmarks" / run_id)
    return {
        "run_id": run_id,
        "engine": args.engine,
        "video": str(args.video),
        "limit_frames": frames,
        "work_dir": str(work_dir),
        "token_budget": token_budget,
        "max_requests": max_requests,
        "rates_usd_per_m": list(args.rates_usd_per_m),
        "cost_ceiling_usd": args.cost_ceiling_usd,
        "model": args.model,
    }


def collect_metrics(work_dir: Path, run_id: str,
                    wall_seconds: float) -> dict:
    """Read report + observations; never fabricate."""
    firstpass = work_dir / "firstpass"
    report = json.loads(
        (firstpass / "reports" / "run_report.json"
         ).read_text(encoding="utf-8"))
    observations: list = []
    for path in sorted(
            (firstpass / "observations").glob("*.json")):
        observations.extend(json.loads(
            path.read_text(encoding="utf-8")))
    stamps = [item.get("timestamp_seconds")
              for item in observations]
    return {
        "run_id": run_id,
        "engine": report.get("pass_id"),
        "wall_elapsed_seconds": round(wall_seconds, 1),
        "frames_total": report.get("frames_total"),
        "frames_observed": report.get("frames_observed"),
        "frames_failed": report.get("frames_failed"),
        "requests_sent": report.get("requests_sent"),
        "groups_completed": report.get("groups_completed"),
        "groups_valid": report.get("groups_valid"),
        "groups_review": report.get("groups_review"),
        "input_tokens": report.get("input_tokens"),
        "output_tokens": report.get("output_tokens"),
        "cost_usd_estimated": report.get(
            "cost_usd_estimated"),
        "terminal_states": report.get("terminal_states"),
        "observation_files": len(list(
            (firstpass / "observations").glob("*.json"))),
        "timestamps_monotonic": (
            stamps == sorted(stamps) if stamps else None),
        "first_timestamp": stamps[0] if stamps else None,
        "last_timestamp": stamps[-1] if stamps else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.video.is_file():
        print(f"error: video not found: {args.video}",
              file=sys.stderr)
        return 2
    if args.limit_frames is not None and (
            args.limit_frames <= 0):
        print("error: --limit-frames must be positive.",
              file=sys.stderr)
        return 2
    config = plan(args)
    print(json.dumps(config, indent=2))
    if not args.execute:
        print("dry-run: add --execute to run billable "
              "inference.")
        return 0
    api_key = resolve_api_key()
    client = MetaSparkClient(
        model=args.model,
        timeout_seconds=args.timeout_seconds, store=False)
    work_dir = Path(config["work_dir"])
    start = time.perf_counter()
    run_pipeline_from_path(
        video_path=args.video, video_name=args.video.name,
        run_id=config["run_id"],
        source_object_id=args.video.name,
        work_dir=work_dir,
        billable_client=client,
        billable_token_budget=config["token_budget"],
        billable_max_requests=config["max_requests"],
        billable_cost_ceiling_usd=args.cost_ceiling_usd,
        billable_rates_usd_per_m=tuple(
            args.rates_usd_per_m),
        billable_strict_cost=False,
        billable_api_key=api_key,
        billable_live=True,
        max_frames=args.limit_frames
        if args.limit_frames is not None else 1000,
        firstpass_engine=args.engine)
    wall = time.perf_counter() - start
    metrics = collect_metrics(
        work_dir, config["run_id"], wall)
    print(json.dumps(metrics, indent=2))
    (work_dir / "benchmark_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
