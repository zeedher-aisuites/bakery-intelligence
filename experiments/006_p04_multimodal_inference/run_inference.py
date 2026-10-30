"""CLI for the P04 full-video multimodal experiment (Exp. 006).

Subcommands: preflight, calibrate, run, select-review, review, report.
Credentials come from the process environment only (META_API_KEY, with
MODEL_API_KEY fallback); this CLI never reads .env files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from meta_client import (  # noqa: E402
    MODEL_SPARK,
    MODEL_WONDER,
    ProviderError,
    list_models,
    resolve_api_key,
)
from prompts import (  # noqa: E402
    PRIMARY_PROMPT,
    REVIEW_PROMPT,
    select_review_candidates,
)
from runner import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGES_PER_REQUEST,
    RunConfig,
    Runner,
    RunnerError,
    load_manifest,
)

MANIFEST_DEFAULT = (
    Path("data/processed/experiment_003/global_1fps_rerun/manifest.csv")
)
CALIBRATION_ROWS = (0, 171, 342, 513, 684)
RUNS_ROOT = Path("data/processed/experiment_006")


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument("--model", default=MODEL_SPARK)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--images-per-request", type=int, default=DEFAULT_IMAGES_PER_REQUEST
    )
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument(
        "--token-budgets", default="1024",
        help="Comma-separated output-token escalation, e.g. 1024,2048,3072.",
    )
    parser.add_argument(
        "--store", dest="store", action="store_true", default=True,
        help="Persist responses server-side (API default).",
    )
    parser.add_argument(
        "--no-store", dest="store", action="store_false",
        help="Send store=false for oversized-payload compatibility.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--price-input", type=float, default=None)
    parser.add_argument("--price-output", type=float, default=None)
    parser.add_argument("--allow-multi-image", action="store_true")


def cmd_preflight(args: argparse.Namespace) -> int:
    """Zero-generation auth check: list models, confirm target access."""
    try:
        api_key = resolve_api_key()
        models = list_models(api_key)
    except ProviderError as exc:
        print(f"PREFLIGHT FAILED [{exc.code}]: {exc.message}")
        return 1
    print(f"Authenticated. {len(models)} model(s) visible.")
    for name in (MODEL_SPARK, MODEL_WONDER):
        print(f"  {name}: {'available' if name in models else 'NOT listed'}")
    if args.model not in models:
        print(f"PREFLIGHT FAILED: requested model {args.model} is not listed.")
        return 1
    return 0


def _parse_budgets(text: str) -> tuple[int, ...]:
    try:
        budgets = tuple(int(part) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise RunnerError(
            "INVALID_CONFIG",
            f"Unparseable --token-budgets: {text!r}.",
        ) from exc
    if not budgets or any(budget <= 0 for budget in budgets):
        raise RunnerError(
            "INVALID_CONFIG",
            f"Unparseable --token-budgets: {text!r}.",
        )
    return budgets


def _run_config_from_args(
    args: argparse.Namespace,
    run_dir: Path,
    prompt_text: str,
    pass_id: str,
    selected_rows: list[int] | None,
) -> RunConfig:
    return RunConfig(
        manifest_path=args.manifest,
        run_dir=run_dir,
        model=args.model,
        prompt_text=prompt_text,
        pass_id=pass_id,
        batch_size=args.batch_size,
        images_per_request=args.images_per_request,
        max_output_tokens=args.max_output_tokens,
        token_budgets=_parse_budgets(args.token_budgets),
        store=args.store,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
        price_input_per_m=args.price_input,
        price_output_per_m=args.price_output,
        allow_multi_image=args.allow_multi_image,
        selected_rows=selected_rows,
    )


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Small paid run (default 5 spread frames) measuring real usage."""
    rows = (
        [int(part) for part in args.rows.split(",")]
        if args.rows else list(CALIBRATION_ROWS)
    )
    run_dir = RUNS_ROOT / "calibration" / args.name
    config = _run_config_from_args(
        args, run_dir, PRIMARY_PROMPT, "calibration", rows
    )
    return _execute(config)


def cmd_run(args: argparse.Namespace) -> int:
    """Full 685-frame run; resume is automatic via checkpoints."""
    run_dir = RUNS_ROOT / "full_run" / args.name
    config = _run_config_from_args(
        args, run_dir, PRIMARY_PROMPT, "baseline_1fps", None
    )
    return _execute(config)


def cmd_select_review(args: argparse.Namespace) -> int:
    """Deterministic Wonder-review selection over primary outputs."""
    obs_dir = Path(args.primary_run_dir) / "observations"
    if not obs_dir.is_dir():
        print(f"No observations directory at {obs_dir}")
        return 1
    observations: list[dict] = []
    for path in sorted(obs_dir.glob("batch_*.json")):
        observations.extend(json.loads(path.read_text(encoding="utf-8")))
    selected = select_review_candidates(observations)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selected) + "\n", encoding="utf-8")
    print(f"Selected {len(selected)} frame(s) for review -> {out}")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    """Selective second-model pass over an explicit index list."""
    wanted = set(json.loads(Path(args.selection).read_text(encoding="utf-8")))
    rows, _ = load_manifest(args.manifest)
    selected_rows = [
        position for position, row in enumerate(rows)
        if row["frame_index"] in wanted
    ]
    if len(selected_rows) != len(wanted):
        print("Selection references frames absent from the manifest.")
        return 1
    obs_dir = Path(args.primary_run_dir) / "observations"
    if not obs_dir.is_dir():
        print(f"No observations directory at {obs_dir}")
        return 1
    primary_obs: dict[int, dict] = {}
    for path in sorted(obs_dir.glob("batch_*.json")):
        for obs in json.loads(path.read_text(encoding="utf-8")):
            primary_obs[int(obs["frame_index"])] = obs
    missing = sorted(wanted - set(primary_obs))
    if missing:
        print(f"Primary outputs missing for frames: {missing}")
        return 1
    run_dir = RUNS_ROOT / "review" / args.name
    config = _run_config_from_args(
        args, run_dir, REVIEW_PROMPT, f"review_{args.model}",
        selected_rows,
    )
    config.is_review = True
    config.primary_obs = primary_obs
    return _execute(config)


def cmd_report(args: argparse.Namespace) -> int:
    """Print the aggregate report of a finished or partial run."""
    report_path = Path(args.run_dir) / "reports" / "run_report.json"
    if not report_path.is_file():
        print(f"No report yet at {report_path}")
        return 1
    print(report_path.read_text(encoding="utf-8"))
    return 0


def _execute(config: RunConfig) -> int:
    try:
        state = Runner(config).run()
    except (RunnerError, ProviderError) as exc:
        print(f"RUN ABORTED [{exc.code}]: {exc.message}")
        return 1
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--model", default=MODEL_SPARK)
    preflight.set_defaults(func=cmd_preflight)
    calibrate = sub.add_parser("calibrate")
    _add_common(calibrate)
    calibrate.add_argument("--name", default="default")
    calibrate.add_argument("--rows", default="")
    calibrate.set_defaults(func=cmd_calibrate)
    run = sub.add_parser("run")
    _add_common(run)
    run.add_argument("--name", default="default")
    run.set_defaults(func=cmd_run)
    select = sub.add_parser("select-review")
    select.add_argument("--primary-run-dir", required=True)
    select.add_argument("--output", required=True)
    select.set_defaults(func=cmd_select_review)
    review = sub.add_parser("review")
    _add_common(review)
    review.add_argument("--name", default="default")
    review.add_argument("--selection", required=True)
    review.add_argument("--primary-run-dir", required=True)
    review.set_defaults(func=cmd_review)
    report = sub.add_parser("report")
    report.add_argument("--run-dir", required=True)
    report.set_defaults(func=cmd_report)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
