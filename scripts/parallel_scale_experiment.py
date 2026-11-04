"""Parallel scale-reading experiment: 5 x 10 anchored images.

Isolated MVP decision experiment (not pipeline code). Sends
exactly five concurrent Muse Spark 1.3 requests, each with
ten labeled scale photos in ONE multimodal call, and scores
verbatim display readings against the verified reference.

Default is dry-run (offline fake client). Live execution
requires --execute plus configured credentials, enforces a
worst-case pre-dispatch cost reservation against a USD 0.50
ceiling, and never retries.

Usage:
  uv run python scripts/parallel_scale_experiment.py --dry-run
  uv run python C:/tmp/mm_with_env.py \
      scripts/parallel_scale_experiment.py --execute
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bakery_intelligence.api.authorization import resolve_api_key
from bakery_intelligence.inference.firstpass import (
    InferenceError,
    InferenceResponse,
    InferenceUsage,
    estimate_cost,
)
from bakery_intelligence.inference.firstpass_eval import (
    Prediction,
    ReferenceRow,
    score_offsets,
)
from bakery_intelligence.inference.firstpass_v3 import (
    visual_frame_id_for,
)
from bakery_intelligence.inference.meta_spark import MetaSparkClient

REF_PATH = Path(
    "runs_api/firstpass_v3_validation/reference_testA.json")
ANCHOR_DIR = Path(
    "runs_api/firstpass_v3_validation/run2A/anchored")
OUT_BASE = Path("runs_api/scale_parallel")
MODEL = "muse-spark-1.3"
BATCH_SIZE = 10
CONCURRENCY = 5
MAX_OUTPUT_TOKENS = 8192
RATES_USD_PER_M = (1.25, 4.25)
CEILING_USD = 0.50
TIMEOUT_SECONDS = 900.0
# Conservative input bound: measured ~650 tokens per
# anchored PNG at 25 images; 10 images + prompt < 9000.
INPUT_TOKEN_BOUND = 9000
SCHEMA_NAME = "ScaleBatch10"

SCALE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "frames": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "frame_id": {"type": "string"},
                    "display_value": {
                        "type": ["string", "null"]},
                    "unit": {"type": ["string", "null"]},
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": ["frame_id", "display_value",
                             "unit", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["frames"],
    "additionalProperties": False,
}
# NOTE: minItems/maxItems deliberately omitted. Provider
# support for length keywords is unverified, and a 400
# rejection would waste one of five requests. Exact
# ten-observation coverage is enforced independently below.


def build_prompt(entries: list[tuple[str, float]]) -> str:
    """Assemble the minimal scale-only batch prompt."""
    lines = [
        "Read the digital scale display in each attached photo.",
        "Each photo has a black identifier band at the top "
        "showing FRAME-ID and CODE. Read the FRAME-ID from "
        "the band and return it as frame_id.",
        "Reply with exactly one JSON object and no other text:",
        '{"frames": [{"frame_id": "F000477", '
        '"display_value": "125.5", "unit": "g", '
        '"confidence": "high"}]}',
        f"RULES: one entry per image, {len(entries)} entries "
        "total, in the table order below.",
        "display_value is the exact visible display text: "
        "preserve decimals, signs, and leading zeros.",
        "Return null when the display is blank or unreadable; "
        "never invent a reading.",
        "Return null for unit unless visibly identifiable.",
        "Never infer tare, ingredients, or net mass.",
        "The band digits are identifiers, never readings.",
        "",
        f"BATCH: {len(entries)} image(s), in this exact order:",
    ]
    for position, (frame_id, stamp) in enumerate(
            entries, start=1):
        lines.append(
            f"IMAGE {position}: frame_id={frame_id} "
            f"(printed on the image), "
            f"timestamp_seconds={stamp}")
    return "\n".join(lines)


@dataclass
class Validation:
    """Independent response validation outcome."""

    ok: bool
    observations: dict[str, dict[str, Any]] | None = None
    error: str | None = None


def validate_response(text: str,
                      expected_ids: list[str]) -> Validation:
    """Require exactly the ten expected frame observations."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return Validation(ok=False,
                          error=f"unparseable_json:{exc}")
    if not isinstance(parsed, dict):
        return Validation(ok=False,
                          error="top_level_not_an_object")
    entries = parsed.get("frames")
    if not isinstance(entries, list):
        return Validation(ok=False, error="frames_not_a_list")
    if len(entries) != len(expected_ids):
        return Validation(
            ok=False,
            error=(f"frame_count_mismatch: got {len(entries)}, "
                   f"expected {len(expected_ids)}"))
    want = set(expected_ids)
    seen: dict[str, dict[str, Any]] = {}
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            return Validation(
                ok=False,
                error=f"entry_not_an_object:position_{position}")
        frame_id = entry.get("frame_id")
        if frame_id not in want:
            return Validation(
                ok=False,
                error=(f"unexpected_frame_id:position_"
                       f"{position}_got_{frame_id!r}"))
        if frame_id in seen:
            return Validation(
                ok=False,
                error=f"duplicate_frame_id:{frame_id}")
        value = entry.get("display_value")
        unit = entry.get("unit")
        confidence = entry.get("confidence")
        if not (value is None or isinstance(value, str)):
            return Validation(
                ok=False,
                error=f"bad_display_value:{frame_id}")
        if not (unit is None or isinstance(unit, str)):
            return Validation(
                ok=False, error=f"bad_unit:{frame_id}")
        if confidence not in ("high", "medium", "low"):
            return Validation(
                ok=False,
                error=f"bad_confidence:{frame_id}")
        seen[frame_id] = {"display_value": value,
                          "unit": unit,
                          "confidence": confidence}
    return Validation(ok=True, observations=seen)


def reserve_worst_case() -> float:
    """Bound the cost of all five requests before dispatch."""
    per_request = ((INPUT_TOKEN_BOUND / 1_000_000)
                   * RATES_USD_PER_M[0]
                   + (MAX_OUTPUT_TOKENS / 1_000_000)
                   * RATES_USD_PER_M[1])
    return round(per_request * CONCURRENCY, 4)


class FakeScaleClient:
    """Deterministic offline double (no network).

    Batch 4 returns malformed JSON to exercise rejection.
    """

    def __init__(self, plan: dict[int, str]) -> None:
        self._plan = plan

    def submit(self, prompt_text: str, image_paths: Any,
               api_key: str, max_output_tokens: int,
               response_schema: Any = None,
               response_schema_name: Any = None,
               *, batch_index: int = 0) -> InferenceResponse:
        _ = (prompt_text, image_paths, api_key,
             max_output_tokens, response_schema,
             response_schema_name)
        return InferenceResponse(
            model="fake-scale-v1", status="completed",
            output_text=self._plan[batch_index],
            usage=InferenceUsage(
                input_tokens=7000, output_tokens=400,
                total_tokens=7400),
            latency_seconds=0.5,
            request_id=f"fake-{batch_index}")


def _fake_payload(ids: list[str],
                  values: dict[str, str | None]) -> str:
    return json.dumps({"frames": [
        {"frame_id": frame_id,
         "display_value": values.get(frame_id),
         "unit": None, "confidence": "high"}
        for frame_id in ids]})


def run_batch(batch_index: int, frame_ids: list[str],
              image_paths: list[Path],
              timestamps: list[float], api_key: str,
              out_dir: Path,
              client_factory: Any) -> dict[str, Any]:
    """Dispatch one request; never retry. Thread-safe."""
    prompt = build_prompt(list(zip(frame_ids, timestamps)))
    started = time.time()
    record: dict[str, Any] = {
        "batch_index": batch_index,
        "frame_ids": frame_ids,
        "images": [path.name for path in image_paths],
        "model": MODEL,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "started_at": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
    }
    try:
        client = client_factory()
        if isinstance(client, FakeScaleClient):
            result = client.submit(
                prompt, image_paths, api_key,
                MAX_OUTPUT_TOKENS, SCALE_SCHEMA, SCHEMA_NAME,
                batch_index=batch_index)
        else:
            result = client.submit(
                prompt, image_paths, api_key,
                MAX_OUTPUT_TOKENS, SCALE_SCHEMA, SCHEMA_NAME)
    except InferenceError as exc:
        record.update({
            "status": "transport_failed",
            "error_code": exc.code,
            "error": str(exc),
            "retried": False,
        })
    except Exception as exc:  # Never retry; record only.
        record.update({
            "status": "transport_failed",
            "error_code": type(exc).__name__,
            "error": str(exc)[:300],
            "retried": False,
        })
    else:
        validation = validate_response(result.output_text,
                                       frame_ids)
        record.update({
            "status": ("completed"
                       if validation.ok else "invalid"),
            "request_id": result.request_id,
            "response_status": result.status,
            "latency_seconds": result.latency_seconds,
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "reasoning_output_tokens":
                result.usage.reasoning_output_tokens,
            "cost_usd_estimated": estimate_cost(
                result.usage.input_tokens,
                result.usage.output_tokens,
                RATES_USD_PER_M),
            "validation_error": validation.error,
            "observations": validation.observations,
            "output_chars": len(result.output_text or ""),
            "output_text": result.output_text,
            "retried": False,
        })
    finally:
        record["ended_at"] = datetime.datetime.now(
            datetime.timezone.utc).isoformat()
        record["wall_seconds"] = round(
            time.time() - started, 3)
    path = out_dir / f"batch_{batch_index}.json"
    path.write_text(json.dumps(record, indent=2),
                    encoding="utf-8")
    return record


def evaluate(out_dir: Path,
             rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score reassembled observations against the reference."""
    by_id: dict[str, dict[str, Any]] = {}
    batches = []
    for batch_index in range(CONCURRENCY):
        record = json.loads((out_dir / f"batch_{batch_index}"
                             ".json").read_text(
                                 encoding="utf-8"))
        batches.append(record)
        if record.get("observations"):
            by_id.update(record["observations"])
    references = [ReferenceRow(
        frame_index=row["frame_index"],
        status=row["status"], reading=row["reading"])
        for row in rows]
    predictions = []
    missing_ids = []
    for row in rows:
        frame_id = visual_frame_id_for(row["frame_index"])
        obs = by_id.get(frame_id)
        if obs is None:
            missing_ids.append(frame_id)
            predictions.append(Prediction(None, "low"))
        else:
            predictions.append(Prediction(
                obs["display_value"], obs["confidence"]))
    scores = score_offsets(predictions, references)
    offsets = {str(offset): {
        "compared": score.compared, "exact": score.exact,
        "missing": score.missing, "wrong": score.wrong,
        "unreadable_with_value":
            score.unreadable_with_value,
        "false_confident": score.false_confident,
    } for offset, score in sorted(scores.items())}
    total_in = sum(
        (batch.get("input_tokens") or 0) for batch in batches)
    total_out = sum(
        (batch.get("output_tokens") or 0) for batch in batches)
    total_reason = sum(
        (batch.get("reasoning_output_tokens") or 0)
        for batch in batches)
    total_cost = round(sum(
        (batch.get("cost_usd_estimated") or 0.0)
        for batch in batches), 4)
    latencies = [batch.get("wall_seconds") for batch in
                 batches]
    report = {
        "batches": [{
            "batch_index": batch["batch_index"],
            "status": batch["status"],
            "latency_seconds": batch.get("latency_seconds"),
            "wall_seconds": batch.get("wall_seconds"),
            "request_id": batch.get("request_id"),
            "input_tokens": batch.get("input_tokens"),
            "output_tokens": batch.get("output_tokens"),
            "reasoning_output_tokens": batch.get(
                "reasoning_output_tokens"),
            "cost_usd_estimated": batch.get(
                "cost_usd_estimated"),
            "validation_error": batch.get("validation_error"),
        } for batch in batches],
        "frames_observed": len(by_id),
        "frames_expected": len(rows),
        "missing_ids": missing_ids,
        "offsets": offsets,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_reasoning_tokens": total_reason,
        "total_cost_usd_estimated": total_cost,
        "max_batch_wall_seconds": max(
            value for value in latencies
            if value is not None),
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "observations.json").write_text(
        json.dumps(by_id, indent=2, sort_keys=True),
        encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parallel 5x10 scale-reading experiment.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.dry_run) == bool(args.execute):
        print("Pass exactly one of --dry-run or --execute.")
        return 2
    live = args.execute
    out_dir = OUT_BASE / ("live1" if live else "dry1")
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"{out_dir} not empty; refusing to mix runs.")
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = json.loads(REF_PATH.read_text(
        encoding="utf-8"))["rows"]
    assert len(rows) == 50, len(rows)
    assert [row["position"] for row in rows] == list(
        range(50))
    groups = [rows[index:index + BATCH_SIZE]
              for index in range(0, 50, BATCH_SIZE)]
    assert len(groups) == CONCURRENCY
    assert all(len(group) == BATCH_SIZE for group in groups)
    for row in rows:
        path = (ANCHOR_DIR
                / f"anchored_{row['frame_index']:08d}.png")
        assert path.is_file(), path
    print(f"groups: {len(groups)} x {BATCH_SIZE}", flush=True)
    if live:
        key = resolve_api_key()
        print("api_key_present:", bool(key), flush=True)
        reservation = reserve_worst_case()
        print(f"worst_case_reservation_usd: {reservation}",
              flush=True)
        if reservation > CEILING_USD:
            print("Reservation exceeds ceiling; aborting "
                  "before any dispatch.")
            return 3

        def factory() -> MetaSparkClient:
            return MetaSparkClient(
                model=MODEL,
                timeout_seconds=TIMEOUT_SECONDS)
    else:
        key = ""
        plan = {}
        for batch_index, group in enumerate(groups):
            ids = [visual_frame_id_for(row["frame_index"])
                   for row in group]
            if batch_index == 4:
                plan[batch_index] = "not json {"
            else:
                plan[batch_index] = _fake_payload(
                    ids, {frame_id: "0" for frame_id in ids})

        def factory() -> FakeScaleClient:
            return FakeScaleClient(plan)

    gate = threading.Semaphore(CONCURRENCY)
    wall_start = time.time()

    def guarded(batch_index: int, group: list[dict[str,
                                                   Any]]):
        with gate:
            return run_batch(
                batch_index,
                [visual_frame_id_for(row["frame_index"])
                 for row in group],
                [ANCHOR_DIR / f"anchored_"
                 f"{row['frame_index']:08d}.png"
                 for row in group],
                [row["timestamp_seconds"] for row in group],
                key, out_dir, factory)

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=CONCURRENCY) as pool:
        futures = [pool.submit(guarded, batch_index, group)
                   for batch_index, group in enumerate(
                       groups)]
        records = [future.result() for future in futures]
    wall_seconds = round(time.time() - wall_start, 3)
    print(f"wall_seconds: {wall_seconds}", flush=True)
    report = evaluate(out_dir, rows)
    report["wall_seconds"] = wall_seconds
    report["mode"] = "live" if live else "dry_run"
    report["reservation_usd"] = (reserve_worst_case()
                                 if live else 0.0)
    report["ceiling_usd"] = CEILING_USD if live else 0.0
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    for record in records:
        print(f"batch_{record['batch_index']}: "
              f"{record['status']} "
              f"wall={record['wall_seconds']}s "
              f"err={record.get('validation_error')}",
              flush=True)
    print(f"observed: {report['frames_observed']}/"
          f"{report['frames_expected']}", flush=True)
    print(f"offsets: {report['offsets']}", flush=True)
    print(f"cost_usd_estimated: "
          f"{report['total_cost_usd_estimated']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
