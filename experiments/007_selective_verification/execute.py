"""Single-event selective verification execution (Exp. 011).

Loads a verification plan, resolves one event's existing frames,
submits a focused multimodal verification request through the shared
Meta client and lifecycle policy, reconciles the result against the
original first-pass observations, and preserves independent evidence.
Dry-run is the default; live inference requires --execute and never
runs from the agent environment in this task.
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_006 = REPO_ROOT / "experiments" / "006_p04_multimodal_inference"
sys.path.insert(0, str(EXPERIMENT_006))

import lifecycle  # noqa: E402
import meta_client  # noqa: E402
from runner import (  # noqa: E402
    ABORT_CODES,
    DEFAULT_BACKOFF_BASE_SECONDS,
    DEFAULT_BACKOFF_CAP_SECONDS,
    DEFAULT_MAX_ATTEMPTS,
    RETRYABLE_CODES,
)

from bakery_intelligence.verification.numerics import (  # noqa: E402
    STATUS_OK,
    normalize_number,
    numbers_match,
)

EXECUTION_VERSION = "1.0.0"
DEFAULT_BUDGETS = (1024, 2048, 3072)
TIMESTAMP_TOLERANCE = 1e-6

MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

VERIFY_PROMPT = """You are verifying scale-display evidence for one video interval.
The verification window spans {start:.3f} to {end:.3f} seconds for event {event_id}. The images below are listed in chronological order:
{image_table}
For each image, read the visible scale display at that image's own timestamp. Report only what the supplied images support. Distinguish directly visible text from inferred numerical values. Record uncertainty explicitly and abstain for any image with insufficient evidence. Never infer ingredient identity or measurement units without visible evidence. Never force agreement across timestamps.
Reply with exactly one JSON object and no other text, using these keys:
{{
  "event_id": "echo the event under verification",
  "readings": [
    {{"timestamp_seconds": number,
      "display_text": "exact digits/segments, or null",
      "numeric_value": "number only when unambiguous, else null",
      "readable": true/false,
      "confidence": "high/medium/low",
      "uncertainty_notes": "short note or null"}}],
  "action_visible": "addition/removal/tare/none/unknown",
  "overall_confidence": "high/medium/low",
  "uncertainty_notes": "short note or null"
}}"""


class ExecuteError(Exception):
    """Structured execution failure with a machine-readable code."""

    def __init__(self, code: str, message: str, entity_id: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.entity_id = entity_id


def sha256_file(path: Path) -> str:
    """SHA-256 over exact file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_plan(path: Path) -> dict[str, Any]:
    """Load and validate a verification plan."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecuteError(
            "PLAN_INVALID", f"Unreadable plan: {path}.", str(path)
        ) from exc
    if not isinstance(payload, dict):
        raise ExecuteError("PLAN_INVALID", "Plan root is not an object.")
    for key in ("plan_version", "entries"):
        if key not in payload:
            raise ExecuteError(
                "PLAN_INVALID", f"Plan lacks key: {key}.")
    if not isinstance(payload["entries"], list):
        raise ExecuteError("PLAN_INVALID", "Plan entries are not a list.")
    for position, entry in enumerate(payload["entries"]):
        for key in ("event_id", "start_timestamp_seconds",
                    "end_timestamp_seconds", "reuse_frames"):
            if not isinstance(entry, dict) or key not in entry:
                raise ExecuteError(
                    "PLAN_INVALID",
                    f"Entry #{position} lacks key: {key}.",
                    str(position),
                )
    return payload


def find_event(plan: dict[str, Any], event_id: str) -> dict[str, Any]:
    """Select exactly one planned event by identifier."""
    for entry in plan["entries"]:
        if entry["event_id"] == event_id:
            return entry
    raise ExecuteError("UNKNOWN_EVENT",
                       f"Event not in plan: {event_id}.", event_id)


def load_extraction_rows(manifest_csv: Path) -> dict[int, dict[str, Any]]:
    """Map frame_index to image name and manifest timestamp."""
    try:
        text = manifest_csv.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExecuteError(
            "MANIFEST_UNREADABLE", f"Unreadable manifest: {manifest_csv}.",
            str(manifest_csv),
        ) from exc
    rows = list(csv.DictReader(text.splitlines()))
    if not rows or "frame_index" not in rows[0] or (
            "image_path" not in rows[0]
            or "timestamp_seconds" not in rows[0]):
        raise ExecuteError(
            "MANIFEST_UNREADABLE",
            "Manifest lacks frame_index/image_path/timestamp_seconds.",
            str(manifest_csv),
        )
    mapping = {}
    for row in rows:
        try:
            mapping[int(row["frame_index"])] = {
                "image_path": row["image_path"],
                "timestamp_seconds": float(row["timestamp_seconds"]),
            }
        except (TypeError, ValueError) as exc:
            raise ExecuteError(
                "MANIFEST_UNREADABLE",
                f"Bad manifest row for frame {row.get('frame_index')!r}.",
                str(manifest_csv),
            ) from exc
    return mapping


def load_originals(obs_dir: Path) -> dict[int, dict[str, Any]]:
    """Index first-pass observations by frame_index (read-only)."""
    if not obs_dir.is_dir():
        raise ExecuteError("OBS_INVALID",
                           f"Not a directory: {obs_dir}.", str(obs_dir))
    indexed: dict[int, dict[str, Any]] = {}
    for path in sorted(obs_dir.glob("batch_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExecuteError(
                "OBS_INVALID", f"Unreadable observations: {path.name}.",
                path.name,
            ) from exc
        if not isinstance(payload, list):
            raise ExecuteError(
                "OBS_INVALID", f"Expected a list: {path.name}.",
                path.name,
            )
        for item in payload:
            if not isinstance(item, dict) or "frame_index" not in item:
                raise ExecuteError(
                    "OBS_INVALID",
                    f"Record without frame_index: {path.name}.",
                    path.name,
                )
            indexed[int(item["frame_index"])] = item
    return indexed


def resolve_frames(entry: dict[str, Any],
                   extraction_rows: dict[int, dict[str, Any]],
                   frames_dir: Path,
                   originals: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve planned frames to files, timestamps, and originals.

    Chronological by frame_index. Never substitutes another frame:
    every planned index resolves to its own file or is recorded
    missing.
    """
    resolved = []
    for ref in entry["reuse_frames"]:
        index = int(ref["frame_index"])
        row = extraction_rows.get(index)
        original = originals.get(index)
        path: Path | None = None
        if row is not None:
            path = frames_dir / row["image_path"]
        exists = path is not None and path.is_file()
        mime = None
        digest = None
        size = None
        if exists:
            assert path is not None
            mime = MIME_BY_SUFFIX.get(path.suffix.lower())
            digest = sha256_file(path)
            size = path.stat().st_size
        timestamp = row["timestamp_seconds"] if row else None
        obs_timestamp = None
        original_fields = None
        batch_id = None
        if original is not None:
            obs_timestamp = original.get("timestamp_seconds")
            fields = original.get("model_fields", {})
            original_fields = {
                "scale_display_verbatim": fields.get(
                    "scale_display_verbatim"),
                "scale_readable": fields.get("scale_readable"),
                "scale_state": fields.get("scale_state"),
                "tare_action_visible": fields.get("tare_action_visible"),
                "addition_or_removal_visible": fields.get(
                    "addition_or_removal_visible"),
                "confidence": fields.get("confidence"),
            }
            batch_id = original.get("batch_id")
        timestamp_note = None
        if timestamp is not None and obs_timestamp is not None and (
                abs(timestamp - obs_timestamp) > TIMESTAMP_TOLERANCE):
            timestamp_note = (
                f"manifest t={timestamp} differs from "
                f"observation t={obs_timestamp}"
            )
        resolved.append({
            "frame_index": index,
            "timestamp_seconds": timestamp,
            "observation_timestamp_seconds": obs_timestamp,
            "timestamp_note": timestamp_note,
            "path": str(path) if path else None,
            "file_exists": exists,
            "mime": mime,
            "sha256": digest,
            "bytes": size,
            "batch_id": batch_id,
            "original": original_fields,
        })
    resolved.sort(key=lambda item: item["frame_index"])
    return resolved


def check_usable(entry: dict[str, Any],
                 resolved: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Eligibility: usable images, originals, and request limits."""
    reasons: list[str] = []
    usable = [item for item in resolved
              if item["file_exists"] and item["mime"] is not None]
    if not usable:
        reasons.append("no_available_images: no planned frame resolves "
                       "to an existing supported image")
    if len(usable) > meta_client.MAX_IMAGES_PER_REQUEST:
        reasons.append(
            f"too_many_images: {len(usable)} exceeds "
            f"{meta_client.MAX_IMAGES_PER_REQUEST}")
    if not any(item["original"] is not None for item in usable):
        reasons.append("no_originals: no usable frame has a first-pass "
                       "observation to compare against")
    for item in resolved:
        if item["file_exists"] and item["mime"] is None:
            reasons.append(
                f"unsupported_encoding: frame {item['frame_index']}")
    return (not reasons, reasons)


def build_prompt(entry: dict[str, Any],
                 usable: list[dict[str, Any]],
                 start: float | None = None,
                 end: float | None = None) -> tuple[str, str]:
    """Frozen verification prompt; timestamps and frames only.

    The stated window defaults to the event bounds; a localized
    verification passes the requested sub-interval instead so the
    prompt never misdescribes the supplied image window.
    """
    lines = [
        f"Image {position}: timestamp {item['timestamp_seconds']:.3f} s "
        f"(source frame {item['frame_index']})"
        for position, item in enumerate(usable, start=1)
    ]
    if start is None:
        start = entry["start_timestamp_seconds"]
    if end is None:
        end = entry["end_timestamp_seconds"]
    text = VERIFY_PROMPT.format(
        start=start,
        end=end,
        event_id=entry["event_id"],
        image_table="\n".join(lines),
    )
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_window_args(start: Any, end: Any,
                      entry: dict[str, Any]
                      ) -> tuple[float | None, float | None]:
    """Validate the optional temporal subset window.

    Returns (None, None) for full-event behavior. Never clamps:
    out-of-event bounds are rejected.
    """
    if start is None and end is None:
        return None, None
    if start is None or end is None:
        raise ExecuteError(
            "INVALID_WINDOW",
            "Temporal selection requires both --window-start and "
            "--window-end.",
            entry["event_id"],
        )
    if not isinstance(start, (int, float)) or not isinstance(
            end, (int, float)) or isinstance(start, bool) or isinstance(
            end, bool) or not math.isfinite(start) or not math.isfinite(
            end):
        raise ExecuteError(
            "INVALID_WINDOW",
            f"Window bounds must be finite numbers: {start!r}, {end!r}.",
            entry["event_id"],
        )
    if start > end:
        raise ExecuteError(
            "INVALID_WINDOW",
            f"Window start {start} exceeds end {end}.",
            entry["event_id"],
        )
    event_start = entry["start_timestamp_seconds"]
    event_end = entry["end_timestamp_seconds"]
    if start < event_start or end > event_end:
        raise ExecuteError(
            "WINDOW_OUT_OF_BOUNDS",
            f"Window {start}..{end} extends outside event bounds "
            f"{event_start}..{event_end}; not clamped.",
            entry["event_id"],
        )
    return float(start), float(end)


def select_window(resolved: list[dict[str, Any]],
                  start: float | None, end: float | None,
                  event_id: str) -> list[dict[str, Any]]:
    """Select existing frames in the inclusive window (or all)."""
    if start is None or end is None:
        return list(resolved)
    tolerance = TIMESTAMP_TOLERANCE
    selected = [
        item for item in resolved
        if item["timestamp_seconds"] is not None
        and start - tolerance <= item["timestamp_seconds"] <= end + tolerance
    ]
    if not selected:
        raise ExecuteError(
            "WINDOW_EMPTY",
            f"Window {start}..{end} selects zero existing frames.",
            event_id,
        )
    return selected


def selection_record(entry: dict[str, Any],
                     start: float | None, end: float | None,
                     selected: list[dict[str, Any]]) -> dict[str, Any]:
    """Auditable record of full-event vs temporal-subset selection."""
    return {
        "temporal_subset": start is not None,
        "original_start_seconds": entry["start_timestamp_seconds"],
        "original_end_seconds": entry["end_timestamp_seconds"],
        "requested_start_seconds": start,
        "requested_end_seconds": end,
        "selected_frame_indices": [item["frame_index"]
                                   for item in selected],
    }


def parse_verification(text: Any) -> tuple[bool, list[Any] | None,
                                           dict[str, Any] | None]:
    """Parse readings plus envelope; no repair attempts."""
    if not isinstance(text, str):
        return False, None, None
    try:
        payload = json.loads(text.strip())
    except (json.JSONDecodeError, AttributeError):
        return False, None, None
    if not isinstance(payload, dict) or not isinstance(
            payload.get("readings"), list):
        return False, None, None
    return True, payload["readings"], payload


def reconcile_units(verified_unit: Any,
                    original_unit: Any) -> tuple[str, str]:
    """Unit comparison with explicit missing-data handling."""
    if verified_unit is None and original_unit is None:
        return "not_comparable", "units absent on both sides"
    if verified_unit is None or original_unit is None:
        return "insufficient_evidence", "unit present on one side only"
    if verified_unit == original_unit:
        return "agreement", "equal units"
    return "disagreement", (
        f"distinct units: {verified_unit!r} != {original_unit!r}")


TEMPORAL_REASONS = frozenset({"temporal_inconsistency"})


def build_temporal_review(review_reasons: Any) -> dict[str, Any]:
    """Temporal review state from the event's original reasons.

    This executor performs no temporal evaluation, so a temporal
    reason can never be marked resolved here. Numerical agreement
    and temporal resolution are independent dimensions.
    """
    reasons = list(review_reasons) if isinstance(review_reasons,
                                                 list) else []
    temporal = [reason for reason in reasons
                if reason in TEMPORAL_REASONS]
    if temporal:
        return {
            "status": "not_evaluated",
            "reasons": temporal,
            "detail": ("Temporal review is not evaluated by this "
                       "executor; numerical agreement does not "
                       "resolve it."),
        }
    return {"status": "not_applicable", "reasons": [],
            "detail": "No temporal review reasons on this event."}


def aggregate_spent(costs: list[float | None]
                    ) -> tuple[float | None, int]:
    """Total known spend, or unknown when any attempt lacks cost.

    Zero requests yield zero spend; any unknown attempt cost yields
    an unknown total, never a false zero or partial sum presented
    as complete.
    """
    unknown = sum(1 for cost in costs if cost is None)
    if not costs:
        return 0.0, 0
    if unknown:
        return None, unknown
    return round(sum(cost for cost in costs
                     if cost is not None), 6), 0


def reconcile(resolved: list[dict[str, Any]],
              readings: list[Any] | None) -> dict[str, Any]:
    """Compare verified readings against originals, frame by frame.

    Readings match frames by chronological order; timestamps are
    checked but order is authoritative. Disagreements are preserved
    for human review; nothing is promoted to ground truth.
    """
    usable = [item for item in resolved
              if item["file_exists"] and item["mime"] is not None]
    per_frame = []
    for position, item in enumerate(usable):
        reading = readings[position] if readings is not None and (
            position < len(readings)) else None
        if not isinstance(reading, dict):
            per_frame.append({
                "frame_index": item["frame_index"],
                "outcome": "insufficient_evidence",
                "detail": "no verified reading for this frame",
                "unit_outcome": "not_comparable",
            })
            continue
        timestamp_note = None
        stamp = reading.get("timestamp_seconds")
        if not isinstance(stamp, (int, float)) or abs(
                stamp - item["timestamp_seconds"]) > TIMESTAMP_TOLERANCE:
            timestamp_note = (
                f"reading timestamp {stamp!r} does not match frame "
                f"t={item['timestamp_seconds']}")
        display = reading.get("display_text")
        if reading.get("readable") is False or not (
                isinstance(display, str) and display):
            per_frame.append({
                "frame_index": item["frame_index"],
                "outcome": "verification_abstained",
                "detail": "verifier abstained or gave no display text",
                "timestamp_note": timestamp_note,
                "unit_outcome": "not_comparable",
            })
            continue
        original = item["original"]
        if original is None:
            per_frame.append({
                "frame_index": item["frame_index"],
                "outcome": "insufficient_evidence",
                "detail": "no first-pass observation for this frame",
                "timestamp_note": timestamp_note,
                "unit_outcome": "not_comparable",
            })
            continue
        verbatim = original.get("scale_display_verbatim")
        normalized = normalize_number(verbatim)
        if normalized.status != STATUS_OK:
            per_frame.append({
                "frame_index": item["frame_index"],
                "outcome": "original_ambiguous",
                "detail": f"original {verbatim!r} is "
                          f"{normalized.status} ({normalized.note})",
                "timestamp_note": timestamp_note,
                "unit_outcome": "not_comparable",
            })
            continue
        verified_value = reading.get("numeric_value")
        if verified_value is None:
            verified_value = display
        match, note = numbers_match(verified_value, normalized.value)
        unit_outcome, unit_note = reconcile_units(
            reading.get("unit"), None)
        if display == verbatim or match:
            outcome = "agreement"
            detail = ("exact display match" if display == verbatim
                      else f"normalized numeric match ({note})")
        else:
            outcome = "disagreement"
            detail = (f"display {display!r} vs original {verbatim!r}; "
                      f"{note}")
        per_frame.append({
            "frame_index": item["frame_index"],
            "outcome": outcome,
            "detail": detail,
            "timestamp_note": timestamp_note,
            "unit_outcome": unit_outcome,
            "unit_detail": unit_note,
        })
    counts: dict[str, int] = {}
    for item in per_frame:
        counts[item["outcome"]] = counts.get(item["outcome"], 0) + 1
    return {
        "per_frame": per_frame,
        "counts": counts,
        "requires_human_review": any(
            item["outcome"] != "agreement" for item in per_frame),
    }


def _backoff_delay(attempt: int, base: float, cap: float,
                   rng: random.Random) -> float:
    delay = min(base * (2.0 ** (attempt - 1)), cap)
    return delay + rng.uniform(0.0, delay * 0.25)


def _send_once(client: Any, prompt_text: str, image_paths: list[Path],
               api_key: str, budget: int) -> dict[str, Any]:
    """One transport attempt; maps provider outcomes to a record."""
    client.max_output_tokens = budget
    try:
        result = client.submit(prompt_text, image_paths, api_key)
    except meta_client.ProviderError as exc:
        return {"transport_ok": False, "code": exc.code,
                "message": exc.message,
                "status_code": exc.status_code,
                "retry_after_seconds": exc.retry_after_seconds}
    return {"transport_ok": True, "result": result}


def execute_event(entry: dict[str, Any], usable: list[dict[str, Any]],
                  prompt_text: str, prompt_sha: str,
                  model: str, budgets: tuple[int, ...], store: bool,
                  max_requests: int, rates: tuple[float, float] | None,
                  cost_ceiling: float | None, hard_ceiling: bool,
                  api_key: str, run_dir: Path, video_id: str | None,
                  client_factory: Any = None,
                  sleeper: Any = None,
                  seed: int = 20261101,
                  selection: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run budgeted verification attempts for one event (paid)."""
    if hard_ceiling and cost_ceiling is not None:
        raise ExecuteError(
            "COST_UNKNOWN",
            "Hard cost ceiling configured but no pre-execution cost "
            "estimate exists; refusing to start.",
            entry["event_id"],
        )
    attempts_dir = run_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    factory = client_factory or meta_client._default_client_factory
    client = meta_client.MetaClient(
        model=model, timeout_seconds=120.0,
        max_output_tokens=budgets[0], store=store,
        client_factory=factory,
    )
    policy = lifecycle.BudgetPolicy(budgets=budgets)
    rng = random.Random(seed)
    sleep = sleeper or (lambda seconds: time.sleep(seconds))
    image_paths = [Path(item["path"]) for item in usable]
    image_hashes = [item["sha256"] for item in usable]
    mimes = [item["mime"] for item in usable]
    attempt_files: list[str] = []
    request_count = 0
    spent_known = 0.0
    attempt_costs: list[float | None] = []
    terminal: str | None = None
    selected: dict[str, Any] | None = None
    final_readings: list[Any] | None = None
    stopped_reason: str | None = None
    started = time.perf_counter()
    for number, budget in enumerate(budgets, start=1):
        transport_log: list[dict[str, Any]] = []
        record: dict[str, Any] | None = None
        transport_error: str | None = None
        for trial in range(DEFAULT_MAX_ATTEMPTS):
            if request_count >= max_requests:
                stopped_reason = (
                    f"max_requests exceeded ({max_requests}); stopping")
                break
            request_count += 1
            outcome = _send_once(client, prompt_text, image_paths,
                                 api_key, budget)
            if not outcome["transport_ok"]:
                code = outcome["code"]
                transport_log.append({
                    "try": trial + 1, "code": code,
                    "message": outcome["message"],
                    "status_code": outcome["status_code"],
                })
                if code in ABORT_CODES:
                    raise ExecuteError(
                        "AUTH_FAILED",
                        f"Aborting on {code}: {outcome['message']}.",
                        entry["event_id"],
                    )
                if code in RETRYABLE_CODES and (
                        trial + 1 < DEFAULT_MAX_ATTEMPTS):
                    wait = outcome["retry_after_seconds"]
                    if wait is None:
                        wait = _backoff_delay(
                            trial + 1, DEFAULT_BACKOFF_BASE_SECONDS,
                            DEFAULT_BACKOFF_CAP_SECONDS, rng)
                    sleep(wait)
                    transport_log[-1]["slept_seconds"] = wait
                    continue
                transport_error = code
                break
            result = outcome["result"]
            record = {
                "status": result.status,
                "incomplete_reason": result.incomplete_reason,
                "output_text": result.output_text,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.total_tokens,
                "reasoning_output_tokens":
                    result.reasoning_output_tokens,
                "latency_seconds": result.latency_seconds,
                "request_id": result.request_id,
            }
            transport_log.append({"try": trial + 1, "code": "ok"})
            break
        if stopped_reason is not None:
            break
        if transport_error is not None:
            decision = lifecycle.decide_attempt(
                None, None, None, False, number, policy,
                transport_error=transport_error)
            attempt = _attempt_record(
                entry, number, budget, prompt_text, prompt_sha,
                image_hashes, mimes, model, store, None,
                transport_error, transport_log, decision, None,
                False, None)
            _write_json(attempts_dir / f"{entry['event_id']}_attempt_"
                                       f"{number:02d}.json", attempt)
            attempt_files.append(str(
                attempts_dir / f"{entry['event_id']}_attempt_"
                               f"{number:02d}.json"))
            attempt_costs.append(None)
            terminal = decision["terminal"]
            break
        assert record is not None
        cost = lifecycle.estimate_cost(
            record["input_tokens"], record["output_tokens"], rates,
            ndigits=6)
        parse_ok, readings, envelope = parse_verification(
            record["output_text"])
        usable_readings = (
            [reading for reading in (readings or [])
             if isinstance(reading, dict)
             and reading.get("readable") is not False
             and isinstance(reading.get("display_text"), str)
             and reading["display_text"]] if parse_ok else [])
        abstained_hint = parse_ok and not usable_readings
        decision = lifecycle.decide_attempt(
            record["status"], record["incomplete_reason"],
            record["output_text"], parse_ok, number, policy,
            abstained_hint=bool(abstained_hint))
        attempt = _attempt_record(
            entry, number, budget, prompt_text, prompt_sha,
            image_hashes, mimes, model, store, record, None,
            transport_log, decision, readings if parse_ok else None,
            parse_ok, cost)
        path = attempts_dir / f"{entry['event_id']}_attempt_{number:02d}.json"
        _write_json(path, attempt)
        attempt_files.append(str(path))
        attempt_costs.append(cost)
        if cost is not None:
            spent_known += cost
        if decision["action"] == "retry":
            if cost_ceiling is not None and spent_known > cost_ceiling:
                stopped_reason = (
                    f"cost_ceiling_exceeded: known spent "
                    f"~{spent_known:.6f} over ceiling {cost_ceiling}; "
                    f"escalation halted")
                terminal = None
                break
            continue
        terminal = decision["terminal"]
        if parse_ok and terminal in (lifecycle.TERMINAL_VALID,
                                     lifecycle.TERMINAL_ABSTAINED):
            final_readings = readings
        if terminal == lifecycle.TERMINAL_VALID:
            selected = {"attempt_number": number,
                        "readings": readings,
                        "envelope_action": (
                            envelope or {}).get("action_visible"),
                        "envelope_confidence": (
                            envelope or {}).get("overall_confidence"),
                        "envelope_notes": (
                            envelope or {}).get("uncertainty_notes")}
        break
    elapsed = time.perf_counter() - started
    reconciliation = reconcile(usable, final_readings)
    temporal_review = build_temporal_review(entry.get("review_reasons"))
    if temporal_review["status"] == "not_evaluated":
        reconciliation["requires_human_review"] = True
    spent_total, spent_unknown = aggregate_spent(attempt_costs)
    return {
        "event_id": entry["event_id"],
        "video_id": video_id,
        "window": {
            "start_timestamp_seconds": entry["start_timestamp_seconds"],
            "end_timestamp_seconds": entry["end_timestamp_seconds"],
        },
        "review_reasons": list(entry.get("review_reasons", [])),
        "selection": selection,
        "originals": [
            {"frame_index": item["frame_index"],
             "timestamp_seconds": item["timestamp_seconds"],
             "batch_id": item["batch_id"],
             "values": item["original"]}
            for item in usable
        ],
        "inputs": [
            {"frame_index": item["frame_index"],
             "timestamp_seconds": item["timestamp_seconds"],
             "path": item["path"], "sha256": item["sha256"],
             "bytes": item["bytes"], "mime": item["mime"]}
            for item in usable
        ],
        "prompt_sha256": prompt_sha,
        "prompt_text": prompt_text,
        "model": model,
        "token_budgets": list(budgets),
        "store": store,
        "max_requests": max_requests,
        "cost_ceiling": cost_ceiling,
        "attempt_files": attempt_files,
        "requests_made": request_count,
        "terminal_state": terminal,
        "stopped_reason": stopped_reason,
        "selected_attempt": (
            selected["attempt_number"] if selected else None),
        "selected_readings": selected["readings"] if selected else None,
        "selected_envelope": (
            {key: selected[key] for key in
             ("envelope_action", "envelope_confidence",
              "envelope_notes")} if selected else None),
        "reconciliation": reconciliation,
        "temporal_review": temporal_review,
        "spent_cost_usd_estimate": spent_total,
        "spent_cost_usd_unknown_attempts": spent_unknown,
        "elapsed_seconds": round(elapsed, 3),
    }


def _attempt_record(entry: dict[str, Any], number: int, budget: int,
                    prompt_text: str, prompt_sha: str,
                    image_hashes: list[str], mimes: list[str],
                    model: str, store: bool,
                    record: dict[str, Any] | None,
                    transport_error: str | None,
                    transport_log: list[dict[str, Any]],
                    decision: dict[str, Any],
                    readings: list[Any] | None, parse_ok: bool,
                    cost: float | None) -> dict[str, Any]:
    """Assemble one persisted attempt record."""
    return {
        "event_id": entry["event_id"],
        "attempt_number": number,
        "max_output_tokens": budget,
        "model": model,
        "prompt_hash": prompt_sha,
        "prompt_text": prompt_text,
        "input_image_hashes": image_hashes,
        "input_mimes": mimes,
        "store": store,
        "request_id": (record or {}).get("request_id"),
        "status": (record or {}).get("status"),
        "incomplete_reason": (record or {}).get("incomplete_reason"),
        "raw_response_text": (record or {}).get("output_text"),
        "normalized_response_text": (
            ((record or {}).get("output_text") or "").strip() or None),
        "json_parse_ok": parse_ok,
        "parsed_readings": readings,
        "input_tokens": (record or {}).get("input_tokens"),
        "output_tokens": (record or {}).get("output_tokens"),
        "reasoning_output_tokens": (
            (record or {}).get("reasoning_output_tokens")),
        "total_tokens": (record or {}).get("total_tokens"),
        "latency_seconds": (record or {}).get("latency_seconds"),
        "transport_error": transport_error,
        "transport_attempts": transport_log,
        "retry_triggered": decision["action"] == "retry",
        "retry_reason": decision["reason"],
        "terminal": decision["terminal"],
        "cost_usd_estimate": cost,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )


def dry_run_report(entry: dict[str, Any], resolved: list[dict[str, Any]],
                   eligible: bool, reasons: list[str],
                   prompt_sha: str | None, model: str,
                   budgets: tuple[int, ...], max_requests: int,
                   cost_ceiling: float | None,
                   output_path: Path,
                   selection: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the dry-run report (zero API requests)."""
    usable = [item for item in resolved
              if item["file_exists"] and item["mime"] is not None]
    if selection is None:
        selection = selection_record(entry, None, None, resolved)
    return {
        "mode": "dry_run",
        "event_id": entry["event_id"],
        "window": {
            "start_timestamp_seconds": entry["start_timestamp_seconds"],
            "end_timestamp_seconds": entry["end_timestamp_seconds"],
        },
        "selection": selection,
        "review_reasons": list(entry.get("review_reasons", [])),
        "image_count": len(usable),
        "images": [
            {"frame_index": item["frame_index"],
             "timestamp_seconds": item["timestamp_seconds"],
             "available": item["file_exists"],
             "encoding": item["mime"],
             "sha256": item["sha256"]}
            for item in resolved
        ],
        "prompt_sha256": prompt_sha,
        "model": model,
        "token_budgets": list(budgets),
        "max_requests": max_requests,
        "cost_ceiling": cost_ceiling,
        "output_path": str(output_path),
        "eligible": eligible,
        "validation_errors": reasons,
    }


def _parse_budgets(text: str) -> tuple[int, ...]:
    try:
        budgets = tuple(
            int(part) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise ExecuteError(
            "INVALID_CONFIG", f"Unparseable --token-budgets: {text!r}."
        ) from exc
    if not budgets or any(budget <= 0 for budget in budgets):
        raise ExecuteError(
            "INVALID_CONFIG", f"Unparseable --token-budgets: {text!r}.")
    return budgets


def main(argv: list[str] | None = None,
         client_factory: Any = None) -> int:
    """CLI: dry-run by default; live only with --execute."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--obs-dir", type=Path, required=True)
    parser.add_argument("--extraction-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(
        "data/processed/experiment_007/selective_verification/executions"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model", default=meta_client.MODEL_SPARK)
    parser.add_argument("--token-budgets", default="1024,2048,3072")
    parser.add_argument("--max-requests", type=int, default=12)
    parser.add_argument("--price-input", type=float, default=None)
    parser.add_argument("--price-output", type=float, default=None)
    parser.add_argument("--cost-ceiling", type=float, default=None)
    parser.add_argument("--hard-ceiling", action="store_true")
    parser.add_argument("--store", dest="store", action="store_true",
                        default=True)
    parser.add_argument("--no-store", dest="store", action="store_false")
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--window-start", type=float, default=None)
    parser.add_argument("--window-end", type=float, default=None)
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        default=None)
    parser.add_argument("--execute", dest="execute", action="store_true",
                        default=False)
    args = parser.parse_args(argv)
    if args.dry_run and args.execute:
        print("EXECUTE FAILED [INVALID_CONFIG]: --dry-run and --execute "
              "are mutually exclusive.")
        return 1
    live = bool(args.execute)
    try:
        plan = load_plan(args.plan)
        entry = find_event(plan, args.event_id)
        rows = load_extraction_rows(args.extraction_manifest)
        originals = load_originals(args.obs_dir)
        resolved = resolve_frames(
            entry, rows, args.extraction_manifest.parent, originals)
        window_start, window_end = parse_window_args(
            args.window_start, args.window_end, entry)
        selected = select_window(
            resolved, window_start, window_end, args.event_id)
        selection = selection_record(
            entry, window_start, window_end, selected)
        eligible, reasons = check_usable(entry, selected)
        budgets = _parse_budgets(args.token_budgets)
        try:
            lifecycle.BudgetPolicy(budgets=budgets)
        except ValueError as exc:
            raise ExecuteError("INVALID_CONFIG", str(exc)) from exc
        if args.max_requests <= 0:
            raise ExecuteError("INVALID_CONFIG",
                               "max_requests must be positive.")
        rates: tuple[float, float] | None = None
        if (args.price_input is None) != (args.price_output is None):
            raise ExecuteError("INVALID_CONFIG",
                               "Price overrides must be paired.")
        if args.price_input is not None:
            rates = (args.price_input, args.price_output)
        stamp = datetime.datetime.now(datetime.timezone.utc)
        run_id = args.run_id or stamp.strftime(
            f"{args.event_id}_%Y%m%dT%H%M%SZ")
        run_dir = args.out / run_id
        usable = [item for item in selected
                  if item["file_exists"] and item["mime"] is not None]
        prompt_text: str | None = None
        prompt_sha: str | None = None
        if eligible:
            prompt_text, prompt_sha = build_prompt(
                entry, usable, window_start, window_end)
        if not live:
            report = dry_run_report(
                entry, selected, eligible, reasons, prompt_sha,
                args.model, budgets, args.max_requests,
                args.cost_ceiling, run_dir / "execution_report.json",
                selection)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if eligible else 1
        if not eligible:
            print(f"EXECUTE FAILED [INELIGIBLE_EVENT]: {'; '.join(reasons)}")
            return 1
        try:
            api_key = meta_client.resolve_api_key()
        except meta_client.ProviderError as exc:
            raise ExecuteError("CREDENTIAL_MISSING", exc.message) from exc
        if args.hard_ceiling and args.cost_ceiling is not None:
            raise ExecuteError(
                "COST_UNKNOWN",
                "Hard cost ceiling configured but no pre-execution "
                "cost estimate exists; refusing to start.")
        if run_dir.exists():
            raise ExecuteError(
                "OUTPUT_EXISTS",
                f"Refusing to overwrite run: {run_dir}.")
        run_dir.mkdir(parents=True)
        assert prompt_text is not None and prompt_sha is not None
        result = execute_event(
            entry, usable, prompt_text, prompt_sha, args.model,
            budgets, args.store, args.max_requests, rates,
            args.cost_ceiling, args.hard_ceiling, api_key, run_dir,
            args.video_id, client_factory,
            selection=selection,
        )
        _write_json(run_dir / "execution_report.json", {
            "execution_version": EXECUTION_VERSION,
            **result,
        })
        print(json.dumps({
            "event_id": result["event_id"],
            "terminal_state": result["terminal_state"],
            "requests_made": result["requests_made"],
            "reconciliation": result["reconciliation"]["counts"],
            "requires_human_review": result["reconciliation"][
                "requires_human_review"],
        }, indent=2, sort_keys=True))
        return 0
    except ExecuteError as exc:
        print(f"EXECUTE FAILED [{exc.code}]: {exc.message}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
