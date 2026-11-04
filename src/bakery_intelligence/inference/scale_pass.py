"""Parallel scale-reading FirstPass engine.

Production integration of the validated 10-image parallel
scale experiment: groups of ten anchored scale photos share
one Muse Spark request, and up to five requests run
concurrently. Each request is independent: no rolling
context, no previous-batch summaries, no shared model
history. Responses are strict per-frame JSON
(``frame_id``/``display_value``/``unit``/``confidence``)
adapted at the boundary into the standard FirstPass
observation envelope, so downstream weighing, review, and
recipe stages consume scale readings through their existing
tolerant readers.

Safety properties: exactly one submit per batch (no
retries, no reruns), worst-case cost reservation for the
whole run before dispatch, per-batch checkpoints with
resume that never rebills a batch whose response was
received, strict unknown-cost halting, and explicit
terminal failure instead of partial reconstruction. The
module performs no network access itself: one fresh client
per batch comes from the injected factory, so automated
tests run fully offline.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import hashlib
import json
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bakery_intelligence.inference.firstpass import (
    STOP_COST_CEILING,
    STOP_REQUEST_LIMIT,
    STOP_STRICT_UNKNOWN_COST,
    TERMINAL_ABSTAINED,
    TERMINAL_EMPTY,
    TERMINAL_EXHAUSTED,
    TERMINAL_FAILED,
    TERMINAL_INVALID_JSON,
    TERMINAL_UNSUPPORTED,
    TERMINAL_VALID,
    TIMESTAMP_BASIS,
    UNCERTAIN_REFUSE,
    UNCERTAIN_RERUN,
    FirstPassError,
    FrameRef,
    InferenceError,
    InferenceResponse,
    InferenceUsage,
    SourceRef,
    estimate_cost,
    is_token_limit_reason,
    prompt_hash,
)
from bakery_intelligence.inference.firstpass_v3 import (
    ANCHOR_DEFAULT_SALT,
    prepare_anchored_image,
    verify_anchored_image,
    visual_frame_id_for,
)
from bakery_intelligence.inference.meta_spark import (
    MAX_IMAGES_PER_REQUEST,
    MODEL_SPARK,
)

SCALE_PASS_ID = "scale_pass_v1"
DEFAULT_SCALE_GROUP_SIZE = 10
DEFAULT_SCALE_CONCURRENCY = 5
DEFAULT_SCALE_TOKEN_BUDGET = 8192
# Conservative per-request input bound (10 anchored PNGs +
# prompt). Measured ~650 tokens per anchored image; the
# bound only sizes the worst-case cost reservation.
DEFAULT_INPUT_TOKEN_BOUND = 9000
SCALE_SCHEMA_NAME = "ScaleBatch10"

SCALE_JSON_SCHEMA: dict[str, Any] = {
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
# Length keywords (minItems/maxItems) are deliberately
# absent: provider support is unverified and a schema
# rejection would waste a paid request. Exact coverage is
# enforced independently by validate_scale_response.


def build_scale_prompt(
        entries: Sequence[tuple[str, float]]) -> str:
    """Assemble the minimal scale-only batch prompt.

    This is the text validated live at 10 images per
    request; keep changes minimal and re-validate live.
    """
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
class ScaleValidation:
    """Outcome of strict validation for one batch reply."""

    ok: bool
    observations: dict[str, dict[str, Any]] | None = None
    error: str | None = None


def validate_scale_response(
        raw_text: str, expected_ids: Sequence[str]
        ) -> ScaleValidation:
    """Require exactly the expected frame observations.

    Acceptance requires: a JSON object, a ``frames`` array
    with exactly one entry per requested image, every
    ``frame_id`` expected exactly once (no missing,
    duplicated, unexpected, or misattributed ids), and
    well-typed ``display_value``/``unit``/``confidence``
    fields. Anything else is rejected with a reason;
    partial payloads never become observations.
    """
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return ScaleValidation(
            ok=False, error=f"unparseable_json:{exc}")
    if not isinstance(parsed, dict):
        return ScaleValidation(
            ok=False, error="top_level_not_an_object")
    entries = parsed.get("frames")
    if not isinstance(entries, list):
        return ScaleValidation(
            ok=False, error="frames_not_a_list")
    if len(entries) != len(expected_ids):
        return ScaleValidation(
            ok=False,
            error=(f"frame_count_mismatch: got {len(entries)}, "
                   f"expected {len(expected_ids)}"))
    want = set(expected_ids)
    seen: dict[str, dict[str, Any]] = {}
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            return ScaleValidation(
                ok=False,
                error=f"entry_not_an_object:position_{position}")
        frame_id = entry.get("frame_id")
        if frame_id not in want:
            return ScaleValidation(
                ok=False,
                error=(f"unexpected_frame_id:position_"
                       f"{position}_got_{frame_id!r}"))
        if frame_id in seen:
            return ScaleValidation(
                ok=False,
                error=f"duplicate_frame_id:{frame_id}")
        value = entry.get("display_value")
        unit = entry.get("unit")
        confidence = entry.get("confidence")
        if not (value is None or isinstance(value, str)):
            return ScaleValidation(
                ok=False,
                error=f"bad_display_value:{frame_id}")
        if not (unit is None or isinstance(unit, str)):
            return ScaleValidation(
                ok=False, error=f"bad_unit:{frame_id}")
        if confidence not in ("high", "medium", "low"):
            return ScaleValidation(
                ok=False,
                error=f"bad_confidence:{frame_id}")
        seen[frame_id] = {"display_value": value,
                          "unit": unit,
                          "confidence": confidence}
    return ScaleValidation(ok=True, observations=seen)


@dataclass
class ScalePassConfig:
    """Complete parallel scale-pass configuration."""

    frames: list[FrameRef]
    source: SourceRef
    out_dir: Path
    model: str = MODEL_SPARK
    provider: str = "meta"
    pass_id: str = SCALE_PASS_ID
    group_size: int = DEFAULT_SCALE_GROUP_SIZE
    max_concurrency: int = DEFAULT_SCALE_CONCURRENCY
    token_budget: int = DEFAULT_SCALE_TOKEN_BUDGET
    max_batches: int | None = None
    rates_usd_per_m: tuple[float, float] | None = None
    cost_ceiling_usd: float | None = None
    strict_cost: bool = False
    billable: bool = False
    uncertain_policy: str = UNCERTAIN_RERUN
    timeout_seconds: float = 120.0
    api_key: str = ""
    anchor_salt: str = ANCHOR_DEFAULT_SALT
    input_token_bound: int = DEFAULT_INPUT_TOKEN_BOUND
    client_factory: Callable[[], Any] | None = None


def _positive_int(name: str, value: Any) -> int:
    if (isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1):
        raise FirstPassError(
            "INVALID_CONFIG",
            f"{name} must be a positive int, got {value!r}.")
    return value


def _validate_scale_config(config: ScalePassConfig) -> None:
    _positive_int("group_size", config.group_size)
    if config.group_size > MAX_IMAGES_PER_REQUEST:
        raise FirstPassError(
            "INVALID_CONFIG",
            f"group_size {config.group_size} exceeds the "
            f"per-request image maximum "
            f"{MAX_IMAGES_PER_REQUEST}.")
    _positive_int("max_concurrency", config.max_concurrency)
    _positive_int("token_budget", config.token_budget)
    _positive_int("input_token_bound",
                  config.input_token_bound)
    if config.max_batches is not None:
        _positive_int("max_batches", config.max_batches)
    if config.uncertain_policy not in (
            UNCERTAIN_RERUN, UNCERTAIN_REFUSE):
        raise FirstPassError(
            "INVALID_CONFIG",
            "uncertain_policy must be "
            f"{UNCERTAIN_RERUN!r} or {UNCERTAIN_REFUSE!r}, "
            f"got {config.uncertain_policy!r}.")
    if not config.frames:
        raise FirstPassError(
            "INVALID_CONFIG", "frames must not be empty.")
    if config.client_factory is None:
        raise FirstPassError(
            "INVALID_CONFIG",
            "client_factory is required (one fresh client "
            "per batch).")
    if config.billable:
        if config.uncertain_policy != UNCERTAIN_REFUSE:
            raise FirstPassError(
                "INVALID_CONFIG",
                "billable execution requires "
                f"uncertain_policy={UNCERTAIN_REFUSE!r}.")
        if (config.cost_ceiling_usd is not None
                and config.rates_usd_per_m is None):
            raise FirstPassError(
                "INVALID_CONFIG",
                "billable cost ceiling requires rates to "
                "price reservations.")


def worst_single_cost_usd(
        *, input_token_bound: int, token_budget: int,
        rates_usd_per_m: tuple[float, float]) -> float:
    """Bound one request at full input bound + full budget."""
    return ((input_token_bound / 1_000_000)
            * rates_usd_per_m[0]
            + (token_budget / 1_000_000)
            * rates_usd_per_m[1])


@dataclass
class _PreparedScaleBatch:
    sequence: int
    batch_id: str
    frames: list[FrameRef]
    visual_ids: list[str]
    image_paths: list[Path]
    source_sha256: list[str]
    prepared_sha256: list[str]
    prompt_text: str
    prompt_sha256: str


class _ScaleLedger:
    """Thread-safe request/cost guard for parallel batches."""

    def __init__(self, *, max_batches: int | None,
                 rates: tuple[float, float] | None,
                 ceiling: float | None,
                 strict: bool) -> None:
        self._max_batches = max_batches
        self._rates = rates
        self._ceiling = ceiling
        self._strict = strict
        self._lock = threading.Lock()
        self.requests = 0
        self.spent = 0.0
        self.unknown = 0
        self.stopped_reason: str | None = None
        self.stopped_detail: dict[str, Any] | None = None

    def pre_dispatch(self) -> str | None:
        """Veto dispatch after a stop was recorded."""
        with self._lock:
            if self.stopped_reason is not None:
                return self.stopped_reason
            if (self._max_batches is not None
                    and self.requests >= self._max_batches):
                self.stopped_reason = STOP_REQUEST_LIMIT
                self.stopped_detail = {
                    "requests": self.requests,
                    "max_batches": self._max_batches,
                }
                return self.stopped_reason
            self.requests += 1
            return None

    def note_usage(self, usage: InferenceUsage) -> bool:
        """Price one completed request; True when halted."""
        with self._lock:
            estimate = (
                estimate_cost(usage.input_tokens,
                              usage.output_tokens,
                              self._rates)
                if self._rates is not None else None)
            if estimate is None:
                self.unknown += 1
                if self._strict:
                    self.stopped_reason = (
                        STOP_STRICT_UNKNOWN_COST)
                    self.stopped_detail = {
                        "unknown_requests": self.unknown,
                    }
                    return True
                return False
            self.spent += estimate
            if (self._ceiling is not None
                    and self.spent > self._ceiling):
                self.stopped_reason = STOP_COST_CEILING
                self.stopped_detail = {
                    "spent_usd": self.spent,
                    "ceiling_usd": self._ceiling,
                }
                return True
            return False

    def snapshot(self) -> dict[str, Any]:
        """Return a consistent copy of guard counters."""
        with self._lock:
            return {
                "requests": self.requests,
                "spent": self.spent,
                "unknown": self.unknown,
                "stopped_reason": self.stopped_reason,
                "stopped_detail": self.stopped_detail,
            }


class DeterministicScaleFakeClient:
    """Deterministic offline double for scale batches.

    Returns schema-valid payloads derived from image file
    names (``anchored_<index>.png``). ``script`` overrides
    single batches by first frame index: a string becomes
    the raw reply text and an exception raises from
    ``submit``.
    """

    MODEL_ID = "deterministic-scale-fake-v1"

    def __init__(
            self,
            script: dict[int, str | Exception] | None = None,
    ) -> None:
        self._script = dict(script or {})
        self._lock = threading.Lock()
        self.calls: list[dict[str, Any]] = []

    @property
    def model_id(self) -> str:
        """Stable fake model identity for provenance."""
        return self.MODEL_ID

    def submit(self, prompt_text: str,
               image_paths: Sequence[Path],
               api_key: str, max_output_tokens: int,
               response_schema: dict[str, Any] | None = None,
               response_schema_name: str | None = None
               ) -> InferenceResponse:
        """Return one deterministic valid batch reply."""
        _ = (prompt_text, api_key, max_output_tokens,
             response_schema)
        paths = list(image_paths)
        with self._lock:
            self.calls.append({
                "images": len(paths),
                "schema": response_schema_name,
            })
        indices = [int(path.stem.split("_")[1])
                   for path in paths]
        override = self._script.get(indices[0])
        if isinstance(override, Exception):
            raise override
        if isinstance(override, str):
            text = override
        else:
            text = json.dumps({"frames": [
                {"frame_id": visual_frame_id_for(index),
                 "display_value": f"{index % 400}",
                 "unit": "g",
                 "confidence": "high"}
                for index in indices]})
        return InferenceResponse(
            model=self.MODEL_ID, status="completed",
            output_text=text,
            usage=InferenceUsage(
                input_tokens=700 * len(paths),
                output_tokens=40 * len(paths),
                total_tokens=740 * len(paths)),
            latency_seconds=0.5, request_id="fake-scale-1")


def _batch_id_for(sequence: int) -> str:
    return f"batch_{sequence:03d}"


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


class ScalePassRunner:
    """Parallel scale-reading execution with safe resume."""

    def __init__(self, config: ScalePassConfig) -> None:
        _validate_scale_config(config)
        self._config = config
        self._stop = threading.Event()

    def _paths(self) -> dict[str, Path]:
        root = self._config.out_dir
        return {
            "root": root,
            "raw": root / "raw_responses",
            "obs": root / "observations",
            "checkpoints": root / "checkpoints",
            "reports": root / "reports",
            "anchored": root / "anchored",
        }

    def worst_case_total_usd(self, batch_count: int) -> float:
        """Bound the whole run before any dispatch."""
        cfg = self._config
        if cfg.rates_usd_per_m is None:
            raise FirstPassError(
                "INVALID_CONFIG",
                "cost reservation requires rates.")
        single = worst_single_cost_usd(
            input_token_bound=cfg.input_token_bound,
            token_budget=cfg.token_budget,
            rates_usd_per_m=cfg.rates_usd_per_m)
        return round(single * batch_count, 4)

    def _prepare_batches(self) -> list[_PreparedScaleBatch]:
        cfg = self._config
        frames = sorted(cfg.frames,
                        key=lambda frame: frame.frame_index)
        indices = [frame.frame_index for frame in frames]
        if len(set(indices)) != len(indices):
            raise FirstPassError(
                "INVALID_CONFIG",
                "frame_index values must be unique.")
        paths = self._paths()
        manifest: list[dict[str, Any]] = []
        prepared: list[_PreparedScaleBatch] = []
        chunk: list[FrameRef] = []
        chunks: list[list[FrameRef]] = []
        for frame in frames:
            chunk.append(frame)
            if len(chunk) == cfg.group_size:
                chunks.append(chunk)
                chunk = []
        if chunk:
            chunks.append(chunk)
        for sequence, group in enumerate(chunks):
            visual_ids: list[str] = []
            image_paths: list[Path] = []
            source_sha: list[str] = []
            prepared_sha: list[str] = []
            for frame in group:
                payload = frame.image_path.read_bytes()
                if payload[:3] != b"\xff\xd8\xff" and (
                        payload[:4] != b"\x89PNG"):
                    raise FirstPassError(
                        "INVALID_IMAGE",
                        f"{frame.image_path.name} is not a "
                        f"JPEG/PNG payload.")
                record = prepare_anchored_image(
                    source=frame.image_path,
                    frame_index=frame.frame_index,
                    timestamp_seconds=frame.timestamp_seconds,
                    dest_dir=paths["anchored"],
                    salt=cfg.anchor_salt)
                if not verify_anchored_image(record):
                    raise FirstPassError(
                        "ANCHOR_INTEGRITY",
                        "prepared pixels differ from source "
                        f"for frame {frame.frame_index}.")
                manifest.append({
                    "frame_index": record.frame_index,
                    "timestamp_seconds":
                        record.timestamp_seconds,
                    "visual_frame_id": record.visual_frame_id,
                    "check_code": record.check_code,
                    "source_name": record.source_path.name,
                    "source_sha256": record.source_sha256,
                    "source_size": list(record.source_size),
                    "prepared_name": record.prepared_path.name,
                    "prepared_sha256": record.prepared_sha256,
                    "prepared_size": list(record.prepared_size),
                    "photo_region_verified": True,
                })
                visual_ids.append(record.visual_frame_id)
                image_paths.append(record.prepared_path)
                source_sha.append(record.source_sha256)
                prepared_sha.append(record.prepared_sha256)
            prompt_text = build_scale_prompt(
                [(frame_id, frame.timestamp_seconds)
                 for frame_id, frame in zip(
                     visual_ids, group)])
            prepared.append(_PreparedScaleBatch(
                sequence=sequence,
                batch_id=_batch_id_for(sequence),
                frames=list(group), visual_ids=visual_ids,
                image_paths=image_paths,
                source_sha256=source_sha,
                prepared_sha256=prepared_sha,
                prompt_text=prompt_text,
                prompt_sha256=prompt_hash(prompt_text)))
        (paths["anchored"] / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
        return prepared

    def _prior_batch_usable(
            self, batch: _PreparedScaleBatch,
            paths: dict[str, Path]) -> dict[str, Any] | None:
        """Adopt a completed batch record, else None.

        Only ``completed`` records whose frame set, prepared
        image hashes, and prompt hash all match the current
        plan are reused. Batches whose response was received
        but rejected are never silently rebilled: they fail
        the run explicitly in ``run``.
        """
        raw_path = paths["raw"] / f"{batch.batch_id}.json"
        obs_path = paths["obs"] / f"{batch.batch_id}.json"
        if not (raw_path.is_file() and obs_path.is_file()):
            return None
        try:
            raw = json.loads(raw_path.read_text(
                encoding="utf-8"))
            obs = json.loads(obs_path.read_text(
                encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict) or raw.get(
                "status") != "completed":
            return None
        if raw.get("frame_indices") != [
                frame.frame_index for frame in batch.frames]:
            return None
        if raw.get("prepared_sha256") != batch.prepared_sha256:
            return None
        if raw.get("prompt_sha256") != batch.prompt_sha256:
            return None
        if not isinstance(obs, list) or len(obs) != len(
                batch.frames):
            return None
        return raw

    def _prior_batch_received(
            self, batch: _PreparedScaleBatch,
            paths: dict[str, Path]) -> dict[str, Any] | None:
        """Return a prior received-but-unusable record, if any.

        A batch whose provider response was recorded
        (request id plus usage) must never be dispatched
        again: the ambiguous billing state fails the run
        explicitly instead of risking a duplicate charge.
        """
        raw_path = paths["raw"] / f"{batch.batch_id}.json"
        if not raw_path.is_file():
            return None
        try:
            raw = json.loads(raw_path.read_text(
                encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(raw, dict):
            return None
        if raw.get("status") == "completed":
            return None
        if (raw.get("request_id") is not None
                and raw.get("usage", {}).get("input_tokens")
                is not None):
            return raw
        return None

    def _run_batch(self, client: Any, batch: _PreparedScaleBatch,
                   ledger: _ScaleLedger, paths: dict[str, Path]
                   ) -> dict[str, Any]:
        """Dispatch one batch exactly once; never retry."""
        cfg = self._config
        started = time.perf_counter()
        veto = ledger.pre_dispatch()
        if veto is not None:
            return self._skipped_record(
                batch, veto, paths, started)
        try:
            result = client.submit(
                batch.prompt_text, batch.image_paths,
                cfg.api_key, cfg.token_budget,
                response_schema=SCALE_JSON_SCHEMA,
                response_schema_name=SCALE_SCHEMA_NAME)
        except InferenceError as exc:
            return self._transport_record(
                batch, paths, started, exc.code, str(exc),
                ledger)
        except Exception as exc:  # Never retry; record only.
            return self._transport_record(
                batch, paths, started, type(exc).__name__,
                str(exc)[:300], ledger)
        halted = ledger.note_usage(result.usage)
        if halted:
            self._stop.set()
        return self._completed_record(
            batch, result, paths, started, ledger, halted)

    def _skipped_record(self, batch: _PreparedScaleBatch,
                        reason: str, paths: dict[str, Path],
                        started: float) -> dict[str, Any]:
        cfg = self._config
        indices = [frame.frame_index
                   for frame in batch.frames]
        record = {
            "batch_id": batch.batch_id,
            "sequence": batch.sequence,
            "frame_indices": indices,
            "visual_frame_ids": list(batch.visual_ids),
            "prepared_sha256": list(batch.prepared_sha256),
            "prompt_sha256": batch.prompt_sha256,
            "status": "skipped",
            "terminal_state": TERMINAL_FAILED,
            "failure_reason": f"veto:{reason}",
            "request_id": None,
            "usage": {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "cached_input_tokens": None,
                "reasoning_output_tokens": None,
            },
            "latency_seconds": None,
            "model": (cfg.model if cfg.billable
                      else "deterministic-scale-fake-v1"),
            "wall_seconds": round(
                time.perf_counter() - started, 3),
        }
        _write_atomic(
            paths["raw"] / f"{batch.batch_id}.json",
            (json.dumps(record, indent=2) + "\n").encode(
                "utf-8"))
        _write_atomic(paths["obs"] / f"{batch.batch_id}.json",
                      b"[]\n")
        return {
            "batch_id": batch.batch_id,
            "sequence": batch.sequence,
            "frame_indices": indices,
            "status": "skipped",
            "terminal_state": TERMINAL_FAILED,
            "failure_reason": f"veto:{reason}",
            "frames_observed": 0,
            "frames_expected": len(indices),
            "submits": 0,
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_output_tokens": None,
            "latency_seconds": None,
            "request_id": None,
            "halted": True,
            "exhausted": False,
            "resumed": False,
        }

    def _transport_record(self, batch: _PreparedScaleBatch,
                          paths: dict[str, Path],
                          started: float, code: str,
                          message: str,
                          ledger: _ScaleLedger) -> dict[str, Any]:
        """Record a batch with no provider response.

        No usage was reported, so under a strict posture the
        run must stop; the batch itself carries no request
        id and may be attempted by an explicit fresh run.
        """
        cfg = self._config
        indices = [frame.frame_index
                   for frame in batch.frames]
        halted = False
        if cfg.strict_cost:
            snapshot = ledger.snapshot()
            if snapshot["stopped_reason"] is None:
                with ledger._lock:
                    ledger.stopped_reason = (
                        STOP_STRICT_UNKNOWN_COST)
                    ledger.stopped_detail = {
                        "batch_id": batch.batch_id,
                        "transport_code": code,
                    }
                    ledger.unknown += 1
            halted = True
            self._stop.set()
        record = {
            "batch_id": batch.batch_id,
            "sequence": batch.sequence,
            "frame_indices": indices,
            "visual_frame_ids": list(batch.visual_ids),
            "prepared_sha256": list(batch.prepared_sha256),
            "prompt_sha256": batch.prompt_sha256,
            "status": "transport_failed",
            "terminal_state": TERMINAL_FAILED,
            "failure_reason": f"transport:{code}:{message}",
            "request_id": None,
            "usage": {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "cached_input_tokens": None,
                "reasoning_output_tokens": None,
            },
            "latency_seconds": None,
            "model": cfg.model,
            "wall_seconds": round(
                time.perf_counter() - started, 3),
        }
        _write_atomic(
            paths["raw"] / f"{batch.batch_id}.json",
            (json.dumps(record, indent=2) + "\n").encode(
                "utf-8"))
        _write_atomic(paths["obs"] / f"{batch.batch_id}.json",
                      b"[]\n")
        return {
            "batch_id": batch.batch_id,
            "sequence": batch.sequence,
            "frame_indices": indices,
            "status": "transport_failed",
            "terminal_state": TERMINAL_FAILED,
            "failure_reason": f"transport:{code}",
            "frames_observed": 0,
            "frames_expected": len(indices),
            "submits": 1,
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_output_tokens": None,
            "latency_seconds": None,
            "request_id": None,
            "halted": halted,
            "exhausted": False,
            "resumed": False,
        }

    def _completed_record(self, batch: _PreparedScaleBatch,
                          result: InferenceResponse,
                          paths: dict[str, Path],
                          started: float, ledger: _ScaleLedger,
                          halted: bool) -> dict[str, Any]:
        cfg = self._config
        indices = [frame.frame_index
                   for frame in batch.frames]
        usage = {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "total_tokens": result.usage.total_tokens,
            "cached_input_tokens":
                result.usage.cached_input_tokens,
            "reasoning_output_tokens":
                result.usage.reasoning_output_tokens,
        }
        observations: list[dict[str, Any]] = []
        status = "invalid"
        terminal: str | None = None
        failure_reason: str | None = None
        if (result.status == "completed"
                and result.output_text.strip()):
            validation = validate_scale_response(
                result.output_text, batch.visual_ids)
            if validation.ok and validation.observations:
                status = "completed"
                terminal = TERMINAL_VALID
                observations = self._adapt_observations(
                    batch, validation.observations, result)
            else:
                status = "invalid"
                terminal = TERMINAL_INVALID_JSON
                failure_reason = (
                    f"validation:{validation.error}")
        else:
            status = "invalid"
            if is_token_limit_reason(
                    result.incomplete_reason):
                terminal = TERMINAL_EXHAUSTED
                failure_reason = (
                    "token_budget_exhausted:no_retry")
            elif not result.output_text.strip():
                terminal = TERMINAL_EMPTY
                failure_reason = "empty_output"
            else:
                terminal = TERMINAL_UNSUPPORTED
                failure_reason = (
                    f"status:{result.status}:"
                    f"{result.incomplete_reason}")
        record = {
            "batch_id": batch.batch_id,
            "sequence": batch.sequence,
            "frame_indices": indices,
            "visual_frame_ids": list(batch.visual_ids),
            "image_order": [path.name
                            for path in batch.image_paths],
            "prepared_sha256": list(batch.prepared_sha256),
            "prompt_sha256": batch.prompt_sha256,
            "incoming_context_kind": "none",
            "status": status,
            "terminal_state": terminal,
            "failure_reason": failure_reason,
            "request_id": result.request_id,
            "usage": usage,
            "latency_seconds": result.latency_seconds,
            "model": result.model,
            "response_status": result.status,
            "incomplete_reason": result.incomplete_reason,
            "output_text": result.output_text,
            "wall_seconds": round(
                time.perf_counter() - started, 3),
        }
        _write_atomic(
            paths["raw"] / f"{batch.batch_id}.json",
            (json.dumps(record, indent=2) + "\n").encode(
                "utf-8"))
        _write_atomic(
            paths["obs"] / f"{batch.batch_id}.json",
            (json.dumps(observations, indent=2) + "\n").encode(
                "utf-8"))
        return {
            "batch_id": batch.batch_id,
            "sequence": batch.sequence,
            "frame_indices": indices,
            "status": status,
            "terminal_state": terminal,
            "failure_reason": failure_reason,
            "frames_observed": len(observations),
            "frames_expected": len(indices),
            "submits": 1,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "reasoning_output_tokens": usage[
                "reasoning_output_tokens"],
            "latency_seconds": result.latency_seconds,
            "request_id": result.request_id,
            "halted": halted,
            "exhausted": terminal == TERMINAL_EXHAUSTED,
            "resumed": False,
        }

    def _adapt_observations(
            self, batch: _PreparedScaleBatch,
            validated: dict[str, dict[str, Any]],
            result: InferenceResponse) -> list[dict[str, Any]]:
        """Map scale readings onto the observation envelope.

        Field mapping (wire -> envelope ``model_fields``):
        ``display_value`` -> ``scale_display_verbatim``,
        readability -> ``scale_readable``, ``unit`` ->
        ``scale_unit_visible``, ``confidence`` unchanged.
        Contextual fields the scale pass never evaluates
        stay explicitly unknown/empty; nothing is invented.
        """
        cfg = self._config
        manifest_ref = {
            "manifest_sha256": cfg.source.sha256,
            "source_name": cfg.source.name,
        }
        observations: list[dict[str, Any]] = []
        for position, frame in enumerate(batch.frames):
            frame_id = batch.visual_ids[position]
            entry = validated[frame_id]
            value = entry["display_value"]
            readable = value is not None
            observations.append({
                "frame_index": frame.frame_index,
                "image_path": frame.image_path.name,
                "timestamp_seconds": frame.timestamp_seconds,
                "source_timestamp_seconds":
                    frame.source_timestamp_seconds,
                "requested_timestamp_seconds":
                    frame.requested_timestamp_seconds,
                "timestamp_basis": TIMESTAMP_BASIS,
                "manifest_ref": manifest_ref,
                "batch_id": batch.batch_id,
                "pass_id": cfg.pass_id,
                "call": {
                    "provider": cfg.provider,
                    "model": cfg.model,
                    "prompt_sha256": batch.prompt_sha256,
                    "group_id": batch.batch_id,
                    "group_sequence": batch.sequence,
                    "image_position": position,
                    "group_size": len(batch.frames),
                    "incoming_context_kind": "none",
                    "max_output_tokens": cfg.token_budget,
                    "store": None,
                    "timeout_seconds": cfg.timeout_seconds,
                    "input_tokens": result.usage.input_tokens,
                    "output_tokens":
                        result.usage.output_tokens,
                    "reasoning_output_tokens":
                        result.usage.reasoning_output_tokens,
                    "latency_seconds": result.latency_seconds,
                    "request_id": result.request_id,
                    "attempts": 1,
                    "response_status": result.status,
                    "incomplete_reason":
                        result.incomplete_reason,
                    "terminal_state": (
                        TERMINAL_ABSTAINED if not readable
                        else TERMINAL_VALID),
                },
                "model_fields": {
                    "scale_readable": readable,
                    "scale_display_verbatim": value,
                    "scale_unit_visible": entry["unit"],
                    "scale_state": "unknown",
                    "tare_action_visible": False,
                    "container_description": None,
                    "container_changed_hint": False,
                    "primary_utensil": None,
                    "primary_visible_material": None,
                    "action": None,
                    "addition_or_removal_visible": "unknown",
                    "confidence": entry["confidence"],
                    "ambiguity_notes": None,
                    "visible_objects": [],
                    "visual_frame_id": frame_id,
                },
                "model_timestamp_seconds":
                    frame.timestamp_seconds,
                "parse_error": None,
                "raw_text_excerpt": json.dumps(
                    {frame_id: entry})[:500],
                "response_status": result.status,
                "incomplete_reason": result.incomplete_reason,
                "terminal_state": (
                    TERMINAL_ABSTAINED if not readable
                    else TERMINAL_VALID),
                "image_ref": {
                    "sha256": batch.source_sha256[position],
                    "encoding": "png",
                    "prepared_sha256":
                        batch.prepared_sha256[position],
                },
            })
        return observations

    def run(self) -> dict[str, Any]:
        """Execute all batches with bounded concurrency."""
        cfg = self._config
        paths = self._paths()
        for key in ("raw", "obs", "checkpoints", "reports"):
            paths[key].mkdir(parents=True, exist_ok=True)
        prepared = self._prepare_batches()
        if (cfg.max_batches is not None
                and len(prepared) > cfg.max_batches):
            raise FirstPassError(
                "INVALID_CONFIG",
                f"{len(prepared)} batches exceed the "
                f"max_batches cap {cfg.max_batches}.")
        if (cfg.billable and cfg.cost_ceiling_usd is not None
                and cfg.rates_usd_per_m is not None):
            reservation = self.worst_case_total_usd(
                len(prepared))
            if reservation > cfg.cost_ceiling_usd:
                raise FirstPassError(
                    "COST_RESERVE",
                    f"worst-case reservation ${reservation} "
                    f"exceeds the ${cfg.cost_ceiling_usd} "
                    f"ceiling for {len(prepared)} batches; "
                    f"refusing dispatch.")
        ledger = _ScaleLedger(
            max_batches=cfg.max_batches,
            rates=cfg.rates_usd_per_m,
            ceiling=cfg.cost_ceiling_usd,
            strict=cfg.strict_cost)
        summaries: dict[int, dict[str, Any]] = {}
        fatal: list[str] = []
        todo: list[_PreparedScaleBatch] = []
        for batch in prepared:
            reused = self._prior_batch_usable(batch, paths)
            if reused is not None:
                summaries[batch.sequence] = (
                    self._resumed_summary(batch, reused))
                continue
            received = self._prior_batch_received(
                batch, paths)
            if received is not None:
                fatal.append(batch.batch_id)
                continue
            todo.append(batch)
        if fatal:
            raise FirstPassError(
                "PRIOR_UNUSABLE_RESPONSE",
                f"batches {fatal} already have a recorded "
                f"provider response that failed validation; "
                f"refusing to rebill. Submit a fresh run.")
        if todo:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=cfg.max_concurrency
                    ) as pool:
                future_of = {
                    pool.submit(self._dispatch_one, batch,
                                ledger, paths): batch
                    for batch in todo
                }
                for future in concurrent.futures.as_completed(
                        future_of):
                    summary = future.result()
                    summaries[summary["sequence"]] = summary
        ordered = [summaries[batch.sequence]
                   for batch in prepared]
        return self._write_report(prepared, ordered, paths,
                                  ledger)

    def _dispatch_one(self, batch: _PreparedScaleBatch,
                      ledger: _ScaleLedger,
                      paths: dict[str, Path]
                      ) -> dict[str, Any]:
        """Dispatch a single batch unless stopped."""
        cfg = self._config
        assert cfg.client_factory is not None
        if self._stop.is_set():
            return self._skipped_record(
                batch, "run_stopped", paths,
                time.perf_counter())
        client = cfg.client_factory()
        try:
            return self._run_batch(client, batch, ledger,
                                   paths)
        except FirstPassError:
            raise
        except Exception as exc:  # Record, never propagate.
            return self._transport_record(
                batch, paths, time.perf_counter(),
                type(exc).__name__, str(exc)[:300], ledger)

    def _resumed_summary(self, batch: _PreparedScaleBatch,
                         raw: dict[str, Any]
                         ) -> dict[str, Any]:
        indices = [frame.frame_index
                   for frame in batch.frames]
        usage = raw.get("usage") or {}
        return {
            "batch_id": batch.batch_id,
            "sequence": batch.sequence,
            "frame_indices": indices,
            "status": "completed",
            "terminal_state": TERMINAL_VALID,
            "failure_reason": None,
            "frames_observed": len(indices),
            "frames_expected": len(indices),
            "submits": 0,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "reasoning_output_tokens": usage.get(
                "reasoning_output_tokens"),
            "latency_seconds": raw.get("latency_seconds"),
            "request_id": raw.get("request_id"),
            "halted": False,
            "exhausted": False,
            "resumed": True,
        }

    def _write_report(
            self, prepared: list[_PreparedScaleBatch],
            ordered: list[dict[str, Any]],
            paths: dict[str, Path],
            ledger: _ScaleLedger) -> dict[str, Any]:
        cfg = self._config
        snapshot = ledger.snapshot()
        indices = [frame.frame_index
                   for batch in prepared
                   for frame in batch.frames]
        frames_observed = sum(
            item["frames_observed"] for item in ordered)
        frames_failed = len(indices) - frames_observed
        states: dict[str, int] = {}
        for item in ordered:
            terminal = item["terminal_state"] or "unknown"
            states[terminal] = states.get(terminal, 0) + 1
        valid = sum(1 for item in ordered
                    if item["status"] == "completed")
        review = sum(1 for item in ordered
                     if item["status"] == "invalid")
        failed = sum(1 for item in ordered
                     if item["status"] in (
                         "transport_failed", "skipped"))
        input_tokens = sum(
            item["input_tokens"] or 0 for item in ordered)
        output_tokens = sum(
            item["output_tokens"] or 0 for item in ordered)
        reasoning_tokens = sum(
            item["reasoning_output_tokens"] or 0
            for item in ordered)
        cost = (estimate_cost(input_tokens, output_tokens,
                              cfg.rates_usd_per_m)
                if cfg.rates_usd_per_m is not None else None)
        stopped = snapshot["stopped_reason"]
        if stopped is None and frames_failed:
            failed_ids = [item["batch_id"] for item in ordered
                          if item["frames_observed"]
                          < item["frames_expected"]]
            stopped = "batches_incomplete"
        else:
            failed_ids = []
        stopped_detail: dict[str, Any] | None = snapshot[
            "stopped_detail"]
        if failed_ids:
            stopped_detail = {
                "incomplete_batches": failed_ids,
                "frames_observed": frames_observed,
                "frames_total": len(indices),
            }
        report = {
            "experiment": SCALE_PASS_ID,
            "pass_id": cfg.pass_id,
            "model": cfg.model,
            "provider": cfg.provider,
            "source_name": cfg.source.name,
            "source_sha256": cfg.source.sha256,
            "frame_indices": indices,
            "group_size": cfg.group_size,
            "max_concurrency": cfg.max_concurrency,
            "token_budget_per_batch": cfg.token_budget,
            "groups_total": len(ordered),
            "groups_valid": valid,
            "groups_review": review,
            "groups_exhausted": failed,
            "groups_resumed": sum(
                1 for item in ordered if item["resumed"]),
            "batches_completed": valid,
            "frames_total": len(indices),
            "frames_observed": frames_observed,
            "frames_failed": frames_failed,
            "frames_resumed": sum(
                item["frames_observed"] for item in ordered
                if item["resumed"]),
            "requests_sent": snapshot["requests"],
            "requests_total": snapshot["requests"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_output_tokens": reasoning_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost_usd_estimated": cost,
            "stopped_reason": stopped,
            "stopped_detail": stopped_detail,
            "terminal_states": states,
            "groups": ordered,
        }
        report_path = paths["reports"] / "run_report.json"
        _write_atomic(report_path, (
            json.dumps(report, indent=2) + "\n").encode(
                "utf-8"))
        state = {
            "frame_indices": indices,
            "batches": [item["batch_id"] for item in ordered],
        }
        _write_atomic(
            paths["checkpoints"] / "run_state.json",
            (json.dumps(state, indent=2) + "\n").encode(
                "utf-8"))
        return report
