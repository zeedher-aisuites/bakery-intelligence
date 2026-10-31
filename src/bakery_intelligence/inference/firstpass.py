"""Video-independent first-pass multimodal inference (M2-06C.2).

Generalizes the proven Exp. 006 runner pattern to arbitrary frame
lists: configurable frames, batches, budgets, and destinations
with per-batch raw/observation artifacts, append-only
checkpoints, resume, and lifecycle terminal states. The
experiment modules stay frozen as the P04 reference; this package
is self-contained because experiment directories are not
importable packages, so shared policy is mirrored here with an
explicit reference comment at each mirror point.

Safety properties: the module performs no network access itself
and wires no live client — inference flows only through an
injected client factory, so automated tests run fully offline.
Live provider wiring arrives with the C.5 authorization
boundary, not here.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

# Terminal vocabulary, mirrored from
# experiments/006_p04_multimodal_inference/lifecycle.py.
TERMINAL_VALID = "valid_prediction"
TERMINAL_INVALID_JSON = "invalid_json"
TERMINAL_EMPTY = "empty_output"
TERMINAL_EXHAUSTED = "exhausted_token_budget"
TERMINAL_FAILED = "request_failed"
TERMINAL_UNSUPPORTED = "unsupported_response"
TERMINAL_ABSTAINED = "abstained"

TERMINAL_STATES = (
    TERMINAL_VALID,
    TERMINAL_INVALID_JSON,
    TERMINAL_EMPTY,
    TERMINAL_EXHAUSTED,
    TERMINAL_FAILED,
    TERMINAL_UNSUPPORTED,
    TERMINAL_ABSTAINED,
)

# Model field schema, mirrored from
# experiments/006_p04_multimodal_inference/prompts.py
# (MODEL_OUTPUT_FIELDS). Downstream temporal analysis requires
# the scale/container/confidence subset of these keys.
MODEL_OUTPUT_FIELDS = (
    "visible_objects",
    "scale_display_verbatim",
    "scale_readable",
    "scale_state",
    "tare_action_visible",
    "addition_or_removal_visible",
    "container_description",
    "container_changed_hint",
    "confidence",
    "ambiguity_notes",
)

# Timestamp method label, mirrored from prompts.py. C.3 extraction
# defines the same basis (index-derived timestamps).
TIMESTAMP_BASIS = "frame_index_over_fps"

# Transport policy, mirrored from experiments/006 runner.py.
RETRYABLE_CODES = frozenset({
    "RATE_LIMITED", "TIMEOUT", "NETWORK_ERROR", "SERVER_ERROR",
})
ABORT_CODES = frozenset({"CREDENTIAL_MISSING", "AUTH_FAILED"})

DEFAULT_BATCH_SIZE = 5
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BACKOFF_BASE_SECONDS = 0.5
DEFAULT_BACKOFF_CAP_SECONDS = 30.0
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_BUDGETS = (1024,)
DEFAULT_JITTER_SEED = 20261028
MAX_IMAGES_PER_REQUEST = 50

JPEG_START = b"\xff\xd8"
JPEG_END = b"\xff\xd9"
PNG_START = b"\x89PNG"


class FirstPassError(Exception):
    """Structured runner failure with a machine-readable code."""

    def __init__(self, code: str, message: str,
                 entity_id: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.entity_id = entity_id


class InferenceError(Exception):
    """Structured inference-client failure (mirrors ProviderError)."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class InferenceUsage:
    """Reported token usage; all fields optional (never invented)."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None


@dataclass(frozen=True)
class InferenceResponse:
    """Normalized outcome of one multimodal request."""

    model: str
    status: str
    output_text: str
    usage: InferenceUsage = field(default_factory=InferenceUsage)
    latency_seconds: float = 0.0
    request_id: str | None = None
    incomplete_reason: str | None = None


class InferenceClient(Protocol):
    """Client seam modeled on MetaClient.submit.

    The token budget travels as an explicit per-request parameter
    instead of 006-style mutable client state, so escalation is
    observable and testable at the seam.
    """

    def submit(
        self,
        prompt_text: str,
        image_paths: Sequence[Path],
        api_key: str,
        max_output_tokens: int,
    ) -> InferenceResponse:
        """Send one multimodal request for a single frame."""
        ...


@dataclass(frozen=True)
class FrameRef:
    """One frame to analyze, with caller-supplied timestamps."""

    frame_index: int
    image_path: Path
    timestamp_seconds: float
    source_timestamp_seconds: float | None = None
    requested_timestamp_seconds: float | None = None

    def resolved_source(self) -> float:
        """Fall back to the frame timestamp when unset."""
        return (self.timestamp_seconds
                if self.source_timestamp_seconds is None
                else self.source_timestamp_seconds)

    def resolved_requested(self) -> float:
        """Fall back to the frame timestamp when unset."""
        return (self.timestamp_seconds
                if self.requested_timestamp_seconds is None
                else self.requested_timestamp_seconds)


@dataclass(frozen=True)
class SourceRef:
    """Integrity anchor for the analyzed frame source."""

    name: str
    sha256: str


@dataclass
class FirstPassConfig:
    """Complete runner configuration (explicit, no hidden paths)."""

    frames: list[FrameRef]
    source: SourceRef
    out_dir: Path
    model: str
    prompt_text: str
    pass_id: str = "firstpass"
    provider: str = "meta"
    batch_size: int = DEFAULT_BATCH_SIZE
    token_budgets: tuple[int, ...] = DEFAULT_BUDGETS
    store: bool = True
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS
    backoff_cap_seconds: float = DEFAULT_BACKOFF_CAP_SECONDS
    jitter_seed: int = DEFAULT_JITTER_SEED
    rates_usd_per_m: tuple[float, float] | None = None
    api_key: str = ""
    client_factory: Callable[[], InferenceClient] | None = None
    sleeper: Callable[[float], None] = time.sleep


@dataclass(frozen=True)
class BudgetPolicy:
    """Output-token escalation schedule (mirrors lifecycle.py)."""

    budgets: tuple[int, ...] = DEFAULT_BUDGETS

    def __post_init__(self) -> None:
        if not self.budgets or any(
            not isinstance(budget, int) or isinstance(budget, bool)
            or budget <= 0 for budget in self.budgets
        ):
            raise ValueError(
                "budgets must be a non-empty tuple of positive ints."
            )


def is_token_limit_reason(reason: Any) -> bool:
    """True only for an explicit output-token-limit reason."""
    if reason is None:
        return False
    text = str(reason).lower()
    return "max_output_tokens" in text or (
        "output_token" in text
        and ("limit" in text or "max" in text or "exhaust" in text)
    )


def decide_attempt(status: str | None, incomplete_reason: Any,
                   raw_text: Any, parse_ok: bool,
                   attempts_made: int, policy: BudgetPolicy,
                   abstained_hint: bool = False,
                   transport_error: str | None = None
                   ) -> dict[str, Any]:
    """Map one attempt outcome to retry/stop plus terminal state.

    Mirrors lifecycle.decide_attempt: retries only on explicit
    token exhaustion with budgets left.
    """
    if transport_error is not None:
        return {"action": "stop", "terminal": TERMINAL_FAILED,
                "reason": f"transport failure ({transport_error}); "
                          "no token escalation"}
    budgets_left = attempts_made < len(policy.budgets)
    if status == "incomplete":
        if is_token_limit_reason(incomplete_reason):
            if budgets_left:
                return {"action": "retry", "terminal": None,
                        "reason": "incomplete: token budget exhausted; "
                                  "escalating to next budget"}
            return {"action": "stop", "terminal": TERMINAL_EXHAUSTED,
                    "reason": "all token budgets exhausted"}
        return {"action": "stop", "terminal": TERMINAL_UNSUPPORTED,
                "reason": "incomplete with non-token reason "
                          f"{incomplete_reason!r}; cause not inferred"}
    if status != "completed":
        return {"action": "stop", "terminal": TERMINAL_UNSUPPORTED,
                "reason": f"unrecognized status {status!r}"}
    text = raw_text if isinstance(raw_text, str) else ""
    if not text.strip():
        if is_token_limit_reason(incomplete_reason):
            if budgets_left:
                return {"action": "retry", "terminal": None,
                        "reason": "empty output with explicit token-limit "
                                  "reason; escalating"}
            return {"action": "stop", "terminal": TERMINAL_EXHAUSTED,
                    "reason": "all token budgets exhausted"}
        return {"action": "stop", "terminal": TERMINAL_EMPTY,
                "reason": "completed with empty output; no token-limit "
                          "reason present"}
    if not parse_ok:
        return {"action": "stop", "terminal": TERMINAL_INVALID_JSON,
                "reason": "completed but output is not valid JSON; "
                          "content alone never triggers a retry"}
    if abstained_hint:
        return {"action": "stop", "terminal": TERMINAL_ABSTAINED,
                "reason": "completed valid response abstains from reading"}
    return {"action": "stop", "terminal": TERMINAL_VALID,
            "reason": "completed valid structured response accepted"}


def estimate_cost(input_tokens: Any, output_tokens: Any,
                  rates: tuple[float, float] | None,
                  ndigits: int = 4) -> float | None:
    """None-safe cost estimate (mirrors lifecycle.estimate_cost)."""
    if rates is None:
        return None
    if not isinstance(input_tokens, (int, float)) or not isinstance(
            output_tokens, (int, float)):
        return None
    if isinstance(input_tokens, bool) or isinstance(output_tokens, bool):
        return None
    return round(
        input_tokens * rates[0] / 1_000_000
        + output_tokens * rates[1] / 1_000_000,
        ndigits,
    )


def prompt_hash(prompt_text: str) -> str:
    """SHA-256 of prompt text (mirrors prompts.prompt_hash)."""
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def parse_model_json(raw_text: str
                     ) -> tuple[dict[str, Any] | None, str | None]:
    """Parse strict model JSON; never invent fields on failure."""
    try:
        payload = json.loads(raw_text)
    except (json.JSONDecodeError, ValueError):
        return None, "unparseable"
    if not isinstance(payload, dict):
        return None, "not_an_object"
    return payload, None


def build_observation(
    frame: FrameRef,
    source: SourceRef,
    batch_id: str,
    pass_id: str,
    call: dict[str, Any],
    raw_text: str,
    response_status: str | None = None,
    incomplete_reason: str | None = None,
    terminal_state: str | None = None,
    image_sha256: str | None = None,
    image_encoding: str | None = None,
) -> dict[str, Any]:
    """Assemble one timestamp-grounded prediction record.

    Mirrors prompts.build_observation key-for-key so outputs stay
    consumable by the existing temporal and reconstruction
    loaders. ``manifest_ref`` carries the caller-supplied source
    anchor; downstream readers require its ``manifest_sha256``.
    """
    parsed, parse_error = parse_model_json(raw_text)
    model_fields: dict[str, Any] = {
        name: None for name in MODEL_OUTPUT_FIELDS
    }
    if parsed is not None:
        for name in MODEL_OUTPUT_FIELDS:
            if name in parsed:
                model_fields[name] = parsed[name]
    return {
        "frame_index": frame.frame_index,
        "image_path": frame.image_path.name,
        "timestamp_seconds": frame.timestamp_seconds,
        "source_timestamp_seconds": frame.resolved_source(),
        "requested_timestamp_seconds": frame.resolved_requested(),
        "timestamp_basis": TIMESTAMP_BASIS,
        "manifest_ref": {
            "manifest_sha256": source.sha256,
            "source_name": source.name,
        },
        "batch_id": batch_id,
        "pass_id": pass_id,
        "call": call,
        "model_fields": model_fields,
        "parse_error": parse_error,
        "raw_text_excerpt": raw_text[:500],
        "response_status": response_status,
        "incomplete_reason": incomplete_reason,
        "terminal_state": terminal_state,
        "image_ref": {
            "sha256": image_sha256,
            "encoding": image_encoding,
        },
    }


def _detect_encoding(blob: bytes) -> str | None:
    if len(blob) >= 4 and blob[:2] == JPEG_START:
        if blob[-2:] == JPEG_END:
            return "jpeg"
        return None
    if len(blob) >= 4 and blob[:4] == PNG_START:
        return "png"
    return None


def _write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically via temp file plus replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    tmp.replace(path)


@dataclass
class _PreparedFrame:
    frame: FrameRef
    image_sha256: str
    image_encoding: str


class FirstPassRunner:
    """Resumable first-pass inference over explicit frame lists.

    Mirrors the Exp. 006 runner flow: upfront validation of every
    frame, contiguous batches, per-batch raw/observation files,
    append-only JSONL checkpoints, atomic run state, and a final
    report. Resume skips batches checkpointed ``completed``;
    batches marked ``needs_human_review`` re-run as a unit.
    """

    def __init__(self, config: FirstPassConfig) -> None:
        self._config = config

    def _paths(self) -> dict[str, Path]:
        root = self._config.out_dir
        return {
            "root": root,
            "raw": root / "raw_responses",
            "obs": root / "observations",
            "checkpoints": root / "checkpoints" / "checkpoints.jsonl",
            "state": root / "checkpoints" / "run_state.json",
            "reports": root / "reports",
        }

    def _backoff_delay(self, attempt: int, rng: random.Random) -> float:
        delay = self._config.backoff_base_seconds * (
            2.0 ** (attempt - 1))
        delay = min(delay, self._config.backoff_cap_seconds)
        return delay + rng.uniform(0.0, delay * 0.25)

    def _load_checkpoint_index(self, path: Path) -> dict[str, Any]:
        completed: dict[str, Any] = {}
        if not path.exists():
            return completed
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            completed[record["batch_id"]] = record
        return completed

    def _prepare_frames(self) -> list[_PreparedFrame]:
        cfg = self._config
        if not cfg.frames:
            raise FirstPassError(
                "INVALID_CONFIG", "Frame list is empty.")
        if cfg.batch_size < 1:
            raise FirstPassError(
                "INVALID_CONFIG", "Batch size must be >= 1.")
        if cfg.client_factory is None:
            raise FirstPassError(
                "INVALID_CONFIG",
                "No inference client wired: client_factory is None.")
        ordered = sorted(
            cfg.frames,
            key=lambda item: (item.timestamp_seconds,
                              item.frame_index),
        )
        seen: set[int] = set()
        prepared: list[_PreparedFrame] = []
        for frame in ordered:
            if frame.frame_index in seen:
                raise FirstPassError(
                    "DUPLICATE_FRAME",
                    f"Duplicate frame_index: {frame.frame_index}.",
                    str(frame.frame_index),
                )
            seen.add(frame.frame_index)
            try:
                blob = frame.image_path.read_bytes()
            except OSError as exc:
                raise FirstPassError(
                    "IMAGE_MISSING",
                    f"Cannot read image for frame "
                    f"{frame.frame_index}.",
                    str(frame.frame_index),
                ) from exc
            encoding = _detect_encoding(blob)
            if encoding is None:
                raise FirstPassError(
                    "IMAGE_MISSING",
                    f"Unreadable image for frame {frame.frame_index}: "
                    "expected JPEG or PNG magic bytes.",
                    str(frame.frame_index),
                )
            prepared.append(_PreparedFrame(
                frame=frame,
                image_sha256=hashlib.sha256(blob).hexdigest(),
                image_encoding=encoding,
            ))
        return prepared

    def _transport_attempts(
        self,
        client: InferenceClient,
        prompt_text: str,
        image: Path,
        api_key: str,
        rng: random.Random,
        budget: int,
        first_number: int,
    ) -> tuple[InferenceResponse | None, list[dict[str, Any]],
                str | None]:
        attempts: list[dict[str, Any]] = []
        for step in range(self._config.max_attempts):
            attempt = first_number + step
            try:
                result = client.submit(
                    prompt_text, [image], api_key, budget)
            except InferenceError as exc:
                attempts.append({
                    "attempt": attempt, "budget": budget,
                    "code": exc.code, "message": exc.message,
                })
                if exc.code in ABORT_CODES:
                    raise
                if (exc.code in RETRYABLE_CODES
                        and step + 1 < self._config.max_attempts):
                    wait = exc.retry_after_seconds
                    if wait is None:
                        wait = self._backoff_delay(step + 1, rng)
                    self._config.sleeper(wait)
                    attempts[-1]["slept_seconds"] = wait
                    continue
                return None, attempts, exc.code
            attempts.append({"attempt": attempt, "budget": budget,
                             "code": "ok"})
            return result, attempts, None
        return None, attempts, "RETRIES_EXHAUSTED"

    def _request_with_retry(
        self,
        client: InferenceClient,
        prompt_text: str,
        image: Path,
        api_key: str,
        rng: random.Random,
        policy: BudgetPolicy,
    ) -> tuple[InferenceResponse | None, list[dict[str, Any]],
                str | None, dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        budgets_used: list[int] = []
        superseded: list[dict[str, int | None]] = []
        transport_retries = 0
        for position, budget in enumerate(policy.budgets):
            first_number = len(attempts) + 1
            result, tried, error = self._transport_attempts(
                client, prompt_text, image, api_key, rng,
                budget, first_number,
            )
            attempts.extend(tried)
            budgets_used.append(budget)
            transport_retries += max(0, len(tried) - 1)
            if error is not None or result is None:
                return None, attempts, error, {
                    "terminal": TERMINAL_FAILED,
                    "incomplete_reason": None,
                    "budgets_used": budgets_used,
                    "transport_retries": transport_retries,
                    "superseded_usage": superseded,
                }
            if result.status == "completed":
                return result, attempts, None, {
                    "terminal": None,
                    "incomplete_reason": result.incomplete_reason,
                    "budgets_used": budgets_used,
                    "transport_retries": transport_retries,
                    "superseded_usage": superseded,
                }
            decision = decide_attempt(
                result.status, result.incomplete_reason,
                result.output_text, False, position + 1, policy,
            )
            if decision["action"] == "retry":
                superseded.append({
                    "input_tokens": result.usage.input_tokens,
                    "output_tokens": result.usage.output_tokens,
                })
                continue
            return None, attempts, "INCOMPLETE_STATUS", {
                "terminal": decision["terminal"],
                "incomplete_reason": result.incomplete_reason,
                "budgets_used": budgets_used,
                "transport_retries": transport_retries,
                "superseded_usage": superseded,
            }
        return None, attempts, "INCOMPLETE_STATUS", {
            "terminal": TERMINAL_EXHAUSTED,
            "incomplete_reason": None,
            "budgets_used": budgets_used,
            "transport_retries": transport_retries,
            "superseded_usage": superseded,
        }

    def _run_batch(
        self,
        client: InferenceClient,
        prepared: list[_PreparedFrame],
        batch_id: str,
        prompt_digest: str,
        api_key: str,
        rng: random.Random,
        paths: dict[str, Path],
        policy: BudgetPolicy,
    ) -> dict[str, Any]:
        cfg = self._config
        raw_records: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []
        attempts_log: list[dict[str, Any]] = []
        requests_sent = retries = 0
        input_tokens = output_tokens = 0
        batch_failed = False
        terminal_states: dict[str, int] = {}
        token_retries = 0
        token_retry_input_tokens = 0
        token_retry_output_tokens = 0
        token_retried_requests = 0
        token_recovered = 0
        for item in prepared:
            result, attempts, error, info = self._request_with_retry(
                client, cfg.prompt_text, item.frame.image_path,
                api_key, rng, policy,
            )
            requests_sent += 1
            retries += info["transport_retries"]
            budgets_used = info["budgets_used"]
            token_retries += max(0, len(budgets_used) - 1)
            for usage in info["superseded_usage"]:
                if usage["input_tokens"] is not None:
                    token_retry_input_tokens += usage["input_tokens"]
                if usage["output_tokens"] is not None:
                    token_retry_output_tokens += usage["output_tokens"]
            if len(budgets_used) > 1:
                token_retried_requests += 1
            attempts_log.append({
                "frame_indices": [item.frame.frame_index],
                "attempts": attempts,
            })
            if error is not None or result is None:
                batch_failed = True
                terminal_states[info["terminal"]] = (
                    terminal_states.get(info["terminal"], 0) + 1
                )
                raw_records.append({
                    "frame_indices": [item.frame.frame_index],
                    "error_code": error, "attempts": attempts,
                    "terminal_state": info["terminal"],
                    "incomplete_reason": info["incomplete_reason"],
                    "budgets_attempted": budgets_used,
                })
                continue
            if result.usage.input_tokens is not None:
                input_tokens += result.usage.input_tokens
            if result.usage.output_tokens is not None:
                output_tokens += result.usage.output_tokens
            parsed, _ = parse_model_json(result.output_text)
            abstained_hint = (
                parsed is not None
                and parsed.get("scale_readable") is False
            )
            terminal = decide_attempt(
                result.status, result.incomplete_reason,
                result.output_text, parsed is not None,
                len(budgets_used), policy, abstained_hint,
            )["terminal"]
            terminal_states[terminal] = (
                terminal_states.get(terminal, 0) + 1
            )
            if terminal == TERMINAL_VALID and len(budgets_used) > 1:
                token_recovered += 1
            raw_records.append({
                "frame_indices": [item.frame.frame_index],
                "model": result.model, "status": result.status,
                "output_text": result.output_text,
                "usage": {
                    "input_tokens": result.usage.input_tokens,
                    "output_tokens": result.usage.output_tokens,
                    "total_tokens": result.usage.total_tokens,
                },
                "latency_seconds": result.latency_seconds,
                "request_id": result.request_id,
                "attempts": attempts,
                "terminal_state": terminal,
                "incomplete_reason": result.incomplete_reason,
                "budgets_attempted": budgets_used,
            })
            call = {
                "provider": cfg.provider, "model": result.model,
                "prompt_sha256": prompt_digest,
                "review_of_sha256": None,
                "max_output_tokens": budgets_used[-1],
                "token_budgets_attempted": budgets_used,
                "store": cfg.store,
                "timeout_seconds": cfg.timeout_seconds,
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "reasoning_output_tokens":
                    result.usage.reasoning_output_tokens,
                "latency_seconds": result.latency_seconds,
                "request_id": result.request_id,
                "attempts": len(attempts),
                "response_status": result.status,
                "incomplete_reason": result.incomplete_reason,
                "terminal_state": terminal,
            }
            observations.append(build_observation(
                item.frame, cfg.source, batch_id, cfg.pass_id,
                call, result.output_text,
                response_status=result.status,
                incomplete_reason=result.incomplete_reason,
                terminal_state=terminal,
                image_sha256=item.image_sha256,
                image_encoding=item.image_encoding,
            ))
        status = "needs_human_review" if batch_failed else "completed"
        raw_path = paths["raw"] / f"{batch_id}.json"
        obs_path = paths["obs"] / f"{batch_id}.json"
        _write_json(raw_path, raw_records)
        _write_json(obs_path, observations)
        record = {
            "batch_id": batch_id,
            "index_range": [
                prepared[0].frame.frame_index,
                prepared[-1].frame.frame_index,
            ],
            "frame_indices": [item.frame.frame_index
                              for item in prepared],
            "status": status,
            "raw_path": raw_path.name,
            "obs_path": obs_path.name,
            "attempts": attempts_log,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        with paths["checkpoints"].open(
                "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return {
            "status": status, "requests_sent": requests_sent,
            "retries": retries, "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "terminal_states": terminal_states,
            "token_retries": token_retries,
            "token_retry_input_tokens": token_retry_input_tokens,
            "token_retry_output_tokens": token_retry_output_tokens,
            "token_retried_requests": token_retried_requests,
            "token_recovered": token_recovered,
            "frames_observed": len(observations),
            "frames_failed": len(prepared) - len(observations),
        }

    def run(self) -> dict[str, Any]:
        """Execute all batches with resume; return the run report."""
        cfg = self._config
        try:
            policy = BudgetPolicy(budgets=tuple(cfg.token_budgets))
        except ValueError as exc:
            raise FirstPassError(
                "INVALID_CONFIG", str(exc)) from exc
        if cfg.client_factory is None:
            raise FirstPassError(
                "INVALID_CONFIG",
                "No inference client wired: client_factory is None.")
        prepared = self._prepare_frames()
        paths = self._paths()
        for key in ("raw", "obs", "reports"):
            paths[key].mkdir(parents=True, exist_ok=True)
        paths["checkpoints"].parent.mkdir(parents=True, exist_ok=True)
        completed = self._load_checkpoint_index(paths["checkpoints"])
        prompt_digest = prompt_hash(cfg.prompt_text)
        assert cfg.client_factory is not None
        client = cfg.client_factory()
        rng = random.Random(cfg.jitter_seed)
        stats: dict[str, Any] = {
            "batches_total": 0,
            "batches_completed": sum(
                1 for record in completed.values()
                if record.get("status") == "completed"
            ),
            "requests_sent": 0, "retries": 0,
            "input_tokens": 0, "output_tokens": 0,
            "needs_review": 0,
            "terminal_states": {},
            "token_retries": 0,
            "token_retry_input_tokens": 0,
            "token_retry_output_tokens": 0,
            "token_retried_requests": 0,
            "token_recovered": 0,
            "frames_total": len(prepared),
            "frames_observed": 0,
            "frames_failed": 0,
        }
        self._check_prior_state(paths["state"], prepared)
        for start in range(0, len(prepared), cfg.batch_size):
            span = prepared[start:start + cfg.batch_size]
            stats["batches_total"] += 1
            batch_id = (
                f"batch_{span[0].frame.frame_index:08d}-"
                f"{span[-1].frame.frame_index:08d}"
            )
            if batch_id in completed:
                prior = completed[batch_id]
                current = [item.frame.frame_index
                           for item in span]
                if prior.get("frame_indices") != current:
                    raise FirstPassError(
                        "CHECKPOINT_MISMATCH",
                        f"Batch {batch_id} checkpoint covers "
                        f"{prior.get('frame_indices')}, current span "
                        f"is {current}; refusing to resume a "
                        "different frame set.",
                        batch_id,
                    )
                if prior.get("status") == "completed":
                    continue
            batch_result = self._run_batch(
                client, span, batch_id, prompt_digest,
                cfg.api_key, rng, paths, policy,
            )
            stats["requests_sent"] += batch_result["requests_sent"]
            stats["retries"] += batch_result["retries"]
            stats["input_tokens"] += batch_result["input_tokens"]
            stats["output_tokens"] += batch_result["output_tokens"]
            stats["frames_observed"] += batch_result["frames_observed"]
            stats["frames_failed"] += batch_result["frames_failed"]
            if batch_result["status"] != "completed":
                stats["needs_review"] += 1
            else:
                stats["batches_completed"] += 1
            for terminal, count in batch_result[
                    "terminal_states"].items():
                stats["terminal_states"][terminal] = (
                    stats["terminal_states"].get(terminal, 0) + count
                )
            stats["token_retries"] += batch_result["token_retries"]
            stats["token_retry_input_tokens"] += batch_result[
                "token_retry_input_tokens"]
            stats["token_retry_output_tokens"] += batch_result[
                "token_retry_output_tokens"]
            stats["token_retried_requests"] += batch_result[
                "token_retried_requests"]
            stats["token_recovered"] += batch_result["token_recovered"]
        return self._write_report(paths, stats, prepared)

    def _check_prior_state(
            self, state_path: Path,
            prepared: list[_PreparedFrame]) -> None:
        """Refuse reuse of a directory completed for other frames."""
        if not state_path.is_file():
            return
        try:
            prior = json.loads(state_path.read_text(encoding="utf-8"))
            prior_frames = prior["frame_indices"]
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            raise FirstPassError(
                "CHECKPOINT_MISMATCH",
                "Prior run state is unreadable; refusing resume.",
            ) from exc
        current = [item.frame.frame_index for item in prepared]
        if prior_frames != current:
            raise FirstPassError(
                "CHECKPOINT_MISMATCH",
                "Prior run state covers a different frame set; "
                "refusing resume into this directory.",
            )

    def _write_report(self, paths: dict[str, Path],
                      stats: dict[str, Any],
                      prepared: list[_PreparedFrame]
                      ) -> dict[str, Any]:
        cfg = self._config
        rates = cfg.rates_usd_per_m
        state = {
            "source_name": cfg.source.name,
            "source_sha256": cfg.source.sha256,
            "frame_indices": [item.frame.frame_index
                              for item in prepared],
            "model": cfg.model,
            "provider": cfg.provider,
            "pass_id": cfg.pass_id,
            "batches_total": stats["batches_total"],
            "batches_completed": stats["batches_completed"],
            "batches_needing_review": stats["needs_review"],
            "requests_sent": stats["requests_sent"],
            "retries": stats["retries"],
            "input_tokens": stats["input_tokens"],
            "output_tokens": stats["output_tokens"],
            "total_tokens": stats["input_tokens"] + stats["output_tokens"],
            "frames_total": stats["frames_total"],
            "frames_observed": stats["frames_observed"],
            "frames_failed": stats["frames_failed"],
        }
        if rates is not None and (
            stats["input_tokens"] or stats["output_tokens"]
        ):
            state["rates_usd_per_m"] = {
                "input": rates[0], "output": rates[1],
            }
            state["cost_usd_estimated"] = estimate_cost(
                stats["input_tokens"], stats["output_tokens"], rates,
            )
            state["cost_status"] = "estimated_from_configured_rates"
        else:
            state["cost_usd_estimated"] = None
            state["cost_status"] = "COST_UNKNOWN"
        terminal_states = stats.get("terminal_states", {})
        classified = sum(terminal_states.get(key, 0) for key in (
            "valid_prediction", "abstained", "invalid_json",
        ))
        retried = stats.get("token_retried_requests", 0)
        state["lifecycle"] = {
            "terminal_states": dict(sorted(terminal_states.items())),
            "classified_total": classified,
            "abstention_rate": (
                terminal_states.get("abstained", 0) / classified
                if classified else None
            ),
            "invalid_rate": (
                terminal_states.get("invalid_json", 0) / classified
                if classified else None
            ),
            "token_retries": stats.get("token_retries", 0),
            "token_retried_requests": retried,
            "recovery_rate": (
                stats.get("token_recovered", 0) / retried
                if retried else None
            ),
            "token_retry_cost_usd_estimated": estimate_cost(
                stats.get("token_retry_input_tokens", 0),
                stats.get("token_retry_output_tokens", 0),
                rates,
            ),
        }
        _write_json(paths["state"], state)
        _write_json(paths["reports"] / "run_report.json", state)
        return state
