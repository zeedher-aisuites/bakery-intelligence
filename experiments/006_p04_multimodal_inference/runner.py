"""Resumable batch runner for the P04 full-video experiment (Exp. 006).

Units: manifest rows (frame identity = ``frame_index``) grouped into
contiguous batches. A batch is complete only after raw responses,
parsed observations, and a checkpoint record are all durable.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lifecycle import (
    TERMINAL_EXHAUSTED,
    TERMINAL_FAILED,
    TERMINAL_VALID,
    BudgetPolicy,
    decide_attempt,
    estimate_cost,
)
from meta_client import (
    MAX_IMAGES_PER_REQUEST,
    InferenceResult,
    MetaClient,
    ProviderError,
    resolve_api_key,
)
from prompts import (
    MODEL_OUTPUT_FIELDS,
    PRIMARY_PROMPT,
    REVIEW_OUTPUT_FIELDS,
    build_observation,
    format_review_prompt,
    parse_model_json,
    prompt_hash,
)

MANIFEST_FIELDS = (
    "image_path",
    "frame_index",
    "timestamp_seconds",
    "source_timestamp_seconds",
    "requested_timestamp_seconds",
)
EXPECTED_ROWS = 685
EXPECTED_STEP = 50
EXPECTED_FIRST_INDEX = 0
EXPECTED_LAST_INDEX = 34200
SOURCE_FPS = 50.0
MODEL_RATES = {
    "muse-spark-1.3": (1.25, 4.25),
    "muse-wonder-1.0": (4.00, 20.00),
}
DEFAULT_BATCH_SIZE = 5
DEFAULT_IMAGES_PER_REQUEST = 1
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BACKOFF_BASE_SECONDS = 0.5
DEFAULT_BACKOFF_CAP_SECONDS = 30.0
RETRYABLE_CODES = frozenset({
    "RATE_LIMITED", "TIMEOUT", "NETWORK_ERROR", "SERVER_ERROR",
})
ABORT_CODES = frozenset({"CREDENTIAL_MISSING", "AUTH_FAILED"})
TIMESTAMP_TOLERANCE = 1e-6


class RunnerError(Exception):
    """Structured runner failure with a machine-readable code."""

    def __init__(self, code: str, message: str, entity_id: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.entity_id = entity_id


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(manifest_path: Path) -> tuple[list[dict[str, Any]], str]:
    """Validate the extraction manifest; fail closed on any defect."""
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RunnerError(
            "MANIFEST_INVALID", f"Cannot read manifest: {exc}",
            manifest_path.name,
        ) from exc
    lines = text.splitlines()
    if not lines or lines[0].strip() != ",".join(MANIFEST_FIELDS):
        raise RunnerError(
            "MANIFEST_INVALID", "Manifest header mismatch.",
            manifest_path.name,
        )
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    reader = csv.DictReader(lines)
    for number, record in enumerate(reader, start=2):
        try:
            index = int(record["frame_index"])
            stamp = float(record["timestamp_seconds"])
            source = float(record["source_timestamp_seconds"])
            requested = float(record["requested_timestamp_seconds"])
        except (TypeError, ValueError) as exc:
            raise RunnerError(
                "MANIFEST_INVALID",
                f"Row {number}: non-numeric index/timestamp.",
                f"row:{number}",
            ) from exc
        expected_name = f"frame_{index:08d}.jpg"
        if record["image_path"] != expected_name:
            raise RunnerError(
                "MANIFEST_INVALID",
                f"Row {number}: image_path does not match frame_index.",
                f"row:{number}",
            )
        if abs(stamp - index / SOURCE_FPS) > TIMESTAMP_TOLERANCE:
            raise RunnerError(
                "MANIFEST_INVALID",
                f"Row {number}: timestamp is not index/50.",
                f"row:{number}",
            )
        if index in seen:
            raise RunnerError(
                "MANIFEST_INVALID", f"Row {number}: duplicate frame_index.",
                f"row:{number}",
            )
        seen.add(index)
        rows.append({
            "image_path": record["image_path"],
            "frame_index": index,
            "timestamp_seconds": stamp,
            "source_timestamp_seconds": source,
            "requested_timestamp_seconds": requested,
        })
    if len(rows) != EXPECTED_ROWS:
        raise RunnerError(
            "MANIFEST_INVALID",
            f"Expected {EXPECTED_ROWS} rows, found {len(rows)}.",
            manifest_path.name,
        )
    indices = [row["frame_index"] for row in rows]
    if indices[0] != EXPECTED_FIRST_INDEX or indices[-1] != EXPECTED_LAST_INDEX:
        raise RunnerError(
            "MANIFEST_INVALID",
            "Manifest window is not anchored at 0..34200.",
            manifest_path.name,
        )
    if any(
        later - earlier != EXPECTED_STEP
        for earlier, later in zip(indices, indices[1:])
    ):
        raise RunnerError(
            "MANIFEST_INVALID",
            "Frame indices are not a strict step-50 sequence.",
            manifest_path.name,
        )
    return rows, sha256_file(manifest_path)


def resolve_rates(
    model: str,
    input_price: float | None,
    output_price: float | None,
) -> tuple[float, float] | None:
    """Explicit paired overrides win; else the verified model table."""
    if (input_price is None) != (output_price is None):
        raise RunnerError(
            "INVALID_CONFIG", "Price overrides must be paired."
        )
    if input_price is not None and output_price is not None:
        return (input_price, output_price)
    return MODEL_RATES.get(model)


def batch_ranges(count: int, size: int) -> list[tuple[int, int]]:
    """Contiguous [first_row, last_row] spans covering all rows."""
    if size < 1:
        raise RunnerError("INVALID_CONFIG", "Batch size must be >= 1.")
    return [
        (start, min(start + size - 1, count - 1))
        for start in range(0, count, size)
    ]


@dataclass
class RunConfig:
    manifest_path: Path
    run_dir: Path
    model: str
    prompt_text: str = PRIMARY_PROMPT
    pass_id: str = "baseline_1fps"
    batch_size: int = DEFAULT_BATCH_SIZE
    images_per_request: int = DEFAULT_IMAGES_PER_REQUEST
    max_output_tokens: int = 1024
    token_budgets: tuple[int, ...] = (1024,)
    store: bool = True
    timeout_seconds: float = 120.0
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    backoff_base_seconds: float = DEFAULT_BACKOFF_BASE_SECONDS
    backoff_cap_seconds: float = DEFAULT_BACKOFF_CAP_SECONDS
    jitter_seed: int = 20261028
    price_input_per_m: float | None = None
    price_output_per_m: float | None = None
    selected_rows: Sequence[int] | None = None
    is_review: bool = False
    primary_obs: dict[int, dict[str, Any]] | None = None
    allow_multi_image: bool = False
    client_factory: Any = None
    sleeper: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.perf_counter


@dataclass
class Runner:
    config: RunConfig

    def _paths(self) -> dict[str, Path]:
        root = self.config.run_dir
        return {
            "root": root,
            "raw": root / "raw_responses",
            "obs": root / "observations",
            "checkpoints": root / "checkpoints" / "checkpoints.jsonl",
            "state": root / "checkpoints" / "run_state.json",
            "reports": root / "reports",
        }

    def _backoff_delay(self, attempt: int, rng: random.Random) -> float:
        delay = self.config.backoff_base_seconds * (2.0 ** (attempt - 1))
        delay = min(delay, self.config.backoff_cap_seconds)
        return delay + rng.uniform(0.0, delay * 0.25)

    def _transport_attempts(
        self,
        client: MetaClient,
        prompt_text: str,
        image_paths: Sequence[Path],
        api_key: str,
        rng: random.Random,
        budget: int,
        first_number: int,
    ) -> tuple[InferenceResult | None, list[dict[str, Any]], str | None]:
        """Transport retry loop for one output-token budget.

        Identical to the historical behavior; each entry additionally
        carries its budget. Transport retries never change budgets.
        """
        attempts: list[dict[str, Any]] = []
        for step in range(self.config.max_attempts):
            attempt = first_number + step
            try:
                result = client.submit(prompt_text, image_paths, api_key)
            except ProviderError as exc:
                attempts.append({
                    "attempt": attempt, "budget": budget,
                    "code": exc.code, "message": exc.message,
                    "status_code": exc.status_code,
                })
                if exc.code in ABORT_CODES:
                    raise
                if exc.code in RETRYABLE_CODES and step + 1 < self.config.max_attempts:
                    wait = exc.retry_after_seconds
                    if wait is None:
                        wait = self._backoff_delay(step + 1, rng)
                    self.config.sleeper(wait)
                    attempts[-1]["slept_seconds"] = wait
                    continue
                return None, attempts, exc.code
            attempts.append({"attempt": attempt, "budget": budget,
                             "code": "ok"})
            return result, attempts, None
        return None, attempts, "RETRIES_EXHAUSTED"

    def _request_with_retry(
        self,
        client: MetaClient,
        prompt_text: str,
        image_paths: Sequence[Path],
        api_key: str,
        rng: random.Random,
        policy: BudgetPolicy,
    ) -> tuple[InferenceResult | None, list[dict[str, Any]],
                str | None, dict[str, Any]]:
        """Budget escalation around transport retries.

        Returns (result, attempts, error_code, lifecycle_info). With
        the default single budget the observable behavior matches the
        historical runner exactly; error codes are preserved and only
        additive lifecycle keys are new.
        """
        attempts: list[dict[str, Any]] = []
        budgets_used: list[int] = []
        superseded: list[dict[str, int | None]] = []
        transport_retries = 0
        for position, budget in enumerate(policy.budgets):
            client.max_output_tokens = budget
            first_number = len(attempts) + 1
            result, tried, error = self._transport_attempts(
                client, prompt_text, image_paths, api_key, rng,
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
                    "terminal": None,  # classified after parsing
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
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
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

    def _load_checkpoint_index(self, path: Path) -> dict[str, Any]:
        completed: dict[str, Any] = {}
        if not path.exists():
            return completed
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RunnerError(
                    "CHECKPOINT_CORRUPT",
                    f"Checkpoint line {number} is not valid JSON.",
                    path.name,
                ) from exc
            completed[record["batch_id"]] = record
        return completed

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )
        tmp.replace(path)

    def run(self) -> dict[str, Any]:
        cfg = self.config
        if cfg.images_per_request < 1:
            raise RunnerError("INVALID_CONFIG", "images_per_request must be >= 1.")
        if cfg.images_per_request > MAX_IMAGES_PER_REQUEST:
            raise RunnerError(
                "INVALID_CONFIG",
                f"images_per_request exceeds {MAX_IMAGES_PER_REQUEST}.",
            )
        if cfg.images_per_request > 1 and not cfg.allow_multi_image:
            raise RunnerError(
                "INVALID_CONFIG",
                "Multi-image requests require explicit allow_multi_image "
                "opt-in (unprobed against the live API).",
            )
        if cfg.is_review:
            if cfg.images_per_request != 1:
                raise RunnerError(
                    "INVALID_CONFIG",
                    "Review passes require exactly 1 image per request.",
                )
            if not cfg.primary_obs:
                raise RunnerError(
                    "INVALID_CONFIG",
                    "Review passes require primary observations.",
                )
        budgets = tuple(cfg.token_budgets)
        if budgets == (1024,) and cfg.max_output_tokens != 1024:
            budgets = (cfg.max_output_tokens,)
        try:
            policy = BudgetPolicy(budgets=budgets)
        except ValueError as exc:
            raise RunnerError("INVALID_CONFIG", str(exc)) from exc
        rates = resolve_rates(
            cfg.model, cfg.price_input_per_m, cfg.price_output_per_m
        )
        api_key = resolve_api_key()
        rows, manifest_hash = load_manifest(cfg.manifest_path)
        manifest_dir = cfg.manifest_path.parent
        if cfg.selected_rows is not None:
            if any(i < 0 or i >= len(rows) for i in cfg.selected_rows):
                raise RunnerError(
                    "INVALID_CONFIG", "selected_rows out of range."
                )
            rows = [rows[i] for i in cfg.selected_rows]
        for row in rows:
            candidate = manifest_dir / row["image_path"]
            if ".." in Path(row["image_path"]).parts or not candidate.is_file():
                raise RunnerError(
                    "IMAGE_MISSING",
                    f"Missing image for frame {row['frame_index']}.",
                    str(row["frame_index"]),
                )
            blob = candidate.read_bytes()
            if len(blob) < 4 or blob[:2] != b"\xff\xd8" or blob[-2:] != b"\xff\xd9":
                raise RunnerError(
                    "IMAGE_MISSING",
                    f"Corrupt JPEG for frame {row['frame_index']}.",
                    str(row["frame_index"]),
                )
            row["image_sha256"] = hashlib.sha256(blob).hexdigest()
            row["image_encoding"] = "jpeg"
        paths = self._paths()
        for key in ("raw", "obs", "reports"):
            paths[key].mkdir(parents=True, exist_ok=True)
        paths["checkpoints"].parent.mkdir(parents=True, exist_ok=True)
        completed = self._load_checkpoint_index(paths["checkpoints"])
        manifest_ref = {
            "manifest_dir": str(manifest_dir),
            "manifest_sha256": manifest_hash,
        }
        prompt_digest = prompt_hash(cfg.prompt_text)
        client_kwargs: dict[str, Any] = {
            "model": cfg.model,
            "timeout_seconds": cfg.timeout_seconds,
            "max_output_tokens": budgets[0],
            "store": cfg.store,
            "clock": cfg.clock,
        }
        if cfg.client_factory is not None:
            client_kwargs["client_factory"] = cfg.client_factory
        client = MetaClient(**client_kwargs)
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
        }
        for first, last in batch_ranges(len(rows), cfg.batch_size):
            stats["batches_total"] += 1
            span = rows[first:last + 1]
            batch_id = (
                f"batch_{span[0]['frame_index']:08d}-"
                f"{span[-1]['frame_index']:08d}"
            )
            if batch_id in completed and completed[batch_id].get("status") == "completed":
                continue
            batch_result = self._run_batch(
                client, cfg, policy, span, batch_id, manifest_ref,
                prompt_digest, api_key, rng, paths,
            )
            stats["requests_sent"] += batch_result["requests_sent"]
            stats["retries"] += batch_result["retries"]
            stats["input_tokens"] += batch_result["input_tokens"]
            stats["output_tokens"] += batch_result["output_tokens"]
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
            if batch_result["status"] == "completed":
                stats["batches_completed"] += 1
            else:
                stats["needs_review"] += 1
        return self._write_report(paths, cfg, manifest_hash, stats, rates)

    def _run_batch(
        self, client: MetaClient, cfg: RunConfig, policy: BudgetPolicy,
        span: list[dict[str, Any]], batch_id: str,
        manifest_ref: dict[str, str], prompt_digest: str,
        api_key: str, rng: random.Random, paths: dict[str, Path],
    ) -> dict[str, Any]:
        manifest_dir = cfg.manifest_path.parent
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
        step = cfg.images_per_request
        fields = REVIEW_OUTPUT_FIELDS if cfg.is_review else MODEL_OUTPUT_FIELDS
        for offset in range(0, len(span), step):
            group = span[offset:offset + step]
            image_paths = [manifest_dir / row["image_path"] for row in group]
            request_text = cfg.prompt_text
            review_hash: str | None = None
            if cfg.is_review:
                primary = (cfg.primary_obs or {}).get(
                    group[0]["frame_index"]
                )
                if primary is None:
                    batch_failed = True
                    raw_records.append({
                        "frame_indices": [row["frame_index"] for row in group],
                        "error_code": "REVIEW_PRIMARY_MISSING",
                        "attempts": [],
                    })
                    continue
                primary_json = json.dumps(primary, sort_keys=True)
                review_hash = hashlib.sha256(
                    primary_json.encode("utf-8")
                ).hexdigest()
                request_text = format_review_prompt(primary_json)
            result, attempts, error, info = self._request_with_retry(
                client, request_text, image_paths, api_key, rng,
                policy,
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
                "frame_indices": [row["frame_index"] for row in group],
                "attempts": attempts,
            })
            if error is not None or result is None:
                batch_failed = True
                terminal_states[info["terminal"]] = (
                    terminal_states.get(info["terminal"], 0) + 1
                )
                raw_records.append({
                    "frame_indices": [row["frame_index"] for row in group],
                    "error_code": error, "attempts": attempts,
                    "terminal_state": info["terminal"],
                    "incomplete_reason": info["incomplete_reason"],
                    "budgets_attempted": budgets_used,
                })
                continue
            if result.input_tokens is not None:
                input_tokens += result.input_tokens
            if result.output_tokens is not None:
                output_tokens += result.output_tokens
            parsed, _ = parse_model_json(result.output_text)
            abstained_hint = (
                not cfg.is_review and parsed is not None
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
                "frame_indices": [row["frame_index"] for row in group],
                "model": result.model, "status": result.status,
                "output_text": result.output_text,
                "usage": {
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "total_tokens": result.total_tokens,
                },
                "latency_seconds": result.latency_seconds,
                "request_id": result.request_id,
                "attempts": attempts,
                "terminal_state": terminal,
                "incomplete_reason": result.incomplete_reason,
                "budgets_attempted": budgets_used,
            })
            call_provenance = {
                "provider": "meta", "model": result.model,
                "prompt_sha256": prompt_digest,
                "review_of_sha256": review_hash,
                "max_output_tokens": budgets_used[-1],
                "token_budgets_attempted": budgets_used,
                "store": cfg.store,
                "timeout_seconds": cfg.timeout_seconds,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "reasoning_output_tokens": result.reasoning_output_tokens,
                "latency_seconds": result.latency_seconds,
                "request_id": result.request_id,
                "attempts": len(attempts),
                "response_status": result.status,
                "incomplete_reason": result.incomplete_reason,
                "terminal_state": terminal,
            }
            for row in group:
                observations.append(build_observation(
                    row, manifest_ref, batch_id, cfg.pass_id,
                    call_provenance, result.output_text, fields,
                    response_status=result.status,
                    incomplete_reason=result.incomplete_reason,
                    terminal_state=terminal,
                    image_sha256=row.get("image_sha256"),
                    image_encoding=row.get("image_encoding"),
                ))
        status = "needs_human_review" if batch_failed else "completed"
        raw_path = paths["raw"] / f"{batch_id}.json"
        obs_path = paths["obs"] / f"{batch_id}.json"
        self._write_json(raw_path, raw_records)
        self._write_json(obs_path, observations)
        record = {
            "batch_id": batch_id,
            "index_range": [
                span[0]["frame_index"], span[-1]["frame_index"],
            ],
            "status": status,
            "raw_path": raw_path.name,
            "obs_path": obs_path.name,
            "attempts": attempts_log,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        with paths["checkpoints"].open("a", encoding="utf-8", newline="\n") as handle:
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
        }

    def _write_report(
        self, paths: dict[str, Path], cfg: RunConfig,
        manifest_hash: str, stats: dict[str, Any],
        rates: tuple[float, float] | None,
    ) -> dict[str, Any]:
        state = {
            "manifest_sha256": manifest_hash,
            "model": cfg.model,
            "pass_id": cfg.pass_id,
            "batches_total": stats["batches_total"],
            "batches_completed": stats["batches_completed"],
            "batches_needing_review": stats["needs_review"],
            "requests_sent": stats["requests_sent"],
            "retries": stats["retries"],
            "input_tokens": stats["input_tokens"],
            "output_tokens": stats["output_tokens"],
            "total_tokens": stats["input_tokens"] + stats["output_tokens"],
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
            state["cost_status"] = "estimated_from_published_rates"
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
        self._write_json(paths["state"], state)
        self._write_json(paths["reports"] / "run_report.json", state)
        return state
