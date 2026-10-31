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

Budget guardrails (S1): an optional request cap and cost
ceiling bound each run. Stops are explicit outcomes recorded
in the report, never silent truncation; resume seeds its
accounting from checkpoints instead of resetting it.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
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

# Budget-halt reasons (S1). Recorded verbatim in reports and
# checkpoints; treat as stable vocabulary.
STOP_REQUEST_LIMIT = "request_limit_reached"
STOP_COST_CEILING = "cost_ceiling_exceeded"
STOP_COST_RESERVE = "cost_reserve_exceeded"
STOP_STRICT_UNKNOWN_COST = "unknown_cost_strict_halt"

# Reserve heuristic label. The pre-dispatch reserve projects
# the next request at the largest single-request estimate seen
# so far; input tokens are unknowable before dispatch, so this
# is explicitly not a guarantee (see the ledger).
RESERVE_BASIS_OBSERVED_MAX = "max_observed_single_request_estimate"

# Batch-start marker files (S2). A marker without a matching
# checkpoint record means the batch may have submitted requests
# whose outcomes were never durably recorded: uncertain.
MARKER_PREFIX = "started_"
MARKER_SUFFIX = ".json"

# Uncertain-batch policies (S2). "rerun" re-attempts uncertain
# batches (safe for free deterministic clients); "refuse"
# fails closed for operator resolution (required for paid use).
UNCERTAIN_RERUN = "rerun"
UNCERTAIN_REFUSE = "refuse"

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


class _BudgetHalt(Exception):
    """Internal budget-stop signal (never escapes the runner)."""

    def __init__(self, reason: str,
                 detail: dict[str, Any]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


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
    max_requests: int | None = None
    cost_ceiling_usd: float | None = None
    strict_cost: bool = False
    # S3: explicit billing declaration. False keeps the historical
    # offline/free retry behavior; True opts into fail-closed
    # single-submit invariants enforced by _validate_billable_config.
    billable: bool = False
    uncertain_policy: str = "rerun"
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


def _validate_budget_config(config: FirstPassConfig) -> None:
    """Reject malformed run guards before any request.

    Zero is a valid limit (it simply allows no submits or no
    spend); negative, non-finite, and mistyped limits are
    configuration defects, never silently coerced. The
    uncertain-batch policy must name a known behavior.
    """
    limit = config.max_requests
    if limit is not None and (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 0):
        raise FirstPassError(
            "INVALID_CONFIG",
            "max_requests must be a non-negative int or None, "
            f"got {limit!r}.")
    ceiling = config.cost_ceiling_usd
    if ceiling is not None and (
            isinstance(ceiling, bool)
            or not isinstance(ceiling, (int, float))
            or not math.isfinite(ceiling)
            or ceiling < 0):
        raise FirstPassError(
            "INVALID_CONFIG",
            "cost_ceiling_usd must be a non-negative finite "
            f"number or None, got {ceiling!r}.")
    if not isinstance(config.strict_cost, bool):
        raise FirstPassError(
            "INVALID_CONFIG",
            "strict_cost must be a bool, "
            f"got {config.strict_cost!r}.")
    if config.uncertain_policy not in (
            UNCERTAIN_RERUN, UNCERTAIN_REFUSE):
        raise FirstPassError(
            "INVALID_CONFIG",
            "uncertain_policy must be 'rerun' or 'refuse', "
            f"got {config.uncertain_policy!r}.")


def _validate_billable_config(config: FirstPassConfig) -> None:
    """Enforce fail-closed single-submit invariants for billable runs.

    Billing status is an explicit declaration, never inferred from
    the client class or any other field. A billable run submits each
    logical frame at most once: no transport retries, no
    token-budget escalation, and no uncertain-batch replay. Any
    ambiguous or unsafe billable configuration is rejected before
    any request is dispatched.
    """
    billable = config.billable
    if not isinstance(billable, bool):
        raise FirstPassError(
            "INVALID_CONFIG",
            "billable must be a bool, "
            f"got {billable!r}.")
    if not billable:
        return
    attempts = config.max_attempts
    if (isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or attempts != 1):
        raise FirstPassError(
            "INVALID_CONFIG",
            "billable execution requires max_attempts=1 "
            "(single submit, no transport retry), "
            f"got {attempts!r}.")
    budgets = config.token_budgets
    if not isinstance(budgets, (tuple, list)) or len(budgets) != 1:
        raise FirstPassError(
            "INVALID_CONFIG",
            "billable execution requires exactly one token budget "
            f"(no escalation replay), got {budgets!r}.")
    if config.uncertain_policy != UNCERTAIN_REFUSE:
        raise FirstPassError(
            "INVALID_CONFIG",
            "billable execution requires uncertain_policy='refuse' "
            "(no uncertain replay), "
            f"got {config.uncertain_policy!r}.")


@dataclass
class _BudgetLedger:
    """Per-run request/cost accounting with enforcement (S1).

    Every submitted request counts exactly once, including
    transport retries and token-budget escalations. Known
    spend sums per-request estimates from reported usage;
    submits without usable usage count as unknown cost, never
    as zero. Unknown totals poison the reported estimate
    (mirroring 007 ``aggregate_spent``) while the known part
    stays available as the enforcement basis.

    Resume seeds the prior counters from checkpoints, so
    enforcement applies to lifetime totals rather than a
    reset zero. Completed, budget-stopped, and
    review-required batches all seed their recorded
    counters; review batches resume only their unobserved
    frames, so known-good requests are never re-submitted
    merely because a batch needs review.

    Guarantees and gaps: the request cap is exact — no
    submit leaves after it trips. The cost ceiling halts
    further spend once known spend exceeds it, but input
    tokens are unknowable before dispatch, so the
    pre-dispatch reserve is a heuristic (largest observed
    single-request estimate), not a bound; overruns are
    detected after the fact. Strict mode additionally
    halts on the first unknown-cost submit instead of
    continuing blind. ``run_manifest.BudgetState`` is not
    reused here because it cannot represent unknown spend.
    """

    max_requests: int | None
    ceiling: float | None
    strict: bool
    rates: tuple[float, float] | None
    uncertain_policy: str = UNCERTAIN_RERUN
    prior_requests: int = 0
    prior_spent: float = 0.0
    prior_unknown: int = 0
    prior_max_single: float | None = None
    prior_completed: set[int] = field(default_factory=set)
    prior_observed: int = 0
    prior_basis: str = "none"
    rates_changed: bool = False
    requests: int = 0
    spent: float = 0.0
    unknown: int = 0
    max_single: float | None = None
    attempted: set[int] = field(default_factory=set)
    stopped_reason: str | None = None
    stopped_detail: dict[str, Any] | None = None
    batch_requests: int = 0
    batch_spent: float = 0.0
    batch_unknown: int = 0
    batch_max_single: float | None = None

    def begin_batch(self) -> None:
        """Reset per-batch counters (checkpoint deltas)."""
        self.batch_requests = 0
        self.batch_spent = 0.0
        self.batch_unknown = 0
        self.batch_max_single = None

    @property
    def total_requests(self) -> int:
        """Submits across prior and current invocations."""
        return self.prior_requests + self.requests

    @property
    def total_spent(self) -> float:
        """Known spend across invocations, rounded for dust."""
        return round(self.prior_spent + self.spent, 4)

    @property
    def total_unknown(self) -> int:
        """Unknown-cost submits across invocations."""
        return self.prior_unknown + self.unknown

    @property
    def reserve(self) -> float | None:
        """Largest single-request estimate seen, if any."""
        candidates = [value for value in
                      (self.prior_max_single, self.max_single)
                      if value is not None]
        return max(candidates) if candidates else None

    def halt(self, reason: str,
             detail: dict[str, Any]) -> None:
        """Record the first halt; later halts keep it."""
        if self.stopped_reason is None:
            self.stopped_reason = reason
            self.stopped_detail = detail

    def pre_dispatch(self) -> tuple[str, dict[str, Any]] | None:
        """Veto the next submit when a limit already trips.

        Exact checks first (request count, known spend,
        strict unknowability), heuristic reserve last.
        Returns None when dispatch is allowed.
        """
        if (self.max_requests is not None
                and self.total_requests >= self.max_requests):
            return STOP_REQUEST_LIMIT, {
                "requests_submitted": self.total_requests,
                "max_requests": self.max_requests,
            }
        if (self.ceiling is not None
                and self.total_spent > self.ceiling):
            return STOP_COST_CEILING, {
                "phase": "pre_dispatch",
                "spent_known_usd": self.total_spent,
                "cost_ceiling_usd": self.ceiling,
            }
        if (self.strict and self.ceiling is not None
                and self.rates is None):
            return STOP_STRICT_UNKNOWN_COST, {
                "phase": "pre_dispatch",
                "cause": "rates_unknown",
                "cost_ceiling_usd": self.ceiling,
            }
        reserve = self.reserve
        if (self.ceiling is not None and self.rates is not None
                and reserve is not None
                and self.total_spent + reserve > self.ceiling):
            return STOP_COST_RESERVE, {
                "spent_known_usd": self.total_spent,
                "reserve_usd_estimate": reserve,
                "reserve_basis": RESERVE_BASIS_OBSERVED_MAX,
                "cost_ceiling_usd": self.ceiling,
            }
        return None

    def note_submitted(self, frame_index: int) -> None:
        """Count one submit leaving for a frame."""
        self.requests += 1
        self.batch_requests += 1
        self.attempted.add(frame_index)

    def note_transport_failure(self) -> bool:
        """Account a submit without usage; True when halting.

        A failed transport may or may not have been billed,
        so its cost is unknown rather than zero.
        """
        self.unknown += 1
        self.batch_unknown += 1
        if self.strict and self.ceiling is not None:
            self.halt(STOP_STRICT_UNKNOWN_COST, {
                "phase": "transport",
                "cause": "transport_failure_without_usage",
                "cost_unknown_requests": self.total_unknown,
                "cost_ceiling_usd": self.ceiling,
            })
            return True
        return False

    def note_usages(self, pairs: list[tuple[Any, Any]]) -> bool:
        """Account billed attempts in order; True when halting.

        Each pair is one submitted request's (input, output)
        token report, oldest first. Unusable reports count as
        unknown cost; known estimates accumulate toward the
        ceiling with an overrun check after each addition.
        Every pair is accounted even when an early pair
        trips a halt, so halting never hides known spend;
        the recorded reason stays the first trigger.
        """
        tripped = False
        for input_tokens, output_tokens in pairs:
            estimate = estimate_cost(
                input_tokens, output_tokens, self.rates)
            if estimate is None:
                self.unknown += 1
                self.batch_unknown += 1
                if self.strict and self.ceiling is not None:
                    self.halt(STOP_STRICT_UNKNOWN_COST, {
                        "phase": "post_response",
                        "cause": "usage_unreported",
                        "cost_unknown_requests": (
                            self.total_unknown),
                        "cost_ceiling_usd": self.ceiling,
                    })
                    tripped = True
                continue
            self.spent += estimate
            self.batch_spent += estimate
            if (self.max_single is None
                    or estimate > self.max_single):
                self.max_single = estimate
            if (self.batch_max_single is None
                    or estimate > self.batch_max_single):
                self.batch_max_single = estimate
            if (self.ceiling is not None
                    and self.total_spent > self.ceiling):
                self.halt(STOP_COST_CEILING, {
                    "phase": "post_response",
                    "spent_known_usd": self.total_spent,
                    "cost_ceiling_usd": self.ceiling,
                })
                tripped = True
        return tripped

    def remaining(self, ordered: list[int]) -> list[int]:
        """Frames never attempted in any invocation, in order."""
        return [index for index in ordered
                if index not in self.prior_completed
                and index not in self.attempted]

    def report_block(
            self, remaining: list[int],
            uncertain: list[int]) -> dict[str, Any]:
        """Assemble the additive budget report section."""
        spent = round(self.spent, 4)
        prior_spent = round(self.prior_spent, 4)
        if self.prior_unknown:
            prior_estimate = None
        elif (self.prior_basis in (
                "reconstructed_from_token_totals", "mixed")
                and self.rates is None
                and self.prior_requests):
            prior_estimate = None
        else:
            prior_estimate = prior_spent
        return {
            "max_requests": self.max_requests,
            "requests_submitted": self.requests,
            "prior_requests_submitted": self.prior_requests,
            "cost_ceiling_usd": self.ceiling,
            "currency": "USD",
            "rates_usd_per_m": (
                list(self.rates)
                if self.rates is not None else None),
            "strict_cost": self.strict,
            "spent_cost_usd_estimate": (
                None if self.unknown else spent),
            "spent_cost_known_usd_estimate": spent,
            "cost_unknown_requests": self.unknown,
            "prior_spent_cost_usd_estimate": prior_estimate,
            "prior_spent_cost_known_usd_estimate": prior_spent,
            "prior_cost_unknown_requests": self.prior_unknown,
            "prior_cost_basis": self.prior_basis,
            "prior_frames_observed": self.prior_observed,
            "rates_changed_on_resume": self.rates_changed,
            "pre_dispatch_reserve_basis": (
                RESERVE_BASIS_OBSERVED_MAX
                if self.reserve is not None else "unavailable"),
            "pre_dispatch_cost_guarantee": False,
            "stopped_reason": self.stopped_reason,
            "stopped_detail": self.stopped_detail,
            "remaining_frame_indices": remaining,
            "uncertain_frame_indices": list(uncertain),
            "uncertain_policy": self.uncertain_policy,
        }


def _seed_ledger(config: FirstPassConfig,
                 index: dict[str, Any]) -> _BudgetLedger:
    """Build the ledger, seeding resume counters from checkpoints.

    Completed spans seed fully; budget-stopped and
    review-required batches seed their recorded counters
    plus their observed frames, since resume re-runs only
    their unobserved remainder. Pre-S1 review records lack
    the observed list; run() fills that gap from the batch
    observation files it is already merging.
    """
    ledger = _BudgetLedger(
        max_requests=config.max_requests,
        ceiling=(None if config.cost_ceiling_usd is None
                 else float(config.cost_ceiling_usd)),
        strict=config.strict_cost,
        rates=config.rates_usd_per_m,
        uncertain_policy=config.uncertain_policy,
    )
    recorded = reconstructed = 0
    for record in index.values():
        status = record.get("status")
        if status == "completed":
            frames = list(record.get("frame_indices", []))
        elif status in ("budget_stopped", "needs_human_review"):
            frames = list(
                record.get("completed_frame_indices", []))
        else:
            continue
        ledger.prior_completed.update(frames)
        ledger.prior_observed += len(frames)
        block = record.get("budget")
        if isinstance(block, dict):
            recorded += 1
            ledger.prior_requests += block.get(
                "requests_submitted", 0)
            ledger.prior_spent += block.get(
                "cost_known_usd_estimate", 0.0)
            ledger.prior_unknown += block.get(
                "cost_unknown_requests", 0)
            candidate = block.get(
                "max_single_request_cost_usd_estimate")
            if isinstance(candidate, (int, float)) and (
                    ledger.prior_max_single is None
                    or candidate > ledger.prior_max_single):
                ledger.prior_max_single = float(candidate)
            prior_rates = block.get("rates_usd_per_m")
            current = (list(config.rates_usd_per_m)
                       if config.rates_usd_per_m is not None
                       else None)
            if current != (list(prior_rates)
                            if prior_rates is not None
                            else None):
                ledger.rates_changed = True
        else:
            reconstructed += 1
            attempts = record.get("attempts", [])
            ledger.prior_requests += sum(
                len(entry.get("attempts", []))
                for entry in attempts
                if isinstance(entry, dict))
            estimate = estimate_cost(
                record.get("input_tokens"),
                record.get("output_tokens"),
                config.rates_usd_per_m)
            if estimate is not None:
                ledger.prior_spent += estimate
    if recorded and reconstructed:
        ledger.prior_basis = "mixed"
    elif recorded:
        ledger.prior_basis = "recorded"
    elif reconstructed:
        ledger.prior_basis = "reconstructed_from_token_totals"
    if (ledger.prior_requests < 0 or ledger.prior_spent < 0
            or ledger.prior_unknown < 0):
        raise FirstPassError(
            "CHECKPOINT_CORRUPT",
            "Checkpoint counters are negative; refusing resume.",
        )
    return ledger


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
        """Index checkpoints by batch, recovering a torn tail.

        Appends are single-writer, so only the final line can
        tear in a crash; it is truncated and ignored, and the
        batch it belonged to resumes as uncertain (its start
        marker survives). A malformed line ahead of more
        content is corruption, and fails closed.
        """
        completed: dict[str, Any] = {}
        if not path.exists():
            return completed
        chunks = path.read_bytes().split(b"\n")
        offset = 0
        for position, chunk in enumerate(chunks):
            start = offset
            offset += len(chunk) + 1
            if not chunk.strip():
                continue
            try:
                record = json.loads(chunk.decode("utf-8"))
            except (UnicodeDecodeError,
                    json.JSONDecodeError) as exc:
                if any(rest.strip()
                       for rest in chunks[position + 1:]):
                    raise FirstPassError(
                        "CHECKPOINT_CORRUPT",
                        "Checkpoint log has a malformed line "
                        "ahead of more content; refusing resume.",
                        path.name,
                    ) from exc
                with path.open("r+b") as handle:
                    handle.truncate(start)
                break
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
        ledger: _BudgetLedger,
        frame_index: int,
        attempts: list[dict[str, Any]],
    ) -> tuple[InferenceResponse | None, str | None]:
        for step in range(self._config.max_attempts):
            attempt = len(attempts) + 1
            stop = ledger.pre_dispatch()
            if stop is not None:
                reason, detail = stop
                ledger.halt(reason, detail)
                raise _BudgetHalt(reason, detail)
            ledger.note_submitted(frame_index)
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
                if ledger.note_transport_failure():
                    reason = ledger.stopped_reason
                    detail = ledger.stopped_detail
                    assert reason is not None
                    assert detail is not None
                    raise _BudgetHalt(reason, detail)
                if (exc.code in RETRYABLE_CODES
                        and step + 1 < self._config.max_attempts):
                    wait = exc.retry_after_seconds
                    if wait is None:
                        wait = self._backoff_delay(step + 1, rng)
                    self._config.sleeper(wait)
                    attempts[-1]["slept_seconds"] = wait
                    continue
                return None, exc.code
            attempts.append({"attempt": attempt, "budget": budget,
                             "code": "ok"})
            return result, None
        return None, "RETRIES_EXHAUSTED"

    def _request_with_retry(
        self,
        client: InferenceClient,
        prompt_text: str,
        image: Path,
        api_key: str,
        rng: random.Random,
        policy: BudgetPolicy,
        ledger: _BudgetLedger,
        frame_index: int,
        attempts: list[dict[str, Any]],
    ) -> tuple[InferenceResponse | None, str | None,
                dict[str, Any]]:
        budgets_used: list[int] = []
        superseded: list[dict[str, int | None]] = []
        transport_retries = 0
        for position, budget in enumerate(policy.budgets):
            before = len(attempts)
            result, error = self._transport_attempts(
                client, prompt_text, image, api_key, rng,
                budget, ledger, frame_index, attempts,
            )
            budgets_used.append(budget)
            transport_retries += max(
                0, len(attempts) - before - 1)
            if error is not None or result is None:
                return None, error, {
                    "terminal": TERMINAL_FAILED,
                    "incomplete_reason": None,
                    "budgets_used": budgets_used,
                    "transport_retries": transport_retries,
                    "superseded_usage": superseded,
                }
            if result.status == "completed":
                return result, None, {
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
            return None, "INCOMPLETE_STATUS", {
                "terminal": decision["terminal"],
                "incomplete_reason": result.incomplete_reason,
                "budgets_used": budgets_used,
                "transport_retries": transport_retries,
                "superseded_usage": superseded,
            }
        return None, "INCOMPLETE_STATUS", {
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
        ledger: _BudgetLedger,
        prior: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        cfg = self._config
        prior = prior or {}
        ledger.begin_batch()
        raw_records: list[dict[str, Any]] = list(
            prior.get("raw", []))
        observations: list[dict[str, Any]] = list(
            prior.get("obs", []))
        attempts_log: list[dict[str, Any]] = list(
            prior.get("attempts", []))
        prior_input = prior.get("input_tokens", 0)
        prior_output = prior.get("output_tokens", 0)
        prior_budget = prior.get("budget", {})
        prior_completed = list(prior.get("completed", []))
        if prior:
            full_indices = list(prior.get("frame_indices", []))
            index_range = list(prior.get("index_range", []))
        else:
            full_indices = [item.frame.frame_index
                            for item in prepared]
            index_range = [
                prepared[0].frame.frame_index,
                prepared[-1].frame.frame_index,
            ]
        requests_sent = retries = 0
        input_tokens = output_tokens = 0
        batch_failed = False
        halted = False
        new_observed: list[int] = []
        terminal_states: dict[str, int] = {}
        token_retries = 0
        token_retry_input_tokens = 0
        token_retry_output_tokens = 0
        token_retried_requests = 0
        token_recovered = 0
        _write_json(
            paths["checkpoints"].parent
            / f"{MARKER_PREFIX}{batch_id}{MARKER_SUFFIX}",
            {"batch_id": batch_id,
             "frame_indices": [item.frame.frame_index
                               for item in prepared],
             "started_at_utc": datetime.datetime.now(
                 datetime.timezone.utc).isoformat()},
        )
        for item in prepared:
            frame_attempts: list[dict[str, Any]] = []
            attempts_log.append({
                "frame_indices": [item.frame.frame_index],
                "attempts": frame_attempts,
            })
            try:
                result, error, info = self._request_with_retry(
                    client, cfg.prompt_text,
                    item.frame.image_path,
                    api_key, rng, policy, ledger,
                    item.frame.frame_index, frame_attempts,
                )
            except _BudgetHalt:
                halted = True
                break
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
            if error is not None or result is None:
                batch_failed = True
                terminal_states[info["terminal"]] = (
                    terminal_states.get(info["terminal"], 0) + 1
                )
                raw_records.append({
                    "frame_indices": [item.frame.frame_index],
                    "error_code": error,
                    "attempts": frame_attempts,
                    "terminal_state": info["terminal"],
                    "incomplete_reason": info["incomplete_reason"],
                    "budgets_attempted": budgets_used,
                })
                abandoned_pairs = [
                    (entry.get("input_tokens"),
                     entry.get("output_tokens"))
                    for entry in info["superseded_usage"]
                ]
                if ledger.note_usages(abandoned_pairs):
                    halted = True
                    break
                continue
            usage_pairs = [
                (entry.get("input_tokens"),
                 entry.get("output_tokens"))
                for entry in info["superseded_usage"]
            ]
            usage_pairs.append((
                result.usage.input_tokens,
                result.usage.output_tokens,
            ))
            if ledger.note_usages(usage_pairs):
                halted = True
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
                "attempts": frame_attempts,
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
                "attempts": len(frame_attempts),
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
            new_observed.append(item.frame.frame_index)
            if halted:
                break
        if halted:
            status = "budget_stopped"
        else:
            status = ("needs_human_review" if batch_failed
                      else "completed")
        raw_path = paths["raw"] / f"{batch_id}.json"
        obs_path = paths["obs"] / f"{batch_id}.json"
        _write_json(raw_path, raw_records)
        _write_json(obs_path, observations)
        completed = prior_completed + new_observed
        merged_spent = round(
            prior_budget.get("cost_known_usd_estimate", 0.0)
            + ledger.batch_spent, 4)
        merged_unknown = (
            prior_budget.get("cost_unknown_requests", 0)
            + ledger.batch_unknown)
        merged_max: float | None = None
        for candidate in (
                prior_budget.get(
                    "max_single_request_cost_usd_estimate"),
                ledger.batch_max_single):
            if isinstance(candidate, (int, float)) and (
                    merged_max is None
                    or candidate > merged_max):
                merged_max = float(candidate)
        record = {
            "batch_id": batch_id,
            "index_range": index_range,
            "frame_indices": full_indices,
            "status": status,
            "completed_frame_indices": completed,
            "raw_path": raw_path.name,
            "obs_path": obs_path.name,
            "attempts": attempts_log,
            "input_tokens": prior_input + input_tokens,
            "output_tokens": prior_output + output_tokens,
            "budget": {
                "requests_submitted": (
                    prior_budget.get("requests_submitted", 0)
                    + ledger.batch_requests),
                "cost_usd_estimate": (
                    None if merged_unknown else merged_spent),
                "cost_known_usd_estimate": merged_spent,
                "cost_unknown_requests": merged_unknown,
                "max_single_request_cost_usd_estimate": (
                    merged_max),
                "rates_usd_per_m": (
                    list(cfg.rates_usd_per_m)
                    if cfg.rates_usd_per_m is not None
                    else None),
                "currency": "USD",
            },
            "stopped_reason": (
                ledger.stopped_reason if halted else None),
            "stopped_detail": (
                ledger.stopped_detail if halted else None),
        }
        with paths["checkpoints"].open(
                "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        attempted_new = [
            item.frame.frame_index for item in prepared
            if item.frame.frame_index in ledger.attempted]
        unprocessed = [
            item.frame.frame_index for item in prepared
            if item.frame.frame_index not in ledger.attempted]
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
            "frames_observed": len(new_observed),
            "frames_failed": (
                len(attempted_new) - len(new_observed)),
            "frames_unprocessed": len(unprocessed),
            "stopped_reason": (
                ledger.stopped_reason if halted else None),
            "stopped_detail": (
                ledger.stopped_detail if halted else None),
        }

    def run(self) -> dict[str, Any]:
        """Execute all batches with resume; return the run report."""
        cfg = self._config
        _validate_budget_config(cfg)
        _validate_billable_config(cfg)
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
        ledger = _seed_ledger(cfg, completed)
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
            "frames_unprocessed": 0,
        }
        self._check_prior_state(paths["state"], prepared)
        expected_spans: dict[str, list[int]] = {}
        for begin in range(0, len(prepared), cfg.batch_size):
            expected = prepared[begin:begin + cfg.batch_size]
            expected_spans[
                f"batch_{expected[0].frame.frame_index:08d}-"
                f"{expected[-1].frame.frame_index:08d}"
            ] = [item.frame.frame_index for item in expected]
        uncertain_map = self._scan_uncertain(
            paths["checkpoints"].parent, completed,
            expected_spans)
        uncertain_frames = sorted({
            frame for span in uncertain_map.values()
            for frame in span})
        if uncertain_map and (
                cfg.uncertain_policy == UNCERTAIN_REFUSE):
            batches = sorted(uncertain_map)
            raise FirstPassError(
                "UNCERTAIN_REQUESTS",
                f"{len(uncertain_frames)} frame(s) in "
                f"{len(batches)} batch(es) lack verified "
                f"outcomes ({', '.join(batches)}); refusing "
                f"automatic replay (uncertain_policy=refuse). "
                f"Resolve explicitly: inspect the batch files, "
                f"then re-run with uncertain_policy='rerun' "
                f"only after confirming replay is safe.",
                batches[0],
            )
        for start in range(0, len(prepared), cfg.batch_size):
            span = prepared[start:start + cfg.batch_size]
            stats["batches_total"] += 1
            batch_id = (
                f"batch_{span[0].frame.frame_index:08d}-"
                f"{span[-1].frame.frame_index:08d}"
            )
            prior: dict[str, Any] | None = None
            if batch_id in completed:
                record = completed[batch_id]
                current = [item.frame.frame_index
                           for item in span]
                if record.get("frame_indices") != current:
                    raise FirstPassError(
                        "CHECKPOINT_MISMATCH",
                        f"Batch {batch_id} checkpoint covers "
                        f"{record.get('frame_indices')}, current "
                        f"span is {current}; refusing to resume a "
                        "different frame set.",
                        batch_id,
                    )
                status = record.get("status")
                if status == "completed":
                    continue
                if status not in (
                        "budget_stopped", "needs_human_review"):
                    raise FirstPassError(
                        "CHECKPOINT_MISMATCH",
                        f"Batch {batch_id} has unrecognized "
                        f"checkpoint status {status!r}; "
                        f"refusing resume.",
                        batch_id,
                    )
                prior = self._load_prior_batch(paths, record)
                fresh = (set(prior["completed"])
                         - ledger.prior_completed)
                ledger.prior_observed += len(fresh)
                ledger.prior_completed.update(fresh)
                done = set(prior["completed"])
                span = [item for item in span
                        if item.frame.frame_index not in done]
                if not span:
                    stats["batches_completed"] += 1
                    continue
            batch_result = self._run_batch(
                client, span, batch_id, prompt_digest,
                cfg.api_key, rng, paths, policy,
                ledger, prior,
            )
            stats["requests_sent"] += batch_result["requests_sent"]
            stats["retries"] += batch_result["retries"]
            stats["input_tokens"] += batch_result["input_tokens"]
            stats["output_tokens"] += batch_result["output_tokens"]
            stats["frames_observed"] += batch_result["frames_observed"]
            stats["frames_failed"] += batch_result["frames_failed"]
            if batch_result["status"] == "completed":
                stats["batches_completed"] += 1
            elif batch_result["status"] == "needs_human_review":
                stats["needs_review"] += 1
            # A budget_stopped batch is neither: the halt and
            # the remaining work stay explicit in the report.
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
            if batch_result["stopped_reason"] is not None:
                break
        remaining = ledger.remaining(
            [item.frame.frame_index for item in prepared])
        stats["frames_unprocessed"] = len(remaining)
        return self._write_report(
            paths, stats, prepared, ledger, remaining,
            uncertain_frames)

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

    def _load_prior_batch(self, paths: dict[str, Path],
                          record: dict[str, Any]
                          ) -> dict[str, Any]:
        """Load partial batch outputs for a partial-batch resume.

        Serves budget-stopped and review-required batches: the
        observation files on disk are the ground truth for
        which frames completed, so known-good requests are
        never re-submitted. Missing or malformed partials fail
        closed: resuming onto vanished evidence would silently
        drop billed work.
        """
        batch_id = record.get("batch_id", "?")
        try:
            raw = json.loads((paths["raw"] / record["raw_path"]
                              ).read_text(encoding="utf-8"))
            obs = json.loads((paths["obs"] / record["obs_path"]
                              ).read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError,
                TypeError) as exc:
            raise FirstPassError(
                "CHECKPOINT_CORRUPT",
                f"Batch {batch_id} partial outputs are missing "
                f"or unreadable; refusing resume.",
                str(batch_id),
            ) from exc
        if not isinstance(raw, list) or not isinstance(obs, list):
            raise FirstPassError(
                "CHECKPOINT_CORRUPT",
                f"Batch {batch_id} partial outputs are "
                f"malformed; refusing resume.",
                str(batch_id),
            )
        observed: list[int] = []
        for item in obs:
            frame = item.get("frame_index") if isinstance(
                item, dict) else None
            if (not isinstance(frame, int)
                    or isinstance(frame, bool)):
                raise FirstPassError(
                    "CHECKPOINT_CORRUPT",
                    f"Batch {batch_id} observations are "
                    f"malformed; refusing resume.",
                    str(batch_id),
                )
            observed.append(frame)
        claimed = record.get("completed_frame_indices")
        if claimed is not None and (
                not isinstance(claimed, list)
                or any(not isinstance(frame, int)
                       or isinstance(frame, bool)
                       for frame in claimed)
                or sorted(claimed) != sorted(observed)):
            raise FirstPassError(
                "CHECKPOINT_CORRUPT",
                f"Batch {batch_id} checkpoint disagrees with "
                f"its observation files; refusing resume.",
                str(batch_id),
            )
        attempts = record.get("attempts", [])
        if not isinstance(attempts, list):
            raise FirstPassError(
                "CHECKPOINT_CORRUPT",
                f"Batch {batch_id} attempt log is malformed; "
                f"refusing resume.",
                str(batch_id),
            )
        prior_input = record.get("input_tokens", 0)
        prior_output = record.get("output_tokens", 0)
        if (not isinstance(prior_input, int)
                or isinstance(prior_input, bool)
                or not isinstance(prior_output, int)
                or isinstance(prior_output, bool)):
            raise FirstPassError(
                "CHECKPOINT_CORRUPT",
                f"Batch {batch_id} token totals are "
                f"malformed; refusing resume.",
                str(batch_id),
            )
        prior_budget = record.get("budget", {})
        if not isinstance(prior_budget, dict):
            raise FirstPassError(
                "CHECKPOINT_CORRUPT",
                f"Batch {batch_id} budget block is "
                f"malformed; refusing resume.",
                str(batch_id),
            )
        for key in ("requests_submitted", "cost_known_usd_estimate",
                    "cost_unknown_requests"):
            value = prior_budget.get(key, 0)
            if (not isinstance(value, (int, float))
                    or isinstance(value, bool) or value < 0):
                raise FirstPassError(
                    "CHECKPOINT_CORRUPT",
                    f"Batch {batch_id} budget block is "
                    f"malformed; refusing resume.",
                    str(batch_id),
                )
        full = record.get("frame_indices", [])
        if not isinstance(full, list) or not full:
            raise FirstPassError(
                "CHECKPOINT_MISMATCH",
                f"Batch {batch_id} checkpoint lacks its frame "
                f"span; refusing resume.",
                str(batch_id),
            )
        index_range = record.get("index_range")
        if (not isinstance(index_range, list)
                or len(index_range) != 2
                or any(not isinstance(bound, int)
                       or isinstance(bound, bool)
                       for bound in index_range)):
            raise FirstPassError(
                "CHECKPOINT_CORRUPT",
                f"Batch {batch_id} index range is malformed; "
                f"refusing resume.",
                str(batch_id),
            )
        return {
            "raw": raw,
            "obs": obs,
            "attempts": list(attempts),
            "input_tokens": prior_input,
            "output_tokens": prior_output,
            "budget": prior_budget,
            "completed": observed,
            "frame_indices": list(full),
            "index_range": list(index_range),
        }

    def _scan_uncertain(
            self, checkpoints_dir: Path,
            index: dict[str, Any],
            expected: dict[str, list[int]]) -> dict[str, list[int]]:
        """Map started-but-uncheckpointed batches to their spans.

        A start marker without a checkpoint record means the
        batch may have submitted requests whose outcomes were
        never durably recorded. Markers for unknown or changed
        spans fail closed instead of being silently dropped.
        """
        uncertain: dict[str, list[int]] = {}
        pattern = f"{MARKER_PREFIX}*{MARKER_SUFFIX}"
        for marker_path in sorted(checkpoints_dir.glob(pattern)):
            try:
                marker = json.loads(
                    marker_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise FirstPassError(
                    "CHECKPOINT_CORRUPT",
                    "Batch start marker is unreadable: "
                    f"{marker_path.name}.",
                    marker_path.name,
                ) from exc
            batch_id = marker.get("batch_id") if isinstance(
                marker, dict) else None
            span = marker.get("frame_indices") if isinstance(
                marker, dict) else None
            if (not isinstance(batch_id, str) or not batch_id
                    or not isinstance(span, list) or not span
                    or any(not isinstance(frame, int)
                           or isinstance(frame, bool)
                           for frame in span)):
                raise FirstPassError(
                    "CHECKPOINT_CORRUPT",
                    "Batch start marker is malformed: "
                    f"{marker_path.name}.",
                    marker_path.name,
                )
            if batch_id in index:
                continue
            current = expected.get(batch_id)
            if current is None or set(span) - set(current):
                raise FirstPassError(
                    "CHECKPOINT_MISMATCH",
                    f"Uncertain batch {batch_id} does not match "
                    f"the current frame set; refusing resume.",
                    batch_id,
                )
            uncertain[batch_id] = list(span)
        return uncertain

    def _write_report(self, paths: dict[str, Path],
                      stats: dict[str, Any],
                      prepared: list[_PreparedFrame],
                      ledger: _BudgetLedger,
                      remaining: list[int],
                      uncertain: list[int]
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
            "billable": cfg.billable,
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
            "frames_unprocessed": stats["frames_unprocessed"],
            "stopped_reason": ledger.stopped_reason,
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
        state["budget"] = ledger.report_block(
            remaining, uncertain)
        _write_json(paths["state"], state)
        _write_json(paths["reports"] / "run_report.json", state)
        return state
