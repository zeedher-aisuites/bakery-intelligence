"""Reusable inference lifecycle policy (Exp. 006).

Canonical home for terminal outcomes, output-token budget
escalation, and cost estimation shared by first-pass inference and
selective verification. Pure policy: no transport, no credentials.
Defaults preserve the historical single-budget behavior exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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

# Historical production behavior: one 1024-token attempt, no escalation.
DEFAULT_BUDGETS = (1024,)
# Experimentally exercised escalation (Exp. 010 probe, one timestamp).
EXPERIMENTAL_BUDGETS = (1024, 2048, 3072)


def is_token_limit_reason(reason: Any) -> bool:
    """True only for an explicit output-token-limit reason."""
    if reason is None:
        return False
    text = str(reason).lower()
    return "max_output_tokens" in text or (
        "output_token" in text
        and ("limit" in text or "max" in text or "exhaust" in text)
    )


@dataclass(frozen=True)
class BudgetPolicy:
    """Output-token escalation schedule with request guardrails."""

    budgets: tuple[int, ...] = DEFAULT_BUDGETS
    max_requests: int | None = None

    def __post_init__(self) -> None:
        if not self.budgets or any(
            not isinstance(budget, int) or isinstance(budget, bool)
            or budget <= 0 for budget in self.budgets
        ):
            raise ValueError(
                "budgets must be a non-empty tuple of positive ints."
            )
        if self.max_requests is not None and (
            not isinstance(self.max_requests, int)
            or self.max_requests <= 0
        ):
            raise ValueError("max_requests must be a positive int.")


def decide_attempt(status: str | None, incomplete_reason: Any,
                   raw_text: Any, parse_ok: bool,
                   attempts_made: int, policy: BudgetPolicy,
                   abstained_hint: bool = False,
                   transport_error: str | None = None) -> dict[str, Any]:
    """Map one attempt outcome to retry/stop plus terminal state.

    Retries only on explicit token exhaustion with budgets left. The
    human reference and model confidence never influence the decision.
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
    """None-safe cost estimate from reported usage and rates.

    Returns None when usage or rates are unavailable. Estimates only;
    never confirmed billing.
    """
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
