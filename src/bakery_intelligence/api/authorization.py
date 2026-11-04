"""Per-run billable authorization grants (M2-06C.5/S5-A).

A grant is an explicit operator-authored document binding one
job submission to an exact billable configuration: workflow,
provider, model, request cap, token budget, frame bound, and
cost controls. Grants never carry credentials; the API key is
loaded from the process environment at the execution
boundary only.

This module is deliberately standalone (no package imports)
so both the job service and the worker executor can share
it without import cycles. Structural validation lives here;
workflow binding (grant matches the submitted/recorded
workflow) is enforced by the callers that own the workflow
constants.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

GRANT_FORMAT = "bakery-billable-grant/1"

# S5-A narrow scope: only the Responses model proven through
# the offline adapter path. Other model ids are refused even
# with a well-formed grant.
ALLOWED_MODELS = frozenset({"muse-spark-1.3"})

# Single supported provider (mirrors META_PROVIDER_ID in the
# pipeline module; kept literal here to avoid a cycle).
GRANT_PROVIDER = "meta"

# Key lookup order mirrors experiments/006 resolve_api_key:
# task convention first, repository convention second.
ENV_KEYS = ("META_API_KEY", "MODEL_API_KEY")

DEFAULT_TIMEOUT_SECONDS = 120.0

_REQUIRED_KEYS = frozenset({
    "grant_format", "workflow", "provider", "model",
    "max_requests", "token_budget", "max_frames",
    "operator", "created_at", "confirmation",
})
_OPTIONAL_KEYS = frozenset({
    "timeout_seconds", "store", "cost_ceiling_usd",
    "rates_usd_per_m", "strict_cost", "firstpass_engine",
})
_KNOWN_KEYS = _REQUIRED_KEYS | _OPTIONAL_KEYS


class GrantError(ValueError):
    """A billable authorization grant is absent or invalid."""


@dataclass(frozen=True)
class Grant:
    """Validated billable authorization (key-free by shape)."""

    workflow: str
    provider: str
    model: str
    max_requests: int
    token_budget: int
    max_frames: int
    operator: str
    created_at: str
    confirmation: str
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    store: bool = True
    cost_ceiling_usd: float | None = None
    rates_usd_per_m: tuple[float, float] | None = None
    strict_cost: bool = False
    firstpass_engine: str | None = None


def confirmation_sentence(workflow: str, model: str,
                           max_requests: int, token_budget: int,
                           max_frames: int, operator: str) -> str:
    """Build the exact operator confirmation sentence."""
    return (
        f"authorize {workflow} model {model} "
        f"up to {max_requests} requests, "
        f"{token_budget} output tokens, "
        f"{max_frames} frames, operator {operator}"
    )


def _is_num(value: Any) -> bool:
    return (isinstance(value, (int, float))
            and not isinstance(value, bool))


def parse_grant(payload: Any) -> Grant:
    """Validate a grant document; reject anything ambiguous.

    Unknown keys are rejected so a mistyped constraint name
    can never silently drop a billing bound.
    """
    if not isinstance(payload, dict):
        raise GrantError(
            f"Grant must be a JSON object, got {type(payload).__name__}.")
    unknown = set(payload) - _KNOWN_KEYS
    if unknown:
        raise GrantError(
            "Grant has unknown keys "
            f"({', '.join(sorted(str(key) for key in unknown))}); "
            "refusing to ignore possible constraints.")
    missing = _REQUIRED_KEYS - set(payload)
    if missing:
        raise GrantError(
            "Grant is missing required keys "
            f"({', '.join(sorted(missing))}).")
    if payload["grant_format"] != GRANT_FORMAT:
        raise GrantError(
            f"Grant format must be {GRANT_FORMAT!r}, "
            f"got {payload['grant_format']!r}.")
    workflow = payload["workflow"]
    if not isinstance(workflow, str) or not workflow:
        raise GrantError("Grant workflow must be a non-empty str.")
    provider = payload["provider"]
    if provider != GRANT_PROVIDER:
        raise GrantError(
            f"Grant provider must be {GRANT_PROVIDER!r}, "
            f"got {provider!r}.")
    model = payload["model"]
    if model not in ALLOWED_MODELS:
        raise GrantError(
            f"Grant model {model!r} is not authorized for "
            "billable execution in S5-A.")
    max_requests = payload["max_requests"]
    if (isinstance(max_requests, bool)
            or not isinstance(max_requests, int)
            or max_requests < 1):
        raise GrantError(
            "Grant max_requests must be a positive int, "
            f"got {max_requests!r}.")
    token_budget = payload["token_budget"]
    if (isinstance(token_budget, bool)
            or not isinstance(token_budget, int)
            or token_budget < 1):
        raise GrantError(
            "Grant token_budget must be a positive int, "
            f"got {token_budget!r}.")
    max_frames = payload["max_frames"]
    if (isinstance(max_frames, bool)
            or not isinstance(max_frames, int)
            or max_frames < 1):
        raise GrantError(
            "Grant max_frames must be a positive int, "
            f"got {max_frames!r}.")
    timeout = payload.get("timeout_seconds",
                          DEFAULT_TIMEOUT_SECONDS)
    if (not _is_num(timeout) or not math.isfinite(timeout)
            or timeout <= 0):
        raise GrantError(
            "Grant timeout_seconds must be a positive finite "
            f"number, got {timeout!r}.")
    store = payload.get("store", True)
    if not isinstance(store, bool):
        raise GrantError(
            f"Grant store must be a bool, got {store!r}.")
    ceiling = payload.get("cost_ceiling_usd")
    if ceiling is not None and (
            not _is_num(ceiling) or not math.isfinite(ceiling)
            or ceiling < 0):
        raise GrantError(
            "Grant cost_ceiling_usd must be a non-negative "
            f"finite number or null, got {ceiling!r}.")
    rates = payload.get("rates_usd_per_m")
    rates_pair: tuple[float, float] | None = None
    if rates is not None:
        if (not isinstance(rates, (list, tuple))
                or len(rates) != 2
                or any(not _is_num(rate)
                       or not math.isfinite(rate) or rate < 0
                       for rate in rates)):
            raise GrantError(
                "Grant rates_usd_per_m must be null or a pair "
                f"of non-negative finite numbers, got {rates!r}.")
        rates_pair = (float(rates[0]), float(rates[1]))
    strict = payload.get("strict_cost", False)
    if not isinstance(strict, bool):
        raise GrantError(
            f"Grant strict_cost must be a bool, got {strict!r}.")
    engine = payload.get("firstpass_engine")
    if engine is not None:
        if (not isinstance(engine, str) or not engine.strip()
                or len(engine) > 64):
            raise GrantError(
                "Grant firstpass_engine must be a non-empty "
                f"engine id, got {engine!r}.")
        engine = engine.strip()
    operator = payload["operator"]
    if not isinstance(operator, str) or not operator:
        raise GrantError(
            "Grant operator must be a non-empty str.")
    created_at = payload["created_at"]
    if not isinstance(created_at, str) or not created_at:
        raise GrantError(
            "Grant created_at must be a non-empty str.")
    try:
        datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise GrantError(
            "Grant created_at must be ISO-8601, "
            f"got {created_at!r}.") from exc
    expected = confirmation_sentence(
        workflow, model, max_requests, token_budget,
        max_frames, operator)
    if payload["confirmation"] != expected:
        raise GrantError(
            "Grant confirmation does not match the granted "
            f"parameters; expected {expected!r}.")
    return Grant(
        workflow=workflow, provider=provider, model=model,
        max_requests=max_requests, token_budget=token_budget,
        max_frames=max_frames, operator=operator,
        created_at=created_at,
        confirmation=payload["confirmation"],
        timeout_seconds=float(timeout), store=store,
        cost_ceiling_usd=(
            None if ceiling is None else float(ceiling)),
        rates_usd_per_m=rates_pair, strict_cost=strict,
        firstpass_engine=engine,
    )


def grant_to_json(grant: Grant) -> str:
    """Serialize a validated grant canonically for storage."""
    payload: dict[str, Any] = {
        "grant_format": GRANT_FORMAT,
        "workflow": grant.workflow,
        "provider": grant.provider,
        "model": grant.model,
        "max_requests": grant.max_requests,
        "token_budget": grant.token_budget,
        "max_frames": grant.max_frames,
        "operator": grant.operator,
        "created_at": grant.created_at,
        "confirmation": grant.confirmation,
        "timeout_seconds": grant.timeout_seconds,
        "store": grant.store,
        "cost_ceiling_usd": grant.cost_ceiling_usd,
        "rates_usd_per_m": (
            list(grant.rates_usd_per_m)
            if grant.rates_usd_per_m is not None else None),
        "strict_cost": grant.strict_cost,
    }
    # The confirmation sentence intentionally excludes the
    # engine: stored grants predate the key and must keep
    # parsing byte-identically. Engine membership and
    # grant/env conflicts are enforced at execution time.
    if grant.firstpass_engine is not None:
        payload["firstpass_engine"] = grant.firstpass_engine
    return json.dumps(payload, sort_keys=True)


def resolve_api_key(env: dict[str, str] | None = None) -> str:
    """Load the provider key at the execution boundary.

    Never persisted by callers; empty values count as
    missing and fail explicitly before any dispatch.
    """
    source = env if env is not None else os.environ
    for name in ENV_KEYS:
        value = source.get(name)
        if value:
            return value
    raise GrantError(
        "CREDENTIAL_MISSING: set META_API_KEY (or "
        "MODEL_API_KEY) in the worker process environment; "
        "refusing billable execution without a key.")


def redact_key(text: str, key: str) -> str:
    """Remove key material from text destined for storage."""
    if not key:
        return text
    return text.replace(key, "***")
