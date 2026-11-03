"""Live recipe-generation providers (Meta proposer, Baseten challenger).

Disabled by default: live calls require
``BAKERY_LIVE_RECIPE_GEN=1`` plus explicit per-request
confirmation, configured keys, and the exact authorized Baseten
model id. Credentials are read only here, server-side, and are
never returned in API payloads.

Verified contracts:
- Meta ``muse-spark-1.3`` via the Responses API
  (``client.responses.create`` with a JSON-schema text format,
  multimodal image data URLs, ``max_output_tokens``,
  no reasoning parameter).
- Baseten ``meta-models/muse-spark-open-1026`` via Chat
  Completions (``client.chat.completions.create`` with text
  messages, usage reported). Vision input to the challenger is
  unverified, so the challenger receives structured text
  evidence only.

Reasoning ``medium`` is a requested configuration recorded for
provenance, never a transmitted parameter.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from pydantic import BaseModel, Field

LIVE_FLAG = "BAKERY_LIVE_RECIPE_GEN"
META_KEY_VAR = "MODEL_API_KEY"
BASETEN_KEY_VAR = "BASETEN_API_KEY"
BASETEN_MODEL_VAR = "BAKERY_BASETEN_MODEL"
BASETEN_BASE_URL = "https://inference.baseten.co/v1"
BASETEN_AUTHORIZED_MODEL = "meta-models/muse-spark-open-1026"
SPARK_MODEL = "muse-spark-1.3"
REQUESTED_REASONING = "medium"

SPARK_RATES_USD_PER_M = (1.25, 4.25)

MAX_PROPOSER_IMAGES = 8
# Live proposer output budget for the next production-like run. The
# Responses API takes max_output_tokens as a pass-through parameter
# (proven at 1024 locally, 4096 in the 006 full run; no vendor cap
# was encountered in-repo). 50,000 sits below every externally
# reported muse-spark output cap and leaves headroom for reasoning
# plus recipe JSON after the observed 2048-token exhaustion
# (22,057 input / 2,048 output / 2,046 reasoning, no text). If the
# provider rejects the value, the failure is persisted with the
# requested budget and never retried.
DEFAULT_MAX_OUTPUT_TOKENS = 50000
# Upper bound enforced on explicit requests: the configured live
# budget. Requests above this are rejected before any billable call.
MAX_OUTPUT_TOKENS_LIMIT = 50000
PROVIDER_TIMEOUT_SECONDS = 120.0

RECIPE_SCHEMA_NAME = "RecipeProposal"
RECIPE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "recipe_name": {"type": ["string", "null"]},
        "ingredients": {"type": "array",
                        "items": {"type": "object"}},
        "steps": {"type": "array", "items": {"type": "object"}},
        "unresolved_questions": {"type": "array",
                                 "items": {"type": "object"}},
    },
    "required": ["ingredients", "steps"],
    "additionalProperties": False,
}


class LiveConfigError(ValueError):
    """Live generation is not authorized or not configured."""


class GenerateRecipeRequest(BaseModel):
    """Explicit generation mode; live never runs by default."""

    mode: str = "simulation"
    confirmed: bool = False
    max_rounds: int = 1
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS

    model_config = {"extra": "forbid"}


class RecipeProposal(BaseModel):
    """Structured recipe output every live provider must produce."""

    recipe_name: str | None = None
    ingredients: list[dict[str, Any]] = Field(default_factory=list)
    steps: list[dict[str, Any]] = Field(default_factory=list)
    unresolved_questions: list[dict[str, Any]] = Field(
        default_factory=list)

    model_config = {"extra": "forbid"}


def _load_local_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def choose_evidence_frames(frames: list[dict[str, Any]],
                           action_frames: set[int],
                           max_images: int) -> list[dict[str, Any]]:
    """Pick bounded visual evidence for culinary reconstruction.

    Priority: frames at action/tare moments first (culinary
    events, not only readable scales), then evenly spaced
    coverage across the video timeline. Deterministic.
    """
    by_index = {frame["frame_index"]: frame for frame in frames}
    ordered = sorted(by_index)
    chosen: list[int] = []
    for frame_index in sorted(action_frames):
        if frame_index in by_index and frame_index not in chosen:
            chosen.append(frame_index)
        if len(chosen) >= max_images:
            return [by_index[index] for index in chosen]
    if ordered:
        step = max(1, len(ordered) // max(max_images - len(chosen), 1))
        for frame_index in ordered[::step]:
            if frame_index not in chosen:
                chosen.append(frame_index)
            if len(chosen) >= max_images:
                break
    return [by_index[index] for index in chosen[:max_images]]


def live_available() -> tuple[bool, list[str]]:
    """Check live prerequisites without exposing secret values."""
    _load_local_env()
    blockers: list[str] = []
    if os.environ.get(LIVE_FLAG, "").lower() not in ("1", "true", "yes"):
        blockers.append(
            f"{LIVE_FLAG} is not enabled; live calls are disabled.")
    if not os.environ.get(META_KEY_VAR):
        blockers.append(f"{META_KEY_VAR} is not configured.")
    if not os.environ.get(BASETEN_KEY_VAR):
        blockers.append(f"{BASETEN_KEY_VAR} is not configured.")
    model = os.environ.get(
        BASETEN_MODEL_VAR, BASETEN_AUTHORIZED_MODEL)
    if model != BASETEN_AUTHORIZED_MODEL:
        blockers.append(
            f"{BASETEN_MODEL_VAR} must be exactly "
            f"{BASETEN_AUTHORIZED_MODEL}.")
    return (not blockers, blockers)


def live_credentials() -> dict[str, str]:
    """Return credentials only for server-side provider calls."""
    available, blockers = live_available()
    if not available:
        raise LiveConfigError("; ".join(blockers))
    meta_key = os.environ.get(META_KEY_VAR, "")
    baseten_key = os.environ.get(BASETEN_KEY_VAR, "")
    if not meta_key or not baseten_key:
        raise LiveConfigError("Provider credentials are not configured.")
    return {"meta": meta_key, "baseten": baseten_key}


def check_live_or_raise(request: GenerateRecipeRequest) -> None:
    if request.mode == "simulation":
        if not 1 <= request.max_rounds <= 3:
            raise LiveConfigError(
                "max_rounds must be between 1 and 3.")
        return
    if request.mode != "live":
        raise LiveConfigError(
            "mode must be simulation or live.")
    if not request.confirmed:
        raise LiveConfigError(
            "Live generation requires explicit confirmation.")
    if not 1 <= request.max_rounds <= 3:
        raise LiveConfigError(
            "max_rounds must be between 1 and 3.")
    if not 1 <= request.max_output_tokens <= MAX_OUTPUT_TOKENS_LIMIT:
        raise LiveConfigError(
            "max_output_tokens must be between 1 and "
            f"{MAX_OUTPUT_TOKENS_LIMIT}.")
    live_credentials()


def cost_entry(*, provider: str, model: str, operation: str,
               usage: dict[str, Any] | None,
               rates: tuple[float, float] | None,
               rates_source: str, request_id: str | None,
               status: str) -> dict[str, Any]:
    """One stage-level cost record; unknown stays null, never zero."""
    entry: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "operation": operation,
        "input_tokens": (usage or {}).get("input_tokens"),
        "output_tokens": (usage or {}).get("output_tokens"),
        "reasoning_tokens": (usage or {}).get("reasoning_tokens"),
        "rates_usd_per_m": list(rates) if rates else None,
        "rates_source": rates_source,
        "estimated_cost_usd": None,
        "cost_status": "unknown",
        "request_id": request_id,
        "status": status,
    }
    if (rates is not None and isinstance(
            entry["input_tokens"], int)
            and isinstance(entry["output_tokens"], int)):
        entry["estimated_cost_usd"] = round(
            entry["input_tokens"] * rates[0] / 1_000_000
            + entry["output_tokens"] * rates[1] / 1_000_000, 4)
        entry["cost_status"] = "estimated_from_configured_rates"
    return entry


class MetaSparkLiveProposer:
    """Live proposer: Meta ``muse-spark-1.3`` Responses API."""

    provider_id = "meta"
    model_id = SPARK_MODEL
    simulated = False

    def __init__(self, client: Any = None,
                 client_factory: Callable[[], Any] | None = None,
                 max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
                 max_images: int = MAX_PROPOSER_IMAGES) -> None:
        self._client = client
        self._client_factory = client_factory
        self.max_output_tokens = max_output_tokens
        self.max_images = max_images
        self.calls = 0
        self.last_model: str | None = None
        self.last_usage: dict[str, Any] | None = None
        self.last_request_id: str | None = None
        self.last_output_length: int | None = None
        self.last_output_excerpt: str | None = None
        self.last_status: str | None = None
        self.last_incomplete_reason: str | None = None

    def propose(self, evidence_text: str, image_paths: list[Any],
                api_key: str) -> tuple[RecipeProposal, dict[str, Any]]:
        from bakery_intelligence.inference.meta_spark import (
            MetaSparkClient,
        )
        client = self._client or (
            self._client_factory() if self._client_factory
            else MetaSparkClient(
                model=self.model_id,
                timeout_seconds=PROVIDER_TIMEOUT_SECONDS))
        self.calls += 1
        self.last_model = self.model_id
        chosen = list(image_paths)[: self.max_images]
        prompt = (
            "Propose a recipe draft grounded only in this evidence. "
            "Reply with exactly one compact JSON object and no prose: "
            "{\"recipe_name\": str|null, \"ingredients\": "
            "[{\"ingredient_id\": str, \"name\": str|null, "
            "\"quantity\": {\"value\": number|null, \"unit\": "
            "str|null, \"derivation\": str}, \"provenance\": "
            "\"ai-extracted\"}], \"steps\": [{\"description\": str, "
            "\"start_seconds\": number|null, \"end_seconds\": "
            "number|null, \"frame_refs\": [int], \"provenance\": "
            "\"ai-extracted\"}], \"unresolved_questions\": "
            "[{\"code\": str, \"message\": str}]}. Keep arrays short. "
            f"Requested reasoning: {REQUESTED_REASONING} "
            "(informational only). "
            f"Evidence: {evidence_text}")
        try:
            response = client.submit(
                prompt,
                chosen,
                api_key,
                self.max_output_tokens,
                RECIPE_JSON_SCHEMA,
                RECIPE_SCHEMA_NAME,
            )
        except TypeError as schema_exc:
            message = str(schema_exc)
            if ("response_schema" not in message
                    and "positional" not in message
                    and "argument" not in message):
                raise
            response = client.submit(
                prompt_text=prompt,
                image_paths=chosen,
                api_key=api_key,
                max_output_tokens=self.max_output_tokens,
            )
        # Actual contract: InferenceResponse(output_text, usage,
        # request_id, model, ...). Usage fields may be None when the
        # provider omits them; unknown stays null, never zero.
        raw_usage = getattr(response, "usage", None)
        def _as_int_or_none(value: Any) -> int | None:
            if isinstance(value, bool) or not isinstance(value, int):
                return None
            return value

        usage = {
            "input_tokens": _as_int_or_none(
                getattr(raw_usage, "input_tokens", None)),
            "output_tokens": _as_int_or_none(
                getattr(raw_usage, "output_tokens", None)),
            "reasoning_tokens": _as_int_or_none(getattr(
                raw_usage, "reasoning_output_tokens", None)),
        }
        request_id = getattr(response, "request_id", None)
        raw_text = getattr(response, "output_text", None)
        response_model = getattr(response, "model", None)
        status = getattr(response, "status", None)
        incomplete_reason = getattr(response, "incomplete_reason", None)
        self.last_usage = usage
        self.last_request_id = (
            request_id if isinstance(request_id, str) else None)
        self.last_model = (
            response_model if isinstance(response_model, str)
            else self.model_id)
        self.last_status = status if isinstance(status, str) else None
        self.last_incomplete_reason = (
            incomplete_reason if isinstance(incomplete_reason, str)
            else None)
        self.last_output_length = (
            len(raw_text) if isinstance(raw_text, str) else None)
        self.last_output_excerpt = (
            raw_text[:2000] if isinstance(raw_text, str) else None)
        context = (
            f"(request_id={self.last_request_id!r}, "
            f"model={self.last_model!r}, "
            f"status={self.last_status!r}, "
            f"incomplete_reason={self.last_incomplete_reason!r}, "
            f"output_length={self.last_output_length!r}, "
            f"usage={self.last_usage!r})")
        if not isinstance(raw_text, str):
            raise LiveConfigError(
                "Live proposer returned unusable output " + context)
        if not raw_text.strip():
            if (self.last_status == "incomplete" or
                    self.last_incomplete_reason is not None):
                raise LiveConfigError(
                    "Live proposer response was incomplete; no recipe "
                    "text was returned " + context)
            raise LiveConfigError(
                "Live proposer returned empty output " + context)
        if self.last_status is not None and self.last_status != "completed":
            raise LiveConfigError(
                "Live proposer response was not completed " + context)
        try:
            payload = json.loads(raw_text)
        except ValueError as exc:
            raise LiveConfigError(
                "Live proposer returned non-JSON output "
                + context) from exc
        if not isinstance(payload, dict):
            raise LiveConfigError(
                "Live proposer output was not a JSON object "
                + context)
        try:
            proposal = RecipeProposal(**payload)
        except (ValueError, TypeError) as exc:
            raise LiveConfigError(
                "Live proposer output failed validation "
                + context) from exc
        return proposal, usage


def _extract_challenger_json(content: str) -> Any | None:
    """Extract one JSON value from challenger text, fences included.

    Accepts raw JSON, Markdown-fenced JSON, or JSON embedded in brief
    prose. Returns the decoded value, or None when no JSON object or
    array can be recovered. Callers classify the failure.
    """
    text = content.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        pass
    if text.startswith("```"):
        lines = text.splitlines()
        body = lines[1:] if len(lines) > 1 else []
        if body and body[-1].strip().startswith("```"):
            body = body[:-1]
        else:
            body = [line for line in body
                    if not line.strip().startswith("```")]
        try:
            return json.loads("\n".join(body).strip())
        except ValueError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except ValueError:
            pass
    return None


class BasetenLiveChallenger:
    """Live challenger: Baseten chat completions, text evidence only."""

    provider_id = "baseten"
    model_id = BASETEN_AUTHORIZED_MODEL
    simulated = False

    def __init__(self, client: Any = None,
                 client_factory: Callable[[], Any] | None = None,
                 max_tokens: int = 4096) -> None:
        self._client = client
        self._client_factory = client_factory
        self.max_tokens = max_tokens
        self.calls = 0
        self.last_usage: dict[str, Any] | None = None
        self.last_request_id: str | None = None
        self.last_finish_reason: str | None = None
        self.last_content_length: int | None = None
        self.last_content_excerpt: str | None = None

    def _get_client(self, api_key: str) -> Any:
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            self._client = self._client_factory()
            return self._client
        from openai import OpenAI
        self._client = OpenAI(
            api_key=api_key, base_url=BASETEN_BASE_URL,
            timeout=PROVIDER_TIMEOUT_SECONDS)
        return self._client

    def critique(self, claims: list[dict[str, Any]],
                 evidence_summary: dict[str, Any], api_key: str,
                 retries: int = 1) -> tuple[list[dict[str, Any]],
                                            dict[str, Any]]:
        client = self._get_client(api_key)
        self.calls += 1
        attempts = 0
        while True:
            attempts += 1
            try:
                response = client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": json.dumps({
                        "task": "Critique each recipe claim. Reply with "
                                "exactly one JSON object and no other "
                                "text. Do not wrap it in code fences. "
                                "Schema: {\"objections\": "
                                "[{\"claim_id\": str, \"objection\": str, "
                                "\"contradiction\": bool}]}. Keep each "
                                "objection to one sentence. Requested "
                                f"reasoning: {REQUESTED_REASONING} "
                                "(informational only).",
                        "claims": claims,
                        "evidence": evidence_summary,
                    }, sort_keys=True, default=str)}],
                    max_tokens=self.max_tokens,
                )
                break
            except Exception:
                if attempts > retries:
                    raise
        self.last_request_id = getattr(response, "id", None)
        if not isinstance(self.last_request_id, str):
            self.last_request_id = None
        usage = getattr(response, "usage", None)
        if usage is not None and hasattr(usage, "model_dump"):
            dumped: dict[str, Any] = usage.model_dump() or {}
        elif isinstance(usage, dict):
            dumped = usage
        else:
            dumped = {}
        def _as_int_or_none(value: Any) -> int | None:
            if isinstance(value, bool) or not isinstance(value, int):
                return None
            return value
        self.last_usage = {
            "input_tokens": _as_int_or_none(dumped.get("prompt_tokens")),
            "output_tokens": _as_int_or_none(
                dumped.get("completion_tokens")),
            "reasoning_tokens": None,
        }
        choices = getattr(response, "choices", None) or []
        choice = choices[0] if choices else None
        message = getattr(choice, "message", None) if choice else None
        content = getattr(message, "content", None) if message else None
        if content is None:
            content = ""
        finish_reason = getattr(choice, "finish_reason", None)
        self.last_finish_reason = (
            finish_reason if isinstance(finish_reason, str) else None)
        self.last_content_length = (
            len(content) if isinstance(content, str) else None)
        self.last_content_excerpt = (
            content[:2000] if isinstance(content, str) else None)
        context = (
            f"(request_id={self.last_request_id!r}, "
            f"finish_reason={self.last_finish_reason!r}, "
            f"content_length={self.last_content_length!r}, "
            f"usage={self.last_usage!r})")
        if not isinstance(content, str) or not content.strip():
            if self.last_finish_reason == "length":
                raise LiveConfigError(
                    "Live challenger response was truncated "
                    f"(failure_category=truncated) {context}")
            raise LiveConfigError(
                "Live challenger returned empty output "
                f"(failure_category=empty) {context}")
        if self.last_finish_reason == "length":
            raise LiveConfigError(
                "Live challenger response was truncated "
                f"(failure_category=truncated) {context}")
        payload = _extract_challenger_json(content)
        if payload is None:
            raise LiveConfigError(
                "Live challenger returned non-JSON output "
                f"(failure_category=non_json) {context}")
        if not isinstance(payload, dict):
            raise LiveConfigError(
                "Live challenger output was not a JSON object "
                f"(failure_category=not_object) {context}")
        objections = payload.get("objections")
        if not isinstance(objections, list):
            raise LiveConfigError(
                "Live challenger output lacks an objections list "
                f"(failure_category=missing_objections) {context}")
        for item in objections:
            if (not isinstance(item, dict)
                    or not isinstance(item.get("claim_id"), str)
                    or not item.get("claim_id")
                    or not isinstance(item.get("objection"), str)
                    or not isinstance(item.get("contradiction"), bool)):
                raise LiveConfigError(
                    "Live challenger objections failed validation "
                    f"(failure_category=invalid_objections) {context}")
        return objections, self.last_usage or {}
