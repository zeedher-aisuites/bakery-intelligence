"""Sequential context-aware multi-image FirstPass (V2 experiment).

Experimental alternative to the single-frame ``FirstPassRunner``
(``firstpass.py``): fixed groups of consecutive frames share one
provider request, and a compact structured rolling context
propagates from each group to the next. For a 645-frame video
with the default group size of 10, a successful run issues 65
requests instead of 645.

Scope and safety properties:

- V1 is untouched and remains the default. On the
  experiment branch, ``pipeline.py`` imports this module
  only when ``firstpass_engine="contextual_batch"`` is
  selected (developer-only setting, never the default);
  no registered workflow selects it implicitly.
- The module performs no network access itself and wires no
  live client: inference flows only through an injected
  client factory, so automated tests run fully offline.
- Execution is strictly sequential: groups dispatch in input
  order, one request per group, no concurrency, no adaptive
  frame selection.
- Billable runs enforce single-submit invariants (one attempt
  per group, no escalation replay, no uncertain replay),
  mirroring the V1 S3 posture.
- Missing or contradictory evidence stays visible: failed
  groups record an explicit gap so later groups never infer
  continuity across unobserved footage.

Downstream compatibility: per-frame records reuse
``build_observation`` key-for-key, so existing temporal and
reconstruction loaders consume V2 observations unchanged.
V2-only provenance travels inside the ``call`` mapping, which
loaders already treat as opaque, plus dedicated group-level
artifacts under the V2 output directory.

Experiment-branch hardening (divergence from the V2 source):
``max_validation_attempts`` replays a group whose provider
response was fully received but unusable (malformed JSON,
truncation, schema mismatch); ambiguous transport is never
replayed. ``max_consecutive_exhausted`` stops the run after
consecutive exhausted groups. Both default to the original
behavior (single attempt, no early stop); the pipeline
builder enables 3 attempts and a 2-group stop for
contextual batches.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from bakery_intelligence.inference.firstpass import (
    MODEL_OUTPUT_FIELDS,
    RETRYABLE_CODES,
    STOP_COST_CEILING,
    STOP_COST_RESERVE,
    STOP_REQUEST_LIMIT,
    STOP_STRICT_UNKNOWN_COST,
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
    build_observation,
    estimate_cost,
    is_token_limit_reason,
    parse_model_json,
    prompt_hash,
)
from bakery_intelligence.inference.meta_spark import (
    MAX_IMAGES_PER_REQUEST,
    MODEL_SPARK,
    OBSERVATION_JSON_SCHEMA,
)

# Experiment identity. Recorded as ``pass_id`` so V1 and V2
# observations never mix silently in shared downstream tools.
PASS_ID_V2 = "firstpass_v2"

# Default group shape: 10 consecutive frames per request, with
# the final group allowed to be partial.
DEFAULT_GROUP_SIZE = 10

# Default output budget for one group request. Ten per-frame
# observations plus a rolling context need materially more
# headroom than the V1 single-frame budget (1024); 4096 matches
# the largest budget proven in Exp. 006. There is no token
# escalation in V2: an exhausted budget is an explicit group
# failure so pilots size the budget from evidence.
DEFAULT_GROUP_TOKEN_BUDGET = 4096

# Rolling-context bound. Roughly 500-800 tokens at ~4 chars
# per token; the model is asked for ~700 tokens and the
# runner hard-truncates anything longer.
CONTEXT_TARGET_TOKENS = 700
CONTEXT_MAX_CHARS = 3200
CONTEXT_TRUNCATION_MARKER = "[context truncated to bound]"

# Group-start marker files, mirroring the V1 S2 convention. A
# marker without a matching checkpoint record means the group
# may have submitted a request whose outcome was never
# durably recorded: uncertain.
MARKER_PREFIX = "started_"
MARKER_SUFFIX = ".json"

# Group response contract name for structured decoding.
GROUP_SCHEMA_NAME = "FrameGroupObservation"

# Client/provider codes that signal the image count itself was
# rejected. These halt the run with a clear error instead of
# burning further doomed requests.
IMAGE_LIMIT_CODES = frozenset({
    "TOO_MANY_IMAGES",
    "PAYLOAD_TOO_LARGE",
    "BAD_REQUEST",
})

# Stop reason recorded when consecutive groups exhaust their
# validation attempts without a usable response.
STOP_CONSECUTIVE_EXHAUSTED = "consecutive_group_exhaustion"

JPEG_START = b"\xff\xd8"
JPEG_END = b"\xff\xd9"
PNG_START = b"\x89PNG"


class GroupClient(Protocol):
    """Client seam for one multi-image group request.

    ``MetaSparkClient.submit`` satisfies this protocol: the
    schema arguments are optional there and required here so
    a group prompt can never be constrained by the
    single-frame observation schema by accident.
    """

    def submit(
        self,
        prompt_text: str,
        image_paths: Sequence[Path],
        api_key: str,
        max_output_tokens: int,
        response_schema: dict[str, Any] | None = None,
        response_schema_name: str | None = None,
    ) -> InferenceResponse:
        """Send one multimodal request for a frame group."""
        ...


def build_group_schema() -> dict[str, Any]:
    """Return the structured-output contract for group replies.

    Each frame entry carries a full single-frame observation
    (same property set as ``OBSERVATION_JSON_SCHEMA``, so
    downstream loaders keep working) plus three explicit
    evidence tiers: ``verified`` (directly visible),
    ``uncertain`` (ambiguous interpretation), and
    ``hypotheses`` (informed guesses requiring later
    confirmation). ``rolling_context`` is the compact state
    the next group receives. Flat object without ``$ref``:
    recursive schemas return HTTP 400.
    """
    observation_props = OBSERVATION_JSON_SCHEMA["properties"]
    return {
        "type": "object",
        "properties": {
            "frames": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "frame_index": {"type": "integer"},
                        "observation": {
                            "type": "object",
                            "properties": dict(observation_props),
                            "required": list(MODEL_OUTPUT_FIELDS),
                            "additionalProperties": False,
                        },
                        "verified": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "uncertain": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "hypotheses": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "frame_index", "observation", "verified",
                        "uncertain", "hypotheses",
                    ],
                    "additionalProperties": False,
                },
            },
            "rolling_context": {"type": "string"},
            "context_notes": {"type": ["string", "null"]},
        },
        "required": ["frames", "rolling_context"],
        "additionalProperties": False,
    }


GROUP_JSON_SCHEMA = build_group_schema()

# Frozen group prompt. The per-request sections (frame table
# and incoming context) are appended by
# ``build_group_prompt``; the rules below mirror the V1
# pipeline prompt so V1/V2 evidence stays comparable.
V2_BASE_PROMPT = """Analyze a short sequence of consecutive bakery-video frames for a baking-evidence study.
The attached images are ordered: IMAGE 1 is the earliest frame, the last image is the latest.
Reply with exactly one JSON object and no other text, using these keys:
{
  "frames": [
    {
      "frame_index": 0,
      "observation": {
        "visible_objects": ["short noun phrases, empty list if unclear"],
        "scale_display_verbatim": "exact digits/segments on the scale display, or null",
        "scale_readable": true/false,
        "scale_state": "stable/changing/zero/off/occluded/unknown",
        "tare_action_visible": true/false,
        "addition_or_removal_visible": "addition/removal/none/unknown",
        "container_description": "short phrase or null",
        "container_changed_hint": true/false,
        "confidence": "high/medium/low",
        "ambiguity_notes": "short note or null"
      },
      "verified": ["facts directly visible in THIS frame"],
      "uncertain": ["ambiguous interpretations of THIS frame"],
      "hypotheses": ["informed guesses needing later confirmation"]
    }
  ],
  "rolling_context": "compact state for the next group, about 700 tokens max",
  "context_notes": "short note or null"
}
Rules: emit one frames entry per attached image, in the same order, with the exact frame_index given below;
report only what is visible in each frame; use null when evidence is insufficient; never infer ingredient
identity, units, or net mass; never convert a display reading into a quantity; a negative display stays a
display string, never a negative quantity; never invent weights or continuity across frames you cannot see;
the rolling_context must summarize only this group plus the incoming context, flagging what is verified,
what is uncertain, and what is hypothesized."""


def estimate_context_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token), labeled as such."""
    return max(0, len(text) // 4)


def bound_context(text: str,
                  max_chars: int = CONTEXT_MAX_CHARS) -> str:
    """Truncate rolling context to the bound with a marker."""
    if len(text) <= max_chars:
        return text
    room = max_chars - len(CONTEXT_TRUNCATION_MARKER) - 1
    return text[:room].rstrip() + "\n" + CONTEXT_TRUNCATION_MARKER


def group_frames(frames: Sequence[FrameRef],
                 group_size: int) -> list[list[FrameRef]]:
    """Chunk frames into fixed consecutive groups (last may vary).

    The input order is preserved verbatim; callers sort before
    grouping when a canonical order is wanted.
    """
    if group_size < 1:
        raise FirstPassError(
            "INVALID_CONFIG",
            f"group_size must be >= 1, got {group_size!r}.")
    return [list(frames[position:position + group_size])
            for position in range(0, len(frames), group_size)]


def group_id_for(sequence: int) -> str:
    """Zero-padded group identifier (``group_0000``)."""
    return f"group_{sequence:04d}"


def build_group_prompt(
    *,
    frames: Sequence[FrameRef],
    group_sequence: int,
    total_groups: int,
    incoming_context: str | None,
    context_kind: str,
) -> str:
    """Assemble the full prompt for one group request.

    ``context_kind`` is ``"global"`` for the initial video
    analysis on the first group (when available), ``"rolling"``
    for propagated context, or ``"none"`` when no context
    exists; it is recorded in artifacts for provenance.
    """
    lines = [V2_BASE_PROMPT, "",
             f"GROUP {group_sequence + 1} OF {total_groups}: "
             f"{len(frames)} image(s), in this exact order:"]
    for position, frame in enumerate(frames, start=1):
        lines.append(
            f"IMAGE {position}: frame_index={frame.frame_index}, "
            f"timestamp_seconds={frame.timestamp_seconds}")
    if incoming_context:
        lines += ["", f"INCOMING {context_kind.upper()} CONTEXT:",
                  incoming_context]
    else:
        lines += ["", "INCOMING CONTEXT: none (first group, no "
                  "prior state; treat every frame on its own "
                  "visible evidence)."]
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class GroupValidation:
    """Outcome of structural validation for one group reply."""

    ok: bool
    entries: tuple[dict[str, Any], ...] = ()
    rolling_context: str = ""
    context_notes: str | None = None
    error: str | None = None


def validate_group_response(
    raw_text: str,
    expected_indices: Sequence[int],
) -> GroupValidation:
    """Validate structure, completeness, and ordering.

    Acceptance requires: a JSON object, a ``frames`` array
    with exactly one entry per requested frame, ``frame_index``
    values matching the request in order, every observation
    carrying all ``MODEL_OUTPUT_FIELDS`` keys, evidence tiers
    as string lists, and a non-empty ``rolling_context``.
    Anything else is rejected with a reason; partial payloads
    never become observations.
    """
    parsed, parse_error = parse_model_json(raw_text)
    if parsed is None:
        return GroupValidation(
            ok=False, error=f"unparseable_json:{parse_error}")
    entries = parsed.get("frames")
    if not isinstance(entries, list):
        return GroupValidation(
            ok=False, error="frames_not_a_list")
    if len(entries) != len(expected_indices):
        return GroupValidation(
            ok=False,
            error=(f"frame_count_mismatch: got {len(entries)}, "
                   f"expected {len(expected_indices)}"))
    for position, (entry, want) in enumerate(
            zip(entries, expected_indices)):
        if not isinstance(entry, dict):
            return GroupValidation(
                ok=False,
                error=f"frame_entry_not_an_object:position_{position}")
        if entry.get("frame_index") != want:
            return GroupValidation(
                ok=False,
                error=(f"frame_order_mismatch:position_{position}_"
                       f"got_{entry.get('frame_index')!r}_"
                       f"expected_{want}"))
        observation = entry.get("observation")
        if not isinstance(observation, dict):
            return GroupValidation(
                ok=False,
                error=f"observation_not_an_object:frame_{want}")
        missing = [name for name in MODEL_OUTPUT_FIELDS
                   if name not in observation]
        if missing:
            return GroupValidation(
                ok=False,
                error=(f"observation_missing_keys:frame_{want}:"
                       f"{','.join(missing)}"))
        for tier in ("verified", "uncertain", "hypotheses"):
            values = entry.get(tier)
            if (not isinstance(values, list)
                    or any(not isinstance(item, str)
                           for item in values)):
                return GroupValidation(
                    ok=False,
                    error=(f"evidence_tier_malformed:frame_{want}:"
                           f"{tier}"))
    context = parsed.get("rolling_context")
    if not isinstance(context, str) or not context.strip():
        return GroupValidation(
            ok=False, error="rolling_context_missing_or_empty")
    notes = parsed.get("context_notes")
    if notes is not None and not isinstance(notes, str):
        return GroupValidation(
            ok=False, error="context_notes_malformed")
    return GroupValidation(
        ok=True, entries=tuple(entries),
        rolling_context=context, context_notes=notes)


@dataclass
class FirstPassV2Config:
    """Complete V2 runner configuration (explicit, no hidden paths)."""

    frames: list[FrameRef]
    source: SourceRef
    out_dir: Path
    model: str = MODEL_SPARK
    provider: str = "meta"
    pass_id: str = PASS_ID_V2
    group_size: int = DEFAULT_GROUP_SIZE
    token_budget: int = DEFAULT_GROUP_TOKEN_BUDGET
    global_context: str | None = None
    global_context_name: str | None = None
    max_groups: int | None = None
    rates_usd_per_m: tuple[float, float] | None = None
    cost_ceiling_usd: float | None = None
    strict_cost: bool = False
    billable: bool = False
    uncertain_policy: str = UNCERTAIN_RERUN
    max_attempts: int = 1
    # Total provider submits per group when each response is
    # received but unusable. Transport without a response is
    # never replayed, regardless of this value.
    max_validation_attempts: int = 1
    # Stop the run after this many consecutive exhausted
    # groups (None disables the stop; a validated group,
    # including a valid abstention, resets the counter).
    max_consecutive_exhausted: int | None = None
    timeout_seconds: float = 120.0
    backoff_base_seconds: float = 0.5
    backoff_cap_seconds: float = 30.0
    # Memory-only credential: excluded from repr so a leaked
    # debug print can never carry key material. Never recorded
    # in reports, checkpoints, or observations.
    api_key: str = field(default="", repr=False)
    client_factory: Callable[[], GroupClient] | None = None
    sleeper: Callable[[float], None] = time.sleep


def _validate_v2_config(config: FirstPassV2Config) -> None:
    """Reject malformed run guards before any request."""
    if not isinstance(config.group_size, int) or isinstance(
            config.group_size, bool) or config.group_size < 1:
        raise FirstPassError(
            "INVALID_CONFIG",
            "group_size must be a positive int, "
            f"got {config.group_size!r}.")
    if config.group_size > MAX_IMAGES_PER_REQUEST:
        raise FirstPassError(
            "TOO_MANY_IMAGES",
            f"group_size {config.group_size} exceeds the client "
            f"maximum of {MAX_IMAGES_PER_REQUEST} images per "
            f"request; no request was sent.")
    if (isinstance(config.token_budget, bool)
            or not isinstance(config.token_budget, int)
            or config.token_budget <= 0):
        raise FirstPassError(
            "INVALID_CONFIG",
            "token_budget must be a positive int, "
            f"got {config.token_budget!r}.")
    limit = config.max_groups
    if limit is not None and (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 0):
        raise FirstPassError(
            "INVALID_CONFIG",
            "max_groups must be a non-negative int or None, "
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
    if not isinstance(config.billable, bool):
        raise FirstPassError(
            "INVALID_CONFIG",
            "billable must be a bool, "
            f"got {config.billable!r}.")
    attempts = config.max_attempts
    if (isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or attempts < 1):
        raise FirstPassError(
            "INVALID_CONFIG",
            "max_attempts must be a positive int, "
            f"got {attempts!r}.")
    validation_attempts = config.max_validation_attempts
    if (isinstance(validation_attempts, bool)
            or not isinstance(validation_attempts, int)
            or validation_attempts < 1):
        raise FirstPassError(
            "INVALID_CONFIG",
            "max_validation_attempts must be a positive int, "
            f"got {validation_attempts!r}.")
    consecutive = config.max_consecutive_exhausted
    if consecutive is not None and (
            isinstance(consecutive, bool)
            or not isinstance(consecutive, int)
            or consecutive < 1):
        raise FirstPassError(
            "INVALID_CONFIG",
            "max_consecutive_exhausted must be a positive "
            f"int or None, got {consecutive!r}.")
    if config.billable:
        if attempts != 1:
            raise FirstPassError(
                "INVALID_CONFIG",
                "billable execution requires max_attempts=1 "
                "(single submit, no transport retry), "
                f"got {attempts!r}.")
        # Validation replays stay permitted: they only rerun
        # a group whose response was fully received (billing
        # is certain) but unusable, and every replay is
        # ledger-counted against the request and cost guards.
        if config.uncertain_policy != UNCERTAIN_REFUSE:
            raise FirstPassError(
                "INVALID_CONFIG",
                "billable execution requires "
                "uncertain_policy='refuse' (no uncertain "
                f"replay), got {config.uncertain_policy!r}.")
    if (config.global_context is not None
            and not isinstance(config.global_context, str)):
        raise FirstPassError(
            "INVALID_CONFIG",
            "global_context must be a str or None, "
            f"got {type(config.global_context).__name__}.")


class DeterministicGroupFakeClient:
    """Offline stand-in for group requests (no network, no cost).

    Returns one observation per attached image, derived from
    frame indices, plus a chained rolling context. Every reply
    carries the fake model identity.
    """

    MODEL_ID = "fake-group-deterministic-v1"

    def __init__(
        self,
        *,
        usage: InferenceUsage | None = None,
        script: dict[int, str | Exception] | None = None,
    ) -> None:
        self._usage = usage or InferenceUsage(
            input_tokens=500, output_tokens=200, total_tokens=700)
        self._script = dict(script or {})
        self.calls: list[dict[str, Any]] = []

    @property
    def model_id(self) -> str:
        """Return the fake model identity recorded in calls."""
        return self.MODEL_ID

    def submit(
        self,
        prompt_text: str,
        image_paths: Sequence[Path],
        api_key: str,
        max_output_tokens: int,
        response_schema: dict[str, Any] | None = None,
        response_schema_name: str | None = None,
    ) -> InferenceResponse:
        """Reply deterministically for the referenced group."""
        _ = (api_key, max_output_tokens, response_schema,
             response_schema_name)
        indices: list[int] = []
        for path in image_paths:
            stem = Path(str(path)).stem
            try:
                indices.append(int(stem.split("_")[1]))
            except (IndexError, ValueError) as exc:
                raise ValueError(
                    "Fake group client needs frame_<index> "
                    f"names, got: {stem}.") from exc
        self.calls.append({
            "prompt_text": prompt_text,
            "frame_indices": list(indices),
        })
        key = indices[0] if indices else -1
        if key in self._script:
            action = self._script[key]
            if isinstance(action, Exception):
                raise action
            raw_text = action
        else:
            entries = [{
                "frame_index": index,
                "observation": {
                    "visible_objects": ["scale"],
                    "scale_display_verbatim": str(
                        100 + (index % 400)),
                    "scale_readable": True,
                    "scale_state": "stable",
                    "tare_action_visible": False,
                    "addition_or_removal_visible": "none",
                    "container_description": "bowl",
                    "container_changed_hint": False,
                    "confidence": "high",
                    "ambiguity_notes": None,
                },
                "verified": [f"frame {index} shows a scale"],
                "uncertain": [],
                "hypotheses": [],
            } for index in indices]
            raw_text = json.dumps({
                "frames": entries,
                "rolling_context": (
                    f"Fake context after frames "
                    f"{indices[0]}-{indices[-1]}: scale stable, "
                    f"bowl present. Verified: scale visible. "
                    f"Uncertain: nothing. Hypotheses: none."),
                "context_notes": None,
            })
        return InferenceResponse(
            model=self.MODEL_ID, status="completed",
            output_text=raw_text, usage=self._usage,
            latency_seconds=0.0,
            request_id=f"fake-group-{key:08d}",
            incomplete_reason=None,
        )


@dataclass
class _V2Ledger:
    """Group-level request/cost accounting with enforcement.

    Mirrors the V1 S1 semantics at group granularity: every
    submitted request counts exactly once, unknown spend
    never reads as zero, resume seeds prior counters from
    checkpoints, and the request cap is exact while the cost
    ceiling relies on a documented pre-dispatch reserve
    heuristic (input tokens are unknowable before dispatch).
    """

    max_groups: int | None
    ceiling: float | None
    strict: bool
    rates: tuple[float, float] | None
    prior_requests: int = 0
    prior_spent: float = 0.0
    prior_unknown: int = 0
    prior_max_single: float | None = None
    requests: int = 0
    spent: float = 0.0
    unknown: int = 0
    max_single: float | None = None
    stopped_reason: str | None = None
    stopped_detail: dict[str, Any] | None = None

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
        """Veto the next submit when a limit already trips."""
        if (self.max_groups is not None
                and self.total_requests >= self.max_groups):
            return STOP_REQUEST_LIMIT, {
                "requests_submitted": self.total_requests,
                "max_requests": self.max_groups,
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
                "reserve_basis":
                    "max_observed_single_request_estimate",
                "cost_ceiling_usd": self.ceiling,
            }
        return None

    def note_usage(self, usage: InferenceUsage) -> bool:
        """Account one submit; True when a halt tripped."""
        estimate = estimate_cost(
            usage.input_tokens, usage.output_tokens, self.rates)
        if estimate is None:
            self.unknown += 1
            if self.strict and self.ceiling is not None:
                self.halt(STOP_STRICT_UNKNOWN_COST, {
                    "phase": "post_response",
                    "cause": "usage_unreported",
                    "cost_unknown_requests": self.total_unknown,
                    "cost_ceiling_usd": self.ceiling,
                })
                return True
            return False
        self.spent += estimate
        if (self.max_single is None
                or estimate > self.max_single):
            self.max_single = estimate
        if (self.ceiling is not None
                and self.total_spent > self.ceiling):
            self.halt(STOP_COST_CEILING, {
                "phase": "post_response",
                "spent_known_usd": self.total_spent,
                "cost_ceiling_usd": self.ceiling,
            })
            return True
        return False

    def note_transport_failure(self) -> bool:
        """Account a submit without usage; True when halting."""
        self.unknown += 1
        if self.strict and self.ceiling is not None:
            self.halt(STOP_STRICT_UNKNOWN_COST, {
                "phase": "transport",
                "cause": "transport_failure_without_usage",
                "cost_unknown_requests": self.total_unknown,
                "cost_ceiling_usd": self.ceiling,
            })
            return True
        return False


def _write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically via temp file plus replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    tmp.replace(path)


def _detect_encoding(blob: bytes) -> str | None:
    """Sniff JPEG/PNG magic bytes; None when unrecognized."""
    if len(blob) >= 4 and blob[:2] == JPEG_START:
        if blob[-2:] == JPEG_END:
            return "jpeg"
        return None
    if len(blob) >= 4 and blob[:4] == PNG_START:
        return "png"
    return None


def _usage_block(usage: InferenceUsage) -> dict[str, int | None]:
    """Persist reported token usage; never invent counts."""
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "reasoning_output_tokens": usage.reasoning_output_tokens,
    }


def _gap_note(group_id: str, indices: Sequence[int],
              reason: str) -> str:
    """Explicit gap marker carried into the next context."""
    return (
        f"[gap: {group_id} frames {indices[0]}-{indices[-1]} "
        f"produced no observations ({reason}); that footage is "
        f"unobserved — do not infer continuity across it.]")


@dataclass
class _PreparedGroup:
    sequence: int
    group_id: str
    frames: list[FrameRef]
    image_paths: list[Path]
    image_sha256: list[str]
    image_encoding: list[str]


class FirstPassV2Runner:
    """Sequential multi-image first pass with rolling context.

    Groups dispatch strictly in order with one request each.
    Each accepted reply yields one V1-compatible observation
    per frame plus the rolling context for the next group.
    Checkpoints are append-only JSONL; resume skips completed
    groups and restores the latest rolling context without
    re-submitting.
    """

    def __init__(self, config: FirstPassV2Config) -> None:
        self._config = config

    def _paths(self) -> dict[str, Path]:
        root = self._config.out_dir
        return {
            "root": root,
            "raw": root / "raw_responses",
            "obs": root / "observations",
            "contexts": root / "contexts",
            "checkpoints": root / "checkpoints" / "checkpoints.jsonl",
            "state": root / "checkpoints" / "run_state.json",
            "reports": root / "reports",
        }

    def _prepare_groups(self) -> list[_PreparedGroup]:
        cfg = self._config
        if not cfg.frames:
            raise FirstPassError(
                "INVALID_CONFIG", "Frame list is empty.")
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
        for frame in ordered:
            if frame.frame_index in seen:
                raise FirstPassError(
                    "DUPLICATE_FRAME",
                    f"Duplicate frame_index: {frame.frame_index}.",
                    str(frame.frame_index),
                )
            seen.add(frame.frame_index)
        prepared: list[_PreparedGroup] = []
        for sequence, span in enumerate(
                group_frames(ordered, cfg.group_size)):
            paths: list[Path] = []
            hashes: list[str] = []
            encodings: list[str] = []
            for frame in span:
                try:
                    blob = frame.image_path.read_bytes()
                except OSError as exc:
                    raise FirstPassError(
                        "IMAGE_MISSING",
                        "Cannot read image for frame "
                        f"{frame.frame_index}.",
                        str(frame.frame_index),
                    ) from exc
                encoding = _detect_encoding(blob)
                if encoding is None:
                    raise FirstPassError(
                        "IMAGE_MISSING",
                        f"Unreadable image for frame "
                        f"{frame.frame_index}: expected JPEG or PNG "
                        f"magic bytes.",
                        str(frame.frame_index),
                    )
                paths.append(frame.image_path)
                hashes.append(
                    hashlib.sha256(blob).hexdigest())
                encodings.append(encoding)
            prepared.append(_PreparedGroup(
                sequence=sequence,
                group_id=group_id_for(sequence),
                frames=span,
                image_paths=paths,
                image_sha256=hashes,
                image_encoding=encodings,
            ))
        return prepared

    def _load_checkpoint_index(self, path: Path) -> dict[str, Any]:
        """Index checkpoints by group, recovering a torn tail."""
        index: dict[str, Any] = {}
        if not path.exists():
            return index
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
            index[record["group_id"]] = record
        return index

    def _seed_ledger(self, index: dict[str, Any]) -> _V2Ledger:
        cfg = self._config
        ledger = _V2Ledger(
            max_groups=cfg.max_groups,
            ceiling=(None if cfg.cost_ceiling_usd is None
                     else float(cfg.cost_ceiling_usd)),
            strict=cfg.strict_cost,
            rates=cfg.rates_usd_per_m,
        )
        for record in index.values():
            if record.get("status") not in (
                    "completed", "budget_stopped",
                    "needs_human_review"):
                continue
            block = record.get("budget", {})
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
        if (ledger.prior_requests < 0 or ledger.prior_spent < 0
                or ledger.prior_unknown < 0):
            raise FirstPassError(
                "CHECKPOINT_CORRUPT",
                "Checkpoint counters are negative; refusing resume.",
            )
        return ledger

    def _scan_uncertain(
        self, checkpoints_dir: Path, index: dict[str, Any],
        expected: dict[str, list[int]],
    ) -> dict[str, list[int]]:
        """Map started-but-uncheckpointed groups to their spans."""
        uncertain: dict[str, list[int]] = {}
        pattern = f"{MARKER_PREFIX}*{MARKER_SUFFIX}"
        for marker_path in sorted(checkpoints_dir.glob(pattern)):
            try:
                marker = json.loads(
                    marker_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise FirstPassError(
                    "CHECKPOINT_CORRUPT",
                    "Group start marker is unreadable: "
                    f"{marker_path.name}.",
                    marker_path.name,
                ) from exc
            group_id = marker.get("group_id") if isinstance(
                marker, dict) else None
            span = marker.get("frame_indices") if isinstance(
                marker, dict) else None
            if (not isinstance(group_id, str) or not group_id
                    or not isinstance(span, list) or not span
                    or any(not isinstance(item, int)
                           or isinstance(item, bool)
                           for item in span)):
                raise FirstPassError(
                    "CHECKPOINT_CORRUPT",
                    "Group start marker is malformed: "
                    f"{marker_path.name}.",
                    marker_path.name,
                )
            if group_id in index:
                continue
            current = expected.get(group_id)
            if current is None or set(span) - set(current):
                raise FirstPassError(
                    "CHECKPOINT_MISMATCH",
                    f"Uncertain group {group_id} does not match "
                    f"the current frame set; refusing resume.",
                    group_id,
                )
            uncertain[group_id] = list(span)
        return uncertain

    def _dispatch(
        self,
        client: GroupClient,
        prompt_text: str,
        group: _PreparedGroup,
        ledger: _V2Ledger,
    ) -> tuple[InferenceResponse | None, str | None,
               list[dict[str, Any]]]:
        """Submit one group request with bounded transport retry.

        Returns (response, error_code, attempts). Error codes
        from the client surface verbatim; image-limit codes
        raise ``FirstPassError`` so the run fails clearly
        instead of repeating doomed requests.
        """
        cfg = self._config
        attempts: list[dict[str, Any]] = []
        for step in range(cfg.max_attempts):
            stop = ledger.pre_dispatch()
            if stop is not None:
                reason, detail = stop
                ledger.halt(reason, detail)
                return None, "BUDGET_HALT", attempts
            ledger.requests += 1
            number = len(attempts) + 1
            try:
                result = client.submit(
                    prompt_text, group.image_paths,
                    cfg.api_key, cfg.token_budget,
                    response_schema=GROUP_JSON_SCHEMA,
                    response_schema_name=GROUP_SCHEMA_NAME,
                )
            except InferenceError as exc:
                attempts.append({
                    "attempt": number, "code": exc.code,
                    "message": exc.message,
                })
                if exc.code in IMAGE_LIMIT_CODES:
                    raise FirstPassError(
                        "PROVIDER_IMAGE_LIMIT",
                        f"Group {group.group_id}: provider or "
                        f"client rejected the {len(group.frames)}-"
                        f"image request ({exc.code}: {exc.message}). "
                        f"Reduce group_size and re-run into a fresh "
                        f"output directory.",
                        group.group_id,
                    ) from exc
                if exc.code in ("CREDENTIAL_MISSING", "AUTH_FAILED"):
                    raise
                ledger.note_transport_failure()
                if (exc.code in RETRYABLE_CODES
                        and step + 1 < cfg.max_attempts):
                    wait = exc.retry_after_seconds
                    if wait is None:
                        wait = min(
                            cfg.backoff_base_seconds * (2.0 ** step),
                            cfg.backoff_cap_seconds)
                    cfg.sleeper(wait)
                    attempts[-1]["slept_seconds"] = wait
                    continue
                return None, exc.code, attempts
            attempts.append({"attempt": number, "code": "ok"})
            return result, None, attempts
        return None, "RETRIES_EXHAUSTED", attempts

    def _run_group(
        self,
        client: GroupClient,
        group: _PreparedGroup,
        total_groups: int,
        incoming_context: str | None,
        context_kind: str,
        ledger: _V2Ledger,
        paths: dict[str, Path],
    ) -> tuple[dict[str, Any], str | None, str]:
        """Run one group; return (summary, next_context, next_kind)."""
        cfg = self._config
        indices = [frame.frame_index for frame in group.frames]
        prompt_text = build_group_prompt(
            frames=group.frames,
            group_sequence=group.sequence,
            total_groups=total_groups,
            incoming_context=incoming_context,
            context_kind=context_kind,
        )
        prompt_digest = prompt_hash(prompt_text)
        _write_json(
            paths["checkpoints"].parent
            / f"{MARKER_PREFIX}{group.group_id}{MARKER_SUFFIX}",
            {"group_id": group.group_id,
             "frame_indices": indices,
             "started_at_utc": datetime.datetime.now(
                 datetime.timezone.utc).isoformat()},
        )
        spent_before = ledger.spent
        unknown_before = ledger.unknown
        max_before = ledger.max_single
        halted = False
        status = "needs_human_review"
        terminal: str | None = None
        failure_reason: str | None = None
        validation: GroupValidation | None = None
        raw_text = ""
        result: InferenceResponse | None = None
        error: str | None = None
        attempts: list[dict[str, Any]] = []
        for validation_try in range(cfg.max_validation_attempts):
            if validation_try:
                wait = min(
                    cfg.backoff_base_seconds * (
                        2.0 ** (validation_try - 1)),
                    cfg.backoff_cap_seconds)
                cfg.sleeper(wait)
                attempts[-1]["slept_seconds"] = wait
            validation = None
            raw_text = ""
            terminal = None
            attempt_result, attempt_error, dispatch_notes = (
                self._dispatch(client, prompt_text, group, ledger))
            for note in dispatch_notes:
                note["attempt"] = len(attempts) + 1
                attempts.append(note)
            result, error = attempt_result, attempt_error
            if error == "BUDGET_HALT":
                halted = True
                status = "budget_stopped"
                failure_reason = (
                    ledger.stopped_reason or "budget_halt")
                break
            if error is not None or result is None:
                status = "needs_human_review"
                terminal = TERMINAL_FAILED
                failure_reason = f"transport:{error}"
                if ledger.stopped_reason is not None:
                    # A recorded halt is effective: strict
                    # ledgers halt on transport without usage
                    # instead of burning further groups.
                    halted = True
                break
            attempts[-1]["latency_seconds"] = (
                result.latency_seconds)
            attempts[-1]["request_id"] = result.request_id
            attempts[-1]["input_tokens"] = (
                result.usage.input_tokens)
            attempts[-1]["output_tokens"] = (
                result.usage.output_tokens)
            attempts[-1]["reasoning_output_tokens"] = (
                result.usage.reasoning_output_tokens)
            raw_text = result.output_text
            if (result.status == "completed"
                    and result.output_text.strip()):
                validation = validate_group_response(
                    raw_text, indices)
                if validation.ok:
                    status = "completed"
                    terminal = TERMINAL_VALID
                    attempts[-1]["validation"] = "ok"
                else:
                    status = "needs_human_review"
                    terminal = TERMINAL_INVALID_JSON
                    failure_reason = (
                        f"validation:{validation.error}")
                    attempts[-1]["validation"] = failure_reason
            else:
                status = "needs_human_review"
                if is_token_limit_reason(
                        result.incomplete_reason):
                    terminal = TERMINAL_EXHAUSTED
                    failure_reason = (
                        "token_budget_exhausted:"
                        "no_escalation_in_v2")
                elif not raw_text.strip():
                    terminal = TERMINAL_EMPTY
                    failure_reason = "empty_output"
                else:
                    terminal = TERMINAL_UNSUPPORTED
                    failure_reason = (
                        f"status:{result.status}:"
                        f"{result.incomplete_reason}")
                if result.status != "completed" and raw_text.strip():
                    failure_reason = (
                        f"status:{result.status}:"
                        f"{result.incomplete_reason}")
                attempts[-1]["validation"] = failure_reason
            halted = ledger.note_usage(result.usage) or halted
            if halted:
                if status == "completed":
                    status = "budget_stopped"
                    failure_reason = (
                        ledger.stopped_reason or "budget_halt")
                break
            if validation is not None and validation.ok:
                break
        observations: list[dict[str, Any]] = []
        next_context: str | None
        next_kind = "rolling"
        if validation is not None and validation.ok:
            next_context = bound_context(
                validation.rolling_context)
            for position, (frame, entry) in enumerate(
                    zip(group.frames, validation.entries)):
                call = {
                    "provider": cfg.provider,
                    "model": result.model if result else cfg.model,
                    "prompt_sha256": prompt_digest,
                    "base_prompt_sha256": prompt_hash(
                        V2_BASE_PROMPT),
                    "group_id": group.group_id,
                    "group_sequence": group.sequence,
                    "image_position": position + 1,
                    "group_size": len(group.frames),
                    "incoming_context_kind": context_kind,
                    "incoming_context_sha256": (
                        hashlib.sha256(
                            incoming_context.encode("utf-8")
                        ).hexdigest()
                        if incoming_context else None),
                    "global_context_sha256": (
                        hashlib.sha256(
                            cfg.global_context.encode("utf-8")
                        ).hexdigest()
                        if cfg.global_context else None),
                    "max_output_tokens": cfg.token_budget,
                    "store": True,
                    "timeout_seconds": cfg.timeout_seconds,
                    "input_tokens": (
                        result.usage.input_tokens
                        if result else None),
                    "output_tokens": (
                        result.usage.output_tokens
                        if result else None),
                    "latency_seconds": (
                        result.latency_seconds if result else None),
                    "request_id": (
                        result.request_id if result else None),
                    "attempts": len(attempts),
                    "response_status": (
                        result.status if result else None),
                    "incomplete_reason": (
                        result.incomplete_reason if result else None),
                    "terminal_state": terminal,
                }
                record = build_observation(
                    frame, cfg.source, group.group_id,
                    cfg.pass_id, call,
                    json.dumps(entry["observation"],
                               sort_keys=True),
                    response_status=(
                        result.status if result else None),
                    incomplete_reason=(
                        result.incomplete_reason if result else None),
                    terminal_state=terminal,
                    image_sha256=group.image_sha256[position],
                    image_encoding=group.image_encoding[position],
                )
                record["evidence_tiers"] = {
                    "verified": entry["verified"],
                    "uncertain": entry["uncertain"],
                    "hypotheses": entry["hypotheses"],
                }
                observations.append(record)
        else:
            carried = incoming_context or ""
            note = _gap_note(
                group.group_id, indices,
                failure_reason or "unknown")
            next_context = bound_context(
                f"{carried}\n{note}".strip()) if carried else note
            next_kind = "rolling_with_gap"
        raw_record: dict[str, Any] = {
            "group_id": group.group_id,
            "group_sequence": group.sequence,
            "frame_indices": indices,
            "image_order": [
                {"position": position + 1,
                 "frame_index": frame.frame_index,
                 "image_path": frame.image_path.name,
                 "sha256": group.image_sha256[position],
                 "encoding": group.image_encoding[position]}
                for position, frame in enumerate(group.frames)
            ],
            "prompt_sha256": prompt_digest,
            "incoming_context_kind": context_kind,
            "incoming_context_chars": (
                len(incoming_context) if incoming_context else 0),
            "incoming_context_token_estimate": (
                estimate_context_tokens(incoming_context)
                if incoming_context else 0),
            "outgoing_context_chars": (
                len(next_context) if next_context else 0),
            "outgoing_context_token_estimate": (
                estimate_context_tokens(next_context)
                if next_context else 0),
            "status": status,
            "terminal_state": terminal,
            "failure_reason": failure_reason,
            "attempts": attempts,
            "output_text": raw_text,
            "usage": (_usage_block(result.usage)
                      if result else _usage_block(InferenceUsage())),
            "latency_seconds": (
                result.latency_seconds if result else None),
            "request_id": result.request_id if result else None,
            "model": result.model if result else cfg.model,
            "response_status": result.status if result else None,
            "incomplete_reason": (
                result.incomplete_reason if result else None),
        }
        raw_path = paths["raw"] / f"{group.group_id}.json"
        obs_path = paths["obs"] / f"{group.group_id}.json"
        _write_json(raw_path, raw_record)
        _write_json(obs_path, observations)
        if next_context:
            context_path = (
                paths["contexts"] / f"{group.group_id}.md")
            context_path.parent.mkdir(
                parents=True, exist_ok=True)
            context_path.write_text(
                next_context + "\n", encoding="utf-8", newline="\n")
        estimate = estimate_cost(
            raw_record["usage"]["input_tokens"],
            raw_record["usage"]["output_tokens"],
            cfg.rates_usd_per_m)
        record = {
            "group_id": group.group_id,
            "group_sequence": group.sequence,
            "frame_indices": indices,
            "status": status,
            "terminal_state": terminal,
            "failure_reason": failure_reason,
            "raw_path": raw_path.name,
            "obs_path": obs_path.name,
            "attempts": attempts,
            "input_tokens":
                raw_record["usage"]["input_tokens"],
            "output_tokens":
                raw_record["usage"]["output_tokens"],
            "rolling_context": next_context,
            "rolling_context_kind": next_kind,
            "frames_observed": len(observations),
            "budget": {
                "requests_submitted": len(attempts),
                "cost_usd_estimate": (
                    estimate
                    if ledger.unknown == unknown_before
                    else None),
                "cost_known_usd_estimate": round(
                    ledger.spent - spent_before, 4),
                "cost_unknown_requests": (
                    ledger.unknown - unknown_before),
                "max_single_request_cost_usd_estimate": (
                    None if ledger.max_single is None
                    or (max_before is not None
                        and ledger.max_single <= max_before)
                    else ledger.max_single),
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
        summary = {
            "group_id": group.group_id,
            "status": status,
            "terminal_state": terminal,
            "submits": len(attempts),
            "frames_observed": len(observations),
            "frames_expected": len(indices),
            "input_tokens":
                raw_record["usage"]["input_tokens"] or 0,
            "output_tokens":
                raw_record["usage"]["output_tokens"] or 0,
            "reasoning_output_tokens": sum(
                note.get("reasoning_output_tokens") or 0
                for note in attempts),
            "latency_seconds": raw_record["latency_seconds"],
            "request_id": raw_record["request_id"],
            "halted": halted,
            "exhausted": (status != "completed" and not halted),
        }
        return summary, next_context, next_kind

    def _check_prior_state(
            self, state_path: Path,
            prepared: list[_PreparedGroup]) -> None:
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
        current = [frame.frame_index for group in prepared
                   for frame in group.frames]
        if prior_frames != current:
            raise FirstPassError(
                "CHECKPOINT_MISMATCH",
                "Prior run state covers a different frame set; "
                "refusing resume into this directory.",
            )

    def run(self) -> dict[str, Any]:
        """Execute all groups sequentially; return the run report."""
        cfg = self._config
        _validate_v2_config(cfg)
        if cfg.client_factory is None:
            raise FirstPassError(
                "INVALID_CONFIG",
                "No inference client wired: client_factory is None.")
        prepared = self._prepare_groups()
        paths = self._paths()
        for key in ("raw", "obs", "contexts", "reports"):
            paths[key].mkdir(parents=True, exist_ok=True)
        paths["checkpoints"].parent.mkdir(
            parents=True, exist_ok=True)
        completed = self._load_checkpoint_index(
            paths["checkpoints"])
        ledger = self._seed_ledger(completed)
        self._check_prior_state(paths["state"], prepared)
        expected = {
            group.group_id: [frame.frame_index
                             for frame in group.frames]
            for group in prepared
        }
        uncertain = self._scan_uncertain(
            paths["checkpoints"].parent, completed, expected)
        uncertain_frames = sorted({
            frame for span in uncertain.values()
            for frame in span})
        if uncertain and (
                cfg.uncertain_policy == UNCERTAIN_REFUSE):
            groups = sorted(uncertain)
            raise FirstPassError(
                "UNCERTAIN_REQUESTS",
                f"{len(uncertain_frames)} frame(s) in "
                f"{len(groups)} group(s) lack verified "
                f"outcomes ({', '.join(groups)}); refusing "
                f"automatic replay (uncertain_policy=refuse).",
                groups[0],
            )
        assert cfg.client_factory is not None
        client = cfg.client_factory()
        total = len(prepared)
        summaries: list[dict[str, Any]] = []
        stats: dict[str, Any] = {
            "groups_total": total,
            "groups_completed": sum(
                1 for record in completed.values()
                if record.get("status") in (
                    "completed", "budget_stopped",
                    "needs_human_review")),
            "requests_sent": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
            "frames_total": sum(
                len(group.frames) for group in prepared),
            "frames_observed": 0,
            "groups_valid": 0,
            "groups_review": 0,
            "groups_exhausted": 0,
            "groups_resumed": 0,
            "frames_resumed": 0,
            "terminal_states": {},
        }
        incoming: str | None = None
        kind = "none"
        if cfg.global_context:
            incoming = bound_context(cfg.global_context)
            kind = "global"
        start_sequence = 0
        for group in prepared:
            record = completed.get(group.group_id)
            if record is None:
                break
            current = [frame.frame_index
                       for frame in group.frames]
            if record.get("frame_indices") != current:
                raise FirstPassError(
                    "CHECKPOINT_MISMATCH",
                    f"Group {group.group_id} checkpoint covers "
                    f"{record.get('frame_indices')}, current span "
                    f"is {current}; refusing to resume a different "
                    f"frame set.",
                    group.group_id,
                )
            if record.get("status") not in (
                    "completed", "budget_stopped",
                    "needs_human_review"):
                raise FirstPassError(
                    "CHECKPOINT_MISMATCH",
                    f"Group {group.group_id} has unrecognized "
                    f"checkpoint status {record.get('status')!r}; "
                    f"refusing resume.",
                    group.group_id,
                )
            if (record.get("status") == "budget_stopped"
                    and not record.get("frames_observed")):
                # Halted before any submit: re-attempt below
                # instead of adopting an empty gap as done.
                break
            stored = record.get("rolling_context")
            if not isinstance(stored, str) or not stored:
                raise FirstPassError(
                    "CHECKPOINT_CORRUPT",
                    f"Group {group.group_id} checkpoint lacks "
                    f"its rolling context; refusing resume.",
                    group.group_id,
                )
            incoming, kind = stored, str(record.get(
                "rolling_context_kind", "rolling"))
            start_sequence = group.sequence + 1
            observed = record.get("frames_observed") or 0
            summaries.append({
                "group_id": group.group_id,
                "status": record.get("status"),
                "terminal_state": record.get("terminal_state"),
                "submits": 0,
                "frames_observed": observed,
                "frames_expected": len(current),
                "input_tokens":
                    record.get("input_tokens") or 0,
                "output_tokens":
                    record.get("output_tokens") or 0,
                "latency_seconds": None,
                "request_id": None,
                "reasoning_output_tokens": 0,
                "halted": record.get("status") == "budget_stopped",
                "exhausted": False,
                "resumed": True,
            })
            stats["groups_resumed"] += 1
            stats["frames_resumed"] += observed
            if record.get("status") == "budget_stopped":
                break
        consecutive_exhausted = 0
        if cfg.max_consecutive_exhausted is not None:
            for prior in prepared[:start_sequence]:
                prior_record = completed.get(prior.group_id)
                if prior_record is None:
                    break
                if prior_record.get("status") == "completed":
                    consecutive_exhausted = 0
                elif (prior_record.get("status")
                        == "needs_human_review"):
                    consecutive_exhausted += 1
                else:
                    break
        pending = prepared[start_sequence:]
        if (cfg.max_consecutive_exhausted is not None
                and consecutive_exhausted
                >= cfg.max_consecutive_exhausted):
            # A resumed run whose adopted tail already meets
            # the stop condition submits nothing further.
            ledger.halt(STOP_CONSECUTIVE_EXHAUSTED, {
                "consecutive_exhausted": consecutive_exhausted,
                "resumed_stop": True,
            })
            pending = []
        for group in pending:
            summary, incoming, kind = self._run_group(
                client, group, total, incoming, kind,
                ledger, paths)
            summaries.append(summary)
            stats["requests_sent"] += summary["submits"]
            stats["input_tokens"] += summary["input_tokens"]
            stats["output_tokens"] += summary["output_tokens"]
            stats["reasoning_output_tokens"] += summary[
                "reasoning_output_tokens"]
            stats["frames_observed"] += summary["frames_observed"]
            if summary["status"] == "completed":
                stats["groups_valid"] += 1
                consecutive_exhausted = 0
            else:
                stats["groups_review"] += 1
                if summary.get("exhausted"):
                    stats["groups_exhausted"] += 1
                    consecutive_exhausted += 1
            terminal = summary.get("terminal_state")
            if terminal:
                stats["terminal_states"][terminal] = (
                    stats["terminal_states"].get(terminal, 0) + 1)
            if (cfg.max_consecutive_exhausted is not None
                    and consecutive_exhausted
                    >= cfg.max_consecutive_exhausted):
                ledger.halt(STOP_CONSECUTIVE_EXHAUSTED, {
                    "consecutive_exhausted":
                        consecutive_exhausted,
                    "group_id": summary["group_id"],
                    "groups_attempted": len(summaries),
                })
                break
            if summary["halted"]:
                break
        attempted_ids = {
            summary["group_id"] for summary in summaries
            if summary.get("submits") or summary.get("resumed")}
        remaining = [frame.frame_index for group in prepared
                     if group.group_id not in attempted_ids
                     for frame in group.frames]
        return self._write_report(
            paths, stats, prepared, summaries, ledger,
            remaining, uncertain_frames)

    def _write_report(
        self,
        paths: dict[str, Path],
        stats: dict[str, Any],
        prepared: list[_PreparedGroup],
        summaries: list[dict[str, Any]],
        ledger: _V2Ledger,
        remaining: list[int],
        uncertain: list[int],
    ) -> dict[str, Any]:
        cfg = self._config
        rates = cfg.rates_usd_per_m
        frames = [frame.frame_index for group in prepared
                  for frame in group.frames]
        state: dict[str, Any] = {
            "experiment": "firstpass_v2",
            "pass_id": cfg.pass_id,
            "source_name": cfg.source.name,
            "source_sha256": cfg.source.sha256,
            "frame_indices": frames,
            "model": cfg.model,
            "provider": cfg.provider,
            "billable": cfg.billable,
            "group_size": cfg.group_size,
            "token_budget_per_group": cfg.token_budget,
            "base_prompt_sha256": prompt_hash(V2_BASE_PROMPT),
            "global_context_sha256": (
                hashlib.sha256(
                    cfg.global_context.encode("utf-8")).hexdigest()
                if cfg.global_context else None),
            "global_context_name": cfg.global_context_name,
            "context_bound_chars": CONTEXT_MAX_CHARS,
            "context_target_tokens": CONTEXT_TARGET_TOKENS,
            "groups_total": stats["groups_total"],
            "groups_completed": stats["groups_completed"],
            "groups_valid": stats["groups_valid"],
            "groups_review": stats["groups_review"],
            "groups_exhausted": stats["groups_exhausted"],
            "groups_resumed": stats["groups_resumed"],
            "frames_resumed": stats["frames_resumed"],
            "requests_sent": stats["requests_sent"],
            "requests_total": ledger.total_requests,
            "input_tokens": stats["input_tokens"],
            "output_tokens": stats["output_tokens"],
            "reasoning_output_tokens":
                stats["reasoning_output_tokens"],
            "total_tokens": (stats["input_tokens"]
                             + stats["output_tokens"]),
            "frames_total": stats["frames_total"],
            "frames_observed": stats["frames_observed"],
            # Pipeline-reader compatibility: the V1 resume
            # verifier requires an int frames_failed key.
            "frames_failed": (
                stats["frames_total"]
                - stats["frames_observed"]),
            "output_completeness": (
                (stats["frames_observed"]
                 + stats["frames_resumed"]) / stats["frames_total"]
                if stats["frames_total"] else None),
            "stopped_reason": ledger.stopped_reason,
            "stopped_detail": ledger.stopped_detail,
            "terminal_states": dict(
                sorted(stats["terminal_states"].items())),
            "groups": summaries,
            "budget": {
                "max_requests": cfg.max_groups,
                "requests_submitted": ledger.requests,
                "prior_requests_submitted":
                    ledger.prior_requests,
                "cost_ceiling_usd": ledger.ceiling,
                "currency": "USD",
                "rates_usd_per_m": (
                    list(rates) if rates is not None else None),
                "strict_cost": ledger.strict,
                "spent_cost_usd_estimate": (
                    None if ledger.unknown
                    else round(ledger.spent, 4)),
                "spent_cost_known_usd_estimate": round(
                    ledger.spent, 4),
                "cost_unknown_requests": ledger.unknown,
                "prior_spent_cost_known_usd_estimate": round(
                    ledger.prior_spent, 4),
                "prior_cost_unknown_requests":
                    ledger.prior_unknown,
                "stopped_reason": ledger.stopped_reason,
                "stopped_detail": ledger.stopped_detail,
                "remaining_frame_indices": remaining,
                "uncertain_frame_indices": list(uncertain),
                "uncertain_policy": cfg.uncertain_policy,
            },
        }
        if rates is not None and (
            stats["input_tokens"] or stats["output_tokens"]
        ):
            state["cost_usd_estimated"] = estimate_cost(
                stats["input_tokens"], stats["output_tokens"], rates)
            state["cost_status"] = "estimated_from_configured_rates"
        else:
            state["cost_usd_estimated"] = None
            state["cost_status"] = "COST_UNKNOWN"
        _write_json(paths["state"], state)
        _write_json(paths["reports"] / "run_report.json", state)
        return state
