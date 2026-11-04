"""Culinary visual evidence enrichment (measurement -> multimodal).

Extends the measurement-focused first pass with structured culinary
observations (ingredients, actions, utensils, containers) and
deterministic temporal event reconstruction. Historical
measurement-only observations are never reinterpreted: absent
culinary fields stay ``not_evaluated``, never negative findings.

Design rules mirrored from the first-pass and live providers:
- The provider client is an injected seam; tests run offline.
- Structured output is strictly validated; failures carry codes.
- Enrichment is explicitly triggered, batched, checkpointed, and
  resumable. Each frame is submitted at most once per enrichment
  version: failed frames are terminal and never auto-retried.
- A displayed scale reading never becomes a net ingredient
  quantity. Net amounts stay unknown with recorded reasons until
  baseline, units, continuity, and tare support all hold.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import os

from bakery_intelligence.api.jobs import utc_now_iso
from bakery_intelligence.api.storage import validate_run_id


LIVE_FLAG = "BAKERY_LIVE_RECIPE_GEN"
META_KEY_VAR = "MODEL_API_KEY"


def enrichment_available() -> tuple[bool, list[str]]:
    """Check enrichment prerequisites without exposing secrets."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        pass
    else:
        load_dotenv(override=False)
    blockers: list[str] = []
    if os.environ.get(LIVE_FLAG, "").lower() not in ("1", "true", "yes"):
        blockers.append(
            f"{LIVE_FLAG} is not enabled; enrichment calls are disabled.")
    if not os.environ.get(META_KEY_VAR):
        blockers.append(f"{META_KEY_VAR} is not configured.")
    return (not blockers, blockers)


def enrichment_api_key() -> str:
    """Return the Meta key for server-side enrichment calls only."""
    ok, blockers = enrichment_available()
    if not ok:
        raise CulinaryError("; ".join(blockers))
    return os.environ.get(META_KEY_VAR, "")

CULINARY_OBSERVATION_VERSION = "culinary-observation/1"
CULINARY_EVENTS_VERSION = "culinary-events/1"
CULINARY_ENRICHMENT_VERSION = "culinary-enrichment/1"
CULINARY_STORE_DIRNAME = "culinary_observations"
CULINARY_MODEL = "muse-spark-1.3"

# Per-field evidence source vocabulary. Observed means directly
# visible; inferred means a stated visual deduction; the rest are
# explicit non-claims. Missing fields in historical observations
# map to not_evaluated, never to a negative finding.
SOURCE_OBSERVED = "observed"
SOURCE_INFERRED = "inferred"
SOURCE_UNKNOWN = "unknown"
SOURCE_NOT_VISIBLE = "not_visible"
SOURCE_NOT_EVALUATED = "not_evaluated"

FIELD_SOURCES = (
    SOURCE_OBSERVED,
    SOURCE_INFERRED,
    SOURCE_UNKNOWN,
    SOURCE_NOT_VISIBLE,
    SOURCE_NOT_EVALUATED,
)

CONFIDENCES = ("high", "medium", "low")

# Closed culinary action vocabulary. Anything else the model emits
# is normalized to unknown_action, never silently kept.
# Explicit verb-form aliases map to their canonical action; only
# unambiguous, individually listed mappings apply. Everything else
# stays unknown_action with the raw value preserved.
ACTION_TYPE_ALIASES = {
    "pouring": "pour",
}
CULINARY_ACTION_TYPES = (
    "add_ingredient",
    "pour",
    "transfer",
    "stir",
    "mix",
    "fold",
    "chop",
    "crack",
    "heat",
    "tare",
    "weigh",
    "rest",
    "unknown_action",
    "no_action_visible",
)

CULINARY_SCHEMA_NAME = "CulinaryObservation"
CULINARY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ingredient_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "confidence": {"type": "string"},
                    "packaging_visible": {"type": "boolean"},
                    "label_text": {"type": ["string", "null"]},
                    "source": {"type": "string"},
                },
                "required": ["confidence", "source"],
                "additionalProperties": False,
            },
        },
        "containers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": ["string", "null"]},
                    "description": {"type": ["string", "null"]},
                    "source": {"type": "string"},
                },
                "required": ["source"],
                "additionalProperties": False,
            },
        },
        "utensils": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": ["string", "null"]},
                    "description": {"type": ["string", "null"]},
                    "source": {"type": "string"},
                },
                "required": ["source"],
                "additionalProperties": False,
            },
        },
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action_type": {"type": "string"},
                    "participants": {"type": "array",
                                    "items": {"type": "string"}},
                    "targets": {"type": "array",
                                "items": {"type": "string"}},
                    "confidence": {"type": "string"},
                    "source": {"type": "string"},
                },
                "required": ["action_type", "confidence", "source"],
                "additionalProperties": False,
            },
        },
        "scale": {
            "type": "object",
            "properties": {
                "display_verbatim": {"type": ["string", "null"]},
                "readable": {"type": ["boolean", "null"]},
                "state": {"type": ["string", "null"]},
                "tare_action_visible": {"type": ["boolean", "null"]},
                "unit": {"type": ["string", "null"]},
                "source": {"type": "string"},
            },
            "required": ["source"],
            "additionalProperties": False,
        },
        "temporal": {
            "type": "object",
            "properties": {
                "continues_previous": {"type": ["boolean", "null"]},
                "transition": {"type": ["string", "null"]},
                "source": {"type": "string"},
            },
            "required": ["source"],
            "additionalProperties": False,
        },
        "uncertainty_notes": {"type": ["string", "null"]},
    },
    "required": ["ingredient_candidates", "containers", "utensils",
                 "actions", "scale", "temporal"],
    "additionalProperties": False,
}

CULINARY_PROMPT = (
    "Describe only what is visually observable in this cooking frame. "
    "Reply with exactly one compact JSON object matching the culinary "
    "observation schema and no prose. "
    "Three separate judgments per claim: "
    "(1) source says HOW the claim is supported and must be exactly "
    "one of observed, inferred, unknown, not_visible, not_evaluated; "
    "it is never a place in the image, so never write frame, image, "
    "left, right, center, or similar words there. "
    "(2) confidence says HOW CERTAIN the claim is and must be exactly "
    "one of high, medium, low; when you cannot judge certainty, "
    "write the word unknown, never observed or a place in the image. "
    "(3) location belongs in the description text, for example "
    "'patterned bowl at frame center holding penne'. "
    "List ingredient candidates only when something edible, its "
    "packaging, or its label is visible; a partially visible bag "
    "with an unreadable label is an unnamed candidate with medium "
    "or low confidence, never a named ingredient. Otherwise leave "
    "the list empty. List containers, utensils, and at most one "
    "culinary action with participants and targets; when no action "
    "is visible, use an empty actions list. Report the scale "
    "display verbatim when readable; never invent quantities, "
    "units, or ingredient identities from context. When the scale "
    "or an object is not visible, say not_visible with confidence "
    "unknown rather than guessing."
)


class CulinaryError(ValueError):
    """Structured culinary enrichment failure."""


def _normalize_source(value: Any) -> tuple[str, Any]:
    """Conservatively normalize a model-supplied source.

    Valid epistemic vocabulary passes through. Anything else
    (including spatial descriptions such as "frame" or
    "frame_center_right") becomes an explicit unknown: spatial
    hints are never converted into observed provenance, and the
    raw value is always preserved for audit.
    """
    if value in FIELD_SOURCES:
        return str(value), value
    return SOURCE_UNKNOWN, value


def _normalize_confidence(value: Any) -> tuple[str | None, Any]:
    """Conservatively normalize a model-supplied confidence.

    Valid vocabulary passes through. Anything else (including a
    misplaced "observed" or an explicit "unknown") becomes an
    explicit unknown (None). The raw value is always preserved
    for audit; it is never transferred into the source field and
    never upgraded to a real confidence level.
    """
    if value in CONFIDENCES:
        return str(value), value
    return None, value


def validate_culinary_observation(
        payload: Any, frame_index: int) -> dict[str, Any]:
    """Validate one frame's culinary observation.

    Unknown actions normalize to unknown_action; unknown
    confidences and non-vocabulary sources normalize to an
    explicit unknown with the raw model value preserved. Spatial
    descriptions are never converted into observed provenance.
    Structural errors (missing sections, wrong shapes, multiple
    actions) still raise instead of guessing.
    """
    if not isinstance(payload, dict):
        raise CulinaryError(
            "INVALID_OBSERVATION: culinary observation must be a "
            f"JSON object for frame {frame_index}.")
    for key in ("ingredient_candidates", "containers", "utensils",
                "actions", "scale", "temporal"):
        if key not in payload:
            raise CulinaryError(
                "INVALID_OBSERVATION: culinary observation for frame "
                f"{frame_index} is missing {key!r}.")
    candidates = payload.get("ingredient_candidates")
    if not isinstance(candidates, list):
        raise CulinaryError(
            "INVALID_OBSERVATION: ingredient_candidates must be a "
            f"list for frame {frame_index}.")
    normalized_candidates = []
    for position, item in enumerate(candidates):
        if not isinstance(item, dict):
            raise CulinaryError(
                "INVALID_OBSERVATION: ingredient candidate "
                f"{position} is not an object for frame {frame_index}.")
        confidence, confidence_raw = _normalize_confidence(
            item.get("confidence"))
        source, source_raw = _normalize_source(item.get("source"))
        normalized_candidates.append({
            "name": item.get("name"),
            "confidence": confidence,
            "confidence_raw": confidence_raw,
            "packaging_visible": bool(
                item.get("packaging_visible")),
            "label_text": item.get("label_text"),
            "source": source,
            "source_raw": source_raw,
        })
    actions = payload.get("actions")
    if not isinstance(actions, list) or len(actions) > 1:
        raise CulinaryError(
            "INVALID_OBSERVATION: actions must list at most one "
            f"culinary action for frame {frame_index}.")
    normalized_actions = []
    for item in actions:
        if not isinstance(item, dict):
            raise CulinaryError(
                "INVALID_OBSERVATION: action entry is not an object "
                f"for frame {frame_index}.")
        raw_action_type = item.get("action_type")
        if raw_action_type in CULINARY_ACTION_TYPES:
            action_type = str(raw_action_type)
        elif (isinstance(raw_action_type, str)
                and ACTION_TYPE_ALIASES.get(
                    raw_action_type.strip().lower()) is not None):
            action_type = ACTION_TYPE_ALIASES[
                raw_action_type.strip().lower()]
        else:
            action_type = "unknown_action"
        confidence, confidence_raw = _normalize_confidence(
            item.get("confidence"))
        source, source_raw = _normalize_source(item.get("source"))
        normalized_actions.append({
            "action_type": action_type,
            "action_type_raw": raw_action_type,
            "participants": list(item.get("participants") or []),
            "targets": list(item.get("targets") or []),
            "confidence": confidence,
            "confidence_raw": confidence_raw,
            "source": source,
            "source_raw": source_raw,
        })

    def _normalize_holders(items: Any, key: str) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            raise CulinaryError(
                f"INVALID_OBSERVATION: {key} must be a list for "
                f"frame {frame_index}.")
        normalized: list[dict[str, Any]] = []
        for position, item in enumerate(items):
            if not isinstance(item, dict):
                raise CulinaryError(
                    f"INVALID_OBSERVATION: {key} entry {position} is "
                    f"not an object for frame {frame_index}.")
            source, source_raw = _normalize_source(item.get("source"))
            normalized.append({
                "kind": item.get("kind"),
                "description": item.get("description"),
                "source": source,
                "source_raw": source_raw,
            })
        return normalized

    scale = payload.get("scale")
    if not isinstance(scale, dict):
        raise CulinaryError(
            "INVALID_OBSERVATION: scale must be an object for frame "
            f"{frame_index}.")
    scale_source, scale_source_raw = _normalize_source(
        scale.get("source"))
    temporal = payload.get("temporal")
    if not isinstance(temporal, dict):
        raise CulinaryError(
            "INVALID_OBSERVATION: temporal must be an object for "
            f"frame {frame_index}.")
    temporal_source, temporal_source_raw = _normalize_source(
        temporal.get("source"))
    return {
        "frame_index": frame_index,
        "ingredient_candidates": normalized_candidates,
        "containers": _normalize_holders(
            payload.get("containers"), "containers"),
        "utensils": _normalize_holders(
            payload.get("utensils"), "utensils"),
        "actions": normalized_actions,
        "scale": {
            "display_verbatim": scale.get("display_verbatim"),
            "readable": scale.get("readable"),
            "state": scale.get("state"),
            "tare_action_visible": scale.get("tare_action_visible"),
            "unit": scale.get("unit"),
            "source": scale_source,
            "source_raw": scale_source_raw,
        },
        "temporal": {
            "continues_previous": temporal.get("continues_previous"),
            "transition": temporal.get("transition"),
            "source": temporal_source,
            "source_raw": temporal_source_raw,
        },
        "uncertainty_notes": payload.get("uncertainty_notes"),
    }


# Review-facing states for culinary exception tasks. None of these
# block recipe generation: they record uncertainty for the chef and
# context for the proposer.
CULINARY_STATE_INGREDIENT_UNCERTAIN = "ingredient_uncertain"
CULINARY_STATE_ACTION_CONFLICTING = "action_conflicting"
CULINARY_STATE_TRANSITION_UNCERTAIN = "transition_uncertain"
CULINARY_STATE_UNIT_MISSING = "unit_missing"
CULINARY_STATE_TARE_UNRESOLVED = "tare_unresolved"

CULINARY_TASK_STATES = (
    CULINARY_STATE_INGREDIENT_UNCERTAIN,
    CULINARY_STATE_ACTION_CONFLICTING,
    CULINARY_STATE_TRANSITION_UNCERTAIN,
    CULINARY_STATE_UNIT_MISSING,
    CULINARY_STATE_TARE_UNRESOLVED,
)

# Maximum temporal gap inside one event, in seconds. Larger gaps
# start a new event instead of silently propagating a claim across
# unsupported intervals. Frame indices are NOT used: frames were
# sampled at 1 FPS from a 50 FPS video, so consecutive observations
# sit 50 indices apart.
EVENT_MAX_GAP_SECONDS = 2.0

# Explicit ingredient-name aliases for event grouping only. Names
# that canonicalize alike are treated as potentially the same
# ingredient when deciding whether consecutive observations belong
# to one event; the raw names are always preserved on the merged
# entry and unrelated identities are never merged.
INGREDIENT_ALIASES = {
    "penne rigati": "penne rigate",
}


def _canonical_ingredient(name: Any) -> str:
    """Grouping key for an ingredient name; raw names are kept."""
    text = str(name or "").strip().lower()
    text = " ".join(text.split())
    return INGREDIENT_ALIASES.get(text, text)

# Temporal values that carry no transition claim on their own.
# Only an actual described transition counts as reported.
_NON_TRANSITIONS = (None, "", "none", "unknown", "not_evaluated",
                    "not_visible", "no_transition")


def _reported_transition(item: dict[str, Any]) -> bool:
    """Whether an observation actually reports a transition."""
    transition = (item.get("temporal") or {}).get("transition")
    if transition is None:
        return False
    if not isinstance(transition, str):
        return bool(transition)
    return transition.strip().lower() not in _NON_TRANSITIONS


def _confidence_rank(value: Any) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(value, 0)


def _event_seed(run_id: str, first: int, last: int,
                operation: str) -> str:
    return hashlib.sha256(json.dumps(
        [run_id, first, last, operation],
        sort_keys=True).encode()).hexdigest()[:12]


def build_culinary_events(
        run_id: str,
        observations: list[dict[str, Any]],
        scale_by_frame: dict[int, Any] | None = None,
        max_gap_seconds: float = EVENT_MAX_GAP_SECONDS,
        ) -> list[dict[str, Any]]:
    """Deterministically group observations into culinary events.

    Consecutive observations (by timestamp) sharing one action
    with compatible ingredients form an event; ingredient
    changes, action changes, and temporal gaps beyond
    max_gap_seconds start new events. Unknown actions never
    merge into known actions. Net quantities are never derived
    here: every event records why instead.
    """
    ordered = sorted(observations,
                     key=lambda item: item["frame_index"])
    scale_by_frame = scale_by_frame or {}
    events: list[dict[str, Any]] = []
    group: list[dict[str, Any]] = []

    def action_of(item: dict[str, Any]) -> str:
        actions = item.get("actions") or []
        if actions and isinstance(actions[0], dict):
            return str(actions[0].get("action_type")
                       or "unknown_action")
        return "no_action_visible"

    def ingredients_of(item: dict[str, Any]) -> set[str]:
        names = set()
        for candidate in item.get("ingredient_candidates") or []:
            if isinstance(candidate, dict) and candidate.get("name"):
                names.add(
                    _canonical_ingredient(candidate.get("name")))
        return names

    def gap_seconds(previous: dict[str, Any],
                    current: dict[str, Any]) -> float | None:
        """Elapsed seconds between observations; None when unknown."""
        before = previous.get("timestamp_seconds")
        after = current.get("timestamp_seconds")
        if isinstance(before, bool) or isinstance(after, bool):
            return None
        if isinstance(before, (int, float)) and isinstance(
                after, (int, float)):
            return float(after) - float(before)
        return None

    def flush() -> None:
        if not group:
            return
        first = group[0]["frame_index"]
        last = group[-1]["frame_index"]
        operation = action_of(group[0])
        seed = _event_seed(run_id, first, last, operation)
        merged: dict[str, dict[str, Any]] = {}
        for item in group:
            for candidate in item.get("ingredient_candidates") or []:
                if not isinstance(candidate, dict):
                    continue
                key = _canonical_ingredient(
                    candidate.get("name") or "unnamed")
                current = merged.get(key)
                rank = _confidence_rank(candidate.get("confidence"))
                if current is None or rank > _confidence_rank(
                        current.get("confidence")):
                    merged[key] = {
                        "name": candidate.get("name"),
                        "confidence": candidate.get("confidence"),
                        "source": candidate.get("source"),
                        "variants": [],
                        "frames": [],
                    }
                raw = candidate.get("name")
                if raw not in merged[key]["variants"]:
                    merged[key]["variants"].append(raw)
                merged[key]["frames"].append(item["frame_index"])
        confidence = min(
            (_confidence_rank((item.get("actions") or [{}])[0].get(
                "confidence")) if item.get("actions") else 0
             for item in group),
            default=0)
        confidence_label = {3: "high", 2: "medium", 1: "low"}.get(
            confidence, "low")
        uncertainties: list[str] = []
        if not merged:
            uncertainties.append("INGREDIENT_UNKNOWN")
        elif any(info["confidence"] != "high"
                 for info in merged.values()):
            uncertainties.append("INGREDIENT_AMBIGUOUS")
        if any(action_of(item) in ("unknown_action",)
               for item in group):
            uncertainties.append("ACTION_UNKNOWN")
        if any(_reported_transition(item) for item in group):
            uncertainties.append("TRANSITION_REPORTED")
        scale_values = [scale_by_frame.get(item["frame_index"])
                        for item in group]
        if any(value is None for value in scale_values):
            uncertainties.append("SCALE_GAP")
        tare_visible = any(
            (item.get("scale") or {}).get("tare_action_visible") is True
            for item in group)
        if tare_visible:
            uncertainties.append("TARE_UNRESOLVED")
        unmet = ["unit_not_observed_in_culinary_schema"]
        if any(value is None for value in scale_values):
            unmet.append("scale_baseline_incomplete")
        unmet.append("net_derivation_not_supported_in_this_pass")
        events.append({
            "event_id": f"culinary_{seed}",
            "operation": operation,
            "start_frame": first,
            "end_frame": last,
            "start_seconds": group[0].get("timestamp_seconds"),
            "end_seconds": group[-1].get("timestamp_seconds"),
            "frame_refs": [item["frame_index"] for item in group],
            "tare_visible": tare_visible,
            "ingredient_candidates": sorted(
                merged.values(), key=lambda info: str(info["name"])),
            "scale_values": scale_values,
            "confidence": confidence_label,
            "uncertainty": uncertainties,
            "net_quantity": {
                "value": None,
                "unit": None,
                "derivation": "unknown",
                "unmet_conditions": unmet,
            },
            "provenance": [
                {
                    "frame_index": item["frame_index"],
                    "model": item.get("model"),
                    "request_id": item.get("request_id"),
                }
                for item in group
            ],
            "unresolved_questions": [
                {"code": code,
                 "message": code.lower().replace("_", " ")}
                for code in uncertainties
            ],
        })
        group.clear()

    for item in ordered:
        if group:
            gap = gap_seconds(group[-1], item)
            same_action = (action_of(item)
                           == action_of(group[-1]))
            shared = (ingredients_of(item)
                      & ingredients_of(group[-1]))
            either_empty = (not ingredients_of(item)
                            or not ingredients_of(group[-1]))
            if (gap is None or gap < 0
                    or gap > max_gap_seconds
                    or not same_action
                    or not (shared or either_empty)):
                flush()
        group.append(item)
    flush()
    return events


class CulinaryStore:
    """Durable per-run culinary observations (outside artifacts)."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def _path(self, run_id: str) -> Path:
        validate_run_id(run_id)
        return self._root / CULINARY_STORE_DIRNAME / f"{run_id}.json"

    def _checkpoints_path(self, run_id: str) -> Path:
        validate_run_id(run_id)
        return self._root / CULINARY_STORE_DIRNAME / f"{run_id}.jsonl"

    def load(self, run_id: str) -> dict[str, Any]:
        """Return stored observations; empty when never enriched."""
        path = self._path(run_id)
        if not path.is_file():
            return {"schema_version": CULINARY_OBSERVATION_VERSION,
                    "run_id": run_id, "observations": {},
                    "failures": {}, "failures_history": [],
                    "requeue_requests": [], "usage": []}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CulinaryError(
                f"Culinary observations for {run_id} are unreadable: "
                f"{exc}")
        if not isinstance(payload, dict) or not isinstance(
                payload.get("observations"), dict):
            raise CulinaryError(
                f"Culinary observations for {run_id} are malformed.")
        payload.setdefault("failures", {})
        payload.setdefault("failures_history", [])
        payload.setdefault("requeue_requests", [])
        payload.setdefault("usage", [])
        return payload

    def _save(self, run_id: str, payload: dict[str, Any]) -> None:
        path = self._path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n")
        tmp.replace(path)

    def record_observation(self, run_id: str, frame_index: int,
                           observation: dict[str, Any],
                           usage: dict[str, Any] | None = None,
                           ) -> None:
        """Persist one validated frame; terminal, never re-submitted."""
        stored = self.load(run_id)
        stored["observations"][str(frame_index)] = observation
        retired = stored.get("failures", {}).pop(str(frame_index), None)
        if retired is not None:
            stored.setdefault("failures_history", []).append({
                "frame_index": frame_index,
                "superseded_by": "observation",
                "at": utc_now_iso(),
                "failure": retired,
            })
        if usage:
            stored.setdefault("usage", []).append(usage)
        stored["updated_at"] = utc_now_iso()
        self._save(run_id, stored)
        self._append_checkpoint(run_id, {
            "at": utc_now_iso(),
            "outcome": "observed",
            "frame_index": frame_index,
        })

    def request_reprocessing(self, run_id: str,
                             frame_indices: list[Any],
                             reason: str, reviewer: str,
                             ) -> dict[str, Any]:
        """Authorize reprocessing of terminal-failed frames (explicit).

        The original failure entries are immutable: each is moved
        into failures_history with its error, timestamp, and this
        authorization, while the live failures map releases the
        frame so a later batch may submit it once. Observed frames
        and never-processed frames cannot be requeued; every call
        is recorded and checkpointed. Nothing is submitted here.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise CulinaryError(
                "INVALID_REQUEST: an explicit reason is required to "
                "reprocess failed frames.")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise CulinaryError(
                "INVALID_REQUEST: a reviewer identity is required to "
                "reprocess failed frames.")
        unique: list[int] = []
        for index in frame_indices or []:
            if isinstance(index, bool) or not isinstance(index, int):
                raise CulinaryError(
                    "INVALID_REQUEST: frame indices must be ints, "
                    f"got {index!r}.")
            if index not in unique:
                unique.append(index)
        if not unique:
            raise CulinaryError(
                "INVALID_REQUEST: no frame indices requested.")
        stored = self.load(run_id)
        failures = stored.setdefault("failures", {})
        observations = stored.get("observations", {})
        approved: list[int] = []
        for index in unique:
            key = str(index)
            if key in observations:
                raise CulinaryError(
                    "INVALID_REQUEST: frame "
                    f"{index} already has an observation; observed "
                    "frames cannot be requeued.")
            entry = failures.get(key)
            if entry is None:
                raise CulinaryError(
                    "INVALID_REQUEST: frame "
                    f"{index} has no terminal failure to reprocess.")
            stored.setdefault("failures_history", []).append({
                "frame_index": index,
                "superseded_by": "requeue",
                "at": utc_now_iso(),
                "failure": entry,
                "reason": reason.strip(),
                "reviewer": reviewer.strip(),
            })
            del failures[key]
            approved.append(index)
        stored.setdefault("requeue_requests", []).append({
            "at": utc_now_iso(),
            "frame_indices": approved,
            "reason": reason.strip(),
            "reviewer": reviewer.strip(),
        })
        stored["updated_at"] = utc_now_iso()
        self._save(run_id, stored)
        for index in approved:
            self._append_checkpoint(run_id, {
                "at": utc_now_iso(),
                "outcome": "requeued",
                "frame_index": index,
                "reason": reason.strip(),
                "reviewer": reviewer.strip(),
            })
        return {"run_id": run_id, "requeued": approved,
                "reason": reason.strip(), "reviewer": reviewer.strip()}

    def record_failure(self, run_id: str, frame_index: int,
                       error: str,
                       usage: dict[str, Any] | None = None,
                       ) -> None:
        """Persist one failed frame as terminal (no auto-retry)."""
        stored = self.load(run_id)
        stored.setdefault("failures", {})[str(frame_index)] = {
            "at": utc_now_iso(), "error": error,
        }
        if usage:
            stored.setdefault("usage", []).append(usage)
        stored["updated_at"] = utc_now_iso()
        self._save(run_id, stored)
        self._append_checkpoint(run_id, {
            "at": utc_now_iso(),
            "outcome": "failed",
            "frame_index": frame_index,
            "error": error,
        })

    def _append_checkpoint(self, run_id: str,
                           entry: dict[str, Any]) -> None:
        path = self._checkpoints_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")

    def completed_indices(self, run_id: str) -> set[int]:
        """Frame indices with a terminal outcome (observed/failed)."""
        stored = self.load(run_id)
        done = {int(key) for key in stored.get("observations", {})}
        done |= {int(key) for key in stored.get("failures", {})}
        return done


def describe_for_prompt(observation: dict[str, Any]) -> str:
    """Compact one-line summary of a validated observation."""
    actions = observation.get("actions") or []
    action = (actions[0].get("action_type")
              if actions and isinstance(actions[0], dict)
              else "no_action_visible")
    names = [str(item.get("name") or "unnamed")
             for item in observation.get("ingredient_candidates", [])
             if isinstance(item, dict)]
    return (f"frame {observation.get('frame_index')}: action={action} "
            f"ingredients=[{', '.join(names) or 'none'}]")


def enrich_next_batch(
        *,
        run_id: str,
        frames: list[dict[str, Any]],
        image_paths: list[Any],
        submit: Any,
        store: CulinaryStore,
        model: str = CULINARY_MODEL,
        api_key: str = "",
        # Headroom against reasoning-token exhaustion: the culinary
        # JSON is small, but reasoning shares this budget.
        max_output_tokens: int = 16384,
        max_frames: int = 5,
        ) -> dict[str, Any]:
    """Enrich the next unprocessed frames; one submit each, no retry.

    Frames already recorded (observed or failed) are skipped, so
    repeated calls resume instead of repeating billable work.
    Returns a progress summary; failures are terminal entries.
    The submit seam mirrors MetaSparkClient.submit.
    """
    if max_frames < 1:
        raise CulinaryError(
            "INVALID_CONFIG: max_frames must be a positive int, "
            f"got {max_frames!r}.")
    done = store.completed_indices(run_id)
    pending = [item for item in frames
               if item["frame_index"] not in done][:max_frames]
    by_index = {item["frame_index"]: path
                for item, path in zip(frames, image_paths)}
    enriched = 0
    failed = 0
    for item in pending:
        frame_index = item["frame_index"]
        path = by_index.get(frame_index)
        if path is None:
            store.record_failure(
                run_id, frame_index,
                "INVALID_REQUEST: no retained image for frame.")
            failed += 1
            continue
        try:
            response = submit(
                CULINARY_PROMPT,
                [path],
                api_key,
                max_output_tokens,
                CULINARY_JSON_SCHEMA,
                CULINARY_SCHEMA_NAME,
            )
        except TypeError:
            response = submit(CULINARY_PROMPT, [path], api_key,
                              max_output_tokens)
        except Exception as exc:
            store.record_failure(
                run_id, frame_index,
                f"{type(exc).__name__}: {exc}")
            failed += 1
            continue
        try:
            import json as _json
            payload = _json.loads(response.output_text)
            observation = validate_culinary_observation(
                payload, frame_index)
        except (ValueError, AttributeError, CulinaryError) as exc:
            store.record_failure(
                run_id, frame_index,
                f"INVALID_OBSERVATION: {exc}")
            failed += 1
            continue
        observation["model"] = getattr(
            response, "model", model) or model
        observation["request_id"] = getattr(
            response, "request_id", None)
        observation["timestamp_seconds"] = item.get(
            "timestamp_seconds")
        usage = getattr(response, "usage", None)
        dumped: dict[str, Any] = {}
        if usage is not None:
            dumped = (usage.model_dump()
                      if hasattr(usage, "model_dump") else dict(usage)
                      if isinstance(usage, dict) else {})
        record = {
            "frame_index": frame_index,
            "model": observation["model"],
            "request_id": observation["request_id"],
            "input_tokens": dumped.get("input_tokens"),
            "output_tokens": dumped.get("output_tokens"),
            "usage_status": ("reported"
                             if dumped.get("input_tokens") is not None
                             or dumped.get("output_tokens") is not None
                             else "unreported"),
            "max_output_tokens": max_output_tokens,
            "status": getattr(response, "status", None),
        }
        store.record_observation(
            run_id, frame_index, observation, record)
        enriched += 1
    total = len(frames)
    completed = len(store.completed_indices(run_id))
    return {
        "run_id": run_id,
        "model": model,
        "enriched": enriched,
        "failed": failed,
        "completed": completed,
        "total": total,
        "done": completed >= total,
    }
