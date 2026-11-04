"""Shared constants and record shapes for evidence fusion."""

from __future__ import annotations

from typing import Any, TypedDict

SOURCE_SPARK = "muse-spark-1.3-full-video"
SOURCE_SPARK_OPEN = "muse-spark-open-720p-segments"
SOURCE_PRELIMINARY = "preliminary-human-annotation"

STATUS_AI_SUGGESTED = "ai_suggested"
STATUS_UNKNOWN = "unknown"

CONFIDENCE_LEVELS = ("high", "medium", "low", "unknown")

# Explicit action aliases only; unmapped actions stay distinct.
# Tokens are normalized (lowercase, spaces to underscores) first, so
# morphological variants map without fuzzy matching.
ACTION_ALIASES = {
    "pouring": "pour",
    "stirring": "stir",
    "adding": "add",
    "mixing": "mix",
    "weighing": "weigh",
    "weighing_ingredient": "weigh",
    "retrieving": "retrieve",
    "retrieve_utensil": "retrieve",
    "retrieve_tool": "retrieve",
    "holding": "hold",
    "transferring": "transfer",
    "cover_bowl": "cover",
    "covering": "cover",
}

# Temporal tolerances (seconds).
OVERLAP_IOU_THRESHOLD = 0.3
NESTING_CONTAINMENT_THRESHOLD = 0.6
CONFLICT_IOU_THRESHOLD = 0.5
READING_MATCH_TOLERANCE = 2.0
INGREDIENT_MATCH_TOLERANCE = 5.0
PRELIMINARY_LINK_TOLERANCE = 30.0
# Coverage gaps inside one step wider than this are flagged so a
# phase-like group never reads as continuous evidence.
STEP_COVERAGE_GAP_THRESHOLD = 15.0


class NormalizedEvent(TypedDict, total=False):
    event_id: str
    source: str
    clip_id: str | None
    source_ref: str
    action_type: str | None
    action_canonical: str | None
    action_description: str | None
    start_seconds: float | None
    end_seconds: float | None
    approximate: bool
    ingredients: list[str]
    utensils: list[str]
    containers: list[str]
    evidence: str | None
    confidence: str
    uncertainty: str | None
    status: str
    note_kind: str


class NormalizedIngredient(TypedDict, total=False):
    record_id: str
    source: str
    clip_id: str | None
    source_ref: str
    normalized_name: str | None
    visible_label: str | None
    supporting_timestamps: list[float]
    used_or_visible: str | None
    evidence: str | None
    ambiguities: list[str]
    status: str


class NormalizedMeasurement(TypedDict, total=False):
    reading_id: str
    source: str
    clip_id: str | None
    source_ref: str
    timestamp_seconds: float | None
    displayed_value: str | None
    visible_unit: str | None
    tare_observable: bool | None
    uncertain: bool
    evidence: str | None
    status: str


def stable_id(*parts: Any) -> str:
    """Deterministic short id from stringified parts."""
    import hashlib

    joined = "|".join("" if part is None else str(part)
                      for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
