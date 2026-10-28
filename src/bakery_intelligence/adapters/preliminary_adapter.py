"""Preliminary annotation adapter: 0.1.0 human review to canonical 1.0.0.

Translates ``preliminary_schema_version 0.1.0`` sources (exemplified by
``experiments/003_video_to_recipe/annotations/preliminary/P04_104.json``)
into canonical ``schema_version 1.0.0`` :class:`Observation` records without
mutating the source file, inventing evidence, or promoting preliminary
notes to verified facts (tasks.md M1-02, plan.md §3.2, §11).

Mapping summary:

- ``timestamp_basis`` is always ``approximate_human_video_review``; the
  numeric ``approximate_timestamp_seconds`` becomes ``timestamp_seconds``.
  An approximate label is never converted into a frame-derived timestamp.
- Provenance is ``{source_type: adapter, source_id: "<file>@<sha256>"}``
  where the hash covers the exact original source bytes.
- ``verification_status`` is ``ambiguous`` for reported readings/states and
  ``unobservable`` when the note asserts absence (not reported, not
  applicable, no reading, display off, unconfirmed tare). It is never
  ``confirmed``.
- A display value is parsed only when the note carries exactly one
  backtick-quoted span holding an unambiguous signed decimal and no
  legibility-uncertainty cue (occluded, obscured, noisy, inferred,
  ambiguous, partial, suspected, apparent, spurious, unclear, illegible,
  unreadable, artifact) appears in the display text. Otherwise
  ``scale_value`` stays null and the span content (single span) or the full
  display text (zero or several spans) is preserved as
  ``raw_reported_value``.
- ``scale_unit`` stays null: the source carries only a
  ``scale_unit_hypothesis``, which is propagated as a hypothesis and never
  promoted to an observed unit.
- ``display_state`` is ``zero`` only for a directly reported ``0``,
  ``occluded`` only when the display text says so, and unset otherwise.
  Stability is never inferred.
- ``observation_type`` is ``scale_display`` for notes with display content,
  ``other`` for the explicit non-scale timer case, and ``action`` for the
  remaining action-centered notes.
- Records marked ``not_frame_verified`` produce no ``FrameEvidence``: no
  frame index, image reference, or evidence link is invented. A
  ``FrameEvidence`` record is emitted only when the source evidence
  carries a non-null frame index or image reference.
- ``description`` concatenates the segment/time label, reported action,
  reported display text, and interpretation note, so no source sentence is
  silently dropped.

Records that cannot be represented faithfully (missing identifier,
missing or invalid timestamp, failed canonical validation) are skipped
and reported as structured issues; nothing is fabricated for them.
``candidate_events`` and file-level video metadata have no canonical
M1-02 target (the output contract covers observations and frame evidence
only) and are therefore not adapted; the source file remains the audit
trail via ``source_hash``.

A parsed display value is a reported reading, not a net ingredient
quantity: this adapter infers no tare events, ingredient identity, net
mass, or measurement stability.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from bakery_intelligence.models import (
    CANONICAL_SCHEMA_VERSION,
    DisplayState,
    FrameEvidence,
    Modality,
    Observation,
    ObservationType,
    Provenance,
    SourceType,
    TimestampBasis,
    VerificationStatus,
)

PRELIMINARY_SCHEMA_VERSION = "0.1.0"

_SPAN_RE = re.compile(r"`([^`]*)`")
_NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")

_UNCERTAINTY_CUES = (
    "occlud",
    "obscur",
    "nois",
    "infer",
    "ambigu",
    "partial",
    "suspect",
    "apparent",
    "spurious",
    "unclear",
    "illegible",
    "unreadable",
    "artifact",
)

_UNOBSERVABLE_CUES = (
    "not reported",
    "not applicable",
    "no scale reading",
    "display off",
    "no confirmed tare",
)


class AdapterError(Exception):
    """Structured file-level adapter failure (code ``invalid_preliminary``)."""

    def __init__(self, code: str, entity_id: str, message: str) -> None:
        super().__init__(f"[{code}] {entity_id}: {message}")
        self.code = code
        self.entity_id = entity_id
        self.message = message


@dataclass
class AdapterIssue:
    """One skipped record or unsupported item with an actionable reason."""

    code: str
    entity_id: str
    message: str


@dataclass
class AdapterResult:
    """Deterministic outcome of adapting one preliminary source file."""

    observations: list[Observation] = field(default_factory=list)
    frame_evidence: list[FrameEvidence] = field(default_factory=list)
    issues: list[AdapterIssue] = field(default_factory=list)
    source_hash: str = ""


def _parse_display(
    display: str,
) -> tuple[float | None, str | None, DisplayState | None]:
    """Split a display note into value, raw text, and display state.

    Returns ``(scale_value, raw_reported_value, display_state)``. A numeric
    value is produced only for exactly one backtick span with unambiguous
    signed-decimal content and no legibility-uncertainty cue in the note.
    """
    text = display.strip()
    lowered = text.lower()
    spans = _SPAN_RE.findall(text)
    uncertain = any(cue in lowered for cue in _UNCERTAINTY_CUES)
    value: float | None = None
    if len(spans) == 1 and not uncertain and _NUMERIC_RE.match(spans[0]):
        value = float(spans[0])
    if value is not None:
        raw: str | None = None
    elif len(spans) == 1:
        raw = spans[0]
    else:
        raw = text
    if value is not None and value == 0.0:
        state: DisplayState | None = DisplayState.ZERO
    elif "occlud" in lowered or "obscur" in lowered:
        state = DisplayState.OCCLUDED
    else:
        state = None
    return value, raw, state


def _classify_type(
    display: str, interpretation: str, span_count: int
) -> ObservationType:
    """Classify the observation; the timer case is explicitly not a scale."""
    lowered_display = display.lower()
    if "timer" in lowered_display or "not a scale" in interpretation.lower():
        return ObservationType.OTHER
    if span_count > 0:
        return ObservationType.SCALE_DISPLAY
    if "display" in lowered_display and (
        re.search(r"\boff\b", lowered_display)
        or re.search(r"\bon\b", lowered_display)
    ):
        return ObservationType.SCALE_DISPLAY
    if "numbers visible" in lowered_display:
        return ObservationType.SCALE_DISPLAY
    return ObservationType.ACTION


def _classify_status(display: str) -> VerificationStatus:
    """Map absence-asserting notes to unobservable, the rest to ambiguous."""
    lowered = display.lower()
    if any(cue in lowered for cue in _UNOBSERVABLE_CUES):
        return VerificationStatus.UNOBSERVABLE
    return VerificationStatus.AMBIGUOUS


def _describe(
    segment_id: str,
    time_label: str,
    action: str,
    display: str,
    interpretation: str,
) -> str:
    """Build a lossless labeled description from the source sentences."""
    parts = [f"[{segment_id} {time_label}]".strip()]
    if action:
        parts.append(f"Reported action: {action}")
    if display:
        parts.append(f"Reported display or state: {display}")
    if interpretation:
        parts.append(f"Interpretation and verification need: {interpretation}")
    return " ".join(parts)


def _adapt_observation(
    raw: Any,
    segment_id: str,
    index: int,
    video_id: str,
    unit_hypothesis: str | None,
    provenance: Provenance,
    result: AdapterResult,
) -> None:
    """Adapt one raw observation or record a structured issue for it."""
    entity = f"{segment_id}[{index}]"
    if not isinstance(raw, dict):
        raise AdapterError(
            "invalid_preliminary",
            entity,
            f"observation at {entity} must be an object",
        )
    observation_id = raw.get("observation_id")
    if not isinstance(observation_id, str) or not observation_id.strip():
        result.issues.append(
            AdapterIssue(
                code="missing_observation_id",
                entity_id=entity,
                message=(
                    f"observation at {entity} has no usable identifier; "
                    "no canonical record emitted"
                ),
            )
        )
        return
    entity_id: str = observation_id.strip()
    stamp = raw.get("approximate_timestamp_seconds")
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
        result.issues.append(
            AdapterIssue(
                code="missing_timestamp",
                entity_id=entity_id,
                message=(
                    f"observation {entity_id} has no approximate timestamp; "
                    "canonical Observation requires exactly one temporal "
                    "form, so no canonical record emitted"
                ),
            )
        )
        return
    display = raw.get("reported_display_or_state", "")
    if not isinstance(display, str):
        display = ""
    interpretation = raw.get("interpretation_and_verification_need", "")
    if not isinstance(interpretation, str):
        interpretation = ""
    action = raw.get("reported_action", "")
    if not isinstance(action, str):
        action = ""
    time_label = raw.get("time_label", "")
    if not isinstance(time_label, str):
        time_label = ""
    value, raw_value, state = _parse_display(display)
    span_count = len(_SPAN_RE.findall(display.strip()))
    evidence_ids: list[str] = []
    evidence = raw.get("evidence")
    if isinstance(evidence, dict):
        frame_index = evidence.get("frame_index")
        image_ref = evidence.get("image_ref")
        if frame_index is not None or image_ref is not None:
            evidence_id = f"{entity_id}:frm"
            try:
                result.frame_evidence.append(
                    FrameEvidence(
                        schema_version=CANONICAL_SCHEMA_VERSION,
                        evidence_id=evidence_id,
                        video_id=video_id,
                        frame_index=frame_index,
                        image_ref=image_ref,
                    )
                )
            except ValidationError as exc:
                result.issues.append(
                    AdapterIssue(
                        code="invalid_evidence",
                        entity_id=entity_id,
                        message=(
                            f"observation {entity_id} carries frame "
                            f"evidence that failed canonical validation "
                            f"and was skipped: {exc}"
                        ),
                    )
                )
                return
            evidence_ids.append(evidence_id)
    try:
        result.observations.append(
            Observation(
                schema_version=CANONICAL_SCHEMA_VERSION,
                observation_id=entity_id,
                modality=Modality.VISUAL,
                timestamp_seconds=float(stamp),
                timestamp_basis=(
                    TimestampBasis.APPROXIMATE_HUMAN_VIDEO_REVIEW
                ),
                observation_type=_classify_type(
                    display, interpretation, span_count
                ),
                scale_value=value,
                scale_unit=None,
                scale_unit_hypothesis=unit_hypothesis,
                display_state=state,
                evidence_ids=evidence_ids,
                raw_reported_value=raw_value,
                description=_describe(
                    segment_id,
                    time_label.strip(),
                    action.strip(),
                    display.strip(),
                    interpretation.strip(),
                ),
                provenance=provenance,
                verification_status=_classify_status(display),
            )
        )
    except ValidationError as exc:
        result.issues.append(
            AdapterIssue(
                code="invalid_observation",
                entity_id=entity_id,
                message=(
                    f"observation {entity_id} failed canonical validation "
                    f"and was skipped: {exc}"
                ),
            )
        )


def adapt_preliminary(path: Path) -> AdapterResult:
    """Adapt a ``0.1.0`` preliminary file to canonical observations.

    Reads the exact source bytes (never modified), verifies the
    preliminary schema version and segment structure, and converts each
    observation deterministically. File-level malformation raises
    :class:`AdapterError`; per-record problems are collected as issues
    while adaptation of the remaining records continues.
    """
    raw_bytes = path.read_bytes()
    source_hash = hashlib.sha256(raw_bytes).hexdigest()
    try:
        document = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise AdapterError(
            "invalid_preliminary",
            path.name,
            f"source file is not valid JSON: {exc}",
        ) from exc
    if not isinstance(document, dict):
        raise AdapterError(
            "invalid_preliminary",
            path.name,
            "source document must be a JSON object",
        )
    version = document.get("preliminary_schema_version")
    if version != PRELIMINARY_SCHEMA_VERSION:
        raise AdapterError(
            "invalid_preliminary",
            path.name,
            f"unsupported preliminary_schema_version: {version!r}; "
            f"expected {PRELIMINARY_SCHEMA_VERSION!r}",
        )
    segments = document.get("segments")
    if not isinstance(segments, list):
        raise AdapterError(
            "invalid_preliminary",
            path.name,
            "source document has no usable segments list",
        )
    hypothesis = document.get("scale_unit_hypothesis")
    unit_hypothesis = (
        hypothesis if isinstance(hypothesis, str) and hypothesis else None
    )
    provenance = Provenance(
        source_type=SourceType.ADAPTER,
        source_id=f"{path.name}@{source_hash}",
    )
    video_id = document.get("video_id", "")
    if not isinstance(video_id, str):
        video_id = ""
    result = AdapterResult(source_hash=source_hash)
    for position, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise AdapterError(
                "invalid_preliminary",
                f"segments[{position}]",
                f"segment at index {position} must be an object",
            )
        segment_id = segment.get("segment_id", f"segments[{position}]")
        if not isinstance(segment_id, str) or not segment_id.strip():
            segment_id = f"segments[{position}]"
        else:
            segment_id = segment_id.strip()
        members = segment.get("observations")
        if not isinstance(members, list):
            raise AdapterError(
                "invalid_preliminary",
                segment_id,
                f"segment {segment_id} has no usable observations list",
            )
        for index, raw in enumerate(members):
            _adapt_observation(
                raw,
                segment_id,
                index,
                video_id,
                unit_hypothesis,
                provenance,
                result,
            )
    return result


def result_to_snapshot(result: AdapterResult) -> dict[str, Any]:
    """Render an AdapterResult as the golden-snapshot mapping.

    Field order is fixed and observations keep source order, so the
    rendered mapping is deterministic for a fixed input file.
    """
    return {
        "preliminary_schema_version": PRELIMINARY_SCHEMA_VERSION,
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "source_hash": result.source_hash,
        "observations": [
            observation.model_dump(mode="json")
            for observation in result.observations
        ],
        "frame_evidence": [
            evidence.model_dump(mode="json")
            for evidence in result.frame_evidence
        ],
        "issues": [
            {
                "code": issue.code,
                "entity_id": issue.entity_id,
                "message": issue.message,
            }
            for issue in result.issues
        ],
    }


__all__ = [
    "PRELIMINARY_SCHEMA_VERSION",
    "AdapterError",
    "AdapterIssue",
    "AdapterResult",
    "adapt_preliminary",
    "result_to_snapshot",
]
