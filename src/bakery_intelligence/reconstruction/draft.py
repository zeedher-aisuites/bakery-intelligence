"""Canonical draft assembly with explicit unknowns (M2-04).

Builds ActionEvent, WeighingEvent, and RecipeDraft records from
assembled measurements. Net quantities and ingredients stay unknown
unless supported; unresolved questions are explicit and actionable.
No step, quantity, or identity is ever invented.
"""

from __future__ import annotations

from typing import Any

from bakery_intelligence.models import (
    CANONICAL_SCHEMA_VERSION,
    ActionType,
    MeasurementMode,
    Provenance,
    SourceType,
    TimeInterval,
    TimestampBasis,
    VerificationStatus,
)
from bakery_intelligence.models.action_event import ActionEvent
from bakery_intelligence.models.recipe_draft import (
    DraftStep,
    EvidenceRefs,
    RecipeDraft,
    UnresolvedQuestion,
)
from bakery_intelligence.models.verification import VerificationIssue
from bakery_intelligence.models.video_source import VideoSource
from bakery_intelligence.models.weighing_event import (
    DisplayReading,
    NetIngredientQuantity,
    WeighingEvent,
)
from bakery_intelligence.verification.reference_checks import (
    check_interval_bounds,
    check_measurement_derivation,
    check_unique_ids,
)

ACTION_MAP = {
    "action_addition": ActionType.ADD,
    "action_removal": ActionType.REMOVE,
    "action_tare": ActionType.TARE,
}

SAMPLING_STEP_SECONDS = 1.0


def _padded_interval(start: float, end: float) -> tuple[float, float]:
    """Widen degenerate instants by one sampling step (documented)."""
    if start < end:
        return start, end
    return start, start + SAMPLING_STEP_SECONDS


def build_action_events(
        temporal_actions: list[dict[str, Any]],
        measurements: list[dict[str, Any]]) -> list[ActionEvent]:
    """Map temporal action candidates to canonical ActionEvents."""
    events = []
    for candidate in temporal_actions:
        start, end = _padded_interval(
            candidate["start_timestamp_seconds"],
            candidate["end_timestamp_seconds"],
        )
        evidence = [
            item["measurement_id"] for item in measurements
            if start <= item["timestamp_seconds"] <= end
        ]
        events.append(ActionEvent(
            schema_version=CANONICAL_SCHEMA_VERSION,
            action_id=f"act_{candidate['event_id']}",
            start_seconds=start,
            end_seconds=end,
            action_type=ACTION_MAP.get(
                candidate["candidate_type"], ActionType.UNKNOWN),
            ingredient_id=None,
            evidence_observation_ids=evidence,
            provenance=Provenance(
                source_type=SourceType.MODEL,
                source_id=f"temporal:{candidate['event_id']}",
            ),
            verification_status=VerificationStatus.AMBIGUOUS,
        ))
    return events


def build_weighing_event(
        temporal_event: dict[str, Any],
        measurements: list[dict[str, Any]],
        tare_observed: bool,
        action_ids: list[str] | None = None) -> WeighingEvent:
    """Build one weighing event; net quantity stays unknown in MVP."""
    usable = [item for item in measurements
              if item["value"] is not None]
    start, end = _padded_interval(
        temporal_event["start_timestamp_seconds"],
        temporal_event["end_timestamp_seconds"],
    )
    if usable:
        final = usable[-1]
        reading = DisplayReading(
            value=final["value"], unit=None,
            observation_id=final["measurement_id"],
        )
    else:
        reading = DisplayReading(
            value=None, unit=None,
            observation_id=measurements[-1]["measurement_id"],
        )
    values = [item["value"] for item in usable]
    return WeighingEvent(
        schema_version=CANONICAL_SCHEMA_VERSION,
        event_id=f"w_{temporal_event['event_id']}",
        start_seconds=start,
        end_seconds=end,
        ingredient_id=None,
        action_ids=list(action_ids or []),
        observation_ids=[item["measurement_id"]
                         for item in measurements],
        final_display_reading=reading,
        net_ingredient_quantity=NetIngredientQuantity(),
        measurement_mode=MeasurementMode.UNKNOWN,
        tare_observed=tare_observed,
        verification_status=VerificationStatus.AMBIGUOUS,
        review_notes=(
            f"{len(usable)} usable reading(s)"
            + (f", range {min(values)}..{max(values)}" if values else "")
            + "; tare observed" if tare_observed else
            f"{len(usable)} usable reading(s)"
            + (f", range {min(values)}..{max(values)}" if values else "")
            + "; no tare observed in window"
        ),
    )


def build_unresolved(event: dict[str, Any], tare_observed: bool,
                     human_unit: str | None) -> list[UnresolvedQuestion]:
    """Explicit gaps blocking a complete recipe."""
    questions = [
        UnresolvedQuestion(
            code="INGREDIENT_UNKNOWN",
            message="No observation identifies the ingredient; "
                    "container text is not an identity.",
            related_entity_ids=[event["event_id"]],
        ),
        UnresolvedQuestion(
            code="NET_QUANTITY_UNSUPPORTED",
            message="Computed deltas exist but no supported derivation "
                    "for a net ingredient quantity.",
            related_entity_ids=[event["event_id"]],
        ),
    ]
    unit_message = "No unit is visible in the source observations."
    if human_unit is not None:
        unit_message += (
            f" Human-assigned unit {human_unit!r} is recorded in the "
            "sidecar with event-window scope; it is not observed.")
    questions.append(UnresolvedQuestion(
        code="UNIT_VISUAL_ABSENT",
        message=unit_message,
        related_entity_ids=[event["event_id"]],
    ))
    if not tare_observed:
        questions.append(UnresolvedQuestion(
            code="UNKNOWN_TARE_HISTORY",
            message="No tare action observed in the window; net "
                    "derivation from differences is unsupported.",
            related_entity_ids=[event["event_id"]],
        ))
    reasons = event.get("review_reasons", [])
    if "temporal_inconsistency" in reasons:
        questions.append(UnresolvedQuestion(
            code="TEMPORAL_NOT_EVALUATED",
            message="Temporal inconsistency was flagged and has not "
                    "been independently resolved.",
            related_entity_ids=[event["event_id"]],
        ))
    return questions


def build_draft(draft_id: str, version: int, video_id: str,
                run_id: str, event: dict[str, Any],
                measurements: list[dict[str, Any]],
                actions: list[ActionEvent],
                unresolved: list[UnresolvedQuestion]) -> RecipeDraft:
    """Assemble a pending-review draft with one factual step."""
    usable = [item for item in measurements
              if item["value"] is not None]
    if usable:
        summary = (
            f"Scale displayed {len(usable)} readings from "
            f"{usable[0]['value']} to {usable[-1]['value']} across "
            f"{usable[0]['timestamp_seconds']:.1f}-"
            f"{usable[-1]['timestamp_seconds']:.1f} s "
            f"({len(actions)} supporting action event(s))."
        )
    else:
        summary = (
            f"No usable scale readings in "
            f"{event['start_timestamp_seconds']:.1f}-"
            f"{event['end_timestamp_seconds']:.1f} s."
        )
    step = DraftStep(
        index=0,
        description=summary,
        action_ids=[action.action_id for action in actions],
        observation_ids=[item["measurement_id"]
                         for item in measurements],
        evidence_summary=summary,
    )
    return RecipeDraft(
        schema_version=CANONICAL_SCHEMA_VERSION,
        draft_id=draft_id,
        version=version,
        video_id=video_id,
        steps=[step],
        ingredients=[],
        unresolved_questions=unresolved,
        evidence_refs=EvidenceRefs(
            observation_ids=[item["measurement_id"]
                             for item in measurements],
            frame_evidence_ids=[],
        ),
        verification_ref="verification_pending",
        run_id=run_id,
    )


def build_video_source(
        video_id: str, obs_index: dict[int, dict[str, Any]],
        source_dataset: str | None = None) -> VideoSource:
    """Source bounds derived from observed timestamps (no guessing)."""
    stamps = sorted(
        record.get("timestamp_seconds") for record in obs_index.values()
        if isinstance(record.get("timestamp_seconds"), (int, float)))
    if not stamps:
        raise ValueError("Cannot bound a source with no timestamps.")
    return VideoSource(
        schema_version=CANONICAL_SCHEMA_VERSION,
        video_id=video_id,
        source_ref=f"{video_id}.MP4",
        source_dataset=source_dataset,
        timestamp_basis=TimestampBasis.FRAME_INDEX_OVER_FPS,
        fps=None,
        frame_count=None,
        duration_seconds=float(stamps[-1]),
        interval=TimeInterval(start_seconds=float(stamps[0]),
                              end_seconds=float(stamps[-1])),
    )


def validate_bundle(
        video_source: VideoSource, actions: list[ActionEvent],
        weighings: list[WeighingEvent]) -> list[VerificationIssue]:
    """Run applicable reference checks (unique, bounds, derivation).

    Full referential integrity against canonical Observation and
    FrameEvidence stores is out of MVP scope: measurement IDs are
    experiment-scoped links, not ingested canonical records.
    """
    issues: list[VerificationIssue] = []
    issues.extend(check_unique_ids(
        action_events=actions, weighing_events=weighings))
    issues.extend(check_interval_bounds(
        video_source, actions, weighings))
    issues.extend(check_measurement_derivation(weighings))
    return issues
