"""Contract-level reference and measurement checks (M1-03).

Pure, deterministic structural checks over canonical ``schema_version
1.0.0`` bundles (``specs/video-to-recipe/tasks.md`` M1-03, ``plan.md`` §7
gates G1–G2). Every function returns a list of :class:`VerificationIssue`
and never raises on domain errors; an empty list means the checked
invariant holds.

Gate assignment follows the plan: G1 covers evidence normalization
(``FrameEvidence`` / ``Observation`` identity, timestamps, and observation
edges); G2 covers event assembly (``ActionEvent`` / ``WeighingEvent``
identity, interval containment, event edges, and measurement derivation).

These checks establish structural and semantic consistency only. A clean
result does not certify that any display, ingredient, or action was
interpreted correctly from pixels (Constitution X, plan.md §12).
"""

from __future__ import annotations

from collections.abc import Sequence

from bakery_intelligence.models import (
    ActionEvent,
    Derivation,
    FrameEvidence,
    GateId,
    Observation,
    VideoSource,
    WeighingEvent,
)
from bakery_intelligence.models.verification import VerificationIssue

SEVERITY_ERROR = "error"

DUPLICATE_ID = "DUPLICATE_ID"
INVALID_INTERVAL = "INVALID_INTERVAL"
TIMESTAMP_OUT_OF_BOUNDS = "TIMESTAMP_OUT_OF_BOUNDS"
DANGLING_REFERENCE = "DANGLING_REFERENCE"
DERIVATION_WITHOUT_UNIT = "DERIVATION_WITHOUT_UNIT"
UNKNOWN_DERIVATION_WITH_VALUE = "UNKNOWN_DERIVATION_WITH_VALUE"

__all__ = [
    "DANGLING_REFERENCE",
    "DERIVATION_WITHOUT_UNIT",
    "DUPLICATE_ID",
    "INVALID_INTERVAL",
    "SEVERITY_ERROR",
    "TIMESTAMP_OUT_OF_BOUNDS",
    "UNKNOWN_DERIVATION_WITH_VALUE",
    "check_interval_bounds",
    "check_measurement_derivation",
    "check_referential_integrity",
    "check_timestamps",
    "check_unique_ids",
]


def _unknown_derivation_violated(
    value: float | None, unit: str | None, derivation: Derivation
) -> bool:
    """Return True when an ``unknown`` derivation carries a value or unit.

    Defensive branch: :class:`NetIngredientQuantity` rejects
    ``derivation == "unknown"`` with non-null ``value``/``unit`` at parse
    time, so this predicate is unreachable via parsed models. It exists as
    a unit-testable guard so ``check_measurement_derivation`` stays total
    even if the model invariant is ever relaxed.
    """
    return derivation == Derivation.UNKNOWN and (
        value is not None or unit is not None
    )


def _duplicate_issues(
    ids_in_order: Sequence[str], label: str, gate: GateId
) -> list[VerificationIssue]:
    """Emit one :data:`DUPLICATE_ID` issue per repeated id value.

    Values are ordered by first appearance; cross-collection collisions
    are valid and never reported here.
    """
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for position, value in enumerate(ids_in_order):
        counts[value] = counts.get(value, 0) + 1
        if value not in first_seen:
            first_seen[value] = position
    duplicates = sorted(
        (value for value, count in counts.items() if count > 1),
        key=lambda value: first_seen[value],
    )
    return [
        VerificationIssue(
            code=DUPLICATE_ID,
            entity_id=value,
            gate=gate,
            message=(
                f"duplicate {label} id {value!r}: "
                f"appears {counts[value]} times"
            ),
            severity=SEVERITY_ERROR,
        )
        for value in duplicates
    ]


def check_unique_ids(
    frame_evidence: Sequence[FrameEvidence] = (),
    observations: Sequence[Observation] = (),
    action_events: Sequence[ActionEvent] = (),
    weighing_events: Sequence[WeighingEvent] = (),
) -> list[VerificationIssue]:
    """Report ids repeated within a single collection.

    Each collection is checked independently: the same id value appearing
    in two different collections is valid. One issue is emitted per
    duplicated value, ordered frame evidence, then observations, then
    action events, then weighing events; within a collection, values are
    ordered by first appearance. Frame evidence and observation issues
    carry gate G1; action and weighing event issues carry gate G2.
    """
    issues: list[VerificationIssue] = []
    issues.extend(
        _duplicate_issues(
            [item.evidence_id for item in frame_evidence],
            "frame evidence",
            GateId.G1,
        )
    )
    issues.extend(
        _duplicate_issues(
            [item.observation_id for item in observations],
            "observation",
            GateId.G1,
        )
    )
    issues.extend(
        _duplicate_issues(
            [item.action_id for item in action_events],
            "action event",
            GateId.G2,
        )
    )
    issues.extend(
        _duplicate_issues(
            [item.event_id for item in weighing_events],
            "weighing event",
            GateId.G2,
        )
    )
    return issues


def check_interval_bounds(
    video_source: VideoSource | None,
    action_events: Sequence[ActionEvent] = (),
    weighing_events: Sequence[WeighingEvent] = (),
) -> list[VerificationIssue]:
    """Report events not contained in the source interval (closed bounds).

    An event violates containment when ``start < source.start`` or
    ``end > source.end``. One :data:`INVALID_INTERVAL` issue (gate G2) is
    emitted per violating event, actions in input order followed by
    weighing events in input order. A ``None`` source yields ``[]``.
    Degenerate intervals (``start >= end``, negative endpoints) are owned
    by the model layer and are not re-checked here.
    """
    if video_source is None:
        return []
    source_start = video_source.interval.start_seconds
    source_end = video_source.interval.end_seconds
    issues: list[VerificationIssue] = []
    for action in action_events:
        if (
            action.start_seconds < source_start
            or action.end_seconds > source_end
        ):
            issues.append(
                VerificationIssue(
                    code=INVALID_INTERVAL,
                    entity_id=action.action_id,
                    gate=GateId.G2,
                    message=(
                        f"action event {action.action_id!r} interval "
                        f"[{action.start_seconds}, {action.end_seconds}] "
                        "not contained in source interval "
                        f"[{source_start}, {source_end}]"
                    ),
                    severity=SEVERITY_ERROR,
                )
            )
    for event in weighing_events:
        if (
            event.start_seconds < source_start
            or event.end_seconds > source_end
        ):
            issues.append(
                VerificationIssue(
                    code=INVALID_INTERVAL,
                    entity_id=event.event_id,
                    gate=GateId.G2,
                    message=(
                        f"weighing event {event.event_id!r} interval "
                        f"[{event.start_seconds}, {event.end_seconds}] "
                        "not contained in source interval "
                        f"[{source_start}, {source_end}]"
                    ),
                    severity=SEVERITY_ERROR,
                )
            )
    return issues


def check_timestamps(
    video_source: VideoSource | None,
    observations: Sequence[Observation] = (),
    frame_evidence: Sequence[FrameEvidence] = (),
) -> list[VerificationIssue]:
    """Report observation/evidence timestamps outside the source interval.

    For observations, either ``timestamp_seconds`` or the ``interval``
    endpoints are checked against the closed source interval; for frame
    evidence, ``timestamp_seconds`` is checked only when present.
    ``source_timestamp_seconds`` and ``requested_timestamp_seconds`` are
    extraction metadata, never domain timestamps, and are not checked.
    One :data:`TIMESTAMP_OUT_OF_BOUNDS` issue (gate G1) is emitted per
    violating record, observations in input order followed by frame
    evidence in input order. A ``None`` source yields ``[]``.
    """
    if video_source is None:
        return []
    source_start = video_source.interval.start_seconds
    source_end = video_source.interval.end_seconds
    issues: list[VerificationIssue] = []
    for observation in observations:
        if observation.timestamp_seconds is not None:
            timestamp = observation.timestamp_seconds
            if timestamp < source_start or timestamp > source_end:
                issues.append(
                    VerificationIssue(
                        code=TIMESTAMP_OUT_OF_BOUNDS,
                        entity_id=observation.observation_id,
                        gate=GateId.G1,
                        message=(
                            f"observation {observation.observation_id!r} "
                            f"timestamp {timestamp} outside source interval "
                            f"[{source_start}, {source_end}]"
                        ),
                        severity=SEVERITY_ERROR,
                    )
                )
        elif observation.interval is not None:
            start = observation.interval.start_seconds
            end = observation.interval.end_seconds
            if start < source_start or end > source_end:
                issues.append(
                    VerificationIssue(
                        code=TIMESTAMP_OUT_OF_BOUNDS,
                        entity_id=observation.observation_id,
                        gate=GateId.G1,
                        message=(
                            f"observation {observation.observation_id!r} "
                            f"interval [{start}, {end}] not contained in "
                            f"source interval [{source_start}, {source_end}]"
                        ),
                        severity=SEVERITY_ERROR,
                    )
                )
    for evidence in frame_evidence:
        if evidence.timestamp_seconds is not None:
            timestamp = evidence.timestamp_seconds
            if timestamp < source_start or timestamp > source_end:
                issues.append(
                    VerificationIssue(
                        code=TIMESTAMP_OUT_OF_BOUNDS,
                        entity_id=evidence.evidence_id,
                        gate=GateId.G1,
                        message=(
                            f"frame evidence {evidence.evidence_id!r} "
                            f"timestamp {timestamp} outside source interval "
                            f"[{source_start}, {source_end}]"
                        ),
                        severity=SEVERITY_ERROR,
                    )
                )
    return issues


def _dangling_issues(
    referrer_id: str,
    referrer_label: str,
    targets_in_order: Sequence[str],
    target_label: str,
    known_ids: set[str],
    gate: GateId,
) -> list[VerificationIssue]:
    """Emit one issue per (referrer, missing target) pair, in list order."""
    issues: list[VerificationIssue] = []
    emitted: set[str] = set()
    for target in targets_in_order:
        if target in known_ids or target in emitted:
            continue
        emitted.add(target)
        issues.append(
            VerificationIssue(
                code=DANGLING_REFERENCE,
                entity_id=referrer_id,
                gate=gate,
                message=(
                    f"{referrer_label} {referrer_id!r} references missing "
                    f"{target_label} {target!r}"
                ),
                severity=SEVERITY_ERROR,
            )
        )
    return issues


def check_referential_integrity(
    frame_evidence: Sequence[FrameEvidence] = (),
    observations: Sequence[Observation] = (),
    action_events: Sequence[ActionEvent] = (),
    weighing_events: Sequence[WeighingEvent] = (),
) -> list[VerificationIssue]:
    """Report references whose target id is absent from the bundle.

    Five edges are checked: ``Observation.evidence_ids`` against frame
    evidence (gate G1); ``ActionEvent.evidence_observation_ids``,
    ``WeighingEvent.action_ids``, ``WeighingEvent.observation_ids``, and
    ``WeighingEvent.final_display_reading.observation_id`` (gate G2).
    ``entity_id`` is the referrer id and the message names the missing
    target. One issue is emitted per (referrer, target) pair, ordered by
    observations in input order, then actions in input order, then
    weighing events in input order (``action_ids``, then
    ``observation_ids``, then ``final_display_reading``).
    """
    evidence_ids = {item.evidence_id for item in frame_evidence}
    observation_ids = {item.observation_id for item in observations}
    action_ids = {item.action_id for item in action_events}
    issues: list[VerificationIssue] = []
    for observation in observations:
        issues.extend(
            _dangling_issues(
                observation.observation_id,
                "observation",
                list(observation.evidence_ids),
                "frame evidence",
                evidence_ids,
                GateId.G1,
            )
        )
    for action in action_events:
        issues.extend(
            _dangling_issues(
                action.action_id,
                "action event",
                list(action.evidence_observation_ids),
                "observation",
                observation_ids,
                GateId.G2,
            )
        )
    for event in weighing_events:
        issues.extend(
            _dangling_issues(
                event.event_id,
                "weighing event",
                list(event.action_ids),
                "action event",
                action_ids,
                GateId.G2,
            )
        )
        issues.extend(
            _dangling_issues(
                event.event_id,
                "weighing event",
                list(event.observation_ids),
                "observation",
                observation_ids,
                GateId.G2,
            )
        )
        issues.extend(
            _dangling_issues(
                event.event_id,
                "weighing event",
                [event.final_display_reading.observation_id],
                "observation",
                observation_ids,
                GateId.G2,
            )
        )
    return issues


def check_measurement_derivation(
    weighing_events: Sequence[WeighingEvent] = (),
) -> list[VerificationIssue]:
    """Report incoherent derivation/value/unit combinations (gate G2).

    A non-``unknown`` derivation requires both a value and a unit; either
    missing yields :data:`DERIVATION_WITHOUT_UNIT`. Presence means
    non-``None``: a blank ``""`` unit is present, not missing. An
    ``unknown`` derivation with a non-null value or unit yields
    :data:`UNKNOWN_DERIVATION_WITH_VALUE`; that branch is defensive and
    unreachable via parsed models, which reject it at parse time (see
    :func:`_unknown_derivation_violated`). Issues follow input order.
    """
    issues: list[VerificationIssue] = []
    for event in weighing_events:
        quantity = event.net_ingredient_quantity
        value = quantity.value
        unit = quantity.unit
        derivation = quantity.derivation
        if _unknown_derivation_violated(value, unit, derivation):
            issues.append(
                VerificationIssue(
                    code=UNKNOWN_DERIVATION_WITH_VALUE,
                    entity_id=event.event_id,
                    gate=GateId.G2,
                    message=(
                        f"weighing event {event.event_id!r} has derivation "
                        f"{derivation.value!r} with non-null value/unit "
                        f"(value={value!r}, unit={unit!r})"
                    ),
                    severity=SEVERITY_ERROR,
                )
            )
        elif derivation != Derivation.UNKNOWN and (
            value is None or unit is None
        ):
            issues.append(
                VerificationIssue(
                    code=DERIVATION_WITHOUT_UNIT,
                    entity_id=event.event_id,
                    gate=GateId.G2,
                    message=(
                        f"weighing event {event.event_id!r} has derivation "
                        f"{derivation.value!r} with missing value/unit "
                        f"(value={value!r}, unit={unit!r})"
                    ),
                    severity=SEVERITY_ERROR,
                )
            )
    return issues
