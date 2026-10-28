"""Reference and measurement validation contract tests (M1-03).

Covers tasks.md M1-03 acceptance: ID uniqueness, interval bounds,
timestamp validity, referential integrity, and derivation-to-value/unit
coherence, per plan.md section 7 gates G1-G2 and constitution III, IV, X.

Expected behaviors derive from the frozen M1-03 brief, tasks.md M1-03,
plan.md section 7, and constitution III/IV/X only. The M1-01 models used
below are parsed fixtures, not oracles: they establish which inputs can
exist, never what the reference checks must report.

Model-layer ownership notes (M1-01, cited where asserted):

- Degenerate event intervals (``start >= end``, negative bounds) are
  rejected by ``ActionEvent``/``WeighingEvent`` at parse time, so no
  parsed-model fixture can carry them into ``check_interval_bounds``.
- ``derivation == "unknown"`` with a non-null value or unit is rejected
  by ``NetIngredientQuantity`` at parse time, so end-to-end
  ``UNKNOWN_DERIVATION_WITH_VALUE`` is unreachable via parsed models.
  The defensive branch is covered through direct unit tests of the
  module-level ``_unknown_derivation_violated`` predicate.

Ordering discipline: every check is pure, deterministic, and
input-ordered. Issue order follows first appearance in the inputs within
each collection and the frozen collection/edge order across them; output
is never sorted. A passing suite attests structural consistency only,
never factual correctness of any visual interpretation (plan.md section
12). No aggregate entry point exists by design; each check is exercised
directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from bakery_intelligence.models import (
    ActionEvent,
    Derivation,
    FrameEvidence,
    GateId,
    Observation,
    VideoSource,
    WeighingEvent,
)
from bakery_intelligence.verification.reference_checks import (
    DANGLING_REFERENCE,
    DERIVATION_WITHOUT_UNIT,
    DUPLICATE_ID,
    INVALID_INTERVAL,
    TIMESTAMP_OUT_OF_BOUNDS,
    UNKNOWN_DERIVATION_WITH_VALUE,
    SEVERITY_ERROR,
    _unknown_derivation_violated,
    check_interval_bounds,
    check_measurement_derivation,
    check_referential_integrity,
    check_timestamps,
    check_unique_ids,
)

GOLDEN = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "data"
    / "canonical"
    / "P04_104.canonical.json"
)

SUPPORTED_DERIVATIONS = [
    "direct_after_tare",
    "difference_between_readings",
    "independently_verified",
]


def _video_source_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "video_id": "SYNTH.FIXTURE.001",
        "source_ref": "synthetic/nominal.mp4",
        "source_dataset": "synthetic-fixture",
        "timestamp_basis": "synthetic_fixture",
        "fps": 25.0,
        "frame_count": 250,
        "duration_seconds": 10.0,
        "interval": {"start_seconds": 0.0, "end_seconds": 10.0},
        "license": "CC-BY-NC-4.0",
    }


def _frame_evidence_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "evidence_id": "frm_00000120",
        "video_id": "SYNTH.FIXTURE.001",
        "frame_index": 120,
        "timestamp_seconds": 4.8,
        "source_timestamp_seconds": 4.8,
        "requested_timestamp_seconds": 4.8,
        "image_ref": "runs/demo/frames/frame_00000120.jpg",
        "region": None,
        "extraction": None,
    }


def _observation_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "observation_id": "obs_001",
        "modality": "visual",
        "timestamp_seconds": 4.8,
        "timestamp_basis": "synthetic_fixture",
        "observation_type": "scale_display",
        "scale_value": 120.0,
        "scale_unit": "g",
        "scale_unit_hypothesis": None,
        "display_state": "stable",
        "evidence_ids": ["frm_00000120"],
        "raw_reported_value": None,
        "description": "Display reads 120 g in the sampled frame.",
        "provenance": {
            "source_type": "human",
            "source_id": "annotator_01",
        },
        "verification_status": "ambiguous",
    }


def _action_event_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "action_id": "act_001",
        "start_seconds": 2.0,
        "end_seconds": 4.0,
        "action_type": "tare",
        "ingredient_id": None,
        "evidence_observation_ids": ["obs_001"],
        "provenance": {
            "source_type": "human",
            "source_id": "annotator_01",
        },
        "verification_status": "ambiguous",
    }


def _weighing_event_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "event_id": "weigh_001",
        "start_seconds": 2.0,
        "end_seconds": 6.0,
        "ingredient_id": "flour",
        "action_ids": ["act_001"],
        "observation_ids": ["obs_001"],
        "final_display_reading": {
            "value": 120.0,
            "unit": "g",
            "observation_id": "obs_001",
        },
        "net_ingredient_quantity": {
            "value": 120.0,
            "unit": "g",
            "derivation": "direct_after_tare",
        },
        "measurement_mode": "individual",
        "tare_observed": True,
        "verification_status": "ambiguous",
        "review_notes": "Tare observed before the addition.",
    }


def _make_video_source(**overrides: Any) -> VideoSource:
    payload = _video_source_payload()
    payload.update(overrides)
    return VideoSource.model_validate(payload)


def _make_evidence(**overrides: Any) -> FrameEvidence:
    payload = _frame_evidence_payload()
    payload.update(overrides)
    return FrameEvidence.model_validate(payload)


def _make_observation(**overrides: Any) -> Observation:
    payload = _observation_payload()
    payload.update(overrides)
    return Observation.model_validate(payload)


def _make_interval_observation(
    observation_id: str, start: float, end: float
) -> Observation:
    return _make_observation(
        observation_id=observation_id,
        timestamp_seconds=None,
        interval={"start_seconds": start, "end_seconds": end},
    )


def _make_action(**overrides: Any) -> ActionEvent:
    payload = _action_event_payload()
    payload.update(overrides)
    return ActionEvent.model_validate(payload)


def _make_weighing(**overrides: Any) -> WeighingEvent:
    payload = _weighing_event_payload()
    payload.update(overrides)
    return WeighingEvent.model_validate(payload)


def _make_item(collection: str, item_id: str) -> Any:
    """Build one real M1-01 model instance for a named collection."""
    if collection == "frame_evidence":
        return _make_evidence(evidence_id=item_id)
    if collection == "observations":
        return _make_observation(observation_id=item_id)
    if collection == "action_events":
        return _make_action(action_id=item_id)
    if collection == "weighing_events":
        return _make_weighing(event_id=item_id)
    raise AssertionError(f"unknown collection: {collection}")


def _bounded_source() -> VideoSource:
    """Source with a strict sub-interval to test closed-bound containment."""
    return _make_video_source(
        interval={"start_seconds": 2.0, "end_seconds": 8.0}
    )


# Valid bundle and issue shape.


def test_valid_bundle_all_checks_pass() -> None:
    """A coherent bundle yields no issues from any of the five checks."""
    source = _make_video_source()
    evidence = [_make_evidence()]
    observations = [_make_observation()]
    actions = [_make_action()]
    weighings = [_make_weighing()]
    assert check_unique_ids(
        frame_evidence=evidence,
        observations=observations,
        action_events=actions,
        weighing_events=weighings,
    ) == []
    assert check_interval_bounds(
        source, action_events=actions, weighing_events=weighings
    ) == []
    assert check_timestamps(
        source, observations=observations, frame_evidence=evidence
    ) == []
    assert check_referential_integrity(
        frame_evidence=evidence,
        observations=observations,
        action_events=actions,
        weighing_events=weighings,
    ) == []
    assert check_measurement_derivation(weighings) == []


def test_issue_codes_match_contract_literals() -> None:
    """Frozen M1-03 code vocabulary pins the tasks.md contract strings."""
    assert DUPLICATE_ID == "DUPLICATE_ID"
    assert INVALID_INTERVAL == "INVALID_INTERVAL"
    assert TIMESTAMP_OUT_OF_BOUNDS == "TIMESTAMP_OUT_OF_BOUNDS"
    assert DANGLING_REFERENCE == "DANGLING_REFERENCE"
    assert DERIVATION_WITHOUT_UNIT == "DERIVATION_WITHOUT_UNIT"
    assert UNKNOWN_DERIVATION_WITH_VALUE == "UNKNOWN_DERIVATION_WITH_VALUE"
    assert SEVERITY_ERROR == "error"


def test_issue_shape_fields_present() -> None:
    """Every issue carries a code, entity, gate, message, and severity."""
    source = _make_video_source()
    dup_a = _make_observation(observation_id="obs_dup")
    dup_b = _make_observation(observation_id="obs_dup")
    bad_action = _make_action(
        action_id="act_bad", start_seconds=20.0, end_seconds=30.0
    )
    bad_obs = _make_observation(
        observation_id="obs_bad", timestamp_seconds=99.0
    )
    dangling = _make_observation(
        observation_id="obs_dangling", evidence_ids=["frm_missing"]
    )
    bad_weighing = _make_weighing(
        net_ingredient_quantity={
            "value": None,
            "unit": "g",
            "derivation": "direct_after_tare",
        }
    )
    all_issues = [
        *check_unique_ids(observations=[dup_a, dup_b]),
        *check_interval_bounds(source, action_events=[bad_action]),
        *check_timestamps(source, observations=[bad_obs]),
        *check_referential_integrity(observations=[dangling]),
        *check_measurement_derivation([bad_weighing]),
    ]
    assert len(all_issues) == 5
    assert all_issues[0].severity == "error"
    for issue in all_issues:
        assert issue.code
        assert isinstance(issue.entity_id, str) and issue.entity_id
        assert issue.message
        assert issue.gate in (GateId.G1, GateId.G2)
        assert issue.severity == SEVERITY_ERROR


def test_checks_never_throw_on_domain_errors() -> None:
    """Failing fixtures return issue lists; domain errors never raise."""
    source = _make_video_source()
    dup_a = _make_observation(observation_id="obs_dup")
    dup_b = _make_observation(observation_id="obs_dup")
    bad_action = _make_action(
        action_id="act_bad", start_seconds=20.0, end_seconds=30.0
    )
    bad_obs = _make_observation(
        observation_id="obs_bad", timestamp_seconds=99.0
    )
    dangling = _make_observation(
        observation_id="obs_dangling", evidence_ids=["frm_missing"]
    )
    bad_weighing = _make_weighing(
        net_ingredient_quantity={
            "value": 120.0,
            "unit": None,
            "derivation": "direct_after_tare",
        }
    )
    results = [
        check_unique_ids(observations=[dup_a, dup_b]),
        check_interval_bounds(source, action_events=[bad_action]),
        check_timestamps(source, observations=[bad_obs]),
        check_referential_integrity(observations=[dangling]),
        check_measurement_derivation([bad_weighing]),
    ]
    for result in results:
        assert isinstance(result, list)
        assert len(result) > 0


def test_checks_empty_inputs_return_empty() -> None:
    source = _make_video_source()
    assert check_unique_ids() == []
    assert check_interval_bounds(source) == []
    assert check_timestamps(source) == []
    assert check_referential_integrity() == []
    assert check_measurement_derivation() == []


# DUPLICATE_ID.


@pytest.mark.parametrize(
    ("collection", "item_id", "expected_gate"),
    [
        ("frame_evidence", "frm_dup", GateId.G1),
        ("observations", "obs_dup", GateId.G1),
        ("action_events", "act_dup", GateId.G2),
        ("weighing_events", "weigh_dup", GateId.G2),
    ],
)
def test_unique_ids_duplicate_within_collection(
    collection: str, item_id: str, expected_gate: GateId
) -> None:
    first = _make_item(collection, item_id)
    second = _make_item(collection, item_id)
    issues = check_unique_ids(**{collection: [first, second]})
    assert len(issues) == 1
    assert issues[0].code == "DUPLICATE_ID"
    assert issues[0].entity_id == item_id
    assert issues[0].gate == expected_gate
    assert issues[0].severity == SEVERITY_ERROR
    assert issues[0].message


def test_unique_ids_single_issue_per_value() -> None:
    first = _make_observation(observation_id="obs_dup")
    second = _make_observation(observation_id="obs_dup")
    third = _make_observation(observation_id="obs_dup")
    issues = check_unique_ids(observations=[first, second, third])
    assert len(issues) == 1
    assert issues[0].code == "DUPLICATE_ID"
    assert issues[0].entity_id == "obs_dup"


def test_unique_ids_collection_order() -> None:
    """Collections report frame_evidence, observations, actions, events."""
    evidence = [
        _make_evidence(evidence_id="frm_dup"),
        _make_evidence(evidence_id="frm_dup"),
    ]
    observations = [
        _make_observation(observation_id="obs_dup"),
        _make_observation(observation_id="obs_dup"),
    ]
    actions = [
        _make_action(action_id="act_dup"),
        _make_action(action_id="act_dup"),
    ]
    weighings = [
        _make_weighing(event_id="weigh_dup"),
        _make_weighing(event_id="weigh_dup"),
    ]
    issues = check_unique_ids(
        frame_evidence=evidence,
        observations=observations,
        action_events=actions,
        weighing_events=weighings,
    )
    assert [issue.entity_id for issue in issues] == [
        "frm_dup",
        "obs_dup",
        "act_dup",
        "weigh_dup",
    ]
    assert [issue.gate for issue in issues] == [
        GateId.G1,
        GateId.G1,
        GateId.G2,
        GateId.G2,
    ]


def test_unique_ids_first_appearance_order() -> None:
    """Within a collection, values report in first-appearance order."""
    items = [
        _make_observation(observation_id="zeta_dup"),
        _make_observation(observation_id="alpha_dup"),
        _make_observation(observation_id="zeta_dup"),
        _make_observation(observation_id="alpha_dup"),
    ]
    issues = check_unique_ids(observations=items)
    assert [issue.entity_id for issue in issues] == [
        "zeta_dup",
        "alpha_dup",
    ]


def test_unique_ids_cross_collection_collision_valid() -> None:
    """The same value across different collections is not a duplicate."""
    evidence = [_make_evidence(evidence_id="shared_001")]
    observations = [_make_observation(observation_id="shared_001")]
    actions = [_make_action(action_id="shared_001")]
    weighings = [_make_weighing(event_id="shared_001")]
    assert check_unique_ids(
        frame_evidence=evidence,
        observations=observations,
        action_events=actions,
        weighing_events=weighings,
    ) == []


def test_unique_ids_shared_timestamps_valid() -> None:
    """Distinct records may share a timestamp without colliding."""
    observations = [
        _make_observation(
            observation_id="obs_a", timestamp_seconds=4.8
        ),
        _make_observation(
            observation_id="obs_b", timestamp_seconds=4.8
        ),
    ]
    evidence = [
        _make_evidence(evidence_id="frm_a", timestamp_seconds=4.8),
        _make_evidence(evidence_id="frm_b", timestamp_seconds=4.8),
    ]
    assert check_unique_ids(
        frame_evidence=evidence, observations=observations
    ) == []


# INVALID_INTERVAL.


@pytest.mark.parametrize("event_kind", ["action", "weighing"])
def test_interval_bounds_start_before_source_start(
    event_kind: str,
) -> None:
    """An event starting before the source interval is not contained."""
    source = _bounded_source()
    if event_kind == "action":
        actions = [
            _make_action(
                action_id="act_early",
                start_seconds=1.0,
                end_seconds=4.0,
            )
        ]
        issues = check_interval_bounds(source, action_events=actions)
        assert len(issues) == 1
        assert issues[0].code == "INVALID_INTERVAL"
        assert issues[0].entity_id == "act_early"
    else:
        weighings = [
            _make_weighing(
                event_id="weigh_early",
                start_seconds=1.0,
                end_seconds=4.0,
            )
        ]
        issues = check_interval_bounds(source, weighing_events=weighings)
        assert len(issues) == 1
        assert issues[0].code == "INVALID_INTERVAL"
        assert issues[0].entity_id == "weigh_early"
    assert issues[0].gate == GateId.G2
    assert issues[0].severity == SEVERITY_ERROR
    assert issues[0].message


@pytest.mark.parametrize("event_kind", ["action", "weighing"])
def test_interval_bounds_end_after_source_end(
    event_kind: str,
) -> None:
    """An event ending after the source interval is not contained."""
    source = _bounded_source()
    if event_kind == "action":
        actions = [
            _make_action(
                action_id="act_late",
                start_seconds=4.0,
                end_seconds=9.0,
            )
        ]
        issues = check_interval_bounds(source, action_events=actions)
        assert len(issues) == 1
        assert issues[0].code == "INVALID_INTERVAL"
        assert issues[0].entity_id == "act_late"
    else:
        weighings = [
            _make_weighing(
                event_id="weigh_late",
                start_seconds=4.0,
                end_seconds=9.0,
            )
        ]
        issues = check_interval_bounds(source, weighing_events=weighings)
        assert len(issues) == 1
        assert issues[0].code == "INVALID_INTERVAL"
        assert issues[0].entity_id == "weigh_late"
    assert issues[0].gate == GateId.G2
    assert issues[0].severity == SEVERITY_ERROR
    assert issues[0].message


def test_interval_bounds_boundary_valid() -> None:
    """Closed bounds: touching the source edges is still contained."""
    source = _bounded_source()
    actions = [
        _make_action(
            action_id="act_exact",
            start_seconds=2.0,
            end_seconds=8.0,
        ),
        _make_action(
            action_id="act_left",
            start_seconds=2.0,
            end_seconds=5.0,
        ),
        _make_action(
            action_id="act_right",
            start_seconds=5.0,
            end_seconds=8.0,
        ),
    ]
    weighings = [
        _make_weighing(
            event_id="weigh_exact",
            start_seconds=2.0,
            end_seconds=8.0,
        )
    ]
    assert check_interval_bounds(
        source, action_events=actions, weighing_events=weighings
    ) == []


def test_interval_bounds_action_before_weighing_order() -> None:
    """Action issues report before weighing issues, in input order."""
    source = _bounded_source()
    actions = [
        _make_action(
            action_id="act_late", start_seconds=4.0, end_seconds=9.0
        )
    ]
    weighings = [
        _make_weighing(
            event_id="weigh_late",
            start_seconds=4.0,
            end_seconds=9.0,
        )
    ]
    issues = check_interval_bounds(
        source, action_events=actions, weighing_events=weighings
    )
    assert [issue.entity_id for issue in issues] == [
        "act_late",
        "weigh_late",
    ]


def test_interval_bounds_none_source_returns_empty() -> None:
    """Without source context, containment cannot be judged: no issues."""
    actions = [
        _make_action(
            action_id="act_far", start_seconds=20.0, end_seconds=30.0
        )
    ]
    weighings = [
        _make_weighing(
            event_id="weigh_far", start_seconds=20.0, end_seconds=30.0
        )
    ]
    assert check_interval_bounds(None, action_events=actions) == []
    assert check_interval_bounds(None, weighing_events=weighings) == []


@pytest.mark.parametrize("model_name", ["ActionEvent", "WeighingEvent"])
@pytest.mark.parametrize(
    ("start", "end"),
    [(4.0, 4.0), (5.0, 4.0), (-1.0, 2.0)],
    ids=["degenerate", "inverted", "negative"],
)
def test_model_rejects_degenerate_event_intervals(
    model_name: str, start: float, end: float
) -> None:
    """Degenerate intervals are M1-01 model-owned: parsing raises, so no
    parsed-model fixture can carry them into check_interval_bounds."""
    if model_name == "ActionEvent":
        payload: dict[str, Any] = _action_event_payload()
        model_cls: Any = ActionEvent
    else:
        payload = _weighing_event_payload()
        model_cls = WeighingEvent
    payload["start_seconds"] = start
    payload["end_seconds"] = end
    with pytest.raises(ValidationError):
        model_cls.model_validate(payload)


# TIMESTAMP_OUT_OF_BOUNDS.


@pytest.mark.parametrize(
    "timestamp", [1.0, 9.0], ids=["before_start", "after_end"]
)
def test_timestamps_observation_out_of_bounds(timestamp: float) -> None:
    source = _bounded_source()
    observations = [
        _make_observation(
            observation_id="obs_bad", timestamp_seconds=timestamp
        )
    ]
    issues = check_timestamps(source, observations=observations)
    assert len(issues) == 1
    assert issues[0].code == "TIMESTAMP_OUT_OF_BOUNDS"
    assert issues[0].entity_id == "obs_bad"
    assert issues[0].gate == GateId.G1
    assert issues[0].severity == SEVERITY_ERROR
    assert issues[0].message


@pytest.mark.parametrize(
    ("start", "end"),
    [(1.0, 4.0), (4.0, 9.0)],
    ids=["start_before_source", "end_after_source"],
)
def test_timestamps_observation_interval_endpoints(
    start: float, end: float
) -> None:
    """Interval-form observations are judged on both endpoints."""
    source = _bounded_source()
    observations = [_make_interval_observation("obs_range", start, end)]
    issues = check_timestamps(source, observations=observations)
    assert len(issues) == 1
    assert issues[0].code == "TIMESTAMP_OUT_OF_BOUNDS"
    assert issues[0].entity_id == "obs_range"
    assert issues[0].gate == GateId.G1


def test_timestamps_frame_evidence_out_of_bounds() -> None:
    source = _bounded_source()
    evidence = [
        _make_evidence(
            evidence_id="frm_bad", timestamp_seconds=9.0
        )
    ]
    issues = check_timestamps(source, frame_evidence=evidence)
    assert len(issues) == 1
    assert issues[0].code == "TIMESTAMP_OUT_OF_BOUNDS"
    assert issues[0].entity_id == "frm_bad"
    assert issues[0].gate == GateId.G1
    assert issues[0].severity == SEVERITY_ERROR


def test_timestamps_frame_evidence_none_valid() -> None:
    """Evidence without a timestamp has nothing to judge: no issue."""
    source = _bounded_source()
    evidence = [
        _make_evidence(
            evidence_id="frm_notimed",
            timestamp_seconds=None,
        )
    ]
    assert check_timestamps(source, frame_evidence=evidence) == []


def test_timestamps_ignores_source_and_requested_timestamps() -> None:
    """Only timestamp_seconds is checked; source/requested stamps are not."""
    source = _bounded_source()
    evidence = [
        _make_evidence(
            evidence_id="frm_aux_only",
            timestamp_seconds=None,
            source_timestamp_seconds=99.0,
            requested_timestamp_seconds=99.0,
        ),
        _make_evidence(
            evidence_id="frm_aux_noisy",
            timestamp_seconds=4.0,
            source_timestamp_seconds=99.0,
            requested_timestamp_seconds=99.0,
        ),
    ]
    assert check_timestamps(source, frame_evidence=evidence) == []


def test_timestamps_boundary_valid() -> None:
    """Closed bounds: exact-edge timestamps and intervals are contained."""
    source = _bounded_source()
    observations = [
        _make_observation(
            observation_id="obs_left", timestamp_seconds=2.0
        ),
        _make_observation(
            observation_id="obs_right", timestamp_seconds=8.0
        ),
        _make_interval_observation("obs_exact", 2.0, 8.0),
    ]
    evidence = [
        _make_evidence(evidence_id="frm_left", timestamp_seconds=2.0),
        _make_evidence(evidence_id="frm_right", timestamp_seconds=8.0),
    ]
    assert check_timestamps(
        source, observations=observations, frame_evidence=evidence
    ) == []


def test_timestamps_observations_before_evidence_order() -> None:
    """Observation issues report before evidence issues, in input order."""
    source = _bounded_source()
    observations = [
        _make_observation(
            observation_id="obs_second", timestamp_seconds=9.0
        ),
        _make_observation(
            observation_id="obs_first", timestamp_seconds=1.0
        ),
    ]
    evidence = [
        _make_evidence(evidence_id="frm_bad", timestamp_seconds=9.0)
    ]
    issues = check_timestamps(
        source, observations=observations, frame_evidence=evidence
    )
    assert [issue.entity_id for issue in issues] == [
        "obs_second",
        "obs_first",
        "frm_bad",
    ]


def test_timestamps_none_source_returns_empty() -> None:
    """Without source context, containment cannot be judged: no issues."""
    observations = [
        _make_observation(
            observation_id="obs_far", timestamp_seconds=99.0
        )
    ]
    evidence = [
        _make_evidence(evidence_id="frm_far", timestamp_seconds=99.0)
    ]
    assert check_timestamps(None, observations=observations) == []
    assert check_timestamps(None, frame_evidence=evidence) == []


# DANGLING_REFERENCE.


def test_referential_integrity_observation_evidence() -> None:
    """Observation evidence_ids must resolve to frame evidence (G1)."""
    observations = [
        _make_observation(
            observation_id="obs_held",
            evidence_ids=["frm_missing"],
        )
    ]
    issues = check_referential_integrity(observations=observations)
    assert len(issues) == 1
    assert issues[0].code == "DANGLING_REFERENCE"
    assert issues[0].entity_id == "obs_held"
    assert issues[0].gate == GateId.G1
    assert issues[0].severity == SEVERITY_ERROR
    assert "frm_missing" in issues[0].message


def test_referential_integrity_action_observations() -> None:
    """Action evidence_observation_ids must resolve to observations (G2)."""
    actions = [
        _make_action(
            action_id="act_held",
            evidence_observation_ids=["obs_missing"],
        )
    ]
    issues = check_referential_integrity(action_events=actions)
    assert len(issues) == 1
    assert issues[0].code == "DANGLING_REFERENCE"
    assert issues[0].entity_id == "act_held"
    assert issues[0].gate == GateId.G2
    assert issues[0].severity == SEVERITY_ERROR
    assert "obs_missing" in issues[0].message


def test_referential_integrity_weighing_action_ids() -> None:
    """Weighing action_ids must resolve to action events (G2)."""
    observations = [_make_observation()]
    weighings = [
        _make_weighing(
            event_id="weigh_held",
            action_ids=["act_missing"],
            observation_ids=["obs_001"],
            final_display_reading={
                "value": 120.0,
                "unit": "g",
                "observation_id": "obs_001",
            },
        )
    ]
    issues = check_referential_integrity(
        observations=observations, weighing_events=weighings
    )
    assert len(issues) == 1
    assert issues[0].code == "DANGLING_REFERENCE"
    assert issues[0].entity_id == "weigh_held"
    assert issues[0].gate == GateId.G2
    assert "act_missing" in issues[0].message


def test_referential_integrity_weighing_observation_ids() -> None:
    """Weighing observation_ids must resolve to observations (G2)."""
    actions = [_make_action()]
    observations = [_make_observation()]
    weighings = [
        _make_weighing(
            event_id="weigh_held",
            action_ids=["act_001"],
            observation_ids=["obs_missing"],
            final_display_reading={
                "value": 120.0,
                "unit": "g",
                "observation_id": "obs_001",
            },
        )
    ]
    issues = check_referential_integrity(
        observations=observations,
        action_events=actions,
        weighing_events=weighings,
    )
    assert len(issues) == 1
    assert issues[0].code == "DANGLING_REFERENCE"
    assert issues[0].entity_id == "weigh_held"
    assert issues[0].gate == GateId.G2
    assert "obs_missing" in issues[0].message


def test_referential_integrity_weighing_final_reading() -> None:
    """Weighing final_display_reading must cite a real observation (G2)."""
    actions = [_make_action()]
    observations = [_make_observation()]
    weighings = [
        _make_weighing(
            event_id="weigh_held",
            action_ids=["act_001"],
            observation_ids=["obs_001"],
            final_display_reading={
                "value": 120.0,
                "unit": "g",
                "observation_id": "obs_missing_final",
            },
        )
    ]
    issues = check_referential_integrity(
        observations=observations,
        action_events=actions,
        weighing_events=weighings,
    )
    assert len(issues) == 1
    assert issues[0].code == "DANGLING_REFERENCE"
    assert issues[0].entity_id == "weigh_held"
    assert issues[0].gate == GateId.G2
    assert "obs_missing_final" in issues[0].message


def test_referential_integrity_empty_lists_valid() -> None:
    """Empty reference lists dangle nowhere; a resolved reading is valid."""
    observations = [_make_observation(evidence_ids=[])]
    actions = [_make_action(evidence_observation_ids=[])]
    weighings = [
        _make_weighing(
            action_ids=[],
            observation_ids=[],
            final_display_reading={
                "value": 120.0,
                "unit": "g",
                "observation_id": "obs_001",
            },
        )
    ]
    assert check_referential_integrity(
        observations=observations,
        action_events=actions,
        weighing_events=weighings,
    ) == []


def test_referential_integrity_multi_holder_attribution() -> None:
    """One issue per (referrer, target) pair, attributed to the referrer."""
    observations = [
        _make_observation(
            observation_id="obs_first",
            evidence_ids=["frm_missing"],
        ),
        _make_observation(
            observation_id="obs_second",
            evidence_ids=["frm_missing"],
        ),
    ]
    issues = check_referential_integrity(observations=observations)
    assert len(issues) == 2
    assert [issue.entity_id for issue in issues] == [
        "obs_first",
        "obs_second",
    ]
    for issue in issues:
        assert issue.code == "DANGLING_REFERENCE"
        assert "frm_missing" in issue.message


def test_referential_integrity_edge_order() -> None:
    """Edges report observations, actions, then weighing lists in the
    frozen order: action_ids, observation_ids, final_display_reading."""
    observations = [
        _make_observation(
            observation_id="obs_held",
            evidence_ids=["frm_x"],
        )
    ]
    actions = [
        _make_action(
            action_id="act_held",
            evidence_observation_ids=["obs_y"],
        )
    ]
    weighings = [
        _make_weighing(
            event_id="weigh_held",
            action_ids=["act_z"],
            observation_ids=["obs_w"],
            final_display_reading={
                "value": 120.0,
                "unit": "g",
                "observation_id": "obs_v",
            },
        )
    ]
    issues = check_referential_integrity(
        observations=observations,
        action_events=actions,
        weighing_events=weighings,
    )
    assert [issue.entity_id for issue in issues] == [
        "obs_held",
        "act_held",
        "weigh_held",
        "weigh_held",
        "weigh_held",
    ]
    assert [issue.gate for issue in issues] == [
        GateId.G1,
        GateId.G2,
        GateId.G2,
        GateId.G2,
        GateId.G2,
    ]
    assert "act_z" in issues[2].message
    assert "obs_w" in issues[3].message
    assert "obs_v" in issues[4].message


def test_referential_integrity_repeated_target_single_issue() -> None:
    """A repeated missing target yields one issue per (referrer, target)."""
    observations = [
        _make_observation(
            observation_id="obs_repeat",
            evidence_ids=["frm_x", "frm_x"],
        )
    ]
    issues = check_referential_integrity(observations=observations)
    assert len(issues) == 1
    assert issues[0].code == "DANGLING_REFERENCE"
    assert issues[0].entity_id == "obs_repeat"
    assert "frm_x" in issues[0].message


# DERIVATION_WITHOUT_UNIT and UNKNOWN_DERIVATION_WITH_VALUE.


@pytest.mark.parametrize("derivation", SUPPORTED_DERIVATIONS)
@pytest.mark.parametrize(
    ("value", "unit"),
    [(None, "g"), (120.0, None), (None, None)],
    ids=["missing_value", "missing_unit", "missing_both"],
)
def test_measurement_derivation_missing_value_or_unit(
    derivation: str, value: float | None, unit: str | None
) -> None:
    """Every supported derivation requires both a value and a unit."""
    weighings = [
        _make_weighing(
            event_id="weigh_partial",
            net_ingredient_quantity={
                "value": value,
                "unit": unit,
                "derivation": derivation,
            },
        )
    ]
    issues = check_measurement_derivation(weighings)
    assert len(issues) == 1
    assert issues[0].code == "DERIVATION_WITHOUT_UNIT"
    assert issues[0].entity_id == "weigh_partial"
    assert issues[0].gate == GateId.G2
    assert issues[0].severity == SEVERITY_ERROR
    assert issues[0].message


def test_measurement_derivation_blank_unit_passes() -> None:
    """Presence is non-None: a blank string unit is present, not missing."""
    weighings = [
        _make_weighing(
            net_ingredient_quantity={
                "value": 120.0,
                "unit": "",
                "derivation": "direct_after_tare",
            }
        )
    ]
    assert check_measurement_derivation(weighings) == []


def test_measurement_derivation_zero_value_valid() -> None:
    """0.0 is a value: presence is non-None, never falsiness."""
    weighings = [
        _make_weighing(
            net_ingredient_quantity={
                "value": 0.0,
                "unit": "g",
                "derivation": "direct_after_tare",
            }
        )
    ]
    assert check_measurement_derivation(weighings) == []


def test_measurement_derivation_unknown_nulls_valid() -> None:
    """Unknown derivation with null value and unit is the honest abstain."""
    weighings = [
        _make_weighing(
            net_ingredient_quantity={
                "value": None,
                "unit": None,
                "derivation": "unknown",
            }
        )
    ]
    assert check_measurement_derivation(weighings) == []


@pytest.mark.parametrize(
    ("value", "unit", "derivation", "expected"),
    [
        (None, None, Derivation.UNKNOWN, False),
        (0.0, None, Derivation.UNKNOWN, True),
        (None, "g", Derivation.UNKNOWN, True),
        (120.0, "g", Derivation.UNKNOWN, True),
        (0.0, "", Derivation.UNKNOWN, True),
        (None, None, Derivation.DIRECT_AFTER_TARE, False),
        (120.0, "g", Derivation.DIRECT_AFTER_TARE, False),
        (None, "g", Derivation.DIRECT_AFTER_TARE, False),
    ],
    ids=[
        "unknown_nulls_ok",
        "unknown_zero_value",
        "unknown_unit_only",
        "unknown_both",
        "unknown_blank_unit",
        "supported_nulls_not_unknown",
        "supported_complete_not_unknown",
        "supported_partial_not_unknown",
    ],
)
def test_unknown_derivation_violated_predicate(
    value: float | None,
    unit: str | None,
    derivation: Derivation,
    expected: bool,
) -> None:
    """Direct predicate tests for the defensive unknown-derivation branch.

    Unknown derivation with any non-None value or unit (including 0.0 and
    blank strings, which are present, not missing) violates the invariant.
    This branch is unreachable via parsed M1-01 models, which raise at
    parse time; it exists so the derivation check stays total even if a
    quantity ever bypasses model validation.
    """
    assert (
        _unknown_derivation_violated(value, unit, derivation) is expected
    )


@pytest.mark.parametrize(
    ("value", "unit"),
    [(120.0, None), (None, "g"), (120.0, "g"), (0.0, None)],
    ids=["value_only", "unit_only", "both", "zero_value"],
)
def test_model_rejects_unknown_derivation_with_value(
    value: float | None, unit: str | None
) -> None:
    """M1-01 model-owned: unknown derivation with value/unit raises at
    parse time, so end-to-end UNKNOWN_DERIVATION_WITH_VALUE is
    unreachable via parsed models and covered via the predicate above."""
    payload = _weighing_event_payload()
    payload["net_ingredient_quantity"] = {
        "value": value,
        "unit": unit,
        "derivation": "unknown",
    }
    with pytest.raises(ValidationError):
        WeighingEvent.model_validate(payload)


# Golden compatibility, determinism, and immutability.


def test_golden_p04_104_compatible() -> None:
    """Adapted P04_104 observations keep unique IDs and contained times."""
    assert GOLDEN.is_file(), f"missing golden snapshot: {GOLDEN}"
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    observations = [
        Observation.model_validate(item) for item in data["observations"]
    ]
    evidence = [
        FrameEvidence.model_validate(item)
        for item in data.get("frame_evidence", [])
    ]
    assert observations

    def _observed_time(item: Observation) -> float:
        if item.timestamp_seconds is not None:
            return item.timestamp_seconds
        assert item.interval is not None
        return item.interval.end_seconds

    max_ts = max(_observed_time(item) for item in observations)
    source = _make_video_source(
        duration_seconds=max_ts,
        interval={"start_seconds": 0.0, "end_seconds": max_ts},
    )
    assert check_unique_ids(
        frame_evidence=evidence, observations=observations
    ) == []
    assert check_timestamps(
        source, observations=observations, frame_evidence=evidence
    ) == []


def test_checks_deterministic_repeat() -> None:
    """Repeating every check on identical inputs yields identical issues."""
    source = _bounded_source()
    dup_a = _make_observation(observation_id="obs_dup")
    dup_b = _make_observation(observation_id="obs_dup")
    bad_action = _make_action(
        action_id="act_late", start_seconds=4.0, end_seconds=9.0
    )
    bad_obs = _make_observation(
        observation_id="obs_bad", timestamp_seconds=9.0
    )
    dangling = _make_observation(
        observation_id="obs_dangling", evidence_ids=["frm_missing"]
    )
    bad_weighing = _make_weighing(
        net_ingredient_quantity={
            "value": None,
            "unit": None,
            "derivation": "direct_after_tare",
        }
    )

    def _run_all() -> list[list[Any]]:
        return [
            check_unique_ids(observations=[dup_a, dup_b]),
            check_interval_bounds(source, action_events=[bad_action]),
            check_timestamps(source, observations=[bad_obs]),
            check_referential_integrity(observations=[dangling]),
            check_measurement_derivation([bad_weighing]),
        ]

    assert _run_all() == _run_all()


def test_checks_follow_input_order_not_sorted() -> None:
    """Issue order documents input-order dependence: first appearance in
    the inputs wins, even when alphabetical sorting would differ."""
    items = [
        _make_observation(observation_id="zeta_dup"),
        _make_observation(observation_id="alpha_dup"),
        _make_observation(observation_id="zeta_dup"),
        _make_observation(observation_id="alpha_dup"),
    ]
    issues = check_unique_ids(observations=items)
    assert [issue.entity_id for issue in issues] == [
        "zeta_dup",
        "alpha_dup",
    ]


def test_checks_do_not_mutate_inputs() -> None:
    """All checks are pure: inputs are unchanged after the calls."""
    source = _bounded_source()
    evidence = [_make_evidence(evidence_id="frm_dup")]
    observations = [
        _make_observation(observation_id="obs_dup"),
        _make_observation(observation_id="obs_dup"),
    ]
    actions = [
        _make_action(
            action_id="act_late", start_seconds=4.0, end_seconds=9.0
        )
    ]
    weighings = [
        _make_weighing(
            event_id="weigh_partial",
            net_ingredient_quantity={
                "value": None,
                "unit": "g",
                "derivation": "direct_after_tare",
            },
        )
    ]
    snapshot = [
        source.model_dump(),
        [item.model_dump() for item in evidence],
        [item.model_dump() for item in observations],
        [item.model_dump() for item in actions],
        [item.model_dump() for item in weighings],
    ]
    check_unique_ids(
        frame_evidence=evidence,
        observations=observations,
        action_events=actions,
        weighing_events=weighings,
    )
    check_interval_bounds(
        source, action_events=actions, weighing_events=weighings
    )
    check_timestamps(
        source, observations=observations, frame_evidence=evidence
    )
    check_referential_integrity(
        frame_evidence=evidence,
        observations=observations,
        action_events=actions,
        weighing_events=weighings,
    )
    check_measurement_derivation(weighings)
    assert source.model_dump() == snapshot[0]
    assert [item.model_dump() for item in evidence] == snapshot[1]
    assert [item.model_dump() for item in observations] == snapshot[2]
    assert [item.model_dump() for item in actions] == snapshot[3]
    assert [item.model_dump() for item in weighings] == snapshot[4]
