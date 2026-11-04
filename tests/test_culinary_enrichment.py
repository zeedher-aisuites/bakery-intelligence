"""Deterministic tests for culinary visual enrichment (no billable calls).

Fake provider clients record requests without network access. The
real Meta submit seam is never invoked: every test injects a fake.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bakery_intelligence.adversarial.culinary import (
    CULINARY_ACTION_TYPES,
    CULINARY_JSON_SCHEMA,
    CULINARY_SCHEMA_NAME,
    CulinaryError,
    CulinaryStore,
    build_culinary_events,
    describe_for_prompt,
    enrich_next_batch,
    validate_culinary_observation,
)
from bakery_intelligence.adversarial.observations import (
    build_review_tasks,
)
from bakery_intelligence.api.jobs import JobRecord
from bakery_intelligence.api.storage import LocalStorageBackend

RUN_ID = "run_c0ffee123456"


def _valid_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ingredient_candidates": [{
            "name": "flour",
            "confidence": "medium",
            "packaging_visible": False,
            "label_text": None,
            "source": "observed",
        }],
        "containers": [{
            "kind": "bowl", "description": "glass bowl",
            "source": "observed",
        }],
        "utensils": [],
        "actions": [{
            "action_type": "pour",
            "participants": ["flour"],
            "targets": ["bowl"],
            "confidence": "medium",
            "source": "observed",
        }],
        "scale": {
            "display_verbatim": "120",
            "readable": True,
            "state": "stable",
            "tare_action_visible": False,
            "unit": None,
            "source": "observed",
        },
        "temporal": {
            "continues_previous": None,
            "transition": None,
            "source": "unknown",
        },
        "uncertainty_notes": None,
    }
    payload.update(overrides)
    return payload


def _observation(frame_index: int, **overrides: Any) -> dict[str, Any]:
    observation = validate_culinary_observation(
        _valid_payload(), frame_index)
    observation.update(overrides)
    observation["timestamp_seconds"] = float(frame_index)
    return observation


class _FakeSubmit:
    """Fake Meta submit seam returning canned InferenceResponses."""

    def __init__(self, texts: list[str], fail_on: int = -1) -> None:
        self.texts = texts
        self.fail_on = fail_on
        self.calls: list[dict[str, Any]] = []

    def __call__(self, prompt_text: str, image_paths: Any,
                 api_key: str, max_output_tokens: int,
                 schema: Any = None,
                 schema_name: Any = None) -> Any:
        index = len(self.calls)
        self.calls.append({
            "prompt_text": prompt_text,
            "image_paths": list(image_paths),
            "api_key": api_key,
            "max_output_tokens": max_output_tokens,
            "schema": schema,
            "schema_name": schema_name,
        })
        if index == self.fail_on:
            raise RuntimeError("provider down")
        return SimpleNamespace(
            output_text=self.texts[index],
            model="muse-spark-1.3",
            request_id=f"resp-{index}",
            status="completed",
            incomplete_reason=None,
            usage=SimpleNamespace(model_dump=lambda: {
                "input_tokens": 3000, "output_tokens": 400,
                "total_tokens": 3400}),
        )


def _frames(indices: list[int]) -> tuple[
        list[dict[str, Any]], list[str]]:
    frames = [{"frame_index": index,
               "timestamp_seconds": float(index)} for index in indices]
    return frames, [f"frame_{index:08d}.jpg" for index in indices]


def test_valid_observation_parses() -> None:
    observation = validate_culinary_observation(_valid_payload(), 7)
    assert observation["actions"][0]["action_type"] == "pour"
    assert observation["ingredient_candidates"][0]["name"] == "flour"
    assert observation["frame_index"] == 7


def test_unknown_action_normalizes_instead_of_failing() -> None:
    payload = _valid_payload()
    payload["actions"][0]["action_type"] = "julienne"
    observation = validate_culinary_observation(payload, 3)
    assert observation["actions"][0]["action_type"] == "unknown_action"


def test_missing_section_is_rejected() -> None:
    payload = _valid_payload()
    del payload["actions"]
    with pytest.raises(CulinaryError, match="INVALID_OBSERVATION"):
        validate_culinary_observation(payload, 1)


def test_unknown_source_is_rejected() -> None:
    # Structural errors still raise: source must live on an object.
    payload = _valid_payload()
    payload["actions"] = ["not-an-object"]
    with pytest.raises(CulinaryError, match="INVALID_OBSERVATION"):
        validate_culinary_observation(payload, 1)
    missing = _valid_payload()
    del missing["scale"]
    with pytest.raises(CulinaryError, match="missing 'scale'"):
        validate_culinary_observation(missing, 1)


def test_spatial_sources_normalize_to_unknown_with_raw() -> None:
    # Exact batch-2 failure shapes: location words in source.
    for frame_index, raw in ((1350, "frame"), (1400, "image"),
                             (1450, "frame_center_right")):
        payload = _valid_payload()
        payload["ingredient_candidates"][0]["source"] = raw
        observation = validate_culinary_observation(payload, frame_index)
        candidate = observation["ingredient_candidates"][0]
        assert candidate["source"] == "unknown"
        assert candidate["source_raw"] == raw
        assert candidate["confidence"] == "medium"
        assert candidate["confidence_raw"] == "medium"


def test_source_never_becomes_observed_from_spatial_hints() -> None:
    payload = _valid_payload()
    payload["containers"][0]["source"] = "frame_center_right"
    payload["scale"]["source"] = "image"
    payload["temporal"]["source"] = "frame"
    observation = validate_culinary_observation(payload, 3)
    assert observation["containers"][0]["source"] == "unknown"
    assert observation["containers"][0]["source_raw"] == (
        "frame_center_right")
    assert observation["scale"]["source"] == "unknown"
    assert observation["scale"]["source_raw"] == "image"
    assert observation["temporal"]["source"] == "unknown"
    assert observation["temporal"]["source_raw"] == "frame"


def test_pouring_alias_maps_to_canonical_pour() -> None:
    payload = _valid_payload()
    payload["actions"][0]["action_type"] = "pouring"
    observation = validate_culinary_observation(payload, 1300)
    (action,) = observation["actions"]
    assert action["action_type"] == "pour"
    assert action["action_type_raw"] == "pouring"


def test_alias_matching_ignores_case_and_whitespace() -> None:
    payload = _valid_payload()
    payload["actions"][0]["action_type"] = " Pouring "
    observation = validate_culinary_observation(payload, 4)
    assert observation["actions"][0]["action_type"] == "pour"
    assert observation["actions"][0]["action_type_raw"] == " Pouring "


def test_genuinely_unknown_actions_stay_unknown() -> None:
    for raw in ("julienne", "plating", None, 42):
        payload = _valid_payload()
        payload["actions"][0]["action_type"] = raw
        observation = validate_culinary_observation(payload, 4)
        assert observation["actions"][0]["action_type"] == (
            "unknown_action")
        assert observation["actions"][0]["action_type_raw"] is raw


def test_action_type_raw_is_preserved_on_normalization() -> None:
    payload = _valid_payload()
    payload["actions"][0]["action_type"] = "sprinkle"
    observation = validate_culinary_observation(payload, 4)
    (action,) = observation["actions"]
    assert action["action_type"] == "unknown_action"
    assert action["action_type_raw"] == "sprinkle"
    payload = _valid_payload()
    observation = validate_culinary_observation(payload, 4)
    assert observation["actions"][0]["action_type"] == "pour"
    assert observation["actions"][0]["action_type_raw"] == "pour"


def test_non_transition_values_do_not_flag_transitions() -> None:
    for value in (None, "not_evaluated", "unknown", "not_visible",
                  "none", "no_transition", "", "  Unknown  "):
        observation = _observation(6)
        observation["temporal"]["transition"] = value
        events = build_culinary_events(RUN_ID, [observation], {6: 10})
        assert "TRANSITION_REPORTED" not in events[0][
            "uncertainty"], value


def test_described_transition_still_flags() -> None:
    observation = _observation(6)
    observation["temporal"]["transition"] = "pouring started"
    events = build_culinary_events(RUN_ID, [observation], {6: 10})
    assert "TRANSITION_REPORTED" in events[0]["uncertainty"]


def test_legacy_observations_without_raw_fields_still_group() -> None:
    legacy = {
        "frame_index": 0,
        "ingredient_candidates": [{
            "name": "flour", "confidence": "high",
            "packaging_visible": False, "label_text": None,
            "source": "observed"}],
        "containers": [], "utensils": [],
        "actions": [{
            "action_type": "pour", "participants": [],
            "targets": [], "confidence": "high",
            "source": "observed"}],
        "scale": {"display_verbatim": "20", "readable": True,
                  "state": None, "tare_action_visible": False,
                  "unit": None, "source": "observed"},
        "temporal": {"continues_previous": None, "transition": None,
                     "source": "unknown"},
        "uncertainty_notes": None,
        "timestamp_seconds": 0.0,
    }
    events = build_culinary_events(RUN_ID, [legacy], {0: 20})
    assert len(events) == 1
    assert events[0]["operation"] == "pour"
    assert events[0]["uncertainty"] == []
    # A supported weighing without an observed unit still asks
    # about the unit; the legacy record itself needed no changes.
    queue = build_review_tasks(
        {"frames": []}, {}, None,
        _culinary_package(events, [legacy]))
    assert [task["state"] for task in queue["tasks"]] == ["unit_missing"]


def test_misplaced_observed_confidence_normalizes_to_unknown() -> None:
    # Exact batch-1 failure shape: confidence held a source value.
    payload = _valid_payload()
    payload["ingredient_candidates"][0]["confidence"] = "observed"
    observation = validate_culinary_observation(payload, 50)
    candidate = observation["ingredient_candidates"][0]
    assert candidate["confidence"] is None
    assert candidate["confidence_raw"] == "observed"
    assert candidate["source"] == "observed"


def test_unknown_confidence_normalizes_without_upgrading() -> None:
    payload = _valid_payload()
    payload["ingredient_candidates"][0]["confidence"] = "unknown"
    payload["actions"][0]["confidence"] = "unknown"
    observation = validate_culinary_observation(payload, 150)
    assert observation["ingredient_candidates"][0]["confidence"] is None
    assert (observation["ingredient_candidates"][0]["confidence_raw"]
            == "unknown")
    assert observation["actions"][0]["confidence"] is None
    assert observation["actions"][0]["confidence_raw"] == "unknown"
    assert observation["actions"][0]["source"] == "observed"


def test_valid_confidences_pass_through_with_raw() -> None:
    for value in ("high", "medium", "low"):
        payload = _valid_payload()
        payload["ingredient_candidates"][0]["confidence"] = value
        observation = validate_culinary_observation(payload, 1)
        candidate = observation["ingredient_candidates"][0]
        assert candidate["confidence"] == value
        assert candidate["confidence_raw"] == value


def test_normalized_unknowns_still_flag_ambiguity() -> None:
    payload = _valid_payload()
    payload["ingredient_candidates"][0]["name"] = "flour"
    payload["ingredient_candidates"][0]["confidence"] = "observed"
    observation = validate_culinary_observation(payload, 2)
    events = build_culinary_events(RUN_ID, [observation], {2: 100})
    assert events[0]["uncertainty"] == ["INGREDIENT_AMBIGUOUS"]
    queue = build_review_tasks(
        {"frames": []}, {}, None,
        _culinary_package(events, [observation]))
    assert queue["tasks"][0]["state"] == "ingredient_uncertain"


def test_multiple_actions_are_rejected() -> None:
    payload = _valid_payload()
    payload["actions"].append(dict(payload["actions"][0]))
    with pytest.raises(CulinaryError, match="at most one"):
        validate_culinary_observation(payload, 1)


def test_historical_gaps_stay_not_evaluated() -> None:
    observation = validate_culinary_observation(_valid_payload(), 9)
    assert observation["temporal"]["source"] == "unknown"
    assert observation["scale"]["unit"] is None
    assert observation["ingredient_candidates"][0]["source"] == "observed"


def _at(index: int, seconds: float, **overrides: Any) -> dict[str, Any]:
    observation = _observation(index, **overrides)
    observation["timestamp_seconds"] = seconds
    return observation


def test_consecutive_compatible_frames_form_one_event() -> None:
    observations = [_at(0, 0.0), _at(50, 1.0), _at(100, 2.0)]
    events = build_culinary_events(
        RUN_ID, observations, {0: 120, 50: 125, 100: 130})
    assert len(events) == 1
    event = events[0]
    assert event["frame_refs"] == [0, 50, 100]
    assert event["operation"] == "pour"
    assert event["event_id"].startswith("culinary_")
    assert event["start_seconds"] == 0.0
    assert event["end_seconds"] == 2.0
    assert event["net_quantity"]["value"] is None
    assert "unit_not_observed_in_culinary_schema" in (
        event["net_quantity"]["unmet_conditions"])
    # Deterministic: same input, same event id.
    again = build_culinary_events(
        RUN_ID, list(reversed(observations)),
        {0: 120, 50: 125, 100: 130})
    assert again[0]["event_id"] == event["event_id"]


def test_temporal_gap_beyond_two_seconds_starts_a_new_event() -> None:
    first = _at(0, 0.0)
    second = _at(50, 3.0)
    events = build_culinary_events(RUN_ID, [first, second], {})
    assert len(events) == 2
    assert events[0]["frame_refs"] == [0]
    assert events[1]["frame_refs"] == [50]


def test_missing_timestamps_never_invent_continuity() -> None:
    first = _at(0, 0.0)
    second = _at(50, 1.0)
    del second["timestamp_seconds"]
    events = build_culinary_events(RUN_ID, [first, second], {})
    assert len(events) == 2


def test_ingredient_change_starts_a_new_event() -> None:
    first = _at(0, 0.0)
    second = _at(50, 1.0)
    second["ingredient_candidates"] = [{
        "name": "sugar", "confidence": "high",
        "packaging_visible": False, "label_text": None,
        "source": "observed"}]
    events = build_culinary_events(RUN_ID, [first, second], {})
    assert len(events) == 2
    assert events[0]["operation"] == events[1]["operation"] == "pour"


def test_packaging_variants_group_without_merging_identities() -> None:
    first = _at(0, 0.0)
    second = _at(50, 1.0)
    second["ingredient_candidates"] = [{
        "name": "PENNE RIGATI", "confidence": "high",
        "packaging_visible": True, "label_text": "PENNE RIGATI 41",
        "source": "observed"}]
    first["ingredient_candidates"] = [{
        "name": "penne rigate", "confidence": "high",
        "packaging_visible": True, "label_text": "PENNE RIGATE 41",
        "source": "observed"}]
    events = build_culinary_events(RUN_ID, [first, second], {})
    assert len(events) == 1
    (merged,) = events[0]["ingredient_candidates"]
    assert merged["confidence"] == "high"
    assert sorted(merged["variants"]) == ["PENNE RIGATI", "penne rigate"]
    third = _at(100, 2.0)
    third["ingredient_candidates"] = [{
        "name": "salt", "confidence": "high",
        "packaging_visible": False, "label_text": None,
        "source": "observed"}]
    split = build_culinary_events(
        RUN_ID, [first, second, third], {})
    assert len(split) == 2


def test_unknown_actions_never_merge_into_known_ones() -> None:
    first = _at(0, 0.0)
    second = _at(50, 1.0)
    second["actions"] = [{
        "action_type": "unknown_action", "participants": [],
        "targets": [], "confidence": "high", "source": "observed"}]
    events = build_culinary_events(RUN_ID, [first, second], {})
    assert len(events) == 2
    assert events[1]["operation"] == "unknown_action"
    twin = _at(100, 2.0)
    twin["actions"] = [{
        "action_type": "unknown_action", "participants": [],
        "targets": [], "confidence": "high", "source": "observed"}]
    together = build_culinary_events(RUN_ID, [second, twin], {})
    assert len(together) == 1


def test_action_change_and_large_gaps_split_events() -> None:
    first = _at(0, 0.0)
    second = _at(50, 1.0)
    second["actions"] = [{
        "action_type": "stir", "participants": [], "targets": [],
        "confidence": "high", "source": "observed"}]
    third = _at(200, 40.0)
    events = build_culinary_events(RUN_ID, [first, second, third], {})
    assert len(events) == 3


def test_tare_frames_are_flagged_without_quantities() -> None:
    observation = _observation(2)
    observation["scale"]["tare_action_visible"] = True
    events = build_culinary_events(RUN_ID, [observation], {2: 50})
    assert events[0]["tare_visible"] is True
    assert "TARE_UNRESOLVED" in events[0]["uncertainty"]
    assert events[0]["net_quantity"]["value"] is None


def test_enrichment_batches_and_resumes(tmp_path: Path) -> None:
    store = CulinaryStore(tmp_path / "culinary-test-resume")
    frames, paths = _frames([0, 1, 2, 3])
    texts = [json.dumps(_valid_payload()) for _ in frames]
    submit = _FakeSubmit(texts)
    first = enrich_next_batch(
        run_id=RUN_ID, frames=frames, image_paths=paths,
        submit=submit, store=store, max_frames=2)
    assert first == {
        "run_id": RUN_ID, "model": "muse-spark-1.3",
        "enriched": 2, "failed": 0, "completed": 2, "total": 4,
        "done": False,
    }
    second = enrich_next_batch(
        run_id=RUN_ID, frames=frames, image_paths=paths,
        submit=submit, store=store, max_frames=5)
    assert second["enriched"] == 2
    assert second["done"] is True
    assert len(submit.calls) == 4
    seen = [call["image_paths"][0] for call in submit.calls]
    assert seen == [f"frame_{i:08d}.jpg" for i in [0, 1, 2, 3]]
    stored = store.load(RUN_ID)
    assert stored["observations"]["0"]["request_id"] == "resp-0"
    assert stored["observations"]["0"]["model"] == "muse-spark-1.3"
    assert stored["usage"][0]["input_tokens"] == 3000


def test_failed_frames_are_terminal_and_never_retried(tmp_path: Path) -> None:
    store = CulinaryStore(tmp_path / "culinary-test-failure")
    frames, paths = _frames([0, 1, 2])
    texts = [json.dumps(_valid_payload()) for _ in frames]
    submit = _FakeSubmit(texts, fail_on=1)
    result = enrich_next_batch(
        run_id=RUN_ID, frames=frames, image_paths=paths,
        submit=submit, store=store, max_frames=5)
    assert result["enriched"] == 2
    assert result["failed"] == 1
    assert result["done"] is True
    resume = enrich_next_batch(
        run_id=RUN_ID, frames=frames, image_paths=paths,
        submit=submit, store=store, max_frames=5)
    assert resume["enriched"] == 0
    assert resume["failed"] == 0
    assert len(submit.calls) == 3
    stored = store.load(RUN_ID)
    assert "1" in stored["failures"]
    assert "1" not in stored["observations"]


def test_malformed_model_output_is_recorded_not_raised(tmp_path: Path) -> None:
    store = CulinaryStore(tmp_path / "culinary-test-malformed")
    frames, paths = _frames([0])
    submit = _FakeSubmit(["{not json"])
    result = enrich_next_batch(
        run_id=RUN_ID, frames=frames, image_paths=paths,
        submit=submit, store=store)
    assert result["failed"] == 1
    assert store.load(RUN_ID)["observations"] == {}


def test_structured_schema_travels_with_each_request(tmp_path: Path) -> None:
    store = CulinaryStore(tmp_path / "culinary-test-schema")
    frames, paths = _frames([0])
    submit = _FakeSubmit([json.dumps(_valid_payload())])
    enrich_next_batch(
        run_id=RUN_ID, frames=frames, image_paths=paths,
        submit=submit, store=store)
    (call,) = submit.calls
    assert call["schema"] == CULINARY_JSON_SCHEMA
    assert call["schema_name"] == CULINARY_SCHEMA_NAME
    assert "visually observable" in call["prompt_text"]
    assert "never invent" in call["prompt_text"]


def _culinary_package(events: list[dict[str, Any]],
                      observations: list[dict[str, Any]]) -> dict[str, Any]:
    return {"events": events, "observations": observations}


def test_no_enrichment_means_no_culinary_tasks() -> None:
    queue = build_review_tasks({"frames": []})
    assert queue["tasks"] == []
    assert queue["counts"]["blocking_errors"] == []


def test_uncertain_ingredient_surfaces_one_event_task() -> None:
    observations = [_at(0, 0.0), _at(50, 1.0)]
    events = build_culinary_events(RUN_ID, observations, {})
    assert len(events) == 1
    queue = build_review_tasks(
        {"frames": []}, {}, None,
        _culinary_package(events, observations))
    assert len(queue["tasks"]) == 1
    task = queue["tasks"][0]
    assert task["state"] == "ingredient_uncertain"
    assert task["frame_refs"] == [0, 50]
    assert task["status"] == "open"
    assert queue["counts"]["blocking_errors"] == []


def test_conflicting_actions_surface_transition_task() -> None:
    first = _at(0, 0.0)
    second = _at(50, 1.0)
    second["actions"] = [{
        "action_type": "stir", "participants": [], "targets": [],
        "confidence": "high", "source": "observed"}]
    events = build_culinary_events(RUN_ID, [first, second], {})
    queue = build_review_tasks(
        {"frames": []}, {}, None,
        _culinary_package(events, [first, second]))
    states = {task["state"] for task in queue["tasks"]}
    assert "action_conflicting" in states


def test_missing_unit_and_tare_tasks_only_when_material() -> None:
    pouring = _observation(0)
    events = build_culinary_events(RUN_ID, [pouring], {0: 120})
    queue = build_review_tasks(
        {"frames": []}, {}, None,
        _culinary_package(events, [pouring]))
    assert len(queue["tasks"]) == 1
    issues = queue["tasks"][0]["issues"]
    assert "unit_missing" in issues
    assert "tare_unresolved" not in issues
    pouring["scale"]["tare_action_visible"] = True
    events = build_culinary_events(RUN_ID, [pouring], {0: 120})
    queue = build_review_tasks(
        {"frames": []}, {}, None,
        _culinary_package(events, [pouring]))
    assert "tare_unresolved" in queue["tasks"][0]["issues"]


def test_culinary_corrections_round_trip_without_touching_scale(
        tmp_path: Path) -> None:
    from bakery_intelligence.adversarial.observations import (
        ObservationReviewStore,
    )
    observations = [_observation(0)]
    events = build_culinary_events(RUN_ID, observations, {})
    culinary = _culinary_package(events, observations)
    package = {"frames": []}
    store = ObservationReviewStore(tmp_path)
    queue = build_review_tasks(package, {}, None, culinary)
    task_id = queue["tasks"][0]["task_id"]
    store.save_correction(
        RUN_ID, task_id, package, "not_identifiable", None,
        "chef-test", culinary=culinary)
    resolved = build_review_tasks(
        package, store.load(RUN_ID)["corrections"], None, culinary)
    assert resolved["tasks"][0]["status"] == "resolved"
    assert resolved["counts"]["open"] == 0
    store.save_correction(
        RUN_ID, task_id, package, "review_later", None,
        "chef-test", culinary=culinary)
    deferred = build_review_tasks(
        package, store.load(RUN_ID)["corrections"], None, culinary)
    assert deferred["tasks"][0]["status"] == "deferred"
    assert deferred["counts"]["blocking_errors"] == []
    undone = store.undo_correction(
        RUN_ID, task_id, package, "chef-test", culinary=culinary)
    assert undone["correction"]["action"] == "not_identifiable"
    restored = build_review_tasks(
        package, store.load(RUN_ID)["corrections"], None, culinary)
    assert restored["tasks"][0]["status"] == "resolved"
    history = store.load(RUN_ID)["correction_history"][task_id]
    assert [entry.get("action") for entry in history] == [
        "not_identifiable", "review_later", "undo"]


def test_evidence_package_carries_culinary_sections(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bakery_intelligence.adversarial.service import (
        RecipeReviewService,
    )
    from bakery_intelligence.adversarial.store import (
        RecipeReviewError,
    )
    monkeypatch.setenv("BAKERY_LIVE_RECIPE_GEN", "1")
    monkeypatch.setenv("MODEL_API_KEY", "dummy-meta")
    storage = LocalStorageBackend(tmp_path / "storage")
    record = JobRecord(run_id=RUN_ID, workflow="video_pipeline_meta")
    service = RecipeReviewService(storage)
    merged = service.consolidated(record)
    assert merged["culinary_observations"] == []
    assert merged["culinary_events"] == []
    assert merged["culinary_unknowns"] == []
    assert isinstance(merged["culinary_hash"], str)
    assert set(merged["content_hash"]) <= set("0123456789abcdef")
    # Existing scale evidence keeps its historical shape.
    assert merged["frames"] == []
    # Enrichment requires explicit confirmation, even with a key.
    with pytest.raises(RecipeReviewError, match="confirmation"):
        service.enrich_culinary(
            record, max_frames=5, confirmed=False,
            submit=_FakeSubmit([]))
    # No retained images exist here, so nothing is submitted.
    progress = service.enrich_culinary(
        record, max_frames=5, confirmed=True,
        submit=_FakeSubmit([]))
    assert progress["total"] == 0
    assert progress["done"] is True
    status = service.culinary_status(record)
    assert status["total"] == 0
    assert status["done"] is True
    assert status["events"] == []


def test_default_per_frame_budget_is_16384(tmp_path: Path) -> None:
    import inspect as _inspect

    from bakery_intelligence.adversarial.service import (
        RecipeReviewService,
    )
    frames, paths = _frames([0])
    submit = _FakeSubmit([json.dumps(_valid_payload())])
    store = CulinaryStore(tmp_path / "culinary-test-budget")
    enrich_next_batch(
        run_id=RUN_ID, frames=frames, image_paths=paths,
        submit=submit, store=store)
    (call,) = submit.calls
    assert call["max_output_tokens"] == 16384
    assert _inspect.signature(
        RecipeReviewService.enrich_culinary).parameters[
            "max_output_tokens"].default == 16384


def test_missing_usage_is_recorded_as_unreported(
        tmp_path: Path) -> None:
    import types

    store = CulinaryStore(tmp_path / "culinary-test-no-usage")
    frames, paths = _frames([0])

    def submit(prompt_text: str, image_paths: Any, api_key: str,
               max_output_tokens: int, schema: Any = None,
               schema_name: Any = None) -> Any:
        return types.SimpleNamespace(
            output_text=json.dumps(_valid_payload()),
            model="muse-spark-1.3",
            request_id="resp-nousage",
            status="completed",
            usage=None)

    result = enrich_next_batch(
        run_id=RUN_ID, frames=frames, image_paths=paths,
        submit=submit, store=store)
    assert result["enriched"] == 1
    (record,) = store.load(RUN_ID)["usage"]
    assert record["usage_status"] == "unreported"
    assert record["input_tokens"] is None
    assert record["request_id"] == "resp-nousage"
    assert record["max_output_tokens"] == 16384


def test_reported_usage_is_recorded_as_reported(tmp_path: Path) -> None:
    store = CulinaryStore(tmp_path / "culinary-test-usage")
    frames, paths = _frames([0])
    submit = _FakeSubmit([json.dumps(_valid_payload())])
    enrich_next_batch(
        run_id=RUN_ID, frames=frames, image_paths=paths,
        submit=submit, store=store)
    (record,) = store.load(RUN_ID)["usage"]
    assert record["usage_status"] == "reported"
    assert record["input_tokens"] == 3000


def test_requeue_releases_failed_frames_with_history(
        tmp_path: Path) -> None:
    store = CulinaryStore(tmp_path / "culinary-test-requeue")
    store.record_failure(RUN_ID, 50, "INVALID_OBSERVATION: bad")
    store.record_failure(RUN_ID, 150, "INVALID_OBSERVATION: bad")
    store.record_observation(RUN_ID, 0, _observation(0))
    before = store.load(RUN_ID)
    assert store.completed_indices(RUN_ID) == {0, 50, 150}
    result = store.request_reprocessing(
        RUN_ID, [50, 150],
        reason="validator now normalizes confidence",
        reviewer="chef-test")
    assert result["requeued"] == [50, 150]
    assert store.completed_indices(RUN_ID) == {0}
    after = store.load(RUN_ID)
    assert after["failures"] == {}
    assert len(after["failures_history"]) == 2
    assert after["failures_history"][0]["failure"] == (
        before["failures"]["50"])
    assert after["requeue_requests"][0]["reason"] == (
        "validator now normalizes confidence")
    assert after["requeue_requests"][0]["reviewer"] == "chef-test"
    # The untouched observation survives; checkpoints append only.
    assert "0" in after["observations"]
    lines = (tmp_path / "culinary-test-requeue"
             / "culinary_observations"
             / f"{RUN_ID}.jsonl").read_text(encoding="utf-8")
    assert '"outcome": "requeued"' in lines
    assert '"outcome": "failed"' in lines


def test_requeue_rejects_observed_unknown_and_blank(
        tmp_path: Path) -> None:
    store = CulinaryStore(tmp_path / "culinary-test-requeue-guards")
    store.record_failure(RUN_ID, 50, "INVALID_OBSERVATION: bad")
    store.record_observation(RUN_ID, 0, _observation(0))
    with pytest.raises(CulinaryError, match="already has an observation"):
        store.request_reprocessing(
            RUN_ID, [0], reason="x", reviewer="chef-test")
    with pytest.raises(CulinaryError, match="no terminal failure"):
        store.request_reprocessing(
            RUN_ID, [51], reason="x", reviewer="chef-test")
    with pytest.raises(CulinaryError, match="reason is required"):
        store.request_reprocessing(
            RUN_ID, [50], reason="  ", reviewer="chef-test")
    with pytest.raises(CulinaryError, match="reviewer identity"):
        store.request_reprocessing(
            RUN_ID, [50], reason="x", reviewer="")
    with pytest.raises(CulinaryError, match="no frame indices"):
        store.request_reprocessing(
            RUN_ID, [], reason="x", reviewer="chef-test")
    assert set(store.completed_indices(RUN_ID)) == {0, 50}


def test_requeued_frame_can_be_observed_again(
        tmp_path: Path) -> None:
    store = CulinaryStore(tmp_path / "culinary-test-reobserve")
    store.record_failure(RUN_ID, 50, "INVALID_OBSERVATION: bad")
    store.request_reprocessing(
        RUN_ID, [50], reason="validator fixed", reviewer="chef-test")
    frames, paths = _frames([50])
    submit = _FakeSubmit([json.dumps(_valid_payload())])
    result = enrich_next_batch(
        run_id=RUN_ID, frames=frames, image_paths=paths,
        submit=submit, store=store)
    assert result["enriched"] == 1
    assert len(submit.calls) == 1
    stored = store.load(RUN_ID)
    assert "50" in stored["observations"]
    assert stored["failures"] == {}
    assert len(stored["failures_history"]) == 1


def test_requeue_preserves_human_corrections(tmp_path: Path) -> None:
    from bakery_intelligence.adversarial.observations import (
        ObservationReviewStore,
        build_review_tasks,
    )
    culinary_store = CulinaryStore(tmp_path / "culinary-test-isolation")
    culinary_store.record_failure(RUN_ID, 50, "INVALID_OBSERVATION: bad")
    package = {"frames": [{
        "frame_index": 0, "timestamp_seconds": 0.0,
        "measurement": {"value": None}, "scale": {"readable": True}}],
        "action_evidence": {"addition_frames": [0]}}
    review_store = ObservationReviewStore(tmp_path)
    queue = build_review_tasks(package)
    review_store.save_correction(
        RUN_ID, queue["tasks"][0]["task_id"], package,
        "review_later", None, "chef-test")
    culinary_store.request_reprocessing(
        RUN_ID, [50], reason="validator fixed", reviewer="chef-test")
    stored = review_store.load(RUN_ID)
    assert queue["tasks"][0]["task_id"] in stored["corrections"]
    assert stored["corrections"][
        queue["tasks"][0]["task_id"]]["action"] == "review_later"


def _endpoint_client(tmp_path: Path):
    import io

    from fastapi.testclient import TestClient

    from bakery_intelligence.api.app import build_app
    from bakery_intelligence.api.jobs import (
        WORKFLOW_STORAGE_SMOKE,
        JobService,
    )
    from bakery_intelligence.api.store import SqliteJobStore

    storage = LocalStorageBackend(tmp_path / "storage")
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    service = JobService(store)
    record = service.submit(
        WORKFLOW_STORAGE_SMOKE, source_object_id="clip.mp4")
    storage.store_source("clip.mp4", io.BytesIO(b"fake"))
    store.claim_next("worker-1", "2026-11-01T00:00:00+00:00")
    store.complete(record.run_id, "worker-1",
                   "2026-11-01T00:01:00+00:00", "a" * 64, {})
    client = TestClient(build_app(
        storage=storage, db_path=tmp_path / "jobs.sqlite3"))
    return client, record.run_id


def test_requeue_endpoint_records_authorization(
        tmp_path: Path) -> None:
    from bakery_intelligence.adversarial.culinary import CulinaryStore

    client, run_id = _endpoint_client(tmp_path)
    storage = LocalStorageBackend(tmp_path / "storage")
    CulinaryStore(storage.base_dir).record_failure(
        run_id, 50, "INVALID_OBSERVATION: bad")
    response = client.post(
        f"/api/analyses/{run_id}/culinary/requeue",
        json={"frame_indices": [50],
              "reason": "validator now normalizes confidence",
              "reviewer": "chef-test"})
    assert response.status_code == 200
    assert response.json()["requeued"] == [50]
    stored = CulinaryStore(storage.base_dir).load(run_id)
    assert stored["failures"] == {}
    assert len(stored["failures_history"]) == 1
    repeat = client.post(
        f"/api/analyses/{run_id}/culinary/requeue",
        json={"frame_indices": [50],
              "reason": "again", "reviewer": "chef-test"})
    assert repeat.status_code == 422
    blank = client.post(
        f"/api/analyses/{run_id}/culinary/requeue",
        json={"frame_indices": [50],
              "reason": "  ", "reviewer": "chef-test"})
    assert blank.status_code == 422


def test_enrich_endpoint_selects_explicit_frames(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BAKERY_LIVE_RECIPE_GEN", "1")
    monkeypatch.setenv("MODEL_API_KEY", "dummy-meta")
    client, run_id = _endpoint_client(tmp_path)
    unconfirmed = client.post(
        f"/api/analyses/{run_id}/culinary/enrich",
        json={"max_frames": 5, "confirmed": False})
    assert unconfirmed.status_code == 422
    invalid = client.post(
        f"/api/analyses/{run_id}/culinary/enrich",
        json={"max_frames": 5, "confirmed": True,
              "frame_indices": [-1]})
    assert invalid.status_code == 422
    # No retained images exist here, so nothing billable happens.
    selected = client.post(
        f"/api/analyses/{run_id}/culinary/enrich",
        json={"max_frames": 5, "confirmed": True,
              "frame_indices": [0, 1, 2]})
    assert selected.status_code == 200
    assert selected.json()["total"] == 0
    assert selected.json()["done"] is True


def test_human_corrections_apply_only_within_exact_scope() -> None:
    # Mirrors frames 1900/1950: the chef's 287 stands exactly where
    # authorized; the raw 281/207 and neighbors are untouched, and
    # the culinary verbatim lives in its own record, never merged.
    from bakery_intelligence.adversarial.consolidation import (
        consolidate,
    )
    package = {
        "run_id": RUN_ID,
        "frames": [
            {"frame_index": 1900,
             "measurement": {"value": 281.0}},
            {"frame_index": 1950,
             "measurement": {"value": 207.0}},
            {"frame_index": 2000,
             "measurement": {"value": 292.0}},
        ],
    }
    corrections = {
        "task_pair": {
            "action": "correct",
            "human_value": 287.0,
            "frame_refs": [1900, 1950],
            "revision": 2,
            "reviewer": "chef-test",
        }
    }
    merged = consolidate(package, corrections)
    by_index = {frame["frame_index"]: frame
                for frame in merged["frames"]}
    assert by_index[1900]["effective"]["value"] == 287.0
    assert by_index[1900]["effective"]["original_value"] == 281.0
    assert by_index[1950]["effective"]["original_value"] == 207.0
    assert "effective" not in by_index[2000]
    assert by_index[2000]["measurement"]["value"] == 292.0


def test_describe_for_prompt_never_invents_quantities() -> None:
    line = describe_for_prompt(_observation(3))
    assert "frame 3" in line
    assert "pour" in line
    assert "flour" in line
    assert "120" not in line
    assert "g" not in line.split("ingredients")[0]


def test_culinary_action_vocabulary_is_closed() -> None:
    assert "pour" in CULINARY_ACTION_TYPES
    assert "tare" in CULINARY_ACTION_TYPES
    assert "weigh" in CULINARY_ACTION_TYPES
