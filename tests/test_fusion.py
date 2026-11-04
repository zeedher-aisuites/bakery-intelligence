"""Deterministic fusion tests (synthetic fixtures only).

No test reads experimental evidence files: adapters take plain
dicts, so every case below is a small hand-built fixture.
"""

import pytest

from bakery_intelligence.fusion import (
    normalize,
    quantities,
    reconcile,
    reconstruct,
)
from bakery_intelligence.fusion.review_adapter import (
    to_store_ingredients,
    to_store_steps,
)


def _spark_operation(**overrides):
    base = {
        "event_id": "E01",
        "action_type": "pour",
        "action_description": "Pour flour.",
        "start_seconds": 10.0,
        "end_seconds": 20.0,
        "ingredients": ["flour"],
        "containers_utensils": ["bowl"],
        "evidence": "visible pour",
        "confidence": "high",
        "uncertainty": None,
    }
    base.update(overrides)
    return base


def _open_event(**overrides):
    base = {
        "event_id": "S01_E01",
        "absolute_start_seconds": 12.0,
        "absolute_end_seconds": 22.0,
        "action_type": "pouring",
        "action_description": "Pouring flour.",
        "evidence": "flour stream",
        "uncertainty": "low",
        "ingredient_candidates": ["flour"],
        "utensils": [],
        "containers": ["bowl"],
        "confidence": "medium",
    }
    base.update(overrides)
    return base


def test_spark_adapter_normalizes_and_ids_deterministically():
    parsed = {"operations": [_spark_operation()]}
    first = normalize.spark_events(parsed, "resp_1")
    second = normalize.spark_events(parsed, "resp_1")
    assert first == second
    event = first[0]
    assert event["event_id"].startswith("fe_")
    assert event["source_ref"] == "resp_1:E01"
    assert event["action_canonical"] == "pour"
    assert event["confidence"] == "high"
    assert event["status"] == "ai_suggested"
    assert event["approximate"] is False


def test_spark_adapter_tolerates_missing_fields():
    events = normalize.spark_events({"operations": [{}]}, None)
    assert events[0]["action_canonical"] is None
    assert events[0]["start_seconds"] is None
    assert events[0]["confidence"] == "unknown"
    assert normalize.spark_events({}, None) == []
    assert normalize.spark_events({"operations": None}, None) == []


def test_confidence_mapping_is_conservative():
    assert normalize.normalize_confidence("HIGH certainty") == "high"
    assert normalize.normalize_confidence("moderate") == "medium"
    assert normalize.normalize_confidence("low - blurry") == "low"
    assert normalize.normalize_confidence("observed") == "unknown"
    assert normalize.normalize_confidence(None) == "unknown"
    assert normalize.normalize_confidence(7) == "unknown"


def test_baseten_adapter_prefers_absolute_time():
    event = normalize.baseten_events(
        {"culinary_events": [_open_event()]})[0]
    assert event["start_seconds"] == 12.0
    assert event["end_seconds"] == 22.0
    assert event["action_canonical"] == "pour"


def test_baseten_ingredients_offset_segment_timestamps():
    aggregated = {"ingredients": [{
        "normalized_name": "Flour",
        "visible_name_or_label": "white powder",
        "supporting_timestamps": [9.4],
        "segment_source_start_seconds": 30.0,
        "actually_used_or_merely_visible": "used",
        "visual_evidence": "bag",
        "ambiguities": "brand unclear",
        "segment_index": 1,
    }]}
    record = normalize.baseten_ingredients(aggregated)[0]
    assert record["normalized_name"] == "flour"
    assert record["supporting_timestamps"] == [39.4]
    assert record["ambiguities"] == ["brand unclear"]


def test_measurements_keep_unknowns_explicit():
    parsed = {"scale_readings": [{
        "timestamp_seconds": 5.0,
        "displayed_value": None,
        "visible_unit": None,
        "tare_observable": None,
        "uncertain": True,
        "evidence": "glare",
    }]}
    reading = normalize.spark_measurements(parsed, None)[0]
    assert reading["displayed_value"] is None
    assert reading["visible_unit"] is None
    assert reading["uncertain"] is True


def test_preliminary_events_are_approximate_context():
    annotation = {"candidate_events": [{
        "candidate_id": "C01",
        "reported_interval": "00:15-01:18",
        "candidate_type": "mixture addition",
        "verification_question": "Is 100 a net mass?",
    }]}
    note = normalize.preliminary_events(annotation)[0]
    assert note["approximate"] is True
    assert note["start_seconds"] == 15.0
    assert note["end_seconds"] == 78.0
    assert note["confidence"] == "unknown"
    assert "net mass" in (note["uncertainty"] or "")


def test_clock_parser_rejects_garbage():
    assert normalize.parse_clock_interval(None) == (None, None)
    assert normalize.parse_clock_interval("soon") == (None, None)
    assert normalize.parse_clock_interval("01:18-00:15") == (None, None)


def test_preliminary_ingests_segment_observations():
    annotation = {
        "candidate_events": [{
            "candidate_id": "C01",
            "reported_interval": "00:15-01:18",
            "candidate_type": "mixture addition",
            "verification_question": "Is 100 a net mass?",
        }],
        "segments": [{
            "segment_id": "S01",
            "observations": [{
                "observation_id": "pre_s1_02",
                "approximate_timestamp_seconds": 78,
                "reported_action": "Bowl removed from scale.",
                "reported_display_or_state": "Last reading 100.",
                "interpretation_and_verification_need":
                    "Net mass not established.",
            }],
        }],
    }
    notes = normalize.preliminary_events(annotation)
    assert len(notes) == 2
    assert all(note["approximate"] is True for note in notes)
    assert notes[0]["note_kind"] == "candidate"
    assert notes[1]["note_kind"] == "segment_observation"
    segment = [note for note in notes
               if note["source_ref"] == "S01:pre_s1_02"][0]
    assert segment["start_seconds"] == 78.0
    assert segment["end_seconds"] == 78.0
    assert "Last reading 100" in (segment["uncertainty"] or "")
    assert "Net mass not established" in (
        segment["uncertainty"] or "")
    assert normalize.preliminary_events({}) == []


def test_normalize_label_canonicalizes_without_identity():
    assert normalize.normalize_label("  Orange   Bag! ") == "orange bag"
    assert normalize.normalize_label(None) is None
    assert normalize.normalize_label(42) is None
    assert normalize.normalize_label("   ") is None


def test_unnamed_records_merge_on_identical_label_only():
    first = normalize.baseten_ingredients({"ingredients": [{
        "normalized_name": None,
        "visible_name_or_label": "Orange bag",
        "segment_index": 1,
        "supporting_timestamps": [5.0],
    }]})[0]
    second = normalize.baseten_ingredients({"ingredients": [{
        "normalized_name": None,
        "visible_name_or_label": "orange  bag!",
        "segment_index": 2,
        "supporting_timestamps": [65.0],
    }]})[0]
    third = normalize.baseten_ingredients({"ingredients": [{
        "normalized_name": None,
        "visible_name_or_label": "Green carton",
        "segment_index": 2,
        "supporting_timestamps": [66.0],
    }]})[0]
    groups = reconcile.group_ingredients([first, second, third])
    assert len(groups) == 2
    merged = [group for group in groups
              if len(group["members"]) == 2][0]
    assert merged["match_key"] == "label:orange bag"
    assert all(member.get("normalized_name") is None
               for member in merged["members"])


def test_ingredient_review_groups_by_identity_status():
    groups = [
        {"group_id": "ig_named", "members": [{
            "normalized_name": "water", "visible_label": "tap water",
            "used_or_visible": "used", "ambiguities": []}],
         "supporting_timestamps": [], "sources": ["s1"],
         "source_count": 1, "member_ids": ["f1"]},
        {"group_id": "ig_desc", "members": [{
            "normalized_name": None, "visible_label": "orange bag",
            "used_or_visible": None, "ambiguities": ["contents?"]}],
         "supporting_timestamps": [], "sources": ["s1"],
         "source_count": 1, "member_ids": ["f2"]},
        {"group_id": "ig_unres", "members": [{
            "normalized_name": None, "visible_label": None,
            "used_or_visible": None, "ambiguities": []}],
         "supporting_timestamps": [], "sources": ["s1"],
         "source_count": 1, "member_ids": ["f3"]},
    ]
    entries = reconstruct.build_ingredient_entries(groups)
    assert [entry["identity_status"] for entry in entries] == [
        "model_proposed", "described", "unresolved"]
    review = reconstruct.build_ingredient_review(entries)
    assert review["counts"] == {"model_proposed": 1, "described": 1,
                                "unresolved": 1}
    assert review["named_candidates"][0]["group_id"] == "ig_named"
    assert review["unresolved"][0]["display_label"] == \
        "unidentified observation"


def test_grouping_merges_overlap_with_alias_actions():
    spark = normalize.spark_events(
        {"operations": [_spark_operation()]}, None)[0]
    opened = normalize.baseten_events(
        {"culinary_events": [_open_event()]})[0]
    groups = reconcile.group_events([spark, opened])
    assert len(groups) == 1
    assert groups[0]["source_count"] == 2
    assert groups[0]["status"] == "ai_suggested"


def test_nested_detail_joins_despite_action_tokens():
    wide = normalize.spark_events(
        {"operations": [_spark_operation(
            action_type="prepare", start_seconds=0.0,
            end_seconds=30.0)]}, None)[0]
    narrow = normalize.baseten_events(
        {"culinary_events": [_open_event(
            action_type="retrieve_utensil",
            absolute_start_seconds=10.0,
            absolute_end_seconds=14.0)]})[0]
    groups = reconcile.group_events([wide, narrow])
    assert len(groups) == 1
    assert groups[0]["source_count"] == 2
    assert reconcile.detect_conflicts([wide, narrow], [], []) == []


def test_long_survey_does_not_absorb_distinct_details():
    survey = normalize.spark_events(
        {"operations": [_spark_operation(
            action_type="survey", start_seconds=530.0,
            end_seconds=639.4)]}, None)[0]
    detail = normalize.baseten_events(
        {"culinary_events": [_open_event(
            action_type="load_dishwasher",
            absolute_start_seconds=540.0,
            absolute_end_seconds=548.0)]})[0]
    groups = reconcile.group_events([survey, detail])
    assert len(groups) == 2
    assert all(group["source_count"] == 1 for group in groups)


def test_same_scale_disagreement_groups_and_conflicts():
    spark = normalize.spark_events(
        {"operations": [_spark_operation(action_type="pour")]}, None)[0]
    opened = normalize.baseten_events(
        {"culinary_events": [_open_event(action_type="stir",
                                         absolute_start_seconds=11.0,
                                         absolute_end_seconds=19.0)]})[0]
    groups = reconcile.group_events([spark, opened])
    assert len(groups) == 1  # nested: one step, disputed action
    conflicts = reconcile.detect_conflicts([spark, opened], [], [])
    assert len(conflicts) == 1
    assert conflicts[0]["kind"] == "action_conflict"


def test_action_aliases_and_token_normalization():
    assert normalize.canonical_action("cover_bowl") == "cover"
    assert normalize.canonical_action("retrieve tool") == "retrieve"
    assert normalize.canonical_action("weighing_ingredient") == "weigh"
    assert normalize.canonical_action("add water") == "add_water"
    assert normalize.canonical_action(None) is None


def test_possible_relatives_never_merge():
    left = {"record_id": "a", "source": "s1",
            "normalized_name": None,
            "visible_label": "wholemeal bread flour bag",
            "supporting_timestamps": []}
    right = {"record_id": "b", "source": "s2",
             "normalized_name": None,
             "visible_label": "open bread flour bag on counter",
             "supporting_timestamps": []}
    groups = reconcile.group_ingredients([left, right])
    assert len(groups) == 2
    assert groups[0]["possibly_related"] == [groups[1]["group_id"]]
    assert groups[1]["possibly_related"] == [groups[0]["group_id"]]


def test_preliminary_never_joins_groups_but_links():
    spark = normalize.spark_events(
        {"operations": [_spark_operation()]}, None)[0]
    note = normalize.preliminary_events({"candidate_events": [{
        "candidate_id": "C01", "reported_interval": "00:10-00:25",
        "candidate_type": "x", "verification_question": "q?"}, ]})[0]
    groups = reconcile.group_events([spark, note])
    assert len(groups) == 1
    assert groups[0]["member_ids"] == [spark["event_id"]]
    reconcile.link_preliminary(groups, [note])
    assert groups[0]["preliminary_links"] == [note["event_id"]]


def test_ingredient_grouping_and_conflicts():
    left = {"record_id": "a", "source": "s1",
            "normalized_name": "Flour",
            "supporting_timestamps": [10.0]}
    right = {"record_id": "b", "source": "s2",
             "normalized_name": " flour ",
             "supporting_timestamps": [11.0]}
    other = {"record_id": "c", "source": "s2",
             "normalized_name": "Sugar",
             "supporting_timestamps": [10.5]}
    groups = reconcile.group_ingredients([left, right, other])
    assert len(groups) == 2
    assert groups[0]["source_count"] == 2
    conflicts = reconcile.detect_conflicts([], [], [left, right, other])
    kinds = [c["kind"] for c in conflicts]
    assert "ingredient_conflict" in kinds


def test_reading_corroboration_never_verifies():
    left = {"reading_id": "r1", "source": "s1",
            "timestamp_seconds": 10.0, "displayed_value": "200"}
    right = {"reading_id": "r2", "source": "s2",
             "timestamp_seconds": 11.0, "displayed_value": "200"}
    same = {"reading_id": "r3", "source": "s2",
            "timestamp_seconds": 10.5, "displayed_value": "200"}
    pairs = reconcile.corroborated_readings([left, right, same])
    assert len(pairs) == 2  # cross-source only
    assert all(p["status"] == "ai_suggested" for p in pairs)
    conflicts = reconcile.detect_conflicts(
        [], [left, {**right, "displayed_value": "210"}], [])
    assert conflicts and conflicts[0]["kind"] == "reading_conflict"


def test_action_conflict_detection():
    left = {"event_id": "e1", "source": "s1",
            "action_canonical": "pour",
            "start_seconds": 10.0, "end_seconds": 20.0}
    right = {"event_id": "e2", "source": "s2",
             "action_canonical": "stir",
             "start_seconds": 11.0, "end_seconds": 19.0}
    conflicts = reconcile.detect_conflicts([left, right], [], [])
    assert len(conflicts) == 1
    assert conflicts[0]["kind"] == "action_conflict"


def test_draft_has_null_quantities_and_questions():
    groups = [{"group_id": "eg_000", "start_seconds": 1.0,
               "end_seconds": 2.0, "sources": ["s1"], "source_count": 1,
               "member_ids": ["e1"],
               "members": [{"event_id": "e1", "action_type": "pour",
                             "ingredients": ["flour"],
                             "uncertainty": "blurry"}],
               "preliminary_links": []}]
    ingredients = [{"group_id": "ig_x", "members": [{
        "normalized_name": None, "visible_label": "powder",
        "used_or_visible": None, "ambiguities": []}],
        "supporting_timestamps": [], "sources": ["s1"],
        "source_count": 1, "member_ids": ["f1"]}]
    readings = [{"displayed_value": None}]
    draft = reconstruct.build_draft(
        "Test bake", groups, ingredients, readings, [], [], [],
        {"fusion_version": "test"})
    assert draft["status"] == "draft_needs_review"
    assert draft["working_title"] == "Test bake"
    assert draft["ingredients"][0]["quantity"] is None
    assert draft["ingredients"][0]["unit"] is None
    assert "quantity_reason" in draft["ingredients"][0]
    assert draft["steps"][0]["step_index"] == 1
    assert any("no normalized name" in q
               for q in draft["unresolved_questions"])
    assert any("illegible" in q
               for q in draft["unresolved_questions"])


def test_questions_keep_candidates_not_segment_notes():
    preliminary = [
        {"source_ref": "C01", "note_kind": "candidate",
         "uncertainty": "Is 100 a net mass?"},
        {"source_ref": "S01:pre_s1_02", "note_kind": "segment_observation",
         "uncertainty": "Net mass not established."},
    ]
    questions = reconstruct.build_questions([], [], [], preliminary)
    assert len(questions) == 1
    assert "C01" in questions[0]


def test_draft_without_title_stays_unknown():
    draft = reconstruct.build_draft(
        None, [], [], [], [], [], [], {})
    assert draft["working_title"] is None
    assert draft["working_title_status"] == "unknown"


def test_review_adapter_shape_and_provenance():
    draft = {"ingredients": [{
        "group_id": "ig_x", "names": ["flour"],
        "visible_labels": ["white powder"], "ambiguities": ["brand"]}],
        "steps": [{"group_id": "eg_000", "actions": ["pour"],
                   "start_seconds": 1.0, "end_seconds": 2.0,
                   "uncertainties": ["blurry"]}]}
    ingredients = to_store_ingredients(draft)
    steps = to_store_steps(draft)
    assert ingredients[0]["ingredient_id"] == "flour"
    assert ingredients[0]["quantity"] == {
        "value": None, "unit": None, "derivation": "unknown"}
    assert ingredients[0]["provenance"] == "ai-extracted"
    assert steps[0]["provenance"] == "ai-extracted"
    assert steps[0]["review_status"] == "pending"
    assert steps[0]["usages"] == []


def test_iou_edge_cases():
    assert reconcile.interval_iou(None, 1.0, 0.0, 1.0) == 0.0
    assert reconcile.interval_iou(0.0, 0.0, 0.0, 1.0) == 0.0
    assert reconcile.interval_iou(0.0, 10.0, 20.0, 30.0) == 0.0
    assert reconcile.interval_iou(0.0, 10.0, 0.0, 10.0) == 1.0


def test_step_members_are_ordered_with_spans():
    groups = [{"group_id": "eg_000", "start_seconds": 10.0,
               "end_seconds": 20.0, "sources": ["s1", "s2"],
               "source_count": 2,
               "member_ids": ["e1", "e2"],
               "members": [
                   {"event_id": "e2", "source": "s2",
                    "source_ref": "r2", "action_type": "stir",
                    "start_seconds": 15.0, "end_seconds": 18.0,
                    "ingredients": []},
                   {"event_id": "e1", "source": "s1",
                    "source_ref": "r1", "action_type": "pour",
                    "start_seconds": 10.0, "end_seconds": 20.0,
                    "ingredients": []}],
               "preliminary_links": []}]
    steps = reconstruct.build_steps(groups)
    members = steps[0]["members"]
    assert [m["member_id"] for m in members] == ["e1", "e2"]
    assert members[0]["source_ref"] == "r1"
    assert members[1]["action"] == "stir"
    assert steps[0]["uncertainties"] == []


def test_duplicate_readings_fold_exactly():
    readings = [
        {"reading_id": "a", "source": "s1", "source_ref": "r1",
         "timestamp_seconds": 124.0, "displayed_value": "318",
         "evidence": "See scale_readings[0]"},
        {"reading_id": "b", "source": "s1", "source_ref": "r2",
         "timestamp_seconds": 124.0, "displayed_value": "318",
         "evidence": "Bowl on scale, LCD shows 318"},
        {"reading_id": "c", "source": "s1", "source_ref": "r3",
         "timestamp_seconds": 124.0, "displayed_value": "319"},
        {"reading_id": "d", "source": "s1", "source_ref": "r4",
         "timestamp_seconds": 130.0, "displayed_value": None,
         "evidence": "Display blurred by motion"},
        {"reading_id": "e", "source": "s1", "source_ref": "r5",
         "timestamp_seconds": 130.0, "displayed_value": None,
         "evidence": "Digits washed out by glare"},
    ]
    unique, duplicates = quantities.consolidate_duplicates(readings)
    assert [r["reading_id"] for r in unique] == ["b", "c", "d", "e"]
    assert duplicates == [{"reading_id": "a", "duplicate_of": "b",
                           "source": "s1", "source_ref": "r1"}]
    assert "duplicate_ids" not in readings[0]


def test_stable_cluster_without_unit_is_unavailable():
    readings = [
        {"reading_id": "a", "source": "s1",
         "timestamp_seconds": 124.0, "displayed_value": "318",
         "visible_unit": None, "tare_observable": False,
         "uncertain": False},
        {"reading_id": "b", "source": "s1",
         "timestamp_seconds": 129.3, "displayed_value": "318",
         "visible_unit": None, "tare_observable": False,
         "uncertain": False},
        {"reading_id": "c", "source": "s1",
         "timestamp_seconds": 200.0, "displayed_value": "318",
         "visible_unit": None, "tare_observable": False,
         "uncertain": False},
    ]
    assessment = quantities.build_quantity_assessment(readings, [])
    assert assessment["readings_unique"] == 3
    assert len(assessment["stable_clusters"]) == 2
    first = assessment["stable_clusters"][0]
    assert first["reading_ids"] == ["a", "b"]
    assert first["derivation_status"] == "unavailable"
    assert any("unit" in item
               for item in first["missing_for_quantity"])
    assert any("tare" in item
               for item in first["missing_for_quantity"])
    assert assessment["quantities_recoverable"] == 0


def test_tare_claims_stay_unverified():
    claims = quantities.normalize_tare_claims(
        ["around 209-215s possible tare but not confirmed", "", None],
        "spark")
    assert len(claims) == 1
    assert claims[0]["status"] == "unverified"
    assert claims[0]["claim_id"].startswith("tc_")
    assert quantities.normalize_tare_claims(None, "spark") == []
    assert quantities.normalize_tare_claims("tare?", "spark") == []


def test_draft_carries_review_and_assessment():
    groups = [{"group_id": "eg_000", "start_seconds": 1.0,
               "end_seconds": 2.0, "sources": ["s1"], "source_count": 1,
               "member_ids": ["e1"],
               "members": [{"event_id": "e1", "action_type": "pour",
                             "ingredients": [],
                             "uncertainty": None}],
               "preliminary_links": []}]
    ingredients = [{"group_id": "ig_x", "members": [{
        "normalized_name": None, "visible_label": "powder",
        "used_or_visible": None, "ambiguities": []}],
        "supporting_timestamps": [], "sources": ["s1"],
        "source_count": 1, "member_ids": ["f1"]}]
    assessment = quantities.build_quantity_assessment([], [])
    draft = reconstruct.build_draft(
        None, groups, ingredients, [], [], [], [], {},
        quantity_assessment=assessment)
    assert draft["ingredient_review"]["counts"] == {
        "model_proposed": 0, "described": 1, "unresolved": 0}
    assert draft["quantity_assessment"]["status"] == "evaluated"
    assert draft["quantity_assessment"][
        "quantities_recoverable"] == 0
    legacy = reconstruct.build_draft(
        None, [], [], [], [], [], [], {})
    assert legacy["quantity_assessment"] == {
        "status": "not_evaluated"}
    assert legacy["ingredient_review"]["counts"] == {
        "model_proposed": 0, "described": 0, "unresolved": 0}


def test_step_coverage_gap_is_flagged():
    groups = [{"group_id": "eg_000", "start_seconds": 0.0,
               "end_seconds": 100.0, "sources": ["s1", "s2"],
               "source_count": 2,
               "member_ids": ["wide", "a", "b"],
               "members": [
                   {"event_id": "wide", "source": "s1",
                    "source_ref": "r", "action_type": "load",
                    "start_seconds": 0.0, "end_seconds": 100.0,
                    "ingredients": []},
                   {"event_id": "a", "source": "s2",
                    "source_ref": "r", "action_type": "place",
                    "start_seconds": 5.0, "end_seconds": 10.0,
                    "ingredients": []},
                   {"event_id": "b", "source": "s2",
                    "source_ref": "r", "action_type": "place",
                    "start_seconds": 70.0, "end_seconds": 75.0,
                    "ingredients": []}],
               "preliminary_links": []}]
    steps = reconstruct.build_steps(groups)
    assert any("internal gap is 60.0s" in u
               for u in steps[0]["uncertainties"])


def test_local_segment_ingredient_aliases_stay_distinct():
    """Agent B 'name/appearance' items must not collapse to one id.

    Regression: local-segment ingredients use short keys while the
    global analyst uses normalized_name/visible_label. Records that
    lose both keys share one stable id and render as duplicate
    'unidentified observation' groups with colliding review targets.
    """
    parsed = {"ingredients": [
        {"name": "dry yeast",
         "appearance": "light-beige granular powder",
         "spanish_name": "levadura seca"},
        {"name": "milk",
         "appearance_visible": "white liquid poured from red mug"},
        {"name": "white powder", "state": "fine powder in bowl"},
    ]}
    records = normalize.spark_ingredients(
        parsed, "resp_local", "clip_01_preparation")
    assert [r["normalized_name"] for r in records] == [
        "dry yeast", "milk", "white powder"]
    assert records[0]["visible_label"] == \
        "light-beige granular powder"
    assert records[1]["visible_label"] == \
        "white liquid poured from red mug"
    assert records[2]["visible_label"] == "fine powder in bowl"
    assert len({r["record_id"] for r in records}) == 3
    groups = reconcile.group_ingredients(records)
    assert len(groups) == 3
    assert len({g["group_id"] for g in groups}) == 3


def test_global_ingredient_keys_keep_precedence():
    """Alias fallback must not alter global-analyst records."""
    parsed = {"ingredients": [{
        "normalized_name": "Flour", "visible_label": "bulk bag",
        "name": "ignored", "appearance": "ignored",
        "supporting_timestamps": [1.0]}]}
    (record,) = normalize.spark_ingredients(parsed, "resp", None)
    assert record["normalized_name"] == "flour"
    assert record["visible_label"] == "bulk bag"


def test_local_segment_event_aliases_and_window_note():
    """Agent B operations map short keys; degenerate ids diverge.

    Local ops use 'action'/'type', 'description'/'detail', and
    carry no event id or absolute timestamps. Each op keeps its
    own record id, and the analyzed window is recorded as an
    uncertainty note instead of invented timestamps.
    """
    parsed = {"operations": [
        {"action": "pour", "description": "powder into bowl",
         "ingredient": "white powder", "tool": "scale"},
        {"type": "tare", "Detail": "display returns to 0"},
    ]}
    events = normalize.spark_events(
        parsed, "resp_seg", "clip_01_preparation",
        window=[165.0, 190.0])
    assert [e["action_type"] for e in events] == ["pour", "tare"]
    assert events[0]["action_description"] == "powder into bowl"
    assert events[1]["action_description"] == "display returns to 0"
    assert events[0]["ingredients"] == ["white powder"]
    assert events[0]["utensils"] == ["scale"]
    assert events[0]["start_seconds"] is None
    assert events[0]["end_seconds"] is None
    assert any("165.0" in (e.get("uncertainty") or "")
               and "190.0" in (e.get("uncertainty") or "")
               for e in events)
    assert len({e["event_id"] for e in events}) == 2


def test_global_event_keys_keep_precedence():
    """Alias fallback must not alter full-video operations."""
    parsed = {"operations": [{
        "event_id": "op1", "action_type": "pour",
        "action": "ignored", "type": "ignored",
        "action_description": "flour pour",
        "description": "ignored",
        "start_seconds": 1.0, "end_seconds": 2.0,
        "ingredients": ["flour"],
        "containers_utensils": ["bowl"]}]}
    (event,) = normalize.spark_events(parsed, "resp", None)
    assert event["action_type"] == "pour"
    assert event["action_description"] == "flour pour"
    assert event["ingredients"] == ["flour"]
    assert event["utensils"] == ["bowl"]
    assert event["uncertainty"] is None


def test_speech_claims_attach_by_turn_index():
    """Culinary claims join their transcript turns; strays counted."""
    turns = [{"transcript": "make rolls", "start_seconds": 0.5},
             {"transcript": "weigh flour", "start_seconds": 5.0}]
    claims = [
        {"claim_id": "c00", "turn_ids": [0, 0],
         "interpretation": "goal stated"},
        {"claim_id": "c01", "turn_ids": [1, 7],
         "interpretation": "flour weighed"},
        {"claim_id": "c02", "turn_ids": [9],
         "interpretation": "stray"},
    ]
    attached, orphaned = normalize.attach_claims(turns, claims)
    assert attached == 2
    assert orphaned == 1
    assert [c["claim_id"] for c in turns[0]["claims"]] == ["c00"]
    assert [c["claim_id"] for c in turns[1]["claims"]] == ["c01"]
    records = normalize.speech_turns(turns, "clip_01_preparation")
    assert records[0]["claims"][0]["interpretation"] == \
        "goal stated"
