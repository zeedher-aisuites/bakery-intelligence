"""Recipe snapshot + PDF export tests (offline fixtures)."""

import json

from bakery_intelligence.recipe_export import pdf as pdf_mod
from bakery_intelligence.recipe_export import snapshot as snap
from bakery_intelligence.recipe_export import v2snapshot as v2


def _draft():
    return {
        "ingredients": [
            {"group_id": "ig_a", "display_label": "white powder",
             "clip_id": "c1", "supporting_timestamps": [10.0],
             "sources": ["s"]},
            {"group_id": "ig_b", "display_label": "white powder",
             "clip_id": "c2", "supporting_timestamps": [],
             "sources": ["s"]},
            {"group_id": "ig_c", "display_label": "mystery",
             "clip_id": "c1", "supporting_timestamps": [],
             "sources": ["s"]},
        ],
        "steps": [
            {"group_id": "eg_000", "clip_id": "c1",
             "actions": ["pour"]},
        ],
        "quantity_assessment": {
            "stable_clusters": [
                {"cluster_id": "qc_1", "clip_id": "c1",
                 "start_seconds": 10.0, "displayed_value": "100"},
            ],
        },
    }


def test_reviewed_name_prefers_correction():
    group = {"display_label": "white powder"}
    assert snap.reviewed_name(
        group, {"action": "correct",
                "value": {"name": "sugar"}}) == "sugar"
    assert snap.reviewed_name(
        group, {"action": "confirm", "value": None}) == \
        "white powder"
    assert snap.reviewed_name(
        group, {"action": "cannot_determine"}) is None
    assert snap.reviewed_name(group, None) is None


def test_consolidation_merges_aliases_and_excludes_unknown():
    real_aliases = dict(snap.CONSOLIDATION_ALIASES)
    snap.CONSOLIDATION_ALIASES["white powder"] = "sugar"
    try:
        draft = _draft()
        decisions = {
            ("ingredient", "ig_a"): {"action": "correct",
                                     "value": {"name": "Sugar"}},
            ("ingredient", "ig_b"): {"action": "confirm",
                                     "value": None},
            ("ingredient", "ig_c"): {"action": "cannot_determine",
                                     "value": None},
        }
        ingredients, excluded = snap.consolidate_ingredients(
            draft, decisions)
        assert len(ingredients) == 1
        row = ingredients[0]
        assert row["name"] == "Sugar"
        assert row["group_ids"] == ["ig_a", "ig_b"]
        assert row["review_status"] == ["human_corrected",
                                        "human_verified"]
        assert row["quantity_status"] == "not_determined"
        assert len(excluded) == 1
        assert excluded[0]["group_id"] == "ig_c"
    finally:
        snap.CONSOLIDATION_ALIASES.clear()
        snap.CONSOLIDATION_ALIASES.update(real_aliases)


def test_quantity_attaches_only_on_matching_supply():
    real_attr = dict(snap.QUANTITY_ATTRIBUTION)
    snap.QUANTITY_ATTRIBUTION["qc_1"] = {
        "ingredient": "sugar", "amount": 100.0, "unit": "g",
        "basis": "test basis",
    }
    try:
        draft = _draft()
        rows = [{"name": "sugar", "quantity": None, "unit": None,
                 "quantity_status": "not_determined",
                 "quantity_basis": None, "evidence": []}]
        snap.attribute_quantities(rows, draft, {
            ("quantity_cluster", "qc_1"): {
                "action": "supply",
                "value": {"amount": 100.0, "unit": "g"}},
        })
        assert rows[0]["quantity"] == 100.0
        assert rows[0]["unit"] == "g"
        assert rows[0]["quantity_status"] == "chef_supplied"

        rows = [{"name": "sugar", "quantity": None, "unit": None,
                 "quantity_status": "not_determined",
                 "quantity_basis": None, "evidence": []}]
        snap.attribute_quantities(rows, draft, {
            ("quantity_cluster", "qc_1"): {
                "action": "supply",
                "value": {"amount": 50.0, "unit": "g"}},
        })
        assert rows[0]["quantity_status"] == "not_determined"
    finally:
        snap.QUANTITY_ATTRIBUTION.clear()
        snap.QUANTITY_ATTRIBUTION.update(real_attr)


def test_method_intervals_must_contain_step_spans():
    import pytest

    real_plan = list(snap.METHOD_PLAN)
    snap.METHOD_PLAN[:] = [{
        "clip_id": "c1", "interval": "0:00-0:01",
        "instruction": "Do it.", "step_group_ids": ["eg_000"],
        "caveats": []}]
    try:
        draft = _draft()
        draft["steps"][0]["start_seconds"] = 50.0
        draft["steps"][0]["end_seconds"] = 60.0
        with pytest.raises(ValueError, match="outside instruction"):
            snap.build_method(draft, {})
    finally:
        snap.METHOD_PLAN[:] = real_plan


def test_method_plan_requires_full_coverage(monkeypatch):
    real_plan = list(snap.METHOD_PLAN)
    real_omitted = dict(snap.OMITTED_DETAIL_STEPS)
    monkeypatch.setattr(snap, "METHOD_PLAN", [{
        "clip_id": "c1", "interval": "0:00-0:01",
        "instruction": "Do it.", "step_group_ids": ["eg_000"],
        "caveats": []}])
    monkeypatch.setattr(snap, "OMITTED_DETAIL_STEPS", {})
    try:
        method, omitted = snap.build_method(
            _draft(), {("step", "eg_000"): {"action": "confirm"}})
        assert len(method) == 1
        assert method[0]["review_status"] == ["human_verified"]
        assert omitted == []
    finally:
        monkeypatch.setattr(snap, "METHOD_PLAN", real_plan)
        monkeypatch.setattr(snap, "OMITTED_DETAIL_STEPS",
                            real_omitted)


def test_snapshot_counts_and_pdf_smoke(tmp_path):
    real_plan = list(snap.METHOD_PLAN)
    real_omitted = dict(snap.OMITTED_DETAIL_STEPS)
    snap.METHOD_PLAN[:] = [{
        "clip_id": "c1", "interval": "0:00-0:01",
        "instruction": "Do it carefully.",
        "step_group_ids": ["eg_000"], "caveats": ["Hot."]}]
    snap.OMITTED_DETAIL_STEPS.clear()
    try:
        decisions = {
            ("ingredient", "ig_a"): {"action": "correct",
                                     "value": {"name": "sugar"}},
            ("ingredient", "ig_b"): {"action": "confirm",
                                     "value": None},
            ("ingredient", "ig_c"): {"action": "cannot_determine",
                                     "value": None},
            ("step", "eg_000"): {"action": "confirm",
                                 "value": None},
        }
        shot = snap.build_snapshot(
            _draft(), decisions, {"request_id": "resp_x"},
            "2026-01-01T00:00:00Z")
        assert shot["snapshot_version"] == snap.SNAPSHOT_VERSION
        assert shot["review_summary"]["decisions"] == 4
        assert shot["provenance"]["agent_f_request_id"] == "resp_x"
        assert "white powder" in {
            r["name"] for r in shot["ingredients"]}

        data = pdf_mod.build_pdf(shot)
        assert data[:5] == b"%PDF-"
        target = tmp_path / "smoke.pdf"
        target.write_bytes(data)
        import pymupdf
        with pymupdf.open(target) as doc:
            assert doc.page_count >= 1
            text = "\n".join(page.get_text()
                              for page in doc)
        assert "Cinnamon Rolls" in text
        assert "Not determined" in text
        assert "Do it carefully." in text
    finally:
        snap.METHOD_PLAN[:] = real_plan
        snap.OMITTED_DETAIL_STEPS.clear()
        snap.OMITTED_DETAIL_STEPS.update(real_omitted)


def test_snapshot_is_json_stable():
    payload = {"b": [1, 2], "a": None}
    assert json.dumps(payload, sort_keys=True) == \
        '{"a": null, "b": [1, 2]}'


def test_v2_completion_attaches_only_valid_estimates():
    rows = [
        {"name": "milk", "quantity": None, "unit": None,
         "quantity_status": "not_determined"},
        {"name": "sugar", "quantity": 100.0, "unit": "g",
         "quantity_status": "chef_supplied"},
    ]
    open_items = v2.apply_completion(rows, [
        {"ingredient": "milk", "amount": 240, "unit": "g",
         "rationale": "hydration", "confidence": "medium"},
        {"ingredient": "sugar", "amount": 5, "unit": "g",
         "rationale": "x", "confidence": "high"},
        {"ingredient": "ghost", "amount": 1, "unit": "g",
         "rationale": "x", "confidence": "low"},
        {"ingredient": "milk", "amount": -2, "unit": "g",
         "rationale": "x", "confidence": "low"},
    ])
    assert rows[0]["quantity"] == 240.0
    assert rows[0]["quantity_status"] == "ai_estimated"
    assert rows[1]["quantity"] == 100.0
    assert len(open_items) == 3


def test_v2_roundings_preserve_displays():
    rows = [{"name": "bread flour", "quantity": 699.0, "unit": "g",
             "quantity_status": "chef_supplied"}]
    v2.apply_roundings(rows)
    assert rows[0]["quantity"] == 700.0
    assert rows[0]["draft_adjustment"]["displayed_value"] == 699.0


def test_v2_excludes_flagged_group_with_reason():
    rows = [{"name": "wheat flour",
             "group_ids": ["ig_keep", "ig_19421ce47de7"]}]
    kept = v2.exclude_flagged(rows)
    assert kept[0]["group_ids"] == ["ig_keep"]


def test_v2_guidance_requires_known_steps():
    method = [{"number": 1, "instruction": "Mix."}]
    open_items = v2.attach_guidance(method, [
        {"step_number": 1, "guidance": "Stir well."},
        {"step_number": 99, "guidance": "Lost."},
    ])
    assert method[0]["ai_guidance"]["provenance"] == "ai_estimated"
    assert len(open_items) == 1
