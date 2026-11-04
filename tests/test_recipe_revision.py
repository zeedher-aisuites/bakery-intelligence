"""Offline tests for multi-recipe records, v3, and revisions.

Synthetic snapshots in tmp storage. Revision route tests patch
the provider call; no network, no inference, no billable cost.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from bakery_intelligence.api import app as app_module
from bakery_intelligence.api.app import build_app
from bakery_intelligence.api.recipe_editor import (
    create_record,
    list_recipes,
    merged_recipe,
)
from bakery_intelligence.recipe_export import revision as revision_mod
from bakery_intelligence.recipe_export.labels import (
    adjustment_note,
    row_source,
)
from bakery_intelligence.recipe_export.v3snapshot import build_v3

SNAPSHOT = {
    "snapshot_version": "recipe-snapshot/2",
    "snapshot_id": "demo_v2",
    "run_name": "bakery_demo_v1",
    "generated_at": "2026-01-01T00:00:00Z",
    "title": "Cinnamon Rolls",
    "recipe_status": "draft",
    "ingredients": [
        {"name": "sugar", "quantity": 100.0, "unit": "g",
         "quantity_status": "chef_supplied",
         "quantity_basis": "Displayed 100; spoken 'cien gramos'.",
         "review_status": ["human_corrected"], "group_ids": ["g1"],
         "evidence": [{"kind": "scale_display"}]},
        {"name": "milk", "quantity": 240.0, "unit": "g",
         "quantity_status": "ai_estimated", "quantity_basis": "est",
         "review_status": ["unknown"], "group_ids": ["g2"],
         "evidence": []},
    ],
    "method": [
        {"number": 1, "clip_id": "clip_01_preparation",
         "interval": "0:00-0:10", "instruction": "Mix. "
         "Exact time was not confirmed.",
         "step_group_ids": ["eg_0"],
         "review_status": ["human_verified"], "caveats": []},
    ],
    "review_flags": [],
    "limitations": ["No ground truth."],
}


def _seed(root: Path, recipe_id: str = "demo_v2") -> Path:
    recipe_dir = root / "recipes" / recipe_id
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "snapshot.json").write_text(
        json.dumps(SNAPSHOT), encoding="utf-8")
    return root


def test_create_and_list_recipes(tmp_path: Path):
    root = _seed(tmp_path / "runs_api")
    record = create_record(root, "XL Cinnamon Rolls", "es",
                           recipe_id="r1")
    assert record["language"] == "es"
    assert record["has_snapshot"] is False
    create_record(root, "Cinnamon Rolls", "en",
                  recipe_id="demo_v2")
    cards = {c["recipe_id"]: c for c in list_recipes(root)}
    assert set(cards) == {"demo_v2", "r1"}
    assert cards["demo_v2"]["ingredients"] == 2
    assert cards["demo_v2"]["steps"] == 1
    assert cards["demo_v2"]["status"] == "draft"
    assert cards["r1"]["status"] == "processing"
    assert cards["r1"]["language"] == "es"


def test_create_rejects_bad_language(tmp_path: Path):
    with pytest.raises(ValueError):
        create_record(tmp_path, "X", "fr")


def test_v3_rewrites_prose_and_preserves_evidence():
    v2 = copy.deepcopy(SNAPSHOT)
    v2["method"] = [{
        "number": 1, "clip_id": "c", "interval": "0:00-0:10",
        "instruction": "Old.", "step_group_ids": [],
        "review_status": [], "caveats": ["old caveat"],
    }]
    snapshot = build_v3(v2, "2026-01-01T00:00:00Z")
    item = snapshot["method"][0]
    assert item["instruction"].startswith("Tare the scale")
    assert item["technical_instruction"] == "Old."
    assert item["interval"] == "0:00-0:10"
    assert item["caveats"] == ["old caveat"]
    assert "not confirmed" not in item["instruction"].lower()
    assert len(snapshot["notes_en"]) == 7
    assert snapshot["ingredients"] == v2["ingredients"]
    with pytest.raises(ValueError):
        build_v3({**v2, "method": [{
            "number": 99, "instruction": "x"}]},
            "2026-01-01T00:00:00Z")


def test_provenance_labels():
    sugar = SNAPSHOT["ingredients"][0]
    milk = SNAPSHOT["ingredients"][1]
    assert row_source(sugar, "en") == \
        "Seen in video + Mentioned in audio"
    assert row_source(sugar, "es") == \
        "Visto en el video + Mencionado en el audio"
    assert row_source(milk, "en") == "Suggested by AI"
    assert row_source({**milk, "chef_edited": True}, "en") == \
        "Edited by chef"
    assert row_source({"quantity_status": "not_determined"},
                      "en") == "Unknown"


def test_adjustment_note():
    row = {"unit": "g", "draft_adjustment": {
        "displayed_value": 699, "draft_value": 700}}
    assert adjustment_note(row, "en") == \
        "Scale showed 699 g; 700 g was mentioned in the audio."
    assert "mostró 699" in (adjustment_note(row, "es") or "")
    assert adjustment_note({}, "en") is None


def test_revision_validation_protects_ingredients():
    revised = {
        "title": "T", "changes_summary": "s",
        "ingredients": [
            {"name": "sugar", "quantity": 100.0, "unit": "g"},
            {"name": "milk", "quantity": 999.0, "unit": "g"},
        ],
        "method": [{"number": 1, "instruction": "Mix well."}],
        "notes": ["n1"],
    }
    with pytest.raises(revision_mod.RevisionError):
        revision_mod.validate_revision(SNAPSHOT, revised, False)
    ok = revision_mod.validate_revision(SNAPSHOT, revised, True)
    assert ok["title"] == "T"
    bad_steps = copy.deepcopy(revised)
    bad_steps["ingredients"][1]["quantity"] = 240.0
    bad_steps["method"] = []
    with pytest.raises(revision_mod.RevisionError):
        revision_mod.validate_revision(SNAPSHOT, bad_steps, False)


def _patched_client(monkeypatch: pytest.MonkeyPatch,
                    root: Path,
                    parsed: dict[str, Any]) -> TestClient:
    def fake_revision(client: Any, bundle: dict[str, Any],
                      work_dir: Path,
                      ) -> tuple[dict[str, Any], Any]:
        record = {
            "model": "meta-models/muse-spark-open-1026",
            "provider": "baseten",
            "prompt_version": "v1.0",
            "request_id": "chat_rev1",
            "finish_reason": "stop",
            "usage": {"input_tokens": 100,
                      "output_tokens": 50,
                      "total_tokens": 150},
            "parsed": parsed,
        }
        (work_dir / "revision_raw.json").write_text(
            json.dumps(record), encoding="utf-8")
        return record, None

    monkeypatch.setattr(app_module, "run_revision_stage",
                        fake_revision)
    monkeypatch.setattr(app_module, "BasetenClient",
                        lambda: object())
    return TestClient(build_app(storage_dir=root))


def test_revise_route_creates_new_version(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = _seed(tmp_path / "runs_api")
    create_record(root, "Cinnamon Rolls", "en", recipe_id="demo_v2")
    parsed = {
        "title": "Cinnamon Rolls", "changes_summary": "Cleaned.",
        "ingredients": [
            {"name": "sugar", "quantity": 100.0, "unit": "g"},
            {"name": "milk", "quantity": 240.0, "unit": "g"},
        ],
        "method": [{"number": 1, "instruction": "Mix smoothly."}],
        "notes": ["Confirm quantities."],
    }
    client = _patched_client(monkeypatch, root, parsed)
    response = client.post("/api/recipes/demo_v2/revise", json={
        "instructions": "Clean the wording.",
        "suggestions": [{"id": "mixer-params", "selected": True,
                         "value": "Speed 2"}],
        "allow_ingredient_changes": False, "confirmed": True})
    assert response.status_code == 201
    payload = response.json()
    assert payload["recipe_id"] == "demo_v2_r1"
    assert payload["provider"] == "baseten"
    assert payload["model"] == \
        "meta-models/muse-spark-open-1026"
    assert payload["request_id"] == "chat_rev1"
    assert payload["actual_usd"] is None
    assert payload["recipe"]["method"][0]["instruction"] == \
        "Mix smoothly."
    assert payload["recipe"]["recipe_status"] == "draft"
    parent = merged_recipe(root, "demo_v2")
    assert parent["method"][0]["instruction"].startswith("Mix. ")
    cards = {c["recipe_id"]: c for c in list_recipes(root)}
    assert cards["demo_v2_r1"]["status"] == "draft"
    assert (root / "recipes" / "demo_v2_r1"
            / "revision_raw.json").is_file()
    child = json.loads(
        (root / "recipes" / "demo_v2_r1" / "snapshot.json"
         ).read_text(encoding="utf-8"))
    assert child["revision"]["parent_version"] == "demo_v2"
    assert child["revision"]["provider"] == "baseten"
    assert child["revision"]["model"] == \
        "meta-models/muse-spark-open-1026"
    assert child["revision"]["chef_instructions"] == \
        "Clean the wording."


def test_revise_route_rejects_unconfirmed_or_violation(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = _seed(tmp_path / "runs_api")
    create_record(root, "Cinnamon Rolls", "en", recipe_id="demo_v2")
    parsed = {
        "title": "T", "changes_summary": "s",
        "ingredients": [
            {"name": "sugar", "quantity": 100.0, "unit": "g"},
            {"name": "milk", "quantity": 240.0, "unit": "g"},
        ],
        "method": [{"number": 1, "instruction": "Mix."}],
        "notes": ["n"],
    }
    client = _patched_client(monkeypatch, root, parsed)
    assert client.post("/api/recipes/demo_v2/revise", json={
        "instructions": "x", "confirmed": False}).status_code == 422
    assert client.post("/api/recipes/demo_v2/revise", json={
        "instructions": "  ", "confirmed": True}).status_code == 422


def test_revise_route_rejects_protected_change(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = _seed(tmp_path / "runs_api")
    create_record(root, "Cinnamon Rolls", "en", recipe_id="demo_v2")
    parsed = {
        "title": "T", "changes_summary": "s",
        "ingredients": [
            {"name": "sugar", "quantity": 100.0, "unit": "g"},
            {"name": "milk", "quantity": 5.0, "unit": "g"},
        ],
        "method": [{"number": 1, "instruction": "Mix."}],
        "notes": ["n"],
    }
    client = _patched_client(monkeypatch, root, parsed)
    response = client.post("/api/recipes/demo_v2/revise", json={
        "instructions": "Change nothing.", "confirmed": True})
    assert response.status_code == 422


def test_revision_bundle_carries_unresolved_and_evidence():
    bundle = revision_mod.revision_bundle(
        SNAPSHOT, "Do it.", [], "en", False, "Demo",
        unresolved_questions=["UNIT_UNKNOWN"],
        evidence={"merged": {"readings_total": 3}})
    assert bundle["unresolved_questions"] == ["UNIT_UNKNOWN"]
    assert bundle["evidence"]["merged"][
        "readings_total"] == 3
    assert bundle["ingredients"][0]["quantity"] == 100.0


def test_revision_bundle_defaults_empty():
    bundle = revision_mod.revision_bundle(
        SNAPSHOT, "Do it.", [], "en", False, "Demo")
    assert bundle["unresolved_questions"] == []
    assert bundle["evidence"] == {}


def test_revision_evidence_missing_run(
        tmp_path: Path):
    from bakery_intelligence.api.app import _revision_evidence

    root = tmp_path / "runs_api"
    assert _revision_evidence(root, None) == {}
    assert _revision_evidence(root, "run_missing") == {}


def test_revision_evidence_caps_merged(
        tmp_path: Path):
    from bakery_intelligence.api.app import _revision_evidence

    root = tmp_path / "runs_api"
    artifacts = root / "runs" / "run_x" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "multimodal_merged.json").write_text(
        json.dumps({
            "readings_total": 5, "segment_count": 2,
            "transcript_turns": 1,
            "unresolved": ["UNIT_UNKNOWN"],
            "items": [{"n": n} for n in range(200)],
        }), encoding="utf-8")
    (artifacts / "multimodal_validation.json").write_text(
        json.dumps({"model": "muse-spark-1.3",
                    "provider": "meta",
                    "result": {"checks": []}}),
        encoding="utf-8")
    (artifacts / "multimodal_segments_index.json"
     ).write_text(json.dumps({
         "count": 2,
         "model": "meta-models/muse-spark-open-1026"}),
        encoding="utf-8")
    evidence = _revision_evidence(root, "run_x")
    assert evidence["merged"]["readings_total"] == 5
    assert len(evidence["merged"]["items"]) == 120
    assert evidence["merged"]["unresolved"] == [
        "UNIT_UNKNOWN"]
    assert evidence["validation"]["provider"] == "meta"
    assert evidence["segments"]["count"] == 2


def test_revise_route_missing_baseten_key(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = _seed(tmp_path / "runs_api")
    create_record(root, "Cinnamon Rolls", "en",
                  recipe_id="demo_v2")

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "revision stage must not run without a key")

    monkeypatch.setattr(app_module, "run_revision_stage",
                        explode)
    client = TestClient(build_app(storage_dir=root))
    # Deleted after build: build_app reloads .env, which
    # would otherwise restore a developer key here.
    monkeypatch.delenv("BASETEN_API_KEY", raising=False)
    response = client.post("/api/recipes/demo_v2/revise", json={
        "instructions": "Clean the wording.",
        "confirmed": True})
    assert response.status_code == 502
    assert "BASETEN_API_KEY" in response.json()["detail"]


def test_revise_route_sends_evidence_to_model(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = _seed(tmp_path / "runs_api")
    create_record(root, "Cinnamon Rolls", "en",
                  recipe_id="demo_v2",
                  source_run_id="run_x")
    artifacts = root / "runs" / "run_x" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "multimodal_merged.json").write_text(
        json.dumps({"readings_total": 5,
                    "segment_count": 1,
                    "transcript_turns": 0,
                    "unresolved": ["UNIT_UNKNOWN"],
                    "items": []}),
        encoding="utf-8")
    seen: dict[str, Any] = {}

    def capture(client: Any, bundle: dict[str, Any],
                work_dir: Path,
                ) -> tuple[dict[str, Any], Any]:
        seen.update(bundle)
        record = {"model":
                  "meta-models/muse-spark-open-1026",
                  "provider": "baseten",
                  "prompt_version": "v1.0",
                  "request_id": "chat_rev1",
                  "finish_reason": "stop", "usage": {},
                  "parsed": {
                      "title": "Cinnamon Rolls",
                      "changes_summary": "ok",
                      "ingredients": [
                          {"name": "sugar", "quantity": 100.0,
                           "unit": "g"},
                          {"name": "milk", "quantity": 240.0,
                           "unit": "g"}],
                      "method": [{"number": 1,
                                  "instruction": "Mix."}],
                      "notes": ["n"],
                  }}
        (work_dir / "revision_raw.json").write_text(
            json.dumps(record), encoding="utf-8")
        return record, None

    monkeypatch.setattr(app_module, "run_revision_stage",
                        capture)
    monkeypatch.setattr(app_module, "BasetenClient",
                        lambda: object())
    client = TestClient(build_app(storage_dir=root))
    response = client.post("/api/recipes/demo_v2/revise", json={
        "instructions": "Clean the wording.",
        "confirmed": True})
    assert response.status_code == 201
    assert seen["evidence"]["merged"][
        "readings_total"] == 5
    assert seen["chef_instructions"] == "Clean the wording."
