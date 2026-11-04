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


def _snapshot() -> dict[str, Any]:
    return {
        "snapshot_id": "r1",
        "title": "Roles XL",
        "language": "es",
        "recipe_status": "draft",
        "ingredients": [
            {"name": "flour", "quantity": 700.0, "unit": "g",
             "quantity_status": "seen_video",
             "quantity_basis": "Display 700.",
             "evidence": []},
            {"name": "salt", "quantity": None, "unit": None,
             "quantity_status": "ai_estimated",
             "quantity_basis": "Estimate.",
             "evidence": []},
        ],
        "method": [
            {"number": 1, "instruction": "Mix."},
            {"number": 2, "instruction": "Bake."},
        ],
        "limitations": ["Confirm salt."],
        "improvement_suggestions": [{
            "id": "flour-amount", "label": "Flour amount",
            "hint": "Grams of flour.", "placeholder": "700"}],
        "open_questions": [],
        "technical_limitations": [],
    }


def _revised(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "title": "Roles XL v2",
        "ingredients": [
            {"name": "flour", "quantity": 700.0, "unit": "g"},
            {"name": "salt", "quantity": 3.0, "unit": "g"},
        ],
        "method": [
            {"number": 1, "instruction": "Mix well."},
            {"number": 2, "instruction": "Bake."},
        ],
        "notes": ["Salt set per chef."],
        "changes_summary": "Set salt to 3 g.",
    }
    base.update(overrides)
    return base


def _client_with_recipe(tmp_path: Path,
                        monkeypatch: pytest.MonkeyPatch,
                        revised: dict[str, Any] | None = None,
                        failure: Exception | None = None):
    root = tmp_path / "runs_api"
    app = build_app(storage_dir=root)
    create_record(root, "Roles XL", "es",
                  source_run_id=None, recipe_id="r1")
    (root / "recipes" / "r1" / "snapshot.json").write_text(
        json.dumps(_snapshot()), encoding="utf-8")
    calls: dict[str, Any] = {}

    def fake_revision(client: Any, bundle: dict[str, Any],
                      out_dir: Path):
        calls["bundle"] = bundle
        calls["count"] = calls.get("count", 0) + 1
        if failure is not None:
            raise failure
        return ({"provider": "baseten", "model": "mm",
                 "request_id": "chat_rev1",
                 "parsed": revised}, None)

    monkeypatch.setattr(
        app_module, "run_revision_stage", fake_revision)
    monkeypatch.setenv("BASETEN_API_KEY", "test-key")
    return TestClient(app), root, calls


def _submission(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "instructions": "",
        "suggestions": [{
            "id": "flour-amount", "selected": True,
            "value": "750"}],
        "allow_ingredient_changes": True,
        "confirmed": True,
    }
    body.update(overrides)
    return body


def test_revise_accepts_valued_suggestions_without_text(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Chef corrections alone satisfy revision validation."""
    client, root, calls = _client_with_recipe(
        tmp_path, monkeypatch, _revised())
    before = (root / "recipes" / "r1" / "snapshot.json"
              ).read_bytes()
    response = client.post("/api/recipes/r1/revise",
                           json=_submission())
    assert response.status_code == 201
    assert calls["count"] == 1
    bundle = calls["bundle"]
    assert bundle["selected_suggestions"] == [{
        "id": "flour-amount", "value": "750"}]
    assert bundle["allow_ingredient_changes"] is True
    payload = response.json()
    assert payload["recipe_id"] == "r1_r1"
    assert payload["parent_version"] == "r1"
    assert payload["request_id"] == "chat_rev1"
    assert (root / "recipes" / "r1_r1" / "snapshot.json"
            ).is_file()
    assert (root / "recipes" / "r1_r1" / "record.json"
            ).is_file()
    assert payload["recipe"]["recipe_status"] == "draft"
    # Parent untouched.
    assert (root / "recipes" / "r1" / "snapshot.json"
            ).read_bytes() == before
    assert not (root / "recipes" / "r1" / "revising.lock"
                ).exists()
    ledger = json.loads(
        (root / "phase2c" / "ledger.json").read_text(
            encoding="utf-8"))
    assert any("revision/r1/r1_r1" in json.dumps(entry)
               for entry in ledger.get("spent") or [])


def test_revise_accepts_text_without_selections(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client, _, calls = _client_with_recipe(
        tmp_path, monkeypatch, _revised())
    response = client.post(
        "/api/recipes/r1/revise",
        json=_submission(instructions="Clarify the steps.",
                         suggestions=[]))
    assert response.status_code == 201
    assert calls["count"] == 1


def test_revise_rejects_empty_form(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client, root, calls = _client_with_recipe(
        tmp_path, monkeypatch, _revised())
    response = client.post(
        "/api/recipes/r1/revise",
        json=_submission(instructions="  ", suggestions=[]))
    assert response.status_code == 422
    assert "at least" in response.json()["detail"]
    assert calls.get("count", 0) == 0
    assert not (root / "recipes" / "r1_r1").exists()
    # Selected-but-empty is not a valid correction either.
    response = client.post(
        "/api/recipes/r1/revise",
        json=_submission(
            instructions="",
            suggestions=[{"id": "flour-amount",
                          "selected": True, "value": "  "}]))
    assert response.status_code == 422
    assert calls.get("count", 0) == 0


def test_revise_rejects_unconfirmed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client, _, calls = _client_with_recipe(
        tmp_path, monkeypatch, _revised())
    response = client.post("/api/recipes/r1/revise",
                           json=_submission(confirmed=False))
    assert response.status_code == 422
    assert calls.get("count", 0) == 0


def test_revise_failure_preserves_original(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from bakery_intelligence.api.multimodal import (
        MultimodalError,
    )

    client, root, calls = _client_with_recipe(
        tmp_path, monkeypatch, _revised(),
        failure=MultimodalError("Revision returned empty "
                                "output (finish=stop)."))
    before = (root / "recipes" / "r1" / "snapshot.json"
              ).read_bytes()
    response = client.post("/api/recipes/r1/revise",
                           json=_submission())
    assert response.status_code == 502
    assert calls["count"] == 1
    assert (root / "recipes" / "r1" / "snapshot.json"
            ).read_bytes() == before
    version = root / "recipes" / "r1_r1"
    assert not (version / "snapshot.json").exists()
    assert not (version / "record.json").exists()
    assert not (root / "recipes" / "r1" / "revising.lock"
                ).exists()


def test_revise_lock_blocks_duplicates(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from bakery_intelligence.api.jobs import utc_now_iso

    client, root, calls = _client_with_recipe(
        tmp_path, monkeypatch, _revised())
    (root / "recipes" / "r1" / "revising.lock").write_text(
        json.dumps({"started_at": utc_now_iso()}),
        encoding="utf-8")
    response = client.post("/api/recipes/r1/revise",
                           json=_submission())
    assert response.status_code == 409
    assert calls.get("count", 0) == 0


def test_revise_stale_lock_allows_retry(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client, root, calls = _client_with_recipe(
        tmp_path, monkeypatch, _revised())
    (root / "recipes" / "r1" / "revising.lock").write_text(
        json.dumps({"started_at": "2020-01-01T00:00:00+00:00"}),
        encoding="utf-8")
    response = client.post("/api/recipes/r1/revise",
                           json=_submission())
    assert response.status_code == 201
    assert calls["count"] == 1


def test_revise_keeps_provenance_of_unchanged_rows(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Only rows the model changed lose seen_video status."""
    client, root, _ = _client_with_recipe(
        tmp_path, monkeypatch, _revised())
    response = client.post("/api/recipes/r1/revise",
                           json=_submission())
    assert response.status_code == 201
    rows = response.json()["recipe"]["ingredients"]
    assert rows[0]["quantity_status"] == "seen_video"
    assert rows[1]["quantity_status"] == "ai_estimated"


def test_revise_rejects_unauthorized_ingredient_change(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client, root, calls = _client_with_recipe(
        tmp_path, monkeypatch, _revised())
    response = client.post(
        "/api/recipes/r1/revise",
        json=_submission(allow_ingredient_changes=False))
    assert response.status_code == 422
    assert "without authorization" in response.json()["detail"]
    assert calls["count"] == 1
    assert not (root / "recipes" / "r1_r1" / "snapshot.json"
                ).exists()


def test_next_version_id_reuses_diagnostics_dir(
        tmp_path: Path):
    root = tmp_path / "runs_api"
    corpse = root / "recipes" / "r1_r1"
    corpse.mkdir(parents=True)
    (corpse / "revision_raw_failed.json").write_text(
        "{}", encoding="utf-8")
    assert app_module._next_version_id(  # noqa: SLF001
        root, "r1") == "r1_r1"
    (corpse / "record.json").write_text("{}", encoding="utf-8")
    assert app_module._next_version_id(  # noqa: SLF001
        root, "r1") == "r1_r2"


def test_validate_revision_protects_rows_and_steps():
    recipe = {"ingredients": [
        {"name": "flour", "quantity": 700.0, "unit": "g"}],
        "method": [{"number": 1,
                    "instruction": "Mix."}]}
    good = {"title": "T", "ingredients": [
        {"name": "flour", "quantity": 700.0, "unit": "g"}],
        "method": [{"number": 1, "instruction": "Mix well."}],
        "notes": ["n"], "changes_summary": "c"}
    assert revision_mod.validate_revision(
        recipe, good, False) == good
    bad = json.loads(json.dumps(good))
    bad["ingredients"][0]["quantity"] = 750.0
    with pytest.raises(revision_mod.RevisionError,
                       match="without authorization"):
        revision_mod.validate_revision(recipe, bad, False)
    assert revision_mod.validate_revision(
        recipe, bad, True) == bad
    bad_steps = json.loads(json.dumps(good))
    bad_steps["method"] = []
    with pytest.raises(revision_mod.RevisionError,
                       match="step count"):
        revision_mod.validate_revision(recipe, bad_steps, True)


def test_assemble_version_relabels_only_changed_rows():
    recipe = {"ingredients": [
        {"name": "flour", "quantity": 700.0, "unit": "g",
         "quantity_status": "seen_video",
         "quantity_basis": "Display 700."}],
        "method": [{"number": 1, "instruction": "Mix."}]}
    revised = {"title": "T", "ingredients": [
        {"name": "flour", "quantity": 750.0, "unit": "g"}],
        "method": [{"number": 1, "instruction": "Mix well."}],
        "notes": ["n"], "changes_summary": "c"}
    meta = {"recipe_id": "r1_r1", "parent_id": "r1",
            "generated_at": "2026-01-01T00:00:00Z",
            "language": "es", "instructions": "i",
            "suggestions": [],
            "allow_ingredient_changes": True}
    snapshot = revision_mod.assemble_version(
        recipe, revised, meta)
    row = snapshot["ingredients"][0]
    assert row["quantity"] == 750.0
    assert row["quantity_status"] == "ai_estimated"
    assert snapshot["recipe_status"] == "draft"
    assert snapshot["revision"]["parent_version"] == "r1"
    assert snapshot["revision"]["request_id"] is None
    same = json.loads(json.dumps(revised))
    same["ingredients"][0]["quantity"] = 700.0
    snapshot = revision_mod.assemble_version(
        recipe, same, meta)
    assert snapshot["ingredients"][0][
        "quantity_status"] == "seen_video"


def test_match_chef_suggestion_cases():
    match = revision_mod.match_chef_suggestion
    suggestions = [
        {"id": "azucar_g", "value": "50"},
        {"id": "vainilla_g", "value": "1 capful"},
        {"id": "flour-amount", "value": 750},
    ]
    labels = {"azucar_g": "Azúcar para la masa",
              "vainilla_g": "Esencia de vainilla",
              "flour-amount": "Flour amount"}
    found = match("Azúcar", 50.0, suggestions, labels)
    assert found is not None and found["id"] == "azucar_g"
    found = match("Esencia de vainilla", 1,
                  suggestions, labels)
    assert found is not None and found["id"] == "vainilla_g"
    found = match("Flour", 750, suggestions, labels)
    assert found is not None
    # Wrong value: same name, no support.
    assert match("Azúcar", 999.0,
                 suggestions, labels) is None
    # Wrong name: same value, no support.
    assert match("Truffle", 50.0,
                 suggestions, labels) is None
    # One correction justifies one row.
    used = {"azucar_g"}
    assert match("Azúcar", 50.0,
                 suggestions, labels, used) is None
    # Degenerate inputs never match.
    assert match("", 50.0, suggestions, labels) is None
    assert match("Azúcar", None,
                 suggestions, labels) is None
    assert match("Azúcar", 50.0, None, labels) is None
    assert match(
        "Azúcar", 50.0,
        [{"id": "azucar_g", "value": "a lot"}],
        labels) is None


def _addition_recipe():
    return {
        "ingredients": [
            {"name": "flour", "quantity": 700.0, "unit": "g",
             "quantity_status": "seen_video",
             "quantity_basis": "Display 700."}],
        "method": [{"number": 1, "instruction": "Mix."}],
        "improvement_suggestions": [
            {"id": "azucar_g",
             "label": "Azúcar para la masa"}],
    }


def _addition_revised(extra):
    return {
        "title": "T",
        "ingredients": [
            {"name": "flour", "quantity": 700.0, "unit": "g"},
            extra],
        "method": [{"number": 1, "instruction": "Mix well."}],
        "notes": ["n"], "changes_summary": "c"}


def test_validate_revision_accepts_supported_addition():
    recipe = _addition_recipe()
    revised = _addition_revised(
        {"name": "Azúcar", "quantity": 50.0, "unit": "g"})
    out = revision_mod.validate_revision(
        recipe, revised, True,
        [{"id": "azucar_g", "value": "50"}])
    assert len(out["ingredients"]) == 2


def test_validate_revision_rejects_unsupported_addition():
    recipe = _addition_recipe()
    # Invented row: nothing selected supports it.
    revised = _addition_revised(
        {"name": "Truffle", "quantity": 5.0, "unit": "g"})
    with pytest.raises(revision_mod.RevisionError,
                       match="unsupported addition"):
        revision_mod.validate_revision(
            recipe, revised, True,
            [{"id": "azucar_g", "value": "50"}])
    # Unknown suggestion id cannot launder an addition.
    revised = _addition_revised(
        {"name": "Truffle", "quantity": 5.0, "unit": "g"})
    with pytest.raises(revision_mod.RevisionError,
                       match="unsupported addition"):
        revision_mod.validate_revision(
            recipe, revised, True,
            [{"id": "truffle_x", "value": "5"}])
    # Additions need authorization.
    revised = _addition_revised(
        {"name": "Azúcar", "quantity": 50.0, "unit": "g"})
    with pytest.raises(revision_mod.RevisionError,
                       match="one ingredient row per input"):
        revision_mod.validate_revision(
            recipe, revised, False,
            [{"id": "azucar_g", "value": "50"}])


def test_validate_revision_rejects_duplicate_support():
    recipe = _addition_recipe()
    revised = _addition_revised(
        {"name": "Azúcar", "quantity": 50.0, "unit": "g"})
    revised["ingredients"].append(
        {"name": "Azúcar refinada", "quantity": 50.0,
         "unit": "g"})
    with pytest.raises(revision_mod.RevisionError,
                       match="unsupported addition"):
        revision_mod.validate_revision(
            recipe, revised, True,
            [{"id": "azucar_g", "value": "50"}])


def test_validate_revision_rejects_deletion_with_allow():
    recipe = _addition_recipe()
    recipe["ingredients"].append(
        {"name": "salt", "quantity": 3.0, "unit": "g"})
    revised = {
        "title": "T",
        "ingredients": [
            {"name": "flour", "quantity": 700.0, "unit": "g"}],
        "method": [{"number": 1, "instruction": "Mix."}],
        "notes": ["n"], "changes_summary": "c"}
    with pytest.raises(revision_mod.RevisionError,
                       match="removed ingredient rows"):
        revision_mod.validate_revision(recipe, revised, True, [])


def test_assemble_version_appends_supported_rows():
    recipe = _addition_recipe()
    revised = _addition_revised(
        {"name": "Azúcar", "quantity": 50.0, "unit": "g"})
    meta = {"recipe_id": "r1_r1", "parent_id": "r1",
            "generated_at": "2026-01-01T00:00:00Z",
            "language": "es", "instructions": "",
            "suggestions": [{"id": "azucar_g",
                             "value": "50"}],
            "allow_ingredient_changes": True}
    snapshot = revision_mod.assemble_version(
        recipe, revised, meta)
    rows = snapshot["ingredients"]
    assert len(rows) == 2
    assert rows[0]["quantity_status"] == "seen_video"
    assert rows[1]["quantity_status"] == "ai_estimated"
    assert "azucar_g" in rows[1]["quantity_basis"]
    assert rows[1]["evidence"] == []
    assert snapshot["revision"]["selected_suggestions"] == [
        {"id": "azucar_g", "value": "50"}]


def test_assemble_version_cites_chef_basis_for_changes():
    recipe = _addition_recipe()
    revised = {
        "title": "T",
        "ingredients": [
            {"name": "flour", "quantity": 750.0, "unit": "g"}],
        "method": [{"number": 1, "instruction": "Mix."}],
        "notes": ["n"], "changes_summary": "c"}
    meta = {"recipe_id": "r1_r1", "parent_id": "r1",
            "generated_at": "2026-01-01T00:00:00Z",
            "language": "es", "instructions": "",
            "suggestions": [{"id": "flour-amount",
                             "value": "750"}],
            "allow_ingredient_changes": True}
    recipe["improvement_suggestions"].append(
        {"id": "flour-amount", "label": "Flour amount"})
    snapshot = revision_mod.assemble_version(
        recipe, revised, meta)
    row = snapshot["ingredients"][0]
    assert row["quantity"] == 750.0
    assert row["quantity_status"] == "ai_estimated"
    assert "flour-amount" in row["quantity_basis"]


def test_assemble_version_pure_rename_keeps_status():
    """A rename without quantity change keeps seen_video."""
    recipe = _addition_recipe()
    revised = {
        "title": "T",
        "ingredients": [
            {"name": "Bread flour", "quantity": 700.0,
             "unit": "g"}],
        "method": [{"number": 1, "instruction": "Mix."}],
        "notes": ["n"], "changes_summary": "c"}
    meta = {"recipe_id": "r1_r1", "parent_id": "r1",
            "generated_at": "2026-01-01T00:00:00Z",
            "language": "es", "instructions": "Rename.",
            "suggestions": [],
            "allow_ingredient_changes": True}
    snapshot = revision_mod.assemble_version(
        recipe, revised, meta)
    row = snapshot["ingredients"][0]
    assert row["name"] == "Bread flour"
    assert row["quantity_status"] == "seen_video"
    assert row["quantity_basis"] == "Display 700."


def test_revise_route_persists_supported_addition(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """End to end: a justified extra row becomes a version."""
    client, root, calls = _client_with_recipe(
        tmp_path, monkeypatch, _revised())
    parsed = _revised()
    parsed["ingredients"].append(
        {"name": "sugar", "quantity": 50.0, "unit": "g"})
    (root / "recipes" / "r1" / "snapshot.json").write_text(
        json.dumps(_snapshot_with_sugar()), encoding="utf-8")

    def fake_revision(client: Any, bundle: dict[str, Any],
                      out_dir: Path):
        calls["count"] = calls.get("count", 0) + 1
        return ({"provider": "baseten", "model": "mm",
                 "request_id": "chat_rev2",
                 "parsed": parsed}, None)

    monkeypatch.setattr(
        app_module, "run_revision_stage", fake_revision)
    body = _submission(suggestions=[
        {"id": "sugar_g", "selected": True, "value": "50"}])
    response = client.post("/api/recipes/r1/revise", json=body)
    assert response.status_code == 201
    rows = response.json()["recipe"]["ingredients"]
    assert len(rows) == 3
    assert rows[2]["name"] == "sugar"
    assert rows[2]["quantity_status"] == "ai_estimated"
    assert "sugar_g" in rows[2]["quantity_basis"]


def _snapshot_with_sugar() -> dict[str, Any]:
    snapshot = _snapshot()
    snapshot["improvement_suggestions"].append(
        {"id": "sugar_g", "label": "Sugar for dough",
         "hint": "Grams of sugar.", "placeholder": "50"})
    return snapshot


def _correction_parsed() -> dict[str, Any]:
    return {
        "title": "T",
        "ingredients": [
            {"name": "Whole milk", "quantity": 238.0,
             "unit": "g"},
            {"name": "Sugar", "quantity": 50.0, "unit": "g"}],
        "method": [
            {"number": 1,
             "instruction": "Pour 238 g of milk."},
            {"number": 2, "instruction": "Add sugar."}],
        "notes": ["Use quality milk."],
        "changes_summary": "Renamed milk."}


def test_apply_corrections_restores_names_and_prose():
    """Audited renames and text edits apply exactly once."""
    parsed = _correction_parsed()
    before = copy.deepcopy(parsed)
    corrected, applied = \
        revision_mod.apply_deterministic_corrections(
            parsed,
            [("Whole milk",
              "Opaque white liquid (milk-like)")],
            [("method", 1, "238 g of milk",
              "238 g of opaque white liquid (milk-like)"),
             ("notes", 0, "quality milk",
              "quality ingredients"),
             ("changes_summary", None, "Renamed milk",
              "Restored observed names")])
    assert parsed == before
    assert corrected["ingredients"][0]["name"] == \
        "Opaque white liquid (milk-like)"
    assert corrected["ingredients"][1]["name"] == "Sugar"
    assert "opaque white liquid" in \
        corrected["method"][0]["instruction"]
    assert corrected["method"][1]["instruction"] == \
        "Add sugar."
    assert corrected["notes"] == ["Use quality ingredients."]
    assert corrected["changes_summary"] == \
        "Restored observed names."
    assert len(applied) == 4


def test_apply_corrections_rejects_ambiguous_match():
    """Zero or duplicate matches abort instead of half-applying."""
    parsed = _correction_parsed()
    with pytest.raises(revision_mod.RevisionError,
                       match="matches 0 rows"):
        revision_mod.apply_deterministic_corrections(
            parsed, [("Missing", "Other")], [])
    parsed["ingredients"].append(
        {"name": "Whole milk", "quantity": 10.0, "unit": "g"})
    with pytest.raises(revision_mod.RevisionError,
                       match="matches 2 rows"):
        revision_mod.apply_deterministic_corrections(
            parsed, [("Whole milk", "Other")], [])


def _substitution_parsed() -> dict[str, Any]:
    """Free-text substitution: milk slot becomes dark chocolate."""
    return {
        "title": "Cinnamon Rolls",
        "ingredients": [
            {"name": "sugar", "quantity": 100.0, "unit": "g"},
            {"name": "dark chocolate chips", "quantity": 80.0,
             "unit": "g"}],
        "method": [{"number": 1,
                    "instruction": "Mix with dark chocolate."}],
        "notes": ["n1"],
        "changes_summary": "Substituted milk with chocolate."}


def test_free_text_substitution_replaces_in_place():
    """Same-count substitution needs no selected suggestion."""
    recipe = copy.deepcopy(SNAPSHOT)
    revised = revision_mod.validate_revision(
        recipe, _substitution_parsed(), True, [])
    snapshot = revision_mod.assemble_version(
        recipe, revised, {
            "recipe_id": "demo_v2_r1",
            "parent_id": "demo_v2",
            "generated_at": "2026-01-01T00:00:00Z",
            "language": "en", "instructions": "Replace the "
            "milk with dark chocolate chips.",
            "suggestions": [],
            "allow_ingredient_changes": True})
    rows = snapshot["ingredients"]
    assert len(rows) == 2
    assert rows[1]["name"] == "dark chocolate chips"
    assert rows[1]["quantity"] == 80.0
    assert rows[1]["quantity_status"] == "ai_estimated"
    assert "chef authorization" in \
        rows[1]["quantity_basis"].lower() \
        or "chef correction" in rows[1]["quantity_basis"]
    assert "Displayed" not in rows[1]["quantity_basis"]
    assert rows[0] == SNAPSHOT["ingredients"][0]
    assert recipe == SNAPSHOT


def test_free_text_substitution_route_mocked(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Instructions-only revision persists a replaced row."""
    root = _seed(tmp_path / "runs_api")
    create_record(root, "Cinnamon Rolls", "en",
                  recipe_id="demo_v2")
    client = _patched_client(
        monkeypatch, root, _substitution_parsed())
    response = client.post("/api/recipes/demo_v2/revise",
                           json={
                               "instructions": "Replace the "
                               "milk with dark chocolate chips.",
                               "suggestions": [],
                               "allow_ingredient_changes": True,
                               "confirmed": True})
    assert response.status_code == 201
    rows = response.json()["recipe"]["ingredients"]
    assert len(rows) == 2
    assert rows[1]["name"] == "dark chocolate chips"
    assert rows[0]["name"] == "sugar"
    parent = json.loads((root / "recipes" / "demo_v2"
                         / "snapshot.json").read_text(
                             encoding="utf-8"))
    assert parent == SNAPSHOT


def test_free_text_append_instead_of_replace_rejected():
    """Appending a duplicate without a suggestion fails."""
    revised = _substitution_parsed()
    revised["ingredients"] = [
        {"name": "sugar", "quantity": 100.0, "unit": "g"},
        {"name": "milk", "quantity": 240.0, "unit": "g"},
        {"name": "dark chocolate chips", "quantity": 80.0,
         "unit": "g"}]
    with pytest.raises(revision_mod.RevisionError,
                       match="unsupported addition"):
        revision_mod.validate_revision(
            copy.deepcopy(SNAPSHOT), revised, True, [])


def test_apply_corrections_rejects_bad_text_target():
    """Out-of-range or missing spans abort the correction."""
    parsed = _correction_parsed()
    with pytest.raises(revision_mod.RevisionError,
                       match="out of range"):
        revision_mod.apply_deterministic_corrections(
            parsed, [], [("method", 9, "x", "y")])
    with pytest.raises(revision_mod.RevisionError,
                       match="matches 0 text spans"):
        revision_mod.apply_deterministic_corrections(
            parsed, [], [("notes", 0, "absent", "y")])
    with pytest.raises(revision_mod.RevisionError,
                       match="Unknown correction target"):
        revision_mod.apply_deterministic_corrections(
            parsed, [], [("title", None, "T", "U")])
