"""Offline tests for automatic first-recipe proposals.

Synthetic run artifacts in tmp storage plus one read-only replay
against the persisted run_52a8b5cd5f24 evidence. Provider calls
are patched; no network, no inference, no billable cost.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from bakery_intelligence.api import app as app_module
from bakery_intelligence.api.app import build_app
from bakery_intelligence.api.jobs import (
    WORKFLOW_STORAGE_SMOKE,
    utc_now_iso,
)
from bakery_intelligence.api.recipe_editor import (
    create_record,
    list_recipes,
    merged_recipe,
)
from bakery_intelligence.phase2c.agents import AgentResult
from bakery_intelligence.recipe_export import proposal as proposal_mod
from bakery_intelligence.recipe_export.labels import row_source

REPO_ROOT = Path(__file__).resolve().parent.parent
REPLAY_RUN = (REPO_ROOT / "runs_api" / "runs" / "run_52a8b5cd5f24")


def _seed_run(run_dir: Path) -> None:
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "pipeline_summary.json").write_text(json.dumps({
        "run_id": run_dir.name,
        "video": {"duration_seconds": 51.0, "frame_count": 51},
    }), encoding="utf-8")
    (artifacts / "measurements.json").write_text(json.dumps({
        "measurements": [
            {"frame_index": 100, "timestamp_seconds": 2.0,
             "value": 100.0, "unit_observed": None,
             "value_status": "ok"},
            {"frame_index": 200, "timestamp_seconds": 4.0,
             "value": None, "unit_observed": None,
             "value_status": "invalid"},
            {"frame_index": 300, "timestamp_seconds": 6.0,
             "value": 250.0, "unit_observed": "g",
             "value_status": "ok"},
        ],
    }), encoding="utf-8")
    (artifacts / "recipe_draft.json").write_text(json.dumps({
        "steps": [{"description": "Scale showed readings."}],
        "unresolved_questions": [{"code": "INGREDIENT_UNKNOWN"}],
    }), encoding="utf-8")
    (artifacts / "weighing_event.json").write_text(json.dumps({
        "tare_observed": True, "review_notes": "tare seen",
    }), encoding="utf-8")
    obs_dir = run_dir / "work" / "firstpass" / "observations"
    obs_dir.mkdir(parents=True)
    (obs_dir / "batch_0.json").write_text(json.dumps([
        {"frame_index": 100, "timestamp_seconds": 2.0,
         "model_fields": {
             "visible_objects": ["kitchen scale", "pasta pack"],
             "container_description": "steel bowl",
             "tare_action_visible": True}},
        {"frame_index": 300, "timestamp_seconds": 6.0,
         "model_fields": {
             "visible_objects": ["pasta pack", "wooden spoon"],
             "container_description": "steel bowl",
             "tare_action_visible": False}},
    ]), encoding="utf-8")


def _parsed(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "meaningful": True,
        "reason": "Readings plus visible packaging support a draft.",
        "title": "Pasta Test",
        "ingredients": [
            {"name": "pasta", "quantity": 100.0, "unit": "g",
             "provenance": "seen_video",
             "basis": "Observed 100 on the display at t=2s."},
            {"name": "salt", "quantity": 3.0, "unit": "g",
             "provenance": "ai_estimated",
             "basis": "Culinary estimate for the pot."},
        ],
        "method": [
            {"number": 1, "instruction": "Boil water."},
            {"number": 2, "instruction": "Add pasta."},
        ],
        "notes": ["Confirm salt to taste."],
        "suggestions": [{
            "id": "water-amount", "label": "Add water amount",
            "hint": "Liters for the pot.",
            "placeholder": "4 liters"}],
        "open_questions": ["q1"],
    }
    base.update(overrides)
    return base


def test_bundle_builder_filters_and_caps(tmp_path: Path):
    run_dir = tmp_path / "runs" / "run_x"
    _seed_run(run_dir)
    bundle = proposal_mod.build_evidence_bundle(
        run_dir, "Roles XL", "es")
    assert bundle["recipe_name"] == "Roles XL"
    assert bundle["language"] == "es"
    assert [r["value"] for r in bundle["readings"]] == [100.0, 250.0]
    labels = [e["object"] for e in bundle["visible_objects"]]
    assert "pasta pack" in labels
    assert "kitchen scale" not in labels
    assert bundle["containers"][0]["description"] == "steel bowl"
    assert bundle["tare_events"] == [2.0]
    assert bundle["weighing"]["tare_observed"] is True
    assert bundle["unresolved"] == ["INGREDIENT_UNKNOWN"]


def test_bundle_builder_missing_run(tmp_path: Path):
    with pytest.raises(proposal_mod.ProposalError):
        proposal_mod.build_evidence_bundle(
            tmp_path / "runs" / "nope", "X", "en")


def test_validate_proposal_accepts_grounded(tmp_path: Path):
    run_dir = tmp_path / "runs" / "run_x"
    _seed_run(run_dir)
    bundle = proposal_mod.build_evidence_bundle(run_dir, "T", "en")
    parsed = proposal_mod.validate_proposal(bundle, _parsed())
    assert parsed["title"] == "Pasta Test"


def test_validate_proposal_rejects_ungrounded_seen(tmp_path: Path):
    run_dir = tmp_path / "runs" / "run_x"
    _seed_run(run_dir)
    bundle = proposal_mod.build_evidence_bundle(run_dir, "T", "en")
    bad = _parsed()
    bad["ingredients"][0]["quantity"] = 999.0
    with pytest.raises(proposal_mod.ProposalError):
        proposal_mod.validate_proposal(bundle, bad)


def test_validate_proposal_shape_rules(tmp_path: Path):
    run_dir = tmp_path / "runs" / "run_x"
    _seed_run(run_dir)
    bundle = proposal_mod.build_evidence_bundle(run_dir, "T", "en")
    bad = _parsed()
    bad["method"][1]["number"] = 3
    with pytest.raises(proposal_mod.ProposalError):
        proposal_mod.validate_proposal(bundle, bad)
    bad = _parsed()
    bad["suggestions"][0]["id"] = "Not A Slug!"
    with pytest.raises(proposal_mod.ProposalError):
        proposal_mod.validate_proposal(bundle, bad)
    bad = _parsed()
    bad["ingredients"].append(dict(bad["ingredients"][0]))
    with pytest.raises(proposal_mod.ProposalError):
        proposal_mod.validate_proposal(bundle, bad)


def test_validate_proposal_accepts_snake_case_slug(tmp_path: Path):
    run_dir = tmp_path / "runs" / "run_x"
    _seed_run(run_dir)
    bundle = proposal_mod.build_evidence_bundle(run_dir, "T", "en")
    # Live fusion emits snake_case ids; they are opaque React
    # keys and selection-map keys, so underscores are safe.
    parsed = proposal_mod.validate_proposal(bundle, _parsed(
        suggestions=[{
            "id": "water_amount", "label": "Add water amount",
            "hint": "Liters for the pot.",
            "placeholder": "4 liters"}]))
    assert parsed["suggestions"][0]["id"] == "water_amount"


def test_validate_non_meaningful(tmp_path: Path):
    run_dir = tmp_path / "runs" / "run_x"
    _seed_run(run_dir)
    bundle = proposal_mod.build_evidence_bundle(run_dir, "T", "en")
    parsed = proposal_mod.validate_proposal(bundle, _parsed(
        meaningful=False, ingredients=[], method=[]))
    assert parsed["meaningful"] is False
    with pytest.raises(proposal_mod.ProposalError):
        proposal_mod.validate_proposal(
            bundle, _parsed(meaningful=False))


def test_assembled_snapshot_merges_and_labels(tmp_path: Path):
    run_dir = tmp_path / "runs" / "run_x"
    _seed_run(run_dir)
    bundle = proposal_mod.build_evidence_bundle(run_dir, "T", "en")
    parsed = proposal_mod.validate_proposal(bundle, _parsed())
    snapshot = proposal_mod.assemble_snapshot(bundle, parsed, {
        "recipe_id": "r1", "run_id": "run_x",
        "generated_at": "2026-01-01T00:00:00Z",
        "request_id": "req_1", "actual_usd": 0.01})
    root = tmp_path / "runs_api"
    recipe_dir = root / "recipes" / "r1"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "snapshot.json").write_text(
        json.dumps(snapshot), encoding="utf-8")
    create_record(root, "Pasta Test", "en",
                  source_run_id="run_x", recipe_id="r1")
    merged = merged_recipe(root, "r1")
    assert merged["ingredients"][0]["quantity"] == 100.0
    assert row_source(merged["ingredients"][0], "en") == \
        "Seen in video"
    assert row_source(merged["ingredients"][1], "en") == \
        "Suggested by AI"
    assert merged["improvement_suggestions"][0]["id"] == \
        "water-amount"
    assert merged["language"] == "en"
    assert merged["recipe_status"] == "draft"


def _client_with_job(tmp_path: Path, status: str,
                     monkeypatch: pytest.MonkeyPatch,
                     parsed: dict[str, Any] | None,
                     ) -> tuple[TestClient, str, str, dict[str, int]]:
    root = tmp_path / "runs_api"
    app = build_app(storage_dir=root)
    service = app.state.jobs
    # storage_smoke needs no grant; the build endpoint only
    # reads job status, never the workflow grant.
    job = service.submit(WORKFLOW_STORAGE_SMOKE)
    run_id = job.run_id
    _seed_run(root / "runs" / run_id)
    if status == "completed":
        service.store.claim_next("test", utc_now_iso())
        assert service.store.complete(
            run_id, "test", utc_now_iso(), None, {})
    elif status == "failed":
        service.store.fail(run_id, None, utc_now_iso(), "boom")
    create_record(root, "Pasta Test", "en",
                  source_run_id=run_id, recipe_id="r1")
    calls = {"count": 0}

    def fake_run(client: Any, ledger: Any, prompt: Any,
                 agent: str, clip_id: str,
                 bundle: dict[str, Any], work_dir: Path,
                 filename: str, estimate: float) -> AgentResult:
        calls["count"] += 1
        if parsed is None:
            return AgentResult(agent=agent, clip_id=clip_id,
                               status="failed", error="provider down",
                               estimate_usd=estimate)
        (work_dir / filename).write_text(
            json.dumps({"request_id": "req_1", "usage": {},
                        "parsed": parsed}), encoding="utf-8")
        return AgentResult(agent=agent, clip_id=clip_id,
                           status="ok", request_id="req_1",
                           estimate_usd=estimate,
                           actual_usd=0.01, payload=parsed)

    monkeypatch.setattr(app_module.agents, "run_text_agent",
                        fake_run)
    return TestClient(app), run_id, "r1", calls


def test_build_endpoint_success_and_idempotent(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client, _, recipe_id, calls = _client_with_job(
        tmp_path, "completed", monkeypatch, _parsed())
    first = client.post(f"/api/recipes/{recipe_id}/build")
    assert first.status_code == 200
    assert first.json()["status"] == "ready"
    assert first.json()["recipe"]["ingredients"][0][
        "quantity"] == 100.0
    assert calls["count"] == 1
    root = tmp_path / "runs_api"
    assert (root / "recipes" / recipe_id / "manifest.json"
            ).is_file()
    assert (root / "recipes" / recipe_id / "proposal_raw.json"
            ).is_file()
    second = client.post(f"/api/recipes/{recipe_id}/build")
    assert second.status_code == 200
    assert calls["count"] == 1
    cards = {c["recipe_id"]: c for c in list_recipes(root)}
    assert cards[recipe_id]["status"] == "draft"
    assert cards[recipe_id]["ingredients"] == 2


def test_build_endpoint_requires_completed_job(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client, _, recipe_id, calls = _client_with_job(
        tmp_path, "queued", monkeypatch, _parsed())
    response = client.post(f"/api/recipes/{recipe_id}/build")
    assert response.status_code == 409
    assert calls["count"] == 0
    client, _, recipe_id, _ = _client_with_job(
        tmp_path / "b", "failed", monkeypatch, _parsed())
    response = client.post(f"/api/recipes/{recipe_id}/build")
    assert response.status_code == 409
    assert "boom" in response.json()["detail"]


def test_build_endpoint_non_meaningful(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client, _, recipe_id, calls = _client_with_job(
        tmp_path, "completed", monkeypatch,
        _parsed(meaningful=False, ingredients=[], method=[],
                reason="Only dark frames; nothing usable."))
    response = client.post(f"/api/recipes/{recipe_id}/build")
    assert response.status_code == 422
    assert "dark frames" in response.json()["detail"]
    root = tmp_path / "runs_api"
    assert (root / "recipes" / recipe_id / "unavailable.json"
            ).is_file()
    assert not (root / "recipes" / recipe_id / "snapshot.json"
                ).exists()
    cards = {c["recipe_id"]: c for c in list_recipes(root)}
    assert cards[recipe_id]["status"] == "failed"
    assert "dark frames" in cards[recipe_id]["build_error"]


def test_build_endpoint_provider_failure(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client, _, recipe_id, _ = _client_with_job(
        tmp_path, "completed", monkeypatch, None)
    response = client.post(f"/api/recipes/{recipe_id}/build")
    assert response.status_code == 502
    root = tmp_path / "runs_api"
    assert (root / "recipes" / recipe_id / "build_error.json"
            ).is_file()
    cards = {c["recipe_id"]: c for c in list_recipes(root)}
    assert cards[recipe_id]["status"] == "failed"


def test_build_endpoint_busy_lock(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client, _, recipe_id, calls = _client_with_job(
        tmp_path, "completed", monkeypatch, _parsed())
    root = tmp_path / "runs_api"
    (root / "recipes" / recipe_id / "building.lock").write_text(
        json.dumps({"started_at": utc_now_iso()}),
        encoding="utf-8")
    response = client.post(f"/api/recipes/{recipe_id}/build")
    assert response.status_code == 409
    assert calls["count"] == 0


@pytest.mark.skipif(not REPLAY_RUN.is_dir(),
                    reason="run_52a8b5cd5f24 evidence not present")
def test_replay_run52a_bundle_shape():
    """Read-only replay: real pipeline artifacts feed the bundle."""
    bundle = proposal_mod.build_evidence_bundle(
        REPLAY_RUN, "Pasta Replay", "en")
    assert bundle["readings_total"] == 22
    values = [r["value"] for r in bundle["readings"]]
    assert min(values) == 20.0
    assert max(values) == 299.0
    assert bundle["visible_objects"], \
        "expected visible objects in replay evidence"
    assert bundle["weighing"]["tare_observed"] is True
    assert "INGREDIENT_UNKNOWN" in bundle["unresolved"]


def test_bundle_builder_multimodal_empty_when_absent(
        tmp_path: Path):
    run_dir = tmp_path / "runs" / "run_x"
    _seed_run(run_dir)
    bundle = proposal_mod.build_evidence_bundle(
        run_dir, "T", "en")
    assert bundle["multimodal"] == {}


def test_bundle_builder_multimodal_context(tmp_path: Path):
    run_dir = tmp_path / "runs" / "run_x"
    _seed_run(run_dir)
    artifacts = run_dir / "artifacts"
    (artifacts / "multimodal_global.json").write_text(
        json.dumps({"request_id": "resp_g",
                    "result": {"overall_activity": "x"}}),
        encoding="utf-8")
    (artifacts / "multimodal_segment_000.json").write_text(
        json.dumps({"index": 0, "start_seconds": 0.0,
                    "end_seconds": 30.0,
                    "request_id": "chat_0",
                    "result": {"events": []}}),
        encoding="utf-8")
    bundle = proposal_mod.build_evidence_bundle(
        run_dir, "T", "en")
    assert bundle["multimodal"]["global"][
        "request_id"] == "resp_g"
    assert bundle["multimodal"]["segments"][0][
        "request_id"] == "chat_0"


def test_assemble_snapshot_provenance_overrides(
        tmp_path: Path):
    run_dir = tmp_path / "runs" / "run_x"
    _seed_run(run_dir)
    bundle = proposal_mod.build_evidence_bundle(
        run_dir, "T", "en")
    snapshot = proposal_mod.assemble_snapshot(
        bundle, _parsed(), {
            "recipe_id": "r1", "run_id": "run_x",
            "generated_at": "2026-11-03T00:00:00Z",
            "request_id": "chat_f", "actual_usd": None,
            "builder": "multimodal-fusion/1",
            "provider": "baseten",
            "model": "meta-models/muse-spark-open-1026",
            "prompt_version": "v1.2",
        })
    provenance = snapshot["provenance"]
    assert provenance["builder"] == "multimodal-fusion/1"
    assert provenance["proposal_provider"] == "baseten"
    assert provenance["proposal_model"] == \
        "meta-models/muse-spark-open-1026"
    assert provenance["proposal_request_id"] == "chat_f"


def test_build_endpoint_reuses_fusion_without_call(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client, run_id, recipe_id, calls = _client_with_job(
        tmp_path, "completed", monkeypatch, _parsed())
    root = tmp_path / "runs_api"
    (root / "runs" / run_id / "artifacts"
     / "multimodal_fusion_recipe.json").write_text(
        json.dumps({
            "model": "meta-models/muse-spark-open-1026",
            "provider": "baseten", "prompt_version": "v1.2",
            "request_id": "chat_fusion", "usage": {},
            "result": _parsed()}),
        encoding="utf-8")
    response = client.post(f"/api/recipes/{recipe_id}/build")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert calls["count"] == 0
    snapshot = json.loads(
        (root / "recipes" / recipe_id / "snapshot.json"
         ).read_text(encoding="utf-8"))
    assert snapshot["provenance"]["builder"] == \
        "multimodal-fusion/1"
    assert snapshot["provenance"]["proposal_model"] == \
        "meta-models/muse-spark-open-1026"
    assert snapshot["provenance"][
        "proposal_request_id"] == "chat_fusion"
    manifest = json.loads(
        (root / "recipes" / recipe_id / "manifest.json"
         ).read_text(encoding="utf-8"))
    assert "multimodal_fusion_recipe.json" in manifest[
        "sources"]


def test_build_endpoint_rejects_invalid_fusion(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    client, run_id, recipe_id, calls = _client_with_job(
        tmp_path, "completed", monkeypatch, _parsed())
    root = tmp_path / "runs_api"
    (root / "runs" / run_id / "artifacts"
     / "multimodal_fusion_recipe.json").write_text(
        json.dumps({
            "model": "meta-models/muse-spark-open-1026",
            "provider": "baseten", "prompt_version": "v1.2",
            "request_id": "chat_bad", "usage": {},
            "result": {"meaningful": True}}),
        encoding="utf-8")
    response = client.post(f"/api/recipes/{recipe_id}/build")
    assert response.status_code == 422
    assert calls["count"] == 0
    assert (root / "recipes" / recipe_id / "build_error.json"
            ).is_file()


def test_build_endpoint_grounds_fusion_in_merged_readings(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build re-check uses pipeline grounding, not FirstPass-only.

    Regression: the pipeline validated a segment-recorded
    238 g, but /build re-checked against FirstPass readings
    only and rejected the same draft with 422.
    """
    client, run_id, recipe_id, calls = _client_with_job(
        tmp_path, "completed", monkeypatch, _parsed())
    root = tmp_path / "runs_api"
    artifacts = root / "runs" / run_id / "artifacts"
    parsed = _parsed(ingredients=[
        {"name": "white liquid", "quantity": 238.0,
         "unit": "g", "provenance": "seen_video",
         "basis": "Stable 238 after tare."}])
    (artifacts / "multimodal_fusion_recipe.json").write_text(
        json.dumps({
            "model": "meta-models/muse-spark-open-1026",
            "provider": "baseten", "prompt_version": "v1.2",
            "request_id": "chat_fusion", "usage": {},
            "result": parsed}),
        encoding="utf-8")
    (artifacts / "multimodal_merged.json").write_text(
        json.dumps({"items": [{
            "source": "segment_006", "kind": "scale_reading",
            "t": 345.0, "value": 238.0, "unit": "g"}]}),
        encoding="utf-8")
    response = client.post(f"/api/recipes/{recipe_id}/build")
    assert response.status_code == 200
    assert calls["count"] == 0
    assert response.json()["recipe"]["ingredients"][0][
        "quantity"] == 238.0


def test_fusion_checks_bundle_falls_back_without_merge(
        tmp_path: Path):
    bundle = {"readings": [{"value": 100.0}]}
    same = app_module._fusion_checks_bundle(  # noqa: SLF001
        bundle, tmp_path / "runs" / "run_x")
    assert same["readings"] == [{"value": 100.0}]


def test_bundle_records_reading_truncation(tmp_path: Path):
    run_dir = tmp_path / "runs" / "run_big"
    _seed_run(run_dir)
    artifacts = run_dir / "artifacts"
    stored = json.loads(
        (artifacts / "measurements.json").read_text(
            encoding="utf-8"))
    stored["measurements"] = [
        {"frame_index": n, "timestamp_seconds": float(n),
         "value": float(n), "unit_observed": None,
         "value_status": "ok"}
        for n in range(45)
    ]
    (artifacts / "measurements.json").write_text(
        json.dumps(stored), encoding="utf-8")
    bundle = proposal_mod.build_evidence_bundle(
        run_dir, "Big", "en")
    assert len(bundle["readings"]) == 40
    assert bundle["readings_total"] == 40
    assert bundle["readings_usable_total"] == 45
    assert bundle["readings_truncated"] is True


def test_bundle_no_truncation_when_small(tmp_path: Path):
    run_dir = tmp_path / "runs" / "run_x"
    _seed_run(run_dir)
    bundle = proposal_mod.build_evidence_bundle(
        run_dir, "Small", "en")
    assert bundle["readings_total"] == 2
    assert bundle["readings_usable_total"] == 2
    assert bundle["readings_truncated"] is False


def test_snapshot_provenance_carries_truncation(
        tmp_path: Path):
    run_dir = tmp_path / "runs" / "run_x"
    _seed_run(run_dir)
    bundle = proposal_mod.build_evidence_bundle(
        run_dir, "Small", "en")
    bundle["readings_usable_total"] = 196
    bundle["readings_truncated"] = True
    snapshot = proposal_mod.assemble_snapshot(
        bundle, _parsed(), {"recipe_id": "r1", "run_id": "x",
                            "generated_at": "t"})
    assert snapshot["provenance"]["readings_used"] == 2
    assert snapshot["provenance"][
        "readings_usable_total"] == 196
    assert snapshot["provenance"]["readings_truncated"] is True


def test_row_source_distinguishes_audio_estimates():
    from bakery_intelligence.recipe_export.labels import (
        ingredient_role,
        split_display_name,
    )
    heard = {"quantity_status": "ai_estimated",
             "quantity_basis": "Audio claims 100 g sugar.",
             "evidence": []}
    assert row_source(heard, "en") == "Mentioned in audio"
    assert row_source(heard, "es") == "Mencionado en el audio"
    seen = {"quantity_status": "ai_estimated",
            "quantity_basis": "Two yolks counted visible.",
            "evidence": []}
    assert row_source(seen, "en") == "Seen in video"
    both = {"quantity_status": "ai_estimated",
            "quantity_basis": "Observed pour; audio says milk.",
            "evidence": []}
    assert row_source(both, "en") == (
        "Seen in video + Mentioned in audio")
    pure = {"quantity_status": "ai_estimated",
            "quantity_basis": "Culinary estimate.",
            "evidence": []}
    assert row_source(pure, "en") == "Suggested by AI"
    assert row_source(
        {"quantity_status": "seen_video",
         "quantity_basis": "x", "evidence": []},
        "en") == "Seen in video"
    assert split_display_name(
        "white powder (audio describes as flour)") == (
        "white powder", "audio describes as flour")
    assert split_display_name("flour") == ("flour", None)
    assert split_display_name("flour (audio") == (
        "flour (audio", None)
    assert split_display_name("(orphan)") == (
        "(orphan)", None)
    assert ingredient_role(
        {"name": "gel (cooked, added back to dough)",
         "quantity_basis": ""}) == "intermediate"
    assert ingredient_role(
        {"name": "flour", "quantity_basis": "display 700"}
    ) == "raw"


def _long_name_draft() -> dict[str, Any]:
    parsed = _parsed()
    parsed["ingredients"] = [
        {"name": "Opaque white liquid in a large steel bowl "
                 "on the bench (audio identifies as milk, "
                 "no carton or label visible - identity "
                 "unconfirmed)",
         "quantity": 238, "unit": "g",
         "provenance": "seen_video",
         "basis": "Scale display 238."},
        {"name": "flour", "quantity": 40, "unit": "g",
         "provenance": "seen_video",
         "basis": "Scale display 40."},
    ]
    return parsed


def test_normalize_splits_long_name_losslessly():
    parsed = _long_name_draft()
    original = parsed["ingredients"][0]["name"]
    assert len(original) > 120
    adjusted = proposal_mod.normalize_proposal_draft(parsed)
    assert len(adjusted) == 1
    assert adjusted[0]["kind"] == "name_split"
    assert adjusted[0]["row"] == 0
    row = parsed["ingredients"][0]
    assert row["name"] == "Opaque white liquid in a large steel bowl on the bench"
    assert len(row["name"]) <= 120
    assert original in row["basis"]
    assert len(row["basis"]) <= 500


def test_normalize_noop_when_names_fit():
    parsed = _parsed()
    assert proposal_mod.normalize_proposal_draft(
        parsed) == []


def test_normalize_rejects_unsplittable_name():
    parsed = _parsed()
    parsed["ingredients"] = [{
        "name": "x" * 121, "quantity": None, "unit": None,
        "provenance": "ai_estimated", "basis": "Estimate."}]
    with pytest.raises(proposal_mod.ProposalError,
                       match="no splittable qualifier"):
        proposal_mod.normalize_proposal_draft(parsed)


def test_normalize_rejects_exhausted_basis():
    parsed = _long_name_draft()
    parsed["ingredients"][0]["basis"] = "y" * 490
    with pytest.raises(proposal_mod.ProposalError,
                       match="basis budget"):
        proposal_mod.normalize_proposal_draft(parsed)


def test_normalize_moves_suggestion_overflow_to_notes():
    parsed = _parsed()
    parsed["suggestions"] = [
        {"id": f"s{i:02d}", "label": f"Label {i}",
         "hint": f"Hint {i}", "placeholder": f"Fill {i}"}
        for i in range(10)
    ]
    notes_before = len(parsed["notes"])
    adjusted = proposal_mod.normalize_proposal_draft(parsed)
    assert len(parsed["suggestions"]) == 8
    assert len(parsed["notes"]) == notes_before + 2
    assert [a["kind"] for a in adjusted] == [
        "suggestion_overflow"] * 2
    assert "'s08'" in parsed["notes"][-2]
    assert "'s09'" in parsed["notes"][-1]


def test_normalize_rejects_notes_budget_exhaustion():
    parsed = _parsed()
    parsed["suggestions"] = [
        {"id": f"s{i:02d}", "label": f"Label {i}",
         "hint": f"Hint {i}", "placeholder": f"Fill {i}"}
        for i in range(10)
    ]
    parsed["notes"] = [f"note {i}" for i in range(15)]
    with pytest.raises(proposal_mod.ProposalError,
                       match="notes budget"):
        proposal_mod.normalize_proposal_draft(parsed)


def test_validator_distinguishes_empty_and_long_names():
    bundle = {"readings": [{"value": 40.0}]}
    empty = _parsed()
    empty["ingredients"] = [{
        "name": "  ", "quantity": None, "unit": None,
        "provenance": "ai_estimated", "basis": "Estimate."}]
    with pytest.raises(proposal_mod.ProposalError,
                       match="needs a name"):
        proposal_mod.validate_proposal(bundle, empty)
    long = _long_name_draft()
    with pytest.raises(proposal_mod.ProposalError,
                       match="exceeds 120 characters"):
        proposal_mod.validate_proposal(bundle, long)
