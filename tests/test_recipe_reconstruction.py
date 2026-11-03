"""Non-billable tests for adversarial recipe reconstruction.

Deterministic fakes only: no network, no provider client, and
no video reprocessing. Covers the ten required scenarios plus
chef-supplied content, agreement-without-evidence, immutable
snapshots, and original-evidence preservation.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from bakery_intelligence.adversarial.claims import (
    evidence_ref,
    make_claim,
)
from bakery_intelligence.adversarial.evidence import (
    build_evidence_package,
)
from bakery_intelligence.adversarial.providers import (
    DeterministicFakeChallenger,
    DeterministicFakeProposer,
    EvidenceProposer,
)
from bakery_intelligence.adversarial.resolution import debate_claims
from bakery_intelligence.adversarial.service import RecipeReviewService
from bakery_intelligence.adversarial.store import RecipeReviewStore
from bakery_intelligence.api.jobs import JobService, WORKFLOW_STORAGE_SMOKE
from bakery_intelligence.api.storage import LocalStorageBackend
from bakery_intelligence.api.store import SqliteJobStore


def _package(**overrides: Any) -> dict[str, Any]:
    package: dict[str, Any] = {
        "schema_version": "evidence-package/1",
        "run_id": "run_0123456789ab",
        "video": {"duration_seconds": 50.0, "source_sha256": "a" * 64},
        "frames": [
            {"frame_index": 0, "timestamp_seconds": 0.0,
             "measurement": {"measurement_id": "m_0", "value": 0.0}},
            {"frame_index": 50, "timestamp_seconds": 1.0,
             "measurement": {"measurement_id": "m_50", "value": 200.0}},
        ],
        "weighing_event": {"verification_status": "confirmed",
                           "tare_observed": True},
        "unknowns": [],
    }
    package.update(overrides)
    return package


def _quantity_claim(**kwargs: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = dict(
        kind="quantity", title="Flour quantity",
        ingredient_id="flour", quantity_value=200.0,
        quantity_unit="g", derivation="difference_between_readings",
        evidence_refs=[evidence_ref("run_0123456789ab", 50, "m_50", 1.0)],
    )
    defaults.update(kwargs)
    return make_claim(**defaults)


def test_supported_ingredient_and_quantity() -> None:
    package = _package()
    ingredient = make_claim(
        kind="ingredient", title="Flour is added",
        ingredient_id="flour",
        evidence_refs=[evidence_ref("run_0123456789ab", 50, "m_50", 1.0)])
    result = debate_claims(
        package, [ingredient, _quantity_claim()],
        DeterministicFakeChallenger())
    statuses = {c["claim_id"]: c["status"] for c in result.claims}
    assert statuses[ingredient["claim_id"]] == "supported"
    assert result.verification.verdict.value == "pass"


def test_unknown_ingredient_stays_unknown() -> None:
    package = _package()
    claims = EvidenceProposer().propose(package)
    result = debate_claims(
        package, claims, DeterministicFakeChallenger())
    ingredient = next(c for c in result.claims if c["kind"] == "ingredient")
    assert ingredient["ingredient_id"] is None
    assert ingredient["status"] == "uncertain"
    assert ingredient["quantity"]["value"] is None


def test_ambiguous_tare_requires_human() -> None:
    package = _package(
        weighing_event={"verification_status": "ambiguous"})
    result = debate_claims(
        package, [_quantity_claim()], DeterministicFakeChallenger())
    assert result.claims[0]["status"] == "requires_human_review"
    codes = [i.code for i in result.verification.issues]
    assert "AMBIGUOUS_TARE" in codes


def test_conflicting_readings_are_contradicted() -> None:
    package = _package()
    claims = [_quantity_claim(),
              _quantity_claim(quantity_value=250.0, claim_id="claim_b")]
    result = debate_claims(
        package, claims, DeterministicFakeChallenger())
    assert all(c["status"] == "contradicted" for c in result.claims)


def test_model_disagreement_preserved_after_max_rounds() -> None:
    package = _package()
    claim = _quantity_claim()
    challenger = DeterministicFakeChallenger(objection_rounds=[
        {claim["claim_id"]: ["Scale state may be cumulative."]},
        {claim["claim_id"]: ["Still unresolved."]},
        {claim["claim_id"]: ["Still unresolved."]},
        {claim["claim_id"]: ["Must never be requested."]},
    ])
    result = debate_claims(package, [claim], challenger)
    assert result.rounds_used == 3
    assert result.termination_reason == "max_rounds_reached"
    assert result.claims[0]["status"] == "requires_human_review"
    assert len(result.claims[0]["history"]) == 6


def test_invalid_evidence_reference_blocks_support() -> None:
    package = _package()
    claim = _quantity_claim(evidence_refs=[
        evidence_ref("run_0123456789ab", 9999, "m_x", 1.0)])
    result = debate_claims(
        package, [claim], DeterministicFakeChallenger())
    assert result.claims[0]["status"] == "requires_human_review"
    assert result.verification.verdict.value == "fail"


def test_agreement_without_evidence_is_not_supported() -> None:
    package = _package()
    claim = _quantity_claim(evidence_refs=[])
    result = debate_claims(
        package, [claim], DeterministicFakeChallenger())
    assert result.claims[0]["status"] != "supported"


def test_chef_supplied_ingredient_absent_from_video() -> None:
    claim = make_claim(
        kind="ingredient", title="Cinnamon (chef supplied)",
        ingredient_id="cinnamon", provenance="chef-supplied",
        evidence_refs=[])
    package = _package()
    result = debate_claims(
        package, [claim], DeterministicFakeChallenger())
    assert result.claims[0]["status"] == "requires_human_review"
    assert result.claims[0]["provenance"] == "chef-supplied"


def _completed_stack(tmp_path: Path):
    storage = LocalStorageBackend(tmp_path / "storage")
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    service = JobService(store)
    record = service.submit(
        WORKFLOW_STORAGE_SMOKE, source_object_id="clip.mp4")
    storage.store_source("clip.mp4", io.BytesIO(b"fake"))
    claimed = store.claim_next("worker-1", "2026-11-01T00:00:00+00:00")
    assert claimed is not None
    artifacts = {
        "measurements.json": {
            "window": {"start_seconds": 0.0, "end_seconds": 2.0},
            "measurements": [
                {"measurement_id": "m_0", "frame_index": 0,
                 "timestamp_seconds": 0.0, "value": None,
                 "value_status": "invalid", "value_source": "none",
                 "unit_observed": None},
                {"measurement_id": "m_50", "frame_index": 50,
                 "timestamp_seconds": 1.0, "value": 200.0,
                 "value_status": "ok", "value_source": "firstpass",
                 "unit_observed": None},
            ],
            "action_evidence": {"tare_frames": [0]},
        },
        "pipeline_summary.json": {
            "video": {"file_name": "clip.mp4",
                      "duration_seconds": 2.0, "fps": 50.0},
            "source_sha256": "a" * 64,
        },
        "firstpass_report.json": {"model": "muse-spark-1.3"},
        "recipe_draft.json": {
            "draft_id": "draft_test", "unresolved_questions": []},
        "weighing_event.json": {
            "ingredient_id": None, "verification_status": "ambiguous"},
    }
    digests = {}
    for artifact_id, payload in artifacts.items():
        data = json.dumps(payload).encode()
        digests[artifact_id] = storage.store_artifact(
            record.run_id, artifact_id, data)
    assert store.complete(record.run_id, "worker-1",
                          "2026-11-01T00:01:00+00:00", "a" * 64, digests)
    return storage, service.get(record.run_id)


def test_human_correction_approval_persistence_and_reload(
        tmp_path: Path) -> None:
    storage, record = _completed_stack(tmp_path)
    assert record is not None
    before = storage.read_artifact(record.run_id, "measurements.json")
    service = RecipeReviewService(storage)
    state = service.get_or_create(record)
    assert state["simulation"] is True
    assert state["status"] == "pending_review"
    package = build_evidence_package(storage, record)
    assert package["run_id"] == record.run_id
    assert package["frames"][1]["measurement"]["value"] == 200.0

    store = RecipeReviewStore(storage.base_dir)
    corrected = store.save_corrections(
        record.run_id,
        ingredients=[{
            "ingredient_id": "cinnamon", "name": "Cinnamon",
            "quantity": {"value": None, "unit": None,
                         "derivation": "unknown"},
            "provenance": "chef-supplied",
        }],
        steps=[{
            "index": 0, "description": "Add cinnamon (chef supplied).",
            "provenance": "chef-supplied",
        }],
        reviewer="chef-maria")
    assert corrected["version"] == 2
    decided = store.decide(record.run_id, "approved", "chef-maria")
    snapshot = decided["approved_snapshots"][0]
    assert decided["status"] == "approved"
    assert snapshot["reviewer_id"] == "chef-maria"

    reloaded = RecipeReviewStore(storage.base_dir).load(record.run_id)
    assert reloaded is not None
    assert reloaded["approved_snapshots"][0]["sha256"] == snapshot["sha256"]
    assert storage.read_artifact(
        record.run_id, "measurements.json") == before

    edited = store.save_corrections(
        record.run_id, ingredients=[], steps=[], reviewer="chef-maria")
    assert edited["version"] == 3
    assert edited["approved_snapshots"][0]["sha256"] == snapshot["sha256"]


def test_recipe_api_approval_survives_app_rebuild(
        tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from bakery_intelligence.api.app import build_app

    storage, record = _completed_stack(tmp_path)
    assert record is not None
    db_path = tmp_path / "jobs.sqlite3"
    client = TestClient(build_app(storage=storage, db_path=db_path))
    created = client.get(f"/api/analyses/{record.run_id}/recipe")
    assert created.status_code == 200
    assert created.json()["simulation"] is True
    corrected = client.post(
        f"/api/analyses/{record.run_id}/recipe/corrections",
        json={
            "ingredients": [{
                "ingredient_id": "cinnamon",
                "quantity": {"value": None, "unit": None,
                             "derivation": "unknown"},
                "provenance": "chef-supplied",
            }],
            "steps": [{
                "index": 0, "description": "Add cinnamon.",
                "provenance": "chef-supplied",
            }],
            "reviewer": "chef-maria",
        })
    assert corrected.status_code == 200
    assert corrected.json()["version"] == 2
    decided = client.post(
        f"/api/analyses/{record.run_id}/recipe/decisions",
        json={"decision": "approved", "reviewer": "chef-maria"})
    assert decided.status_code == 200
    rebuilt = TestClient(build_app(storage=storage, db_path=db_path))
    reloaded = rebuilt.get(f"/api/analyses/{record.run_id}/recipe")
    assert reloaded.json()["status"] == "approved"
    assert len(reloaded.json()["approved_snapshots"]) == 1
    assert reloaded.json()["decisions"][0]["reviewer_id"] == "chef-maria"


def test_temporal_links_persist_and_reload(tmp_path: Path) -> None:
    storage, record = _completed_stack(tmp_path)
    assert record is not None
    store = RecipeReviewStore(storage.base_dir)
    RecipeReviewService(storage).get_or_create(record)
    saved = store.save_corrections(
        record.run_id,
        ingredients=[{
            "ingredient_id": "flour", "name": "Flour",
            "quantity": {"value": None, "unit": None,
                         "derivation": "unknown"},
            "provenance": "chef-supplied",
        }],
        steps=[{
            "step_id": "add_flour", "description": "Add flour.",
            "start_seconds": 23.0, "end_seconds": 29.0,
            "frame_refs": [{"frame_index": 50}],
            "ingredient_refs": ["flour"],
            "provenance": "chef-supplied",
            "not_observed": True, "review_status": "not_observed",
        }],
        reviewer="chef-maria")
    step = saved["steps"][0]
    assert step["start_seconds"] == 23.0 and step["end_seconds"] == 29.0
    reloaded = RecipeReviewStore(storage.base_dir).load(record.run_id)
    assert reloaded is not None
    assert reloaded["steps"][0]["frame_refs"] == [{"frame_index": 50}]
    assert reloaded["steps"][0]["ingredient_refs"] == ["flour"]


def test_invalid_temporal_and_reference_rejected(
        tmp_path: Path) -> None:
    storage, record = _completed_stack(tmp_path)
    assert record is not None
    store = RecipeReviewStore(storage.base_dir)
    RecipeReviewService(storage).get_or_create(record)
    from bakery_intelligence.adversarial.store import (
        RecipeReviewError,
    )
    with pytest.raises(RecipeReviewError, match="ends before"):
        store.save_corrections(
            record.run_id, ingredients=[],
            steps=[{"description": "Bad interval.",
                    "start_seconds": 30.0, "end_seconds": 20.0}],
            reviewer="chef-maria")
    with pytest.raises(RecipeReviewError, match="unknown ingredient"):
        store.save_corrections(
            record.run_id, ingredients=[],
            steps=[{"description": "Bad ref.",
                    "ingredient_refs": ["ghost"]}],
            reviewer="chef-maria")
    service = RecipeReviewService(storage)
    with pytest.raises(RecipeReviewError, match="missing frame"):
        service.save_corrections(
            record, ingredients=[],
            steps=[{"description": "Bad frame.",
                    "frame_refs": [{"frame_index": 9999}]}],
            reviewer="chef-maria")


def test_legacy_state_without_temporal_fields_reads(
        tmp_path: Path) -> None:
    storage, record = _completed_stack(tmp_path)
    assert record is not None
    service = RecipeReviewService(storage)
    state = service.get_or_create(record)
    legacy = dict(state)
    legacy["ingredients"] = [{"ingredient_id": "flour"}]
    legacy["steps"] = [{"description": "Legacy step."}]
    RecipeReviewStore(storage.base_dir).save(legacy)
    view = RecipeReviewService(storage).get_or_create(record)
    assert view["steps"][0]["start_seconds"] is None
    assert view["steps"][0]["frame_refs"] == []
    assert view["steps"][0]["step_id"] == "step_1"


def test_measurement_claim_marks_representative_evidence() -> None:
    package = _package()
    claims = EvidenceProposer().propose(package)
    measurement = next(c for c in claims if c["kind"] == "measurement")
    assert measurement["evidence_scope"] == "representative"
    assert measurement["evidence_total"] == 2
    assert len(measurement["evidence_refs"]) == 2


def test_no_provider_calls_beyond_fake_counters() -> None:
    package = _package()
    proposer = DeterministicFakeProposer(claims=[_quantity_claim()])
    challenger = DeterministicFakeChallenger()
    claims = proposer.propose(package)
    result = debate_claims(package, claims, challenger)
    assert proposer.calls == 1
    assert challenger.calls == result.rounds_used == 1
    # Fakes receive claims and the package only; no frame/video input.
    assert challenger.claims_seen == [[claims[0]["claim_id"]]]
    assert "video_bytes" not in json.dumps(package)


def _usage_stack(tmp_path: Path):
    storage, record = _completed_stack(tmp_path)
    assert record is not None
    store = RecipeReviewStore(storage.base_dir)
    RecipeReviewService(storage).get_or_create(record)
    return store, record


def test_step_usages_persist_quantities_units_and_provenance(
        tmp_path: Path) -> None:
    store, record = _usage_stack(tmp_path)
    saved = store.save_corrections(
        record.run_id,
        ingredients=[{"ingredient_id": "flour",
                      "provenance": "chef-supplied"}],
        steps=[{
            "step_id": "add_flour", "description": "Add flour.",
            "ingredient_refs": ["flour"],
            "usages": [{
                "ingredient_id": "flour",
                "quantity": {"value": 200, "unit": "g",
                             "derivation": "chef_supplied"},
                "provenance": "chef-supplied",
            }],
            "provenance": "chef-supplied",
        }],
        reviewer="chef-maria")
    usage = saved["steps"][0]["usages"][0]
    assert usage["ingredient_id"] == "flour"
    assert usage["quantity"] == {"value": 200, "unit": "g",
                                 "derivation": "chef_supplied"}
    assert usage["provenance"] == "chef-supplied"
    # Summary step_refs stay consistent without double-counting.
    assert saved["ingredients"][0]["step_refs"] == ["add_flour"]
    reloaded = store.load(record.run_id)
    assert reloaded is not None
    assert reloaded["steps"][0]["usages"][0]["quantity"]["value"] == 200


def test_step_usages_reload_round_trip(tmp_path: Path) -> None:
    store, record = _usage_stack(tmp_path)
    store.save_corrections(
        record.run_id,
        ingredients=[{"ingredient_id": "flour",
                      "provenance": "chef-supplied"}],
        steps=[{
            "step_id": "add_flour", "description": "Add flour.",
            "ingredient_refs": [],
            "usages": [{"ingredient_id": "flour",
                        "quantity": {"value": None, "unit": None,
                                     "derivation": "chef_supplied"}}],
            "provenance": "chef-supplied",
        }],
        reviewer="chef-maria")
    reloaded = store.load(record.run_id)
    assert reloaded is not None
    usage = reloaded["steps"][0]["usages"][0]
    assert usage["quantity"] == {"value": None, "unit": None,
                                 "derivation": "chef_supplied"}
    # Usage-only links still join the summary union.
    assert reloaded["ingredients"][0]["step_refs"] == ["add_flour"]


def test_step_usages_reject_bad_quantities_and_ai_labels(
        tmp_path: Path) -> None:
    from bakery_intelligence.adversarial.store import RecipeReviewError
    store, record = _usage_stack(tmp_path)
    base_step = {"step_id": "add_flour", "description": "Add flour.",
                 "ingredient_refs": [], "provenance": "chef-supplied"}
    base_ingredient = [{"ingredient_id": "flour",
                        "provenance": "chef-supplied"}]
    for bad_quantity in ({"value": "lots"}, {"value": -5},
                         {"value": float("inf")}, {"value": True},
                         "200"):
        step = dict(base_step, usages=[{"ingredient_id": "flour",
                                        "quantity": bad_quantity}])
        with pytest.raises(RecipeReviewError):
            store.save_corrections(
                record.run_id, ingredients=base_ingredient,
                steps=[step], reviewer="chef-maria")
    for bad_usage in ({"quantity": {}}, {"ingredient_id": "  ",
                                          "quantity": {}},
                      {"ingredient_id": "flour",
                       "provenance": "ai_observed"}):
        step = dict(base_step, usages=[bad_usage])
        with pytest.raises(RecipeReviewError):
            store.save_corrections(
                record.run_id, ingredients=base_ingredient,
                steps=[step], reviewer="chef-maria")
