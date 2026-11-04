"""Offline tests for the editable recipe API.

Synthetic v2-shaped snapshot in tmp storage. No network, no
inference, no reads from experimental evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bakery_intelligence.api.app import build_app
from bakery_intelligence.api.recipe_editor import RECIPE_ID

SNAPSHOT = {
    "snapshot_version": "recipe-snapshot/2",
    "snapshot_id": RECIPE_ID,
    "run_name": "bakery_demo_v1",
    "generated_at": "2026-01-01T00:00:00Z",
    "title": "Cinnamon Rolls",
    "title_basis": "spoken",
    "recipe_status": "draft",
    "review_summary": {"reviewer": "ai-chef", "decisions": 1},
    "provenance": {},
    "ingredients": [
        {"name": "sugar", "quantity": 100.0, "unit": "g",
         "quantity_status": "chef_supplied", "quantity_basis": "b",
         "review_status": ["human_corrected"], "group_ids": ["g1"],
         "display_labels": ["sugar"], "evidence": [], "note": None},
        {"name": "milk", "quantity": 240.0, "unit": "g",
         "quantity_status": "ai_estimated", "quantity_basis": "est",
         "review_status": ["human_verified"],
         "group_ids": ["g2"], "display_labels": ["milk"],
         "evidence": [], "note": None},
    ],
    "excluded_observations": [],
    "method": [
        {"number": 1, "clip_id": "c1", "interval": "0:00-0:10",
         "instruction": "Mix.", "step_group_ids": ["eg_0"],
         "review_status": ["human_verified"], "caveats": []},
    ],
    "omitted_detail_steps": [],
    "review_flags": [],
    "completion_open_items": [],
    "limitations": ["No ground truth."],
}


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    root = tmp_path / "runs_api"
    recipe_dir = root / "recipes" / RECIPE_ID
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "snapshot.json").write_text(
        json.dumps(SNAPSHOT), encoding="utf-8")
    return TestClient(build_app(storage_dir=root))


def test_get_recipe_merges_untouched_draft(client: TestClient):
    response = client.get(f"/api/recipes/{RECIPE_ID}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["recipe_status"] == "draft"
    assert payload["ingredients"][0]["chef_edited"] is False
    assert payload["ingredients"][1]["quantity"] == 240.0
    assert payload["unknown-id" if False else "title"] == \
        "Cinnamon Rolls"


def test_unknown_recipe_is_404(client: TestClient):
    assert client.get("/api/recipes/nope").status_code == 404


def test_save_persists_edits_with_originals(client: TestClient):
    response = client.put(f"/api/recipes/{RECIPE_ID}", json={
        "ingredients": [{"index": 1, "quantity": 250.0}],
        "method": [{"number": 1, "instruction": "Mix well."}],
    })
    assert response.status_code == 200
    payload = response.json()
    milk = payload["ingredients"][1]
    assert milk["quantity"] == 250.0
    assert milk["chef_edited"] is True
    assert milk["original"]["quantity"] == 240.0
    assert payload["method"][0]["instruction"] == "Mix well."
    assert payload["last_saved_at"]

    reread = client.get(f"/api/recipes/{RECIPE_ID}").json()
    assert reread["ingredients"][1]["quantity"] == 250.0
    assert reread["method"][0]["chef_edited"] is True


def test_save_rejects_unknown_targets(client: TestClient):
    response = client.put(f"/api/recipes/{RECIPE_ID}", json={
        "ingredients": [{"index": 9, "quantity": 1.0}],
        "method": [],
    })
    assert response.status_code == 422
    response = client.put(f"/api/recipes/{RECIPE_ID}", json={
        "ingredients": [],
        "method": [{"number": 7, "instruction": "x"}],
    })
    assert response.status_code == 422


def test_status_transitions_persist(client: TestClient):
    response = client.post(f"/api/recipes/{RECIPE_ID}/status",
                           json={"status": "approved"})
    assert response.status_code == 200
    assert response.json()["recipe_status"] == "approved"
    response = client.post(
        f"/api/recipes/{RECIPE_ID}/status",
        json={"status": "needs_changes", "note": "Check salt."})
    payload = response.json()
    assert payload["recipe_status"] == "needs_changes"
    assert payload["status_note"] == "Check salt."
    assert client.post(f"/api/recipes/{RECIPE_ID}/status",
                       json={"status": "bogus"}).status_code == 422


def test_exports_reflect_saved_state(client: TestClient,
                                     tmp_path: Path):
    client.put(f"/api/recipes/{RECIPE_ID}", json={
        "ingredients": [{"index": 0, "quantity": 120.0}],
        "method": [],
    })
    pdf = client.get(f"/api/recipes/{RECIPE_ID}/export.pdf")
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content[:5] == b"%PDF-"
    target = tmp_path / "export.pdf"
    target.write_bytes(pdf.content)
    import pymupdf
    with pymupdf.open(target) as doc:
        text = "\n".join(page.get_text() for page in doc)
    assert "120" in text
    assert "Edited by chef" in text
    assert "Evidence appendix" not in text

    docx = client.get(f"/api/recipes/{RECIPE_ID}/export.docx")
    assert docx.status_code == 200
    assert docx.content[:2] == b"PK"
    doctarget = tmp_path / "export.docx"
    doctarget.write_bytes(docx.content)
    from docx import Document
    words = "\n".join(p.text for p in Document(doctarget).paragraphs)
    assert "Cinnamon Rolls" in words
    assert "Mix." in words


def _seed_job_and_recipe(tmp_path: Path, run_id: str,
                         name: str) -> TestClient:
    from bakery_intelligence.api.jobs import JobRecord
    from bakery_intelligence.api.recipe_editor import (
        create_record,
    )
    from bakery_intelligence.api.store import SqliteJobStore

    root = tmp_path / "runs_api"
    create_record(root, name, "en", source_run_id=run_id)
    store = SqliteJobStore(root / "jobs.sqlite3")
    store.create(JobRecord(
        run_id=run_id,
        workflow="video_pipeline_multimodal",
        source_object_id="clip.mp4"))
    return TestClient(build_app(storage_dir=root))


def test_cards_mark_failed_run_as_failed(
        tmp_path: Path) -> None:
    from bakery_intelligence.api.store import SqliteJobStore

    client = _seed_job_and_recipe(
        tmp_path, "run_failed_1", "Failed Dish")
    store = SqliteJobStore(
        tmp_path / "runs_api" / "jobs.sqlite3")
    assert store.fail("run_failed_1", None,
                      "2026-11-04T00:00:00+00:00",
                      "Stage audio failed: truncated")
    cards = client.get("/api/recipes").json()["recipes"]
    assert len(cards) == 1
    assert cards[0]["status"] == "failed"
    assert "truncated" in (cards[0]["build_error"] or "")
    assert cards[0]["ingredients"] == 0


def test_cards_mark_completed_run_awaiting_build(
        tmp_path: Path) -> None:
    from bakery_intelligence.api.store import SqliteJobStore

    client = _seed_job_and_recipe(
        tmp_path, "run_done_1", "Done Dish")
    store = SqliteJobStore(
        tmp_path / "runs_api" / "jobs.sqlite3")
    claimed = store.claim_next(
        "w1", "2026-11-04T00:00:00+00:00")
    assert claimed is not None
    assert store.complete("run_done_1", "w1",
                          "2026-11-04T00:00:00+00:00",
                          None, {})
    cards = client.get("/api/recipes").json()["recipes"]
    assert cards[0]["status"] == "awaiting_build"


def test_cards_keep_active_run_processing(
        tmp_path: Path) -> None:
    client = _seed_job_and_recipe(
        tmp_path, "run_live_1", "Live Dish")
    cards = client.get("/api/recipes").json()["recipes"]
    assert cards[0]["status"] == "processing"


def test_cards_snapshot_status_untouched(
        client: TestClient) -> None:
    cards = client.get("/api/recipes").json()["recipes"]
    assert cards[0]["status"] == "draft"
