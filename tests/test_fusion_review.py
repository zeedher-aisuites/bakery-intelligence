"""Offline tests for fusion draft serving and chef decisions.

Fixtures are synthetic and tmp-local. No network, no inference,
no paid services, no fixture reads from experimental evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bakery_intelligence.api.app import build_app
from bakery_intelligence.api.fusion_review import REVIEW_VIDEOS

RUN = "p04_104_v6"


def _draft() -> dict:
    return {
        "schema": "fusion-recipe-draft/v1",
        "status": "draft_needs_review",
        "working_title": "Test bake",
        "ingredients": [{
            "group_id": "ig_water", "identity_status": "model_proposed",
            "display_label": "water", "names": ["water"],
        }],
        "steps": [{
            "group_id": "eg_000", "step_index": 1,
            "start_seconds": 0.0, "end_seconds": 10.0,
            "actions": ["pour"],
        }],
        "conflicts": [{
            "conflict_id": "cf_1", "kind": "action_conflict",
            "detail": "pour vs stir",
        }],
        "unresolved_questions": ["Is the bowl tared?"],
        "quantity_assessment": {
            "status": "evaluated",
            "stable_clusters": [{
                "cluster_id": "qc_1", "displayed_value": "318",
                "start_seconds": 124.0, "end_seconds": 129.3,
                "derivation_status": "unavailable",
            }],
        },
    }


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    root = tmp_path / "runs_api"
    draft_dir = root / "fusion" / RUN
    draft_dir.mkdir(parents=True)
    (draft_dir / "fusion_draft.json").write_text(
        json.dumps(_draft()), encoding="utf-8")
    (draft_dir / "evaluation.json").write_text(
        json.dumps({"schema": "fusion-evaluation/v1"}),
        encoding="utf-8")
    return TestClient(build_app(storage_dir=root))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_get_draft_serves_payload(client: TestClient):
    response = client.get(f"/api/fusion/drafts/{RUN}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_name"] == RUN
    assert payload["draft"]["working_title"] == "Test bake"
    assert payload["evaluation"]["schema"] == "fusion-evaluation/v1"


def test_get_draft_rejects_unknown_run(client: TestClient):
    assert client.get("/api/fusion/drafts/run_evil").status_code == 422
    assert client.get(
        "/api/fusion/drafts/p04_104_v99").status_code == 404


def test_video_missing_reports_404(client: TestClient):
    response = client.get(f"/api/fusion/drafts/{RUN}/video")
    assert response.status_code == 404


def test_video_served_when_present(tmp_path: Path):
    root = tmp_path / "runs_api"
    draft_dir = root / "fusion" / RUN
    draft_dir.mkdir(parents=True)
    (draft_dir / "fusion_draft.json").write_text(
        json.dumps(_draft()), encoding="utf-8")
    (draft_dir / "evaluation.json").write_text("{}", encoding="utf-8")
    video = tmp_path / REVIEW_VIDEOS[("p04_104", None)]
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42-test-bytes")
    client = TestClient(build_app(storage_dir=root))
    response = client.get(f"/api/fusion/drafts/{RUN}/video")
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"


def test_clip_video_routing(tmp_path: Path, client: TestClient):
    demo = "bakery_demo_v1"
    draft_dir = tmp_path / "runs_api" / "fusion" / demo
    draft_dir.mkdir(parents=True)
    (draft_dir / "fusion_draft.json").write_text(
        json.dumps({"session": {"clips": []}}), encoding="utf-8")
    (draft_dir / "evaluation.json").write_text(
        "{}", encoding="utf-8")
    assert client.get(
        f"/api/fusion/drafts/{demo}").status_code == 200
    assert client.get(
        f"/api/fusion/drafts/{demo}/video/"
        "clip_01_preparation").status_code == 404
    assert client.get(
        "/api/fusion/drafts/p04_104_v6/video/evil").status_code == 404


def test_reviews_start_empty(client: TestClient):
    response = client.get(f"/api/fusion/drafts/{RUN}/reviews")
    assert response.status_code == 200
    assert response.json()["decisions"] == []


def test_confirm_persists_and_reloads(client: TestClient):
    saved = client.post(
        f"/api/fusion/drafts/{RUN}/reviews",
        json={"target_type": "step", "target_id": "eg_000",
              "action": "confirm", "reviewer": "chef-ana"})
    assert saved.status_code == 200
    assert saved.json()["action"] == "confirm"
    reloaded = client.get(f"/api/fusion/drafts/{RUN}/reviews")
    decisions = reloaded.json()["decisions"]
    assert len(decisions) == 1
    assert decisions[0]["reviewer"] == "chef-ana"


def test_correct_requires_value_and_reviewer(client: TestClient):
    base = {"target_type": "ingredient", "target_id": "ig_water",
            "action": "correct", "reviewer": "chef-ana"}
    assert client.post(f"/api/fusion/drafts/{RUN}/reviews",
                       json=base).status_code == 422
    empty = {**base, "value": {"name": "  "}}
    assert client.post(f"/api/fusion/drafts/{RUN}/reviews",
                       json=empty).status_code == 422
    anonymous = {**base, "value": {"name": "flour"},
                 "reviewer": " "}
    assert client.post(f"/api/fusion/drafts/{RUN}/reviews",
                       json=anonymous).status_code == 422
    good = {**base, "value": {"name": "flour"}}
    assert client.post(f"/api/fusion/drafts/{RUN}/reviews",
                       json=good).status_code == 200


def test_supply_accepts_unit_or_amount(client: TestClient):
    base = {"target_type": "quantity_cluster", "target_id": "qc_1",
            "action": "supply", "reviewer": "chef-ana"}
    assert client.post(f"/api/fusion/drafts/{RUN}/reviews",
                       json={**base, "value": {}}).status_code == 422
    bad = json.dumps({**base, "value": {"amount": float("inf")}},
                       allow_nan=True)
    assert client.post(f"/api/fusion/drafts/{RUN}/reviews",
                       content=bad,
                       headers={"Content-Type": "application/json"}
                       ).status_code == 422
    good = client.post(
        f"/api/fusion/drafts/{RUN}/reviews",
        json={**base, "value": {"unit": "g", "amount": 318}})
    assert good.status_code == 200
    assert good.json()["value"] == {"unit": "g", "amount": 318.0}


def test_value_forbidden_on_confirm(client: TestClient):
    response = client.post(
        f"/api/fusion/drafts/{RUN}/reviews",
        json={"target_type": "step", "target_id": "eg_000",
              "action": "confirm", "value": {"name": "x"},
              "reviewer": "chef-ana"})
    assert response.status_code == 422


def test_unknown_target_reports_404(client: TestClient):
    response = client.post(
        f"/api/fusion/drafts/{RUN}/reviews",
        json={"target_type": "step", "target_id": "eg_999",
              "action": "confirm", "reviewer": "chef-ana"})
    assert response.status_code == 404


def test_decisions_never_touch_draft(tmp_path: Path, client: TestClient):
    draft_path = (tmp_path / "runs_api" / "fusion" / RUN
                  / "fusion_draft.json")
    before = _sha(draft_path)
    client.post(
        f"/api/fusion/drafts/{RUN}/reviews",
        json={"target_type": "conflict", "target_id": "cf_1",
              "action": "cannot_determine", "reviewer": "chef-ana",
              "notes": "needs the video"})
    assert _sha(draft_path) == before
