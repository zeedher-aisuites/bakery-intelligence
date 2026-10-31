"""Offline tests for the offline_reconstruct workflow (M2-06B.1).

Bundle fixtures are synthetic and tmp-local, modeled on the proven
inputs in tests/test_reconstruction.py. No network, no inference,
no paid services.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from bakery_intelligence.api.jobs import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    JobService,
    WORKFLOW_OFFLINE_RECONSTRUCT,
)
from bakery_intelligence.api.offline import (
    BUNDLE_FORMAT,
    OFFLINE_SUMMARY_ARTIFACT,
    OfflineBundleError,
    parse_offline_bundle,
)
from bakery_intelligence.api.storage import (
    LocalStorageBackend,
    sha256_hex,
)
from bakery_intelligence.api.store import SqliteJobStore
from bakery_intelligence.api.worker import run_once
from bakery_intelligence.reconstruction.units import (
    assign_unit,
    new_sidecar,
)

FRAMES = (7400, 7450, 7500, 7550, 7600)
VERBATIMS = ("353", "421", "5 15", "575", "643")
EVENT_ID = "scale_visibility_test"
VIDEO_ID = "P04_104"


def _obs_record(frame: int, verbatim: str) -> dict[str, Any]:
    return {
        "frame_index": frame,
        "image_path": f"frame_{frame:08d}.jpg",
        "timestamp_seconds": frame / 50.0,
        "batch_id": "batch_test",
        "model_fields": {
            "scale_display_verbatim": verbatim,
            "scale_readable": True,
            "scale_state": "stable",
            "tare_action_visible": False,
            "addition_or_removal_visible": "addition",
            "confidence": "high",
        },
    }


def _temporal_events() -> list[dict[str, Any]]:
    return [
        {"event_id": EVENT_ID,
         "candidate_type": "scale_visibility",
         "start_timestamp_seconds": 148.0,
         "end_timestamp_seconds": 152.0,
         "verification_status": "needs_review",
         "frame_indices": list(FRAMES),
         "review_reasons": ["temporal_inconsistency"],
         "uncertainty_notes": []},
        {"event_id": "action_addition_test",
         "candidate_type": "action_addition",
         "start_timestamp_seconds": 148.0,
         "end_timestamp_seconds": 152.0,
         "verification_status": "candidate",
         "frame_indices": list(FRAMES),
         "review_reasons": [],
         "uncertainty_notes": []},
    ]


def _readings() -> list[dict[str, Any]]:
    return [{
        "timestamp_seconds": frame / 50.0,
        "display_text": display,
        "numeric_value": float(display),
        "readable": True,
        "confidence": "high",
        "uncertainty_notes": None,
    } for frame, display in zip(
        FRAMES, ("353", "421", "515", "575", "643"))]


def _bundle_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "bundle_format": BUNDLE_FORMAT,
        "video_id": VIDEO_ID,
        "event_id": EVENT_ID,
        "observations": [_obs_record(frame, verbatim)
                         for frame, verbatim in zip(FRAMES,
                                                   VERBATIMS)],
        "temporal": {"artifact_version": "1.1.0",
                     "events": _temporal_events()},
    }
    payload.update(overrides)
    return payload


def _bundle_bytes(**overrides: Any) -> bytes:
    return (json.dumps(_bundle_payload(**overrides)) + "\n"
            ).encode("utf-8")


def _stack(tmp_path: Path):
    storage = LocalStorageBackend(tmp_path / "storage")
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    return storage, store, JobService(store)


def _submit_offline(storage: LocalStorageBackend,
                    service: JobService,
                    raw: bytes,
                    object_id: str = "bundle.json"):
    storage.store_source(object_id, io.BytesIO(raw))
    return service.submit(WORKFLOW_OFFLINE_RECONSTRUCT,
                          source_object_id=object_id)


def _read_json(storage: LocalStorageBackend, run_id: str,
               artifact_id: str) -> Any:
    return json.loads(storage.read_artifact(run_id, artifact_id))


# Bundle parsing -----------------------------------------------------


def test_parse_accepts_minimal_bundle() -> None:
    raw = _bundle_bytes()
    bundle = parse_offline_bundle(raw)
    assert bundle.video_id == VIDEO_ID
    assert bundle.event_id == EVENT_ID
    assert bundle.draft_version == 1
    assert bundle.source_fps is None
    assert bundle.window is None
    assert bundle.units is None
    assert bundle.execution_reports == []
    assert bundle.bundle_sha256 == sha256_hex(raw)


def test_parse_accepts_window_and_options() -> None:
    bundle = parse_offline_bundle(_bundle_bytes(
        draft_version=3, source_dataset="synthetic",
        window={"start": 148.5, "end": 151.5}))
    assert bundle.draft_version == 3
    assert bundle.source_dataset == "synthetic"
    assert bundle.window == (148.5, 151.5)


def test_parse_accepts_source_fps() -> None:
    bundle = parse_offline_bundle(_bundle_bytes(source_fps=25.0))
    assert bundle.source_fps == 25.0
    integer = parse_offline_bundle(_bundle_bytes(source_fps=25))
    assert integer.source_fps == 25.0


def test_parse_rejects_malformed_json() -> None:
    with pytest.raises(OfflineBundleError, match="JSON"):
        parse_offline_bundle(b"{not json")


def test_parse_rejects_structural_problems() -> None:
    cases = {
        "format_tag": {"bundle_format": "bogus/9"},
        "video_id": {"video_id": ""},
        "event_id": {"event_id": None},
        "draft_zero": {"draft_version": 0},
        "draft_bool": {"draft_version": True},
        "draft_str": {"draft_version": "1"},
        "dataset": {"source_dataset": 7},
        "fps_zero": {"source_fps": 0},
        "fps_negative": {"source_fps": -25.0},
        "fps_nan": {"source_fps": float("nan")},
        "fps_inf": {"source_fps": float("inf")},
        "fps_str": {"source_fps": "25"},
        "fps_bool": {"source_fps": True},
        "window_shape": {"window": "narrow"},
        "window_start": {"window": {"start": "x", "end": 2.0}},
        "observations_empty": {"observations": []},
        "observations_shape": {"observations": {}},
        "temporal_shape": {"temporal": []},
        "reports_shape": {"execution_reports": {}},
        "report_event": {"execution_reports": [{"terminal": 1}]},
        "units_shape": {"units": {"assignments": {}}},
    }
    for label, override in cases.items():
        with pytest.raises(OfflineBundleError, match="."):
            parse_offline_bundle(_bundle_bytes(**override))
    payload = _bundle_payload()
    del payload["event_id"]
    with pytest.raises(OfflineBundleError, match="event_id"):
        parse_offline_bundle(json.dumps(payload).encode("utf-8"))


# Successful offline execution ---------------------------------------


def test_worker_completes_offline_reconstruct(tmp_path: Path) -> None:
    storage, store, service = _stack(tmp_path)
    raw = _bundle_bytes()
    record = _submit_offline(storage, service, raw)
    assert run_once(store, storage, "worker-1") == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_COMPLETED
    assert final.source_sha256 == sha256_hex(raw)
    assert final.artifact_ids == [
        "action_events.json",
        "measurements.json",
        "offline_summary.json",
        "recipe_draft.json",
        "validation.json",
        "weighing_event.json",
    ]
    for artifact_id in final.artifact_ids:
        content = storage.read_artifact(record.run_id, artifact_id)
        assert final.artifact_sha256[artifact_id] == sha256_hex(
            content)
    draft = _read_json(storage, record.run_id, "recipe_draft.json")
    assert draft["draft_id"] == (
        f"draft_{VIDEO_ID}_{EVENT_ID}_v1")
    measurements = _read_json(
        storage, record.run_id, "measurements.json")
    assert len(measurements["measurements"]) == len(FRAMES)
    summary = _read_json(
        storage, record.run_id, OFFLINE_SUMMARY_ARTIFACT)
    assert summary["workflow"] == WORKFLOW_OFFLINE_RECONSTRUCT
    assert summary["inference_performed"] is False
    assert summary["network_accessed"] is False
    assert summary["paid_services_used"] is False
    assert summary["bundle_sha256"] == sha256_hex(raw)
    assert summary["measurements"] == len(FRAMES)


def test_worker_offline_prefers_verified_readings(
        tmp_path: Path) -> None:
    storage, store, service = _stack(tmp_path)
    raw = _bundle_bytes(execution_reports=[{
        "event_id": EVENT_ID,
        "terminal_state": "valid_prediction",
        "selected_readings": _readings(),
    }])
    record = _submit_offline(storage, service, raw)
    assert run_once(store, storage, "worker-1") == record.run_id
    assert service.get(record.run_id).status == STATUS_COMPLETED
    measurements = _read_json(
        storage, record.run_id, "measurements.json")
    target = [item for item in measurements["measurements"]
              if item["frame_index"] == 7500][0]
    assert target["value_source"] == "verification"
    assert target["value"] == 515.0


def test_worker_offline_applies_units_sidecar(
        tmp_path: Path) -> None:
    storage, store, service = _stack(tmp_path)
    sidecar = new_sidecar()
    assign_unit(sidecar, "g", "event_window", EVENT_ID,
                148.0, 152.0, reviewer="tester")
    raw = _bundle_bytes(units=sidecar)
    record = _submit_offline(storage, service, raw)
    assert run_once(store, storage, "worker-1") == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_COMPLETED
    assert "units_applied.json" in final.artifact_ids
    applied = _read_json(
        storage, record.run_id, "units_applied.json")
    assert applied["assignments"][0]["unit"] == "g"
    measurements = _read_json(
        storage, record.run_id, "measurements.json")
    resolutions = measurements["unit_resolutions"]
    assert resolutions[0]["unit"] == "g"
    assert resolutions[0]["provenance"] == "human_assigned"


def test_worker_offline_uses_bundle_source_fps(
        tmp_path: Path) -> None:
    observations = [_obs_record(frame, verbatim)
                    for frame, verbatim in zip(FRAMES, VERBATIMS)
                    if frame != 7500]
    raw = _bundle_bytes(observations=observations,
                        source_fps=25.0)
    storage, store, service = _stack(tmp_path)
    record = _submit_offline(storage, service, raw)
    assert run_once(store, storage, "worker-1") == record.run_id
    assert service.get(record.run_id).status == STATUS_COMPLETED
    measurements = _read_json(
        storage, record.run_id, "measurements.json")
    target = [item for item in measurements["measurements"]
              if item["frame_index"] == 7500][0]
    assert target["timestamp_seconds"] == pytest.approx(300.0)
    assert target["original"] is None
    assert target["value_source"] == "none"
    assert target["value_status"] == "missing"


# Invalid inputs and missing artifacts -------------------------------


def test_worker_fails_on_malformed_bundle(tmp_path: Path) -> None:
    storage, store, service = _stack(tmp_path)
    record = _submit_offline(storage, service, b"{not json")
    assert run_once(store, storage, "worker-1") == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_FAILED
    assert final.error is not None
    assert "JSON" in final.error
    assert final.artifact_ids == []


def test_worker_fails_on_unknown_event(tmp_path: Path) -> None:
    storage, store, service = _stack(tmp_path)
    raw = _bundle_bytes(event_id="absent_event")
    record = _submit_offline(storage, service, raw)
    assert run_once(store, storage, "worker-1") == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_FAILED
    assert final.error is not None
    assert "absent_event" in final.error
    assert final.artifact_ids == []


def test_worker_fails_for_missing_source(tmp_path: Path) -> None:
    storage, store, service = _stack(tmp_path)
    record = service.submit(
        WORKFLOW_OFFLINE_RECONSTRUCT,
        source_object_id="absent.json")
    assert run_once(store, storage, "worker-1") == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_FAILED
    assert final.error is not None
    assert final.artifact_ids == []


# Worker failure semantics -------------------------------------------


def test_failed_offline_job_is_not_retried(tmp_path: Path) -> None:
    storage, store, service = _stack(tmp_path)
    record = _submit_offline(storage, service, b"{not json")
    assert run_once(store, storage, "worker-1") == record.run_id
    assert run_once(store, storage, "worker-1") is None
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_FAILED
    assert final.attempts == 1


def test_crashed_offline_job_recovers_on_next_pass(
        tmp_path: Path) -> None:
    storage, store, service = _stack(tmp_path)
    record = _submit_offline(storage, service, _bundle_bytes())
    claimed = store.claim_next(
        "crashed-worker", "2020-01-01T00:00:00+00:00")
    assert claimed is not None
    assert run_once(store, storage, "worker-2") == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_COMPLETED
    assert final.attempts == 2


# HTTP workflow selection --------------------------------------------


def test_create_accepts_explicit_offline_workflow(
        tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from bakery_intelligence.api.app import build_app

    storage = LocalStorageBackend(tmp_path)
    db_path = tmp_path / "jobs.sqlite3"
    client = TestClient(build_app(storage=storage, db_path=db_path))
    store = SqliteJobStore(db_path)
    response = client.post(
        "/api/analyses",
        files={"file": ("bundle.json", _bundle_bytes())},
        data={"workflow": WORKFLOW_OFFLINE_RECONSTRUCT},
    )
    assert response.status_code == 202
    created = response.json()
    assert created["workflow"] == WORKFLOW_OFFLINE_RECONSTRUCT
    assert run_once(store, storage, "worker-1") == created["run_id"]
    status = client.get(f"/api/analyses/{created['run_id']}")
    assert status.json()["status"] == "completed"
    artifact = client.get(
        f"/api/analyses/{created['run_id']}/artifacts/"
        f"{OFFLINE_SUMMARY_ARTIFACT}")
    assert artifact.status_code == 200
    assert artifact.json()["inference_performed"] is False


def test_create_rejects_unknown_workflow(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from bakery_intelligence.api.app import build_app

    storage = LocalStorageBackend(tmp_path)
    client = TestClient(build_app(
        storage=storage, db_path=tmp_path / "jobs.sqlite3"))
    response = client.post(
        "/api/analyses",
        files={"file": ("bundle.json", _bundle_bytes())},
        data={"workflow": "paid_magic"},
    )
    assert response.status_code == 422
