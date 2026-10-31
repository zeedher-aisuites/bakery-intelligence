"""Offline tests for the video_reconstruct workflow (M2-06C.1).

Synthetic evidence sets are tmp-local and exercise manifest
validation, recorded-layout assembly, equivalence with
offline_reconstruct, and worker failure semantics. One test runs
against real recorded P04 evidence when present and skips
explicitly otherwise. No network, no inference, no paid services.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from bakery_intelligence.api import video as video_api
from bakery_intelligence.api.jobs import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    JobService,
    WORKFLOW_OFFLINE_RECONSTRUCT,
    WORKFLOW_VIDEO_RECONSTRUCT,
)
from bakery_intelligence.api.storage import (
    LocalStorageBackend,
    sha256_hex,
)
from bakery_intelligence.api.store import SqliteJobStore
from bakery_intelligence.api.video import (
    VIDEO_MANIFEST_FORMAT,
    VIDEO_SUMMARY_ARTIFACT,
    VideoManifestError,
    parse_video_manifest,
)
from bakery_intelligence.api.worker import run_once
from bakery_intelligence.reconstruction.units import (
    assign_unit,
    new_sidecar,
)

FRAMES = (7400, 7450, 7500, 7550, 7600)
VERBATIMS = ("353", "421", "5 15", "575", "643")
EVENT_ID = "scale_visibility_vid"
VIDEO_ID = "SYNTH.VID.001"
SYNTH_SET = "synth_video_test"
_SYNTH_SPEC = {
    "observations": "synth/obs",
    "temporal": "synth/temporal.json",
    "executions": "synth/exec",
    "units": "synth/units.json",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDED_ROOT = REPO_ROOT / "data" / "processed"


def _obs_record(frame: int, verbatim: str) -> dict[str, Any]:
    return {
        "frame_index": frame,
        "image_path": f"frame_{frame:08d}.jpg",
        "timestamp_seconds": frame / 50.0,
        "batch_id": "batch_synth",
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
        {"event_id": "action_addition_vid",
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8")


def _allow_synth_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Authorize the synthetic set within test scope only."""
    monkeypatch.setitem(video_api.EVIDENCE_SETS, SYNTH_SET,
                        _SYNTH_SPEC)


def _synthetic_set(monkeypatch: pytest.MonkeyPatch,
                   tmp_path: Path,
                   with_optionals: bool = True) -> Path:
    """Write a native-layout evidence set; return its root."""
    root = tmp_path / "evidence"
    first = [_obs_record(frame, verbatim)
             for frame, verbatim in zip(FRAMES[:3], VERBATIMS[:3])]
    second = [_obs_record(frame, verbatim)
              for frame, verbatim in zip(FRAMES[3:], VERBATIMS[3:])]
    _write_json(root / "synth" / "obs" / "batch_a.json", first)
    _write_json(root / "synth" / "obs" / "batch_b.json", second)
    _write_json(root / "synth" / "temporal.json",
                {"artifact_version": "1.1.0",
                 "events": _temporal_events()})
    if with_optionals:
        _write_json(
            root / "synth" / "exec" / "run_1" / "execution_report.json",
            {"event_id": EVENT_ID,
             "terminal_state": "valid_prediction",
             "selected_readings": _readings()})
        sidecar = new_sidecar()
        assign_unit(sidecar, "g", "event_window", EVENT_ID,
                    148.0, 152.0, reviewer="tester")
        _write_json(root / "synth" / "units.json", sidecar)
    _allow_synth_set(monkeypatch)
    return root


def _manifest_bytes(evidence_set: str = SYNTH_SET,
                    **overrides: Any) -> bytes:
    payload: dict[str, Any] = {
        "manifest_format": VIDEO_MANIFEST_FORMAT,
        "evidence_set": evidence_set,
        "event_id": EVENT_ID,
        "video_id": VIDEO_ID,
    }
    payload.update(overrides)
    return (json.dumps(payload) + "\n").encode("utf-8")


def _stack(tmp_path: Path):
    storage = LocalStorageBackend(tmp_path / "storage")
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    return storage, store, JobService(store)


def _submit(storage: LocalStorageBackend, service: JobService,
            workflow: str, raw: bytes,
            object_id: str = "input.json"):
    storage.store_source(object_id, io.BytesIO(raw))
    return service.submit(workflow, source_object_id=object_id)


def _read_json(storage: LocalStorageBackend, run_id: str,
               artifact_id: str) -> Any:
    return json.loads(storage.read_artifact(run_id, artifact_id))


# Manifest validation ------------------------------------------------


def test_parse_accepts_manifest_with_defaults(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_synth_set(monkeypatch)
    raw = _manifest_bytes()
    manifest = parse_video_manifest(raw)
    assert manifest.evidence_set == SYNTH_SET
    assert manifest.event_id == EVENT_ID
    assert manifest.video_id == VIDEO_ID
    assert manifest.draft_version == 1
    assert manifest.window is None
    assert manifest.manifest_sha256 == sha256_hex(raw)


def test_parse_rejects_manifest_problems(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _allow_synth_set(monkeypatch)
    cases = {
        "format_tag": {"manifest_format": "bogus/9"},
        "evidence_set": {"evidence_set": "no_such_set"},
        "event_id": {"event_id": ""},
        "video_id": {"video_id": None},
        "draft_zero": {"draft_version": 0},
        "draft_bool": {"draft_version": True},
        "dataset": {"source_dataset": 7},
        "window_shape": {"window": [1.0, 2.0]},
        "window_end": {"window": {"start": 1.0, "end": "x"}},
    }
    for label, override in cases.items():
        with pytest.raises(VideoManifestError, match="."):
            parse_video_manifest(_manifest_bytes(**override))
    with pytest.raises(VideoManifestError, match="JSON"):
        parse_video_manifest(b"{not json")


# Successful recorded-layout execution -------------------------------


def test_worker_completes_video_reconstruct(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage, store, service = _stack(tmp_path)
    evroot = _synthetic_set(monkeypatch, tmp_path)
    raw = _manifest_bytes()
    record = _submit(storage, service,
                     WORKFLOW_VIDEO_RECONSTRUCT, raw, "job.json")
    assert run_once(store, storage, "worker-1",
                    evidence_dir=evroot) == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_COMPLETED
    assert final.source_sha256 == sha256_hex(raw)
    assert final.artifact_ids == [
        "action_events.json",
        "measurements.json",
        "recipe_draft.json",
        "units_applied.json",
        "validation.json",
        "video_summary.json",
        "weighing_event.json",
    ]
    for artifact_id in final.artifact_ids:
        content = storage.read_artifact(record.run_id, artifact_id)
        assert final.artifact_sha256[artifact_id] == sha256_hex(
            content)
    summary = _read_json(storage, record.run_id,
                         VIDEO_SUMMARY_ARTIFACT)
    assert summary["workflow"] == WORKFLOW_VIDEO_RECONSTRUCT
    assert summary["evidence_set"] == SYNTH_SET
    assert summary["manifest_sha256"] == sha256_hex(raw)
    assert summary["inference_performed"] is False
    assert summary["network_accessed"] is False
    assert summary["paid_services_used"] is False
    assert summary["draft_id"] == (
        f"draft_{VIDEO_ID}_{EVENT_ID}_v1")
    assert summary["measurements"] == len(FRAMES)
    roles = sorted(entry["role"] for entry in summary["inputs"])
    assert roles == ["execution_report", "observations",
                     "observations", "temporal", "units"]
    assert all(entry["present"] for entry in summary["inputs"])
    assert all(entry["sha256"] for entry in summary["inputs"])
    draft = _read_json(storage, record.run_id, "recipe_draft.json")
    assert draft["draft_id"] == summary["draft_id"]
    measurements = _read_json(
        storage, record.run_id, "measurements.json")
    target = [item for item in measurements["measurements"]
              if item["frame_index"] == 7500][0]
    assert target["value"] == 515.0
    assert target["value_source"] == "verification"


def test_worker_video_records_absent_optionals(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage, store, service = _stack(tmp_path)
    evroot = _synthetic_set(monkeypatch, tmp_path,
                            with_optionals=False)
    record = _submit(storage, service,
                     WORKFLOW_VIDEO_RECONSTRUCT,
                     _manifest_bytes(), "job.json")
    assert run_once(store, storage, "worker-1",
                    evidence_dir=evroot) == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_COMPLETED
    assert "units_applied.json" not in final.artifact_ids
    summary = _read_json(storage, record.run_id,
                         VIDEO_SUMMARY_ARTIFACT)
    absent = {entry["role"] for entry in summary["inputs"]
              if not entry["present"]}
    assert absent == {"execution_reports", "units"}


def test_video_matches_offline_for_equivalent_evidence(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage, store, service = _stack(tmp_path)
    evroot = _synthetic_set(monkeypatch, tmp_path)
    obs_ordered = ([_obs_record(frame, verbatim)
                    for frame, verbatim in zip(FRAMES[:3],
                                              VERBATIMS[:3])]
                   + [_obs_record(frame, verbatim)
                      for frame, verbatim in zip(FRAMES[3:],
                                                VERBATIMS[3:])])
    sidecar = new_sidecar()
    assign_unit(sidecar, "g", "event_window", EVENT_ID,
                148.0, 152.0, reviewer="tester")
    bundle = {
        "bundle_format": "bakery-offline-bundle/1",
        "video_id": VIDEO_ID,
        "event_id": EVENT_ID,
        "observations": obs_ordered,
        "temporal": {"artifact_version": "1.1.0",
                     "events": _temporal_events()},
        "execution_reports": [{
            "event_id": EVENT_ID,
            "terminal_state": "valid_prediction",
            "selected_readings": _readings()}],
        "units": sidecar,
    }
    bundle_raw = (json.dumps(bundle) + "\n").encode("utf-8")
    offline = _submit(storage, service,
                      WORKFLOW_OFFLINE_RECONSTRUCT,
                      bundle_raw, "bundle.json")
    video = _submit(storage, service, WORKFLOW_VIDEO_RECONSTRUCT,
                    _manifest_bytes(), "job.json")
    assert run_once(store, storage, "worker-1",
                    evidence_dir=evroot) == offline.run_id
    assert run_once(store, storage, "worker-1",
                    evidence_dir=evroot) == video.run_id
    assert service.get(offline.run_id).status == STATUS_COMPLETED
    assert service.get(video.run_id).status == STATUS_COMPLETED
    for artifact_id in ("measurements.json", "validation.json",
                        "weighing_event.json",
                        "action_events.json",
                        "units_applied.json"):
        left = storage.read_artifact(offline.run_id, artifact_id)
        right = storage.read_artifact(video.run_id, artifact_id)
        assert left == right
    offline_draft = _read_json(storage, offline.run_id,
                               "recipe_draft.json")
    video_draft = _read_json(storage, video.run_id,
                             "recipe_draft.json")
    assert offline_draft["draft_id"] == video_draft["draft_id"]
    assert video_draft["run_id"] == video.run_id
    offline_draft["run_id"] = video_draft["run_id"] = "ignored"
    assert offline_draft == video_draft


# Failure semantics --------------------------------------------------


def test_worker_fails_on_unknown_evidence_set(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage, store, service = _stack(tmp_path)
    evroot = _synthetic_set(monkeypatch, tmp_path)
    record = _submit(storage, service,
                     WORKFLOW_VIDEO_RECONSTRUCT,
                     _manifest_bytes("no_such_set"), "job.json")
    assert run_once(store, storage, "worker-1",
                    evidence_dir=evroot) == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_FAILED
    assert final.error is not None
    assert "no_such_set" in final.error
    assert final.artifact_ids == []


def test_worker_fails_on_missing_evidence(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage, store, service = _stack(tmp_path)
    evroot = tmp_path / "empty-root"
    evroot.mkdir()
    _allow_synth_set(monkeypatch)
    record = _submit(storage, service,
                     WORKFLOW_VIDEO_RECONSTRUCT,
                     _manifest_bytes(), "job.json")
    assert run_once(store, storage, "worker-1",
                    evidence_dir=evroot) == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_FAILED
    assert final.error is not None
    assert final.artifact_ids == []


def test_failed_video_job_is_not_retried(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage, store, service = _stack(tmp_path)
    evroot = _synthetic_set(monkeypatch, tmp_path)
    record = _submit(storage, service,
                     WORKFLOW_VIDEO_RECONSTRUCT,
                     b"{not json", "job.json")
    assert run_once(store, storage, "worker-1",
                    evidence_dir=evroot) == record.run_id
    assert run_once(store, storage, "worker-1",
                    evidence_dir=evroot) is None
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_FAILED
    assert final.attempts == 1


def test_crashed_video_job_recovers_on_next_pass(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage, store, service = _stack(tmp_path)
    evroot = _synthetic_set(monkeypatch, tmp_path)
    record = _submit(storage, service,
                     WORKFLOW_VIDEO_RECONSTRUCT,
                     _manifest_bytes(), "job.json")
    claimed = store.claim_next(
        "crashed-worker", "2020-01-01T00:00:00+00:00")
    assert claimed is not None
    assert run_once(store, storage, "worker-1",
                    evidence_dir=evroot) == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_COMPLETED
    assert final.attempts == 2


# Recorded evidence (host data, skipped when absent) ------------------


def _recorded_set() -> dict[str, Path]:
    spec = video_api.EVIDENCE_SETS["p04_full_4096"]
    return {
        "observations": RECORDED_ROOT / spec["observations"],
        "temporal": RECORDED_ROOT / spec["temporal"],
        "executions": RECORDED_ROOT / spec["executions"],
        "units": RECORDED_ROOT / spec["units"],
    }


def _recorded_available() -> bool:
    paths = _recorded_set()
    batches = sorted(paths["observations"].glob("batch_*.json")) \
        if paths["observations"].is_dir() else []
    return (bool(batches)
            and paths["temporal"].is_file()
            and paths["units"].is_file())


def _recorded_files() -> list[Path]:
    paths = _recorded_set()
    files = sorted(paths["observations"].glob("batch_*.json"))
    files.append(paths["temporal"])
    if paths["executions"].is_dir():
        files.extend(sorted(
            paths["executions"].glob("*/execution_report.json")))
    files.append(paths["units"])
    return files


def _hash_all(files: list[Path]) -> dict[str, str]:
    return {path.relative_to(REPO_ROOT).as_posix(): sha256_hex(
        path.read_bytes()) for path in files}


@pytest.mark.skipif(not _recorded_available(),
                    reason="recorded P04 evidence not present")
def test_recorded_p04_evidence_reconstructs(tmp_path: Path) -> None:
    storage, store, service = _stack(tmp_path)
    event_id = "scale_visibility_00007100_00007650"
    manifest_raw = _manifest_bytes(
        "p04_full_4096", event_id=event_id, video_id="P04_104")
    consumed = _recorded_files()
    before = _hash_all(consumed)
    record = _submit(storage, service,
                     WORKFLOW_VIDEO_RECONSTRUCT,
                     manifest_raw, "p04.json")
    assert run_once(store, storage, "worker-1",
                    evidence_dir=RECORDED_ROOT) == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_COMPLETED
    assert VIDEO_SUMMARY_ARTIFACT in final.artifact_ids
    summary = _read_json(storage, record.run_id,
                         VIDEO_SUMMARY_ARTIFACT)
    assert summary["draft_id"] == f"draft_P04_104_{event_id}_v1"
    assert summary["measurements"] > 0
    assert summary["inference_performed"] is False
    assert all(entry["present"] for entry in summary["inputs"])
    roles = {entry["role"] for entry in summary["inputs"]}
    assert {"observations", "temporal", "execution_report",
            "units"} <= roles
    applied = _read_json(storage, record.run_id,
                         "units_applied.json")
    assert applied["assignments"][0]["reviewer"] == "operator-1"
    assert _hash_all(consumed) == before
