"""Offline tests for the video_pipeline_offline workflow (M2-06C.4).

Synthetic MJPG videos are generated locally; inference uses the
deterministic fake client only. Covers worker success,
determinism, invalid inputs, uncertain and partial
observations, provenance, recovery, and an API-to-artifacts
pass. No network, no live models, no paid services.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from bakery_intelligence.api.jobs import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    JobService,
    WORKFLOW_VIDEO_PIPELINE_OFFLINE,
)
from bakery_intelligence.api.pipeline import (
    FIRSTPASS_REPORT_ARTIFACT,
    PIPELINE_SUMMARY_ARTIFACT,
    PipelineResumeError,
    run_pipeline_from_path,
    run_pipeline_offline,
)
from bakery_intelligence.api.storage import (
    LocalStorageBackend,
    sha256_hex,
)
from bakery_intelligence.api.store import SqliteJobStore
from bakery_intelligence.api.worker import run_once
from bakery_intelligence.inference.fake import (
    MODEL_ID as FAKE_MODEL_ID,
    PROVIDER_ID as FAKE_PROVIDER_ID,
    DeterministicFakeClient,
)
from bakery_intelligence.inference.firstpass import InferenceError


def _make_video(path: Path, fps: float = 10.0,
                frame_count: int = 30,
                shade_offset: int = 0) -> Path:
    """Write a synthetic MJPG video; skip when encoding fails.

    Output depends only on the encoding parameters, never on
    the filename; ``shade_offset`` shifts frame shades for
    tests needing deterministically different videos.
    """
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    if not writer.isOpened():
        pytest.skip("MJPG video encoding is unavailable.")
    for index in range(frame_count):
        shade = (index * 7 + shade_offset) % 256
        writer.write(np.full((48, 64, 3), shade, dtype=np.uint8))
    writer.release()
    return path


def _stack(tmp_path: Path):
    storage = LocalStorageBackend(tmp_path / "storage")
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    return storage, store, JobService(store)


def _submit_video(storage: LocalStorageBackend,
                  service: JobService, raw: bytes,
                  object_id: str = "clip.avi"):
    storage.store_source(object_id, io.BytesIO(raw))
    return service.submit(WORKFLOW_VIDEO_PIPELINE_OFFLINE,
                          source_object_id=object_id)


def _read_json(storage: LocalStorageBackend, run_id: str,
               artifact_id: str) -> Any:
    return json.loads(storage.read_artifact(run_id, artifact_id))


def test_worker_completes_pipeline(tmp_path: Path) -> None:
    storage, store, service = _stack(tmp_path)
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    record = _submit_video(storage, service, raw)
    assert run_once(store, storage, "worker-1") == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_COMPLETED
    assert final.source_sha256 == sha256_hex(raw)
    assert final.artifact_ids == [
        "action_events.json",
        "firstpass_report.json",
        "measurements.json",
        "pipeline_summary.json",
        "recipe_draft.json",
        "validation.json",
        "weighing_event.json",
    ]
    for artifact_id in final.artifact_ids:
        content = storage.read_artifact(record.run_id, artifact_id)
        assert final.artifact_sha256[artifact_id] == sha256_hex(
            content)
    summary = _read_json(storage, record.run_id,
                         PIPELINE_SUMMARY_ARTIFACT)
    assert summary["workflow"] == WORKFLOW_VIDEO_PIPELINE_OFFLINE
    assert summary["synthetic_inference"] is True
    assert summary["inference_performed"] is False
    assert summary["network_accessed"] is False
    assert summary["paid_services_used"] is False
    assert summary["fake_model_id"] == FAKE_MODEL_ID
    assert summary["fake_provider_id"] == FAKE_PROVIDER_ID
    assert summary["video"]["fps"] == pytest.approx(10.0)
    assert summary["video"]["frame_count"] == 30
    assert summary["extraction"]["frame_indices"] == [0, 10, 20]
    assert summary["firstpass"]["frames_observed"] == 3
    assert summary["firstpass"]["frames_failed"] == 0
    assert summary["firstpass"]["model"] == FAKE_MODEL_ID
    assert summary["measurements"] == 3
    draft = _read_json(storage, record.run_id, "recipe_draft.json")
    assert draft["draft_id"] == summary["draft_id"]
    assert draft["video_id"].startswith("upload_")


def test_pipeline_is_deterministic(tmp_path: Path) -> None:
    storage, store, service = _stack(tmp_path)
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    digest = sha256_hex(raw)
    first = _submit_video(storage, service, raw, "a.avi")
    second = _submit_video(storage, service, raw, "b.avi")
    assert run_once(store, storage, "worker-1") == first.run_id
    assert run_once(store, storage, "worker-1") == second.run_id
    assert service.get(first.run_id).status == STATUS_COMPLETED
    assert service.get(second.run_id).status == STATUS_COMPLETED
    for artifact_id in ("measurements.json", "validation.json",
                        "weighing_event.json",
                        "action_events.json"):
        assert (storage.read_artifact(first.run_id, artifact_id)
                == storage.read_artifact(second.run_id, artifact_id))
    first_fp = _read_json(storage, first.run_id,
                          "firstpass_report.json")
    second_fp = _read_json(storage, second.run_id,
                           "firstpass_report.json")
    assert first_fp["source_name"] == (
        f"video_{digest[:16]}_upload_a@1.0fps")
    assert second_fp["source_name"] == (
        f"video_{digest[:16]}_upload_b@1.0fps")
    first_fp["source_name"] = second_fp["source_name"] = "ignored"
    assert first_fp == second_fp
    first_draft = _read_json(storage, first.run_id,
                             "recipe_draft.json")
    second_draft = _read_json(storage, second.run_id,
                              "recipe_draft.json")
    assert first_draft["draft_id"] == second_draft["draft_id"]
    first_draft["run_id"] = second_draft["run_id"] = "ignored"
    assert first_draft == second_draft
    first_summary = _read_json(
        storage, first.run_id, PIPELINE_SUMMARY_ARTIFACT)
    second_summary = _read_json(
        storage, second.run_id, PIPELINE_SUMMARY_ARTIFACT)
    assert first_summary["source_object_id"] == "a.avi"
    assert second_summary["source_object_id"] == "b.avi"
    assert first_summary["source_sha256"] == digest
    assert second_summary["source_sha256"] == digest
    assert first_summary["video"]["file_name"] == "upload_a.avi"
    assert second_summary["video"]["file_name"] == "upload_b.avi"
    # Bundle fingerprints differ by design: observations embed the
    # staged source name, so identical bytes under different upload
    # names hash differently. The evidence content stays identical.
    assert len(first_summary["bundle_sha256"]) == 64
    assert len(second_summary["bundle_sha256"]) == 64
    assert (first_summary["bundle_sha256"]
            != second_summary["bundle_sha256"])
    assert (first_summary["extraction"]
            == second_summary["extraction"])
    assert first_summary["run_id"] != second_summary["run_id"]


def test_worker_fails_on_non_video(tmp_path: Path) -> None:
    storage, store, service = _stack(tmp_path)
    record = _submit_video(storage, service, b"\x00" * 4096)
    assert run_once(store, storage, "worker-1") == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_FAILED
    assert final.error is not None
    assert "Cannot open video container" in final.error
    assert final.artifact_ids == []


def test_worker_fails_on_oversized_video(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "BAKERY_PIPELINE_MAX_VIDEO_BYTES", str(16 * 1024 * 1024))
    storage, store, service = _stack(tmp_path)
    record = _submit_video(storage, service, b"\x01" * (17 * 1024 * 1024))
    assert run_once(store, storage, "worker-1") == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_FAILED
    assert final.error is not None
    assert "limit is 16777216" in final.error
    assert final.artifact_ids == []


def test_uncertain_and_empty_observations(tmp_path: Path) -> None:
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    abstain = json.dumps({"scale_readable": False,
                          "confidence": "high"})
    fake = DeterministicFakeClient(
        overrides={10: "", 20: abstain})
    artifacts = run_pipeline_offline(
        video_bytes=raw, video_name="clip.avi", run_id="testrun",
        source_object_id="clip.avi",
        source_sha256=sha256_hex(raw),
        work_dir=tmp_path / "work", fake=fake)
    report = json.loads(artifacts[FIRSTPASS_REPORT_ARTIFACT])
    assert report["lifecycle"]["terminal_states"] == {
        "abstained": 1, "empty_output": 1, "valid_prediction": 1}
    measurements = json.loads(artifacts["measurements.json"])
    by_frame = {item["frame_index"]: item
                for item in measurements["measurements"]}
    assert by_frame[0]["value"] == 120.0
    assert by_frame[0]["value_source"] == "firstpass"
    assert by_frame[10]["value"] is None
    assert by_frame[20]["value"] is None
    assert artifacts["recipe_draft.json"] is not None


def test_partial_transport_failure_is_explicit(
        tmp_path: Path) -> None:
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    fake = DeterministicFakeClient(overrides={
        10: InferenceError("BAD_REQUEST", "nope")})
    artifacts = run_pipeline_offline(
        video_bytes=raw, video_name="clip.avi", run_id="testrun",
        source_object_id="clip.avi",
        source_sha256=sha256_hex(raw),
        work_dir=tmp_path / "work", fake=fake)
    report = json.loads(artifacts[FIRSTPASS_REPORT_ARTIFACT])
    assert report["frames_observed"] == 2
    assert report["frames_failed"] == 1
    assert report["batches_needing_review"] == 1
    measurements = json.loads(artifacts["measurements.json"])
    by_frame = {item["frame_index"]: item
                for item in measurements["measurements"]}
    assert by_frame[10]["value"] is None
    assert by_frame[10]["value_source"] == "none"
    assert by_frame[10]["value_status"] == "missing"


def test_pipeline_missing_frame_uses_video_fps(
        tmp_path: Path) -> None:
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    fake = DeterministicFakeClient(overrides={
        10: InferenceError("BAD_REQUEST", "nope")})
    artifacts = run_pipeline_offline(
        video_bytes=raw, video_name="clip.avi", run_id="testrun",
        source_object_id="clip.avi",
        source_sha256=sha256_hex(raw),
        work_dir=tmp_path / "work", fake=fake)
    measurements = json.loads(artifacts["measurements.json"])
    by_frame = {item["frame_index"]: item
                for item in measurements["measurements"]}
    assert by_frame[10]["timestamp_seconds"] == pytest.approx(1.0)
    assert by_frame[10]["original"] is None
    assert by_frame[10]["value_source"] == "none"
    assert by_frame[10]["value_status"] == "missing"
    assert by_frame[0]["timestamp_seconds"] == pytest.approx(0.0)
    assert by_frame[20]["timestamp_seconds"] == pytest.approx(2.0)


def test_total_inference_failure_fails_explicitly(
        tmp_path: Path) -> None:
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    fake = DeterministicFakeClient(overrides={
        0: InferenceError("BAD_REQUEST", "nope"),
        10: InferenceError("BAD_REQUEST", "nope"),
        20: InferenceError("BAD_REQUEST", "nope"),
    })
    with pytest.raises(ValueError, match="no observations"):
        run_pipeline_offline(
            video_bytes=raw, video_name="clip.avi",
            run_id="testrun", source_object_id="clip.avi",
            source_sha256=sha256_hex(raw),
            work_dir=tmp_path / "work", fake=fake)


def test_provenance_chain(tmp_path: Path) -> None:
    storage, store, service = _stack(tmp_path)
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    record = _submit_video(storage, service, raw)
    assert run_once(store, storage, "worker-1") == record.run_id
    summary = _read_json(storage, record.run_id,
                         PIPELINE_SUMMARY_ARTIFACT)
    assert summary["source_sha256"] == sha256_hex(raw)
    assert summary["video"]["sha256"] == sha256_hex(raw)
    assert summary["video"]["file_name"] == "upload_clip.avi"
    assert len(summary["extraction"]["manifest_sha256"]) == 64
    assert len(summary["bundle_sha256"]) == 64
    firstpass = _read_json(storage, record.run_id,
                           FIRSTPASS_REPORT_ARTIFACT)
    assert firstpass["model"] == FAKE_MODEL_ID
    assert firstpass["provider"] == FAKE_PROVIDER_ID


def test_failed_pipeline_job_is_not_retried(
        tmp_path: Path) -> None:
    storage, store, service = _stack(tmp_path)
    record = _submit_video(storage, service, b"\x00" * 4096)
    assert run_once(store, storage, "worker-1") == record.run_id
    assert run_once(store, storage, "worker-1") is None
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_FAILED
    assert final.attempts == 1


def test_crashed_pipeline_job_recovers(tmp_path: Path) -> None:
    storage, store, service = _stack(tmp_path)
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    record = _submit_video(storage, service, raw)
    claimed = store.claim_next(
        "crashed-worker", "2020-01-01T00:00:00+00:00")
    assert claimed is not None
    assert run_once(store, storage, "worker-1") == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_COMPLETED
    assert final.attempts == 2


def test_api_end_to_end_pipeline(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from bakery_intelligence.api.app import build_app

    storage = LocalStorageBackend(tmp_path)
    db_path = tmp_path / "jobs.sqlite3"
    client = TestClient(build_app(storage=storage, db_path=db_path))
    store = SqliteJobStore(db_path)
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    response = client.post(
        "/api/analyses",
        files={"file": ("clip.avi", raw, "video/avi")},
        data={"workflow": WORKFLOW_VIDEO_PIPELINE_OFFLINE},
    )
    assert response.status_code == 202
    created = response.json()
    assert created["workflow"] == WORKFLOW_VIDEO_PIPELINE_OFFLINE
    assert run_once(store, storage, "worker-1") == created["run_id"]
    status = client.get(f"/api/analyses/{created['run_id']}")
    assert status.json()["status"] == "completed"
    draft = client.get(
        f"/api/analyses/{created['run_id']}/artifacts/recipe_draft.json")
    assert draft.status_code == 200
    assert draft.json()["draft_id"].startswith("draft_upload_")
    summary = client.get(
        f"/api/analyses/{created['run_id']}/artifacts/"
        f"{PIPELINE_SUMMARY_ARTIFACT}")
    assert summary.status_code == 200
    assert summary.json()["synthetic_inference"] is True


# Durable checkpoints and safe recovery (M2-06C.5/S2) --------------------


class _CountingFake(DeterministicFakeClient):
    """Deterministic fake that records submitted frame indices."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.indices: list[int] = []

    def submit(self, prompt_text: str, image_paths: Any,
               api_key: str, max_output_tokens: int) -> Any:
        stem = Path(str(image_paths[0])).stem
        self.indices.append(int(stem.split("_")[1]))
        return super().submit(prompt_text, image_paths, api_key,
                              max_output_tokens)


def _pipeline_kwargs(tmp_path: Path, raw: bytes,
                     work_dir: Path) -> dict[str, Any]:
    return {
        "video_bytes": raw,
        "video_name": "clip.avi",
        "run_id": "run_0123456789ab",
        "source_object_id": "clip.avi",
        "source_sha256": sha256_hex(raw),
        "work_dir": work_dir,
    }


def test_pipeline_crash_resumes_without_replay(
        tmp_path: Path) -> None:
    raw = _make_video(tmp_path / "clip.avi",
                      frame_count=70).read_bytes()
    work_dir = tmp_path / "work"
    kwargs = _pipeline_kwargs(tmp_path, raw, work_dir)
    crashing = _CountingFake(
        overrides={50: RuntimeError("simulated crash")})
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_pipeline_offline(fake=crashing, **kwargs)
    assert crashing.indices == [0, 10, 20, 30, 40, 50]
    state = json.loads((work_dir / "pipeline_state.json"
                        ).read_text(encoding="utf-8"))
    assert state["stage"] == "firstpass_running"
    clean = _CountingFake()
    artifacts = run_pipeline_offline(fake=clean, **kwargs)
    assert clean.indices == [50, 60]
    measurements = json.loads(artifacts["measurements.json"])
    assert sorted(item["frame_index"]
                  for item in measurements["measurements"]) == [
                      0, 10, 20, 30, 40, 50, 60]
    summary = json.loads(artifacts["pipeline_summary.json"])
    assert summary["resumed"] is True
    assert summary["firstpass"]["frames_observed"] == 7
    assert summary["firstpass"]["frames_failed"] == 0
    firstpass = json.loads(artifacts["firstpass_report.json"])
    assert firstpass["budget"]["prior_requests_submitted"] == 5
    assert firstpass["budget"]["requests_submitted"] == 2
    assert firstpass["budget"]["uncertain_frame_indices"] == [
        50, 60]


def test_pipeline_done_stage_skips_rerun(tmp_path: Path) -> None:
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    work_dir = tmp_path / "work"
    kwargs = _pipeline_kwargs(tmp_path, raw, work_dir)
    first = _CountingFake()
    before = run_pipeline_offline(fake=first, **kwargs)
    assert first.indices == [0, 10, 20]
    second = _CountingFake()
    after = run_pipeline_offline(fake=second, **kwargs)
    assert second.indices == []
    assert after["measurements.json"] == before[
        "measurements.json"]
    summary = json.loads(after["pipeline_summary.json"])
    assert summary["resumed"] is True


def test_pipeline_fresh_run_records_unresumed_state(
        tmp_path: Path) -> None:
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    work_dir = tmp_path / "work"
    artifacts = run_pipeline_offline(
        **_pipeline_kwargs(tmp_path, raw, work_dir))
    state = json.loads((work_dir / "pipeline_state.json"
                        ).read_text(encoding="utf-8"))
    assert state["state_version"] == "pipeline-workdir/1"
    assert state["stage"] == "firstpass_done"
    summary = json.loads(artifacts["pipeline_summary.json"])
    assert summary["resumed"] is False


def test_pipeline_resume_mismatches_refused(
        tmp_path: Path) -> None:
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    work_dir = tmp_path / "work"
    kwargs = _pipeline_kwargs(tmp_path, raw, work_dir)
    run_pipeline_offline(**kwargs)
    with pytest.raises(PipelineResumeError, match="sample_fps"):
        run_pipeline_offline(sample_fps=2.0, **kwargs)
    other = _make_video(tmp_path / "other.avi",
                        shade_offset=100).read_bytes()
    assert sha256_hex(other) != sha256_hex(raw)
    with pytest.raises(PipelineResumeError,
                       match="source_sha256"):
        run_pipeline_offline(video_bytes=other,
                             source_sha256=sha256_hex(other),
                             **{key: value for key, value in
                                 kwargs.items()
                                 if key not in ("video_bytes",
                                                "source_sha256")})
    alien = DeterministicFakeClient(model_id="alien-model")
    with pytest.raises(PipelineResumeError, match="model"):
        run_pipeline_offline(fake=alien, **kwargs)


def test_pipeline_tampered_frame_refused(tmp_path: Path) -> None:
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    work_dir = tmp_path / "work"
    kwargs = _pipeline_kwargs(tmp_path, raw, work_dir)
    run_pipeline_offline(**kwargs)
    frame = work_dir / "frames" / "frame_00000010.jpg"
    data = bytearray(frame.read_bytes())
    data[10] ^= 0xFF
    frame.write_bytes(bytes(data))
    with pytest.raises(PipelineResumeError, match="frame 10"):
        run_pipeline_offline(**kwargs)


def test_from_path_long_upload_name_keeps_mp4_container(
        tmp_path: Path) -> None:
    """Regression: a staged upload id must not truncate `.mp4` to `.m`.

    Mirrors the failed Meta run shape: the stored object id is
    ``upload_<sha16>_<original stem>.mp4`` (50 chars). Sanitizing
    the full filename truncated it to 48 chars, leaving a `.m`
    suffix that intake rejected as an unsupported container.
    Offline fake inference only; no billable client is involved.
    """
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    staged = tmp_path / "upload_18e2e3850f6922aa_P35_101_clip_0209_0300.mp4"
    staged.write_bytes(raw)
    object_id = staged.name
    artifacts = run_pipeline_from_path(
        video_path=staged, video_name=object_id,
        run_id="run_0123456789ab", source_object_id=object_id,
        work_dir=tmp_path / "work")
    summary = json.loads(artifacts[PIPELINE_SUMMARY_ARTIFACT])
    assert summary["video"]["file_name"].endswith(".mp4")
    assert summary["source_object_id"] == object_id
    input_dir = tmp_path / "work" / "input"
    input_names = [path.name for path in input_dir.iterdir()]
    assert len(input_names) == 1
    assert input_names[0].endswith(".mp4")


def test_pipeline_stateless_firstpass_debris_refused(
        tmp_path: Path) -> None:
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    work_dir = tmp_path / "work"
    kwargs = _pipeline_kwargs(tmp_path, raw, work_dir)
    crashing = _CountingFake(
        overrides={10: RuntimeError("simulated crash")})
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_pipeline_offline(fake=crashing, **kwargs)
    (work_dir / "pipeline_state.json").unlink()
    clean = _CountingFake()
    with pytest.raises(PipelineResumeError,
                       match="without pipeline state"):
        run_pipeline_offline(fake=clean, **kwargs)
    assert clean.indices == []


def test_worker_restart_resumes_pipeline_without_replay(
        tmp_path: Path) -> None:
    storage, store, service = _stack(tmp_path)
    raw = _make_video(tmp_path / "clip.avi",
                      frame_count=70).read_bytes()
    record = _submit_video(storage, service, raw)
    work_dir = storage.run_work_dir(record.run_id)
    crashing = _CountingFake(
        overrides={50: RuntimeError("simulated crash")})
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_pipeline_offline(
            video_bytes=raw, video_name="clip.avi",
            run_id=record.run_id, source_object_id="clip.avi",
            source_sha256=sha256_hex(raw), work_dir=work_dir,
            fake=crashing)
    assert crashing.indices == [0, 10, 20, 30, 40, 50]
    claimed = store.claim_next(
        "crashed-worker", "2020-01-01T00:00:00+00:00")
    assert claimed is not None
    assert run_once(store, storage, "worker-1") == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_COMPLETED
    assert final.attempts == 2
    firstpass = _read_json(storage, record.run_id,
                            FIRSTPASS_REPORT_ARTIFACT)
    assert firstpass["budget"]["prior_requests_submitted"] == 5
    assert firstpass["budget"]["requests_submitted"] == 2
    assert firstpass["budget"]["uncertain_frame_indices"] == [
        50, 60]
    summary = _read_json(storage, record.run_id,
                          PIPELINE_SUMMARY_ARTIFACT)
    assert summary["resumed"] is True
    assert summary["firstpass"]["frames_observed"] == 7
