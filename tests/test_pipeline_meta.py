"""Offline Meta-backed pipeline integration (M2-06C.5/S4-B).

Exercises the pipeline billable seam end to end with
MetaSparkClient over a recording fake SDK: success,
fingerprints, crash/uncertain refusal, review resume
without replay, unknown-cost accounting, and proof that
no registered workflow can trigger the Meta client. No
network, no credentials, no paid calls.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from bakery_intelligence.api import video as video_api
from bakery_intelligence.api.jobs import (
    STATUS_COMPLETED,
    SUPPORTED_WORKFLOWS,
    JobService,
    WORKFLOW_OFFLINE_RECONSTRUCT,
    WORKFLOW_STORAGE_SMOKE,
    WORKFLOW_VIDEO_PIPELINE_OFFLINE,
    WORKFLOW_VIDEO_RECONSTRUCT,
)
from bakery_intelligence.api.pipeline import (
    META_PROVIDER_ID,
    PIPELINE_BATCH_SIZE,
    PIPELINE_PASS_ID,
    PIPELINE_PROMPT,
    PipelineResumeError,
    run_pipeline_offline,
)
from bakery_intelligence.api.storage import (
    LocalStorageBackend,
    sha256_hex,
)
from bakery_intelligence.api.store import SqliteJobStore
from bakery_intelligence.api.video import VIDEO_MANIFEST_FORMAT
from bakery_intelligence.api.worker import run_once
from bakery_intelligence.inference.fake import DeterministicFakeClient
from bakery_intelligence.inference.firstpass import (
    FirstPassError,
    prompt_hash,
)
from bakery_intelligence.inference.meta_spark import (
    MODEL_SPARK,
    MetaSparkClient,
)

VALID_OBS = json.dumps({
    "visible_objects": ["scale"],
    "scale_display_verbatim": "120",
    "scale_readable": True,
    "scale_state": "stable",
    "tare_action_visible": False,
    "addition_or_removal_visible": "none",
    "container_description": "bowl",
    "container_changed_hint": False,
    "confidence": "high",
    "ambiguity_notes": None,
})

FRAMES = (7400, 7450, 7500, 7550, 7600)
VERBATIMS = ("353", "421", "5 15", "575", "643")
BUNDLE_VIDEO_ID = "P04_104"
BUNDLE_EVENT_ID = "scale_visibility_test"
VID_VIDEO_ID = "SYNTH.VID.001"
VID_EVENT_ID = "scale_visibility_vid"
SYNTH_SET = "synth_video_test"
_SYNTH_SPEC = {
    "observations": "synth/obs",
    "temporal": "synth/temporal.json",
    "executions": "synth/exec",
    "units": "synth/units.json",
}


class _APIStatusError(Exception):
    """Fake SDK HTTP error with status and headers."""

    def __init__(self, message: str, *, status_code: Any,
                 headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(headers=headers or {})


class _FakeSDK:
    """Recording stand-in for the OpenAI SDK client."""

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    @property
    def responses(self) -> _FakeSDK:
        return self

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        action = self._script.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def _ok_response(request_id: str,
                 usage: Any = "default") -> SimpleNamespace:
    if usage == "default":
        usage = {"input_tokens": 100, "output_tokens": 50,
                 "total_tokens": 150}
    return SimpleNamespace(
        status="completed", output_text=VALID_OBS,
        usage=usage, model=MODEL_SPARK, id=request_id,
        incomplete_details=None)


def _meta_client(script: list[Any]
                 ) -> tuple[MetaSparkClient, _FakeSDK]:
    sdk = _FakeSDK(script)
    client = MetaSparkClient(
        client_factory=lambda key, timeout: sdk)
    return client, sdk


def _make_video(path: Path, fps: float = 10.0,
                frame_count: int = 30) -> Path:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    if not writer.isOpened():
        pytest.skip("MJPG video encoding is unavailable.")
    for index in range(frame_count):
        shade = (index * 7) % 256
        writer.write(np.full((48, 64, 3), shade, dtype=np.uint8))
    writer.release()
    return path


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


def _read_state(work_dir: Path) -> dict[str, Any]:
    return json.loads((work_dir / "pipeline_state.json"
                       ).read_text(encoding="utf-8"))


def _read_observations(work_dir: Path) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for path in sorted(
            (work_dir / "firstpass" / "observations").glob("*.json")):
        found.extend(json.loads(path.read_text(encoding="utf-8")))
    return found


def _stack(tmp_path: Path):
    storage = LocalStorageBackend(tmp_path / "storage")
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    return storage, store, JobService(store)


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


def _temporal_events(event_id: str) -> list[dict[str, Any]]:
    return [
        {"event_id": event_id,
         "candidate_type": "scale_visibility",
         "start_timestamp_seconds": 148.0,
         "end_timestamp_seconds": 152.0,
         "verification_status": "needs_review",
         "frame_indices": list(FRAMES),
         "review_reasons": ["temporal_inconsistency"],
         "uncertainty_notes": []},
        {"event_id": f"action_{event_id}",
         "candidate_type": "action_addition",
         "start_timestamp_seconds": 148.0,
         "end_timestamp_seconds": 152.0,
         "verification_status": "candidate",
         "frame_indices": list(FRAMES),
         "review_reasons": [],
         "uncertainty_notes": []},
    ]


def _bundle_bytes() -> bytes:
    payload = {
        "bundle_format": "bakery-offline-bundle/1",
        "video_id": BUNDLE_VIDEO_ID,
        "event_id": BUNDLE_EVENT_ID,
        "observations": [_obs_record(frame, verbatim)
                         for frame, verbatim in zip(FRAMES,
                                                   VERBATIMS)],
        "temporal": {"artifact_version": "1.1.0",
                     "events": _temporal_events(BUNDLE_EVENT_ID)},
    }
    return (json.dumps(payload) + "\n").encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8")


def _evidence_root(tmp_path: Path) -> Path:
    root = tmp_path / "evidence"
    first = [_obs_record(frame, verbatim)
             for frame, verbatim in zip(FRAMES[:3], VERBATIMS[:3])]
    second = [_obs_record(frame, verbatim)
              for frame, verbatim in zip(FRAMES[3:], VERBATIMS[3:])]
    _write_json(root / "synth" / "obs" / "batch_a.json", first)
    _write_json(root / "synth" / "obs" / "batch_b.json", second)
    _write_json(root / "synth" / "temporal.json",
                {"artifact_version": "1.1.0",
                 "events": _temporal_events(VID_EVENT_ID)})
    return root


def _manifest_bytes() -> bytes:
    payload = {
        "manifest_format": VIDEO_MANIFEST_FORMAT,
        "evidence_set": SYNTH_SET,
        "event_id": VID_EVENT_ID,
        "video_id": VID_VIDEO_ID,
    }
    return (json.dumps(payload) + "\n").encode("utf-8")


def test_meta_success_end_to_end(tmp_path: Path) -> None:
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    work_dir = tmp_path / "work"
    ids = ["resp_meta_1", "resp_meta_2", "resp_meta_3"]
    client, sdk = _meta_client([_ok_response(request_id)
                                for request_id in ids])
    artifacts = run_pipeline_offline(
        billable_client=client,
        **_pipeline_kwargs(tmp_path, raw, work_dir))
    assert sdk.calls and len(sdk.calls) == 3
    assert {call["model"] for call in sdk.calls} == {MODEL_SPARK}
    assert {call["max_output_tokens"]
            for call in sdk.calls} == {1024}
    summary = json.loads(artifacts["pipeline_summary.json"])
    assert summary["firstpass"]["model"] == MODEL_SPARK
    assert summary["firstpass"]["provider"] == META_PROVIDER_ID
    assert summary["firstpass"]["frames_observed"] == 3
    assert summary["firstpass"]["frames_failed"] == 0
    assert summary["inference_performed"] is False
    assert summary["network_accessed"] is False
    assert summary["paid_services_used"] is False
    assert summary["synthetic_inference"] is True
    assert "caller-supplied" in summary["note"]
    assert "test-double" in summary["note"]
    report = json.loads(artifacts["firstpass_report.json"])
    assert report["billable"] is True
    assert report["provider"] == META_PROVIDER_ID
    assert report["model"] == MODEL_SPARK
    assert report["input_tokens"] == 300
    assert report["output_tokens"] == 150
    assert report["budget"]["requests_submitted"] == 3
    assert report["budget"]["stopped_reason"] is None
    measurements = json.loads(artifacts["measurements.json"])
    assert sorted(item["frame_index"]
                  for item in measurements["measurements"]) == [
                      0, 10, 20]
    observations = _read_observations(work_dir)
    assert sorted(item["frame_index"]
                  for item in observations) == [0, 10, 20]
    assert sorted(item["timestamp_seconds"]
                  for item in observations) == [0.0, 1.0, 2.0]
    assert sorted(item["call"]["request_id"]
                  for item in observations) == sorted(ids)


def test_meta_custom_token_budget_honored(tmp_path: Path) -> None:
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    work_dir = tmp_path / "work"
    client, sdk = _meta_client(
        [_ok_response(f"resp_{index}") for index in range(3)])
    run_pipeline_offline(
        billable_client=client, billable_token_budget=512,
        **_pipeline_kwargs(tmp_path, raw, work_dir))
    assert len(sdk.calls) == 3
    assert {call["max_output_tokens"]
            for call in sdk.calls} == {512}
    assert _read_state(work_dir)["firstpass"][
        "token_budgets"] == [512]


def test_meta_rejects_ambiguous_clients(tmp_path: Path) -> None:
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    work_dir = tmp_path / "work"
    client, _ = _meta_client([])
    with pytest.raises(ValueError, match="at most one"):
        run_pipeline_offline(
            fake=DeterministicFakeClient(),
            billable_client=client,
            **_pipeline_kwargs(tmp_path, raw, work_dir))
    assert not (work_dir / "pipeline_state.json").exists()


@pytest.mark.parametrize("budget", [0, -1, True, "1024"])
def test_meta_rejects_bad_budget(tmp_path: Path,
                                 budget: Any) -> None:
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    work_dir = tmp_path / "work"
    client, _ = _meta_client([])
    with pytest.raises(ValueError, match="positive int"):
        run_pipeline_offline(
            billable_client=client,
            billable_token_budget=budget,
            **_pipeline_kwargs(tmp_path, raw, work_dir))
    assert not (work_dir / "pipeline_state.json").exists()


def test_meta_fingerprint_records_policy(tmp_path: Path) -> None:
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    work_dir = tmp_path / "work"
    client, _ = _meta_client(
        [_ok_response(f"resp_{index}") for index in range(3)])
    run_pipeline_offline(
        billable_client=client,
        **_pipeline_kwargs(tmp_path, raw, work_dir))
    assert _read_state(work_dir)["firstpass"] == {
        "model": MODEL_SPARK,
        "provider": META_PROVIDER_ID,
        "pass_id": PIPELINE_PASS_ID,
        "batch_size": PIPELINE_BATCH_SIZE,
        "token_budgets": [1024],
        "prompt_sha256": prompt_hash(PIPELINE_PROMPT),
        "max_requests": None,
        "cost_ceiling_usd": None,
        "strict_cost": False,
        "max_attempts": 1,
        "billable": True,
        "uncertain_policy": "refuse",
    }


def test_meta_cross_path_resume_refused(tmp_path: Path) -> None:
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    fake_dir = tmp_path / "fake_work"
    run_pipeline_offline(
        **_pipeline_kwargs(tmp_path, raw, fake_dir))
    meta_client, _ = _meta_client([])
    with pytest.raises(PipelineResumeError, match="difference"):
        run_pipeline_offline(
            billable_client=meta_client,
            **_pipeline_kwargs(tmp_path, raw, fake_dir))
    meta_dir = tmp_path / "meta_work"
    first, _ = _meta_client(
        [_ok_response(f"resp_{index}") for index in range(3)])
    run_pipeline_offline(
        billable_client=first,
        **_pipeline_kwargs(tmp_path, raw, meta_dir))
    with pytest.raises(PipelineResumeError, match="difference"):
        run_pipeline_offline(
            **_pipeline_kwargs(tmp_path, raw, meta_dir))


def test_meta_uncertain_crash_refuses(tmp_path: Path) -> None:
    raw = _make_video(tmp_path / "clip.avi",
                      frame_count=70).read_bytes()
    work_dir = tmp_path / "work"
    kwargs = _pipeline_kwargs(tmp_path, raw, work_dir)
    # Crash after the 6th dispatch but before its result is
    # recorded: the clock raises outside the adapter's mapped
    # region, so the provider may have been hit (true
    # charge-then-crash uncertainty).
    doomed_sdk = _FakeSDK(
        [_ok_response(f"resp_{index}") for index in range(6)])
    ticks = iter(range(100))

    def crashing_clock() -> float:
        tick = next(ticks)
        if tick == 11:
            raise RuntimeError("simulated crash")
        return float(tick)

    crashing = MetaSparkClient(
        client_factory=lambda key, timeout: doomed_sdk,
        clock=crashing_clock)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_pipeline_offline(billable_client=crashing, **kwargs)
    assert len(doomed_sdk.calls) == 6
    assert _read_state(work_dir)["stage"] == "firstpass_running"
    clean, sdk = _meta_client(
        [_ok_response(f"retry_{index}") for index in range(7)])
    with pytest.raises(FirstPassError) as caught:
        run_pipeline_offline(billable_client=clean, **kwargs)
    assert caught.value.code == "UNCERTAIN_REQUESTS"
    assert sdk.calls == []


def test_meta_review_preserved_without_replay(
        tmp_path: Path) -> None:
    # A certainly-failed frame (HTTP 400, recorded outcome)
    # stays failed: the completed first pass verifies and
    # reuses its outputs instead of re-dispatching. No
    # automatic billable replay exists; retrying failed
    # frames needs an explicit authorized mechanism (future
    # work, not an implicit resume side effect).
    raw = _make_video(tmp_path / "clip.avi",
                      frame_count=70).read_bytes()
    work_dir = tmp_path / "work"
    kwargs = _pipeline_kwargs(tmp_path, raw, work_dir)
    script: list[Any] = [
        _ok_response(f"resp_r1_{index}") for index in range(7)]
    script[2] = _APIStatusError("bad frame", status_code=400)
    first, _ = _meta_client(script)
    partial = run_pipeline_offline(
        billable_client=first, **kwargs)
    report = json.loads(partial["firstpass_report.json"])
    assert report["frames_observed"] == 6
    assert report["frames_failed"] == 1
    assert report["batches_needing_review"] == 1
    assert report["budget"]["requests_submitted"] == 7
    assert _read_state(work_dir)["stage"] == "firstpass_done"
    second, sdk = _meta_client(
        [_ok_response(f"retry_{index}") for index in range(7)])
    resumed = run_pipeline_offline(
        billable_client=second, **kwargs)
    assert sdk.calls == []
    assert resumed["firstpass_report.json"] == (
        partial["firstpass_report.json"])
    summary = json.loads(resumed["pipeline_summary.json"])
    assert summary["resumed"] is True
    assert summary["firstpass"]["frames_observed"] == 6
    assert summary["firstpass"]["frames_failed"] == 1
    # Established artifact semantics: every temporal-window
    # frame gets a measurement row, and the failed frame's
    # row is an explicit missing placeholder with no value,
    # no source, and no original fields.
    measurements = json.loads(resumed["measurements.json"])
    by_frame = {item["frame_index"]: item
                for item in measurements["measurements"]}
    assert sorted(by_frame) == [0, 10, 20, 30, 40, 50, 60]
    failed = by_frame[20]
    assert failed["value"] is None
    assert failed["value_source"] == "none"
    assert failed["value_status"] == "missing"
    assert failed["original"] is None
    assert failed["verified"] is None
    assert failed["delta_from_previous"] is None
    assert failed["timestamp_seconds"] == pytest.approx(2.0)
    usable = by_frame[0]
    assert usable["value"] == 120.0
    assert usable["value_source"] == "firstpass"
    assert usable["value_status"] == "ok"
    # The failed frame leaks into no quantity, ingredient,
    # or reading: the weighing's final reading is the last
    # usable frame, and the draft claims no ingredients.
    weighing = json.loads(resumed["weighing_event.json"])
    final = weighing["final_display_reading"]
    assert final["value"] == 120.0
    assert final["observation_id"] == "m_00000060"
    draft = json.loads(resumed["recipe_draft.json"])
    assert draft["ingredients"] == []
    observations = _read_observations(work_dir)
    assert sorted(item["frame_index"]
                  for item in observations) == [
                      0, 10, 30, 40, 50, 60]


def test_meta_unknown_cost_never_zero(tmp_path: Path) -> None:
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    work_dir = tmp_path / "work"
    client, sdk = _meta_client(
        [_ok_response(f"resp_{index}", usage=None)
         for index in range(3)])
    artifacts = run_pipeline_offline(
        billable_client=client,
        **_pipeline_kwargs(tmp_path, raw, work_dir))
    assert len(sdk.calls) == 3
    report = json.loads(artifacts["firstpass_report.json"])
    assert report["input_tokens"] == 0
    assert report["output_tokens"] == 0
    budget = report["budget"]
    assert budget["requests_submitted"] == 3
    assert budget["cost_unknown_requests"] == 3
    assert budget["spent_cost_usd_estimate"] is None
    assert budget["spent_cost_known_usd_estimate"] == 0.0
    measurements = json.loads(artifacts["measurements.json"])
    assert len(measurements["measurements"]) == 3


def test_registered_workflows_cannot_trigger_meta(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(
            "MetaSparkClient must not trigger via a "
            "registered workflow.")

    monkeypatch.setattr(MetaSparkClient, "__init__", explode)
    monkeypatch.setattr(MetaSparkClient, "submit", explode)
    monkeypatch.setitem(video_api.EVIDENCE_SETS, SYNTH_SET,
                        _SYNTH_SPEC)
    storage, store, service = _stack(tmp_path)
    evroot = _evidence_root(tmp_path)
    video_raw = _make_video(tmp_path / "clip.avi").read_bytes()

    def submit_smoke() -> Any:
        storage.store_source("clip01.mp4",
                             io.BytesIO(b"fake-video"))
        return service.submit(WORKFLOW_STORAGE_SMOKE,
                              source_object_id="clip01.mp4")

    def submit_offline() -> Any:
        storage.store_source("bundle.json",
                             io.BytesIO(_bundle_bytes()))
        return service.submit(WORKFLOW_OFFLINE_RECONSTRUCT,
                              source_object_id="bundle.json")

    def submit_video() -> Any:
        storage.store_source("job.json",
                             io.BytesIO(_manifest_bytes()))
        return service.submit(WORKFLOW_VIDEO_RECONSTRUCT,
                              source_object_id="job.json")

    def submit_pipeline() -> Any:
        storage.store_source("clip.avi", io.BytesIO(video_raw))
        return service.submit(WORKFLOW_VIDEO_PIPELINE_OFFLINE,
                              source_object_id="clip.avi")

    cases = {
        WORKFLOW_STORAGE_SMOKE: submit_smoke,
        WORKFLOW_OFFLINE_RECONSTRUCT: submit_offline,
        WORKFLOW_VIDEO_RECONSTRUCT: submit_video,
        WORKFLOW_VIDEO_PIPELINE_OFFLINE: submit_pipeline,
    }
    assert set(cases) == set(SUPPORTED_WORKFLOWS)
    for workflow, setup in cases.items():
        record = setup()
        if workflow == WORKFLOW_VIDEO_RECONSTRUCT:
            assert run_once(store, storage, "worker-1",
                            evidence_dir=evroot) == record.run_id
        else:
            assert run_once(store, storage, "worker-1") == (
                record.run_id)
        final = service.get(record.run_id)
        assert final is not None
        assert final.status == STATUS_COMPLETED
