"""Offline adapter tests for the contextual FirstPass engine.

Experiment branch only. No network, no credentials, no
inference: deterministic fakes and scripted doubles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bakery_intelligence.api.multimodal import (
    _global_context_for_firstpass,
    frame_stage_calls,
)
from bakery_intelligence.api.pipeline import (
    FIRSTPASS_BATCH_GROUP_SIZE,
    FIRSTPASS_ENGINE_BATCH,
    FIRSTPASS_ENGINE_FOCUSED,
    FIRSTPASS_ENGINE_SINGLE,
    PipelineResumeError,
    firstpass_engine_from_env,
    run_pipeline_from_path,
)
from bakery_intelligence.inference.firstpass import TIMESTAMP_BASIS

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - optional test dep
    cv2 = None
    np = None


def _make_video(path: Path, frame_count: int = 120) -> Path:
    if cv2 is None or np is None:
        pytest.skip("cv2/numpy are unavailable.")
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        (64, 48))
    if not writer.isOpened():
        pytest.skip("MJPG video encoding is unavailable.")
    for index in range(frame_count):
        shade = (index * 7) % 256
        writer.write(
            np.full((48, 64, 3), shade, dtype=np.uint8))
    writer.release()
    return path


def _run_kwargs(video: Path, work_dir: Path,
                **over: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "video_path": video,
        "video_name": video.name,
        "run_id": "run_0123456789ab",
        "source_object_id": video.name,
        "work_dir": work_dir,
    }
    kwargs.update(over)
    return kwargs


def _read_report(work_dir: Path) -> dict[str, Any]:
    return json.loads((
        work_dir / "firstpass" / "reports"
        / "run_report.json").read_text(encoding="utf-8"))


def _read_observations(work_dir: Path) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for path in sorted(
            (work_dir / "firstpass" / "observations").glob(
                "*.json")):
        found.extend(json.loads(path.read_text(
            encoding="utf-8")))
    return found


def _run_script(*argv: str):
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, "scripts/benchmark_firstpass.py",
         *argv],
        capture_output=True, text=True, cwd=Path.cwd())


def test_benchmark_script_help() -> None:
    completed = _run_script("--help")
    assert completed.returncode == 0
    assert "--execute" in completed.stdout


def test_benchmark_dry_run_plans_without_executing(
        tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    completed = _run_script(
        "--video", str(video), "--engine",
        "contextual_batch", "--limit-frames", "30")
    assert completed.returncode == 0
    lines = completed.stdout.splitlines()
    stop = lines.index(
        "dry-run: add --execute to run billable inference.")
    plan = json.loads(chr(10).join(lines[:stop]))
    assert plan["engine"] == "contextual_batch"
    assert plan["token_budget"] == 16384
    assert plan["max_requests"] == 4
    assert plan["cost_ceiling_usd"] == 2.0


def test_benchmark_missing_video_fails() -> None:
    completed = _run_script(
        "--video", "does-not-exist.mp4")
    assert completed.returncode == 2


def test_engine_from_env_defaults_and_validates(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FIRSTPASS_ENGINE", raising=False)
    assert firstpass_engine_from_env() == "single_frame"
    monkeypatch.setenv("FIRSTPASS_ENGINE", "contextual_batch")
    assert firstpass_engine_from_env() == "contextual_batch"
    monkeypatch.setenv("FIRSTPASS_ENGINE", "turbo")
    with pytest.raises(ValueError, match="FIRSTPASS_ENGINE"):
        firstpass_engine_from_env()


def test_unknown_engine_rejected(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "clip.avi")
    with pytest.raises(ValueError, match="firstpass_engine"):
        run_pipeline_from_path(**_run_kwargs(
            video, tmp_path / "work",
            firstpass_engine="turbo"))


def test_batch_rejects_single_frame_fake(tmp_path: Path) -> None:
    from bakery_intelligence.inference.fake import (
        DeterministicFakeClient,
    )

    video = _make_video(tmp_path / "clip.avi")
    with pytest.raises(ValueError, match="incompatible"):
        run_pipeline_from_path(**_run_kwargs(
            video, tmp_path / "work",
            fake=DeterministicFakeClient(),
            firstpass_engine=FIRSTPASS_ENGINE_BATCH))


def test_contextual_fake_run_groups_and_orders(
        tmp_path: Path) -> None:
    video = _make_video(tmp_path / "clip.avi")
    work_dir = tmp_path / "work"
    artifacts = run_pipeline_from_path(**_run_kwargs(
        video, work_dir,
        firstpass_engine=FIRSTPASS_ENGINE_BATCH))
    assert "firstpass_report.json" in artifacts
    report = _read_report(work_dir)
    assert report["pass_id"] == "firstpass_v2"
    assert report["group_size"] == FIRSTPASS_BATCH_GROUP_SIZE
    assert report["requests_sent"] == 2
    assert report["groups_valid"] == 2
    assert report["groups_review"] == 0
    assert report["frames_total"] == 12
    assert report["frames_observed"] == 12
    assert report["frames_failed"] == 0
    assert (work_dir / "firstpass" / "contexts").is_dir()
    observations = _read_observations(work_dir)
    assert len(observations) == 12
    indices = [item["frame_index"] for item in observations]
    assert indices == sorted(indices)
    assert len(set(indices)) == 12
    stamps = [item["timestamp_seconds"]
              for item in observations]
    assert stamps == sorted(stamps)
    assert all(
        item["timestamp_basis"] == TIMESTAMP_BASIS
        for item in observations
    )


def test_contextual_resume_reuses_without_rebill(
        tmp_path: Path) -> None:
    video = _make_video(tmp_path / "clip.avi")
    work_dir = tmp_path / "work"
    kwargs = _run_kwargs(
        video, work_dir,
        firstpass_engine=FIRSTPASS_ENGINE_BATCH)
    first = run_pipeline_from_path(**kwargs)
    before = sorted(
        (work_dir / "firstpass" / "raw_responses").glob(
            "*.json"))
    second = run_pipeline_from_path(**kwargs)
    after = sorted(
        (work_dir / "firstpass" / "raw_responses").glob(
            "*.json"))
    assert [p.name for p in before] == [
        p.name for p in after]
    assert (first["firstpass_report.json"]
            == second["firstpass_report.json"])


def test_cross_engine_resume_refused(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "clip.avi")
    work_dir = tmp_path / "work"
    run_pipeline_from_path(**_run_kwargs(
        video, work_dir,
        firstpass_engine=FIRSTPASS_ENGINE_SINGLE))
    with pytest.raises(PipelineResumeError, match="engine"):
        run_pipeline_from_path(**_run_kwargs(
            video, work_dir,
            firstpass_engine=FIRSTPASS_ENGINE_BATCH))
    work_dir2 = tmp_path / "work2"
    run_pipeline_from_path(**_run_kwargs(
        video, work_dir2,
        firstpass_engine=FIRSTPASS_ENGINE_BATCH))
    with pytest.raises(PipelineResumeError, match="engine"):
        run_pipeline_from_path(**_run_kwargs(
            video, work_dir2,
            firstpass_engine=FIRSTPASS_ENGINE_SINGLE))


def test_legacy_state_backfills_single_engine(
        tmp_path: Path) -> None:
    video = _make_video(tmp_path / "clip.avi")
    work_dir = tmp_path / "work"
    run_pipeline_from_path(**_run_kwargs(
        video, work_dir,
        firstpass_engine=FIRSTPASS_ENGINE_SINGLE))
    state_path = work_dir / "pipeline_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    del state["firstpass"]["engine"]
    del state["firstpass"]["group_size"]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    resumed = run_pipeline_from_path(**_run_kwargs(
        video, work_dir,
        firstpass_engine=FIRSTPASS_ENGINE_SINGLE))
    assert "firstpass_report.json" in resumed


def test_contextual_billable_double_groups_requests(
        tmp_path: Path) -> None:
    from bakery_intelligence.inference.firstpass_v2 import (
        DeterministicGroupFakeClient,
    )

    video = _make_video(tmp_path / "clip.avi")
    work_dir = tmp_path / "work"
    double = DeterministicGroupFakeClient()
    kwargs = _run_kwargs(
        video, work_dir,
        billable_client=double,
        billable_token_budget=16384,
        billable_max_requests=10,
        billable_rates_usd_per_m=(1.25, 4.25),
        firstpass_engine=FIRSTPASS_ENGINE_BATCH)
    artifacts = run_pipeline_from_path(**kwargs)
    assert len(double.calls) == 2
    report = json.loads(artifacts[
        "firstpass_report.json"].decode("utf-8"))
    assert report["billable"] is True
    assert report["requests_sent"] == 2
    assert report["frames_observed"] == 12
    assert report["cost_usd_estimated"] is not None
    # Resume issues zero new billable submits.
    run_pipeline_from_path(**kwargs)
    assert len(double.calls) == 2


def test_frame_stage_calls_carries_group_usage() -> None:
    calls = frame_stage_calls({
        "input_tokens": 10, "output_tokens": 20,
        "requests_sent": 2, "batches_completed": 0,
        "groups_completed": 2, "groups_valid": 2,
        "cost_usd_estimated": 1.5})
    assert calls[0].usage["groups_completed"] == 2
    assert calls[0].usage["groups_valid"] == 2
    legacy = frame_stage_calls({
        "input_tokens": 10, "output_tokens": 20,
        "requests_sent": 5, "batches_completed": 1})
    assert legacy[0].usage["groups_completed"] == 0
    assert legacy[0].usage["groups_valid"] == 0


def test_global_context_helper_bounds_and_absent() -> None:
    assert _global_context_for_firstpass(None) is None
    assert _global_context_for_firstpass({}) is None
    assert _global_context_for_firstpass(
        {"result": None}) is None
    context = _global_context_for_firstpass(
        {"result": {"phases": ["weigh", "mix"]}})
    assert context is not None
    assert "weigh" in context
    big = _global_context_for_firstpass(
        {"result": {"blob": "x" * 10000}})
    assert big is not None and len(big) <= 4000


# --- Contextual-batch retry, early-stop, and budget policy. ---


def _png_bytes() -> bytes:
    return b"\x89PNG\r\n\x1a\n"


def _v2_frames(tmp_path: Path, count: int) -> list[Any]:
    from bakery_intelligence.inference.firstpass import FrameRef

    img_dir = tmp_path / "v2imgs"
    img_dir.mkdir(exist_ok=True)
    frames = []
    for index in range(count):
        path = img_dir / f"frame_{index:08d}.png"
        path.write_bytes(_png_bytes())
        frames.append(FrameRef(
            frame_index=index, image_path=path,
            timestamp_seconds=float(index)))
    return frames


def _v2_source() -> Any:
    from bakery_intelligence.inference.firstpass import SourceRef

    return SourceRef(name="test@1fps", sha256="ab" * 32)


class _ScriptedGroupClient:
    """Per-submit scripted GroupClient double.

    Steps are consumed in order: "valid" delegates to the
    deterministic fake, str steps become completed text
    responses, InferenceResponse values return as-is, and
    Exceptions raise. Every submit records its budget.
    """

    def __init__(self, steps: list[Any]) -> None:
        from bakery_intelligence.inference.firstpass_v2 import (
            DeterministicGroupFakeClient,
        )

        self._steps = list(steps)
        self._fake = DeterministicGroupFakeClient()
        self.calls: list[dict[str, Any]] = []

    @property
    def model_id(self) -> str:
        return "scripted-group-v1"

    def submit(self, prompt_text: str, image_paths: Any,
               api_key: str, max_output_tokens: int,
               response_schema: Any = None,
               response_schema_name: Any = None) -> Any:
        _ = (prompt_text, api_key, response_schema)
        self.calls.append({
            "images": len(list(image_paths)),
            "max_output_tokens": max_output_tokens,
            "schema": response_schema_name,
        })
        if not self._steps:
            raise AssertionError("script exhausted")
        step = self._steps.pop(0)
        if callable(step):
            step = step(image_paths)
        if step == "valid":
            return self._fake.submit(
                prompt_text, image_paths, api_key,
                max_output_tokens, response_schema,
                response_schema_name)
        if isinstance(step, str):
            return _v2_response(
                step, request_id=f"r{len(self.calls)}")
        if isinstance(step, Exception):
            raise step
        return step


def _v2_config(tmp_path: Path, frames: list[Any],
               client: Any, **over: Any) -> Any:
    from bakery_intelligence.inference.firstpass_v2 import (
        FirstPassV2Config,
    )

    kwargs: dict[str, Any] = {
        "frames": frames,
        "source": _v2_source(),
        "out_dir": tmp_path / "fp",
        "group_size": 2,
        "max_attempts": 1,
        "max_validation_attempts": 3,
        "max_consecutive_exhausted": 2,
        "client_factory": lambda: client,
        "sleeper": lambda seconds: None,
    }
    kwargs.update(over)
    return FirstPassV2Config(**kwargs)


def _v2_response(text: str, *, status: str = "completed",
                 request_id: str = "r1",
                 input_tokens: int = 100,
                 output_tokens: int = 50,
                 reasoning: int | None = None,
                 incomplete: str | None = None) -> Any:
    from bakery_intelligence.inference.firstpass import (
        InferenceResponse,
        InferenceUsage,
    )

    return InferenceResponse(
        model="scripted-group-v1", status=status,
        output_text=text,
        usage=InferenceUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            reasoning_output_tokens=reasoning),
        latency_seconds=1.25, request_id=request_id,
        incomplete_reason=incomplete)


def _abstention_text(indices: list[int]) -> str:
    from bakery_intelligence.inference.firstpass import (
        MODEL_OUTPUT_FIELDS,
    )

    entries = [{
        "frame_index": index,
        "observation": {
            name: None for name in MODEL_OUTPUT_FIELDS},
        "verified": [],
        "uncertain": ["unreadable"],
        "hypotheses": [],
    } for index in indices]
    return json.dumps({
        "frames": entries,
        "rolling_context": "nothing readable; abstaining.",
    })


def _abstain_for(image_paths: Any) -> str:
    """Build an abstention reply for the submitted images."""
    indices = []
    for path in image_paths:
        stem = Path(str(path)).stem
        indices.append(int(stem.split("_")[1]))
    return _abstention_text(indices)


def _billable_batch_kwargs(video: Path, work_dir: Path,
                           client: Any, **over: Any
                           ) -> dict[str, Any]:
    kwargs = _run_kwargs(
        video, work_dir, billable_client=client,
        billable_token_budget=4096,
        billable_max_requests=30,
        billable_rates_usd_per_m=(1.25, 4.25),
        firstpass_engine=FIRSTPASS_ENGINE_BATCH)
    kwargs.update(over)
    return kwargs


def _raw_group(tmp_path: Path, position: int = 0) -> dict[str, Any]:
    paths = sorted(
        (tmp_path / "fp" / "raw_responses").glob("group_*.json"))
    return json.loads(paths[position].read_text(encoding="utf-8"))


def test_batch_token_budget_derives_from_grant() -> None:
    from bakery_intelligence.api.pipeline import _batch_token_budget

    assert _batch_token_budget(4096) == 32768
    assert _batch_token_budget(1024) == 10240
    assert _batch_token_budget(100) == 1000
    for bad in (0, -5, True, "4096"):
        with pytest.raises(ValueError):
            _batch_token_budget(bad)  # type: ignore[arg-type]


def test_batch_effective_budget_reaches_request(
        tmp_path: Path) -> None:
    video = _make_video(tmp_path / "clip.avi")
    work_dir = tmp_path / "work"
    client = _ScriptedGroupClient(["valid"] * 4)
    run_pipeline_from_path(**_run_kwargs(
        video, work_dir, billable_client=client,
        billable_token_budget=4096,
        billable_max_requests=10,
        billable_rates_usd_per_m=(1.25, 4.25),
        firstpass_engine=FIRSTPASS_ENGINE_BATCH))
    assert len(client.calls) == 2
    assert {call["max_output_tokens"]
            for call in client.calls} == {32768}
    report = _read_report(work_dir)
    assert report["token_budget_per_group"] == 32768
    state = json.loads((
        work_dir / "pipeline_state.json").read_text(
            encoding="utf-8"))
    assert state["firstpass"]["token_budgets"] == [32768]
    assert state["firstpass"]["max_validation_attempts"] == 3
    assert state["firstpass"]["max_consecutive_exhausted"] == 2
    observations = _read_observations(work_dir)
    assert observations[0]["call"]["max_output_tokens"] == 32768


def test_batch_offline_budget_scales_default(
        tmp_path: Path) -> None:
    video = _make_video(tmp_path / "clip.avi")
    work_dir = tmp_path / "work"
    run_pipeline_from_path(**_run_kwargs(
        video, work_dir,
        firstpass_engine=FIRSTPASS_ENGINE_BATCH))
    report = _read_report(work_dir)
    assert report["token_budget_per_group"] == 10240
    state = json.loads((
        work_dir / "pipeline_state.json").read_text(
            encoding="utf-8"))
    assert state["firstpass"]["token_budgets"] == [10240]


def test_validation_retry_succeeds_second_attempt(
        tmp_path: Path) -> None:
    from bakery_intelligence.inference.firstpass_v2 import (
        FirstPassV2Runner,
    )

    frames = _v2_frames(tmp_path, 2)
    client = _ScriptedGroupClient(["not json {{{", "valid"])
    report = FirstPassV2Runner(_v2_config(
        tmp_path, frames, client)).run()
    assert report["groups_valid"] == 1
    assert report["requests_sent"] == 2
    assert report["groups"][0]["submits"] == 2
    raw = _raw_group(tmp_path)
    assert len(raw["attempts"]) == 2
    first, second = raw["attempts"]
    assert first["validation"].startswith("validation:")
    assert first["request_id"] == "r1"
    assert first["latency_seconds"] == 1.25
    assert first["input_tokens"] == 100
    assert first["slept_seconds"] == 0.5
    assert second["validation"] == "ok"


def test_validation_retry_succeeds_third_attempt(
        tmp_path: Path) -> None:
    from bakery_intelligence.inference.firstpass_v2 import (
        FirstPassV2Runner,
    )

    frames = _v2_frames(tmp_path, 2)
    truncated = _v2_response(
        "", status="incomplete",
        incomplete="max_output_tokens limit reached",
        request_id="rt")
    client = _ScriptedGroupClient(["{{{bad", truncated, "valid"])
    report = FirstPassV2Runner(_v2_config(
        tmp_path, frames, client)).run()
    assert report["groups_valid"] == 1
    assert report["requests_sent"] == 3
    raw = _raw_group(tmp_path)
    assert len(raw["attempts"]) == 3
    assert raw["attempts"][1]["validation"].startswith(
        "token_budget_exhausted")
    assert raw["attempts"][2]["validation"] == "ok"


def test_exhaustion_after_three_attempts(
        tmp_path: Path) -> None:
    from bakery_intelligence.inference.firstpass_v2 import (
        FirstPassV2Runner,
    )

    frames = _v2_frames(tmp_path, 2)
    client = _ScriptedGroupClient(["bad"] * 5)
    report = FirstPassV2Runner(_v2_config(
        tmp_path, frames, client)).run()
    assert report["requests_sent"] == 3
    assert report["groups_review"] == 1
    assert report["groups_exhausted"] == 1
    group = report["groups"][0]
    assert group["status"] == "needs_human_review"
    assert group["exhausted"] is True
    assert group["terminal_state"] == "invalid_json"
    raw = _raw_group(tmp_path)
    assert len(raw["attempts"]) == 3
    assert all(
        note["validation"].startswith("validation:")
        for note in raw["attempts"])


def test_no_retry_on_ambiguous_transport(
        tmp_path: Path) -> None:
    from bakery_intelligence.inference.firstpass import (
        TERMINAL_FAILED,
        InferenceError,
    )
    from bakery_intelligence.inference.firstpass_v2 import (
        FirstPassV2Runner,
    )

    frames = _v2_frames(tmp_path, 2)
    client = _ScriptedGroupClient(
        [InferenceError("TIMEOUT", "timed out")] * 3)
    report = FirstPassV2Runner(_v2_config(
        tmp_path, frames, client)).run()
    assert report["requests_sent"] == 1
    group = report["groups"][0]
    assert group["terminal_state"] == TERMINAL_FAILED
    raw = _raw_group(tmp_path)
    assert len(raw["attempts"]) == 1
    assert "slept_seconds" not in raw["attempts"][0]
    assert raw["failure_reason"] == "transport:TIMEOUT"


def test_no_submit_after_budget_halt(tmp_path: Path) -> None:
    from bakery_intelligence.inference.firstpass_v2 import (
        FirstPassV2Runner,
    )

    frames = _v2_frames(tmp_path, 4)
    client = _ScriptedGroupClient(["valid"] * 4)
    report = FirstPassV2Runner(_v2_config(
        tmp_path, frames, client, max_groups=0)).run()
    assert report["requests_sent"] == 0
    assert report["stopped_reason"] == "request_limit_reached"
    assert report["groups"][0]["status"] == "budget_stopped"
    assert len(client.calls) == 0


def test_early_stop_after_two_consecutive_exhausted(
        tmp_path: Path) -> None:
    from bakery_intelligence.inference.firstpass_v2 import (
        FirstPassV2Runner,
    )

    frames = _v2_frames(tmp_path, 6)
    client = _ScriptedGroupClient(["bad"] * 12)
    report = FirstPassV2Runner(_v2_config(
        tmp_path, frames, client)).run()
    assert report["groups_total"] == 3
    assert len(report["groups"]) == 2
    assert report["requests_sent"] == 6
    assert report["groups_exhausted"] == 2
    assert (report["stopped_reason"]
            == "consecutive_group_exhaustion")
    assert (report["budget"]["remaining_frame_indices"]
            == [4, 5])
    assert len(client.calls) == 6


def test_consecutive_counter_resets_after_success(
        tmp_path: Path) -> None:
    from bakery_intelligence.inference.firstpass_v2 import (
        FirstPassV2Runner,
    )

    frames = _v2_frames(tmp_path, 10)
    client = _ScriptedGroupClient(
        ["bad"] * 3 + ["valid"] + ["bad"] * 6)
    report = FirstPassV2Runner(_v2_config(
        tmp_path, frames, client)).run()
    assert report["requests_sent"] == 10
    assert report["groups_valid"] == 1
    assert report["groups_exhausted"] == 3
    assert (report["stopped_reason"]
            == "consecutive_group_exhaustion")
    assert (report["budget"]["remaining_frame_indices"]
            == [8, 9])


def test_valid_abstention_no_retry(tmp_path: Path) -> None:
    from bakery_intelligence.inference.firstpass_v2 import (
        FirstPassV2Runner,
    )

    frames = _v2_frames(tmp_path, 2)
    client = _ScriptedGroupClient([_abstention_text([0, 1])])
    report = FirstPassV2Runner(_v2_config(
        tmp_path, frames, client)).run()
    assert report["requests_sent"] == 1
    assert report["groups_valid"] == 1
    raw = _raw_group(tmp_path)
    assert len(raw["attempts"]) == 1
    assert raw["attempts"][0]["validation"] == "ok"


def test_transport_halt_effective_under_strict(
        tmp_path: Path) -> None:
    from bakery_intelligence.inference.firstpass import (
        InferenceError,
    )
    from bakery_intelligence.inference.firstpass_v2 import (
        FirstPassV2Runner,
    )

    frames = _v2_frames(tmp_path, 4)
    client = _ScriptedGroupClient(
        [InferenceError("NETWORK_ERROR", "down")] * 4)
    report = FirstPassV2Runner(_v2_config(
        tmp_path, frames, client, strict_cost=True,
        cost_ceiling_usd=20.0,
        rates_usd_per_m=(1.25, 4.25),
        uncertain_policy="refuse")).run()
    assert report["requests_sent"] == 1
    assert (report["stopped_reason"]
            == "unknown_cost_strict_halt")
    assert len(report["groups"]) == 1
    assert len(client.calls) == 1


def test_request_counting_across_retries(
        tmp_path: Path) -> None:
    from bakery_intelligence.inference.firstpass_v2 import (
        FirstPassV2Runner,
    )

    frames = _v2_frames(tmp_path, 8)
    client = _ScriptedGroupClient(
        ["bad", "valid", "valid", "valid", "valid"])
    report = FirstPassV2Runner(_v2_config(
        tmp_path, frames, client, max_groups=4)).run()
    assert report["requests_sent"] == 4
    assert report["requests_total"] == 4
    assert (report["stopped_reason"]
            == "request_limit_reached")
    assert report["groups"][-1]["status"] == "budget_stopped"
    assert report["groups"][-1]["submits"] == 0


def test_resume_after_early_stop_submits_nothing(
        tmp_path: Path) -> None:
    from bakery_intelligence.inference.firstpass_v2 import (
        FirstPassV2Runner,
    )

    frames = _v2_frames(tmp_path, 6)
    failing = _ScriptedGroupClient(["bad"] * 12)
    first = FirstPassV2Runner(_v2_config(
        tmp_path, frames, failing)).run()
    assert first["stopped_reason"] == (
        "consecutive_group_exhaustion")
    passing = _ScriptedGroupClient(["valid"] * 6)
    second = FirstPassV2Runner(_v2_config(
        tmp_path, frames, passing)).run()
    assert len(passing.calls) == 0
    assert second["requests_sent"] == 0
    assert second["groups_resumed"] == 2
    assert (second["stopped_reason"]
            == "consecutive_group_exhaustion")


def test_firstpass_elapsed_flows_to_stage_call() -> None:
    payload = {
        "input_tokens": 10, "output_tokens": 20,
        "requests_sent": 2, "batches_completed": 0,
        "groups_completed": 2, "groups_valid": 2,
        "cost_usd_estimated": 1.5}
    calls = frame_stage_calls(payload, elapsed_seconds=12.5)
    assert calls[0].elapsed_seconds == 12.5
    default = frame_stage_calls(payload)
    assert default[0].elapsed_seconds == 0.0


def test_no_single_frame_fallback_on_batch_failure(
        tmp_path: Path) -> None:
    video = _make_video(tmp_path / "clip.avi")
    work_dir = tmp_path / "work"
    client = _ScriptedGroupClient(["{{{" ] * 12)
    with pytest.raises(ValueError,
                        match="consecutive_group_exhaustion"):
        run_pipeline_from_path(**_run_kwargs(
            video, work_dir, billable_client=client,
            billable_token_budget=4096,
            billable_max_requests=30,
            billable_rates_usd_per_m=(1.25, 4.25),
            firstpass_engine=FIRSTPASS_ENGINE_BATCH))
    report = _read_report(work_dir)
    assert report["frames_observed"] == 0
    obs_files = sorted(
        (work_dir / "firstpass" / "observations").glob("*.json"))
    assert obs_files
    assert all(path.name.startswith("group_")
               for path in obs_files)
    assert not [path for path in obs_files
                if path.name.startswith("batch_")]


def test_reasoning_tokens_aggregated(tmp_path: Path) -> None:
    from bakery_intelligence.inference.firstpass_v2 import (
        FirstPassV2Runner,
    )

    frames = _v2_frames(tmp_path, 4)
    steps = [
        _v2_response(_abstention_text([0, 1]), reasoning=100,
                     request_id="a"),
        _v2_response(_abstention_text([2, 3]), reasoning=50,
                     request_id="b"),
    ]
    client = _ScriptedGroupClient(steps)
    report = FirstPassV2Runner(_v2_config(
        tmp_path, frames, client)).run()
    assert report["groups_valid"] == 2
    assert report["reasoning_output_tokens"] == 150


def test_retry_policy_config_validation(tmp_path: Path) -> None:
    from bakery_intelligence.inference.firstpass import (
        FirstPassError,
    )
    from bakery_intelligence.inference.firstpass_v2 import (
        FirstPassV2Runner,
    )

    frames = _v2_frames(tmp_path, 2)
    with pytest.raises(FirstPassError) as first:
        FirstPassV2Runner(_v2_config(
            tmp_path, frames, _ScriptedGroupClient([]),
            max_validation_attempts=0)).run()
    assert first.value.code == "INVALID_CONFIG"
    with pytest.raises(FirstPassError) as second:
        FirstPassV2Runner(_v2_config(
            tmp_path, frames, _ScriptedGroupClient([]),
            max_consecutive_exhausted=0)).run()
    assert second.value.code == "INVALID_CONFIG"


def test_terminated_firstpass_fails_despite_partial_observations(
        tmp_path: Path) -> None:
    video = _make_video(tmp_path / "clip.avi", frame_count=400)
    work_dir = tmp_path / "work"
    client = _ScriptedGroupClient(["valid"] + ["bad"] * 6)
    with pytest.raises(ValueError, match="terminated"):
        run_pipeline_from_path(**_billable_batch_kwargs(
            video, work_dir, client))
    # Group 0 valid, groups 1-2 exhausted, group 3 never sent.
    assert len(client.calls) == 7
    report = _read_report(work_dir)
    assert (report["stopped_reason"]
            == "consecutive_group_exhaustion")
    assert report["frames_observed"] == 10
    assert report["frames_failed"] == 30
    assert len(_read_observations(work_dir)) == 10
    assert (work_dir / "firstpass" / "checkpoints"
            / "checkpoints.jsonl").is_file()
    assert len(list((work_dir / "firstpass" / "raw_responses"
                     ).glob("*.json"))) == 3
    state = json.loads((
        work_dir / "pipeline_state.json").read_text(
            encoding="utf-8"))
    assert state["firstpass_report_sha256"] is None


def test_budget_halt_fails_with_partial_observations(
        tmp_path: Path) -> None:
    video = _make_video(tmp_path / "clip.avi")
    work_dir = tmp_path / "work"
    client = _ScriptedGroupClient(["valid"] * 4)
    with pytest.raises(ValueError, match="request_limit_reached"):
        run_pipeline_from_path(**_billable_batch_kwargs(
            video, work_dir, client, billable_max_requests=1))
    assert len(client.calls) == 1
    report = _read_report(work_dir)
    assert report["frames_observed"] == 10
    assert report["frames_failed"] == 2


def test_abstentions_do_not_fail_run(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "clip.avi")
    work_dir = tmp_path / "work"
    client = _ScriptedGroupClient(["valid", _abstain_for])
    run_pipeline_from_path(**_billable_batch_kwargs(
        video, work_dir, client))
    report = _read_report(work_dir)
    assert report["stopped_reason"] is None
    observations = _read_observations(work_dir)
    assert len(observations) == 12
    abstained = [
        item for item in observations
        if all(value is None
               for value in item["model_fields"].values())]
    assert len(abstained) == 2


def test_terminated_run_fails_again_on_resume_without_rebill(
        tmp_path: Path) -> None:
    video = _make_video(tmp_path / "clip.avi", frame_count=400)
    work_dir = tmp_path / "work"
    client = _ScriptedGroupClient(["valid"] + ["bad"] * 6)
    kwargs = _billable_batch_kwargs(video, work_dir, client)
    with pytest.raises(ValueError, match="terminated"):
        run_pipeline_from_path(**kwargs)
    assert len(client.calls) == 7
    with pytest.raises(ValueError, match="terminated"):
        run_pipeline_from_path(**kwargs)
    assert len(client.calls) == 7


def test_reject_terminated_firstpass_unit() -> None:
    from bakery_intelligence.api.pipeline import (
        _reject_terminated_firstpass,
    )

    _reject_terminated_firstpass(
        "run_x", {"stopped_reason": None})
    with pytest.raises(ValueError) as first:
        _reject_terminated_firstpass("run_x", {
            "stopped_reason": "consecutive_group_exhaustion",
            "frames_total": 40, "frames_observed": 10,
            "frames_failed": 30})
    assert "consecutive_group_exhaustion" in str(first.value)
    assert "10 of 40" in str(first.value)
    assert "abstentions" in str(first.value)
    with pytest.raises(ValueError) as second:
        _reject_terminated_firstpass("run_x", {
            "stopped_reason": "request_limit_reached",
            "frames_total": 645, "frames_observed": 600,
            "frames_failed": 45, "frames_unprocessed": 45})
    assert "request_limit_reached" in str(second.value)


def test_frame_timeout_seconds_selects_by_engine() -> None:
    from bakery_intelligence.api.pipeline import (
        FIRSTPASS_BATCH_TIMEOUT_SECONDS,
        frame_timeout_seconds,
    )

    assert FIRSTPASS_BATCH_TIMEOUT_SECONDS == 240.0
    assert frame_timeout_seconds(
        FIRSTPASS_ENGINE_BATCH, 120.0) == 240.0
    assert frame_timeout_seconds(
        FIRSTPASS_ENGINE_SINGLE, 45.5) == 45.5
    with pytest.raises(ValueError, match="firstpass_engine"):
        frame_timeout_seconds("turbo", 120.0)


def test_batch_builder_records_batch_timeout(
        tmp_path: Path) -> None:
    video = _make_video(tmp_path / "clip.avi")
    work_dir = tmp_path / "work"
    client = _ScriptedGroupClient(["valid"] * 4)
    run_pipeline_from_path(**_billable_batch_kwargs(
        video, work_dir, client))
    observations = _read_observations(work_dir)
    assert observations
    assert {item["call"]["timeout_seconds"]
            for item in observations} == {240.0}


def test_timeout_fingerprint_by_engine(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "clip.avi")
    batch_dir = tmp_path / "batch"
    run_pipeline_from_path(**_run_kwargs(
        video, batch_dir,
        firstpass_engine=FIRSTPASS_ENGINE_BATCH))
    batch_state = json.loads((
        batch_dir / "pipeline_state.json").read_text(
            encoding="utf-8"))
    assert (batch_state["firstpass"]["timeout_seconds"]
            == 240.0)
    single_dir = tmp_path / "single"
    run_pipeline_from_path(**_run_kwargs(
        video, single_dir,
        firstpass_engine=FIRSTPASS_ENGINE_SINGLE))
    single_state = json.loads((
        single_dir / "pipeline_state.json").read_text(
            encoding="utf-8"))
    assert single_state["firstpass"]["timeout_seconds"] is None


def test_legacy_batch_state_without_timeout_refuses_resume(
        tmp_path: Path) -> None:
    video = _make_video(tmp_path / "clip.avi")
    work_dir = tmp_path / "work"
    kwargs = _run_kwargs(
        video, work_dir,
        firstpass_engine=FIRSTPASS_ENGINE_BATCH)
    run_pipeline_from_path(**kwargs)
    state_path = work_dir / "pipeline_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    del state["firstpass"]["timeout_seconds"]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(PipelineResumeError,
                       match="timeout_seconds"):
        run_pipeline_from_path(**kwargs)


def test_client_timeout_reaches_sdk_factory() -> None:
    from bakery_intelligence.inference.meta_spark import (
        MetaSparkClient,
    )

    captured: dict[str, Any] = {}

    def factory(api_key: str, timeout: float) -> Any:
        captured["timeout"] = timeout
        return object()

    client = MetaSparkClient(
        model="muse-spark-1.3", timeout_seconds=240.0,
        client_factory=factory)
    assert client._require_client("k") is not None  # noqa: SLF001
    assert captured["timeout"] == 240.0


def _billable_focused_kwargs(video: Path, work_dir: Path,
                             client: Any, **over: Any
                             ) -> dict[str, Any]:
    kwargs = _run_kwargs(
        video, work_dir, billable_client=client,
        billable_token_budget=4096,
        billable_max_requests=30,
        billable_rates_usd_per_m=(1.25, 4.25),
        firstpass_engine=FIRSTPASS_ENGINE_FOCUSED)
    kwargs.update(over)
    return kwargs


def test_focused_engine_fake_run(tmp_path: Path) -> None:
    video = _make_video(tmp_path / "clip.avi")
    work_dir = tmp_path / "work"
    run_pipeline_from_path(**_run_kwargs(
        video, work_dir,
        firstpass_engine=FIRSTPASS_ENGINE_FOCUSED))
    report = _read_report(work_dir)
    assert report["pass_id"] == "firstpass_v3"
    assert report["groups_valid"] == 1
    assert report["frames_observed"] == 12
    assert report["stopped_reason"] is None
    observations = _read_observations(work_dir)
    assert len(observations) == 12
    assert observations[0]["pass_id"] == "firstpass_v3"


def test_focused_fingerprint_records_policy(
        tmp_path: Path) -> None:
    video = _make_video(tmp_path / "clip.avi")
    work_dir = tmp_path / "work"
    run_pipeline_from_path(**_run_kwargs(
        video, work_dir,
        firstpass_engine=FIRSTPASS_ENGINE_FOCUSED))
    state = json.loads((
        work_dir / "pipeline_state.json").read_text(
            encoding="utf-8"))
    firstpass = state["firstpass"]
    assert firstpass["engine"] == FIRSTPASS_ENGINE_FOCUSED
    assert firstpass["group_size"] == 50
    assert firstpass["token_budgets"] == [51200]
    assert firstpass["max_validation_attempts"] == 3
    assert firstpass["max_consecutive_exhausted"] == 2
    assert firstpass["timeout_seconds"] == 240.0


def test_focused_effective_budget_reaches_request(
        tmp_path: Path) -> None:
    from bakery_intelligence.inference.firstpass_v3 import (
        DeterministicFocusedFakeClient,
    )

    video = _make_video(tmp_path / "clip.avi")
    work_dir = tmp_path / "work"
    observed: dict[str, Any] = {}
    real_submit = DeterministicFocusedFakeClient.submit

    def spy_submit(self: Any, prompt_text: str,
                   image_paths: Any, api_key: str,
                   max_output_tokens: int,
                   response_schema: Any = None,
                   response_schema_name: Any = None) -> Any:
        observed["max_output_tokens"] = max_output_tokens
        observed["schema"] = response_schema_name
        return real_submit(
            self, prompt_text, image_paths, api_key,
            max_output_tokens, response_schema,
            response_schema_name)

    DeterministicFocusedFakeClient.submit = spy_submit  # type: ignore[method-assign]
    try:
        run_pipeline_from_path(**_run_kwargs(
            video, work_dir,
            firstpass_engine=FIRSTPASS_ENGINE_FOCUSED))
    finally:
        DeterministicFocusedFakeClient.submit = (  # type: ignore[method-assign]
            real_submit)
    assert observed["max_output_tokens"] == 51200
    assert observed["schema"] == "FocusedFrameBatch"


def test_focused_billable_budget_caps_at_max(
        tmp_path: Path) -> None:
    video = _make_video(tmp_path / "clip.avi")
    work_dir = tmp_path / "work"
    from bakery_intelligence.inference.firstpass_v3 import (
        DeterministicFocusedFakeClient,
    )

    run_pipeline_from_path(**_billable_focused_kwargs(
        video, work_dir, DeterministicFocusedFakeClient()))
    report = _read_report(work_dir)
    assert report["token_budget_per_group"] == 65536
    state = json.loads((
        work_dir / "pipeline_state.json").read_text(
            encoding="utf-8"))
    assert state["firstpass"]["token_budgets"] == [65536]


def test_focused_env_selection(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIRSTPASS_ENGINE", "focused_batch")
    assert firstpass_engine_from_env() == "focused_batch"
    monkeypatch.setenv("FIRSTPASS_ENGINE", "bogus")
    with pytest.raises(ValueError, match="FIRSTPASS_ENGINE"):
        firstpass_engine_from_env()


def test_focused_fake_override_refused(
        tmp_path: Path) -> None:
    from bakery_intelligence.inference.fake import (
        DeterministicFakeClient,
    )

    video = _make_video(tmp_path / "clip.avi")
    work_dir = tmp_path / "work"
    with pytest.raises(ValueError, match="incompatible"):
        run_pipeline_from_path(**_run_kwargs(
            video, work_dir, fake=DeterministicFakeClient(),
            firstpass_engine=FIRSTPASS_ENGINE_FOCUSED))


def test_focused_cross_engine_resume_refused(
        tmp_path: Path) -> None:
    video = _make_video(tmp_path / "clip.avi")
    work_dir = tmp_path / "work"
    run_pipeline_from_path(**_run_kwargs(
        video, work_dir,
        firstpass_engine=FIRSTPASS_ENGINE_SINGLE))
    with pytest.raises(PipelineResumeError, match="engine"):
        run_pipeline_from_path(**_run_kwargs(
            video, work_dir,
            firstpass_engine=FIRSTPASS_ENGINE_FOCUSED))


def test_focused_terminated_gate(tmp_path: Path) -> None:
    from bakery_intelligence.inference.firstpass_v3 import (
        DeterministicFocusedFakeClient,
    )

    video = _make_video(tmp_path / "clip.avi", frame_count=1200)
    work_dir = tmp_path / "work"
    script = {500: "{{{", 1000: "{{{"}
    client = DeterministicFocusedFakeClient(script=script)
    with pytest.raises(ValueError,
                       match="consecutive_group_exhaustion"):
        run_pipeline_from_path(**_billable_focused_kwargs(
            video, work_dir, client))
    report = _read_report(work_dir)
    assert report["groups_valid"] == 1
    assert report["frames_observed"] == 50


def test_frame_timeout_focused_matches_batch() -> None:
    from bakery_intelligence.api.pipeline import (
        frame_timeout_seconds,
    )

    assert frame_timeout_seconds(
        FIRSTPASS_ENGINE_FOCUSED, 120.0) == 240.0
    assert frame_timeout_seconds(
        FIRSTPASS_ENGINE_SINGLE, 45.0) == 45.0


def test_scale_engine_registered_and_env(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from bakery_intelligence.api.pipeline import (
        FIRSTPASS_ENGINE_SCALE,
        FIRSTPASS_ENGINES,
    )

    assert FIRSTPASS_ENGINE_SCALE == "parallel_scale"
    assert FIRSTPASS_ENGINE_SCALE in FIRSTPASS_ENGINES
    monkeypatch.setenv("FIRSTPASS_ENGINE", "parallel_scale")
    assert firstpass_engine_from_env() == "parallel_scale"


def test_scale_token_budget_derives_from_grant() -> None:
    from bakery_intelligence.api.pipeline import (
        _scale_token_budget,
    )

    assert _scale_token_budget(4096) == 8192
    assert _scale_token_budget(100) == 1000
    with pytest.raises(ValueError, match="grant token budget"):
        _scale_token_budget(0)


def test_scale_timeout_seconds() -> None:
    from bakery_intelligence.api.pipeline import (
        FIRSTPASS_ENGINE_SCALE,
        FIRSTPASS_SCALE_TIMEOUT_SECONDS,
        frame_timeout_seconds,
    )

    assert frame_timeout_seconds(
        FIRSTPASS_ENGINE_SCALE, 120.0) == (
            FIRSTPASS_SCALE_TIMEOUT_SECONDS)
    assert frame_timeout_seconds(
        FIRSTPASS_ENGINE_SINGLE, 120.0) == 120.0


def test_resolve_firstpass_engine_precedence(
        monkeypatch: pytest.MonkeyPatch) -> None:
    from bakery_intelligence.api.authorization import (
        GrantError,
    )
    from bakery_intelligence.api.pipeline import (
        FIRSTPASS_ENGINE_BATCH,
        FIRSTPASS_ENGINE_SCALE,
        resolve_firstpass_engine,
    )

    monkeypatch.delenv("FIRSTPASS_ENGINE", raising=False)
    assert resolve_firstpass_engine(None) == "single_frame"
    monkeypatch.setenv(
        "FIRSTPASS_ENGINE", FIRSTPASS_ENGINE_BATCH)
    assert resolve_firstpass_engine(None) == (
        FIRSTPASS_ENGINE_BATCH)
    monkeypatch.delenv("FIRSTPASS_ENGINE", raising=False)
    assert resolve_firstpass_engine(
        FIRSTPASS_ENGINE_SCALE) == FIRSTPASS_ENGINE_SCALE
    with pytest.raises(GrantError, match="firstpass_engine"):
        resolve_firstpass_engine("turbo")
    monkeypatch.setenv(
        "FIRSTPASS_ENGINE", FIRSTPASS_ENGINE_BATCH)
    with pytest.raises(GrantError, match="conflicts"):
        resolve_firstpass_engine(FIRSTPASS_ENGINE_SCALE)
    monkeypatch.setenv(
        "FIRSTPASS_ENGINE", FIRSTPASS_ENGINE_SCALE)
    assert resolve_firstpass_engine(
        FIRSTPASS_ENGINE_SCALE) == FIRSTPASS_ENGINE_SCALE


def test_scale_fake_run_batches_orders_adapts(
        tmp_path: Path) -> None:
    from bakery_intelligence.api.pipeline import (
        FIRSTPASS_ENGINE_SCALE,
        FIRSTPASS_SCALE_GROUP_SIZE,
    )

    video = _make_video(tmp_path / "clip.avi")
    work_dir = tmp_path / "work"
    artifacts = run_pipeline_from_path(**_run_kwargs(
        video, work_dir,
        firstpass_engine=FIRSTPASS_ENGINE_SCALE))
    assert "firstpass_report.json" in artifacts
    report = _read_report(work_dir)
    assert report["pass_id"] == "scale_pass_v1"
    assert report["group_size"] == FIRSTPASS_SCALE_GROUP_SIZE
    assert report["groups_valid"] == 2
    assert report["frames_total"] == 12
    assert report["frames_observed"] == 12
    assert report["frames_failed"] == 0
    assert report["requests_sent"] == 2
    assert report["stopped_reason"] is None
    sizes = [len(json.loads(
        (work_dir / "firstpass" / "observations"
         / f"batch_{index:03d}.json").read_text(
             encoding="utf-8"))) for index in range(2)]
    assert sizes == [10, 2]
    observations = _read_observations(work_dir)
    assert len(observations) == 12
    first = observations[0]
    assert first["model_fields"][
        "scale_display_verbatim"] == "0"
    assert first["model_fields"]["scale_readable"] is True
    assert first["model_fields"]["scale_unit_visible"] == "g"
    assert first["model_fields"]["scale_state"] == "unknown"
    assert first["model_fields"]["visible_objects"] == []
    assert first["model_fields"]["visual_frame_id"] == (
        "F000000")
    assert first["timestamp_basis"] == TIMESTAMP_BASIS
    manifest = json.loads(
        (work_dir / "firstpass" / "anchored"
         / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 12
    assert manifest[0]["photo_region_verified"] is True
    state = json.loads(
        (work_dir / "pipeline_state.json").read_text(
            encoding="utf-8"))
    assert state["firstpass"]["engine"] == (
        FIRSTPASS_ENGINE_SCALE)
    assert state["firstpass"]["group_size"] == 10
    assert state["firstpass"]["timeout_seconds"] == 240.0
    assert state["firstpass"]["token_budgets"] == [8192]
    assert state["firstpass"]["max_validation_attempts"] == 1


def test_scale_rejects_single_frame_fake(
        tmp_path: Path) -> None:
    from bakery_intelligence.api.pipeline import (
        FIRSTPASS_ENGINE_SCALE,
    )
    from bakery_intelligence.inference.fake import (
        DeterministicFakeClient,
    )

    video = _make_video(tmp_path / "clip.avi")
    with pytest.raises(ValueError, match="incompatible"):
        run_pipeline_from_path(**_run_kwargs(
            video, tmp_path / "work",
            fake=DeterministicFakeClient(),
            firstpass_engine=FIRSTPASS_ENGINE_SCALE))


def test_scale_cross_engine_resume_refused(
        tmp_path: Path) -> None:
    from bakery_intelligence.api.pipeline import (
        FIRSTPASS_ENGINE_SCALE,
    )

    video = _make_video(tmp_path / "clip.avi")
    work_dir = tmp_path / "work"
    run_pipeline_from_path(**_run_kwargs(
        video, work_dir,
        firstpass_engine=FIRSTPASS_ENGINE_SCALE))
    with pytest.raises(PipelineResumeError):
        run_pipeline_from_path(**_run_kwargs(
            video, work_dir,
            firstpass_engine=FIRSTPASS_ENGINE_SINGLE))

def test_scale_run_feeds_reconstruction_and_review(
        tmp_path: Path) -> None:
    from bakery_intelligence.api.jobs import JobRecord
    from bakery_intelligence.api.pipeline import (
        FIRSTPASS_ENGINE_SCALE,
    )
    from bakery_intelligence.api.review import (
        build_review_bundle,
    )
    from bakery_intelligence.api.storage import (
        LocalStorageBackend,
    )

    storage = LocalStorageBackend(tmp_path / "storage")
    run_id = "run_0123456789ab"
    work_dir = (storage.base_dir / "runs" / run_id
                / "work")
    video = _make_video(tmp_path / "clip.avi")
    artifacts = run_pipeline_from_path(**_run_kwargs(
        video, work_dir, run_id=run_id,
        firstpass_engine=FIRSTPASS_ENGINE_SCALE))
    measurements = json.loads(
        artifacts["measurements.json"].decode("utf-8"))
    assert measurements["measurements"], (
        "reconstruction consumed no scale readings")
    assert json.loads(
        artifacts["weighing_event.json"].decode(
            "utf-8")) is not None
    assert json.loads(
        artifacts["recipe_draft.json"].decode(
            "utf-8")) is not None
    record = JobRecord(
        run_id=run_id,
        workflow="video_pipeline_offline")
    bundle = build_review_bundle(storage, record)
    assert len(bundle["frames"]) == 12
    first = bundle["frames"][0]["observation"]
    assert first["model_fields"][
        "scale_display_verbatim"] == "0"
    assert first["request_id"] == "fake-scale-1"
    assert bundle["frames"][0]["image_available"] is True
