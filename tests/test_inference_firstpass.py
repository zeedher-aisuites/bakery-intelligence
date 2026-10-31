"""Offline tests for generalized first-pass inference (M2-06C.2).

A scripted fake client stands in for the provider seam: no
network, no credentials, no paid calls. Covers ordering,
batching, retries, budget escalation, partial failures, resume,
provenance, and structural compatibility with recorded P04
outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bakery_intelligence.inference.firstpass import (
    MODEL_OUTPUT_FIELDS,
    FirstPassConfig,
    FirstPassError,
    FirstPassRunner,
    FrameRef,
    InferenceError,
    InferenceResponse,
    InferenceUsage,
    SourceRef,
    prompt_hash,
)

FAKE_JPEG = b"\xff\xd8" + b"0" * 64 + b"\xff\xd9"
FAKE_PNG = b"\x89PNG" + b"0" * 64
PROMPT = "TEST PROMPT"
SOURCE = SourceRef(name="test-source", sha256="abc123")

VALID_JSON = json.dumps({
    "visible_objects": ["scale"], "scale_display_verbatim": "120",
    "scale_readable": True, "scale_state": "stable",
    "tare_action_visible": False,
    "addition_or_removal_visible": "none",
    "container_description": "bowl", "container_changed_hint": False,
    "confidence": "high", "ambiguity_notes": None,
})

REPO_ROOT = Path(__file__).resolve().parents[1]
RECORDED_RAW = (REPO_ROOT / "data" / "processed" / "experiment_006"
                / "full_run" / "p04_full_4096" / "raw_responses"
                / "batch_00000000-00000200.json")
RECORDED_OBS = (REPO_ROOT / "data" / "processed" / "experiment_006"
                / "full_run" / "p04_full_4096" / "observations"
                / "batch_00000000-00000200.json")


class _FakeClient:
    """Scripted provider double keyed by frame index."""

    def __init__(self, script: dict[int, list[Any]] | None = None,
                 default: Any = None) -> None:
        self._script = {key: list(value)
                        for key, value in (script or {}).items()}
        self._default = default
        self.calls: list[tuple[str, list[str], str]] = []
        self.budgets: list[int] = []

    def submit(self, prompt_text: str, image_paths: Any,
               api_key: str,
               max_output_tokens: int) -> InferenceResponse:
        self.calls.append((prompt_text,
                           [str(path) for path in image_paths],
                           api_key))
        self.budgets.append(max_output_tokens)
        stem = Path(str(image_paths[0])).stem
        index = int(stem.split("_")[1])
        queue = self._script.get(index, [])
        action = queue.pop(0) if queue else self._default
        if isinstance(action, Exception):
            raise action
        assert action is not None
        return action


def _frame(frames_dir: Path, index: int, timestamp: float,
           blob: bytes = FAKE_JPEG,
           suffix: str = ".jpg") -> FrameRef:
    frames_dir.mkdir(parents=True, exist_ok=True)
    path = frames_dir / f"frame_{index:08d}{suffix}"
    path.write_bytes(blob)
    return FrameRef(frame_index=index, image_path=path,
                    timestamp_seconds=timestamp)


def _ok(text: str = VALID_JSON, usage: InferenceUsage | None = None,
        **overrides: Any) -> InferenceResponse:
    payload: dict[str, Any] = {
        "model": "fake-model", "status": "completed",
        "output_text": text,
        "usage": usage or InferenceUsage(10, 20, 30),
        "latency_seconds": 0.1, "request_id": "req-1",
        "incomplete_reason": None,
    }
    payload.update(overrides)
    return InferenceResponse(**payload)


def _config(frames: list[FrameRef], out_dir: Path,
            fake: _FakeClient, **overrides: Any) -> FirstPassConfig:
    payload: dict[str, Any] = {
        "frames": frames, "source": SOURCE, "out_dir": out_dir,
        "model": "fake-model", "prompt_text": PROMPT,
        "client_factory": lambda: fake,
        "sleeper": lambda seconds: None,
    }
    payload.update(overrides)
    return FirstPassConfig(**payload)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _obs_index(out_dir: Path) -> dict[int, dict[str, Any]]:
    indexed = {}
    for path in sorted((out_dir / "observations").glob("*.json")):
        for item in _read_json(path):
            indexed[item["frame_index"]] = item
    return indexed


# Happy path and layout ------------------------------------------------


def test_run_produces_observations_and_layout(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames = [_frame(frames_dir, index, float(index))
              for index in (3, 1, 2, 9, 7)]
    fake = _FakeClient(default=_ok())
    out_dir = tmp_path / "run"
    report = FirstPassRunner(_config(
        frames, out_dir, fake, batch_size=2)).run()
    assert report["frames_total"] == 5
    assert report["frames_observed"] == 5
    assert report["frames_failed"] == 0
    assert report["batches_total"] == 3
    assert report["batches_completed"] == 3
    assert report["requests_sent"] == 5
    obs_files = sorted((out_dir / "observations").glob("*.json"))
    raw_files = sorted((out_dir / "raw_responses").glob("*.json"))
    assert [path.name for path in obs_files] == [
        "batch_00000001-00000002.json",
        "batch_00000003-00000007.json",
        "batch_00000009-00000009.json",
    ]
    assert [path.name for path in raw_files] == [
        path.name for path in obs_files]
    checkpoints = (out_dir / "checkpoints" / "checkpoints.jsonl"
                   ).read_text(encoding="utf-8").strip().split("\n")
    assert len(checkpoints) == 3
    assert all(json.loads(line)["status"] == "completed"
               for line in checkpoints)
    assert (out_dir / "checkpoints" / "run_state.json").is_file()
    assert (out_dir / "reports" / "run_report.json").is_file()
    indexed = _obs_index(out_dir)
    assert sorted(indexed) == [1, 2, 3, 7, 9]
    first = indexed[1]
    assert first["timestamp_seconds"] == 1.0
    assert first["batch_id"] == "batch_00000001-00000002"
    assert first["pass_id"] == "firstpass"
    assert first["terminal_state"] == "valid_prediction"
    assert first["model_fields"]["scale_display_verbatim"] == "120"
    assert first["manifest_ref"]["manifest_sha256"] == "abc123"
    assert first["call"]["prompt_sha256"] == prompt_hash(PROMPT)
    assert first["call"]["model"] == "fake-model"


def test_single_frame_single_batch(tmp_path: Path) -> None:
    frames = [_frame(tmp_path / "frames", 42, 4.2)]
    fake = _FakeClient(default=_ok())
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake, batch_size=5)).run()
    assert report["batches_total"] == 1
    assert (tmp_path / "run" / "observations"
            / "batch_00000042-00000042.json").is_file()


def test_exact_batch_multiple(tmp_path: Path) -> None:
    frames = [_frame(tmp_path / "frames", index, float(index))
              for index in (10, 20, 30, 40)]
    fake = _FakeClient(default=_ok())
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake, batch_size=2)).run()
    assert report["batches_total"] == 2
    assert report["frames_observed"] == 4


def test_png_frames_accepted(tmp_path: Path) -> None:
    frames = [_frame(tmp_path / "frames", 5, 0.5, FAKE_PNG, ".png")]
    fake = _FakeClient(default=_ok())
    FirstPassRunner(_config(frames, tmp_path / "run", fake)).run()
    indexed = _obs_index(tmp_path / "run")
    assert indexed[5]["image_ref"]["encoding"] == "png"


# Configuration failures (before any request) --------------------------


def test_empty_frames_rejected(tmp_path: Path) -> None:
    fake = _FakeClient(default=_ok())
    with pytest.raises(FirstPassError) as caught:
        FirstPassRunner(_config(
            [], tmp_path / "run", fake)).run()
    assert caught.value.code == "INVALID_CONFIG"
    assert fake.calls == []


def test_duplicate_frame_index_rejected(tmp_path: Path) -> None:
    frames = [_frame(tmp_path / "frames", 7, 0.7),
              _frame(tmp_path / "frames", 7, 0.8)]
    fake = _FakeClient(default=_ok())
    with pytest.raises(FirstPassError) as caught:
        FirstPassRunner(_config(
            frames, tmp_path / "run", fake)).run()
    assert caught.value.code == "DUPLICATE_FRAME"
    assert fake.calls == []


def test_missing_image_rejected_before_any_request(
        tmp_path: Path) -> None:
    frames = [_frame(tmp_path / "frames", 1, 0.1),
              FrameRef(frame_index=2,
                       image_path=tmp_path / "frames" / "absent.jpg",
                       timestamp_seconds=0.2)]
    fake = _FakeClient(default=_ok())
    with pytest.raises(FirstPassError) as caught:
        FirstPassRunner(_config(
            frames, tmp_path / "run", fake)).run()
    assert caught.value.code == "IMAGE_MISSING"
    assert fake.calls == []


def test_corrupt_image_rejected(tmp_path: Path) -> None:
    frames = [_frame(tmp_path / "frames", 1, 0.1, b"not-an-image")]
    fake = _FakeClient(default=_ok())
    with pytest.raises(FirstPassError) as caught:
        FirstPassRunner(_config(
            frames, tmp_path / "run", fake)).run()
    assert caught.value.code == "IMAGE_MISSING"


def test_bad_batch_size_and_budgets_rejected(
        tmp_path: Path) -> None:
    frames = [_frame(tmp_path / "frames", 1, 0.1)]
    with pytest.raises(FirstPassError):
        FirstPassRunner(_config(
            frames, tmp_path / "r1", _FakeClient(default=_ok()),
            batch_size=0)).run()
    with pytest.raises(FirstPassError):
        FirstPassRunner(_config(
            frames, tmp_path / "r2", _FakeClient(default=_ok()),
            token_budgets=())).run()


def test_missing_client_factory_rejected(tmp_path: Path) -> None:
    frames = [_frame(tmp_path / "frames", 1, 0.1)]
    config = _config(frames, tmp_path / "run",
                     _FakeClient(default=_ok()))
    config.client_factory = None
    with pytest.raises(FirstPassError) as caught:
        FirstPassRunner(config).run()
    assert "client_factory" in caught.value.message


# Transport retries and escalation -------------------------------------


def test_retryable_error_retries_then_succeeds(
        tmp_path: Path) -> None:
    frames = [_frame(tmp_path / "frames", 1, 0.1)]
    sleeps: list[float] = []
    fake = _FakeClient(
        script={1: [InferenceError("TIMEOUT", "slow"), _ok()]})
    config = _config(frames, tmp_path / "run", fake,
                     sleeper=sleeps.append)
    report = FirstPassRunner(config).run()
    assert report["retries"] == 1
    assert report["frames_observed"] == 1
    assert len(sleeps) == 1
    raw = _read_json(tmp_path / "run" / "raw_responses"
                     / "batch_00000001-00000001.json")
    assert [entry["code"] for entry in raw[0]["attempts"]] == [
        "TIMEOUT", "ok"]
    assert "slept_seconds" in raw[0]["attempts"][0]


def test_retry_after_header_honored(tmp_path: Path) -> None:
    frames = [_frame(tmp_path / "frames", 1, 0.1)]
    sleeps: list[float] = []
    fake = _FakeClient(script={1: [
        InferenceError("RATE_LIMITED", "busy",
                       retry_after_seconds=12.5),
        _ok(),
    ]})
    FirstPassRunner(_config(
        frames, tmp_path / "run", fake,
        sleeper=sleeps.append)).run()
    assert sleeps == [12.5]


def test_exhausted_retries_fail_frame_explicitly(
        tmp_path: Path) -> None:
    frames = [_frame(tmp_path / "frames", 1, 0.1)]
    fake = _FakeClient(script={1: [
        InferenceError("SERVER_ERROR", "down")] * 4})
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake)).run()
    assert report["frames_observed"] == 0
    assert report["frames_failed"] == 1
    assert report["batches_needing_review"] == 1
    assert _obs_index(tmp_path / "run") == {}
    raw = _read_json(tmp_path / "run" / "raw_responses"
                     / "batch_00000001-00000001.json")
    assert raw[0]["error_code"] == "SERVER_ERROR"
    assert raw[0]["terminal_state"] == "request_failed"


def test_token_budget_escalation_recovers(tmp_path: Path) -> None:
    frames = [_frame(tmp_path / "frames", 1, 0.1)]
    fake = _FakeClient(script={1: [
        _ok("", InferenceUsage(5, 5, 10),
            status="incomplete",
            incomplete_reason="max_output_tokens reached"),
        _ok(),
    ]})
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake,
        token_budgets=(1024, 2048),
        rates_usd_per_m=(1.25, 4.25))).run()
    assert fake.budgets == [1024, 2048]
    lifecycle = report["lifecycle"]
    assert lifecycle["token_retries"] == 1
    assert lifecycle["token_retried_requests"] == 1
    assert lifecycle["recovery_rate"] == 1.0
    assert lifecycle["token_retry_cost_usd_estimated"] == round(
        5 * 1.25 / 1_000_000 + 5 * 4.25 / 1_000_000, 4)
    assert report["input_tokens"] == 10
    assert report["frames_observed"] == 1
    raw = _read_json(tmp_path / "run" / "raw_responses"
                     / "batch_00000001-00000001.json")
    assert raw[0]["budgets_attempted"] == [1024, 2048]
    assert raw[0]["terminal_state"] == "valid_prediction"


def test_exhausted_budgets_end_without_observation(
        tmp_path: Path) -> None:
    frames = [_frame(tmp_path / "frames", 1, 0.1)]
    incomplete = _ok("", InferenceUsage(5, 5, 10),
                     status="incomplete",
                     incomplete_reason="output_token limit hit")
    fake = _FakeClient(script={1: [incomplete, incomplete]})
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake,
        token_budgets=(1024, 2048))).run()
    assert report["frames_observed"] == 0
    assert report["lifecycle"]["terminal_states"] == {
        "exhausted_token_budget": 1}
    raw = _read_json(tmp_path / "run" / "raw_responses"
                     / "batch_00000001-00000001.json")
    assert raw[0]["error_code"] == "INCOMPLETE_STATUS"


# Terminal classification ----------------------------------------------


def test_invalid_json_keeps_null_fields_explicit(
        tmp_path: Path) -> None:
    frames = [_frame(tmp_path / "frames", 1, 0.1)]
    fake = _FakeClient(default=_ok("not json at all"))
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake)).run()
    assert report["lifecycle"]["terminal_states"] == {
        "invalid_json": 1}
    indexed = _obs_index(tmp_path / "run")
    assert indexed[1]["parse_error"] == "unparseable"
    assert indexed[1]["model_fields"] == {
        name: None for name in MODEL_OUTPUT_FIELDS}


def test_empty_output_and_abstention_classified(
        tmp_path: Path) -> None:
    frames = [_frame(tmp_path / "frames", 1, 0.1),
              _frame(tmp_path / "frames", 2, 0.2)]
    abstain = json.dumps({"scale_readable": False,
                          "confidence": "high"})
    fake = _FakeClient(script={
        1: [_ok("   ")],
        2: [_ok(abstain)],
    })
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake, batch_size=5)).run()
    assert report["lifecycle"]["terminal_states"] == {
        "abstained": 1, "empty_output": 1}
    assert report["lifecycle"]["abstention_rate"] == 1.0


def test_partial_batch_reports_mixed_outcomes(tmp_path: Path) -> None:
    frames = [_frame(tmp_path / "frames", index, float(index))
              for index in (1, 2, 3)]
    fake = _FakeClient(
        script={2: [InferenceError("BAD_REQUEST", "nope")]},
        default=_ok())
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake, batch_size=5)).run()
    assert report["frames_observed"] == 2
    assert report["frames_failed"] == 1
    assert report["batches_needing_review"] == 1
    assert report["lifecycle"]["terminal_states"] == {
        "request_failed": 1, "valid_prediction": 2}
    indexed = _obs_index(tmp_path / "run")
    assert sorted(indexed) == [1, 3]
    raw = _read_json(tmp_path / "run" / "raw_responses"
                     / "batch_00000001-00000003.json")
    failed = [entry for entry in raw
              if entry["frame_indices"] == [2]][0]
    assert failed["error_code"] == "BAD_REQUEST"


# Abort, resume, and provenance ----------------------------------------


def test_abort_code_raises_with_partial_progress(
        tmp_path: Path) -> None:
    frames = [_frame(tmp_path / "frames", index, float(index))
              for index in (1, 2, 3, 4)]
    fake = _FakeClient(
        script={3: [InferenceError("AUTH_FAILED", "bad key")]},
        default=_ok())
    with pytest.raises(InferenceError) as caught:
        FirstPassRunner(_config(
            frames, tmp_path / "run", fake, batch_size=2)).run()
    assert caught.value.code == "AUTH_FAILED"
    indexed = _obs_index(tmp_path / "run")
    assert sorted(indexed) == [1, 2]


def test_resume_skips_completed_batches(tmp_path: Path) -> None:
    frames = [_frame(tmp_path / "frames", index, float(index))
              for index in (1, 2, 3, 4)]
    failing = _FakeClient(
        script={3: [InferenceError("AUTH_FAILED", "bad key")]},
        default=_ok())
    with pytest.raises(InferenceError):
        FirstPassRunner(_config(
            frames, tmp_path / "run", failing, batch_size=2)).run()
    assert len(failing.calls) == 3
    resumed = _FakeClient(default=_ok())
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", resumed, batch_size=2)).run()
    resumed_indices = sorted(
        int(Path(call[1][0]).stem.split("_")[1])
        for call in resumed.calls)
    assert resumed_indices == [3, 4]
    assert report["batches_completed"] == 2
    assert report["requests_sent"] == 2
    assert sorted(_obs_index(tmp_path / "run")) == [1, 2, 3, 4]


def test_provenance_and_cost_reporting(tmp_path: Path) -> None:
    frames = [_frame(tmp_path / "frames", 1, 0.1)]
    usage = InferenceUsage(100, 200, 300,
                           reasoning_output_tokens=50)
    fake = _FakeClient(default=_ok(usage=usage))
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake,
        rates_usd_per_m=(1.25, 4.25))).run()
    assert report["cost_status"] == "estimated_from_configured_rates"
    assert report["cost_usd_estimated"] == round(
        100 * 1.25 / 1_000_000 + 200 * 4.25 / 1_000_000, 4)
    indexed = _obs_index(tmp_path / "run")
    call = indexed[1]["call"]
    assert call["provider"] == "meta"
    assert call["max_output_tokens"] == 1024
    assert call["token_budgets_attempted"] == [1024]
    assert call["reasoning_output_tokens"] == 50
    assert call["request_id"] == "req-1"
    assert indexed[1]["raw_text_excerpt"] == VALID_JSON[:500]
    assert indexed[1]["image_ref"]["encoding"] == "jpeg"


def test_resume_with_different_frames_refused(
        tmp_path: Path) -> None:
    frames = [_frame(tmp_path / "frames", index, float(index))
              for index in (1, 2, 3, 4)]
    FirstPassRunner(_config(
        frames, tmp_path / "run", _FakeClient(default=_ok()),
        batch_size=2)).run()
    changed = [_frame(tmp_path / "frames", index, float(index))
               for index in (1, 2, 3, 4, 5)]
    fresh = _FakeClient(default=_ok())
    with pytest.raises(FirstPassError) as caught:
        FirstPassRunner(_config(
            changed, tmp_path / "run", fresh, batch_size=2)).run()
    assert caught.value.code == "CHECKPOINT_MISMATCH"
    assert fresh.calls == []


def test_resume_with_different_span_refused(tmp_path: Path) -> None:
    frames = [_frame(tmp_path / "frames", 1, 0.1),
              _frame(tmp_path / "frames", 9, 0.9)]
    FirstPassRunner(_config(
        frames, tmp_path / "run", _FakeClient(default=_ok()),
        batch_size=5)).run()
    (tmp_path / "run" / "checkpoints" / "run_state.json").unlink()
    changed = [_frame(tmp_path / "frames", 1, 0.1),
               _frame(tmp_path / "frames", 5, 0.5),
               _frame(tmp_path / "frames", 9, 0.9)]
    fresh = _FakeClient(default=_ok())
    with pytest.raises(FirstPassError) as caught:
        FirstPassRunner(_config(
            changed, tmp_path / "run", fresh, batch_size=5)).run()
    assert caught.value.code == "CHECKPOINT_MISMATCH"
    assert fresh.calls == []


def test_cost_unknown_without_rates(tmp_path: Path) -> None:
    frames = [_frame(tmp_path / "frames", 1, 0.1)]
    fake = _FakeClient(default=_ok())
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake)).run()
    assert report["cost_usd_estimated"] is None
    assert report["cost_status"] == "COST_UNKNOWN"


# Recorded P04 structural compatibility --------------------------------


def _recorded_available() -> bool:
    return RECORDED_RAW.is_file() and RECORDED_OBS.is_file()


@pytest.mark.skipif(not _recorded_available(),
                    reason="recorded P04 raw batch not present")
def test_recorded_raw_replay_matches_observation_shape(
        tmp_path: Path) -> None:
    """Replay recorded model texts; compare shapes, not live output.

    Structural compatibility only: identical recorded replies must
    yield identical model fields and a key-compatible record. This
    says nothing about what a live model would output today.
    """
    raw_groups = _read_json(RECORDED_RAW)
    recorded = {item["frame_index"]: item
                for item in _read_json(RECORDED_OBS)}
    frames_dir = tmp_path / "frames"
    frames = []
    script: dict[int, list[Any]] = {}
    for group in raw_groups:
        index = group["frame_indices"][0]
        assert index in recorded
        frames.append(_frame(
            frames_dir, index,
            recorded[index]["timestamp_seconds"]))
        usage = group.get("usage", {})
        script[index] = [_ok(
            group["output_text"],
            InferenceUsage(usage.get("input_tokens"),
                           usage.get("output_tokens"),
                           usage.get("total_tokens")),
            model=group.get("model", "fake-model"),
            request_id=group.get("request_id"),
        )]
    manifest_sha = recorded[raw_groups[0]["frame_indices"][0]][
        "manifest_ref"]["manifest_sha256"]
    fake = _FakeClient(script=script)
    config = _config(frames, tmp_path / "run", fake,
                     batch_size=len(frames), pass_id="baseline_1fps",
                     source=SourceRef(name="p04_full_4096",
                                      sha256=manifest_sha))
    FirstPassRunner(config).run()
    indexed = _obs_index(tmp_path / "run")
    assert sorted(indexed) == sorted(recorded)
    for index, expected in recorded.items():
        actual = indexed[index]
        assert set(expected) <= set(actual)
        assert actual["model_fields"] == expected["model_fields"]
        assert actual["frame_index"] == expected["frame_index"]
        assert (actual["timestamp_seconds"]
                == expected["timestamp_seconds"])
        assert actual["image_path"] == expected["image_path"]
        assert actual["batch_id"] == expected["batch_id"]
        assert actual["pass_id"] == expected["pass_id"]
        assert (actual["manifest_ref"]["manifest_sha256"]
                == manifest_sha)
        assert actual["parse_error"] == expected.get("parse_error")
        if "terminal_state" in expected:
            assert (actual["terminal_state"]
                    == expected["terminal_state"])
        else:
            assert actual["terminal_state"] in (
                "valid_prediction", "abstained")
    assert set(recorded[raw_groups[0]["frame_indices"][0]][
        "model_fields"]) <= set(MODEL_OUTPUT_FIELDS)
