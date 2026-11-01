"""Offline tests for generalized first-pass inference (M2-06C.2).

A scripted fake client stands in for the provider seam: no
network, no credentials, no paid calls. Covers ordering,
batching, retries, budget escalation, partial failures, resume,
provenance, request/cost guardrails (M2-06C.5/S1), and
structural compatibility with recorded P04 outputs.
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


# Request and cost budget guardrails (M2-06C.5/S1) ----------------------

# One dollar per token in each direction, so estimates stay
# exact and readable ($30 for the standard 10/20 usage).
DOLLAR_RATES = (1_000_000.0, 1_000_000.0)
DOLLAR_USAGE = InferenceUsage(10, 20, 30)


def _seq_frames(tmp_path: Path, count: int = 4) -> list[FrameRef]:
    return [_frame(tmp_path / "frames", index, float(index))
            for index in range(1, count + 1)]


def _call_indices(fake: _FakeClient) -> list[int]:
    return sorted(
        int(Path(call[1][0]).stem.split("_")[1])
        for call in fake.calls)


def _checkpoints(out_dir: Path) -> list[dict[str, Any]]:
    text = (out_dir / "checkpoints" / "checkpoints.jsonl"
            ).read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [json.loads(line) for line in text.split("\n")]


def test_request_limit_stops_before_next_request(
        tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path)
    fake = _FakeClient(default=_ok())
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake, batch_size=5,
        max_requests=2)).run()
    assert len(fake.calls) == 2
    assert _call_indices(fake) == [1, 2]
    assert report["frames_observed"] == 2
    assert report["frames_failed"] == 0
    assert report["frames_unprocessed"] == 2
    assert report["batches_total"] == 1
    assert report["batches_completed"] == 0
    assert report["batches_needing_review"] == 0
    assert report["stopped_reason"] == "request_limit_reached"
    budget = report["budget"]
    assert budget["max_requests"] == 2
    assert budget["requests_submitted"] == 2
    assert budget["prior_requests_submitted"] == 0
    assert budget["stopped_reason"] == "request_limit_reached"
    assert budget["stopped_detail"] == {
        "requests_submitted": 2, "max_requests": 2}
    assert budget["remaining_frame_indices"] == [3, 4]
    assert sorted(_obs_index(tmp_path / "run")) == [1, 2]
    records = _checkpoints(tmp_path / "run")
    assert len(records) == 1
    assert records[0]["status"] == "budget_stopped"
    assert records[0]["frame_indices"] == [1, 2, 3, 4]
    assert records[0]["completed_frame_indices"] == [1, 2]
    assert records[0]["budget"]["requests_submitted"] == 2


def test_request_limit_zero_submits_nothing(
        tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path)
    fake = _FakeClient(default=_ok())
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake, batch_size=5,
        max_requests=0)).run()
    assert fake.calls == []
    assert report["frames_observed"] == 0
    assert report["frames_failed"] == 0
    assert report["frames_unprocessed"] == 4
    budget = report["budget"]
    assert budget["stopped_reason"] == "request_limit_reached"
    assert budget["stopped_detail"] == {
        "requests_submitted": 0, "max_requests": 0}
    assert budget["remaining_frame_indices"] == [1, 2, 3, 4]
    assert _obs_index(tmp_path / "run") == {}
    records = _checkpoints(tmp_path / "run")
    assert len(records) == 1
    assert records[0]["status"] == "budget_stopped"
    assert records[0]["completed_frame_indices"] == []


def test_retry_attempts_consume_request_budget(
        tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path, 2)
    fake = _FakeClient(
        script={1: [InferenceError("TIMEOUT", "slow"), _ok()]},
        default=_ok())
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake, batch_size=5,
        max_requests=2)).run()
    assert len(fake.calls) == 2
    assert _call_indices(fake) == [1, 1]
    assert report["retries"] == 1
    assert report["frames_observed"] == 1
    assert report["frames_unprocessed"] == 1
    assert report["stopped_reason"] == "request_limit_reached"
    assert report["budget"]["remaining_frame_indices"] == [2]


def test_request_limit_exact_completion(tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path, 2)
    fake = _FakeClient(default=_ok())
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake, batch_size=5,
        max_requests=2)).run()
    assert len(fake.calls) == 2
    assert report["frames_observed"] == 2
    assert report["frames_unprocessed"] == 0
    assert report["stopped_reason"] is None
    assert report["budget"]["remaining_frame_indices"] == []
    records = _checkpoints(tmp_path / "run")
    assert records[0]["status"] == "completed"


def test_cost_ceiling_pre_dispatch_reserve_halts(
        tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path)
    fake = _FakeClient(
        default=_ok(usage=DOLLAR_USAGE))
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake, batch_size=5,
        rates_usd_per_m=DOLLAR_RATES,
        cost_ceiling_usd=75.0)).run()
    assert len(fake.calls) == 2
    assert sorted(_obs_index(tmp_path / "run")) == [1, 2]
    budget = report["budget"]
    assert budget["stopped_reason"] == "cost_reserve_exceeded"
    assert budget["stopped_detail"] == {
        "spent_known_usd": 60.0,
        "reserve_usd_estimate": 30.0,
        "reserve_basis": "max_observed_single_request_estimate",
        "cost_ceiling_usd": 75.0,
    }
    assert budget["spent_cost_usd_estimate"] == 60.0
    assert budget["cost_unknown_requests"] == 0
    assert budget["remaining_frame_indices"] == [3, 4]
    assert report["frames_unprocessed"] == 2


def test_cost_ceiling_post_response_overrun_halts(
        tmp_path: Path) -> None:
    small = InferenceUsage(4, 6, 10)
    big = InferenceUsage(40, 60, 100)
    frames = _seq_frames(tmp_path)
    fake = _FakeClient(
        script={1: [_ok(usage=small)],
                2: [_ok(usage=big)]},
        default=_ok(usage=small))
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake, batch_size=5,
        rates_usd_per_m=DOLLAR_RATES,
        cost_ceiling_usd=50.0)).run()
    assert len(fake.calls) == 2
    # The overrunning response is billed work: its
    # observation is kept, and only later frames remain.
    assert sorted(_obs_index(tmp_path / "run")) == [1, 2]
    budget = report["budget"]
    assert budget["stopped_reason"] == "cost_ceiling_exceeded"
    assert budget["stopped_detail"] == {
        "phase": "post_response",
        "spent_known_usd": 110.0,
        "cost_ceiling_usd": 50.0,
    }
    assert budget["spent_cost_usd_estimate"] == 110.0
    assert budget["remaining_frame_indices"] == [3, 4]


def test_unknown_cost_non_strict_continues(
        tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path, 3)
    fake = _FakeClient(
        script={2: [_ok(usage=InferenceUsage())]},
        default=_ok(usage=DOLLAR_USAGE))
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake, batch_size=5,
        rates_usd_per_m=DOLLAR_RATES,
        cost_ceiling_usd=1000.0)).run()
    assert len(fake.calls) == 3
    assert report["stopped_reason"] is None
    assert report["frames_observed"] == 3
    budget = report["budget"]
    assert budget["cost_unknown_requests"] == 1
    assert budget["spent_cost_usd_estimate"] is None
    assert budget["spent_cost_known_usd_estimate"] == 60.0
    assert budget["remaining_frame_indices"] == []


def test_unknown_cost_strict_halts(tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path, 3)
    fake = _FakeClient(
        script={2: [_ok(usage=InferenceUsage())]},
        default=_ok(usage=DOLLAR_USAGE))
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake, batch_size=5,
        rates_usd_per_m=DOLLAR_RATES,
        cost_ceiling_usd=1000.0, strict_cost=True)).run()
    assert len(fake.calls) == 2
    assert sorted(_obs_index(tmp_path / "run")) == [1, 2]
    budget = report["budget"]
    assert budget["stopped_reason"] == "unknown_cost_strict_halt"
    assert budget["stopped_detail"] == {
        "phase": "post_response",
        "cause": "usage_unreported",
        "cost_unknown_requests": 1,
        "cost_ceiling_usd": 1000.0,
    }
    assert budget["spent_cost_usd_estimate"] is None
    assert budget["spent_cost_known_usd_estimate"] == 30.0
    assert budget["remaining_frame_indices"] == [3]


def test_strict_without_rates_refuses_dispatch(
        tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path, 2)
    fake = _FakeClient(default=_ok())
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake, batch_size=5,
        cost_ceiling_usd=100.0, strict_cost=True)).run()
    assert fake.calls == []
    assert report["frames_observed"] == 0
    budget = report["budget"]
    assert budget["stopped_reason"] == "unknown_cost_strict_halt"
    assert budget["stopped_detail"] == {
        "phase": "pre_dispatch",
        "cause": "rates_unknown",
        "cost_ceiling_usd": 100.0,
    }
    assert budget["remaining_frame_indices"] == [1, 2]


def test_strict_transport_failure_halts(tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path, 3)
    fake = _FakeClient(
        script={1: [InferenceError("BAD_REQUEST", "nope")]},
        default=_ok(usage=DOLLAR_USAGE))
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake, batch_size=5,
        rates_usd_per_m=DOLLAR_RATES,
        cost_ceiling_usd=1000.0, strict_cost=True)).run()
    assert len(fake.calls) == 1
    assert report["frames_observed"] == 0
    assert report["frames_failed"] == 1
    assert report["frames_unprocessed"] == 2
    budget = report["budget"]
    assert budget["stopped_reason"] == "unknown_cost_strict_halt"
    assert budget["stopped_detail"] == {
        "phase": "transport",
        "cause": "transport_failure_without_usage",
        "cost_unknown_requests": 1,
        "cost_ceiling_usd": 1000.0,
    }
    assert budget["remaining_frame_indices"] == [2, 3]
    raw = _read_json(tmp_path / "run" / "raw_responses"
                     / "batch_00000001-00000003.json")
    assert raw == []
    records = _checkpoints(tmp_path / "run")
    tries = records[0]["attempts"][0]["attempts"]
    assert [entry["code"] for entry in tries] == ["BAD_REQUEST"]


def test_transport_failure_counts_unknown_cost(
        tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path, 3)
    fake = _FakeClient(
        script={1: [InferenceError("BAD_REQUEST", "nope")]},
        default=_ok(usage=DOLLAR_USAGE))
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake, batch_size=5,
        rates_usd_per_m=DOLLAR_RATES,
        cost_ceiling_usd=1000.0)).run()
    assert len(fake.calls) == 3
    assert report["stopped_reason"] is None
    assert report["frames_failed"] == 1
    assert report["lifecycle"]["terminal_states"] == {
        "request_failed": 1, "valid_prediction": 2}
    budget = report["budget"]
    assert budget["cost_unknown_requests"] == 1
    assert budget["spent_cost_usd_estimate"] is None
    assert budget["spent_cost_known_usd_estimate"] == 60.0


def test_ceiling_zero_non_strict_reports_overrun(
        tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path, 2)
    fake = _FakeClient(default=_ok(usage=DOLLAR_USAGE))
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake, batch_size=5,
        rates_usd_per_m=DOLLAR_RATES,
        cost_ceiling_usd=0.0)).run()
    assert len(fake.calls) == 1
    assert sorted(_obs_index(tmp_path / "run")) == [1]
    budget = report["budget"]
    assert budget["stopped_reason"] == "cost_ceiling_exceeded"
    assert budget["stopped_detail"] == {
        "phase": "post_response",
        "spent_known_usd": 30.0,
        "cost_ceiling_usd": 0.0,
    }
    assert budget["remaining_frame_indices"] == [2]


def test_resume_preserves_request_accounting(
        tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path)
    out_dir = tmp_path / "run"
    first = _FakeClient(default=_ok())
    capped = FirstPassRunner(_config(
        frames, out_dir, first, batch_size=2,
        max_requests=2)).run()
    assert _call_indices(first) == [1, 2]
    assert capped["stopped_reason"] == "request_limit_reached"
    assert capped["budget"]["remaining_frame_indices"] == [3, 4]
    second = _FakeClient(default=_ok())
    report = FirstPassRunner(_config(
        frames, out_dir, second, batch_size=2,
        max_requests=4)).run()
    assert _call_indices(second) == [3, 4]
    assert report["stopped_reason"] is None
    assert report["budget"]["remaining_frame_indices"] == []
    budget = report["budget"]
    assert budget["requests_submitted"] == 2
    assert budget["prior_requests_submitted"] == 2
    assert budget["prior_cost_basis"] == "recorded"
    assert budget["prior_frames_observed"] == 2
    assert sorted(_obs_index(out_dir)) == [1, 2, 3, 4]
    merged = _read_json(out_dir / "observations"
                        / "batch_00000003-00000004.json")
    assert [item["frame_index"] for item in merged] == [3, 4]
    records = _checkpoints(out_dir)
    assert len(records) == 3
    assert records[-1]["status"] == "completed"
    assert records[-1]["frame_indices"] == [3, 4]


def test_resume_enforces_cost_ceiling_on_lifetime_total(
        tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path)
    out_dir = tmp_path / "run"
    base = {"batch_size": 5, "rates_usd_per_m": DOLLAR_RATES}
    first = _FakeClient(default=_ok(usage=DOLLAR_USAGE))
    capped = FirstPassRunner(_config(
        frames, out_dir, first, cost_ceiling_usd=75.0,
        **base)).run()
    assert _call_indices(first) == [1, 2]
    assert capped["stopped_reason"] == "cost_reserve_exceeded"
    stalled = _FakeClient(default=_ok(usage=DOLLAR_USAGE))
    report = FirstPassRunner(_config(
        frames, out_dir, stalled, cost_ceiling_usd=75.0,
        **base)).run()
    assert stalled.calls == []
    assert report["stopped_reason"] == "cost_reserve_exceeded"
    assert report["budget"]["prior_spent_cost_known_usd_estimate"
                 ] == 60.0
    assert report["budget"]["remaining_frame_indices"] == [3, 4]
    raised = _FakeClient(default=_ok(usage=DOLLAR_USAGE))
    final = FirstPassRunner(_config(
        frames, out_dir, raised, cost_ceiling_usd=200.0,
        **base)).run()
    assert _call_indices(raised) == [3, 4]
    assert final["stopped_reason"] is None
    assert final["budget"]["remaining_frame_indices"] == []
    assert final["budget"]["requests_submitted"] == 2
    assert final["budget"]["prior_requests_submitted"] == 2
    assert final["budget"]["spent_cost_known_usd_estimate"
                 ] == 60.0
    assert final["budget"]["prior_spent_cost_known_usd_estimate"
                 ] == 60.0
    assert final["budget"]["prior_cost_basis"] == "recorded"
    assert sorted(_obs_index(out_dir)) == [1, 2, 3, 4]
    merged = _read_json(out_dir / "observations"
                        / "batch_00000001-00000004.json")
    assert [item["frame_index"] for item in merged] == [1, 2, 3, 4]
    records = _checkpoints(out_dir)
    assert records[-1]["status"] == "completed"
    assert records[-1]["completed_frame_indices"] == [1, 2, 3, 4]
    assert records[-1]["budget"]["requests_submitted"] == 4


def test_old_checkpoints_seed_reconstructed_basis(
        tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    checkpoints = out_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    legacy = {
        "attempts": [
            {"attempts": [{"attempt": 1, "budget": 1024,
                            "code": "ok"}],
             "frame_indices": [1]},
            {"attempts": [{"attempt": 1, "budget": 1024,
                            "code": "ok"}],
             "frame_indices": [2]},
        ],
        "batch_id": "batch_00000001-00000002",
        "frame_indices": [1, 2],
        "index_range": [1, 2],
        "input_tokens": 20,
        "obs_path": "batch_00000001-00000002.json",
        "output_tokens": 40,
        "raw_path": "batch_00000001-00000002.json",
        "status": "completed",
    }
    (checkpoints / "checkpoints.jsonl").write_text(
        json.dumps(legacy) + "\n", encoding="utf-8")
    frames = _seq_frames(tmp_path)
    fake = _FakeClient(default=_ok(usage=DOLLAR_USAGE))
    report = FirstPassRunner(_config(
        frames, out_dir, fake, batch_size=2,
        rates_usd_per_m=DOLLAR_RATES,
        max_requests=3)).run()
    assert _call_indices(fake) == [3]
    assert report["stopped_reason"] == "request_limit_reached"
    budget = report["budget"]
    assert budget["requests_submitted"] == 1
    assert budget["prior_requests_submitted"] == 2
    assert budget["prior_cost_basis"] == (
        "reconstructed_from_token_totals")
    assert budget["prior_spent_cost_known_usd_estimate"] == 60.0
    assert budget["prior_spent_cost_usd_estimate"] == 60.0
    assert budget["prior_frames_observed"] == 2
    assert budget["remaining_frame_indices"] == [4]


def test_default_run_has_no_budget_limits(
        tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path, 2)
    fake = _FakeClient(default=_ok())
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake, batch_size=5)).run()
    budget = report["budget"]
    assert budget["max_requests"] is None
    assert budget["cost_ceiling_usd"] is None
    assert budget["strict_cost"] is False
    assert budget["requests_submitted"] == 2
    assert budget["prior_requests_submitted"] == 0
    assert budget["stopped_reason"] is None
    assert budget["remaining_frame_indices"] == []
    assert budget["cost_unknown_requests"] == 2
    assert budget["spent_cost_usd_estimate"] is None
    rated = FirstPassRunner(_config(
        _seq_frames(tmp_path, 2), tmp_path / "rated",
        _FakeClient(default=_ok(usage=DOLLAR_USAGE)),
        batch_size=5, rates_usd_per_m=DOLLAR_RATES)).run()
    assert rated["budget"]["cost_unknown_requests"] == 0
    assert rated["budget"]["spent_cost_usd_estimate"] == 60.0


@pytest.mark.parametrize("overrides", [
    {"max_requests": -1},
    {"max_requests": True},
    {"max_requests": 2.0},
    {"max_requests": "2"},
    {"max_requests": float("nan")},
    {"cost_ceiling_usd": -1.0},
    {"cost_ceiling_usd": True},
    {"cost_ceiling_usd": "10"},
    {"cost_ceiling_usd": float("nan")},
    {"cost_ceiling_usd": float("inf")},
    {"cost_ceiling_usd": float("-inf")},
    {"strict_cost": 1},
    {"strict_cost": "yes"},
    {"strict_cost": None},
])
def test_invalid_budget_limits_rejected(
        tmp_path: Path, overrides: dict[str, Any]) -> None:
    frames = _seq_frames(tmp_path, 1)
    fake = _FakeClient(default=_ok())
    with pytest.raises(FirstPassError) as caught:
        FirstPassRunner(_config(
            frames, tmp_path / "run", fake,
            **overrides)).run()
    assert caught.value.code == "INVALID_CONFIG"
    assert fake.calls == []


def test_strict_without_ceiling_is_vacuous(
        tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path, 2)
    fake = _FakeClient(
        script={1: [InferenceError("BAD_REQUEST", "nope")]},
        default=_ok())
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake, batch_size=5,
        strict_cost=True)).run()
    assert len(fake.calls) == 2
    assert report["stopped_reason"] is None
    assert report["frames_failed"] == 1
    assert report["frames_observed"] == 1
    # Both submits lack calculable cost for different reasons:
    # the failed transport reports no usage, and the success
    # reports usage but no rates are configured. Neither is
    # treated as zero.
    assert report["budget"]["cost_unknown_requests"] == 2
    assert report["budget"]["spent_cost_usd_estimate"] is None
    assert report["budget"]["spent_cost_known_usd_estimate"] == 0.0
    assert report["budget"]["remaining_frame_indices"] == []


def test_superseded_usage_counts_toward_spend(
        tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path, 1)
    fake = _FakeClient(script={1: [
        _ok("", InferenceUsage(5, 5, 10),
            status="incomplete",
            incomplete_reason="max_output_tokens reached"),
        _ok(usage=DOLLAR_USAGE),
    ]})
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake,
        token_budgets=(1024, 2048),
        rates_usd_per_m=DOLLAR_RATES)).run()
    assert fake.budgets == [1024, 2048]
    budget = report["budget"]
    assert budget["requests_submitted"] == 2
    assert budget["spent_cost_known_usd_estimate"] == 40.0
    assert budget["spent_cost_usd_estimate"] == 40.0
    assert budget["cost_unknown_requests"] == 0
    assert report["cost_usd_estimated"] == 30.0


def test_capped_run_completing_under_budget(
        tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path, 3)
    fake = _FakeClient(default=_ok(usage=DOLLAR_USAGE))
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake, batch_size=5,
        rates_usd_per_m=DOLLAR_RATES, max_requests=10,
        cost_ceiling_usd=1000.0, strict_cost=True)).run()
    assert len(fake.calls) == 3
    assert report["stopped_reason"] is None
    assert report["frames_observed"] == 3
    assert report["budget"]["remaining_frame_indices"] == []
    assert report["budget"]["spent_cost_usd_estimate"] == 90.0
    assert report["budget"]["cost_unknown_requests"] == 0


# Durable checkpoints and safe recovery (M2-06C.5/S2) --------------------


class _CrashingClient:
    """Fake client that dies once, simulating a worker crash."""

    def __init__(self, crash_on: int,
                 usage: InferenceUsage | None = None) -> None:
        self.crash_on = crash_on
        self.usage = usage or InferenceUsage(10, 20, 30)
        self.calls: list[int] = []
        self.crashed = False

    def submit(self, prompt_text: str, image_paths: list[Path],
               api_key: str, max_output_tokens: int
               ) -> InferenceResponse:
        _ = (prompt_text, api_key, max_output_tokens)
        index = int(Path(image_paths[0]).stem.split("_")[1])
        self.calls.append(index)
        if index == self.crash_on and not self.crashed:
            self.crashed = True
            raise RuntimeError("simulated worker crash")
        return _ok(usage=self.usage)


def test_review_batch_resumes_only_unobserved_frames(
        tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path, 3)
    out_dir = tmp_path / "run"
    first = _FakeClient(
        script={2: [InferenceError("BAD_REQUEST", "nope")]},
        default=_ok())
    partial = FirstPassRunner(_config(
        frames, out_dir, first, batch_size=5)).run()
    assert partial["batches_needing_review"] == 1
    assert sorted(_obs_index(out_dir)) == [1, 3]
    records = _checkpoints(out_dir)
    assert records[0]["status"] == "needs_human_review"
    second = _FakeClient(default=_ok())
    report = FirstPassRunner(_config(
        frames, out_dir, second, batch_size=5)).run()
    assert _call_indices(second) == [2]
    assert sorted(_obs_index(out_dir)) == [1, 2, 3]
    assert report["batches_needing_review"] == 0
    assert report["batches_completed"] == 1
    budget = report["budget"]
    assert budget["requests_submitted"] == 1
    assert budget["prior_requests_submitted"] == 3
    assert budget["prior_cost_basis"] == "recorded"
    merged = _read_json(out_dir / "observations"
                        / "batch_00000001-00000003.json")
    assert sorted(
        item["frame_index"] for item in merged) == [1, 2, 3]
    records = _checkpoints(out_dir)
    assert records[-1]["status"] == "completed"
    assert sorted(
        records[-1]["completed_frame_indices"]) == [1, 2, 3]
    assert records[-1]["budget"]["requests_submitted"] == 4


def test_unknown_checkpoint_status_refused(
        tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    checkpoints = out_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "checkpoints.jsonl").write_text(
        json.dumps({"batch_id": "batch_00000001-00000002",
                    "frame_indices": [1, 2],
                    "status": "mystery"}) + "\n",
        encoding="utf-8")
    frames = _seq_frames(tmp_path, 2)
    fake = _FakeClient(default=_ok())
    with pytest.raises(FirstPassError) as caught:
        FirstPassRunner(_config(
            frames, out_dir, fake, batch_size=5)).run()
    assert caught.value.code == "CHECKPOINT_MISMATCH"
    assert "mystery" in caught.value.message
    assert fake.calls == []


def test_torn_checkpoint_tail_truncated(tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path)
    out_dir = tmp_path / "run"
    first = _FakeClient(default=_ok())
    FirstPassRunner(_config(
        frames, out_dir, first, batch_size=2)).run()
    log = out_dir / "checkpoints" / "checkpoints.jsonl"
    before = log.read_bytes()
    log.write_bytes(before + b'{"batch_id": "batch_0000000')
    second = _FakeClient(default=_ok())
    report = FirstPassRunner(_config(
        frames, out_dir, second, batch_size=2)).run()
    assert second.calls == []
    assert report["frames_total"] == 4
    assert log.read_bytes() == before


def test_midfile_checkpoint_corruption_refused(
        tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path)
    out_dir = tmp_path / "run"
    first = _FakeClient(default=_ok())
    FirstPassRunner(_config(
        frames, out_dir, first, batch_size=2)).run()
    log = out_dir / "checkpoints" / "checkpoints.jsonl"
    lines = log.read_bytes().strip().split(b"\n")
    assert len(lines) == 2
    log.write_bytes(
        b"\n".join([lines[0], b"NOT JSON AT ALL", lines[1]])
        + b"\n")
    second = _FakeClient(default=_ok())
    with pytest.raises(FirstPassError) as caught:
        FirstPassRunner(_config(
            frames, out_dir, second, batch_size=2)).run()
    assert caught.value.code == "CHECKPOINT_CORRUPT"
    assert second.calls == []


def test_uncertain_refuse_policy_blocks_replay(
        tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path)
    out_dir = tmp_path / "run"
    crashing = _CrashingClient(crash_on=3)
    with pytest.raises(RuntimeError, match="simulated worker"):
        FirstPassRunner(_config(
            frames, out_dir, crashing, batch_size=2)).run()
    assert crashing.calls == [1, 2, 3]
    cautious = _FakeClient(default=_ok())
    with pytest.raises(FirstPassError) as caught:
        FirstPassRunner(_config(
            frames, out_dir, cautious, batch_size=2,
            uncertain_policy="refuse")).run()
    assert caught.value.code == "UNCERTAIN_REQUESTS"
    assert "batch_00000003-00000004" in caught.value.message
    assert cautious.calls == []


def test_uncertain_rerun_reports_replayed_frames(
        tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path)
    out_dir = tmp_path / "run"
    crashing = _CrashingClient(crash_on=3)
    with pytest.raises(RuntimeError, match="simulated worker"):
        FirstPassRunner(_config(
            frames, out_dir, crashing, batch_size=2)).run()
    second = _FakeClient(default=_ok())
    report = FirstPassRunner(_config(
        frames, out_dir, second, batch_size=2)).run()
    assert _call_indices(second) == [3, 4]
    assert sorted(_obs_index(out_dir)) == [1, 2, 3, 4]
    budget = report["budget"]
    assert budget["uncertain_frame_indices"] == [3, 4]
    assert budget["uncertain_policy"] == "rerun"
    assert budget["prior_requests_submitted"] == 2
    assert budget["requests_submitted"] == 2
    assert budget["stopped_reason"] is None
    assert budget["remaining_frame_indices"] == []


@pytest.mark.parametrize("policy", ["sometimes", "", None, True])
def test_invalid_uncertain_policy_rejected(
        tmp_path: Path, policy: Any) -> None:
    frames = _seq_frames(tmp_path, 1)
    fake = _FakeClient(default=_ok())
    with pytest.raises(FirstPassError) as caught:
        FirstPassRunner(_config(
            frames, tmp_path / "run", fake,
            uncertain_policy=policy)).run()
    assert caught.value.code == "INVALID_CONFIG"
    assert fake.calls == []


def test_missing_partial_files_refused(tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path, 3)
    out_dir = tmp_path / "run"
    first = _FakeClient(
        script={2: [InferenceError("BAD_REQUEST", "nope")]},
        default=_ok())
    FirstPassRunner(_config(
        frames, out_dir, first, batch_size=5)).run()
    (out_dir / "observations"
     / "batch_00000001-00000003.json").unlink()
    second = _FakeClient(default=_ok())
    with pytest.raises(FirstPassError) as caught:
        FirstPassRunner(_config(
            frames, out_dir, second, batch_size=5)).run()
    assert caught.value.code == "CHECKPOINT_CORRUPT"
    assert second.calls == []


def test_checkpoint_observation_disagreement_refused(
        tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path, 3)
    out_dir = tmp_path / "run"
    first = _FakeClient(
        script={2: [InferenceError("BAD_REQUEST", "nope")]},
        default=_ok())
    FirstPassRunner(_config(
        frames, out_dir, first, batch_size=5)).run()
    log = out_dir / "checkpoints" / "checkpoints.jsonl"
    record = json.loads(log.read_text(encoding="utf-8").strip())
    record["completed_frame_indices"] = [1]
    log.write_text(json.dumps(record) + "\n", encoding="utf-8")
    second = _FakeClient(default=_ok())
    with pytest.raises(FirstPassError) as caught:
        FirstPassRunner(_config(
            frames, out_dir, second, batch_size=5)).run()
    assert caught.value.code == "CHECKPOINT_CORRUPT"
    assert second.calls == []


def test_legacy_review_checkpoint_resumes_selectively(
        tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    checkpoints = out_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    obs_dir = out_dir / "observations"
    obs_dir.mkdir(parents=True)
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(parents=True)
    batch = "batch_00000001-00000003"
    legacy = {
        "attempts": [
            {"attempts": [{"attempt": 1, "budget": 1024,
                            "code": "ok"}],
             "frame_indices": [1]},
            {"attempts": [{"attempt": 1, "budget": 1024,
                            "code": "BAD_REQUEST"}],
             "frame_indices": [2]},
            {"attempts": [{"attempt": 1, "budget": 1024,
                            "code": "ok"}],
             "frame_indices": [3]},
        ],
        "batch_id": batch,
        "frame_indices": [1, 2, 3],
        "index_range": [1, 3],
        "input_tokens": 20,
        "obs_path": f"{batch}.json",
        "output_tokens": 40,
        "raw_path": f"{batch}.json",
        "status": "needs_human_review",
    }
    (checkpoints / "checkpoints.jsonl").write_text(
        json.dumps(legacy) + "\n", encoding="utf-8")
    (obs_dir / f"{batch}.json").write_text(json.dumps([
        {"frame_index": 1, "terminal_state": "valid_prediction"},
        {"frame_index": 3, "terminal_state": "valid_prediction"},
    ]), encoding="utf-8")
    (raw_dir / f"{batch}.json").write_text(
        json.dumps([]), encoding="utf-8")
    frames = _seq_frames(tmp_path, 3)
    fake = _FakeClient(default=_ok())
    report = FirstPassRunner(_config(
        frames, out_dir, fake, batch_size=5,
        rates_usd_per_m=DOLLAR_RATES)).run()
    assert _call_indices(fake) == [2]
    assert sorted(_obs_index(out_dir)) == [1, 2, 3]
    budget = report["budget"]
    assert budget["prior_requests_submitted"] == 3
    assert budget["requests_submitted"] == 1
    assert budget["prior_cost_basis"] == (
        "reconstructed_from_token_totals")
    assert budget["prior_spent_cost_known_usd_estimate"] == 60.0
    assert budget["prior_frames_observed"] == 2


# Paid-execution retry safety (M2-06C.5/S3) -------------------------------


def _billable(**extra: Any) -> dict[str, Any]:
    """Valid billable overrides: single submit, no replay."""
    payload: dict[str, Any] = {
        "billable": True, "max_attempts": 1,
        "token_budgets": (1024,), "uncertain_policy": "refuse",
    }
    payload.update(extra)
    return payload


@pytest.mark.parametrize("flag", [1, 0, "yes", None])
def test_billable_flag_type_rejected(
        tmp_path: Path, flag: Any) -> None:
    frames = _seq_frames(tmp_path, 1)
    fake = _FakeClient(default=_ok())
    with pytest.raises(FirstPassError) as caught:
        FirstPassRunner(_config(
            frames, tmp_path / "run", fake,
            **_billable(billable=flag))).run()
    assert caught.value.code == "INVALID_CONFIG"
    assert "billable must be a bool" in caught.value.message
    assert fake.calls == []


@pytest.mark.parametrize("attempts", [0, 2, 4, True, 1.0, "1"])
def test_billable_rejects_retry_enabled_config(
        tmp_path: Path, attempts: Any) -> None:
    frames = _seq_frames(tmp_path, 1)
    fake = _FakeClient(default=_ok())
    with pytest.raises(FirstPassError) as caught:
        FirstPassRunner(_config(
            frames, tmp_path / "run", fake,
            **_billable(max_attempts=attempts))).run()
    assert caught.value.code == "INVALID_CONFIG"
    assert "max_attempts=1" in caught.value.message
    assert fake.calls == []


@pytest.mark.parametrize("budgets", [(1024, 2048), (), None, "1024"])
def test_billable_rejects_token_escalation_config(
        tmp_path: Path, budgets: Any) -> None:
    frames = _seq_frames(tmp_path, 1)
    fake = _FakeClient(default=_ok())
    with pytest.raises(FirstPassError) as caught:
        FirstPassRunner(_config(
            frames, tmp_path / "run", fake,
            **_billable(token_budgets=budgets))).run()
    assert caught.value.code == "INVALID_CONFIG"
    assert "exactly one token budget" in caught.value.message
    assert fake.calls == []


def test_billable_requires_uncertain_refusal(tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path, 1)
    fake = _FakeClient(default=_ok())
    with pytest.raises(FirstPassError) as caught:
        FirstPassRunner(_config(
            frames, tmp_path / "run", fake,
            **_billable(uncertain_policy="rerun"))).run()
    assert caught.value.code == "INVALID_CONFIG"
    assert "uncertain_policy='refuse'" in caught.value.message
    assert fake.calls == []


def test_billable_transport_failure_submits_once(
        tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path, 2)
    fake = _FakeClient(
        script={1: [InferenceError("TIMEOUT", "slow")]},
        default=_ok())
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake, batch_size=5,
        **_billable())).run()
    assert _call_indices(fake) == [1, 2]
    assert report["retries"] == 0
    assert report["frames_observed"] == 1
    assert report["frames_failed"] == 1
    assert report["lifecycle"]["terminal_states"] == {
        "request_failed": 1, "valid_prediction": 1}
    assert report["batches_needing_review"] == 1
    assert report["budget"]["stopped_reason"] is None


def test_billable_token_limit_does_not_escalate(
        tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path, 1)
    incomplete = _ok("", InferenceUsage(5, 5, 10),
                     status="incomplete",
                     incomplete_reason="max_output_tokens reached")
    fake = _FakeClient(script={1: [incomplete]})
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake,
        **_billable())).run()
    assert len(fake.calls) == 1
    assert fake.budgets == [1024]
    assert report["lifecycle"]["token_retries"] == 0
    assert report["lifecycle"]["terminal_states"] == {
        "exhausted_token_budget": 1}
    assert report["frames_observed"] == 0
    assert report["frames_failed"] == 1


def test_billable_uncertain_checkpoint_refused(
        tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path)
    out_dir = tmp_path / "run"
    crashing = _CrashingClient(crash_on=3)
    with pytest.raises(RuntimeError, match="simulated worker"):
        FirstPassRunner(_config(
            frames, out_dir, crashing, batch_size=2,
            **_billable())).run()
    assert crashing.calls == [1, 2, 3]
    cautious = _FakeClient(default=_ok())
    with pytest.raises(FirstPassError) as caught:
        FirstPassRunner(_config(
            frames, out_dir, cautious, batch_size=2,
            **_billable())).run()
    assert caught.value.code == "UNCERTAIN_REQUESTS"
    assert cautious.calls == []


def test_billable_valid_run_completes(tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path, 2)
    fake = _FakeClient(default=_ok())
    report = FirstPassRunner(_config(
        frames, tmp_path / "run", fake, batch_size=5,
        **_billable())).run()
    assert len(fake.calls) == 2
    assert report["frames_observed"] == 2
    assert report["frames_failed"] == 0
    assert report["batches_needing_review"] == 0
    assert report["budget"]["stopped_reason"] is None
    assert report["billable"] is True


def test_offline_default_reports_not_billable(
        tmp_path: Path) -> None:
    frames = _seq_frames(tmp_path, 1)
    fake = _FakeClient(default=_ok())
    config = _config(frames, tmp_path / "run", fake)
    assert config.billable is False
    report = FirstPassRunner(config).run()
    assert report["billable"] is False
    assert report["frames_observed"] == 1


# Incomplete-response evidence and accounting (S5-B P0) ----------------------


def test_billable_exhausted_preserves_final_response(
        tmp_path: Path) -> None:
    """Live-incident shape: one billable submit ends incomplete
    with reported usage. The raw record must keep the response;
    the submit must count as unknown cost without rates."""
    frames = _seq_frames(tmp_path, 1)
    usage = InferenceUsage(
        2913, 1240, 4153, cached_input_tokens=128,
        reasoning_output_tokens=542)
    partial = '{"visible_objects": ["scale"], "scale_dis'
    incomplete = _ok(partial, usage, status="incomplete",
                     incomplete_reason="max_output_tokens",
                     request_id="resp_live_1",
                     latency_seconds=1.7)
    fake = _FakeClient(script={1: [incomplete]})
    out_dir = tmp_path / "run"
    report = FirstPassRunner(_config(
        frames, out_dir, fake,
        **_billable(token_budgets=(512,)))).run()
    assert len(fake.calls) == 1
    assert fake.budgets == [512]
    assert report["frames_observed"] == 0
    assert report["frames_failed"] == 1
    assert report["batches_needing_review"] == 1
    assert report["lifecycle"]["terminal_states"] == {
        "exhausted_token_budget": 1}
    assert report["input_tokens"] == 2913
    assert report["output_tokens"] == 1240
    assert report["cost_usd_estimated"] is None
    assert report["cost_status"] == "COST_UNKNOWN"
    budget = report["budget"]
    assert budget["requests_submitted"] == 1
    assert budget["cost_unknown_requests"] == 1
    assert budget["spent_cost_usd_estimate"] is None
    assert budget["spent_cost_known_usd_estimate"] == 0.0
    assert _obs_index(out_dir) == {}
    raw = _read_json(out_dir / "raw_responses"
                     / "batch_00000001-00000001.json")
    assert len(raw) == 1
    record = raw[0]
    assert record["error_code"] == "INCOMPLETE_STATUS"
    assert record["terminal_state"] == "exhausted_token_budget"
    assert record["incomplete_reason"] == "max_output_tokens"
    assert record["budgets_attempted"] == [512]
    assert record["attempts"] == [
        {"attempt": 1, "budget": 512, "code": "ok"}]
    assert record["provider"] == "meta"
    assert record["model"] == "fake-model"
    assert record["status"] == "incomplete"
    assert record["output_text"] == partial
    assert record["usage"] == {
        "input_tokens": 2913, "output_tokens": 1240,
        "total_tokens": 4153, "cached_input_tokens": 128,
        "reasoning_output_tokens": 542,
    }
    assert record["latency_seconds"] == 1.7
    assert record["request_id"] == "resp_live_1"


def test_incomplete_without_usage_counts_unknown(
        tmp_path: Path) -> None:
    """An incomplete response without usage is unknown cost,
    never zero, even when rates are configured."""
    frames = _seq_frames(tmp_path, 1)
    incomplete = _ok('{"partial": true', InferenceUsage(),
                     status="incomplete",
                     incomplete_reason="max_output_tokens")
    fake = _FakeClient(script={1: [incomplete]})
    out_dir = tmp_path / "run"
    report = FirstPassRunner(_config(
        frames, out_dir, fake,
        rates_usd_per_m=DOLLAR_RATES)).run()
    assert report["frames_observed"] == 0
    assert report["input_tokens"] == 0
    assert report["output_tokens"] == 0
    assert report["cost_usd_estimated"] is None
    assert report["cost_status"] == "COST_UNKNOWN"
    budget = report["budget"]
    assert budget["requests_submitted"] == 1
    assert budget["cost_unknown_requests"] == 1
    assert budget["spent_cost_usd_estimate"] is None
    assert budget["spent_cost_known_usd_estimate"] == 0.0
    raw = _read_json(out_dir / "raw_responses"
                     / "batch_00000001-00000001.json")
    assert raw[0]["output_text"] == '{"partial": true'
    assert raw[0]["usage"] == {
        "input_tokens": None, "output_tokens": None,
        "total_tokens": None, "cached_input_tokens": None,
        "reasoning_output_tokens": None,
    }


def test_incomplete_empty_text_still_counts_usage(
        tmp_path: Path) -> None:
    """Empty partial text stays a failure, but its reported
    usage still counts toward totals and known spend."""
    frames = _seq_frames(tmp_path, 1)
    incomplete = _ok("", DOLLAR_USAGE, status="incomplete",
                     incomplete_reason="max_output_tokens")
    fake = _FakeClient(script={1: [incomplete]})
    out_dir = tmp_path / "run"
    report = FirstPassRunner(_config(
        frames, out_dir, fake,
        rates_usd_per_m=DOLLAR_RATES)).run()
    assert report["frames_observed"] == 0
    assert report["lifecycle"]["terminal_states"] == {
        "exhausted_token_budget": 1}
    assert report["input_tokens"] == 10
    assert report["output_tokens"] == 20
    assert report["cost_usd_estimated"] == 30.0
    budget = report["budget"]
    assert budget["cost_unknown_requests"] == 0
    assert budget["spent_cost_usd_estimate"] == 30.0
    assert budget["spent_cost_known_usd_estimate"] == 30.0
    raw = _read_json(out_dir / "raw_responses"
                     / "batch_00000001-00000001.json")
    assert raw[0]["output_text"] == ""
    assert raw[0]["usage"]["input_tokens"] == 10


def test_completed_raw_usage_carries_reasoning_and_cached(
        tmp_path: Path) -> None:
    """Success raw records persist the full usage block.
    Reasoning tokens are evidence only: billed spend still
    derives from input/output totals alone."""
    frames = _seq_frames(tmp_path, 1)
    usage = InferenceUsage(10, 20, 30, cached_input_tokens=4,
                           reasoning_output_tokens=6)
    fake = _FakeClient(default=_ok(usage=usage))
    out_dir = tmp_path / "run"
    report = FirstPassRunner(_config(
        frames, out_dir, fake,
        rates_usd_per_m=DOLLAR_RATES)).run()
    assert report["frames_observed"] == 1
    raw = _read_json(out_dir / "raw_responses"
                     / "batch_00000001-00000001.json")
    assert raw[0]["usage"] == {
        "input_tokens": 10, "output_tokens": 20,
        "total_tokens": 30, "cached_input_tokens": 4,
        "reasoning_output_tokens": 6,
    }
    assert report["budget"]["spent_cost_usd_estimate"] == 30.0


def test_exhausted_multi_budget_counts_final_usage_once(
        tmp_path: Path) -> None:
    """Escalation ending exhausted: every submit counts toward
    spend exactly once; totals carry the final round only."""
    frames = _seq_frames(tmp_path, 1)
    first = _ok("", InferenceUsage(5, 5, 10),
                status="incomplete",
                incomplete_reason="max_output_tokens")
    second = _ok("", InferenceUsage(7, 9, 16),
                 status="incomplete",
                 incomplete_reason="max_output_tokens")
    fake = _FakeClient(script={1: [first, second]})
    out_dir = tmp_path / "run"
    report = FirstPassRunner(_config(
        frames, out_dir, fake, token_budgets=(1024, 2048),
        rates_usd_per_m=DOLLAR_RATES)).run()
    assert fake.budgets == [1024, 2048]
    assert report["frames_observed"] == 0
    assert report["lifecycle"]["terminal_states"] == {
        "exhausted_token_budget": 1}
    assert report["lifecycle"]["token_retries"] == 1
    budget = report["budget"]
    assert budget["requests_submitted"] == 2
    assert budget["cost_unknown_requests"] == 0
    assert budget["spent_cost_known_usd_estimate"] == 26.0
    assert report["input_tokens"] == 7
    assert report["output_tokens"] == 9
    assert report["cost_usd_estimated"] == 16.0
    raw = _read_json(out_dir / "raw_responses"
                     / "batch_00000001-00000001.json")
    assert raw[0]["budgets_attempted"] == [1024, 2048]
    assert raw[0]["usage"]["input_tokens"] == 7
    assert raw[0]["usage"]["output_tokens"] == 9


def test_incomplete_unknown_cost_strict_halts(
        tmp_path: Path) -> None:
    """Strict mode halts on an incomplete response without
    usage, same as any other unreported usage; the failure
    record is still durably written."""
    frames = _seq_frames(tmp_path, 3)
    incomplete = _ok("", InferenceUsage(),
                     status="incomplete",
                     incomplete_reason="max_output_tokens")
    fake = _FakeClient(
        script={1: [incomplete]},
        default=_ok(usage=DOLLAR_USAGE))
    out_dir = tmp_path / "run"
    report = FirstPassRunner(_config(
        frames, out_dir, fake, batch_size=5,
        rates_usd_per_m=DOLLAR_RATES,
        cost_ceiling_usd=1000.0, strict_cost=True)).run()
    assert len(fake.calls) == 1
    assert report["frames_observed"] == 0
    assert report["frames_failed"] == 1
    assert report["frames_unprocessed"] == 2
    budget = report["budget"]
    assert budget["stopped_reason"] == "unknown_cost_strict_halt"
    assert budget["stopped_detail"] == {
        "phase": "post_response",
        "cause": "usage_unreported",
        "cost_unknown_requests": 1,
        "cost_ceiling_usd": 1000.0,
    }
    assert budget["cost_unknown_requests"] == 1
    assert budget["spent_cost_usd_estimate"] is None
    assert budget["remaining_frame_indices"] == [2, 3]
    raw = _read_json(out_dir / "raw_responses"
                     / "batch_00000001-00000003.json")
    assert len(raw) == 1
    assert raw[0]["error_code"] == "INCOMPLETE_STATUS"


def test_exhausted_usage_seeds_resume_without_double_count(
        tmp_path: Path) -> None:
    """Failed-frame spend seeds resume accounting exactly
    once; the resumed batch merges prior evidence with the
    new outcome instead of re-counting the old submit."""
    frames = _seq_frames(tmp_path, 1)
    out_dir = tmp_path / "run"
    incomplete = _ok("", InferenceUsage(5, 5, 10),
                     status="incomplete",
                     incomplete_reason="max_output_tokens")
    first = _FakeClient(script={1: [incomplete]})
    partial = FirstPassRunner(_config(
        frames, out_dir, first,
        rates_usd_per_m=DOLLAR_RATES)).run()
    assert partial["batches_needing_review"] == 1
    assert (partial["budget"]["spent_cost_known_usd_estimate"]
            == 10.0)
    second = _FakeClient(default=_ok(usage=DOLLAR_USAGE))
    report = FirstPassRunner(_config(
        frames, out_dir, second,
        rates_usd_per_m=DOLLAR_RATES)).run()
    assert _call_indices(second) == [1]
    assert report["frames_observed"] == 1
    budget = report["budget"]
    assert budget["prior_requests_submitted"] == 1
    assert budget["requests_submitted"] == 1
    assert budget["prior_spent_cost_known_usd_estimate"] == 10.0
    assert budget["spent_cost_known_usd_estimate"] == 30.0
    assert budget["cost_unknown_requests"] == 0
    assert budget["prior_cost_unknown_requests"] == 0
    records = _checkpoints(out_dir)
    assert len(records) == 2
    assert records[1]["input_tokens"] == 15
    assert records[1]["output_tokens"] == 25
    raw = _read_json(out_dir / "raw_responses"
                     / "batch_00000001-00000001.json")
    assert len(raw) == 2
    assert raw[0]["error_code"] == "INCOMPLETE_STATUS"
    assert "error_code" not in raw[1]


def test_failed_raw_record_holds_no_secrets(
        tmp_path: Path) -> None:
    """The threaded credential must reach the client seam but
    never land in persisted evidence or reports."""
    key = "SYNTHETIC-KEY-9f8e7d6c5b4a"
    frames = _seq_frames(tmp_path, 1)
    incomplete = _ok('{"partial": true', DOLLAR_USAGE,
                     status="incomplete",
                     incomplete_reason="max_output_tokens",
                     request_id="resp_synth_1")
    fake = _FakeClient(script={1: [incomplete]})
    out_dir = tmp_path / "run"
    config = _config(frames, out_dir, fake, api_key=key)
    assert key not in repr(config)
    FirstPassRunner(config).run()
    assert fake.calls[0][2] == key
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            assert key not in path.read_text(encoding="utf-8")
