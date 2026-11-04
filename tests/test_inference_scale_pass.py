"""Parallel scale-reading engine (scale_pass) regression tests.

Offline coverage for 10-image batching, bounded
concurrency, strict frame-ID validation, abstentions,
transport failures without retry, worst-case cost
reservation, checkpoint resume without rebilling, and the
downstream observation envelope. All tests run offline via
deterministic doubles.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import pytest

from bakery_intelligence.inference.firstpass import (
    FirstPassError,
    FrameRef,
    InferenceError,
    InferenceResponse,
    InferenceUsage,
    SourceRef,
)
from bakery_intelligence.inference.firstpass_v3 import (
    visual_frame_id_for,
)
from bakery_intelligence.inference.scale_pass import (
    DEFAULT_SCALE_CONCURRENCY,
    DEFAULT_SCALE_GROUP_SIZE,
    DEFAULT_SCALE_TOKEN_BUDGET,
    DeterministicScaleFakeClient,
    ScalePassConfig,
    ScalePassRunner,
    build_scale_prompt,
    validate_scale_response,
    worst_single_cost_usd,
)


def _frames(tmp_path: Path, count: int,
            start: int = 0) -> list[FrameRef]:
    from PIL import Image

    img_dir = tmp_path / "scalephotos"
    img_dir.mkdir(exist_ok=True)
    frames = []
    for offset in range(count):
        index = start + offset
        path = img_dir / f"frame_{index:08d}.jpg"
        shade = (index * 53) % 256
        Image.new("RGB", (48, 64),
                  (shade, 120, 40)).save(path, format="JPEG")
        frames.append(FrameRef(
            frame_index=index, image_path=path,
            timestamp_seconds=float(index)))
    return frames


def _source() -> SourceRef:
    return SourceRef(name="clip.mp4", sha256="ab" * 32)


def _config(tmp_path: Path, frames: list[FrameRef],
            client: Any, **overrides: Any
            ) -> ScalePassConfig:
    params: dict[str, Any] = {
        "frames": frames,
        "source": _source(),
        "out_dir": tmp_path / "sp",
        "rates_usd_per_m": (1.25, 4.25),
        "client_factory": lambda: client,
    }
    params.update(overrides)
    return ScalePassConfig(**params)


def test_batching_ten_and_partial_final(tmp_path: Path) -> None:
    frames = _frames(tmp_path, 25)
    client = DeterministicScaleFakeClient()
    report = ScalePassRunner(
        _config(tmp_path, frames, client)).run()
    assert report["groups_total"] == 3
    assert report["groups_valid"] == 3
    assert report["frames_observed"] == 25
    assert report["requests_sent"] == 3
    assert report["stopped_reason"] is None
    sizes = [len(json.loads(
        (tmp_path / "sp" / "observations"
         / f"batch_{index:03d}.json").read_text(
             encoding="utf-8"))) for index in range(3)]
    assert sizes == [10, 10, 5]


def test_defaults_match_validated_experiment() -> None:
    assert DEFAULT_SCALE_GROUP_SIZE == 10
    assert DEFAULT_SCALE_CONCURRENCY == 5
    assert DEFAULT_SCALE_TOKEN_BUDGET == 8192


def test_batches_run_concurrently(tmp_path: Path) -> None:
    barrier = threading.Barrier(5)
    entered = threading.Event()

    class _BarrierFake(DeterministicScaleFakeClient):
        def submit(self, prompt_text: str,
                   image_paths: Sequence[Path],
                   api_key: str, max_output_tokens: int,
                   response_schema: Any = None,
                   response_schema_name: Any | None = None,
                   ) -> InferenceResponse:
            entered.set()
            barrier.wait(timeout=30)
            return super().submit(
                prompt_text, image_paths, api_key,
                max_output_tokens, response_schema,
                response_schema_name)

    frames = _frames(tmp_path, 50)
    client = _BarrierFake()
    report = ScalePassRunner(
        _config(tmp_path, frames, client)).run()
    assert entered.is_set()
    assert report["groups_valid"] == 5
    assert report["frames_observed"] == 50


def test_out_of_order_completion_reassembles(
        tmp_path: Path) -> None:
    class _SlowFirstFake(DeterministicScaleFakeClient):
        def submit(self, prompt_text: str,
                   image_paths: Sequence[Path],
                   api_key: str, max_output_tokens: int,
                   response_schema: Any = None,
                   response_schema_name: Any | None = None,
                   ) -> InferenceResponse:
            first = int(list(image_paths)[0].stem.split(
                "_")[1])
            time.sleep(max(0.0, (20 - first) * 0.01))
            return super().submit(
                prompt_text, image_paths, api_key,
                max_output_tokens, response_schema,
                response_schema_name)

    frames = _frames(tmp_path, 30)
    client = _SlowFirstFake()
    report = ScalePassRunner(
        _config(tmp_path, frames, client)).run()
    assert report["frames_observed"] == 30
    seen = {}
    for index in range(3):
        records = json.loads(
            (tmp_path / "sp" / "observations"
             / f"batch_{index:03d}.json").read_text(
                 encoding="utf-8"))
        for record in records:
            seen[record["frame_index"]] = record[
                "model_fields"]["visual_frame_id"]
    assert len(seen) == 30
    for index in range(30):
        assert seen[index] == visual_frame_id_for(index)


def _payload(ids: list[str]) -> str:
    return json.dumps({"frames": [
        {"frame_id": frame_id, "display_value": "12",
         "unit": "g", "confidence": "high"}
        for frame_id in ids]})


def test_scale_prompt_states_strict_rules() -> None:
    prompt = build_scale_prompt([("F000000", 0.0)])
    assert "FRAME-ID" in prompt
    assert "never invent a reading" in prompt
    assert "F000000" in prompt


def test_validation_accepts_exact_coverage() -> None:
    ids = ["F000000", "F000001"]
    result = validate_scale_response(_payload(ids), ids)
    assert result.ok
    assert result.observations is not None
    assert sorted(result.observations) == ids


def test_validation_rejects_id_mismatch() -> None:
    ids = ["F000000", "F000001"]
    short = validate_scale_response(_payload(ids[:1]), ids)
    assert not short.ok
    assert short.error is not None
    assert short.error.startswith("frame_count_mismatch")
    swapped = json.loads(_payload(ids))
    swapped["frames"][1]["frame_id"] = "F000009"
    unexpected = validate_scale_response(
        json.dumps(swapped), ids)
    assert not unexpected.ok
    assert unexpected.error is not None
    assert unexpected.error.startswith("unexpected_frame_id")
    doubled = json.loads(_payload(ids))
    doubled["frames"][1]["frame_id"] = "F000000"
    duplicate = validate_scale_response(
        json.dumps(doubled), ids)
    assert not duplicate.ok
    assert duplicate.error == (
        "duplicate_frame_id:F000000")
    broken = validate_scale_response("not json {", ids)
    assert not broken.ok


def test_validation_rejects_bad_field_types() -> None:
    ids = ["F000000"]
    payload = json.loads(_payload(ids))
    payload["frames"][0]["confidence"] = "certain"
    assert not validate_scale_response(
        json.dumps(payload), ids).ok
    payload["frames"][0]["confidence"] = "high"
    payload["frames"][0]["display_value"] = 12
    assert not validate_scale_response(
        json.dumps(payload), ids).ok


def test_runner_rejects_invalid_batch_without_retry(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path, 10)
    ids = [visual_frame_id_for(frame.frame_index)
           for frame in frames]
    payload = json.loads(_payload(ids))
    payload["frames"].pop()
    client = DeterministicScaleFakeClient(
        {0: json.dumps(payload)})
    report = ScalePassRunner(
        _config(tmp_path, frames, client)).run()
    assert report["groups_valid"] == 0
    assert report["groups_review"] == 1
    assert report["frames_failed"] == 10
    assert report["requests_sent"] == 1
    assert len(client.calls) == 1
    assert report["stopped_reason"] == "batches_incomplete"


def test_abstention_counts_as_observed(tmp_path: Path) -> None:
    frames = _frames(tmp_path, 10)
    ids = [visual_frame_id_for(frame.frame_index)
           for frame in frames]
    payload = {"frames": [
        {"frame_id": frame_id, "display_value": None,
         "unit": None, "confidence": "low"}
        for frame_id in ids]}
    client = DeterministicScaleFakeClient(
        {0: json.dumps(payload)})
    report = ScalePassRunner(
        _config(tmp_path, frames, client)).run()
    assert report["groups_valid"] == 1
    assert report["frames_observed"] == 10
    assert report["stopped_reason"] is None
    records = json.loads(
        (tmp_path / "sp" / "observations"
         / "batch_000.json").read_text(encoding="utf-8"))
    assert records[0]["terminal_state"] == "abstained"
    assert records[0]["model_fields"][
        "scale_display_verbatim"] is None
    assert records[0]["model_fields"][
        "scale_readable"] is False


def test_transport_failure_never_retries(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path, 10)

    def _boom() -> DeterministicScaleFakeClient:
        raise AssertionError("unreachable")

    class _TimeoutFake(DeterministicScaleFakeClient):
        def submit(self, *args: Any, **kwargs: Any
                   ) -> InferenceResponse:
            raise InferenceError("TIMEOUT", "timed out")

    client = _TimeoutFake()
    report = ScalePassRunner(
        _config(tmp_path, frames, client)).run()
    assert report["groups_valid"] == 0
    assert report["groups_exhausted"] == 1
    assert report["requests_sent"] == 1
    assert report["stopped_reason"] == "batches_incomplete"
    raw = json.loads(
        (tmp_path / "sp" / "raw_responses"
         / "batch_000.json").read_text(encoding="utf-8"))
    assert raw["status"] == "transport_failed"
    assert raw["request_id"] is None


def test_strict_transport_unknown_cost_halts(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path, 20)

    class _TimeoutFake(DeterministicScaleFakeClient):
        def submit(self, *args: Any, **kwargs: Any
                   ) -> InferenceResponse:
            raise InferenceError("TIMEOUT", "timed out")

    client = _TimeoutFake()
    report = ScalePassRunner(_config(
        tmp_path, frames, client, billable=True,
        uncertain_policy="refuse",
        strict_cost=True, cost_ceiling_usd=20.0)).run()
    assert report["stopped_reason"] in (
        "unknown_cost_strict_halt", "batches_incomplete")
    assert report["groups_valid"] == 0


def test_worst_case_reservation_refuses_dispatch(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path, 20)
    calls: list[str] = []

    def _factory() -> Any:
        calls.append("client")
        return DeterministicScaleFakeClient()

    config = _config(tmp_path, frames,
                     DeterministicScaleFakeClient(),
                     billable=True, strict_cost=True,
                     uncertain_policy="refuse",
                     cost_ceiling_usd=0.01)
    config.client_factory = _factory
    runner = ScalePassRunner(config)
    try:
        runner.run()
    except FirstPassError as exc:
        assert exc.code == "COST_RESERVE"
    else:
        raise AssertionError("expected COST_RESERVE")
    assert calls == []


def test_worst_case_math_covers_full_video() -> None:
    single = worst_single_cost_usd(
        input_token_bound=9000, token_budget=8192,
        rates_usd_per_m=(1.25, 4.25))
    assert single == pytest.approx(
        9000 / 1_000_000 * 1.25
        + 8192 / 1_000_000 * 4.25)
    assert round(single * 65, 4) < 20.0


def test_resume_skips_completed_without_rebill(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path, 25)
    failing = DeterministicScaleFakeClient({
        20: InferenceError("TIMEOUT", "timed out")})
    first = ScalePassRunner(
        _config(tmp_path, frames, failing)).run()
    assert first["groups_valid"] == 2
    assert first["frames_failed"] == 5
    clean = DeterministicScaleFakeClient()
    second = ScalePassRunner(
        _config(tmp_path, frames, clean)).run()
    assert second["groups_valid"] == 3
    assert second["groups_resumed"] == 2
    assert second["frames_observed"] == 25
    assert second["requests_sent"] == 1
    assert second["stopped_reason"] is None
    assert len(clean.calls) == 1


def test_resume_refuses_received_invalid_batch(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path, 10)
    failing = DeterministicScaleFakeClient(
        {0: "not json {"})
    first = ScalePassRunner(
        _config(tmp_path, frames, failing)).run()
    assert first["groups_review"] == 1

    def _factory() -> Any:
        raise AssertionError(
            "must not dispatch on rebill refusal")

    config = _config(tmp_path, frames,
                     DeterministicScaleFakeClient())
    config.client_factory = _factory
    try:
        ScalePassRunner(config).run()
    except FirstPassError as exc:
        assert exc.code == "PRIOR_UNUSABLE_RESPONSE"
    else:
        raise AssertionError("expected refusal")


def test_observation_envelope_matches_contract(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path, 10)
    client = DeterministicScaleFakeClient()
    ScalePassRunner(
        _config(tmp_path, frames, client)).run()
    records = json.loads(
        (tmp_path / "sp" / "observations"
         / "batch_000.json").read_text(encoding="utf-8"))
    record = records[0]
    for key in ("frame_index", "image_path",
                "timestamp_seconds", "manifest_ref",
                "batch_id", "pass_id", "call",
                "model_fields", "parse_error",
                "response_status", "terminal_state",
                "image_ref"):
        assert key in record, key
    fields = record["model_fields"]
    assert fields["scale_display_verbatim"] == "0"
    assert fields["scale_readable"] is True
    assert fields["scale_unit_visible"] == "g"
    assert fields["scale_state"] == "unknown"
    assert fields["visible_objects"] == []
    assert fields["visual_frame_id"] == "F000000"
    assert record["image_ref"]["encoding"] == "png"
    assert "prepared_sha256" in record["image_ref"]
    raw = json.loads(
        (tmp_path / "sp" / "raw_responses"
         / "batch_000.json").read_text(encoding="utf-8"))
    assert raw["frame_indices"] == list(range(10))
    assert raw["incoming_context_kind"] == "none"
    manifest = json.loads(
        (tmp_path / "sp" / "anchored"
         / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 10
    assert manifest[0]["photo_region_verified"] is True


def test_max_batches_cap_enforced(tmp_path: Path) -> None:
    frames = _frames(tmp_path, 25)
    client = DeterministicScaleFakeClient()
    try:
        ScalePassRunner(_config(
            tmp_path, frames, client,
            max_batches=2)).run()
    except FirstPassError as exc:
        assert exc.code == "INVALID_CONFIG"
    else:
        raise AssertionError("expected cap refusal")
