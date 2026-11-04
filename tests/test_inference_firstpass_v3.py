"""Focused large-batch FirstPass (V3) regression tests.

Offline coverage for grouping, schema validation,
deterministic previous-batch context, runner safeguards,
and checkpoint compatibility. All tests run offline via
the deterministic fake or scripted doubles.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    ANCHOR_BAND_HEIGHT,
    ANCHOR_CODE_LENGTH,
    DEFAULT_FOCUSED_GROUP_SIZE,
    DEFAULT_FOCUSED_TOKEN_BUDGET,
    FOCUSED_ANCHORED_JSON_SCHEMA,
    FOCUSED_JSON_SCHEMA,
    FOCUSED_SCHEMA_NAME,
    PASS_ID_V3,
    PREV_CONTEXT_MAX_CHARS,
    V3_ANCHORED_OUTPUT_FIELDS,
    V3_MODEL_OUTPUT_FIELDS,
    DeterministicFocusedFakeClient,
    FirstPassV3Config,
    FirstPassV3Runner,
    build_focused_schema,
    build_focused_prompt,
    build_previous_batch_context,
    prepare_anchored_image,
    validate_focused_response,
    verify_anchored_image,
    visual_check_code,
    visual_frame_id_for,
)


def _frames(tmp_path: Path, count: int,
            start: int = 0) -> list[FrameRef]:
    img_dir = tmp_path / "v3imgs"
    img_dir.mkdir(exist_ok=True)
    frames = []
    for offset in range(count):
        index = start + offset
        path = img_dir / f"frame_{index:08d}.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
        frames.append(FrameRef(
            frame_index=index, image_path=path,
            timestamp_seconds=float(index)))
    return frames


def _source() -> SourceRef:
    return SourceRef(name="test@1fps", sha256="ab" * 32)


class _ScriptedFocusedClient:
    """Per-submit scripted double; records prompts."""

    def __init__(self, steps: list[Any]) -> None:
        self._steps = list(steps)
        self._fake = DeterministicFocusedFakeClient()
        self.calls: list[dict[str, Any]] = []
        self.prompts: list[str] = []

    @property
    def model_id(self) -> str:
        return "scripted-focused-v1"

    def submit(self, prompt_text: str, image_paths: Any,
               api_key: str, max_output_tokens: int,
               response_schema: Any = None,
               response_schema_name: Any = None) -> Any:
        _ = (api_key, response_schema)
        self.calls.append({
            "images": len(list(image_paths)),
            "max_output_tokens": max_output_tokens,
            "schema": response_schema_name,
        })
        self.prompts.append(prompt_text)
        if not self._steps:
            raise AssertionError("script exhausted")
        step = self._steps.pop(0)
        if step == "valid":
            return self._fake.submit(
                prompt_text, image_paths, api_key,
                max_output_tokens, response_schema,
                response_schema_name)
        if isinstance(step, str):
            return _reply(step, request_id=f"r{len(self.calls)}")
        if isinstance(step, Exception):
            raise step
        return step


def _reply(text: str, *, status: str = "completed",
           request_id: str = "r1",
           incomplete: str | None = None) -> InferenceResponse:
    return InferenceResponse(
        model="scripted-focused-v1", status=status,
        output_text=text,
        usage=InferenceUsage(
            input_tokens=100, output_tokens=50,
            total_tokens=150),
        latency_seconds=1.25, request_id=request_id,
        incomplete_reason=incomplete)


def _config(tmp_path: Path, frames: list[FrameRef],
            client: Any, **over: Any) -> FirstPassV3Config:
    kwargs: dict[str, Any] = {
        "frames": frames,
        "source": _source(),
        "out_dir": tmp_path / "fp",
        "group_size": 2,
        "max_attempts": 1,
        "max_validation_attempts": 3,
        "max_consecutive_exhausted": 2,
        "client_factory": lambda: client,
        "sleeper": lambda seconds: None,
    }
    kwargs.update(over)
    return FirstPassV3Config(**kwargs)


def _run(tmp_path: Path, frames: list[FrameRef],
         client: Any, **over: Any) -> dict[str, Any]:
    return FirstPassV3Runner(
        _config(tmp_path, frames, client, **over)).run()


def _raw(tmp_path: Path, position: int = 0) -> dict[str, Any]:
    paths = sorted(
        (tmp_path / "fp" / "raw_responses").glob("group_*.json"))
    return json.loads(paths[position].read_text(encoding="utf-8"))


def _valid_text(indices: list[int]) -> str:
    entries = [{
        "frame_index": index,
        "timestamp_seconds": float(index),
        "observation": {
            "scale_readable": True,
            "scale_display_verbatim": "42",
            "scale_unit_visible": "g",
            "scale_state": "stable",
            "tare_action_visible": False,
            "container_description": "bowl",
            "container_changed_hint": False,
            "primary_utensil": "spoon",
            "primary_visible_material": "flour",
            "action": "stirring",
            "addition_or_removal_visible": "none",
            "confidence": "high",
            "ambiguity_notes": None,
        },
    } for index in indices]
    return json.dumps({"frames": entries, "events": []})


# --- Grouping and configuration. ---

def test_grouping_50_with_partial_final(tmp_path: Path) -> None:
    assert DEFAULT_FOCUSED_GROUP_SIZE == 50
    assert DEFAULT_FOCUSED_TOKEN_BUDGET == 65536
    frames = _frames(tmp_path, 645)
    client = _ScriptedFocusedClient(["valid"] * 13)
    report = _run(tmp_path, frames, client, group_size=50,
                  max_consecutive_exhausted=None)
    assert report["groups_total"] == 13
    sizes = [group["frames_expected"]
             for group in report["groups"]]
    assert sizes == [50] * 12 + [45]
    assert report["frames_observed"] == 645
    assert report["token_budget_per_group"] == 65536


def test_group_size_over_image_max_rejected(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path, 4)
    client = _ScriptedFocusedClient([])
    with pytest.raises(FirstPassError) as caught:
        _run(tmp_path, frames, client, group_size=51)
    assert caught.value.code == "INVALID_CONFIG"


def test_config_rejects_bad_policy(tmp_path: Path) -> None:
    frames = _frames(tmp_path, 2)
    bad = [
        {"group_size": 0},
        {"token_budget": 0},
        {"max_attempts": 0},
        {"max_validation_attempts": 0},
        {"max_consecutive_exhausted": 0},
        {"max_groups": -1},
        {"uncertain_policy": "maybe"},
    ]
    for knob in bad:
        with pytest.raises(FirstPassError) as caught:
            _run(tmp_path, frames, _ScriptedFocusedClient([]),
                 **knob)
        assert caught.value.code == "INVALID_CONFIG"


def test_billable_requires_strict_posture(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path, 2)
    with pytest.raises(FirstPassError) as first:
        _run(tmp_path, frames, _ScriptedFocusedClient([]),
             billable=True, max_attempts=2,
             uncertain_policy="refuse", api_key="k")
    assert first.value.code == "INVALID_CONFIG"
    with pytest.raises(FirstPassError) as second:
        _run(tmp_path, frames, _ScriptedFocusedClient([]),
             billable=True, uncertain_policy="rerun",
             api_key="k")
    assert second.value.code == "INVALID_CONFIG"
    allowed = _config(tmp_path, frames,
                      _ScriptedFocusedClient([]),
                      billable=True, uncertain_policy="refuse",
                      api_key="")
    FirstPassV3Runner(allowed)


# --- Schema validation. ---

def test_schema_accepts_full_batch() -> None:
    result = validate_focused_response(
        _valid_text([0, 1]), [(0, 0.0), (1, 1.0)])
    assert result.ok
    assert len(result.entries) == 2
    assert result.events == ()


def test_schema_rejects_coverage_and_order() -> None:
    assert not validate_focused_response(
        _valid_text([0]), [(0, 0.0), (1, 1.0)]).ok
    assert not validate_focused_response(
        _valid_text([1, 0]), [(0, 0.0), (1, 1.0)]).ok
    assert not validate_focused_response(
        _valid_text([0, 0]), [(0, 0.0), (1, 1.0)]).ok
    assert not validate_focused_response(
        "not json", [(0, 0.0)]).ok
    assert not validate_focused_response(
        json.dumps({"frames": "nope"}), [(0, 0.0)]).ok


def test_schema_rejects_bad_observation_and_events() -> None:
    payload = json.loads(_valid_text([0]))
    del payload["frames"][0]["observation"]["action"]
    assert not validate_focused_response(
        json.dumps(payload), [(0, 0.0)]).ok
    payload = json.loads(_valid_text([0]))
    payload["frames"][0]["timestamp_seconds"] = "soon"
    assert not validate_focused_response(
        json.dumps(payload), [(0, 0.0)]).ok
    payload = json.loads(_valid_text([0]))
    payload["events"] = [{"description": "x"}]
    assert not validate_focused_response(
        json.dumps(payload), [(0, 0.0)]).ok
    payload = json.loads(_valid_text([0]))
    payload["events"] = [{"description": "pour",
                          "frame_indices": [0]}]
    assert validate_focused_response(
        json.dumps(payload), [(0, 0.0)]).ok


def test_schema_dedupes_identical_repeat() -> None:
    payload = json.loads(_valid_text([0, 1]))
    payload["frames"].append(payload["frames"][-1])
    result = validate_focused_response(
        json.dumps(payload), [(0, 0.0), (1, 1.0)])
    assert result.ok
    assert result.deduped == 1
    assert [entry["frame_index"]
            for entry in result.entries] == [0, 1]


def test_schema_rejects_conflicting_repeat() -> None:
    payload = json.loads(_valid_text([0, 1]))
    repeat = json.loads(json.dumps(payload["frames"][-1]))
    repeat["observation"]["scale_display_verbatim"] = "43"
    payload["frames"].append(repeat)
    result = validate_focused_response(
        json.dumps(payload), [(0, 0.0), (1, 1.0)])
    assert not result.ok
    assert result.error == "duplicate_frame_conflict:frame_1"


def test_schema_rejects_non_duplicate_surplus() -> None:
    result = validate_focused_response(
        _valid_text([0, 1, 2]), [(0, 0.0), (1, 1.0)])
    assert not result.ok
    assert result.error is not None
    assert result.error.startswith("frame_count_mismatch")


# --- Deterministic previous-batch context. ---

def _entry(index: int, **fields: Any) -> dict[str, Any]:
    observation = {
        "scale_readable": False,
        "scale_display_verbatim": None,
        "scale_unit_visible": None,
        "scale_state": "unknown",
        "tare_action_visible": False,
        "container_description": None,
        "container_changed_hint": False,
        "primary_utensil": None,
        "primary_visible_material": None,
        "action": None,
        "addition_or_removal_visible": "unknown",
        "confidence": "low",
        "ambiguity_notes": None,
    }
    observation.update(fields)
    return {"frame_index": index, "observation": observation}


def test_context_builder_extracts_previous_batch() -> None:
    entries = [
        _entry(0, scale_readable=True,
               scale_display_verbatim="120",
               scale_unit_visible="g",
               container_description="steel bowl",
               primary_visible_material="white powder"),
        _entry(1, scale_readable=True,
               scale_display_verbatim="135",
               action="pouring",
               primary_utensil="jug",
               ambiguity_notes="label unreadable"),
    ]
    text = build_previous_batch_context(
        prev_group_id="group_0000", entries=entries)
    assert "group_0000" in text
    assert "'135' at frame 1" in text
    assert "'g'" in text
    assert "steel bowl" in text
    assert "pouring" in text
    assert "label unreadable" in text
    assert len(text) <= PREV_CONTEXT_MAX_CHARS


def test_context_builder_empty_batch_is_fresh_start() -> None:
    text = build_previous_batch_context(
        prev_group_id="group_0003", entries=[])
    assert "group_0003" in text
    assert "fresh start" in text


def test_context_builder_bounds_output() -> None:
    entries = [_entry(index, ambiguity_notes="word " * 200)
               for index in range(50)]
    text = build_previous_batch_context(
        prev_group_id="group_0000", entries=entries)
    assert len(text) <= PREV_CONTEXT_MAX_CHARS


def test_context_never_accumulates(tmp_path: Path) -> None:
    frames = _frames(tmp_path, 6)
    client = _ScriptedFocusedClient(["valid"] * 3)
    report = _run(tmp_path, frames, client)
    assert report["groups_valid"] == 3
    assert len(client.prompts) == 3
    assert "no earlier observations" in client.prompts[0]
    assert "PREVIOUS-BATCH CONTEXT" in client.prompts[1]
    # Batch 2 sees batch 1 only: batch-0 evidence (frame 0
    # reading "100") must not leak two hops forward.
    assert "frame 0" not in client.prompts[2]
    assert "group_0001" in client.prompts[2]
    first_raw = _raw(tmp_path, 0)
    assert first_raw["incoming_context_kind"] == "none"
    assert first_raw["incoming_context_chars"] == 0
    second_raw = _raw(tmp_path, 1)
    assert (second_raw["incoming_context_kind"]
            == "previous_batch")
    assert second_raw["incoming_context_chars"] > 0
    for position in range(3):
        payload = json.loads(
            _raw(tmp_path, position)["output_text"])
        assert set(payload.keys()) <= {"frames", "events"}
        assert "rolling_context" not in payload


# --- Runner behavior. ---

def test_runner_fake_success_with_partial_final(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path, 105)
    client = _ScriptedFocusedClient(["valid"] * 3)
    report = _run(tmp_path, frames, client, group_size=50,
                  max_consecutive_exhausted=None)
    assert report["pass_id"] == PASS_ID_V3
    assert report["groups_valid"] == 3
    assert report["frames_observed"] == 105
    assert report["stopped_reason"] is None
    assert {call["schema"] for call in client.calls} == {
        FOCUSED_SCHEMA_NAME}


def test_runner_retry_then_success(tmp_path: Path) -> None:
    frames = _frames(tmp_path, 2)
    client = _ScriptedFocusedClient(["garbage", "valid"])
    report = _run(tmp_path, frames, client)
    assert report["groups_valid"] == 1
    assert report["requests_sent"] == 2
    raw = _raw(tmp_path)
    assert len(raw["attempts"]) == 2
    assert raw["attempts"][0]["validation"].startswith(
        "validation:")
    assert raw["attempts"][1]["validation"] == "ok"


def test_runner_exhaustion_and_early_stop(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path, 6)
    client = _ScriptedFocusedClient(["bad"] * 12)
    report = _run(tmp_path, frames, client)
    assert report["requests_sent"] == 6
    assert report["groups_exhausted"] == 2
    assert (report["stopped_reason"]
            == "consecutive_group_exhaustion")
    assert report["budget"]["remaining_frame_indices"] == [4, 5]


def test_runner_counter_resets_after_success(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path, 10)
    client = _ScriptedFocusedClient(
        ["bad"] * 3 + ["valid"] + ["bad"] * 6)
    report = _run(tmp_path, frames, client)
    assert report["requests_sent"] == 10
    assert report["groups_valid"] == 1
    assert report["groups_exhausted"] == 3
    assert (report["budget"]["remaining_frame_indices"]
            == [8, 9])


def test_runner_no_retry_on_transport(tmp_path: Path) -> None:
    frames = _frames(tmp_path, 2)
    client = _ScriptedFocusedClient(
        [InferenceError("TIMEOUT", "slow")] * 3)
    report = _run(tmp_path, frames, client)
    assert report["requests_sent"] == 1
    raw = _raw(tmp_path)
    assert raw["terminal_state"] == "request_failed"
    assert len(raw["attempts"]) == 1


def test_runner_budget_halt_before_dispatch(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path, 4)
    client = _ScriptedFocusedClient(["valid"] * 4)
    report = _run(tmp_path, frames, client, max_groups=0)
    assert report["requests_sent"] == 0
    assert report["stopped_reason"] == "request_limit_reached"
    assert len(client.calls) == 0


def test_runner_strict_transport_halt(tmp_path: Path) -> None:
    frames = _frames(tmp_path, 4)
    client = _ScriptedFocusedClient(
        [InferenceError("NETWORK_ERROR", "down")] * 4)
    report = _run(tmp_path, frames, client, strict_cost=True,
                  cost_ceiling_usd=20.0,
                  rates_usd_per_m=(1.25, 4.25),
                  uncertain_policy="refuse")
    assert report["requests_sent"] == 1
    assert (report["stopped_reason"]
            == "unknown_cost_strict_halt")
    assert len(report["groups"]) == 1


def test_runner_request_counting_across_retries(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path, 8)
    client = _ScriptedFocusedClient(
        ["bad", "valid", "valid", "valid", "valid"])
    report = _run(tmp_path, frames, client, max_groups=4)
    assert report["requests_sent"] == 4
    assert report["requests_total"] == 4
    assert (report["stopped_reason"]
            == "request_limit_reached")


def test_runner_resume_rebuilds_context(tmp_path: Path) -> None:
    class _Crash(Exception):
        pass

    frames = _frames(tmp_path, 6)
    crashing = _ScriptedFocusedClient(["valid", _Crash()])
    with pytest.raises(_Crash):
        _run(tmp_path, frames, crashing)
    assert len(crashing.calls) == 2
    passing = _ScriptedFocusedClient(["valid"] * 4)
    report = _run(tmp_path, frames, passing)
    assert report["requests_sent"] == 2
    assert report["groups_resumed"] == 1
    assert report["groups_valid"] == 2
    resumed_raw = _raw(tmp_path, 1)
    assert (resumed_raw["incoming_context_kind"]
            == "previous_batch")
    assert resumed_raw["incoming_context_chars"] > 0
    assert "group_0000" in passing.prompts[0]


def test_runner_resume_after_stop_submits_nothing(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path, 6)
    failing = _ScriptedFocusedClient(["bad"] * 12)
    first = _run(tmp_path, frames, failing)
    assert (first["stopped_reason"]
            == "consecutive_group_exhaustion")
    passing = _ScriptedFocusedClient(["valid"] * 6)
    second = _run(tmp_path, frames, passing)
    assert len(passing.calls) == 0
    assert second["requests_sent"] == 0
    assert second["groups_resumed"] == 2


def test_runner_refuses_uncertain_resume(
        tmp_path: Path) -> None:
    class _Crash(Exception):
        pass

    frames = _frames(tmp_path, 4)
    crashing = _ScriptedFocusedClient([_Crash()])
    kwargs = {"billable": True, "uncertain_policy": "refuse",
              "api_key": "test"}
    with pytest.raises(_Crash):
        _run(tmp_path, frames, crashing, **kwargs)
    with pytest.raises(FirstPassError) as caught:
        _run(tmp_path, frames,
             _ScriptedFocusedClient(["valid"] * 4), **kwargs)
    assert caught.value.code == "UNCERTAIN_REQUESTS"


def test_runner_refuses_frame_set_change_on_resume(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path, 4)
    _run(tmp_path, frames,
         _ScriptedFocusedClient(["valid"] * 2))
    other = _frames(tmp_path, 4, start=10)
    with pytest.raises(FirstPassError) as caught:
        _run(tmp_path, other,
             _ScriptedFocusedClient(["valid"] * 2))
    assert caught.value.code == "CHECKPOINT_MISMATCH"


def test_runner_refuses_missing_observations(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path, 4)
    _run(tmp_path, frames,
         _ScriptedFocusedClient(["valid"] * 2))
    victim = next(
        (tmp_path / "fp" / "observations").glob("group_*.json"))
    victim.unlink()
    with pytest.raises(FirstPassError) as caught:
        _run(tmp_path, frames,
             _ScriptedFocusedClient(["valid"] * 2))
    assert caught.value.code == "CHECKPOINT_CORRUPT"


def test_runner_events_preserved_in_raw(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path, 2)
    payload = json.loads(_valid_text([0, 1]))
    payload["events"] = [{"description": "tare press",
                          "frame_indices": [0, 1]}]
    client = _ScriptedFocusedClient([json.dumps(payload)])
    report = _run(tmp_path, frames, client)
    assert report["groups_valid"] == 1
    raw = _raw(tmp_path)
    assert raw["events"] == [{"description": "tare press",
                              "frame_indices": [0, 1]}]


def test_runner_abstention_needs_no_retry(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path, 2)
    payload = json.loads(_valid_text([0, 1]))
    for entry in payload["frames"]:
        observation = entry["observation"]
        for name in observation:
            if name in ("scale_readable",
                        "tare_action_visible",
                        "container_changed_hint"):
                observation[name] = False
            elif name == "scale_state":
                observation[name] = "unknown"
            elif name == "addition_or_removal_visible":
                observation[name] = "unknown"
            elif name == "confidence":
                observation[name] = "low"
            else:
                observation[name] = None
    client = _ScriptedFocusedClient([json.dumps(payload)])
    report = _run(tmp_path, frames, client)
    assert report["requests_sent"] == 1
    assert report["groups_valid"] == 1


def test_observation_envelope_matches_downstream_contract(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path, 2)
    client = _ScriptedFocusedClient(["valid"])
    _run(tmp_path, frames, client)
    obs_path = next(
        (tmp_path / "fp" / "observations").glob("group_*.json"))
    records = json.loads(obs_path.read_text(encoding="utf-8"))
    assert len(records) == 2
    record = records[0]
    for key in ("frame_index", "image_path",
                "timestamp_seconds", "manifest_ref",
                "batch_id", "pass_id", "call", "model_fields",
                "parse_error", "response_status",
                "terminal_state", "image_ref"):
        assert key in record, key
    assert record["pass_id"] == PASS_ID_V3
    assert (sorted(record["model_fields"].keys())
            == sorted(V3_MODEL_OUTPUT_FIELDS))
    assert record["call"]["incoming_context_kind"] == "none"
    assert record["timestamp_seconds"] == 0.0


# --- Visual frame anchoring. ---

def _photo_frames(tmp_path: Path, count: int,
                  start: int = 0) -> list[FrameRef]:
    from PIL import Image

    img_dir = tmp_path / "v3photos"
    img_dir.mkdir(exist_ok=True)
    frames = []
    for offset in range(count):
        index = start + offset
        path = img_dir / f"frame_{index:08d}.jpg"
        shade = (index * 37) % 256
        Image.new("RGB", (48, 64),
                  (shade, 100, 200)).save(path, format="JPEG")
        frames.append(FrameRef(
            frame_index=index, image_path=path,
            timestamp_seconds=float(index)))
    return frames


def _valid_anchored_text(indices: list[int]) -> str:
    payload = json.loads(_valid_text(indices))
    for entry in payload["frames"]:
        entry["observation"]["visual_frame_id"] = (
            visual_frame_id_for(entry["frame_index"]))
    return json.dumps(payload)


def test_visual_ids_deterministic_and_unique() -> None:
    assert visual_frame_id_for(477) == "F000477"
    assert (visual_check_code(12)
            == visual_check_code(12))
    assert (visual_check_code(12, salt="other")
            != visual_check_code(12))
    codes = {visual_check_code(index) for index in range(645)}
    assert len(codes) == 645
    wide = {visual_check_code(index)
            for index in range(20000)}
    assert len(wide) == 20000
    for bad in (True, -1, "477", None):
        with pytest.raises(FirstPassError):
            visual_frame_id_for(bad)  # type: ignore[arg-type]


def test_anchor_preparation_preserves_photo(
        tmp_path: Path) -> None:
    frame = _photo_frames(tmp_path, 1)[0]
    before = frame.image_path.read_bytes()
    record = prepare_anchored_image(
        source=frame.image_path, frame_index=frame.frame_index,
        timestamp_seconds=frame.timestamp_seconds,
        dest_dir=tmp_path / "anchored")
    assert frame.image_path.read_bytes() == before
    assert record.visual_frame_id == "F000000"
    assert len(record.check_code) == ANCHOR_CODE_LENGTH
    assert record.source_size == (48, 64)
    assert record.prepared_size == (
        48, 64 + ANCHOR_BAND_HEIGHT)
    assert record.prepared_sha256 != record.source_sha256
    assert verify_anchored_image(record)


def test_anchor_verification_rejects_tampered_photo(
        tmp_path: Path) -> None:
    from PIL import Image

    frame = _photo_frames(tmp_path, 1)[0]
    record = prepare_anchored_image(
        source=frame.image_path, frame_index=frame.frame_index,
        timestamp_seconds=frame.timestamp_seconds,
        dest_dir=tmp_path / "anchored")
    with Image.open(record.prepared_path) as handle:
        tampered = handle.convert("RGB")
    tampered.putpixel((0, ANCHOR_BAND_HEIGHT + 1),
                      (1, 2, 3))
    tampered.save(record.prepared_path, format="PNG")
    assert not verify_anchored_image(record)


def test_anchored_schema_requires_visual_id() -> None:
    assert "visual_frame_id" in (
        FOCUSED_ANCHORED_JSON_SCHEMA["properties"]["frames"]
        ["items"]["properties"]["observation"]["properties"])
    assert "visual_frame_id" in (
        FOCUSED_ANCHORED_JSON_SCHEMA["properties"]["frames"]
        ["items"]["properties"]["observation"]["required"])
    plain = FOCUSED_JSON_SCHEMA["properties"]["frames"][
        "items"]["properties"]["observation"]
    assert "visual_frame_id" not in plain["properties"]
    assert "visual_frame_id" not in plain["required"]
    assert (set(V3_ANCHORED_OUTPUT_FIELDS)
            == set(V3_MODEL_OUTPUT_FIELDS)
            | {"visual_frame_id"})
    assert (build_focused_schema()["properties"]["frames"]
            ["items"]["properties"]["observation"]["required"]
            == list(V3_MODEL_OUTPUT_FIELDS))


def test_anchored_prompt_labels_every_image(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path, 2)
    ids = [visual_frame_id_for(frame.frame_index)
           for frame in frames]
    prompt = build_focused_prompt(
        frames=frames, group_sequence=0, total_groups=1,
        incoming_context=None, visual_ids=ids)
    assert "visual_frame_id" in prompt
    assert "F000000" in prompt and "F000001" in prompt
    assert "never assign one image's reading" in prompt
    with pytest.raises(FirstPassError):
        build_focused_prompt(
            frames=frames, group_sequence=0, total_groups=1,
            incoming_context=None, visual_ids=["F000000"])
    plain = build_focused_prompt(
        frames=frames, group_sequence=0, total_groups=1,
        incoming_context=None)
    assert "visual_frame_id" not in plain


def test_anchored_validation_accepts_and_rejects() -> None:
    ids = ["F000000", "F000001"]
    expected = [(0, 0.0), (1, 1.0)]
    assert validate_focused_response(
        _valid_anchored_text([0, 1]), expected,
        visual_ids=ids).ok
    assert not validate_focused_response(
        _valid_text([0, 1]), expected, visual_ids=ids).ok
    swapped = json.loads(_valid_anchored_text([0, 1]))
    first = swapped["frames"][0]["observation"]
    second = swapped["frames"][1]["observation"]
    first["visual_frame_id"], second["visual_frame_id"] = (
        second["visual_frame_id"], first["visual_frame_id"])
    result = validate_focused_response(
        json.dumps(swapped), expected, visual_ids=ids)
    assert not result.ok
    assert result.error is not None
    assert result.error.startswith("visual_id_mismatch")
    with pytest.raises(FirstPassError):
        validate_focused_response(
            _valid_anchored_text([0, 1]), expected,
            visual_ids=["F000000"])


def test_anchored_dedupe_and_conflicts() -> None:
    ids = ["F000000", "F000001"]
    expected = [(0, 0.0), (1, 1.0)]
    payload = json.loads(_valid_anchored_text([0, 1]))
    payload["frames"].append(payload["frames"][-1])
    replay = validate_focused_response(
        json.dumps(payload), expected, visual_ids=ids)
    assert replay.ok and replay.deduped == 1
    payload = json.loads(_valid_anchored_text([0, 1]))
    repeat = json.loads(json.dumps(payload["frames"][-1]))
    repeat["visual_frame_id"] = "F000001"
    repeat["observation"]["visual_frame_id"] = "F000000"
    payload["frames"].append(repeat)
    conflict = validate_focused_response(
        json.dumps(payload), expected, visual_ids=ids)
    assert not conflict.ok
    assert conflict.error == (
        "duplicate_frame_conflict:frame_1")
    payload = json.loads(_valid_anchored_text([0, 1]))
    payload["frames"][1]["frame_index"] = 0
    payload["frames"][1]["observation"][
        "visual_frame_id"] = "F000000"
    clash = validate_focused_response(
        json.dumps(payload), expected, visual_ids=ids)
    assert not clash.ok


def test_runner_anchored_batches_and_manifest(
        tmp_path: Path) -> None:
    frames = _photo_frames(tmp_path, 55)
    client = DeterministicFocusedFakeClient(visual_anchor=True)
    report = _run(tmp_path, frames, client,
                  visual_anchor=True, group_size=50,
                  max_validation_attempts=1)
    assert report["groups_valid"] == 2
    assert report["frames_observed"] == 55
    manifest_path = (tmp_path / "fp" / "anchored"
                     / "manifest.json")
    manifest = json.loads(manifest_path.read_text(
        encoding="utf-8"))
    assert len(manifest) == 55
    assert manifest[0]["visual_frame_id"] == "F000000"
    assert manifest[0]["photo_region_verified"] is True
    assert (manifest[0]["prepared_size"][1]
            == 64 + ANCHOR_BAND_HEIGHT)
    raw = _raw(tmp_path)
    assert raw["visual_anchor"] is True
    assert raw["image_order"][0].startswith("anchored_")
    assert client.calls[0]["schema"] == (
        "FocusedFrameBatchAnchored")
    obs_path = next(
        (tmp_path / "fp" / "observations").glob(
            "group_0000.json"))
    records = json.loads(obs_path.read_text(encoding="utf-8"))
    assert (records[0]["model_fields"]["visual_frame_id"]
            == "F000000")
    assert "prepared_sha256" in records[0]["image_ref"]
    assert "sha256" in records[0]["image_ref"]
    assert len(raw["prompt_sha256"]) == 64
    assert set(records[0]) == {
        "frame_index", "image_path", "timestamp_seconds",
        "source_timestamp_seconds",
        "requested_timestamp_seconds", "timestamp_basis",
        "manifest_ref", "batch_id", "pass_id", "call",
        "model_fields", "model_timestamp_seconds",
        "parse_error", "raw_text_excerpt",
        "response_status", "incomplete_reason",
        "terminal_state", "image_ref"}


def test_runner_anchored_25_and_partial_final(
        tmp_path: Path) -> None:
    frames = _photo_frames(tmp_path, 30)
    client = DeterministicFocusedFakeClient(visual_anchor=True)
    report = _run(tmp_path, frames, client,
                  visual_anchor=True, group_size=25,
                  max_validation_attempts=1)
    assert report["groups_valid"] == 2
    assert report["frames_observed"] == 30
    obs = json.loads(next(
        (tmp_path / "fp" / "observations").glob(
            "group_0001.json")).read_text(encoding="utf-8"))
    assert len(obs) == 5
    assert (obs[0]["model_fields"]["visual_frame_id"]
            == visual_frame_id_for(25))


def test_runner_anchored_context_unchanged(
        tmp_path: Path) -> None:
    frames = _photo_frames(tmp_path, 3)
    client = _ScriptedFocusedClient(
        [_valid_anchored_text([0, 1]),
         _valid_anchored_text([2])])
    report = _run(tmp_path, frames, client,
                  visual_anchor=True, group_size=2,
                  max_validation_attempts=1)
    assert report["groups_valid"] == 2
    assert "no earlier observations" in client.prompts[0]
    assert "PREVIOUS BATCH" in client.prompts[1]
    assert "F000002" in client.prompts[1]


def test_runner_single_attempt_sends_once(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path, 2)
    client = _ScriptedFocusedClient(["not json {"])
    report = _run(tmp_path, frames, client,
                  max_validation_attempts=1)
    assert report["requests_sent"] == 1
    assert report["groups_valid"] == 0
    assert report["groups_review"] == 1
    assert len(client.calls) == 1
