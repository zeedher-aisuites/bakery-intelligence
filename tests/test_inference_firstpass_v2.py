"""Offline tests for sequential contextual FirstPass V2.

A scripted fake group client stands in for the provider seam:
no network, no credentials, no paid calls. Covers grouping,
one-request-per-group dispatch, rolling-context propagation,
response validation (completeness, ordering, JSON shape),
checkpoint/resume without re-submit, budget caps, billable
config guards, and the provider image-limit failure path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from bakery_intelligence.inference.firstpass import (
    MODEL_OUTPUT_FIELDS,
    FirstPassError,
    FrameRef,
    InferenceError,
    InferenceResponse,
    InferenceUsage,
    SourceRef,
)
from bakery_intelligence.inference.firstpass_v2 import (
    CONTEXT_MAX_CHARS,
    DEFAULT_GROUP_SIZE,
    GROUP_JSON_SCHEMA,
    PASS_ID_V2,
    DeterministicGroupFakeClient,
    FirstPassV2Config,
    FirstPassV2Runner,
    bound_context,
    build_group_prompt,
    estimate_context_tokens,
    group_frames,
    validate_group_response,
)
from bakery_intelligence.inference.meta_spark import (
    MAX_IMAGES_PER_REQUEST,
)

FAKE_JPEG = b"\xff\xd8" + b"0" * 64 + b"\xff\xd9"
SOURCE = SourceRef(name="test-source", sha256="abc123")


def _frame(frames_dir: Path, index: int,
           timestamp: float) -> FrameRef:
    frames_dir.mkdir(parents=True, exist_ok=True)
    path = frames_dir / f"frame_{index:08d}.jpg"
    path.write_bytes(FAKE_JPEG)
    return FrameRef(frame_index=index, image_path=path,
                    timestamp_seconds=timestamp)


def _frames(frames_dir: Path, count: int,
            start: int = 0) -> list[FrameRef]:
    return [_frame(frames_dir, start + position,
                   float(start + position))
            for position in range(count)]


def _observation(index: int) -> dict[str, Any]:
    return {
        "visible_objects": ["scale"],
        "scale_display_verbatim": str(100 + index),
        "scale_readable": True,
        "scale_state": "stable",
        "tare_action_visible": False,
        "addition_or_removal_visible": "none",
        "container_description": "bowl",
        "container_changed_hint": False,
        "confidence": "high",
        "ambiguity_notes": None,
    }


def _group_payload(indices: Sequence[int],
                   context: str = "rolled context") -> str:
    return json.dumps({
        "frames": [{
            "frame_index": index,
            "observation": _observation(index),
            "verified": [f"frame {index} visible"],
            "uncertain": [],
            "hypotheses": [],
        } for index in indices],
        "rolling_context": context,
        "context_notes": None,
    })


def _ok(text: str, **overrides: Any) -> InferenceResponse:
    payload: dict[str, Any] = {
        "model": "fake-group-model", "status": "completed",
        "output_text": text,
        "usage": InferenceUsage(500, 200, 700),
        "latency_seconds": 0.25, "request_id": "req-group-1",
        "incomplete_reason": None,
    }
    payload.update(overrides)
    return InferenceResponse(**payload)


class _ScriptedGroupClient:
    """Group double keyed by group sequence; records calls."""

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    def submit(self, prompt_text: str,
               image_paths: Sequence[Path], api_key: str,
               max_output_tokens: int,
               response_schema: Any = None,
               response_schema_name: Any = None
               ) -> InferenceResponse:
        self.calls.append({
            "prompt_text": prompt_text,
            "image_paths": [Path(path).name
                            for path in image_paths],
            "api_key": api_key,
            "max_output_tokens": max_output_tokens,
            "response_schema": response_schema,
            "response_schema_name": response_schema_name,
        })
        action = self._script.pop(0)
        if isinstance(action, Exception):
            raise action
        assert action is not None
        return action


def _config(frames: list[FrameRef], out_dir: Path,
            client: Any, **overrides: Any) -> FirstPassV2Config:
    payload: dict[str, Any] = {
        "frames": frames, "source": SOURCE, "out_dir": out_dir,
        "client_factory": lambda: client,
        "sleeper": lambda seconds: None,
    }
    payload.update(overrides)
    return FirstPassV2Config(**payload)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# Grouping and prompts.


def test_group_frames_fixed_ten_with_partial_tail() -> None:
    assert len(group_frames(list(range(25)), 10)) == 3
    groups = group_frames(list(range(25)), 10)
    assert [len(group) for group in groups] == [10, 10, 5]
    assert groups[0] == list(range(10))
    assert groups[2] == list(range(20, 25))


def test_group_frames_rejects_empty_size() -> None:
    with pytest.raises(FirstPassError, match="group_size"):
        group_frames([1, 2], 0)


def test_group_prompt_lists_frames_in_order_with_context() -> None:
    frames = [FrameRef(frame_index=index,
                       image_path=Path(f"frame_{index:08d}.jpg"),
                       timestamp_seconds=float(index))
              for index in (4, 5)]
    prompt = build_group_prompt(
        frames=frames, group_sequence=1, total_groups=3,
        incoming_context="prior state", context_kind="rolling")
    first = prompt.index("frame_index=4")
    second = prompt.index("frame_index=5")
    assert first < second
    assert "GROUP 2 OF 3" in prompt
    assert "INCOMING ROLLING CONTEXT" in prompt
    assert "prior state" in prompt


def test_group_prompt_marks_missing_context() -> None:
    frames = [FrameRef(frame_index=0,
                       image_path=Path("frame_00000000.jpg"),
                       timestamp_seconds=0.0)]
    prompt = build_group_prompt(
        frames=frames, group_sequence=0, total_groups=1,
        incoming_context=None, context_kind="none")
    assert "INCOMING CONTEXT: none" in prompt


def test_group_schema_requires_per_frame_observations() -> None:
    assert GROUP_JSON_SCHEMA["required"] == [
        "frames", "rolling_context"]
    item = GROUP_JSON_SCHEMA["properties"]["frames"]["items"]
    assert item["required"] == [
        "frame_index", "observation", "verified",
        "uncertain", "hypotheses"]
    assert (sorted(item["properties"]["observation"]["required"])
            == sorted(MODEL_OUTPUT_FIELDS))


def test_context_bound_truncates_with_marker() -> None:
    long = "x" * (CONTEXT_MAX_CHARS + 500)
    bounded = bound_context(long)
    assert len(bounded) <= CONTEXT_MAX_CHARS
    assert bounded.endswith("[context truncated to bound]")
    assert estimate_context_tokens("abcd") == 1


# Validation.


def test_validate_accepts_well_formed_group() -> None:
    result = validate_group_response(
        _group_payload([0, 1], context="next state"), [0, 1])
    assert result.ok
    assert len(result.entries) == 2
    assert result.rolling_context == "next state"


@pytest.mark.parametrize("raw_text,expected,fragment", [
    ("not json", [0], "unparseable_json"),
    (json.dumps({"frames": "nope", "rolling_context": "c"}),
     [0], "frames_not_a_list"),
    (_group_payload([0]), [0, 1], "frame_count_mismatch"),
    (_group_payload([1]), [0], "frame_order_mismatch"),
    (json.dumps({"frames": [{"frame_index": 0,
                             "observation": {"scale_readable": True},
                             "verified": [], "uncertain": [],
                             "hypotheses": []}],
                 "rolling_context": "c"}),
     [0], "observation_missing_keys"),
    (json.dumps({"frames": [{"frame_index": 0,
                             "observation": _observation(0),
                             "verified": "nope", "uncertain": [],
                             "hypotheses": []}],
                 "rolling_context": "c"}),
     [0], "evidence_tier_malformed"),
    (_group_payload([0], context="  "), [0],
     "rolling_context_missing"),
])
def test_validate_rejects_malformed_groups(
        raw_text: str, expected: list[int],
        fragment: str) -> None:
    result = validate_group_response(raw_text, expected)
    assert not result.ok
    assert fragment in (result.error or "")


# Runner dispatch and context propagation.


def test_one_request_per_group_with_ordered_images(
        tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames = _frames(frames_dir, 25)
    client = _ScriptedGroupClient([
        _ok(_group_payload(range(0, 10), context="ctx-0")),
        _ok(_group_payload(range(10, 20), context="ctx-1")),
        _ok(_group_payload(range(20, 25), context="ctx-2")),
    ])
    report = FirstPassV2Runner(_config(
        frames, tmp_path / "v2", client)).run()
    assert report["requests_sent"] == 3
    assert report["frames_observed"] == 25
    assert report["output_completeness"] == 1.0
    assert [len(call["image_paths"]) for call in client.calls] == [
        10, 10, 5]
    assert client.calls[0]["image_paths"][0] == "frame_00000000.jpg"
    assert client.calls[1]["image_paths"][0] == "frame_00000010.jpg"
    assert client.calls[2]["image_paths"][-1] == "frame_00000024.jpg"
    for call in client.calls:
        assert (call["response_schema_name"]
                == "FrameGroupObservation")
        assert call["response_schema"] is GROUP_JSON_SCHEMA


def test_rolling_context_propagates_across_groups(
        tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames = _frames(frames_dir, 20)
    client = _ScriptedGroupClient([
        _ok(_group_payload(range(0, 10), context="STATE-ALPHA")),
        _ok(_group_payload(range(10, 20), context="STATE-BETA")),
    ])
    report = FirstPassV2Runner(_config(
        frames, tmp_path / "v2", client,
        global_context="GLOBAL-VIDEO-STATE")).run()
    assert report["groups_valid"] == 2
    assert "GLOBAL-VIDEO-STATE" in client.calls[0]["prompt_text"]
    assert "STATE-ALPHA" in client.calls[1]["prompt_text"]
    assert "GLOBAL-VIDEO-STATE" not in client.calls[1]["prompt_text"]
    context_file = (tmp_path / "v2" / "contexts"
                    / "group_0000.md")
    assert "STATE-ALPHA" in context_file.read_text(encoding="utf-8")


def test_observations_keep_v1_compatible_shape(
        tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames = _frames(frames_dir, 10)
    client = _ScriptedGroupClient(
        [_ok(_group_payload(range(10), context="ctx"))])
    FirstPassV2Runner(_config(
        frames, tmp_path / "v2", client)).run()
    records = _read_json(
        tmp_path / "v2" / "observations" / "group_0000.json")
    assert len(records) == 10
    first = records[0]
    assert first["frame_index"] == 0
    assert first["pass_id"] == PASS_ID_V2
    assert first["batch_id"] == "group_0000"
    assert (sorted(first["model_fields"]) ==
            sorted(MODEL_OUTPUT_FIELDS))
    assert first["model_fields"]["scale_display_verbatim"] == "100"
    assert first["evidence_tiers"]["verified"] == [
        "frame 0 visible"]
    assert first["call"]["group_sequence"] == 0
    assert first["call"]["image_position"] == 1
    assert first["call"]["incoming_context_kind"] == "none"


def test_invalid_group_marks_review_and_carries_gap(
        tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames = _frames(frames_dir, 20)
    client = _ScriptedGroupClient([
        _ok("not json"),
        _ok(_group_payload(range(10, 20), context="after-gap")),
    ])
    report = FirstPassV2Runner(_config(
        frames, tmp_path / "v2", client)).run()
    assert report["groups_valid"] == 1
    assert report["groups_review"] == 1
    assert report["frames_observed"] == 10
    assert "[gap: group_0000" in client.calls[1]["prompt_text"]
    assert "do not infer continuity" in client.calls[1][
        "prompt_text"].lower()
    assert report["groups"][0]["terminal_state"] == "invalid_json"


def test_checkpoint_resume_avoids_resubmission(
        tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames = _frames(frames_dir, 20)
    out_dir = tmp_path / "v2"
    first = _ScriptedGroupClient(
        [_ok(_group_payload(range(0, 10), context="CTX-0"))])
    partial = FirstPassV2Runner(_config(
        frames, out_dir, first, max_groups=1)).run()
    assert partial["requests_sent"] == 1
    assert partial["stopped_reason"] == "request_limit_reached"
    second = _ScriptedGroupClient(
        [_ok(_group_payload(range(10, 20), context="CTX-1"))])
    resumed = FirstPassV2Runner(_config(
        frames, out_dir, second)).run()
    assert resumed["requests_sent"] == 1
    assert resumed["frames_observed"] == 10
    assert resumed["frames_resumed"] == 10
    assert resumed["groups_resumed"] == 1
    assert resumed["output_completeness"] == 1.0
    assert len(second.calls) == 1
    assert "CTX-0" in second.calls[0]["prompt_text"]
    assert resumed["budget"]["prior_requests_submitted"] == 1


def test_billable_config_requires_single_submit_guards(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path / "frames", 10)
    client = DeterministicGroupFakeClient()
    with pytest.raises(FirstPassError, match="max_attempts"):
        FirstPassV2Runner(_config(
            frames, tmp_path / "v2", client, billable=True,
            max_attempts=2, uncertain_policy="refuse")).run()
    with pytest.raises(FirstPassError, match="uncertain_policy"):
        FirstPassV2Runner(_config(
            frames, tmp_path / "v2", client, billable=True,
            uncertain_policy="rerun")).run()


def test_group_size_above_client_cap_fails_before_dispatch(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path / "frames", 5)
    client = DeterministicGroupFakeClient()
    with pytest.raises(FirstPassError) as excinfo:
        FirstPassV2Runner(_config(
            frames, tmp_path / "v2", client,
            group_size=MAX_IMAGES_PER_REQUEST + 1)).run()
    assert excinfo.value.code == "TOO_MANY_IMAGES"
    assert client.calls == []


def test_provider_image_rejection_fails_clearly(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path / "frames", 20)
    client = _ScriptedGroupClient([
        InferenceError("PAYLOAD_TOO_LARGE", "too many images"),
    ])
    with pytest.raises(FirstPassError) as excinfo:
        FirstPassV2Runner(_config(
            frames, tmp_path / "v2", client)).run()
    assert excinfo.value.code == "PROVIDER_IMAGE_LIMIT"


def test_empty_output_on_token_exhaustion_is_exhausted(
        tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames = _frames(frames_dir, 10)
    client = _ScriptedGroupClient([
        _ok("", status="incomplete",
            incomplete_reason="max_output_tokens"),
    ])
    report = FirstPassV2Runner(_config(
        frames, tmp_path / "v2", client)).run()
    assert report["groups"][0]["terminal_state"] == (
        "exhausted_token_budget")


def test_empty_output_without_token_reason_stays_empty(
        tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames = _frames(frames_dir, 10)
    client = _ScriptedGroupClient([
        _ok("   ", status="completed",
            incomplete_reason=None),
    ])
    report = FirstPassV2Runner(_config(
        frames, tmp_path / "v2", client)).run()
    assert report["groups"][0]["terminal_state"] == "empty_output"


def test_deterministic_fake_client_chains_offline(
        tmp_path: Path) -> None:
    frames = _frames(tmp_path / "frames", 30)
    fake = DeterministicGroupFakeClient()
    report = FirstPassV2Runner(_config(
        frames, tmp_path / "v2", fake,
        group_size=DEFAULT_GROUP_SIZE)).run()
    assert report["requests_sent"] == 3
    assert report["frames_observed"] == 30
    assert len(fake.calls) == 3
    assert fake.calls[0]["frame_indices"] == list(range(10))
    assert "Fake context after frames 0-9" in fake.calls[1][
        "prompt_text"]
    assert "Fake context after frames 10-19" in fake.calls[2][
        "prompt_text"]
