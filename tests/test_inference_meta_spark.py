"""Offline tests for the Meta Muse Spark adapter (M2-06C.5/S4-A).

A recording fake stands in for the OpenAI SDK seam: no
network, no credentials, no paid calls. Covers request
serialization, MIME handling, response parsing, usage
accounting, error mapping, the S3 billable-config factory,
and proof that no hidden retries exist.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bakery_intelligence.inference.firstpass import (
    MODEL_OUTPUT_FIELDS,
    FirstPassError,
    FirstPassRunner,
    FrameRef,
    InferenceError,
    SourceRef,
)
from bakery_intelligence.inference.meta_spark import (
    BASE_URL,
    MODEL_SPARK,
    OBSERVATION_JSON_SCHEMA,
    OBSERVATION_SCHEMA_NAME,
    MetaSparkClient,
    billable_runner_config,
)

PROMPT = "TEST PROMPT"
SOURCE = SourceRef(name="test-source", sha256="abc123")
# Framed like the proven first-pass fixture: JPEG start marker,
# filler, and the FFD9 end marker the runner requires.
JPEG_BYTES = b"\xff\xd8\xff" + b"0" * 64 + b"\xff\xd9"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 64
GIF_BYTES = b"GIF89a" + b"0" * 64

FULL_USAGE = {
    "input_tokens": 2913,
    "output_tokens": 1240,
    "total_tokens": 4153,
    "input_tokens_details": {"cached_tokens": 128},
    "output_tokens_details": {"reasoning_tokens": 542},
}


class APITimeoutError(Exception):
    """Fake SDK timeout (mirrors the mapped exception name)."""


class APIConnectionError(Exception):
    """Fake SDK connection failure."""


class APIStatusError(Exception):
    """Fake SDK HTTP error with status and headers."""

    def __init__(self, message: str, *, status_code: Any,
                 headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(headers=headers or {})


class _FakeResponses:
    """Recording stand-in for client.responses."""

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        action = self._script.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


class _FakeSDK:
    """Recording stand-in for the OpenAI SDK client."""

    def __init__(self, script: list[Any]) -> None:
        self.responses = _FakeResponses(script)


class _UsageWithDump:
    """Usage object exposing model_dump like the real SDK."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def model_dump(self) -> dict[str, Any]:
        return dict(self._payload)


def _sdk_response(**overrides: Any) -> SimpleNamespace:
    payload: dict[str, Any] = {
        "status": "completed",
        "output_text": '{"scale_readable": true}',
        "usage": dict(FULL_USAGE),
        "model": MODEL_SPARK,
        "id": "resp_test123",
        "incomplete_details": None,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


def _image(path: Path, blob: bytes = JPEG_BYTES) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return path


def _client(script: list[Any], **overrides: Any) -> MetaSparkClient:
    made: list[_FakeSDK] = []

    def factory(api_key: str, timeout: float) -> _FakeSDK:
        _ = (api_key, timeout)
        sdk = _FakeSDK(script)
        made.append(sdk)
        return sdk

    client = MetaSparkClient(client_factory=factory, **overrides)
    client._made = made  # type: ignore[attr-defined]
    return client


def _sdk_of(client: MetaSparkClient) -> _FakeSDK:
    return client._made[0]  # type: ignore[attr-defined]


def test_request_serialization_matches_verified_shape(
        tmp_path: Path) -> None:
    image = _image(tmp_path / "frame.jpg")
    client = _client([_sdk_response()])
    client.submit(PROMPT, [image], "key", 1024)
    (call,) = _sdk_of(client).responses.calls
    assert call["model"] == MODEL_SPARK
    assert call["max_output_tokens"] == 1024
    assert call["store"] is True
    assert call["input"] == [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": PROMPT},
            {"type": "input_image",
             "image_url": "data:image/jpeg;base64,"
                          + base64.b64encode(JPEG_BYTES).decode("ascii")},
        ],
    }]


def test_png_mime_matches_payload(tmp_path: Path) -> None:
    image = _image(tmp_path / "frame.png", PNG_BYTES)
    client = _client([_sdk_response()])
    client.submit(PROMPT, [image], "key", 1024)
    (call,) = _sdk_of(client).responses.calls
    blocks = call["input"][0]["content"]
    assert blocks[1] == {
        "type": "input_image",
        "image_url": "data:image/png;base64,"
                     + base64.b64encode(PNG_BYTES).decode("ascii"),
    }


@pytest.mark.parametrize("blob", [GIF_BYTES, b""])
def test_unsupported_image_refused(
        tmp_path: Path, blob: bytes) -> None:
    image = _image(tmp_path / "frame.jpg", blob)
    sdk = _FakeSDK([_sdk_response()])
    client = MetaSparkClient(
        client_factory=lambda key, timeout: sdk)
    with pytest.raises(InferenceError) as caught:
        client.submit(PROMPT, [image], "key", 1024)
    assert caught.value.code == "UNSUPPORTED_IMAGE"
    assert sdk.responses.calls == []


def test_empty_request_rejected(tmp_path: Path) -> None:
    sdk = _FakeSDK([_sdk_response()])
    client = MetaSparkClient(
        client_factory=lambda key, timeout: sdk)
    with pytest.raises(InferenceError) as caught:
        client.submit(PROMPT, [], "key", 1024)
    assert caught.value.code == "EMPTY_REQUEST"
    assert sdk.responses.calls == []


def test_too_many_images_rejected(tmp_path: Path) -> None:
    paths = [tmp_path / f"frame_{index}.jpg" for index in range(51)]
    sdk = _FakeSDK([_sdk_response()])
    client = MetaSparkClient(
        client_factory=lambda key, timeout: sdk)
    with pytest.raises(InferenceError) as caught:
        client.submit(PROMPT, paths, "key", 1024)
    assert caught.value.code == "TOO_MANY_IMAGES"
    assert sdk.responses.calls == []


def test_missing_image_refused(tmp_path: Path) -> None:
    sdk = _FakeSDK([_sdk_response()])
    client = MetaSparkClient(
        client_factory=lambda key, timeout: sdk)
    with pytest.raises(InferenceError) as caught:
        client.submit(
            PROMPT, [tmp_path / "absent.jpg"], "key", 1024)
    assert caught.value.code == "IMAGE_MISSING"
    assert sdk.responses.calls == []


@pytest.mark.parametrize("budget", [0, -1, True, "1024", None])
def test_invalid_budget_rejected(
        tmp_path: Path, budget: Any) -> None:
    image = _image(tmp_path / "frame.jpg")
    sdk = _FakeSDK([_sdk_response()])
    client = MetaSparkClient(
        client_factory=lambda key, timeout: sdk)
    with pytest.raises(InferenceError) as caught:
        client.submit(PROMPT, [image], "key", budget)
    assert caught.value.code == "INVALID_REQUEST"
    assert sdk.responses.calls == []


def test_response_normalized_with_usage(tmp_path: Path) -> None:
    image = _image(tmp_path / "frame.jpg")
    ticks = iter([10.0, 12.5])
    client = _client([_sdk_response()], clock=lambda: next(ticks))
    result = client.submit(PROMPT, [image], "key", 1024)
    assert result.model == MODEL_SPARK
    assert result.status == "completed"
    assert result.output_text == '{"scale_readable": true}'
    assert result.usage.input_tokens == 2913
    assert result.usage.output_tokens == 1240
    assert result.usage.total_tokens == 4153
    assert result.usage.cached_input_tokens == 128
    assert result.usage.reasoning_output_tokens == 542
    assert result.latency_seconds == pytest.approx(2.5)
    assert result.request_id == "resp_test123"
    assert result.incomplete_reason is None


def test_model_dump_usage_supported(tmp_path: Path) -> None:
    image = _image(tmp_path / "frame.jpg")
    response = _sdk_response(usage=_UsageWithDump(dict(FULL_USAGE)))
    client = _client([response])
    result = client.submit(PROMPT, [image], "key", 1024)
    assert result.usage.input_tokens == 2913
    assert result.usage.reasoning_output_tokens == 542


@pytest.mark.parametrize("usage", [None, {}])
def test_missing_usage_stays_unknown(
        tmp_path: Path, usage: Any) -> None:
    image = _image(tmp_path / "frame.jpg")
    client = _client([_sdk_response(usage=usage)])
    result = client.submit(PROMPT, [image], "key", 1024)
    assert result.usage.input_tokens is None
    assert result.usage.output_tokens is None
    assert result.usage.total_tokens is None
    assert result.usage.cached_input_tokens is None
    assert result.usage.reasoning_output_tokens is None


def test_non_int_usage_degrades_to_unknown(tmp_path: Path) -> None:
    image = _image(tmp_path / "frame.jpg")
    usage = {
        "input_tokens": "2913",
        "output_tokens": True,
        "total_tokens": 41.53,
        "input_tokens_details": {"cached_tokens": "128"},
        "output_tokens_details": ["not-a-dict"],
    }
    client = _client([_sdk_response(usage=usage)])
    result = client.submit(PROMPT, [image], "key", 1024)
    assert result.usage.input_tokens is None
    assert result.usage.output_tokens is None
    assert result.usage.total_tokens is None
    assert result.usage.cached_input_tokens is None
    assert result.usage.reasoning_output_tokens is None


@pytest.mark.parametrize("details", [
    {"reason": "max_output_tokens"},
    SimpleNamespace(reason="max_output_tokens"),
])
def test_incomplete_reason_preserved(
        tmp_path: Path, details: Any) -> None:
    image = _image(tmp_path / "frame.jpg")
    response = _sdk_response(status="incomplete",
                             output_text="",
                             incomplete_details=details)
    client = _client([response])
    result = client.submit(PROMPT, [image], "key", 1024)
    assert result.status == "incomplete"
    assert result.incomplete_reason == "max_output_tokens"


def test_incomplete_response_keeps_usage_and_identity(
        tmp_path: Path) -> None:
    """Boundary contract the runner's failure evidence relies
    on: partial text, full usage, model echo, and request id
    survive normalization on an incomplete response."""
    image = _image(tmp_path / "frame.jpg")
    response = _sdk_response(
        status="incomplete",
        output_text='{"visible_objects": ["scale"], ',
        incomplete_details={"reason": "max_output_tokens"},
        id="resp_inc_7",
    )
    client = _client([response])
    result = client.submit(PROMPT, [image], "key", 512)
    assert result.status == "incomplete"
    assert result.incomplete_reason == "max_output_tokens"
    assert result.output_text == (
        '{"visible_objects": ["scale"], ')
    assert result.model == MODEL_SPARK
    assert result.request_id == "resp_inc_7"
    assert result.latency_seconds >= 0.0
    assert result.usage.input_tokens == 2913
    assert result.usage.output_tokens == 1240
    assert result.usage.total_tokens == 4153
    assert result.usage.cached_input_tokens == 128
    assert result.usage.reasoning_output_tokens == 542


@pytest.mark.parametrize("field,value", [
    ("status", None), ("status", 200),
    ("output_text", None), ("output_text", 42),
])
def test_malformed_response_rejected(
        tmp_path: Path, field: str, value: Any) -> None:
    image = _image(tmp_path / "frame.jpg")
    client = _client([_sdk_response(**{field: value})])
    with pytest.raises(InferenceError) as caught:
        client.submit(PROMPT, [image], "key", 1024)
    assert caught.value.code == "RESPONSE_MALFORMED"


@pytest.mark.parametrize("echo", [None, ""])
def test_model_echo_fallback(tmp_path: Path, echo: Any) -> None:
    image = _image(tmp_path / "frame.jpg")
    client = _client([_sdk_response(model=echo)])
    result = client.submit(PROMPT, [image], "key", 1024)
    assert result.model == MODEL_SPARK


def _status_cases() -> list[tuple[Any, str, float | None]]:
    return [
        (APITimeoutError("slow"), "TIMEOUT", None),
        (APIConnectionError("down"), "NETWORK_ERROR", None),
        (APIStatusError("denied", status_code=401), "AUTH_FAILED", None),
        (APIStatusError("busy", status_code=429,
                        headers={"retry-after": "7.5"}),
         "RATE_LIMITED", 7.5),
        (APIStatusError("busy", status_code=429,
                        headers={"retry-after": "soon"}),
         "RATE_LIMITED", None),
        (APIStatusError("bad", status_code=400), "BAD_REQUEST", None),
        (APIStatusError("big", status_code=413),
         "PAYLOAD_TOO_LARGE", None),
        (APIStatusError("oops", status_code=500),
         "SERVER_ERROR", None),
        (APIStatusError("oops", status_code=503,
                        headers={"retry-after": "2"}),
         "SERVER_ERROR", 2.0),
        (APIStatusError("pay", status_code=402),
         "PROVIDER_ERROR", None),
        (APIStatusError("odd", status_code="weird"),
         "PROVIDER_ERROR", None),
        (RuntimeError("boom"), "PROVIDER_ERROR", None),
    ]


@pytest.mark.parametrize("exc,code,retry_after", _status_cases())
def test_exception_mapping(tmp_path: Path, exc: Exception,
                           code: str,
                           retry_after: float | None) -> None:
    image = _image(tmp_path / "frame.jpg")
    client = _client([exc])
    with pytest.raises(InferenceError) as caught:
        client.submit(PROMPT, [image], "key", 1024)
    assert caught.value.code == code
    assert caught.value.retry_after_seconds == retry_after
    assert len(_sdk_of(client).responses.calls) == 1


def test_timeout_name_variant_mapped(tmp_path: Path) -> None:
    class ReadTimeout(Exception):
        """SDK-style timeout without the exact class name."""

    image = _image(tmp_path / "frame.jpg")
    client = _client([ReadTimeout("slow")])
    with pytest.raises(InferenceError) as caught:
        client.submit(PROMPT, [image], "key", 1024)
    assert caught.value.code == "TIMEOUT"


def test_key_never_in_error_text(tmp_path: Path) -> None:
    image = _image(tmp_path / "frame.jpg")
    client = _client([RuntimeError("boom")])
    with pytest.raises(InferenceError) as caught:
        client.submit(PROMPT, [image], "sk-secret-123", 1024)
    assert "sk-secret-123" not in caught.value.message


def test_client_factory_lazy_and_cached(tmp_path: Path) -> None:
    image = _image(tmp_path / "frame.jpg")
    built: list[str] = []

    def factory(api_key: str, timeout: float) -> _FakeSDK:
        built.append(api_key)
        return _FakeSDK([_sdk_response(), _sdk_response()])

    client = MetaSparkClient(client_factory=factory)
    assert built == []
    client.submit(PROMPT, [image], "key", 1024)
    client.submit(PROMPT, [image], "key", 1024)
    assert built == ["key"]


def test_model_id_property() -> None:
    client = MetaSparkClient(
        client_factory=lambda key, timeout: _FakeSDK([]))
    assert client.model_id == MODEL_SPARK
    assert MetaSparkClient(
        model="other-model",
        client_factory=lambda key, timeout: _FakeSDK([]),
    ).model_id == "other-model"


def test_default_factory_wires_sdk_without_io() -> None:
    from bakery_intelligence.inference.meta_spark import (
        _default_client_factory,
    )

    built = _default_client_factory("sk-test", 45.0)
    assert str(built.base_url).rstrip("/") == BASE_URL
    assert built.api_key == "sk-test"
    assert built.max_retries == 0


def test_billable_factory_builds_safe_config(
        tmp_path: Path) -> None:
    frames = [FrameRef(frame_index=1,
                       image_path=tmp_path / "frame.jpg",
                       timestamp_seconds=0.1)]
    made: list[MetaSparkClient] = []

    def factory() -> MetaSparkClient:
        client = MetaSparkClient(
            client_factory=lambda key, timeout: _FakeSDK([]))
        made.append(client)
        return client

    config = billable_runner_config(
        frames=frames, source=SOURCE, out_dir=tmp_path / "run",
        prompt_text=PROMPT, client_factory=factory)
    assert config.billable is True
    assert config.max_attempts == 1
    assert config.token_budgets == (1024,)
    assert config.uncertain_policy == "refuse"
    assert config.provider == "meta"
    assert config.model == MODEL_SPARK
    assert made == []


@pytest.mark.parametrize("budget", [0, -5, True, "1024", None])
def test_billable_factory_rejects_bad_budget(
        tmp_path: Path, budget: Any) -> None:
    with pytest.raises(FirstPassError) as caught:
        billable_runner_config(
            frames=[], source=SOURCE, out_dir=tmp_path,
            prompt_text=PROMPT,
            client_factory=lambda: MetaSparkClient(
                client_factory=lambda key, timeout: _FakeSDK([])),
            token_budget=budget)
    assert caught.value.code == "INVALID_CONFIG"


def test_billable_factory_passes_pass_and_batch(
        tmp_path: Path) -> None:
    config = billable_runner_config(
        frames=[], source=SOURCE, out_dir=tmp_path,
        prompt_text=PROMPT,
        client_factory=lambda: MetaSparkClient(
            client_factory=lambda key, timeout: _FakeSDK([])),
        pass_id="pipeline_firstpass", batch_size=5)
    assert config.pass_id == "pipeline_firstpass"
    assert config.batch_size == 5
    assert config.billable is True
    assert config.max_attempts == 1


@pytest.mark.parametrize("field,value", [
    ("pass_id", ""), ("pass_id", None), ("pass_id", 7),
    ("batch_size", 0), ("batch_size", -2),
    ("batch_size", True), ("batch_size", "5"),
])
def test_billable_factory_rejects_bad_pass_or_batch(
        tmp_path: Path, field: str, value: Any) -> None:
    with pytest.raises(FirstPassError) as caught:
        billable_runner_config(
            frames=[], source=SOURCE, out_dir=tmp_path,
            prompt_text=PROMPT,
            client_factory=lambda: MetaSparkClient(
                client_factory=lambda key, timeout: _FakeSDK([])),
            **{field: value})
    assert caught.value.code == "INVALID_CONFIG"


def test_adapter_runs_offline_end_to_end(tmp_path: Path) -> None:
    image = _image(tmp_path / "frames" / "frame_00000001.jpg")
    frames = [FrameRef(frame_index=1, image_path=image,
                       timestamp_seconds=0.1)]
    sdk = _FakeSDK([_sdk_response()])
    client = MetaSparkClient(
        client_factory=lambda key, timeout: sdk)
    config = billable_runner_config(
        frames=frames, source=SOURCE,
        out_dir=tmp_path / "run", prompt_text=PROMPT,
        client_factory=lambda: client, token_budget=512)
    report = FirstPassRunner(config).run()
    assert report["billable"] is True
    assert report["frames_observed"] == 1
    assert report["budget"]["stopped_reason"] is None
    (call,) = sdk.responses.calls
    assert call["max_output_tokens"] == 512


def test_request_constrains_observation_schema(tmp_path: Path) -> None:
    image = _image(tmp_path / "frame.jpg")
    client = _client([_sdk_response()])
    client.submit(PROMPT, [image], "key", 1024)
    (call,) = _sdk_of(client).responses.calls
    assert call["text"] == {
        "format": {
            "type": "json_schema",
            "name": OBSERVATION_SCHEMA_NAME,
            "schema": OBSERVATION_JSON_SCHEMA,
        }
    }
    assert set(OBSERVATION_JSON_SCHEMA["properties"]) == set(
        MODEL_OUTPUT_FIELDS)
    assert sorted(OBSERVATION_JSON_SCHEMA["required"]) == sorted(
        MODEL_OUTPUT_FIELDS)
    assert OBSERVATION_JSON_SCHEMA["additionalProperties"] is False
    assert "$ref" not in json.dumps(OBSERVATION_JSON_SCHEMA)
