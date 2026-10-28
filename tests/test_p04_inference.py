"""Mocked-provider tests for the P04 full-video experiment (Exp. 006).

No test here touches the network, real credentials, or paid APIs: the
provider seam (client factory) is always faked, and API keys are
monkeypatched. Oracles come from the verified Meta contract, the
extractor manifest format, and the frozen runner semantics.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "006_p04_multimodal_inference"
sys.path.insert(0, str(EXPERIMENT_DIR))

import meta_client  # noqa: E402
import prompts  # noqa: E402
import runner  # noqa: E402

FAKE_JPEG = b"\xff\xd8" + b"0" * 64 + b"\xff\xd9"
VALID_JSON = json.dumps({
    "visible_objects": ["scale"], "scale_display_verbatim": "120",
    "scale_readable": True, "scale_state": "stable",
    "tare_action_visible": False,
    "addition_or_removal_visible": "none",
    "container_description": "bowl", "container_changed_hint": False,
    "confidence": "high", "ambiguity_notes": None,
})


class APIStatusError(Exception):
    def __init__(self, message: str, status_code: int,
                 retry_after: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        headers = {"retry-after": retry_after} if retry_after else {}
        self.response = types.SimpleNamespace(headers=headers)


class APIConnectionError(Exception):
    pass


class APITimeoutError(APIConnectionError):
    pass


def _usage(prompt: int, completion: int) -> Any:
    return types.SimpleNamespace(model_dump=lambda: {
        "input_tokens": prompt, "output_tokens": completion,
        "total_tokens": prompt + completion,
        "input_tokens_details": {}, "output_tokens_details": {},
    })


def _response(text: str = VALID_JSON, status: str = "completed",
              usage: Any = None) -> Any:
    return types.SimpleNamespace(
        status=status, output_text=text, model="muse-spark-1.3",
        id="req-test", incomplete_details=None,
        usage=usage if usage is not None else _usage(2900, 550),
    )


class _FakeResponses:
    def __init__(self, script: list, calls: list) -> None:
        self.script = script
        self.calls = calls

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        action = self.script.pop(0)
        if isinstance(action, BaseException):
            raise action
        return action


class _FakeClient:
    def __init__(self, script: list, calls: list) -> None:
        self.responses = _FakeResponses(script, calls)


def _factory(script: list, constructions: list, calls: list):  # noqa: ANN202
    def build(api_key: str, timeout: float) -> _FakeClient:
        constructions.append((api_key, timeout))
        return _FakeClient(script, calls)

    return build


def _write_manifest(path: Path, count: int, offset: float = 0.0) -> None:
    lines = ["image_path,frame_index,timestamp_seconds,"
             "source_timestamp_seconds,requested_timestamp_seconds"]
    for row in range(count):
        index = row * 50
        stamp = index / 50.0
        lines.append(
            f"frame_{index:08d}.jpg,{index},{stamp},{stamp + offset},{stamp}"
        )
        (path.parent / f"frame_{index:08d}.jpg").write_bytes(FAKE_JPEG)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                 count: int = 4, **overrides: Any) -> runner.RunConfig:
    frames = tmp_path / "frames"
    frames.mkdir()
    manifest = frames / "manifest.csv"
    _write_manifest(manifest, count)
    monkeypatch.setattr(runner, "EXPECTED_ROWS", count)
    monkeypatch.setattr(runner, "EXPECTED_LAST_INDEX", (count - 1) * 50)
    monkeypatch.setenv("META_API_KEY", "test-key")
    params: dict[str, Any] = {
        "manifest_path": manifest, "run_dir": tmp_path / "run",
        "model": "muse-spark-1.3", "batch_size": 2,
        "sleeper": lambda seconds: None,
    }
    params.update(overrides)
    return runner.RunConfig(**params)


def test_missing_api_key_aborts_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    calls: list = []
    config = _make_config(tmp_path, monkeypatch)
    monkeypatch.delenv("META_API_KEY", raising=False)
    config.client_factory = _factory([], [], calls)
    with pytest.raises(meta_client.ProviderError) as exc_info:
        runner.Runner(config).run()
    assert exc_info.value.code == "CREDENTIAL_MISSING"
    assert calls == []
    assert not (tmp_path / "run").exists()


def test_invalid_manifest_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    manifest = frames / "manifest.csv"
    manifest.write_text("wrong,header\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(runner, "EXPECTED_ROWS", 1)
    monkeypatch.setenv("META_API_KEY", "test-key")
    calls: list = []
    config = runner.RunConfig(
        manifest_path=manifest, run_dir=tmp_path / "run",
        model="muse-spark-1.3",
        client_factory=_factory([], [], calls),
    )
    with pytest.raises(runner.RunnerError) as exc_info:
        runner.Runner(config).run()
    assert exc_info.value.code == "MANIFEST_INVALID"
    assert calls == []


def test_missing_image_not_billed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    manifest = frames / "manifest.csv"
    _write_manifest(manifest, 2)
    (frames / "frame_00000050.jpg").unlink()
    monkeypatch.setattr(runner, "EXPECTED_ROWS", 2)
    monkeypatch.setattr(runner, "EXPECTED_LAST_INDEX", 50)
    monkeypatch.setenv("META_API_KEY", "test-key")
    calls: list = []
    config = runner.RunConfig(
        manifest_path=manifest, run_dir=tmp_path / "run",
        model="muse-spark-1.3",
        client_factory=_factory([_response()], [], calls),
    )
    with pytest.raises(runner.RunnerError) as exc_info:
        runner.Runner(config).run()
    assert exc_info.value.code == "IMAGE_MISSING"
    assert exc_info.value.entity_id == "50"
    assert calls == []


def test_malformed_provider_response_quarantined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = types.SimpleNamespace(status=None, output_text=None)
    script: list = [bad, bad, bad, bad]
    calls: list = []
    config = _make_config(tmp_path, monkeypatch, count=2)
    config.client_factory = _factory(script, [], calls)
    state = runner.Runner(config).run()
    assert state["batches_needing_review"] == 1
    assert state["batches_completed"] == 0
    raw_path = next((tmp_path / "run" / "raw_responses").glob("*.json"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    assert raw[0]["error_code"] == "RESPONSE_MALFORMED"


def test_auth_failure_aborts_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script: list = [APIStatusError("bad key", 401)]
    calls: list = []
    config = _make_config(tmp_path, monkeypatch, count=2)
    config.client_factory = _factory(script, [], calls)
    with pytest.raises(meta_client.ProviderError) as exc_info:
        runner.Runner(config).run()
    assert exc_info.value.code == "AUTH_FAILED"
    assert len(calls) == 1
    run_dir = tmp_path / "run"
    blob = ""
    if run_dir.is_dir():
        blob = "".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in run_dir.rglob("*") if path.is_file()
        )
    assert "test-key" not in blob


def test_rate_limit_retries_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script: list = [
        APIStatusError("slow", 429, retry_after="2"),
        APIStatusError("slow", 429),
        _response(),
    ]
    calls: list = []
    sleeps: list = []
    config = _make_config(tmp_path, monkeypatch, count=1)
    config.client_factory = _factory(script, [], calls)
    config.sleeper = sleeps.append
    state = runner.Runner(config).run()
    assert state["batches_completed"] == 1
    assert len(calls) == 3
    assert sleeps[0] == 2.0
    assert 1.0 <= sleeps[1] <= 1.25


def test_persistent_rate_limit_marks_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script: list = [APIStatusError("slow", 429)] * 6
    calls: list = []
    config = _make_config(tmp_path, monkeypatch, count=1)
    config.client_factory = _factory(script, [], calls)
    state = runner.Runner(config).run()
    assert state["batches_needing_review"] == 1
    assert len(calls) == 4
    obs_path = next((tmp_path / "run" / "observations").glob("*.json"))
    obs = json.loads(obs_path.read_text(encoding="utf-8"))
    assert obs == []


def test_timeout_retryable_without_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script: list = [APITimeoutError("slow"), _response()]
    calls: list = []
    config = _make_config(tmp_path, monkeypatch, count=1)
    config.client_factory = _factory(script, [], calls)
    state = runner.Runner(config).run()
    assert state["batches_completed"] == 1
    assert state["input_tokens"] == 2900
    assert state["output_tokens"] == 550


def test_partial_batch_failure_preserves_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = types.SimpleNamespace(status=None, output_text=None)
    script: list = [_response(), bad, _response()]
    calls: list = []
    config = _make_config(tmp_path, monkeypatch, count=3, batch_size=3)
    config.client_factory = _factory(script, [], calls)
    state = runner.Runner(config).run()
    assert state["batches_needing_review"] == 1
    obs_path = next((tmp_path / "run" / "observations").glob("*.json"))
    obs = json.loads(obs_path.read_text(encoding="utf-8"))
    assert [item["frame_index"] for item in obs] == [0, 100]


def test_resume_skips_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # KeyboardInterrupt (BaseException) models a real interruption: it
    # bypasses MetaClient.submit's `except Exception` provider mapping
    # exactly like Ctrl-C, while a RuntimeError would be correctly
    # converted to a PROVIDER_ERROR failure record, not propagated.
    boom: list = [_response(), _response(), KeyboardInterrupt()]
    calls_one: list = []
    config = _make_config(tmp_path, monkeypatch, count=4, batch_size=2)
    config.client_factory = _factory(boom, [], calls_one)
    with pytest.raises(KeyboardInterrupt):
        runner.Runner(config).run()
    first_batch = tmp_path / "run" / "observations" / "batch_00000000-00000050.json"
    before = first_batch.read_bytes()
    script: list = [_response(), _response()]
    calls_two: list = []
    config.client_factory = _factory(script, [], calls_two)
    state = runner.Runner(config).run()
    assert state["batches_completed"] == 2
    assert len(calls_two) == 2
    assert first_batch.read_bytes() == before


def test_completed_batches_never_rebilled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path, monkeypatch, count=2, batch_size=2)
    calls_one: list = []
    config.client_factory = _factory([_response(), _response()], [], calls_one)
    first = runner.Runner(config).run()
    calls_two: list = []
    config.client_factory = _factory([], [], calls_two)
    second = runner.Runner(config).run()
    assert calls_two == []
    assert second["input_tokens"] == 0
    assert first["input_tokens"] == 5800


def test_timestamp_provenance_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    manifest = frames / "manifest.csv"
    _write_manifest(manifest, 1, offset=225.0)
    monkeypatch.setattr(runner, "EXPECTED_ROWS", 1)
    monkeypatch.setattr(runner, "EXPECTED_LAST_INDEX", 0)
    monkeypatch.setenv("META_API_KEY", "test-key")
    config = runner.RunConfig(
        manifest_path=manifest, run_dir=tmp_path / "run",
        model="muse-spark-1.3",
        client_factory=_factory([_response()], [], []),
    )
    runner.Runner(config).run()
    obs_path = next((tmp_path / "run" / "observations").glob("*.json"))
    obs = json.loads(obs_path.read_text(encoding="utf-8"))[0]
    assert obs["timestamp_seconds"] == 0.0
    assert obs["source_timestamp_seconds"] == 225.0
    assert obs["requested_timestamp_seconds"] == 0.0
    assert obs["timestamp_basis"] == "frame_index_over_fps"


def test_invalid_model_output_never_becomes_quantity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    texts = ['{"scale_display_verbatim": "54\\u201355"}',
             '{"scale_display_verbatim": "-750"}',
             "not json at all"]
    script: list = [_response(text=text) for text in texts]
    config = _make_config(tmp_path, monkeypatch, count=3, batch_size=3)
    config.client_factory = _factory(script, [], [])
    runner.Runner(config).run()
    obs_path = next((tmp_path / "run" / "observations").glob("*.json"))
    obs = json.loads(obs_path.read_text(encoding="utf-8"))
    assert obs[0]["model_fields"]["scale_display_verbatim"] == "54\u201355"
    assert obs[0]["parse_error"] is None
    assert obs[1]["model_fields"]["scale_display_verbatim"] == "-750"
    assert "net_ingredient_quantity" not in json.dumps(obs)
    assert obs[2]["parse_error"] == "unparseable"
    assert obs[2]["model_fields"]["scale_display_verbatim"] is None


def test_cost_aggregation_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script: list = [
        _response(usage=_usage(1000, 100)),
        _response(usage=_usage(2000, 200)),
    ]
    config = _make_config(tmp_path, monkeypatch, count=2, batch_size=2)
    config.client_factory = _factory(script, [], [])
    state = runner.Runner(config).run()
    assert state["input_tokens"] == 3000
    assert state["output_tokens"] == 300
    assert state["cost_usd_estimated"] == round(
        3000 * 1.25 / 1_000_000 + 300 * 4.25 / 1_000_000, 4
    )
    assert state["cost_status"] == "estimated_from_published_rates"


def test_missing_usage_yields_cost_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _response()
    response.usage = None
    config = _make_config(tmp_path, monkeypatch, count=1)
    config.client_factory = _factory([response], [], [])
    state = runner.Runner(config).run()
    assert state["cost_usd_estimated"] is None
    assert state["cost_status"] == "COST_UNKNOWN"


def test_output_isolation_and_no_key_echo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path, monkeypatch, count=2)
    config.client_factory = _factory([_response(), _response()], [], [])
    manifest_before = (tmp_path / "frames" / "manifest.csv").read_bytes()
    runner.Runner(config).run()
    assert (tmp_path / "frames" / "manifest.csv").read_bytes() == manifest_before
    for path in (tmp_path / "run").rglob("*"):
        assert tmp_path in path.resolve().parents
    blob = "".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (tmp_path / "run").rglob("*") if path.is_file()
    )
    assert "test-key" not in blob
    assert "api_key" not in blob


def test_images_per_request_cap_enforced() -> None:
    client = meta_client.MetaClient(client_factory=lambda k, t: None)
    with pytest.raises(meta_client.ProviderError) as exc_info:
        client.submit("prompt", [Path(f"f{i}.jpg") for i in range(51)], "key")
    assert exc_info.value.code == "TOO_MANY_IMAGES"


def test_review_selection_deterministic() -> None:
    observations = [
        {"frame_index": 100, "parse_error": None,
         "model_fields": {"confidence": "high"}},
        {"frame_index": 0, "parse_error": None,
         "model_fields": {"scale_readable": False}},
        {"frame_index": 50, "parse_error": "unparseable",
         "model_fields": {}},
        {"frame_index": 150, "parse_error": None,
         "model_fields": {"confidence": "low"}},
        {"frame_index": 200, "parse_error": None,
         "model_fields": {"tare_action_visible": True}},
    ]
    assert prompts.select_review_candidates(observations) == [0, 50, 150, 200]


def test_key_precedence_meta_over_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("META_API_KEY", "meta-key")
    monkeypatch.setenv("MODEL_API_KEY", "model-key")
    assert meta_client.resolve_api_key() == "meta-key"
    monkeypatch.delenv("META_API_KEY")
    assert meta_client.resolve_api_key() == "model-key"
    monkeypatch.setenv("MODEL_API_KEY", "")
    with pytest.raises(meta_client.ProviderError) as exc_info:
        meta_client.resolve_api_key()
    assert exc_info.value.code == "CREDENTIAL_MISSING"


def _custom_manifest(path: Path, rows: list[tuple]) -> None:
    lines = ["image_path,frame_index,timestamp_seconds,"
             "source_timestamp_seconds,requested_timestamp_seconds"]
    for name, index, stamp in rows:
        lines.append(f"{name},{index},{stamp},{stamp},{stamp}")
        (path.parent / name).write_bytes(FAKE_JPEG)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.parametrize("mutation", [
    "bad_step", "bad_stamp", "bad_name", "duplicate", "shifted",
])
def test_manifest_negative_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    manifest = frames / "manifest.csv"
    if mutation == "bad_step":
        rows = [("frame_00000000.jpg", 0, 0.0), ("frame_00000060.jpg", 60, 1.2)]
    elif mutation == "bad_stamp":
        rows = [("frame_00000000.jpg", 0, 0.0), ("frame_00000050.jpg", 50, 9.9)]
    elif mutation == "bad_name":
        rows = [("frame_00000000.jpg", 0, 0.0), ("other.jpg", 50, 1.0)]
    elif mutation == "duplicate":
        rows = [("frame_00000000.jpg", 0, 0.0), ("frame_00000000.jpg", 0, 0.0)]
    else:
        rows = [("frame_00000050.jpg", 50, 1.0), ("frame_00000100.jpg", 100, 2.0)]
    _custom_manifest(manifest, rows)
    monkeypatch.setattr(runner, "EXPECTED_ROWS", 2)
    monkeypatch.setattr(runner, "EXPECTED_LAST_INDEX", 50)
    with pytest.raises(runner.RunnerError) as exc_info:
        runner.load_manifest(manifest)
    assert exc_info.value.code == "MANIFEST_INVALID"


def test_corrupt_jpeg_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path, monkeypatch, count=1)
    (tmp_path / "frames" / "frame_00000000.jpg").write_bytes(b"nope")
    config.client_factory = _factory([], [], [])
    with pytest.raises(runner.RunnerError) as exc_info:
        runner.Runner(config).run()
    assert exc_info.value.code == "IMAGE_MISSING"


def test_incomplete_status_marks_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script: list = [_response(status="incomplete")]
    config = _make_config(tmp_path, monkeypatch, count=1)
    config.client_factory = _factory(script, [], [])
    state = runner.Runner(config).run()
    assert state["batches_needing_review"] == 1
    raw_path = next((tmp_path / "run" / "raw_responses").glob("*.json"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    assert raw[0]["error_code"] == "INCOMPLETE_STATUS"


def test_bad_request_terminal_no_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    script: list = [APIStatusError("bad", 400)]
    calls: list = []
    config = _make_config(tmp_path, monkeypatch, count=1)
    config.client_factory = _factory(script, [], calls)
    state = runner.Runner(config).run()
    assert state["batches_needing_review"] == 1
    assert len(calls) == 1


def test_review_pass_grounds_and_prices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib

    primary = {"frame_index": 0,
               "model_fields": {"scale_display_verbatim": "54"}}
    reply = json.dumps({
        "agree": False, "corrected_fields": {"scale_state": "changing"},
        "missed_evidence": None, "contradiction_flag": True,
        "confidence": "low",
    })
    script: list = [_response(text=reply, usage=_usage(1000, 100))]
    calls: list = []
    config = _make_config(tmp_path, monkeypatch, count=1)
    config.model = "muse-wonder-1.0"
    config.is_review = True
    config.primary_obs = {0: primary}
    config.client_factory = _factory(script, [], calls)
    state = runner.Runner(config).run()
    assert state["batches_completed"] == 1
    sent_text = calls[0]["input"][0]["content"][0]["text"]
    dumped = json.dumps(primary, sort_keys=True)
    assert "PRIMARY JSON" in sent_text and dumped in sent_text
    obs_path = next((tmp_path / "run" / "observations").glob("*.json"))
    obs = json.loads(obs_path.read_text(encoding="utf-8"))[0]
    assert obs["model_fields"]["agree"] is False
    assert obs["model_fields"]["corrected_fields"] == {
        "scale_state": "changing"}
    assert obs["parse_error"] is None
    assert obs["call"]["review_of_sha256"] == hashlib.sha256(
        dumped.encode("utf-8")).hexdigest()
    assert state["rates_usd_per_m"] == {"input": 4.0, "output": 20.0}
    assert state["cost_usd_estimated"] == 0.006


def test_multi_image_requires_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path, monkeypatch, count=2)
    config.images_per_request = 2
    config.client_factory = _factory([], [], [])
    with pytest.raises(runner.RunnerError) as exc_info:
        runner.Runner(config).run()
    assert exc_info.value.code == "INVALID_CONFIG"


def test_checkpoint_corrupt_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path, monkeypatch, count=1)
    checkpoints = tmp_path / "run" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "checkpoints.jsonl").write_text(
        "{not json\n", encoding="utf-8")
    config.client_factory = _factory([], [], [])
    with pytest.raises(runner.RunnerError) as exc_info:
        runner.Runner(config).run()
    assert exc_info.value.code == "CHECKPOINT_CORRUPT"
