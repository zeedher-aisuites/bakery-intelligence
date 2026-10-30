"""Mocked tests for the reusable inference lifecycle (Exp. 006).

Covers terminal classification, budget escalation, reason capture,
request-storage control, and runner integration. No network or paid
calls; provider and filesystem fixtures are synthetic and tmp-local.
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

import lifecycle  # noqa: E402
import meta_client  # noqa: E402
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
ABSTAIN_JSON = json.dumps({
    "visible_objects": ["scale"], "scale_display_verbatim": None,
    "scale_readable": False, "scale_state": "occluded",
    "tare_action_visible": False,
    "addition_or_removal_visible": "none",
    "container_description": "bowl", "container_changed_hint": False,
    "confidence": "low", "ambiguity_notes": "occluded",
})


def _usage(prompt: int, completion: int) -> Any:
    return types.SimpleNamespace(model_dump=lambda: {
        "input_tokens": prompt, "output_tokens": completion,
        "total_tokens": prompt + completion,
        "input_tokens_details": {}, "output_tokens_details": {},
    })


def _response(text: str = VALID_JSON, status: str = "completed",
              usage: Any = None, reason: Any = None) -> Any:
    return types.SimpleNamespace(
        status=status, output_text=text, model="muse-spark-1.3",
        id="req-test", incomplete_details=reason, usage=usage,
    )


def _reason(text: str) -> Any:
    return types.SimpleNamespace(reason=text)


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


def _factory(script: list, calls: list):  # noqa: ANN202
    def build(api_key: str, timeout: float) -> Any:
        namespace = types.SimpleNamespace()
        namespace.responses = _FakeResponses(script, calls)
        return namespace

    return build


def _write_manifest(path: Path, count: int) -> None:
    lines = ["image_path,frame_index,timestamp_seconds,"
             "source_timestamp_seconds,requested_timestamp_seconds"]
    for row in range(count):
        index = row * 50
        stamp = index / 50.0
        lines.append(
            f"frame_{index:08d}.jpg,{index},{stamp},{stamp},{stamp}"
        )
        (path.parent / f"frame_{index:08d}.jpg").write_bytes(FAKE_JPEG)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                 count: int = 1, **overrides: Any) -> runner.RunConfig:
    frames = tmp_path / "frames"
    frames.mkdir()
    manifest = frames / "manifest.csv"
    _write_manifest(manifest, count)
    monkeypatch.setattr(runner, "EXPECTED_ROWS", count)
    monkeypatch.setattr(runner, "EXPECTED_LAST_INDEX", (count - 1) * 50)
    monkeypatch.setenv("META_API_KEY", "test-key")
    params: dict[str, Any] = {
        "manifest_path": manifest, "run_dir": tmp_path / "run",
        "model": "muse-spark-1.3", "batch_size": 1,
        "sleeper": lambda seconds: None,
    }
    params.update(overrides)
    return runner.RunConfig(**params)


def _report(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "reports" / "run_report.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _single(dir_path: Path) -> dict[str, Any]:
    files = sorted(dir_path.glob("batch_*.json"))
    assert len(files) == 1
    records = json.loads(files[0].read_text(encoding="utf-8"))
    assert len(records) == 1
    return records[0]


def test_policy_defaults_and_validation() -> None:
    policy = lifecycle.BudgetPolicy()
    assert policy.budgets == (1024,)
    with pytest.raises(ValueError):
        lifecycle.BudgetPolicy(budgets=())
    with pytest.raises(ValueError):
        lifecycle.BudgetPolicy(budgets=(1024, 0))
    with pytest.raises(ValueError):
        lifecycle.BudgetPolicy(budgets=(1024, True))
    with pytest.raises(ValueError):
        lifecycle.BudgetPolicy(max_requests=0)


def test_token_limit_reason_matching() -> None:
    assert lifecycle.is_token_limit_reason("max_output_tokens") is True
    assert lifecycle.is_token_limit_reason(None) is False
    assert lifecycle.is_token_limit_reason("content_filter") is False
    assert lifecycle.is_token_limit_reason("") is False


def test_decide_attempt_matrix() -> None:
    policy = lifecycle.BudgetPolicy(budgets=(1024, 2048))
    decide = lifecycle.decide_attempt
    assert decide("completed", None, VALID_JSON, True, 1,
                  policy)["terminal"] == "valid_prediction"
    assert decide("completed", None, "{oops", False, 1,
                  policy)["terminal"] == "invalid_json"
    assert decide("completed", None, "  ", False, 1,
                  policy)["terminal"] == "empty_output"
    assert decide("completed", None, VALID_JSON, True, 1, policy,
                  abstained_hint=True)["terminal"] == "abstained"
    failed = decide("completed", None, VALID_JSON, True, 1, policy,
                    transport_error="TIMEOUT")
    assert failed["terminal"] == "request_failed"
    assert failed["action"] == "stop"
    retry = decide("incomplete", "max_output_tokens", "", False, 1,
                   policy)
    assert retry["action"] == "retry"
    exhausted = decide("incomplete", "max_output_tokens", "", False, 2,
                       policy)
    assert exhausted["terminal"] == "exhausted_token_budget"
    assert decide("incomplete", "content_filter", "", False, 1,
                  policy)["terminal"] == "unsupported_response"
    assert decide("incomplete", None, "", False, 1,
                  policy)["terminal"] == "unsupported_response"
    assert decide("failed", None, "", False, 1,
                  policy)["terminal"] == "unsupported_response"


def test_estimate_cost_values() -> None:
    rates = (1.25, 4.25)
    assert lifecycle.estimate_cost(1000, 2000, rates) == round(
        1000 * 1.25 / 1_000_000 + 2000 * 4.25 / 1_000_000, 4)
    assert lifecycle.estimate_cost(1000, 2000, None) is None
    assert lifecycle.estimate_cost(None, 2000, rates) is None
    assert lifecycle.estimate_cost(True, 2000, rates) is None


def test_store_flag_and_reason_capture(tmp_path: Path) -> None:
    image = tmp_path / "a.jpg"
    image.write_bytes(FAKE_JPEG)
    calls: list[dict] = []
    client = meta_client.MetaClient(
        store=False,
        client_factory=_factory(
            [_response(usage=_usage(10, 5),
                       reason=_reason("max_output_tokens"))],
            calls,
        ),
    )
    result = client.submit("prompt", [image], "key")
    assert calls[0]["store"] is False
    assert result.incomplete_reason == "max_output_tokens"


def test_store_defaults_true(tmp_path: Path) -> None:
    image = tmp_path / "a.jpg"
    image.write_bytes(FAKE_JPEG)
    calls: list[dict] = []
    client = meta_client.MetaClient(
        client_factory=_factory(
            [_response(usage=_usage(10, 5))], calls),
    )
    client.submit("prompt", [image], "key")
    assert calls[0]["store"] is True


def test_incomplete_reason_dict_shape(tmp_path: Path) -> None:
    image = tmp_path / "a.jpg"
    image.write_bytes(FAKE_JPEG)
    calls: list[dict] = []
    client = meta_client.MetaClient(
        client_factory=_factory(
            [_response(usage=_usage(10, 5),
                       reason={"reason": "max_output_tokens"})],
            calls),
    )
    result = client.submit("prompt", [image], "key")
    assert result.incomplete_reason == "max_output_tokens"


def test_runner_escalates_then_recovers(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []
    script = [
        _response("", "incomplete", _usage(2900, 1024),
                  _reason("max_output_tokens")),
        _response(VALID_JSON, "completed", _usage(2900, 60)),
    ]
    config = _make_config(
        tmp_path, monkeypatch, client_factory=_factory(script, calls),
        token_budgets=(1024, 2048),
    )
    state = runner.Runner(config).run()
    assert [call["max_output_tokens"] for call in calls] == [1024, 2048]
    raw = _single(tmp_path / "run" / "raw_responses")
    assert raw["budgets_attempted"] == [1024, 2048]
    assert raw["terminal_state"] == "valid_prediction"
    obs = _single(tmp_path / "run" / "observations")
    assert obs["terminal_state"] == "valid_prediction"
    assert obs["image_ref"]["encoding"] == "jpeg"
    assert len(obs["image_ref"]["sha256"]) == 64
    assert obs["call"]["token_budgets_attempted"] == [1024, 2048]
    assert state["lifecycle"]["recovery_rate"] == 1.0
    assert state["lifecycle"]["token_retries"] == 1
    assert state["requests_sent"] == 1
    assert state["retries"] == 0


def test_runner_default_single_budget_preserves_incomplete_code(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = [_response("", "incomplete", _usage(2900, 1024),
                        _reason("max_output_tokens"))]
    config = _make_config(
        tmp_path, monkeypatch,
        client_factory=_factory(script, []),
    )
    runner.Runner(config).run()
    raw = _single(tmp_path / "run" / "raw_responses")
    assert raw["error_code"] == "INCOMPLETE_STATUS"
    assert raw["terminal_state"] == "exhausted_token_budget"
    assert raw["incomplete_reason"] == "max_output_tokens"


def test_runner_abstention_classified(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = [_response(ABSTAIN_JSON, "completed", _usage(2900, 60))]
    config = _make_config(
        tmp_path, monkeypatch,
        client_factory=_factory(script, []),
    )
    state = runner.Runner(config).run()
    obs = _single(tmp_path / "run" / "observations")
    assert obs["terminal_state"] == "abstained"
    assert state["lifecycle"]["abstention_rate"] == 1.0
    assert state["lifecycle"]["terminal_states"] == {"abstained": 1}


def test_runner_invalid_json_classified(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = [_response("{oops", "completed", _usage(2900, 12))]
    config = _make_config(
        tmp_path, monkeypatch,
        client_factory=_factory(script, []),
    )
    state = runner.Runner(config).run()
    raw = _single(tmp_path / "run" / "raw_responses")
    assert raw["terminal_state"] == "invalid_json"
    assert state["lifecycle"]["invalid_rate"] == 1.0
