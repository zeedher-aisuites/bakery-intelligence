"""Offline tests for probe adaptive recovery (Exp. 010).

All API interactions use injected fake clients with scripted
responses; no test performs network or paid calls. The API key is a
synthetic monkeypatched value, never a credential.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "007_selective_verification"
sys.path.insert(0, str(EXPERIMENT_DIR))

import probe_temporal_spatial as probe  # noqa: E402


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("META_API_KEY", "test-key-not-a-credential")
    monkeypatch.delenv("MODEL_API_KEY", raising=False)


class APIStatusError(Exception):
    """Fake transport error shaped for meta_client mapping."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int,
                 reasoning: int = 0) -> None:
        self._data = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_details": {},
            "output_tokens_details": (
                {"reasoning_tokens": reasoning} if reasoning else {}),
        }

    def model_dump(self) -> dict[str, Any]:
        return dict(self._data)


class FakeResponse:
    def __init__(self, status: str, output_text: str,
                 usage: FakeUsage | None = None,
                 incomplete_reason: str | None = None) -> None:
        self.status = status
        self.output_text = output_text
        self.usage = usage or FakeUsage(10, 5)
        self.model = "muse-spark-1.3"
        self.id = "resp_test"
        self.incomplete_details = (
            SimpleNamespace(reason=incomplete_reason)
            if incomplete_reason is not None else None)

    def model_dump(self, mode: str | None = None) -> dict[str, Any]:
        return {"status": self.status, "output_text": self.output_text,
                "model": self.model, "id": self.id,
                "usage": self.usage.model_dump(),
                "incomplete_details": (
                    {"reason": self.incomplete_details.reason}
                    if self.incomplete_details is not None else None)}


class FakeResponses:
    def __init__(self, script: list[Any], calls: list[dict]) -> None:
        self._script = script
        self._calls = calls

    def create(self, **kwargs: Any) -> Any:
        self._calls.append(kwargs)
        action = self._script[len(self._calls) - 1]
        if isinstance(action, Exception):
            raise action
        return action


def make_factory(script: list[Any],
                 calls: list[dict]) -> Any:
    def factory(api_key: str, timeout_seconds: float) -> Any:
        assert api_key == "test-key-not-a-credential"
        namespace = SimpleNamespace()
        namespace.responses = FakeResponses(script, calls)
        return namespace

    return factory


def _prediction(display: str = "515",
                numeric: float | None = 515.0) -> str:
    return json.dumps({
        "target_timestamp_seconds": 150.0,
        "display_text": display,
        "numeric_value": numeric,
        "readable": True,
        "confidence": "high",
        "evidence_timestamps": [150.0],
        "direct_visual_evidence": "digits 5, 1, 5",
        "temporal_inference_used": False,
        "uncertainty_notes": None,
    })


def _write_manifest(out_root: Path,
                    condition_ids: tuple[str, ...] = ("A",),
                    suffix: str = ".jpg") -> None:
    conditions = []
    for cid in condition_ids:
        name = f"img_{cid}{suffix}"
        path = out_root / name
        path.write_bytes(f"fake-bytes-{cid}".encode("utf-8"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        conditions.append({
            "id": cid,
            "purpose": f"synthetic {cid}",
            "eligible": True,
            "eligibility_notes": ["test"],
            "images": [{"path": str(path), "sha256": digest,
                        "actual_timestamp": 150.0,
                        "frame_index": 7500,
                        "requested_timestamp": 150.0,
                        "variant": "test", "bytes": path.stat().st_size}],
        })
    (out_root / "input_manifest.json").write_text(json.dumps({
        "target_timestamp_seconds": 150.0,
        "resolution_finding": "test",
        "conditions": conditions,
    }), encoding="utf-8")


def _attempt(out_root: Path, run_id: str, name: str) -> dict[str, Any]:
    path = out_root / "runs" / run_id / "attempts" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_retry_recovers_at_second_budget(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    calls: list[dict] = []
    script = [
        FakeResponse("incomplete", "",
                     FakeUsage(100, 1024),
                     incomplete_reason="max_output_tokens"),
        FakeResponse("completed", _prediction(), FakeUsage(100, 60)),
    ]
    report = probe.run_conditions(
        tmp_path, "run_t01", make_factory(script, calls))
    assert [call["max_output_tokens"] for call in calls] == [1024, 2048]
    assert all(call["store"] is False for call in calls)
    assert report["conditions"]["A"]["status"] == "valid_prediction"
    assert report["conditions"]["A"]["selected_attempt"] == 2
    assert report["recovery"]["total_requests"] == 2
    assert report["recovery"]["adaptive_retries"] == 1
    assert report["recovery"]["conditions_recovered"] == 1
    assert report["conditions"]["A"]["evaluation"][
        "display_exact_match"] is True


def test_all_budgets_exhausted(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    calls: list[dict] = []
    script = [
        FakeResponse("incomplete", "", FakeUsage(100, budget),
                     incomplete_reason="max_output_tokens")
        for budget in (1024, 2048, 3072)
    ]
    report = probe.run_conditions(
        tmp_path, "run_t02", make_factory(script, calls))
    assert [call["max_output_tokens"] for call in calls] == [
        1024, 2048, 3072]
    assert report["conditions"]["A"]["status"] == "exhausted_token_budget"
    assert report["conditions"]["A"]["selected_attempt"] is None
    assert report["recovery"]["conditions_exhausted"] == 1


def test_first_attempt_success_no_retry(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    calls: list[dict] = []
    script = [FakeResponse("completed", _prediction(),
                           FakeUsage(100, 60))]
    report = probe.run_conditions(
        tmp_path, "run_t03", make_factory(script, calls))
    assert len(calls) == 1
    assert report["conditions"]["A"]["status"] == "valid_prediction"
    assert report["recovery"]["adaptive_retries"] == 0


def test_completed_invalid_json_recorded(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    calls: list[dict] = []
    script = [FakeResponse("completed", "{oops", FakeUsage(100, 12))]
    report = probe.run_conditions(
        tmp_path, "run_t04", make_factory(script, calls))
    assert len(calls) == 1
    assert report["conditions"]["A"]["status"] == "invalid_json"


def test_completed_empty_output_recorded(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    calls: list[dict] = []
    script = [FakeResponse("completed", "  ", FakeUsage(100, 4))]
    report = probe.run_conditions(
        tmp_path, "run_t05", make_factory(script, calls))
    assert len(calls) == 1
    assert report["conditions"]["A"]["status"] == "empty_output"


def test_413_has_no_token_escalation(tmp_path: Path) -> None:
    _write_manifest(tmp_path, ("E",), ".png")
    calls: list[dict] = []
    script = [APIStatusError("too large", 413)]
    report = probe.run_conditions(
        tmp_path, "run_t06", make_factory(script, calls))
    assert len(calls) == 1
    assert calls[0]["max_output_tokens"] == 1024
    assert report["conditions"]["E"]["status"] == "request_failed"
    attempt = _attempt(tmp_path, "run_t06", "E_attempt_01.json")
    assert attempt["error"]["type"] == "PAYLOAD_TOO_LARGE"
    assert attempt["retry_triggered"] is False


def test_429_has_no_automatic_retry(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    calls: list[dict] = []
    script = [APIStatusError("limited", 429)]
    report = probe.run_conditions(
        tmp_path, "run_t07", make_factory(script, calls))
    assert len(calls) == 1
    assert report["conditions"]["A"]["status"] == "request_failed"
    attempt = _attempt(tmp_path, "run_t07", "A_attempt_01.json")
    assert attempt["error"]["type"] == "RATE_LIMITED"


def test_valid_but_incorrect_accepted_without_retry(
        tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    calls: list[dict] = []
    script = [FakeResponse("completed", _prediction("514", 514.0),
                           FakeUsage(100, 60))]
    report = probe.run_conditions(
        tmp_path, "run_t08", make_factory(script, calls))
    assert len(calls) == 1
    result = report["conditions"]["A"]
    assert result["status"] == "valid_prediction"
    assert result["evaluation"]["display_exact_match"] is False
    assert result["evaluation"]["numeric_match"] is False


def test_input_hashes_preserved(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    calls: list[dict] = []
    script = [FakeResponse("completed", _prediction(),
                           FakeUsage(100, 60))]
    probe.run_conditions(tmp_path, "run_t09",
                         make_factory(script, calls))
    manifest = json.loads(
        (tmp_path / "input_manifest.json").read_text(encoding="utf-8"))
    expected = [image["sha256"]
                for image in manifest["conditions"][0]["images"]]
    attempt = _attempt(tmp_path, "run_t09", "A_attempt_01.json")
    assert attempt["input_image_hashes"] == expected


def test_attempt_paths_distinct(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    calls: list[dict] = []
    script = [
        FakeResponse("incomplete", "", FakeUsage(100, 1024),
                     incomplete_reason="max_output_tokens"),
        FakeResponse("completed", _prediction(), FakeUsage(100, 60)),
    ]
    probe.run_conditions(tmp_path, "run_t10",
                         make_factory(script, calls))
    first = _attempt(tmp_path, "run_t10", "A_attempt_01.json")
    second = _attempt(tmp_path, "run_t10", "A_attempt_02.json")
    assert first["attempt_number"] == 1
    assert second["attempt_number"] == 2
    assert first["max_output_tokens"] == 1024
    assert second["max_output_tokens"] == 2048
    assert first["retry_triggered"] is True
    assert second["retry_triggered"] is False


def test_run_overwrite_protected(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    script = [FakeResponse("completed", _prediction(),
                           FakeUsage(100, 60))]
    probe.run_conditions(tmp_path, "run_keep",
                         make_factory(script, []))
    sentinel = tmp_path / "runs" / "run_keep" / "sentinel.txt"
    sentinel.write_text("prior", encoding="utf-8")
    with pytest.raises(probe.ProbeError) as exc_info:
        probe.run_conditions(tmp_path, "run_keep",
                             make_factory(script, []))
    assert exc_info.value.code == "OUTPUT_EXISTS"
    assert sentinel.read_text(encoding="utf-8") == "prior"


def test_cost_aggregation_across_attempts(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    calls: list[dict] = []
    script = [
        FakeResponse("incomplete", "", FakeUsage(1000, 1024),
                     incomplete_reason="max_output_tokens"),
        FakeResponse("completed", _prediction(),
                     FakeUsage(1000, 60)),
    ]
    report = probe.run_conditions(
        tmp_path, "run_t12", make_factory(script, calls))
    recovery = report["recovery"]
    assert recovery["input_tokens"] == 2000
    assert recovery["output_tokens"] == 1084
    assert recovery["total_tokens"] == 3084
    expected_total = round(
        2000 / 1e6 * 1.25 + 1084 / 1e6 * 4.25, 6)
    expected_wasted = round(
        1000 / 1e6 * 1.25 + 1024 / 1e6 * 4.25, 6)
    assert recovery["cost_usd_estimate"] == expected_total
    assert recovery["cost_usd_wasted_estimate"] == expected_wasted
    assert report["pricing_basis"][
        "status"] == "REPORTED_NOT_DIRECTLY_VERIFIED"


def test_store_false_recorded(tmp_path: Path) -> None:
    _write_manifest(tmp_path, ("E",), ".png")
    calls: list[dict] = []
    script = [FakeResponse("completed", _prediction(),
                           FakeUsage(100, 60))]
    report = probe.run_conditions(
        tmp_path, "run_t13", make_factory(script, calls))
    assert calls[0]["store"] is False
    attempt = _attempt(tmp_path, "run_t13", "E_attempt_01.json")
    assert attempt["store"] is False
    assert report["request_store"] is False


def test_incomplete_reason_preserved(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    calls: list[dict] = []
    script = [FakeResponse("incomplete", "", FakeUsage(100, 3072),
                           incomplete_reason="max_output_tokens")]
    # Force single-budget exhaustion path via three identical attempts.
    script = script * 3
    report = probe.run_conditions(
        tmp_path, "run_t14", make_factory(script, calls))
    assert report["conditions"]["A"]["status"] == "exhausted_token_budget"
    attempt = _attempt(tmp_path, "run_t14", "A_attempt_03.json")
    assert attempt["incomplete_reason"] == "max_output_tokens"
    assert attempt["raw_response"]["incomplete_details"] == {
        "reason": "max_output_tokens"}
    assert "token budgets exhausted" in attempt["retry_reason"]


def test_incomplete_without_reason_not_guessed(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    calls: list[dict] = []
    script = [FakeResponse("incomplete", "", FakeUsage(100, 40))]
    report = probe.run_conditions(
        tmp_path, "run_t15", make_factory(script, calls))
    assert len(calls) == 1
    assert report["conditions"]["A"]["status"] == "unsupported_response"


def test_all_five_conditions_rerun_from_first_budget(
        tmp_path: Path) -> None:
    _write_manifest(tmp_path, ("A", "B", "C", "D", "E"))
    calls: list[dict] = []
    script = [FakeResponse("completed", _prediction(),
                           FakeUsage(100, 60)) for _ in range(5)]
    report = probe.run_conditions(
        tmp_path, "run_t16", make_factory(script, calls))
    assert len(calls) == 5
    assert len(report["conditions"]) == 5
    for cid in ("A", "B", "C", "D", "E"):
        assert report["conditions"][cid]["status"] == "valid_prediction"
        assert report["conditions"][cid]["selected_attempt"] == 1
    assert [call["max_output_tokens"] for call in calls] == [1024] * 5
    assert report["recovery"]["total_requests"] == 5


def test_string_numeric_matches_float_reference(
        tmp_path: Path) -> None:
    """Probe-level regression for the run_adaptive_01 mismatch."""
    _write_manifest(tmp_path)
    calls: list[dict] = []
    payload = json.loads(_prediction())
    payload["numeric_value"] = "515"
    script = [FakeResponse("completed", json.dumps(payload),
                           FakeUsage(100, 60))]
    report = probe.run_conditions(
        tmp_path, "run_t18", make_factory(script, calls))
    evaluation = report["conditions"]["A"]["evaluation"]
    assert evaluation["display_exact_match"] is True
    assert evaluation["numeric_match"] is True


def test_reasoning_tokens_preserved(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    calls: list[dict] = []
    script = [FakeResponse("completed", _prediction(),
                           FakeUsage(100, 60, reasoning=25))]
    report = probe.run_conditions(
        tmp_path, "run_t17", make_factory(script, calls))
    attempt = _attempt(tmp_path, "run_t17", "A_attempt_01.json")
    assert attempt["reasoning_output_tokens"] == 25
    assert report["conditions"]["A"]["evaluation"]["tokens"][
        "reasoning_output_tokens"] == 25
