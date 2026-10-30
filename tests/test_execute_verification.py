"""Mocked tests for single-event verification execution (Exp. 011).

All provider interactions use injected fake transports with scripted
responses; no test performs network or paid calls. Fixtures for the
plan, frames, and first-pass observations are synthetic and tmp-local.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "007_selective_verification"
sys.path.insert(0, str(EXPERIMENT_DIR))

import execute  # noqa: E402


@pytest.fixture()
def _fake_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("META_API_KEY", "test-key-not-a-credential")
    monkeypatch.delenv("MODEL_API_KEY", raising=False)


class APIStatusError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def _usage(prompt: int, completion: int) -> Any:
    return types.SimpleNamespace(model_dump=lambda: {
        "input_tokens": prompt, "output_tokens": completion,
        "total_tokens": prompt + completion,
        "input_tokens_details": {}, "output_tokens_details": {},
    })


def _response(text: str, status: str = "completed",
              usage: Any = None, reason: Any = None) -> Any:
    return types.SimpleNamespace(
        status=status, output_text=text, model="muse-spark-1.3",
        id="resp_test", incomplete_details=reason,
        usage=usage if usage is not None else _usage(50, 20),
    )


def _reason(text: str) -> Any:
    return types.SimpleNamespace(reason=text)


class _FakeResponses:
    def __init__(self, script: list, calls: list) -> None:
        self.script = script
        self.calls = calls

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        action = self.script[len(self.calls) - 1]
        if isinstance(action, Exception):
            raise action
        return action


def _factory(script: list, calls: list):  # noqa: ANN202
    def build(api_key: str, timeout: float) -> Any:
        namespace = types.SimpleNamespace()
        namespace.responses = _FakeResponses(script, calls)
        return namespace

    return build


def _readings_payload(displays: list, abstain: bool = False) -> str:
    readings = []
    for position, display in enumerate(displays):
        readings.append({
            "timestamp_seconds": 2.0 + position,
            "display_text": None if abstain else display,
            "numeric_value": None,
            "readable": not abstain,
            "confidence": "low" if abstain else "high",
            "uncertainty_notes": None,
        })
    return json.dumps({
        "event_id": "evt_test",
        "readings": readings,
        "action_visible": "none",
        "overall_confidence": "high",
        "uncertainty_notes": None,
    })


def _write_inputs(tmp_path: Path, frames: tuple[int, ...] = (100, 150),
                  verbatims: tuple[str | None, ...] = ("120", "120"),
                  event_id: str = "evt_test",
                  reasons: tuple[str, ...] = ()) -> dict[str, Path]:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "plan_version": "1.0.0",
        "windows_in_scope": ["V09"],
        "entries": [{
            "event_id": event_id,
            "candidate_type": "scale_visibility",
            "start_timestamp_seconds": 2.0,
            "end_timestamp_seconds": 3.0,
            "review_reasons": list(reasons),
            "uncertainty_notes": [],
            "reuse_frames": [{"frame_index": index,
                              "frame_available": True}
                             for index in frames],
            "token_budgets": [1024, 2048, 3072],
            "max_requests": 3,
        }],
        "summary": {},
    }), encoding="utf-8")
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    rows = ["frame_index,image_path,timestamp_seconds"]
    for index in frames:
        name = f"frame_{index:08d}.jpg"
        (frames_dir / name).write_bytes(b"\xff\xd8fake" + bytes([index]))
        rows.append(f"{index},{name},{index / 50.0}")
    csv_path = frames_dir / "manifest.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    obs_dir = tmp_path / "obs"
    obs_dir.mkdir()
    records = []
    for index, verbatim in zip(frames, verbatims):
        records.append({
            "frame_index": index,
            "image_path": f"frame_{index:08d}.jpg",
            "timestamp_seconds": index / 50.0,
            "batch_id": "batch_test",
            "model_fields": {
                "scale_display_verbatim": verbatim,
                "scale_readable": verbatim is not None,
                "scale_state": "stable",
                "tare_action_visible": False,
                "addition_or_removal_visible": "none",
                "confidence": "high",
            },
        })
    (obs_dir / "batch_test.json").write_text(
        json.dumps(records), encoding="utf-8")
    return {"plan": plan_path, "csv": csv_path, "obs": obs_dir,
            "frames": frames_dir}


def _base_args(paths: dict[str, Path],
               event_id: str = "evt_test") -> list[str]:
    return ["--plan", str(paths["plan"]), "--event-id", event_id,
            "--obs-dir", str(paths["obs"]),
            "--extraction-manifest", str(paths["csv"]),
            "--out", str(paths["plan"].parent / "executions")]


def test_valid_single_event_selection(tmp_path: Path, capsys: Any) -> None:
    paths = _write_inputs(tmp_path)
    code = execute.main(_base_args(paths) + ["--dry-run"])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["event_id"] == "evt_test"
    assert report["image_count"] == 2
    assert report["eligible"] is True
    assert [image["frame_index"] for image in report["images"]] == [100, 150]


def test_unknown_event_rejected(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    code = execute.main(_base_args(paths, "evt_nope") + ["--dry-run"])
    assert code == 1


def test_malformed_plan_rejected(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    paths["plan"].write_text("{not json", encoding="utf-8")
    assert execute.main(_base_args(paths) + ["--dry-run"]) == 1
    paths["plan"].write_text(json.dumps({"entries": "nope"}),
                             encoding="utf-8")
    assert execute.main(_base_args(paths) + ["--dry-run"]) == 1


def test_missing_image_ineligible(tmp_path: Path, capsys: Any) -> None:
    paths = _write_inputs(tmp_path)
    for name in ("frame_00000100.jpg", "frame_00000150.jpg"):
        (paths["frames"] / name).unlink()
    code = execute.main(_base_args(paths) + ["--dry-run"])
    assert code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["eligible"] is False
    assert any("no_available_images" in error
               for error in report["validation_errors"])


def test_chronological_frame_ordering(tmp_path: Path, capsys: Any) -> None:
    paths = _write_inputs(tmp_path, frames=(150, 100))
    code = execute.main(_base_args(paths) + ["--dry-run"])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert [image["frame_index"] for image in report["images"]] == [100, 150]


def test_dry_run_needs_no_key_and_makes_no_calls(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    paths = _write_inputs(tmp_path)

    def _exploding_factory(api_key: str, timeout: float) -> Any:
        raise AssertionError("must not construct transport")

    code = execute.main(_base_args(paths) + ["--dry-run"],
                        client_factory=_exploding_factory)
    assert code == 0
    assert not (tmp_path / "executions").exists()


def test_live_requires_explicit_authorization(
        tmp_path: Path, _fake_key: None) -> None:
    paths = _write_inputs(tmp_path)

    def _exploding_factory(api_key: str, timeout: float) -> Any:
        raise AssertionError("must not construct transport")

    code = execute.main(_base_args(paths),
                        client_factory=_exploding_factory)
    assert code == 0
    assert not (tmp_path / "executions").exists()


def test_live_without_key_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    paths = _write_inputs(tmp_path)
    code = execute.main(_base_args(paths) + ["--execute"])
    assert code == 1
    assert not (tmp_path / "executions").exists()


def _run_live(tmp_path: Path, paths: dict[str, Path], script: list,
              calls: list, extra: list[str] | None = None,
              run_id: str = "run_test") -> dict[str, Any]:
    args = _base_args(paths) + ["--execute", "--run-id", run_id,
                                "--price-input", "1.25",
                                "--price-output", "4.25"]
    if extra:
        args.extend(extra)
    code = execute.main(args, client_factory=_factory(script, calls))
    assert code == 0
    report_path = (tmp_path / "executions" / run_id
                   / "execution_report.json")
    return json.loads(report_path.read_text(encoding="utf-8"))


def test_adaptive_recovery(tmp_path: Path, _fake_key: None) -> None:
    paths = _write_inputs(tmp_path)
    calls: list[dict] = []
    script = [
        _response("", "incomplete", _usage(100, 1024),
                  _reason("max_output_tokens")),
        _response(_readings_payload(["120", "120"]), "completed",
                  _usage(100, 60)),
    ]
    report = _run_live(tmp_path, paths, script, calls)
    assert [call["max_output_tokens"] for call in calls] == [1024, 2048]
    assert report["terminal_state"] == "valid_prediction"
    assert report["requests_made"] == 2
    assert report["reconciliation"]["counts"] == {"agreement": 2}
    assert report["reconciliation"]["requires_human_review"] is False


def test_exhausted_budgets(tmp_path: Path, _fake_key: None) -> None:
    paths = _write_inputs(tmp_path)
    calls: list[dict] = []
    script = [
        _response("", "incomplete", _usage(100, budget),
                  _reason("max_output_tokens"))
        for budget in (1024, 2048, 3072)
    ]
    report = _run_live(tmp_path, paths, script, calls)
    assert len(calls) == 3
    assert report["terminal_state"] == "exhausted_token_budget"
    assert report["selected_attempt"] is None


def test_invalid_json_terminal(tmp_path: Path, _fake_key: None) -> None:
    paths = _write_inputs(tmp_path)
    calls: list[dict] = []
    script = [_response("{oops", "completed", _usage(100, 12))]
    report = _run_live(tmp_path, paths, script, calls)
    assert len(calls) == 1
    assert report["terminal_state"] == "invalid_json"


def test_empty_response_terminal(tmp_path: Path, _fake_key: None) -> None:
    paths = _write_inputs(tmp_path)
    calls: list[dict] = []
    script = [_response("   ", "completed", _usage(100, 4))]
    report = _run_live(tmp_path, paths, script, calls)
    assert report["terminal_state"] == "empty_output"


def test_model_abstention_terminal(tmp_path: Path, _fake_key: None) -> None:
    paths = _write_inputs(tmp_path)
    calls: list[dict] = []
    script = [_response(_readings_payload(["120", "120"], abstain=True),
                        "completed", _usage(100, 40))]
    report = _run_live(tmp_path, paths, script, calls)
    assert report["terminal_state"] == "abstained"
    assert report["reconciliation"]["counts"] == {
        "verification_abstained": 2}


def test_transport_failure_no_escalation(
        tmp_path: Path, _fake_key: None) -> None:
    paths = _write_inputs(tmp_path)
    calls: list[dict] = []
    script = [APIStatusError("too large", 413)]
    report = _run_live(tmp_path, paths, script, calls)
    assert len(calls) == 1
    assert report["terminal_state"] == "request_failed"
    attempt_path = (tmp_path / "executions" / "run_test" / "attempts"
                    / "evt_test_attempt_01.json")
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert attempt["transport_error"] == "PAYLOAD_TOO_LARGE"
    assert attempt["retry_triggered"] is False


def test_attempt_persistence_and_provenance(
        tmp_path: Path, _fake_key: None) -> None:
    paths = _write_inputs(tmp_path)
    calls: list[dict] = []
    script = [_response(_readings_payload(["120", "120"]), "completed",
                        _usage(100, 60))]
    report = _run_live(tmp_path, paths, script, calls)
    assert len(report["attempt_files"]) == 1
    attempt = json.loads(Path(report["attempt_files"][0]).read_text(
        encoding="utf-8"))
    assert attempt["attempt_number"] == 1
    assert attempt["max_output_tokens"] == 1024
    assert attempt["store"] is True
    assert attempt["prompt_hash"] == report["prompt_sha256"]
    expected = [hashlib.sha256(
        (paths["frames"] / name).read_bytes()).hexdigest()
        for name in ("frame_00000100.jpg", "frame_00000150.jpg")]
    assert attempt["input_image_hashes"] == expected
    assert report["inputs"][0]["mime"] == "image/jpeg"
    assert report["originals"][0]["values"][
        "scale_display_verbatim"] == "120"


def test_run_overwrite_protected(tmp_path: Path, _fake_key: None) -> None:
    paths = _write_inputs(tmp_path)
    script = [_response(_readings_payload(["120", "120"]), "completed",
                        _usage(100, 60))]
    _run_live(tmp_path, paths, list(script), [])
    sentinel = (tmp_path / "executions" / "run_test" / "sentinel.txt")
    sentinel.write_text("prior", encoding="utf-8")
    args = _base_args(paths) + ["--execute", "--run-id", "run_test"]
    code = execute.main(args, client_factory=_factory(list(script), []))
    assert code == 1
    assert sentinel.read_text(encoding="utf-8") == "prior"


def test_numerical_disagreement(tmp_path: Path, _fake_key: None) -> None:
    paths = _write_inputs(tmp_path)
    calls: list[dict] = []
    script = [_response(_readings_payload(["120", "121"]), "completed",
                        _usage(100, 60))]
    report = _run_live(tmp_path, paths, script, calls)
    assert report["reconciliation"]["counts"] == {
        "agreement": 1, "disagreement": 1}
    assert report["reconciliation"]["requires_human_review"] is True


def test_ambiguous_original_reading(tmp_path: Path, _fake_key: None) -> None:
    paths = _write_inputs(tmp_path, verbatims=("5 15", "120"))
    calls: list[dict] = []
    script = [_response(_readings_payload(["515", "120"]), "completed",
                        _usage(100, 60))]
    report = _run_live(tmp_path, paths, script, calls)
    assert report["reconciliation"]["counts"] == {
        "original_ambiguous": 1, "agreement": 1}


def test_reconcile_units_branches() -> None:
    assert execute.reconcile_units(None, None)[0] == "not_comparable"
    assert execute.reconcile_units("g", None)[0] == (
        "insufficient_evidence")
    assert execute.reconcile_units("g", "g")[0] == "agreement"
    assert execute.reconcile_units("g", "kg")[0] == "disagreement"


def test_evidence_separation(tmp_path: Path, _fake_key: None) -> None:
    paths = _write_inputs(tmp_path)
    before = {
        name: hashlib.sha256(
            (paths["obs"] / name).read_bytes()).hexdigest()
        for name in ("batch_test.json",)}
    calls: list[dict] = []
    script = [_response(_readings_payload(["120", "120"]), "completed",
                        _usage(100, 60))]
    report = _run_live(tmp_path, paths, script, calls)
    after = {
        name: hashlib.sha256(
            (paths["obs"] / name).read_bytes()).hexdigest()
        for name in ("batch_test.json",)}
    assert before == after
    assert report["event_id"] == "evt_test"
    assert "execution_report.json" not in str(paths["obs"])


def test_cost_ceiling_stops_escalation(
        tmp_path: Path, _fake_key: None) -> None:
    paths = _write_inputs(tmp_path)
    calls: list[dict] = []
    script = [
        _response("", "incomplete", _usage(100000, 1024),
                  _reason("max_output_tokens")),
        _response(_readings_payload(["120", "120"]), "completed",
                  _usage(100000, 60)),
    ]
    report = _run_live(tmp_path, paths, script, calls,
                       ["--cost-ceiling", "0.000001"])
    assert len(calls) == 1
    assert report["stopped_reason"] is not None
    assert "cost_ceiling_exceeded" in report["stopped_reason"]
    assert (tmp_path / "executions" / "run_test" / "attempts"
            / "evt_test_attempt_01.json").is_file()


def test_hard_ceiling_without_estimate_refuses_start(
        tmp_path: Path, _fake_key: None) -> None:
    paths = _write_inputs(tmp_path)
    calls: list[dict] = []
    args = _base_args(paths) + ["--execute", "--run-id", "run_test",
                                "--cost-ceiling", "1.0", "--hard-ceiling"]
    code = execute.main(args, client_factory=_factory(calls, calls))
    assert code == 1
    assert calls == []
    assert not (tmp_path / "executions" / "run_test").exists()


def test_no_automatic_batch_execution(
        tmp_path: Path, _fake_key: None) -> None:
    paths = _write_inputs(tmp_path)
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    second = dict(plan["entries"][0])
    second["event_id"] = "evt_other"
    plan["entries"].append(second)
    paths["plan"].write_text(json.dumps(plan), encoding="utf-8")
    calls: list[dict] = []
    script = [_response(_readings_payload(["120", "120"]), "completed",
                        _usage(100, 60))]
    report = _run_live(tmp_path, paths, script, calls)
    assert report["event_id"] == "evt_test"
    assert len(calls) == 1


def test_mixed_abstention_reconciled_per_reading(
        tmp_path: Path, _fake_key: None) -> None:
    """Abstention follows explicit readable/display flags per reading."""
    paths = _write_inputs(tmp_path)
    calls: list[dict] = []
    payload = json.dumps({
        "event_id": "evt_test",
        "readings": [
            {"timestamp_seconds": 2.0, "display_text": "120",
             "numeric_value": 120.0, "readable": True,
             "confidence": "high", "uncertainty_notes": None},
            {"timestamp_seconds": 3.0, "display_text": None,
             "numeric_value": None, "readable": False,
             "confidence": "low", "uncertainty_notes": "occluded"},
        ],
        "action_visible": "none",
        "overall_confidence": "medium",
        "uncertainty_notes": None,
    })
    script = [_response(payload, "completed", _usage(100, 60))]
    report = _run_live(tmp_path, paths, script, calls)
    assert report["terminal_state"] == "valid_prediction"
    assert report["reconciliation"]["counts"] == {
        "agreement": 1, "verification_abstained": 1}
    assert report["reconciliation"]["requires_human_review"] is True


def test_invalid_json_keeps_insufficient_evidence(
        tmp_path: Path, _fake_key: None) -> None:
    """Unparseable verification output is not an abstention."""
    paths = _write_inputs(tmp_path)
    calls: list[dict] = []
    script = [_response("{oops", "completed", _usage(100, 12))]
    report = _run_live(tmp_path, paths, script, calls)
    assert report["terminal_state"] == "invalid_json"
    assert report["reconciliation"]["counts"] == {
        "insufficient_evidence": 2}


def test_no_ground_truth_leakage_into_prompt(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, verbatims=("120", "121"))
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    entry = execute.find_event(plan, "evt_test")
    rows = execute.load_extraction_rows(paths["csv"])
    originals = execute.load_originals(paths["obs"])
    resolved = execute.resolve_frames(
        entry, rows, paths["csv"].parent, originals)
    usable = [item for item in resolved if item["file_exists"]]
    prompt_text, _ = execute.build_prompt(entry, usable)
    assert "120" not in prompt_text
    assert "121" not in prompt_text
    assert "515" not in prompt_text
    assert "150.000" not in prompt_text


def test_aggregate_spent_zero_requests() -> None:
    total, unknown = execute.aggregate_spent([])
    assert total == 0.0
    assert unknown == 0


def test_aggregate_spent_known_and_mixed() -> None:
    total, unknown = execute.aggregate_spent([0.01, 0.02])
    assert total == 0.03
    assert unknown == 0
    total, unknown = execute.aggregate_spent([0.01, None])
    assert total is None
    assert unknown == 1


def test_unknown_pricing_yields_unknown_total(
        tmp_path: Path, _fake_key: None) -> None:
    paths = _write_inputs(tmp_path)
    calls: list[dict] = []
    script = [_response(_readings_payload(["120", "120"]), "completed",
                        _usage(100, 60))]
    args = _base_args(paths) + ["--execute", "--run-id", "run_test"]
    code = execute.main(args, client_factory=_factory(script, calls))
    assert code == 0
    report_path = (tmp_path / "executions" / "run_test"
                   / "execution_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["spent_cost_usd_estimate"] is None
    assert report["spent_cost_usd_unknown_attempts"] == 1


def test_multiple_unknown_no_false_zero(
        tmp_path: Path, _fake_key: None) -> None:
    paths = _write_inputs(tmp_path)
    calls: list[dict] = []
    script = [
        _response("", "incomplete", _usage(100, 1024),
                  _reason("max_output_tokens")),
        _response(_readings_payload(["120", "120"]), "completed",
                  _usage(100, 60)),
    ]
    args = _base_args(paths) + ["--execute", "--run-id", "run_test"]
    code = execute.main(args, client_factory=_factory(script, calls))
    assert code == 0
    report_path = (tmp_path / "executions" / "run_test"
                   / "execution_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["spent_cost_usd_estimate"] is None
    assert report["spent_cost_usd_unknown_attempts"] == 2


def test_known_costs_sum_to_total(
        tmp_path: Path, _fake_key: None) -> None:
    paths = _write_inputs(tmp_path)
    calls: list[dict] = []
    script = [
        _response("", "incomplete", _usage(1000, 1024),
                  _reason("max_output_tokens")),
        _response(_readings_payload(["120", "120"]), "completed",
                  _usage(1000, 60)),
    ]
    report = _run_live(tmp_path, paths, script, calls)
    expected = round(2000 / 1e6 * 1.25 + 1084 / 1e6 * 4.25, 6)
    assert report["spent_cost_usd_estimate"] == expected
    assert report["spent_cost_usd_unknown_attempts"] == 0


def test_temporal_reason_retains_review_despite_agreement(
        tmp_path: Path, _fake_key: None) -> None:
    paths = _write_inputs(
        tmp_path, frames=(100, 150, 200),
        verbatims=("238", "238", "238"),
        reasons=("temporal_inconsistency",))
    calls: list[dict] = []
    script = [_response(
        _readings_payload(["238", "238", "238"]), "completed",
        _usage(100, 60))]
    report = _run_live(tmp_path, paths, script, calls)
    assert report["terminal_state"] == "valid_prediction"
    assert report["reconciliation"]["counts"] == {"agreement": 3}
    assert report["reconciliation"]["requires_human_review"] is True
    assert report["temporal_review"]["status"] == "not_evaluated"
    assert report["temporal_review"]["reasons"] == [
        "temporal_inconsistency"]
    assert report["review_reasons"] == ["temporal_inconsistency"]


def test_no_temporal_reasons_may_clear_review(
        tmp_path: Path, _fake_key: None) -> None:
    paths = _write_inputs(
        tmp_path, frames=(100, 150, 200),
        verbatims=("238", "238", "238"))
    calls: list[dict] = []
    script = [_response(
        _readings_payload(["238", "238", "238"]), "completed",
        _usage(100, 60))]
    report = _run_live(tmp_path, paths, script, calls)
    assert report["reconciliation"]["counts"] == {"agreement": 3}
    assert report["reconciliation"]["requires_human_review"] is False
    assert report["temporal_review"]["status"] == "not_applicable"


def _write_window_inputs(tmp_path: Path) -> dict[str, Path]:
    frames = tuple(7100 + 50 * step for step in range(12))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "plan_version": "1.0.0",
        "windows_in_scope": ["V01"],
        "entries": [{
            "event_id": "evt_window",
            "candidate_type": "scale_visibility",
            "start_timestamp_seconds": 142.0,
            "end_timestamp_seconds": 153.0,
            "review_reasons": ["ambiguous_parsing",
                               "temporal_inconsistency"],
            "uncertainty_notes": [],
            "reuse_frames": [{"frame_index": index,
                              "frame_available": True}
                             for index in frames],
            "token_budgets": [1024, 2048, 3072],
            "max_requests": 3,
        }],
        "summary": {},
    }), encoding="utf-8")
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    rows = ["frame_index,image_path,timestamp_seconds"]
    records = []
    for index in frames:
        name = f"frame_{index:08d}.jpg"
        (frames_dir / name).write_bytes(
            b"\xff\xd8win" + str(index).encode("utf-8"))
        rows.append(f"{index},{name},{index / 50.0}")
        records.append({
            "frame_index": index,
            "image_path": name,
            "timestamp_seconds": index / 50.0,
            "batch_id": "batch_test",
            "model_fields": {
                "scale_display_verbatim": "60",
                "scale_readable": True,
                "scale_state": "stable",
                "tare_action_visible": False,
                "addition_or_removal_visible": "none",
                "confidence": "high",
            },
        })
    csv_path = frames_dir / "manifest.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    obs_dir = tmp_path / "obs"
    obs_dir.mkdir()
    (obs_dir / "batch_test.json").write_text(
        json.dumps(records), encoding="utf-8")
    return {"plan": plan_path, "csv": csv_path, "obs": obs_dir,
            "frames": frames_dir}


def _window_args(paths: dict[str, Path], event_id: str,
                 start: str | None = None,
                 end: str | None = None) -> list[str]:
    args = ["--plan", str(paths["plan"]), "--event-id", event_id,
            "--obs-dir", str(paths["obs"]),
            "--extraction-manifest", str(paths["csv"]),
            "--out", str(paths["plan"].parent / "executions")]
    if start is not None:
        args.extend(["--window-start", start])
    if end is not None:
        args.extend(["--window-end", end])
    return args


def test_default_full_event_unchanged(tmp_path: Path, capsys: Any) -> None:
    paths = _write_inputs(tmp_path)
    code = execute.main(_base_args(paths) + ["--dry-run"])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["image_count"] == 2
    assert report["selection"]["temporal_subset"] is False
    assert report["selection"]["requested_start_seconds"] is None
    assert report["selection"]["selected_frame_indices"] == [100, 150]


def test_inclusive_window_selection(tmp_path: Path, capsys: Any) -> None:
    paths = _write_inputs(tmp_path)
    code = execute.main(
        _base_args(paths) + ["--dry-run", "--window-start", "2.0",
                             "--window-end", "3.0"])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["image_count"] == 2
    assert report["selection"]["temporal_subset"] is True


def test_five_frame_148_152_selection(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: Any) -> None:
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    paths = _write_window_inputs(tmp_path)
    code = execute.main(
        _window_args(paths, "evt_window", "148", "152") + ["--dry-run"])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["eligible"] is True
    assert report["image_count"] == 5
    assert [image["frame_index"] for image in report["images"]] == [
        7400, 7450, 7500, 7550, 7600]
    assert [image["timestamp_seconds"] for image in report["images"]] == [
        148.0, 149.0, 150.0, 151.0, 152.0]
    assert report["selection"]["selected_frame_indices"] == [
        7400, 7450, 7500, 7550, 7600]


def test_window_preserves_order_and_hashes(tmp_path: Path) -> None:
    plan = json.loads(
        (lambda paths: paths["plan"].read_text(encoding="utf-8"))(
            _write_window_inputs(tmp_path)))
    entry = execute.find_event(plan, "evt_window")
    entry["reuse_frames"] = list(reversed(entry["reuse_frames"]))
    csv_path = tmp_path / "frames" / "manifest.csv"
    obs_dir = tmp_path / "obs"
    rows = execute.load_extraction_rows(csv_path)
    originals = execute.load_originals(obs_dir)
    resolved = execute.resolve_frames(
        entry, rows, csv_path.parent, originals)
    selected = execute.select_window(resolved, 148.0, 152.0,
                                     "evt_window")
    assert [item["frame_index"] for item in selected] == [
        7400, 7450, 7500, 7550, 7600]
    for item in selected:
        raw = (csv_path.parent / f"frame_{item['frame_index']:08d}.jpg"
               ).read_bytes()
        assert item["sha256"] == hashlib.sha256(raw).hexdigest()


def test_partial_window_rejected(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    code = execute.main(
        _base_args(paths) + ["--dry-run", "--window-start", "2.0"])
    assert code == 1


def test_reversed_window_rejected(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    code = execute.main(
        _base_args(paths) + ["--dry-run", "--window-start", "3.0",
                             "--window-end", "2.0"])
    assert code == 1


def test_out_of_event_window_rejected(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    low = execute.main(
        _base_args(paths) + ["--dry-run", "--window-start", "1.0",
                             "--window-end", "2.5"])
    high = execute.main(
        _base_args(paths) + ["--dry-run", "--window-start", "2.5",
                             "--window-end", "4.0"])
    assert low == 1
    assert high == 1


def test_empty_window_rejected(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    code = execute.main(
        _base_args(paths) + ["--dry-run", "--window-start", "2.2",
                             "--window-end", "2.8"])
    assert code == 1


def test_invalid_window_values_rejected(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        execute.main(
            _base_args(paths) + ["--dry-run", "--window-start", "abc",
                                 "--window-end", "3.0"])
    assert exc_info.value.code == 2
    assert execute.main(
        _base_args(paths) + ["--dry-run", "--window-start", "nan",
                             "--window-end", "3.0"]) == 1
    assert execute.main(
        _base_args(paths) + ["--dry-run", "--window-start", "2.0",
                             "--window-end", "inf"]) == 1


def test_prompt_covers_selected_only(
        tmp_path: Path, _fake_key: None) -> None:
    paths = _write_inputs(tmp_path)
    calls: list[dict] = []
    script = [_response(_readings_payload(["120"]), "completed",
                        _usage(100, 60))]
    args = _base_args(paths) + ["--execute", "--run-id", "run_test",
                                "--window-start", "2.0",
                                "--window-end", "2.0"]
    code = execute.main(args, client_factory=_factory(script, calls))
    assert code == 0
    attempt_path = (tmp_path / "executions" / "run_test" / "attempts"
                    / "evt_test_attempt_01.json")
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert "2.000" in attempt["prompt_text"]
    assert "3.000" not in attempt["prompt_text"]
    assert len(attempt["input_image_hashes"]) == 1


def test_window_prompt_leak_free(tmp_path: Path, _fake_key: None) -> None:
    paths = _write_inputs(tmp_path, verbatims=("9 99", "120"))
    calls: list[dict] = []
    script = [_response(_readings_payload(["120"]), "completed",
                        _usage(100, 60))]
    args = _base_args(paths) + ["--execute", "--run-id", "run_test",
                                "--window-start", "2.0",
                                "--window-end", "2.0"]
    code = execute.main(args, client_factory=_factory(script, calls))
    assert code == 0
    attempt_path = (tmp_path / "executions" / "run_test" / "attempts"
                    / "evt_test_attempt_01.json")
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert "9 99" not in attempt["prompt_text"]
    assert "515" not in attempt["prompt_text"]


def test_reconciliation_subset_only(
        tmp_path: Path, _fake_key: None) -> None:
    paths = _write_inputs(tmp_path)
    calls: list[dict] = []
    script = [_response(_readings_payload(["120"]), "completed",
                        _usage(100, 60))]
    args = _base_args(paths) + ["--execute", "--run-id", "run_test",
                                "--window-start", "2.0",
                                "--window-end", "2.0"]
    code = execute.main(args, client_factory=_factory(script, calls))
    assert code == 0
    report_path = (tmp_path / "executions" / "run_test"
                   / "execution_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert [item["frame_index"]
            for item in report["reconciliation"]["per_frame"]] == [100]
    assert report["reconciliation"]["counts"] == {"agreement": 1}
    assert [item["frame_index"] for item in report["originals"]] == [100]
    assert report["selection"]["temporal_subset"] is True


def test_window_temporal_review_preserved(
        tmp_path: Path, _fake_key: None) -> None:
    paths = _write_inputs(tmp_path, reasons=("temporal_inconsistency",))
    calls: list[dict] = []
    script = [_response(_readings_payload(["120"]), "completed",
                        _usage(100, 60))]
    args = _base_args(paths) + ["--execute", "--run-id", "run_test",
                                "--window-start", "2.0",
                                "--window-end", "2.0"]
    code = execute.main(args, client_factory=_factory(script, calls))
    assert code == 0
    report_path = (tmp_path / "executions" / "run_test"
                   / "execution_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["review_reasons"] == ["temporal_inconsistency"]
    assert report["temporal_review"]["status"] == "not_evaluated"
    assert report["reconciliation"]["requires_human_review"] is True


def test_windowed_retry_intact(tmp_path: Path, _fake_key: None) -> None:
    paths = _write_inputs(tmp_path)
    calls: list[dict] = []
    script = [
        _response("", "incomplete", _usage(100, 1024),
                  _reason("max_output_tokens")),
        _response(_readings_payload(["120"]), "completed",
                  _usage(100, 60)),
    ]
    args = _base_args(paths) + ["--execute", "--run-id", "run_test",
                                "--window-start", "2.0",
                                "--window-end", "2.0"]
    code = execute.main(args, client_factory=_factory(script, calls))
    assert code == 0
    assert [call["max_output_tokens"] for call in calls] == [1024, 2048]
    report_path = (tmp_path / "executions" / "run_test"
                   / "execution_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["terminal_state"] == "valid_prediction"
