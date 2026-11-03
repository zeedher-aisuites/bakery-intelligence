"""Mocked tests for live recipe generation (no billable calls).

Fake provider clients record requests without network access.
Live mode is exercised only through injected fakes with
dummy credentials from the test environment.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("fastapi")

from bakery_intelligence.adversarial.live import (  # noqa: E402
    GenerateRecipeRequest,
    check_live_or_raise,
)
from bakery_intelligence.adversarial.service import (  # noqa: E402
    RecipeReviewService,
)
from bakery_intelligence.api.jobs import (  # noqa: E402
    JobService,
    WORKFLOW_STORAGE_SMOKE,
)
from bakery_intelligence.api.storage import (  # noqa: E402
    LocalStorageBackend,
)
from bakery_intelligence.api.store import SqliteJobStore  # noqa: E402


class _FakeMetaProposer:
    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[dict[str, Any]] = []
        self.last_request_id = "meta-req-1"

    def propose(self, evidence_text: str, image_paths: list[Any],
                api_key: str) -> tuple[Any, dict[str, Any]]:
        self.calls += 1
        assert api_key == "dummy-meta"
        assert len(image_paths) <= 8
        self.requests.append({"images": len(image_paths)})
        proposal = SimpleNamespace(
            steps=[{"description": "Add flour.",
                    "frame_refs": [0],
                    "start_seconds": 0.0,
                    "end_seconds": 1.0}],
            ingredients=[{"ingredient_id": "flour"}],
            model_dump=lambda mode="json": {"steps": [], "ingredients": []},
        )
        return proposal, {"input_tokens": 1000, "output_tokens": 100,
                          "reasoning_tokens": None}


class _FakeBasetenChallenger:
    def __init__(self, rounds: list[list[dict[str, Any]]]) -> None:
        self._rounds = rounds
        self.calls = 0
        self.last_request_id = "baseten-req-1"

    def critique(self, claims: list[dict[str, Any]],
                 evidence_summary: dict[str, Any], api_key: str,
                 retries: int = 1) -> tuple[list[dict[str, Any]],
                                            dict[str, Any]]:
        self.calls += 1
        assert api_key == "dummy-baseten"
        assert "model" not in json.dumps(evidence_summary)
        assert "video_bytes" not in json.dumps(evidence_summary)
        objections = (self._rounds.pop(0) if self._rounds else [])
        return objections, {"input_tokens": 500, "output_tokens": 50,
                            "reasoning_tokens": None}


def _stack(tmp_path: Path):
    storage = LocalStorageBackend(tmp_path / "storage")
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    service = JobService(store)
    record = service.submit(
        WORKFLOW_STORAGE_SMOKE, source_object_id="clip.mp4")
    storage.store_source("clip.mp4", io.BytesIO(b"fake"))
    store.claim_next("worker-1", "2026-11-01T00:00:00+00:00")
    artifacts = {
        "measurements.json": {
            "window": {"start_seconds": 0.0, "end_seconds": 2.0},
            "measurements": [
                {"measurement_id": "m_0", "frame_index": 0,
                 "timestamp_seconds": 0.0, "value": 0.0,
                 "value_status": "ok", "value_source": "firstpass",
                 "unit_observed": "g"},
                {"measurement_id": "m_50", "frame_index": 50,
                 "timestamp_seconds": 1.0, "value": 12.0,
                 "value_status": "ok", "value_source": "firstpass",
                 "unit_observed": "g"},
            ],
            "action_evidence": {"addition_frames": [50],
                                "tare_frames": []}},
        "pipeline_summary.json": {
            "video": {"file_name": "clip.mp4", "duration_seconds": 2.0},
            "source_sha256": "a" * 64},
        "firstpass_report.json": {"model": "muse-spark-1.3"},
        "recipe_draft.json": {"draft_id": "draft_test",
                              "unresolved_questions": []},
        "weighing_event.json": {"ingredient_id": None,
                                "verification_status": "confirmed"},
    }
    digests = {}
    for artifact_id, payload in artifacts.items():
        digests[artifact_id] = storage.store_artifact(
            record.run_id, artifact_id, json.dumps(payload).encode())
    store.complete(record.run_id, "worker-1",
                   "2026-11-01T00:01:00+00:00", "a" * 64, digests)
    return storage, service.get(record.run_id)


def _live_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BAKERY_LIVE_RECIPE_GEN", "1")
    monkeypatch.setenv("MODEL_API_KEY", "dummy-meta")
    monkeypatch.setenv("BASETEN_API_KEY", "dummy-baseten")
    monkeypatch.delenv("BAKERY_BASETEN_MODEL", raising=False)


def test_evidence_prefers_action_moments_with_coverage() -> None:
    from bakery_intelligence.adversarial.live import choose_evidence_frames
    frames = [{"frame_index": index, "timestamp_seconds": float(index)}
              for index in range(10)]
    chosen = choose_evidence_frames(frames, {3, 7}, 4)
    assert [frame["frame_index"] for frame in chosen] == [3, 7, 0, 5]
    assert len(choose_evidence_frames(frames, set(range(10)), 4)) == 4
    assert choose_evidence_frames(frames, {3, 7}, 4) == chosen


def test_live_defaults_to_simulation_without_calls(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _live_env(monkeypatch)
    storage, record = _stack(tmp_path)
    assert record is not None
    state = RecipeReviewService(storage).generate_recipe(
        record, GenerateRecipeRequest(),
        proposer=_FakeMetaProposer(),
        challenger=_FakeBasetenChallenger([]))
    assert state["generation"]["mode"] == "simulation"
    assert state["generation"]["simulated"] is True


def test_live_requires_explicit_confirmation(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _live_env(monkeypatch)
    storage, record = _stack(tmp_path)
    assert record is not None
    with pytest.raises(Exception, match="confirmation"):
        RecipeReviewService(storage).generate_recipe(
            record, GenerateRecipeRequest(mode="live"),
            proposer=_FakeMetaProposer(),
            challenger=_FakeBasetenChallenger([]))


def test_live_generation_records_costs_and_roles(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _live_env(monkeypatch)
    storage, record = _stack(tmp_path)
    assert record is not None
    proposer = _FakeMetaProposer()
    challenger = _FakeBasetenChallenger([[], []])
    service = RecipeReviewService(storage)
    state = service.generate_recipe(
        record, GenerateRecipeRequest(
            mode="live", confirmed=True, max_rounds=2),
        proposer=proposer, challenger=challenger)
    generation = state["generation"]
    assert generation["simulated"] is False
    assert generation["providers"][0]["model"] == "muse-spark-1.3"
    assert generation["providers"][1]["model"] == (
        "meta-models/muse-spark-open-1026")
    assert generation["providers"][1]["input_modality"] == (
        "text-evidence-only")
    operations = [item["operation"] for item in generation["costs"]]
    assert operations[0] == "recipe-proposal"
    assert generation["costs"][0]["estimated_cost_usd"] == round(
        1000 * 1.25 / 1e6 + 100 * 4.25 / 1e6, 4)
    assert generation["costs"][1]["cost_status"] == "unknown"
    assert generation["estimated_total_cost_usd"] == round(
        1000 * 1.25 / 1e6 + 100 * 4.25 / 1e6, 4)
    assert proposer.calls == 1 and challenger.calls == 1
    assert state["claims"]


def test_live_idempotent_without_new_provider_calls(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _live_env(monkeypatch)
    storage, record = _stack(tmp_path)
    assert record is not None
    service = RecipeReviewService(storage)
    request = GenerateRecipeRequest(
        mode="live", confirmed=True, max_rounds=1)
    proposer = _FakeMetaProposer()
    challenger = _FakeBasetenChallenger([[]])
    first = service.generate_recipe(
        record, request, proposer=proposer, challenger=challenger)
    second = service.generate_recipe(
        record, request, proposer=proposer, challenger=challenger)
    assert (first["generation"]["consolidated_hash"]
            == second["generation"]["consolidated_hash"])
    assert proposer.calls == 1 and challenger.calls == 1


def test_live_provider_error_never_falls_back(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _live_env(monkeypatch)
    storage, record = _stack(tmp_path)
    assert record is not None

    class _Failing:
        def critique(self, claims: Any, evidence_summary: Any,
                     api_key: str, retries: int = 1) -> Any:
            raise RuntimeError("baseten down")

    service = RecipeReviewService(storage)
    with pytest.raises(RuntimeError, match="baseten down"):
        service.generate_recipe(
            record, GenerateRecipeRequest(
                mode="live", confirmed=True, max_rounds=1),
            proposer=_FakeMetaProposer(), challenger=_Failing())
    assert "generation" not in (service.store.load(record.run_id) or {})


def test_live_blocked_when_flag_disabled(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Set (not merely delete) the flag: live_available() loads the
    # local .env for missing vars, so delenv alone still sees an
    # enabled flag on machines with live credentials configured.
    monkeypatch.setenv("BAKERY_LIVE_RECIPE_GEN", "0")
    with pytest.raises(Exception, match="disabled"):
        check_live_or_raise(
            GenerateRecipeRequest(mode="live", confirmed=True))


def _recipe_payload() -> dict[str, Any]:
    return {
        "recipe_name": "Test loaf",
        "ingredients": [],
        "steps": [{"description": "Mix.",
                   "frame_refs": [],
                   "provenance": "ai-extracted"}],
        "unresolved_questions": [],
    }


class _InferenceResponseClient:
    """Fake Meta client returning the real InferenceResponse shape."""

    def __init__(self, response: Any) -> None:
        self._response = response
        self.requests: list[dict[str, Any]] = []

    def submit(self, prompt_text: str, image_paths: Any, api_key: str,
               max_output_tokens: int) -> Any:
        self.requests.append({
            "prompt_text": prompt_text,
            "image_paths": list(image_paths),
            "api_key": api_key,
            "max_output_tokens": max_output_tokens,
        })
        return self._response


def test_meta_proposer_uses_inference_response_contract() -> None:
    from bakery_intelligence.adversarial.live import MetaSparkLiveProposer
    from bakery_intelligence.inference.firstpass import (
        InferenceResponse,
        InferenceUsage,
    )

    response = InferenceResponse(
        model="muse-spark-1.3",
        status="completed",
        output_text=json.dumps(_recipe_payload()),
        usage=InferenceUsage(
            input_tokens=1000, output_tokens=100,
            reasoning_output_tokens=None),
        latency_seconds=0.1,
        request_id="resp-live-1",
        incomplete_reason=None,
    )
    proposer = MetaSparkLiveProposer(
        client=_InferenceResponseClient(response))
    proposal, usage = proposer.propose("evidence", ["frame.jpg"], "key")
    assert proposal.recipe_name == "Test loaf"
    assert usage == {"input_tokens": 1000, "output_tokens": 100,
                     "reasoning_tokens": None}
    assert proposer.last_request_id == "resp-live-1"
    assert proposer.last_usage == usage
    assert proposer.last_model == "muse-spark-1.3"


def test_meta_proposer_handles_missing_usage_gracefully() -> None:
    from bakery_intelligence.adversarial.live import (
        MetaSparkLiveProposer,
        cost_entry,
    )
    from bakery_intelligence.inference.firstpass import (
        InferenceResponse,
        InferenceUsage,
    )

    response = InferenceResponse(
        model="muse-spark-1.3",
        status="completed",
        output_text=json.dumps(_recipe_payload()),
        usage=InferenceUsage(),
        latency_seconds=0.0,
        request_id=None,
        incomplete_reason=None,
    )
    proposer = MetaSparkLiveProposer(
        client=_InferenceResponseClient(response))
    _, usage = proposer.propose("evidence", [], "key")
    assert usage == {"input_tokens": None, "output_tokens": None,
                     "reasoning_tokens": None}
    assert proposer.last_request_id is None
    entry = cost_entry(
        provider="meta", model="muse-spark-1.3",
        operation="recipe-proposal", usage=usage,
        rates=(1.25, 4.25), rates_source="test",
        request_id=proposer.last_request_id, status="ok")
    assert entry["cost_status"] == "unknown"
    assert entry["estimated_cost_usd"] is None


def test_meta_proposer_preserves_request_id_on_parse_failure() -> None:
    from bakery_intelligence.adversarial.live import (
        LiveConfigError,
        MetaSparkLiveProposer,
    )
    from bakery_intelligence.inference.firstpass import (
        InferenceResponse,
        InferenceUsage,
    )

    response = InferenceResponse(
        model="muse-spark-1.3",
        status="completed",
        output_text="not json",
        usage=InferenceUsage(input_tokens=10, output_tokens=5),
        latency_seconds=0.0,
        request_id="resp-billable-9",
        incomplete_reason=None,
    )
    proposer = MetaSparkLiveProposer(
        client=_InferenceResponseClient(response))
    with pytest.raises(LiveConfigError, match="resp-billable-9"):
        proposer.propose("evidence", [], "key")
    assert proposer.last_request_id == "resp-billable-9"
    assert proposer.last_usage == {"input_tokens": 10,
                                   "output_tokens": 5,
                                   "reasoning_tokens": None}


def test_meta_proposer_preserves_request_id_on_validation_failure() -> None:
    from bakery_intelligence.adversarial.live import (
        LiveConfigError,
        MetaSparkLiveProposer,
    )
    from bakery_intelligence.inference.firstpass import (
        InferenceResponse,
        InferenceUsage,
    )

    response = InferenceResponse(
        model="muse-spark-1.3",
        status="completed",
        output_text=json.dumps({"recipe_name": 123}),
        usage=InferenceUsage(input_tokens=10, output_tokens=5),
        latency_seconds=0.0,
        request_id="resp-billable-10",
        incomplete_reason=None,
    )
    proposer = MetaSparkLiveProposer(
        client=_InferenceResponseClient(response))
    with pytest.raises(LiveConfigError, match="resp-billable-10"):
        proposer.propose("evidence", [], "key")
    assert proposer.last_request_id == "resp-billable-10"


class _SchemaAwareClient:
    """Fake Meta client accepting the recipe schema contract."""

    def __init__(self, response: Any) -> None:
        self._response = response
        self.requests: list[dict[str, Any]] = []

    def submit(self, prompt_text: str, image_paths: Any, api_key: str,
               max_output_tokens: int,
               response_schema: Any | None = None,
               response_schema_name: Any | None = None) -> Any:
        self.requests.append({
            "prompt_text": prompt_text,
            "image_paths": list(image_paths),
            "api_key": api_key,
            "max_output_tokens": max_output_tokens,
            "response_schema": response_schema,
            "response_schema_name": response_schema_name,
        })
        return self._response


def _inference_response(**overrides: Any) -> Any:
    from bakery_intelligence.inference.firstpass import (
        InferenceResponse,
        InferenceUsage,
    )

    payload: dict[str, Any] = {
        "model": "muse-spark-1.3",
        "status": "completed",
        "output_text": json.dumps(_recipe_payload()),
        "usage": InferenceUsage(
            input_tokens=1000, output_tokens=100),
        "latency_seconds": 0.0,
        "request_id": "resp-test-1",
        "incomplete_reason": None,
    }
    payload.update(overrides)
    return InferenceResponse(**payload)


def test_recipe_schema_sent_and_tokens_honored() -> None:
    from bakery_intelligence.adversarial.live import (
        MetaSparkLiveProposer,
        RECIPE_JSON_SCHEMA,
        RECIPE_SCHEMA_NAME,
    )

    client = _SchemaAwareClient(_inference_response())
    proposer = MetaSparkLiveProposer(
        client=client, max_output_tokens=2048)
    _, usage = proposer.propose("evidence", ["frame.jpg"], "key")
    assert usage["input_tokens"] == 1000
    (call,) = client.requests
    assert call["max_output_tokens"] == 2048
    assert call["response_schema"] == RECIPE_JSON_SCHEMA
    assert call["response_schema_name"] == RECIPE_SCHEMA_NAME
    assert "compact JSON object" in call["prompt_text"]
    assert proposer.calls == 1


def test_empty_output_completed_reports_empty() -> None:
    from bakery_intelligence.adversarial.live import (
        LiveConfigError,
        MetaSparkLiveProposer,
    )
    from bakery_intelligence.inference.firstpass import InferenceUsage

    proposer = MetaSparkLiveProposer(
        client=_SchemaAwareClient(_inference_response(
            output_text="", status="completed",
            incomplete_reason=None,
            usage=InferenceUsage(
                input_tokens=2000, output_tokens=10),
            request_id="resp-empty-1")))
    with pytest.raises(LiveConfigError, match="empty output"):
        proposer.propose("evidence", [], "key")
    assert proposer.last_request_id == "resp-empty-1"
    assert proposer.last_status == "completed"
    assert proposer.last_output_length == 0
    assert proposer.calls == 1


def test_reasoning_only_incomplete_reports_incomplete() -> None:
    from bakery_intelligence.adversarial.live import (
        LiveConfigError,
        MetaSparkLiveProposer,
    )
    from bakery_intelligence.inference.firstpass import InferenceUsage

    proposer = MetaSparkLiveProposer(
        client=_SchemaAwareClient(_inference_response(
            output_text="", status="incomplete",
            incomplete_reason="max_output_tokens",
            usage=InferenceUsage(
                input_tokens=5000, output_tokens=2048,
                reasoning_output_tokens=2048),
            request_id="resp-reason-1")))
    with pytest.raises(LiveConfigError, match="incomplete"):
        proposer.propose("evidence", [], "key")
    assert proposer.last_request_id == "resp-reason-1"
    assert proposer.last_status == "incomplete"
    assert proposer.last_incomplete_reason == "max_output_tokens"
    assert proposer.last_usage["reasoning_tokens"] == 2048
    assert proposer.calls == 1


def test_incomplete_partial_json_reports_not_completed() -> None:
    from bakery_intelligence.adversarial.live import (
        LiveConfigError,
        MetaSparkLiveProposer,
    )

    proposer = MetaSparkLiveProposer(
        client=_SchemaAwareClient(_inference_response(
            output_text='{"ingredients": [',
            status="incomplete",
            incomplete_reason="max_output_tokens",
            request_id="resp-partial-1")))
    with pytest.raises(LiveConfigError, match="not completed"):
        proposer.propose("evidence", [], "key")
    assert proposer.last_incomplete_reason == "max_output_tokens"
    assert proposer.calls == 1


def test_malformed_json_reports_request_id() -> None:
    from bakery_intelligence.adversarial.live import (
        LiveConfigError,
        MetaSparkLiveProposer,
    )

    proposer = MetaSparkLiveProposer(
        client=_SchemaAwareClient(_inference_response(
            output_text="{not json",
            status="completed",
            request_id="resp-malformed-1")))
    with pytest.raises(LiveConfigError, match="resp-malformed-1"):
        proposer.propose("evidence", [], "key")
    assert proposer.calls == 1


def test_valid_structured_recipe_output_succeeds() -> None:
    from bakery_intelligence.adversarial.live import MetaSparkLiveProposer

    proposer = MetaSparkLiveProposer(
        client=_SchemaAwareClient(_inference_response(
            request_id="resp-valid-1")))
    proposal, _ = proposer.propose("evidence", [], "key")
    assert proposal.recipe_name == "Test loaf"
    assert proposer.last_status == "completed"
    assert proposer.last_request_id == "resp-valid-1"


def test_recipe_token_budget_default() -> None:
    from bakery_intelligence.adversarial.live import (
        DEFAULT_MAX_OUTPUT_TOKENS,
        GenerateRecipeRequest,
        MAX_OUTPUT_TOKENS_LIMIT,
        MetaSparkLiveProposer,
    )

    assert DEFAULT_MAX_OUTPUT_TOKENS == 50000
    assert MAX_OUTPUT_TOKENS_LIMIT == 50000
    assert GenerateRecipeRequest().max_output_tokens == 50000
    assert MetaSparkLiveProposer().max_output_tokens == 50000


def test_live_output_budget_50000_accepted_and_bounded(
        monkeypatch: pytest.MonkeyPatch) -> None:
    _live_env(monkeypatch)
    check_live_or_raise(GenerateRecipeRequest(
        mode="live", confirmed=True, max_output_tokens=50000))
    check_live_or_raise(GenerateRecipeRequest(
        mode="live", confirmed=True))
    with pytest.raises(Exception, match="between 1 and 50000"):
        check_live_or_raise(GenerateRecipeRequest(
            mode="live", confirmed=True, max_output_tokens=50001))
    with pytest.raises(Exception, match="between 1 and 50000"):
        check_live_or_raise(GenerateRecipeRequest(
            mode="live", confirmed=True, max_output_tokens=0))


def test_proposer_sends_configured_50000_budget() -> None:
    from bakery_intelligence.adversarial.live import MetaSparkLiveProposer

    client = _SchemaAwareClient(_inference_response())
    proposer = MetaSparkLiveProposer(client=client)
    proposer.propose("evidence", ["frame.jpg"], "key")
    (call,) = client.requests
    assert call["max_output_tokens"] == 50000


def test_reasoning_exhaustion_reports_incomplete_without_retry() -> None:
    from bakery_intelligence.adversarial.live import (
        LiveConfigError,
        MetaSparkLiveProposer,
    )
    from bakery_intelligence.inference.firstpass import InferenceUsage

    proposer = MetaSparkLiveProposer(
        client=_SchemaAwareClient(_inference_response(
            output_text="", status="incomplete",
            incomplete_reason="max_output_tokens",
            usage=InferenceUsage(
                input_tokens=22057, output_tokens=2048,
                reasoning_output_tokens=2046),
            request_id="resp-exhausted-1")))
    with pytest.raises(LiveConfigError, match="incomplete"):
        proposer.propose("evidence", [], "key")
    assert proposer.last_request_id == "resp-exhausted-1"
    assert proposer.last_status == "incomplete"
    assert proposer.last_incomplete_reason == "max_output_tokens"
    assert proposer.last_output_length == 0
    assert proposer.last_usage == {
        "input_tokens": 22057, "output_tokens": 2048,
        "reasoning_tokens": 2046}
    assert proposer.calls == 1


def test_proposer_failure_persisted_without_retry(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bakery_intelligence.adversarial.live import (
        LiveConfigError,
        MetaSparkLiveProposer,
    )

    _live_env(monkeypatch)
    storage, record = _stack(tmp_path)
    assert record is not None
    proposer = MetaSparkLiveProposer(
        client=_SchemaAwareClient(_inference_response(
            output_text="", status="incomplete",
            incomplete_reason="max_output_tokens",
            request_id="resp-billable-99")))
    service = RecipeReviewService(storage)
    with pytest.raises(LiveConfigError, match="resp-billable-99"):
        service.generate_recipe(
            record, GenerateRecipeRequest(
                mode="live", confirmed=True, max_rounds=1),
            proposer=proposer,
            challenger=_FakeBasetenChallenger([]))
    assert proposer.calls == 1
    stored = service.store.load(record.run_id) or {}
    assert "generation" not in stored
    attempts = stored.get("live_attempts") or []
    assert len(attempts) == 1
    failure = attempts[0]
    assert failure["request_id"] == "resp-billable-99"
    assert failure["operation"] == "recipe-proposal"
    assert failure["status"] == "incomplete"
    assert failure["incomplete_reason"] == "max_output_tokens"
    assert failure["output_length"] == 0
    assert "dummy-meta" not in json.dumps(failure)


def _chat_message_response(content: Any, finish_reason: Any = "stop",
                           request_id: Any = "chatcmpl-test",
                           usage: Any = "default") -> Any:
    if usage == "default":
        usage = SimpleNamespace(model_dump=lambda: {
            "prompt_tokens": 120, "completion_tokens": 40,
            "total_tokens": 160})
    return SimpleNamespace(
        id=request_id,
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content),
            finish_reason=finish_reason)],
        usage=usage,
    )


class _FakeChatClient:
    """Fake Chat Completions client returning scripted responses."""

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.requests: list[dict[str, Any]] = []

    @property
    def chat(self) -> Any:
        return SimpleNamespace(completions=SimpleNamespace(
            create=self.create))

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        action = self._script.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def _objections_payload() -> dict[str, Any]:
    return {"objections": [{
        "claim_id": "claim_1", "objection": "Tare is ambiguous.",
        "contradiction": False}]}


def test_challenger_parses_fenced_json() -> None:
    from bakery_intelligence.adversarial.live import BasetenLiveChallenger

    fenced = "```json\n" + json.dumps(_objections_payload()) + "\n```"
    challenger = BasetenLiveChallenger(client=_FakeChatClient(
        [_chat_message_response(fenced)]))
    objections, usage = challenger.critique(
        [{"claim_id": "claim_1"}], {"run_id": "r"}, "key")
    assert objections[0]["claim_id"] == "claim_1"
    assert usage["input_tokens"] == 120
    assert challenger.last_finish_reason == "stop"
    assert challenger.last_request_id == "chatcmpl-test"


def test_challenger_parses_prose_wrapped_json() -> None:
    from bakery_intelligence.adversarial.live import BasetenLiveChallenger

    wrapped = ("Here is the critique:\n"
               + json.dumps(_objections_payload())
               + "\nLet me know if you need more.")
    challenger = BasetenLiveChallenger(client=_FakeChatClient(
        [_chat_message_response(wrapped)]))
    objections, _ = challenger.critique(
        [{"claim_id": "claim_1"}], {"run_id": "r"}, "key")
    assert objections[0]["objection"] == "Tare is ambiguous."


def test_challenger_empty_output_reports_category() -> None:
    from bakery_intelligence.adversarial.live import (
        BasetenLiveChallenger,
        LiveConfigError,
    )

    challenger = BasetenLiveChallenger(client=_FakeChatClient(
        [_chat_message_response("   ", finish_reason="stop",
                                request_id="chatcmpl-empty")]))
    with pytest.raises(LiveConfigError, match="failure_category=empty"):
        challenger.critique([{"claim_id": "c"}], {}, "key")
    assert challenger.last_request_id == "chatcmpl-empty"
    assert challenger.last_content_length == 3
    assert challenger.calls == 1


def test_challenger_truncated_output_reports_category() -> None:
    from bakery_intelligence.adversarial.live import (
        BasetenLiveChallenger,
        LiveConfigError,
    )

    challenger = BasetenLiveChallenger(client=_FakeChatClient(
        [_chat_message_response('{"objections": [',
                                finish_reason="length",
                                request_id="chatcmpl-trunc")]))
    with pytest.raises(LiveConfigError, match="failure_category=truncated"):
        challenger.critique([{"claim_id": "c"}], {}, "key")
    assert challenger.last_finish_reason == "length"


def test_challenger_non_json_reports_category() -> None:
    from bakery_intelligence.adversarial.live import (
        BasetenLiveChallenger,
        LiveConfigError,
    )

    challenger = BasetenLiveChallenger(client=_FakeChatClient(
        [_chat_message_response("The scale looks fine today.",
                                request_id="chatcmpl-prose")]))
    with pytest.raises(LiveConfigError, match="failure_category=non_json"):
        challenger.critique([{"claim_id": "c"}], {}, "key")
    assert challenger.last_request_id == "chatcmpl-prose"


def test_challenger_missing_and_invalid_objections_rejected() -> None:
    from bakery_intelligence.adversarial.live import (
        BasetenLiveChallenger,
        LiveConfigError,
    )

    missing = BasetenLiveChallenger(client=_FakeChatClient(
        [_chat_message_response(json.dumps({"verdict": "ok"}))]))
    with pytest.raises(LiveConfigError,
                       match="failure_category=missing_objections"):
        missing.critique([{"claim_id": "c"}], {}, "key")

    invalid = BasetenLiveChallenger(client=_FakeChatClient(
        [_chat_message_response(json.dumps(
            {"objections": [{"claim_id": "c"}]}))]))
    with pytest.raises(LiveConfigError,
                       match="failure_category=invalid_objections"):
        invalid.critique([{"claim_id": "c"}], {}, "key")


def test_challenger_request_uses_exact_model_without_reasoning() -> None:
    from bakery_intelligence.adversarial.live import BasetenLiveChallenger

    client = _FakeChatClient([_chat_message_response(
        json.dumps(_objections_payload()))])
    challenger = BasetenLiveChallenger(client=client)
    challenger.critique([{"claim_id": "claim_1"}], {"run_id": "r"}, "key")
    (request,) = client.requests
    assert request["model"] == "meta-models/muse-spark-open-1026"
    assert request["max_tokens"] == 4096
    assert set(request) == {"model", "messages", "max_tokens"}
    assert "reasoning_effort" not in request
    assert "reasoning" not in request
    assert isinstance(request["messages"], list)


def test_challenger_handles_dict_and_missing_usage() -> None:
    from bakery_intelligence.adversarial.live import BasetenLiveChallenger

    as_dict = BasetenLiveChallenger(client=_FakeChatClient(
        [_chat_message_response(
            json.dumps(_objections_payload()),
            usage={"prompt_tokens": 7, "completion_tokens": 3,
                   "total_tokens": 10})]))
    _, usage = as_dict.critique([{"claim_id": "claim_1"}], {}, "key")
    assert usage["input_tokens"] == 7
    assert usage["output_tokens"] == 3

    missing = BasetenLiveChallenger(client=_FakeChatClient(
        [_chat_message_response(
            json.dumps(_objections_payload()), usage=None)]))
    _, empty_usage = missing.critique([{"claim_id": "claim_1"}], {}, "key")
    assert empty_usage == {"input_tokens": None, "output_tokens": None,
                           "reasoning_tokens": None}


def test_challenger_failure_preserves_proposal_and_resumes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bakery_intelligence.adversarial.live import (
        BasetenLiveChallenger,
        LiveConfigError,
        MetaSparkLiveProposer,
    )

    _live_env(monkeypatch)
    storage, record = _stack(tmp_path)
    assert record is not None
    proposer = MetaSparkLiveProposer(
        client=_SchemaAwareClient(_inference_response(
            request_id="resp-resume-proposal")))
    failing = BasetenLiveChallenger(client=_FakeChatClient(
        [_chat_message_response("not json",
                                request_id="chatcmpl-fail-1")]))
    service = RecipeReviewService(storage)
    with pytest.raises(LiveConfigError, match="failure_category=non_json"):
        service.generate_recipe(
            record, GenerateRecipeRequest(
                mode="live", confirmed=True, max_rounds=1),
            proposer=proposer, challenger=failing)
    assert proposer.calls == 1
    assert failing.calls == 1
    stored = service.store.load(record.run_id) or {}
    assert "generation" not in stored
    assert stored["live_proposal"]["proposal"]["ingredients"] == []
    attempts = stored.get("live_attempts") or []
    assert len(attempts) == 1
    assert attempts[0]["request_id"] == "chatcmpl-fail-1"
    assert attempts[0]["failure_category"] == "non_json"
    assert attempts[0]["finish_reason"] == "stop"
    assert "dummy-baseten" not in json.dumps(attempts[0])
    assert "dummy-meta" not in json.dumps(stored)

    working = BasetenLiveChallenger(client=_FakeChatClient(
        [_chat_message_response(
            json.dumps({"objections": []}),
            request_id="chatcmpl-ok-2")]))
    state = service.generate_recipe(
        record, GenerateRecipeRequest(
            mode="live", confirmed=True, max_rounds=1),
        proposer=proposer, challenger=working)
    assert proposer.calls == 1
    assert working.calls == 1
    assert state["generation"]["simulated"] is False
    assert state["generation"]["proposal"]["ingredients"] == []
    stored_after = service.store.load(record.run_id) or {}
    assert "live_proposal" not in stored_after
    assert len(stored_after.get("live_attempts") or []) == 1


def test_challenger_budget_4096_and_resume_reuses_proposal(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from bakery_intelligence.adversarial.live import (
        DEFAULT_MAX_OUTPUT_TOKENS,
        BasetenLiveChallenger,
        LiveConfigError,
        MetaSparkLiveProposer,
    )

    assert BasetenLiveChallenger().max_tokens == 4096
    assert MetaSparkLiveProposer().max_output_tokens == 50000
    assert DEFAULT_MAX_OUTPUT_TOKENS == 50000

    _live_env(monkeypatch)
    storage, record = _stack(tmp_path)
    assert record is not None
    proposer = MetaSparkLiveProposer(
        client=_SchemaAwareClient(_inference_response(
            request_id="resp-budget-resume")))
    truncated = BasetenLiveChallenger(client=_FakeChatClient(
        [_chat_message_response(
            "", finish_reason="length", request_id="chatcmpl-budget-1",
            usage=SimpleNamespace(model_dump=lambda: {
                "prompt_tokens": 1814, "completion_tokens": 1024,
                "total_tokens": 2838}))]))
    service = RecipeReviewService(storage)
    with pytest.raises(LiveConfigError, match="failure_category=truncated"):
        service.generate_recipe(
            record, GenerateRecipeRequest(
                mode="live", confirmed=True, max_rounds=1),
            proposer=proposer, challenger=truncated)
    assert proposer.calls == 1
    stored = service.store.load(record.run_id) or {}
    assert "generation" not in stored
    assert stored["live_proposal"]["proposal"]["ingredients"] == []

    working = BasetenLiveChallenger(client=_FakeChatClient(
        [_chat_message_response(
            json.dumps({"objections": []}),
            request_id="chatcmpl-budget-2")]))
    state = service.generate_recipe(
        record, GenerateRecipeRequest(
            mode="live", confirmed=True, max_rounds=1),
        proposer=proposer, challenger=working)
    assert proposer.calls == 1
    assert working.calls == 1
    (request,) = working._client.requests
    assert request["max_tokens"] == 4096
    assert state["generation"]["simulated"] is False
    stored_after = service.store.load(record.run_id) or {}
    assert "live_proposal" not in stored_after
    attempts = stored_after.get("live_attempts") or []
    assert len(attempts) == 1
    assert attempts[0]["failure_category"] == "truncated"
    assert attempts[0]["request_id"] == "chatcmpl-budget-1"
