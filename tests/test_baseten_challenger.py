"""Offline tests for the Baseten challenger adapter.

A fake Chat Completions client records every request; no live
Baseten call is made. Verifies the exact authorized model ID,
the absence of unverified reasoning parameters, and usage
provenance recording.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from bakery_intelligence.adversarial.baseten_challenger import (
    BASETEN_BASE_URL,
    BASETEN_CHALLENGER_MODEL,
    BasetenChallenger,
)


class _FakeCompletions:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        payload = {"objections": [{
            "claim_id": "claim_1", "objection": "Tare is ambiguous.",
            "contradiction": False}]}
        return SimpleNamespace(
            id="chatcmpl-fake-1",
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(payload)),
                finish_reason="stop")],
            usage=SimpleNamespace(model_dump=lambda: {
                "prompt_tokens": 100, "completion_tokens": 20,
                "total_tokens": 120}),
        )


def test_baseten_challenger_uses_exact_model_without_reasoning() -> None:
    completions = _FakeCompletions()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions))
    challenger = BasetenChallenger(client=client)
    objections = challenger.critique(
        [{"claim_id": "claim_1", "title": "Flour 200 g"}],
        {"run_id": "run_0123456789ab", "frames": []})
    assert BASETEN_BASE_URL == "https://inference.baseten.co/v1"
    assert BASETEN_CHALLENGER_MODEL == "meta-models/muse-spark-open-1026"
    request = completions.requests[0]
    assert request["model"] == "meta-models/muse-spark-open-1026"
    assert "reasoning" not in json.dumps(request).lower().replace(
        "requested_reasoning", "")
    assert objections[0]["objection"] == "Tare is ambiguous."
    assert challenger.last_request_id == "chatcmpl-fake-1"
    assert challenger.last_usage == {
        "prompt_tokens": 100, "completion_tokens": 20,
        "total_tokens": 120}
    assert challenger.simulated is False
