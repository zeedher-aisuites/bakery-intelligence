"""Baseten challenger adapter (authorized hackathon model).

Verified from Experiment 004 only:
- Provider base URL: ``https://inference.baseten.co/v1``
- Exact authorized model ID: ``meta-models/muse-spark-open-1026``
  (the API key is restricted to this ID; never substitute
  ``meta-models/muse-spark-open``)
- Interface: OpenAI SDK Chat Completions with text/image content
- Usage recorded on the experiment results (prompt/completion/
  total tokens); USD pricing was not measured there.

Not verified, therefore not sent: any reasoning parameter.
``medium`` reasoning is a requested configuration recorded for
provenance, never transmitted until the provider documents it.
Structured output via Chat Completions JSON schema is likewise
unverified; this adapter prompts for JSON and parses it,
labeling the exchange accordingly.

No client is constructed until ``critique`` runs with an
injected or lazily built client. Tests inject fakes; no live
call is made by importing or configuring this module.
"""

from __future__ import annotations

import json
from typing import Any, Callable

BASETEN_BASE_URL = "https://inference.baseten.co/v1"
BASETEN_CHALLENGER_MODEL = "meta-models/muse-spark-open-1026"
REQUESTED_REASONING = "medium"


class BasetenChallenger:
    """Critiques claims through Baseten Chat Completions."""

    provider_id = "baseten"
    simulated = False

    def __init__(self, client: Any | None = None,
                 client_factory: Callable[[], Any] | None = None,
                 max_tokens: int = 4096) -> None:
        self._client = client
        self._client_factory = client_factory
        self.max_tokens = max_tokens
        self.calls = 0
        self.last_usage: dict[str, Any] | None = None
        self.last_request_id: str | None = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self._client_factory is None:
            raise RuntimeError(
                "Baseten challenger requires an explicit client "
                "or factory; live construction is not implicit.")
        self._client = self._client_factory()
        return self._client

    def critique(self, claims: list[dict[str, Any]],
                 package: dict[str, Any],
                 ) -> list[dict[str, Any]]:
        client = self._get_client()
        self.calls += 1
        prompt = json.dumps({
            "task": (
                "Critique each recipe claim against the evidence. "
                "Reply with exactly one JSON object: "
                "{\"objections\": [{\"claim_id\": str, "
                "\"objection\": str, \"contradiction\": bool}]}. "
                "Flag unsupported identities, quantities, units, "
                "tare confusion, missing actions, ordering, and "
                "invalid evidence references."),
            "requested_reasoning": REQUESTED_REASONING,
            "claims": claims,
            "evidence": {
                "run_id": package.get("run_id"),
                "frames": package.get("frames"),
                "weighing_event": package.get("weighing_event"),
                "unknowns": package.get("unknowns"),
            },
        }, sort_keys=True, default=str)
        response = client.chat.completions.create(
            model=BASETEN_CHALLENGER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
        )
        self.last_request_id = getattr(response, "id", None)
        usage = getattr(response, "usage", None)
        self.last_usage = (
            usage.model_dump() if usage is not None else None)
        content = response.choices[0].message.content or ""
        try:
            payload = json.loads(content)
        except ValueError:
            return [{
                "claim_id": claim.get("claim_id"),
                "objection": "Challenger response was not valid JSON.",
                "contradiction": False,
            } for claim in claims]
        objections = payload.get("objections")
        return objections if isinstance(objections, list) else []
