"""Phase 2C provider clients (Meta Model API + Voice ASR).

Thin httpx clients over verified contracts: Files upload +
Responses input_video for Spark, multipart ASR transcribe for
voice. No automatic retries: the orchestrator retries explicitly
against the budget ledger. Every billable call is pre-checked
and recorded; provider usage is stored separately from
estimates. No credentials are logged or persisted.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

BASE_URL = "https://api.meta.ai/v1"
MODEL_SPARK = "muse-spark-1.3"
# Paths are relative to BASE_URL (which already ends in /v1).
FILES_PATH = "/files"
RESPONSES_PATH = "/responses"
ASR_TRANSCRIBE_PATH = "/asr/transcribe"

RATE_SPARK_IN_PER_M = 1.25
RATE_SPARK_OUT_PER_M = 4.25
RATE_ASR_PER_HOUR = 0.18


class ProviderError(RuntimeError):
    """A provider request failed (billable status uncertain)."""


class BudgetExceeded(RuntimeError):
    """A planned call would break the spending ceiling."""


def api_key() -> str:
    key = os.environ.get("MODEL_API_KEY") or os.environ.get(
        "META_API_KEY")
    if not key:
        raise ProviderError(
            "No Meta API key in MODEL_API_KEY/META_API_KEY.")
    return key


@dataclass
class BudgetLedger:
    """Application-level spending guard (no provider-side cap)."""

    path: Path
    budget_usd: float = 30.0
    max_single_call_usd: float = 2.0
    spent_usd: float = 0.0
    calls: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "BudgetLedger":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        ledger = cls(
            path=path,
            budget_usd=float(payload.get("budget_usd", 30.0)),
            max_single_call_usd=float(payload.get("gates", {}).get(
                "max_single_call_usd", 2.0)))
        for entry in payload.get("spent", []) or []:
            if isinstance(entry, dict):
                ledger.calls.append(entry)
                try:
                    ledger.spent_usd += float(
                        entry.get("actual_usd",
                                  entry.get("estimate_usd", 0.0)))
                except (TypeError, ValueError):
                    continue
        return ledger

    def remaining(self) -> float:
        return self.budget_usd - self.spent_usd

    def check(self, estimate_usd: float, label: str) -> None:
        if estimate_usd > self.max_single_call_usd:
            raise BudgetExceeded(
                f"{label}: estimated ${estimate_usd:.2f} exceeds "
                f"the ${self.max_single_call_usd:.2f} "
                f"single-call gate.")
        if estimate_usd > self.remaining():
            raise BudgetExceeded(
                f"{label}: estimated ${estimate_usd:.2f} exceeds "
                f"the remaining ${self.remaining():.2f}.")

    def record(self, entry: dict[str, Any]) -> None:
        self.calls.append(entry)
        try:
            self.spent_usd += float(entry.get(
                "actual_usd", entry.get("estimate_usd", 0.0)))
        except (TypeError, ValueError):
            pass
        payload: dict[str, Any] = {"budget_usd": self.budget_usd,
                                   "spent": self.calls}
        if self.path.is_file():
            try:
                existing = json.loads(
                    self.path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    payload = {**existing, **payload}
            except ValueError:
                pass
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")


def estimate_spark_usd(input_tokens: int,
                       output_tokens: int) -> float:
    return (input_tokens / 1_000_000 * RATE_SPARK_IN_PER_M
            + output_tokens / 1_000_000 * RATE_SPARK_OUT_PER_M)


def estimate_asr_usd(audio_seconds: float) -> float:
    import math

    return math.ceil(max(audio_seconds, 0)) / 3600 * RATE_ASR_PER_HOUR


@dataclass
class ProviderCall:
    agent: str
    label: str
    request_id: str | None
    status: str
    elapsed_seconds: float
    usage: dict[str, Any]
    estimate_usd: float
    actual_usd: float | None
    error: str | None = None


class MetaResponsesClient:
    """Files + Responses video/text client (single attempt each)."""

    def __init__(self, timeout_seconds: float = 1800.0,
                 transport: httpx.BaseTransport | None = None):
        self._timeout = timeout_seconds
        self._transport = transport

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {api_key()}"},
            timeout=self._timeout, transport=self._transport)

    def upload_file(self, path: Path,
                    purpose: str = "user_data") -> str:
        with self._client() as client:
            with path.open("rb") as handle:
                response = client.post(
                    FILES_PATH,
                    files={"file": (path.name, handle)},
                    data={"purpose": purpose})
        if response.status_code >= 400:
            raise ProviderError(
                f"File upload failed ({response.status_code}): "
                f"{response.text[:200]}")
        payload = response.json()
        file_id = payload.get("id")
        if not file_id:
            raise ProviderError(
                f"Upload response has no id: "
                f"{response.text[:200]}")
        return str(file_id)

    def create_response(
            self, model: str,
            content: list[dict[str, Any]],
            max_output_tokens: int = 16384) -> dict[str, Any]:
        started = time.monotonic()
        with self._client() as client:
            response = client.post(
                RESPONSES_PATH,
                json={"model": model,
                      "input": [{"role": "user",
                                 "content": content}],
                      "max_output_tokens": max_output_tokens,
                      "store": False})
        elapsed = time.monotonic() - started
        if response.status_code >= 400:
            raise ProviderError(
                f"Responses call failed "
                f"({response.status_code}): "
                f"{response.text[:300]}")
        payload = response.json()
        payload["_elapsed_seconds"] = elapsed
        return payload


def extract_output_text(payload: dict[str, Any]) -> str:
    texts: list[str] = []
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("text"):
                    texts.append(str(part["text"]))
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct:
        texts.append(direct)
    return "\n".join(texts)


class VoiceTranscribeClient:
    """Multipart ASR transcription client (single attempt each)."""

    def __init__(self, timeout_seconds: float = 1200.0,
                 transport: httpx.BaseTransport | None = None):
        self._timeout = timeout_seconds
        self._transport = transport

    def transcribe(self, audio_path: Path,
                   mode: str = "DIARIZATION") -> dict[str, Any]:
        started = time.monotonic()
        settings = json.dumps(
            {"mode": mode,
             "model": "muse-voice-transcribe-1.0",
             "audioEncoding": "WAV"})
        with httpx.Client(
                base_url=BASE_URL,
                headers={"Authorization": f"Bearer {api_key()}"},
                timeout=self._timeout,
                transport=self._transport) as client:
            with audio_path.open("rb") as handle:
                response = client.post(
                    ASR_TRANSCRIBE_PATH,
                    files={
                        "request": (None, settings,
                                    "application/json"),
                        "audio": (audio_path.name, handle,
                                  "audio/wav"),
                    })
        elapsed = time.monotonic() - started
        if response.status_code >= 400:
            raise ProviderError(
                f"ASR transcribe failed "
                f"({response.status_code}): "
                f"{response.text[:300]}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderError(
                "ASR response is not JSON.") from exc
        if not isinstance(payload, dict):
            raise ProviderError("ASR response is not an object.")
        payload["_elapsed_seconds"] = elapsed
        return payload
