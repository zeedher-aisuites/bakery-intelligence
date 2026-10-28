"""Meta Model API client for the P04 full-video experiment (Exp. 006).

Thin wrapper over the OpenAI SDK pointed at ``https://api.meta.ai/v1``,
using the Responses API with ``input_image`` blocks (base64 data URLs).
Verified contract elements are marked below; anything else is an
explicit unknown. No credentials are logged, persisted, or echoed.
"""

from __future__ import annotations

import base64
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BASE_URL = "https://api.meta.ai/v1"
MODEL_SPARK = "muse-spark-1.3"
MODEL_WONDER = "muse-wonder-1.0"
ENV_KEYS = ("META_API_KEY", "MODEL_API_KEY")
MAX_IMAGES_PER_REQUEST = 50  # Verified-docs hard cap (HTTP 400 above).
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_OUTPUT_TOKENS = 1024  # Locally proven value (001/004).


class ProviderError(Exception):
    """Structured provider failure with a machine-readable code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class InferenceResult:
    """Normalized outcome of one Responses API call."""

    model: str
    status: str
    output_text: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cached_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    latency_seconds: float = 0.0
    request_id: str | None = None
    raw_status_details: Any = None


def resolve_api_key(env: dict[str, str] | None = None) -> str:
    """Return the API key without logging or persisting it.

    Precedence: ``META_API_KEY`` (task convention), then
    ``MODEL_API_KEY`` (repository convention, used by Exp. 001/004).
    """
    source = env if env is not None else os.environ
    for name in ENV_KEYS:
        value = source.get(name)
        if value:
            return value
    raise ProviderError(
        "CREDENTIAL_MISSING",
        "Set META_API_KEY (or MODEL_API_KEY) in the process environment.",
    )


def _default_client_factory(
    api_key: str, timeout_seconds: float
) -> Any:
    from openai import OpenAI  # Lazy: unit tests inject a fake factory.

    return OpenAI(
        base_url=BASE_URL,
        api_key=api_key,
        timeout=timeout_seconds,
        max_retries=0,
    )


def _retry_after_seconds(exc: Any) -> float | None:
    try:
        headers = exc.response.headers
    except AttributeError:
        return None
    if not headers:
        return None
    try:
        value = headers.get("retry-after")
    except AttributeError:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class MetaClient:
    """Single-model Responses API client with injectable seams."""

    model: str = MODEL_SPARK
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    client_factory: Callable[[str, float], Any] = _default_client_factory
    clock: Callable[[], float] = time.perf_counter
    _client: Any = field(default=None, repr=False)

    def _require_client(self, api_key: str) -> Any:
        if self._client is None:
            self._client = self.client_factory(api_key, self.timeout_seconds)
        return self._client

    def submit(
        self,
        prompt_text: str,
        image_paths: Sequence[Path],
        api_key: str,
    ) -> InferenceResult:
        """Send one multimodal request; map outcomes to data or codes."""
        if not image_paths:
            raise ProviderError("EMPTY_REQUEST", "At least one image required.")
        if len(image_paths) > MAX_IMAGES_PER_REQUEST:
            raise ProviderError(
                "TOO_MANY_IMAGES",
                f"{len(image_paths)} images exceeds the verified "
                f"maximum of {MAX_IMAGES_PER_REQUEST} per request.",
            )
        chunks: list[dict[str, Any]] = [
            {"type": "input_text", "text": prompt_text}
        ]
        for path in image_paths:
            try:
                raw = path.read_bytes()
            except OSError as exc:
                raise ProviderError(
                    "IMAGE_MISSING",
                    f"Cannot read image bytes: {path.name}",
                ) from exc
            encoded = base64.b64encode(raw).decode("ascii")
            chunks.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{encoded}",
            })
        client = self._require_client(api_key)
        start = self.clock()
        try:
            response = client.responses.create(
                model=self.model,
                input=[{"role": "user", "content": chunks}],
                max_output_tokens=self.max_output_tokens,
            )
        except Exception as exc:  # Mapped below; never leaks the key.
            raise self._map_exception(exc) from exc
        latency = self.clock() - start
        return self._normalize(response, latency)

    def _map_exception(self, exc: Exception) -> ProviderError:
        name = type(exc).__name__
        status = getattr(exc, "status_code", None)
        if name == "APITimeoutError" or "Timeout" in name:
            return ProviderError("TIMEOUT", f"Request timed out: {exc}")
        if name == "APIConnectionError":
            return ProviderError("NETWORK_ERROR", f"Connection failed: {exc}")
        if name == "APIStatusError" or status is not None:
            retry_after = _retry_after_seconds(exc)
            if status == 401:
                return ProviderError(
                    "AUTH_FAILED", f"HTTP 401: {exc}", status_code=status
                )
            if status == 429:
                return ProviderError(
                    "RATE_LIMITED", f"HTTP 429: {exc}",
                    status_code=status, retry_after_seconds=retry_after,
                )
            if status == 400:
                return ProviderError(
                    "BAD_REQUEST", f"HTTP 400: {exc}", status_code=status
                )
            if status == 413:
                return ProviderError(
                    "PAYLOAD_TOO_LARGE", f"HTTP 413: {exc}",
                    status_code=status,
                )
            if isinstance(status, int) and 500 <= status < 600:
                return ProviderError(
                    "SERVER_ERROR", f"HTTP {status}: {exc}",
                    status_code=status, retry_after_seconds=retry_after,
                )
            return ProviderError(
                "PROVIDER_ERROR", f"HTTP {status}: {exc}",
                status_code=status,
            )
        return ProviderError("PROVIDER_ERROR", f"{name}: {exc}")

    def _normalize(self, response: Any, latency: float) -> InferenceResult:
        status = getattr(response, "status", None)
        text = getattr(response, "output_text", None)
        if not isinstance(status, str) or not isinstance(text, str):
            raise ProviderError(
                "RESPONSE_MALFORMED",
                "Response lacks a string status/output_text.",
            )
        usage = getattr(response, "usage", None)
        input_tokens = output_tokens = total_tokens = None
        cached = reasoning = None
        if usage is not None:
            dumped = (
                usage.model_dump() if hasattr(usage, "model_dump")
                else dict(usage) if isinstance(usage, dict) else {}
            )
            input_tokens = dumped.get("input_tokens")
            output_tokens = dumped.get("output_tokens")
            total_tokens = dumped.get("total_tokens")
            details_in = dumped.get("input_tokens_details") or {}
            details_out = dumped.get("output_tokens_details") or {}
            cached = details_in.get("cached_tokens")
            reasoning = details_out.get("reasoning_tokens")
        return InferenceResult(
            model=getattr(response, "model", self.model),
            status=status,
            output_text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_input_tokens=cached,
            reasoning_output_tokens=reasoning,
            latency_seconds=latency,
            request_id=getattr(response, "id", None),
            raw_status_details=getattr(
                response, "incomplete_details", None
            ),
        )


def list_models(api_key: str, timeout_seconds: float = 20.0) -> list[str]:
    """Zero-generation auth check mirroring Exp. 001 (verified-local)."""
    import httpx  # Lazy: keeps unit tests network-free.

    try:
        response = httpx.get(
            f"{BASE_URL}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
        )
    except Exception as exc:
        raise ProviderError("NETWORK_ERROR", f"Model list failed: {exc}") from exc
    if response.status_code == 401:
        raise ProviderError("AUTH_FAILED", "HTTP 401 from /v1/models.")
    if response.status_code != 200:
        raise ProviderError(
            "PROVIDER_ERROR",
            f"/v1/models returned HTTP {response.status_code}.",
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise ProviderError(
            "RESPONSE_MALFORMED", "Non-JSON /v1/models body."
        ) from exc
    models = data.get("data", [])
    return [str(item.get("id")) for item in models if item.get("id")]
