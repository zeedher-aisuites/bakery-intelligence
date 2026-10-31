"""Meta Muse Spark 1.3 adapter for FirstPassRunner (M2-06C.5/S4-A).

Production-oriented client over the verified Meta Responses API
call shape (experiments 001/004/006), with an injectable SDK
seam so every test runs offline. No live path is wired: the
default factory needs the OpenAI SDK plus an explicit key, and
no workflow, grant, or UI option enables paid execution.

Verified contract elements are marked below; anything else is
an explicit unknown (see docs/m206/meta-spark-adapter.md). No
credentials are logged, persisted, or echoed.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bakery_intelligence.inference.firstpass import (
    DEFAULT_BATCH_SIZE,
    FirstPassConfig,
    FirstPassError,
    FrameRef,
    InferenceClient,
    InferenceError,
    InferenceResponse,
    InferenceUsage,
    SourceRef,
)

# Verified 001/004/006: SDK base URL and Responses model id.
BASE_URL = "https://api.meta.ai/v1"
MODEL_SPARK = "muse-spark-1.3"
# Inherited from 006 (comment claims verified-docs hard cap).
MAX_IMAGES_PER_REQUEST = 50
# Inherited from the 006 full-run client (001 used 30s for text,
# 004-meta 60s for vision); configurable per client.
DEFAULT_TIMEOUT_SECONDS = 120.0

_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


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


def _as_int(value: Any) -> int | None:
    """Keep provider token counts only when they are real ints.

    Anything else degrades to unknown (None) instead of
    crashing downstream cost accounting; values are never
    invented or coerced from strings.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _mime_type(raw: bytes, name: str) -> str:
    """Sniff JPEG/PNG magic bytes; refuse anything else.

    The MIME label always matches the actual payload. Unlike
    the historical client, which labeled every image as JPEG,
    unknown or empty payloads fail explicitly instead of
    being mislabeled.
    """
    if raw[:3] == _JPEG_MAGIC:
        return "image/jpeg"
    if raw[:8] == _PNG_MAGIC:
        return "image/png"
    raise InferenceError(
        "UNSUPPORTED_IMAGE",
        f"Cannot determine JPEG/PNG format for image: {name}.",
    )


@dataclass
class MetaSparkClient:
    """Single-model Responses API client with injectable seams.

    Mirrors the verified experiments/006 MetaClient call shape,
    adapted to the generalized InferenceClient seam: the token
    budget travels per request instead of mutable client state.
    """

    model: str = MODEL_SPARK
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    store: bool = True
    client_factory: Callable[[str, float], Any] = (
        _default_client_factory
    )
    clock: Callable[[], float] = time.perf_counter
    _client: Any = field(default=None, repr=False)

    @property
    def model_id(self) -> str:
        """Return the configured model identity for provenance."""
        return self.model

    def _require_client(self, api_key: str) -> Any:
        if self._client is None:
            self._client = self.client_factory(
                api_key, self.timeout_seconds)
        return self._client

    def submit(
        self,
        prompt_text: str,
        image_paths: Sequence[Path],
        api_key: str,
        max_output_tokens: int,
    ) -> InferenceResponse:
        """Send one multimodal request; map outcomes to data/codes."""
        if not image_paths:
            raise InferenceError(
                "EMPTY_REQUEST", "At least one image required.")
        if len(image_paths) > MAX_IMAGES_PER_REQUEST:
            raise InferenceError(
                "TOO_MANY_IMAGES",
                f"{len(image_paths)} images exceeds the verified "
                f"maximum of {MAX_IMAGES_PER_REQUEST} per request.",
            )
        if (isinstance(max_output_tokens, bool)
                or not isinstance(max_output_tokens, int)
                or max_output_tokens <= 0):
            raise InferenceError(
                "INVALID_REQUEST",
                "max_output_tokens must be a positive int, "
                f"got {max_output_tokens!r}.",
            )
        chunks: list[dict[str, Any]] = [
            {"type": "input_text", "text": prompt_text}
        ]
        for path in image_paths:
            try:
                raw = path.read_bytes()
            except OSError as exc:
                raise InferenceError(
                    "IMAGE_MISSING",
                    f"Cannot read image bytes: {path.name}",
                ) from exc
            mime = _mime_type(raw, path.name)
            encoded = base64.b64encode(raw).decode("ascii")
            chunks.append({
                "type": "input_image",
                "image_url": f"data:{mime};base64,{encoded}",
            })
        client = self._require_client(api_key)
        start = self.clock()
        try:
            response = client.responses.create(
                model=self.model,
                input=[{"role": "user", "content": chunks}],
                max_output_tokens=max_output_tokens,
                store=self.store,
            )
        except Exception as exc:  # Mapped below; never leaks the key.
            raise self._map_exception(exc) from exc
        latency = self.clock() - start
        return self._normalize(response, latency)

    def _map_exception(self, exc: Exception) -> InferenceError:
        name = type(exc).__name__
        status = getattr(exc, "status_code", None)
        if name == "APITimeoutError" or "Timeout" in name:
            return InferenceError(
                "TIMEOUT", f"Request timed out: {exc}")
        if name == "APIConnectionError":
            return InferenceError(
                "NETWORK_ERROR", f"Connection failed: {exc}")
        if name == "APIStatusError" or status is not None:
            retry_after = _retry_after_seconds(exc)
            if status == 401:
                return InferenceError(
                    "AUTH_FAILED", f"HTTP 401: {exc}",
                    status_code=status,
                )
            if status == 429:
                return InferenceError(
                    "RATE_LIMITED", f"HTTP 429: {exc}",
                    status_code=status,
                    retry_after_seconds=retry_after,
                )
            if status == 400:
                return InferenceError(
                    "BAD_REQUEST", f"HTTP 400: {exc}",
                    status_code=status,
                )
            if status == 413:
                return InferenceError(
                    "PAYLOAD_TOO_LARGE", f"HTTP 413: {exc}",
                    status_code=status,
                )
            if isinstance(status, int) and 500 <= status < 600:
                return InferenceError(
                    "SERVER_ERROR", f"HTTP {status}: {exc}",
                    status_code=status,
                    retry_after_seconds=retry_after,
                )
            return InferenceError(
                "PROVIDER_ERROR", f"HTTP {status}: {exc}",
                status_code=status,
            )
        return InferenceError("PROVIDER_ERROR", f"{name}: {exc}")

    def _normalize(
        self, response: Any, latency: float
    ) -> InferenceResponse:
        status = getattr(response, "status", None)
        text = getattr(response, "output_text", None)
        if not isinstance(status, str) or not isinstance(text, str):
            raise InferenceError(
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
            input_tokens = _as_int(dumped.get("input_tokens"))
            output_tokens = _as_int(dumped.get("output_tokens"))
            total_tokens = _as_int(dumped.get("total_tokens"))
            details_in = dumped.get("input_tokens_details") or {}
            details_out = dumped.get("output_tokens_details") or {}
            if isinstance(details_in, dict):
                cached = _as_int(details_in.get("cached_tokens"))
            if isinstance(details_out, dict):
                reasoning = _as_int(
                    details_out.get("reasoning_tokens"))
        details = getattr(response, "incomplete_details", None)
        if isinstance(details, dict):
            incomplete_reason = details.get("reason")
        else:
            incomplete_reason = getattr(details, "reason", None)
        if incomplete_reason is not None:
            incomplete_reason = str(incomplete_reason)
        model = getattr(response, "model", None) or self.model
        return InferenceResponse(
            model=model,
            status=status,
            output_text=text,
            usage=InferenceUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cached_input_tokens=cached,
                reasoning_output_tokens=reasoning,
            ),
            latency_seconds=latency,
            request_id=getattr(response, "id", None),
            incomplete_reason=incomplete_reason,
        )


def billable_runner_config(
    *,
    frames: list[FrameRef],
    source: SourceRef,
    out_dir: Path,
    prompt_text: str,
    client_factory: Callable[[], InferenceClient],
    model: str = MODEL_SPARK,
    token_budget: int = 1024,
    api_key: str = "",
    pass_id: str = "firstpass",
    batch_size: int = DEFAULT_BATCH_SIZE,
    rates_usd_per_m: tuple[float, float] | None = None,
    cost_ceiling_usd: float | None = None,
    strict_cost: bool = False,
) -> FirstPassConfig:
    """Build an S3-safe billable runner config for this adapter.

    The safety posture is structural, not advisory: billing is
    declared, attempts and budgets are fixed to one, and
    uncertain replay is refused. Key provisioning stays with
    the caller (S5 owns authorization grants).
    """
    if (isinstance(token_budget, bool)
            or not isinstance(token_budget, int)
            or token_budget <= 0):
        raise FirstPassError(
            "INVALID_CONFIG",
            "token_budget must be a positive int, "
            f"got {token_budget!r}.")
    if not isinstance(pass_id, str) or not pass_id:
        raise FirstPassError(
            "INVALID_CONFIG",
            "pass_id must be a non-empty str, "
            f"got {pass_id!r}.")
    if (isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1):
        raise FirstPassError(
            "INVALID_CONFIG",
            "batch_size must be a positive int, "
            f"got {batch_size!r}.")
    return FirstPassConfig(
        frames=frames,
        source=source,
        out_dir=out_dir,
        model=model,
        prompt_text=prompt_text,
        provider="meta",
        pass_id=pass_id,
        batch_size=batch_size,
        token_budgets=(token_budget,),
        max_attempts=1,
        billable=True,
        uncertain_policy="refuse",
        api_key=api_key,
        client_factory=client_factory,
        rates_usd_per_m=rates_usd_per_m,
        cost_ceiling_usd=cost_ceiling_usd,
        strict_cost=strict_cost,
    )
