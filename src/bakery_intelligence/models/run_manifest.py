"""Canonical ``RunManifest`` record (plan.md §4.2, §9.1)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from bakery_intelligence.models import (
    SHA256_HEX_PATTERN,
    NonBlankStr,
    RunStatus,
    SchemaVersionedModel,
    TimeInterval,
    Verdict,
)
from bakery_intelligence.models.video_source import VideoSource


class RunConfig(BaseModel):
    """Frozen analysis configuration snapshot for the run."""

    model_config = ConfigDict(extra="forbid")

    sample_fps: Optional[float] = Field(default=None, gt=0)
    interval: TimeInterval
    frame_budget: int = Field(ge=0)
    model_budget: int = Field(ge=0)
    retry_limit: int = Field(ge=0)
    cost_budget_usd: Optional[float] = Field(default=None, ge=0)


class BudgetState(BaseModel):
    """Consumed-budget counters and exhaustion marker (plan.md §9.1)."""

    model_config = ConfigDict(extra="forbid")

    frames_used: int = Field(ge=0)
    model_calls_used: int = Field(ge=0)
    cost_usd_spent: float = Field(ge=0)
    exhausted: bool
    exhausted_at: Optional[datetime] = None


class RunOutputs(BaseModel):
    """Artifact paths produced by the run, with hashes when computed.

    All fields are optional because partial and failed runs persist a
    manifest before every artifact exists. Output hashes are added by the
    full manifest writer (M5); the M2 minimal seed records paths only.
    """

    model_config = ConfigDict(extra="forbid")

    ledger_path: Optional[str] = None
    events_path: Optional[str] = None
    draft_path: Optional[str] = None
    verification_path: Optional[str] = None
    ledger_sha256: Optional[str] = Field(
        default=None, pattern=SHA256_HEX_PATTERN
    )
    events_sha256: Optional[str] = Field(
        default=None, pattern=SHA256_HEX_PATTERN
    )
    draft_sha256: Optional[str] = Field(
        default=None, pattern=SHA256_HEX_PATTERN
    )
    verification_sha256: Optional[str] = Field(
        default=None, pattern=SHA256_HEX_PATTERN
    )


class AttemptRecord(BaseModel):
    """One recorded stage attempt with its verdict."""

    model_config = ConfigDict(extra="forbid")

    stage: NonBlankStr
    verdict: Verdict
    issues_ref: Optional[str] = None


class TokenUsage(BaseModel):
    """Provider token accounting for one call."""

    model_config = ConfigDict(extra="forbid")

    prompt: int = Field(ge=0)
    completion: int = Field(ge=0)


class ProviderCallRecord(BaseModel):
    """Metadata for one provider call (never raw media under license)."""

    model_config = ConfigDict(extra="forbid")

    provider: NonBlankStr
    model_id: NonBlankStr
    prompt_hash: NonBlankStr
    settings: Optional[dict[str, Any]] = None
    started_at: Optional[datetime] = None
    latency_ms: Optional[int] = Field(default=None, ge=0)
    tokens: Optional[TokenUsage] = None
    cost_usd: Optional[float] = Field(default=None, ge=0)
    error_code: Optional[str] = None


class RunError(BaseModel):
    """One structured run error."""

    model_config = ConfigDict(extra="forbid")

    stage: NonBlankStr
    code: NonBlankStr
    message: str


class CostSummary(BaseModel):
    """Aggregated variable cost. Offline runs stay at zero."""

    model_config = ConfigDict(extra="forbid")

    total_cost_usd: float = Field(default=0.0, ge=0)
    currency: str = "USD"


class RunManifest(SchemaVersionedModel):
    """Reproducible audit trail of one run, including partial/failed runs.

    ``status`` always reflects the truthful outcome; failures never
    fabricate success (Constitution VI). The offline minimal seed (M2)
    populates identity, input snapshots, ``started_at``, ``budget_state``,
    output paths, and ``status``; the full writer (M5) adds finalization,
    hashes, attempts, provider calls, errors, and cost.
    """

    run_id: NonBlankStr
    code_revision: NonBlankStr
    started_at: datetime
    finished_at: Optional[datetime] = None
    elapsed_ms: Optional[int] = Field(default=None, ge=0)
    video_source_snapshot: VideoSource
    run_config_snapshot: RunConfig
    timestamp_assumption: NonBlankStr
    budget_state: BudgetState
    outputs: RunOutputs = Field(default_factory=RunOutputs)
    attempts: list[AttemptRecord] = Field(default_factory=list)
    provider_calls: list[ProviderCallRecord] = Field(default_factory=list)
    errors: list[RunError] = Field(default_factory=list)
    status: RunStatus
    status_reason: Optional[str] = None
    cost_summary: CostSummary = Field(default_factory=CostSummary)
