"""Canonical ``VerificationReport`` record (plan.md §4.2, §7)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from bakery_intelligence.models import (
    GateId,
    NonBlankStr,
    SchemaVersionedModel,
    Verdict,
)


class VerificationIssue(BaseModel):
    """One machine-readable verification finding.

    ``severity`` carries a required marker string; its controlled vocabulary
    is defined by the reference checks (M1-03) and verifier gates (M2-03),
    not by this model.
    """

    model_config = ConfigDict(extra="forbid")

    code: NonBlankStr
    entity_id: str
    gate: GateId
    message: str
    severity: str


class VerificationReport(SchemaVersionedModel):
    """The persisted outcome of deterministic verification gates.

    A ``pass`` verdict attests structural and semantic consistency only; it
    does not certify that any display, ingredient, or action was interpreted
    correctly from pixels (plan.md §12).
    """

    report_id: NonBlankStr
    verdict: Verdict
    issues: list[VerificationIssue] = Field(default_factory=list)
    checked_ids: list[str] = Field(default_factory=list)
    verifier_version: NonBlankStr
    checked_at: datetime
