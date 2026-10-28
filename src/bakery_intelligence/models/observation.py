"""Canonical ``Observation`` record (plan.md §4.2)."""

from __future__ import annotations

from typing import Optional

from pydantic import Field, model_validator

from bakery_intelligence.models import (
    DisplayState,
    Modality,
    NonBlankStr,
    ObservationType,
    Provenance,
    SchemaVersionedModel,
    TimeInterval,
    TimestampBasis,
    VerificationStatus,
)


class Observation(SchemaVersionedModel):
    """A single claim about visible evidence, with provenance and status.

    ``scale_value`` is the displayed value when legible; it is never a net
    ingredient mass (Constitution III). ``scale_unit_hypothesis`` is an
    annotation that must never be promoted to ``scale_unit``.

    Exactly one of ``timestamp_seconds`` or ``interval`` must be set, so a
    single timestamp and a time range are never confused. Timestamps are
    nonnegative at this layer; containment within the source interval needs
    bundle context and is enforced by the reference checks (M1-03).

    This model is representational: it neither promotes nor demotes
    ``verification_status``. Status ceilings for adapter/model/preliminary
    sources are enforced by the adapter (M1-02); promotion to ``confirmed``
    is reserved to the deterministic verifier (M2-03) and to the isolated
    human-adjudication process (plan.md §5.1).
    """

    observation_id: NonBlankStr
    modality: Modality = Modality.VISUAL
    timestamp_seconds: Optional[float] = Field(default=None, ge=0)
    interval: Optional[TimeInterval] = None
    timestamp_basis: TimestampBasis
    observation_type: ObservationType
    scale_value: Optional[float] = None
    scale_unit: Optional[str] = None
    scale_unit_hypothesis: Optional[str] = None
    display_state: Optional[DisplayState] = None
    evidence_ids: list[str] = Field(default_factory=list)
    raw_reported_value: Optional[str] = None
    description: Optional[str] = None
    provenance: Provenance
    verification_status: VerificationStatus

    @model_validator(mode="after")
    def _require_exactly_one_time_form(self) -> "Observation":
        if (self.timestamp_seconds is None) == (self.interval is None):
            raise ValueError(
                "exactly one of timestamp_seconds or interval must be set"
            )
        return self
