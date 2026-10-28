"""Canonical ``ActionEvent`` record (plan.md §4.2)."""

from __future__ import annotations

from typing import Optional

from pydantic import Field, model_validator

from bakery_intelligence.models import (
    ActionType,
    NonBlankStr,
    Provenance,
    SchemaVersionedModel,
    VerificationStatus,
)


class ActionEvent(SchemaVersionedModel):
    """An ordered kitchen action interval with supporting observation links.

    The interval satisfies ``0 <= start_seconds < end_seconds`` at this
    layer; containment within the source duration needs bundle context and
    is enforced by the reference checks (M1-03).
    """

    action_id: NonBlankStr
    start_seconds: float = Field(ge=0)
    end_seconds: float
    action_type: ActionType
    ingredient_id: Optional[str] = None
    evidence_observation_ids: list[str] = Field(default_factory=list)
    provenance: Provenance
    verification_status: VerificationStatus

    @model_validator(mode="after")
    def _check_interval(self) -> "ActionEvent":
        if not self.start_seconds < self.end_seconds:
            raise ValueError(
                "action interval must satisfy "
                "0 <= start_seconds < end_seconds"
            )
        return self
