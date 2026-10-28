"""Canonical ``WeighingEvent`` record (plan.md §4.2)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bakery_intelligence.models import (
    Derivation,
    MeasurementMode,
    NonBlankStr,
    Provenance,
    SchemaVersionedModel,
    VerificationStatus,
)


class DisplayReading(BaseModel):
    """Final displayed reading: what the scale showed, not a net mass."""

    model_config = ConfigDict(extra="forbid")

    value: Optional[float] = None
    unit: Optional[str] = None
    observation_id: NonBlankStr


class NetIngredientQuantity(BaseModel):
    """Derived net ingredient quantity with its derivation.

    ``derivation == "unknown"`` requires ``value`` and ``unit`` to be null
    (plan.md §6.1). Presence of value/unit for non-unknown derivations is
    additionally checked at the bundle/reference-check layer (M1-03,
    ``DERIVATION_WITHOUT_UNIT``).
    """

    model_config = ConfigDict(extra="forbid")

    value: Optional[float] = None
    unit: Optional[str] = None
    derivation: Derivation = Derivation.UNKNOWN

    @model_validator(mode="after")
    def _check_unknown_derivation(self) -> "NetIngredientQuantity":
        if self.derivation == Derivation.UNKNOWN and (
            self.value is not None or self.unit is not None
        ):
            raise ValueError(
                'derivation "unknown" requires value=null and unit=null'
            )
        return self


class WeighingEvent(SchemaVersionedModel):
    """A weighing event linking actions, observations, and quantities.

    ``final_display_reading`` preserves the observed display; only
    ``net_ingredient_quantity`` carries a derived mass, and only with a
    supported (non-unknown) derivation (Constitution III).
    """

    event_id: NonBlankStr
    start_seconds: float = Field(ge=0)
    end_seconds: float
    ingredient_id: Optional[str] = None
    action_ids: list[str] = Field(default_factory=list)
    observation_ids: list[str] = Field(default_factory=list)
    final_display_reading: DisplayReading
    net_ingredient_quantity: NetIngredientQuantity
    measurement_mode: MeasurementMode
    tare_observed: Optional[bool] = None
    verification_status: VerificationStatus
    review_notes: Optional[str] = None

    @model_validator(mode="after")
    def _check_interval(self) -> "WeighingEvent":
        if not self.start_seconds < self.end_seconds:
            raise ValueError(
                "event interval must satisfy "
                "0 <= start_seconds < end_seconds"
            )
        return self
