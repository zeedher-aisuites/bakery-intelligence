"""Canonical ``RecipeDraft`` record (plan.md §4.2)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bakery_intelligence.models import (
    Derivation,
    DraftStatus,
    IngredientQuantityStatus,
    NonBlankStr,
    SchemaVersionedModel,
)


class DraftStep(BaseModel):
    """One ordered procedure step with claim-to-evidence links."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    description: str
    action_ids: list[str] = Field(default_factory=list)
    observation_ids: list[str] = Field(default_factory=list)
    evidence_summary: str


class DraftQuantity(BaseModel):
    """A draft ingredient quantity. Quantities exist only with a supported
    (non-unknown) derivation; otherwise value and unit stay null."""

    model_config = ConfigDict(extra="forbid")

    value: Optional[float] = None
    unit: Optional[str] = None
    derivation: Derivation = Derivation.UNKNOWN
    source_event_id: Optional[str] = None

    @model_validator(mode="after")
    def _check_unknown_derivation(self) -> "DraftQuantity":
        if self.derivation == Derivation.UNKNOWN and (
            self.value is not None or self.unit is not None
        ):
            raise ValueError(
                'derivation "unknown" requires value=null and unit=null'
            )
        return self


class DraftIngredient(BaseModel):
    """An ingredient entry with quantity and support state."""

    model_config = ConfigDict(extra="forbid")

    ingredient_id: NonBlankStr
    quantity: DraftQuantity
    status: IngredientQuantityStatus


class UnresolvedQuestion(BaseModel):
    """An explicit gap surfaced to the reviewer."""

    model_config = ConfigDict(extra="forbid")

    code: NonBlankStr
    message: str
    related_entity_ids: list[str] = Field(default_factory=list)


class EvidenceRefs(BaseModel):
    """Navigable claim-to-evidence indices for the draft."""

    model_config = ConfigDict(extra="forbid")

    observation_ids: list[str] = Field(default_factory=list)
    frame_evidence_ids: list[str] = Field(default_factory=list)


class RecipeDraft(SchemaVersionedModel):
    """An evidence-linked recipe/procedure draft awaiting human review.

    New drafts are ``pending_review`` (the default). ``approved`` is
    representable for historical versions but must only result from an
    explicit human ``ReviewDecision`` through the review boundary and gate
    G3 (Constitution V); no generator path may set it.
    """

    draft_id: NonBlankStr
    version: int = Field(ge=1)
    video_id: NonBlankStr
    steps: list[DraftStep] = Field(default_factory=list)
    ingredients: list[DraftIngredient] = Field(default_factory=list)
    unresolved_questions: list[UnresolvedQuestion] = Field(
        default_factory=list
    )
    evidence_refs: EvidenceRefs = Field(default_factory=EvidenceRefs)
    verification_ref: NonBlankStr
    run_id: NonBlankStr
    status: DraftStatus = DraftStatus.PENDING_REVIEW
