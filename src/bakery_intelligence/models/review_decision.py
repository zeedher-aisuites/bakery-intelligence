"""Canonical ``ReviewDecision`` record (plan.md §4.2, §8)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field

from bakery_intelligence.models import (
    NonBlankStr,
    ReviewDecisionValue,
    SchemaVersionedModel,
)


class ReviewDecision(SchemaVersionedModel):
    """An explicit, version-pinned human decision on one draft version.

    Approving version N never approves version N+1 (plan.md §8.1). No
    automatic processing path may create an ``approved`` decision
    (Constitution V).
    """

    decision_id: NonBlankStr
    draft_id: NonBlankStr
    draft_version: int = Field(ge=1)
    reviewer_id: NonBlankStr
    decision: ReviewDecisionValue
    notes: Optional[str] = None
    corrections_ref: Optional[str] = None
    decided_at: datetime
