"""Recipe claim shape and controlled vocabularies.

A claim is the unit of debate: one ingredient, quantity, step,
or equipment assertion with evidence references, provenance,
and an explicit status. Provenance distinguishes information
extracted by a model, corrected by a human reviewer, and
supplied by the chef without video observation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

STATUS_SUPPORTED = "supported"
STATUS_UNCERTAIN = "uncertain"
STATUS_CONTRADICTED = "contradicted"
STATUS_REQUIRES_HUMAN = "requires_human_review"

CLAIM_STATUSES = (
    STATUS_SUPPORTED,
    STATUS_UNCERTAIN,
    STATUS_CONTRADICTED,
    STATUS_REQUIRES_HUMAN,
)

PROVENANCE_AI = "ai-extracted"
PROVENANCE_HUMAN = "human-corrected"
PROVENANCE_CHEF = "chef-supplied"
PROVENANCES = (PROVENANCE_AI, PROVENANCE_HUMAN, PROVENANCE_CHEF)

CLAIM_KINDS = ("ingredient", "quantity", "step", "equipment", "measurement")


def make_claim(
    *,
    kind: str,
    title: str,
    evidence_refs: list[dict[str, Any]] | None = None,
    ingredient_id: str | None = None,
    quantity_value: float | None = None,
    quantity_unit: str | None = None,
    derivation: str = "unknown",
    step_index: int | None = None,
    provenance: str = PROVENANCE_AI,
    uncertainty: str | None = None,
    claim_id: str | None = None,
) -> dict[str, Any]:
    """Create one claim dict; unknowns stay null, never defaulted."""
    if kind not in CLAIM_KINDS:
        raise ValueError(f"Unsupported claim kind: {kind!r}.")
    if provenance not in PROVENANCES:
        raise ValueError(f"Unsupported provenance: {provenance!r}.")
    refs = list(evidence_refs or [])
    seed = json.dumps(
        [kind, title, ingredient_id, quantity_value, refs],
        sort_keys=True, default=str)
    return {
        "claim_id": claim_id or (
            f"claim_{hashlib.sha256(seed.encode()).hexdigest()[:12]}"),
        "kind": kind,
        "title": title,
        "ingredient_id": ingredient_id,
        "quantity": {
            "value": quantity_value,
            "unit": quantity_unit,
            "derivation": derivation,
        },
        "step_index": step_index,
        "evidence_refs": refs,
        "provenance": provenance,
        "status": STATUS_UNCERTAIN,
        "uncertainty": uncertainty,
        "history": [],
        "objections": [],
    }


def evidence_ref(run_id: str, frame_index: int,
                 measurement_id: str | None = None,
                 timestamp_seconds: float | None = None,
                 ) -> dict[str, Any]:
    """Build one navigable evidence reference."""
    return {
        "run_id": run_id,
        "frame_index": frame_index,
        "measurement_id": measurement_id,
        "timestamp_seconds": timestamp_seconds,
    }
