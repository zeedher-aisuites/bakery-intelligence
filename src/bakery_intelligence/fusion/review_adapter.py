"""Adapter: fusion draft -> recipe-review store rows (offline only).

Shapes draft content for the existing RecipeReviewStore contract
without writing anything. Every quantity stays null with an
explicit unknown derivation; provenance is always ai-extracted.
"""

from __future__ import annotations

from typing import Any


def to_store_ingredients(
        draft: dict[str, Any]) -> list[dict[str, Any]]:
    """Draft ingredients as store ingredient rows (no quantities)."""
    rows: list[dict[str, Any]] = []
    for entry in draft.get("ingredients") or []:
        names = entry.get("names") or []
        labels = entry.get("visible_labels") or []
        identifier = names[0] if names else (
            labels[0] if labels else str(entry.get("group_id")))
        rows.append({
            "ingredient_id": identifier,
            "quantity": {"value": None, "unit": None,
                         "derivation": "unknown"},
            "provenance": "ai-extracted",
            "uncertainty": "; ".join(entry.get("ambiguities") or [])
            or None,
            "step_refs": [],
        })
    return rows


def to_store_steps(draft: dict[str, Any]) -> list[dict[str, Any]]:
    """Draft steps as store step rows with evidence notes."""
    rows: list[dict[str, Any]] = []
    for step in draft.get("steps") or []:
        actions = step.get("actions") or []
        description = "; ".join(actions) if actions else (
            f"Fused event group {step.get('group_id')}")
        uncertainties = step.get("uncertainties") or []
        rows.append({
            "step_id": str(step.get("group_id")),
            "description": description,
            "start_seconds": step.get("start_seconds"),
            "end_seconds": step.get("end_seconds"),
            "frame_refs": [],
            "ingredient_refs": [],
            "usages": [],
            "provenance": "ai-extracted",
            "review_status": "pending",
            "uncertainty": "; ".join(uncertainties) or None,
            "not_observed": False,
        })
    return rows
