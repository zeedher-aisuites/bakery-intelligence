"""Proposed v2 recipe: reviewed v1 plus AI culinary completion.

Deterministic merge of the frozen v1 snapshot with the recorded
completion proposal. Evidence-supported values are never altered
except for two explained draft roundings (699->700 g flour,
112->113 g butter: spoken baker's targets, displays preserved).
AI estimates attach with their own provenance; anything invalid
stays not_determined. The ambiguous ig_19421ce47de7 group is
excluded from the v2 wheat-flour row (documented below) instead
of being silently absorbed.
"""

from __future__ import annotations

from typing import Any

V2_SNAPSHOT_ID = "bakery_demo_v1_proposed_v2"
V2_VERSION = "recipe-snapshot/2"

ALLOWED_UNITS = {"g", "kg", "ml", "l", "whole", "pcs", "tsp",
                 "tbsp", "cup", "pinch"}

# Draft roundings toward the spoken baker's target. The displayed
# value stays in the record; the draft uses the target.
DRAFT_ROUNDINGS = {
    "bread flour": {
        "amount": 700.0,
        "reason": ("Draft uses the spoken baker's target of 700 g; "
                   "the scale displayed 699 g (1 g short, likely "
                   "settling). Both values are preserved."),
    },
    "butter": {
        "amount": 113.0,
        "reason": ("Draft uses the spoken baker's target of 113 g; "
                   "the scale displayed 112 g and the speaker read "
                   "back 'ciento, doce'. Both values preserved."),
    },
}

# Groups excluded from v2 recipe rows (kept out of v1 silently
# absorbed state only with an explicit documented reason).
V2_EXCLUDED_GROUPS = {
    "ig_19421ce47de7": ("Excluded from the v2 wheat-flour row: the "
                        "group was corrected to wheat flour but its "
                        "clip-02 segment evidence matches the 24 g "
                        "milk-powder weighing (see review flag). It "
                        "must not be treated as verified wheat "
                        "flour; awaiting human re-review."),
}


def _valid_proposal(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    name = item.get("ingredient")
    amount = item.get("amount")
    unit = item.get("unit")
    if not isinstance(name, str) or not name.strip():
        return None
    if isinstance(amount, bool) or not isinstance(
            amount, (int, float)) or amount <= 0:
        return None
    if unit not in ALLOWED_UNITS:
        return None
    confidence = item.get("confidence")
    return {
        "ingredient": name.strip(),
        "amount": float(amount),
        "unit": unit,
        "rationale": item.get("rationale")
        if isinstance(item.get("rationale"), str) else "",
        "confidence": confidence
        if confidence in ("high", "medium", "low") else "low",
    }


def apply_completion(
        ingredients: list[dict[str, Any]],
        proposals: list[dict[str, Any]],
        ) -> list[str]:
    """Attach AI estimates to unresolved rows; returns open items."""
    open_items: list[str] = []
    by_name = {row["name"].lower(): row for row in ingredients}
    seen: set[str] = set()
    for raw in proposals:
        proposal = _valid_proposal(raw)
        if proposal is None:
            open_items.append(
                f"Rejected invalid proposal: {raw!r}"[:200])
            continue
        key = proposal["ingredient"].lower()
        if key in seen:
            open_items.append(
                f"Duplicate proposal for {proposal['ingredient']} "
                f"ignored.")
            continue
        seen.add(key)
        row = by_name.get(key)
        if row is None:
            open_items.append(
                f"Proposal for unknown ingredient "
                f"{proposal['ingredient']} ignored.")
            continue
        if row.get("quantity_status") != "not_determined":
            open_items.append(
                f"Proposal for already-supported "
                f"{proposal['ingredient']} ignored.")
            continue
        row["quantity"] = proposal["amount"]
        row["unit"] = proposal["unit"]
        row["quantity_status"] = "ai_estimated"
        row["quantity_basis"] = (
            f"AI estimate ({proposal['confidence']} confidence): "
            f"{proposal['rationale']}")
    return open_items


def apply_roundings(ingredients: list[dict[str, Any]]) -> None:
    """Apply explained draft roundings, preserving displays."""
    for row in ingredients:
        spec = DRAFT_ROUNDINGS.get(row["name"].lower())
        if spec is None:
            continue
        if row.get("quantity_status") != "chef_supplied":
            continue
        displayed = row.get("quantity")
        row["quantity"] = spec["amount"]
        row["draft_adjustment"] = {
            "displayed_value": displayed,
            "draft_value": spec["amount"],
            "reason": spec["reason"],
        }


def exclude_flagged(ingredients: list[dict[str, Any]]
                    ) -> list[dict[str, Any]]:
    """Drop flagged groups from v2 rows, recording each removal."""
    removed: list[dict[str, Any]] = []
    for row in ingredients:
        kept = [g for g in (row.get("group_ids") or [])
                if g not in V2_EXCLUDED_GROUPS]
        for group_id in (row.get("group_ids") or []):
            if group_id in V2_EXCLUDED_GROUPS:
                removed.append({
                    "group_id": group_id,
                    "from_ingredient": row["name"],
                    "reason": V2_EXCLUDED_GROUPS[group_id],
                })
        row["group_ids"] = kept
    return [row for row in ingredients if row.get("group_ids")]


def attach_guidance(method: list[dict[str, Any]],
                    guidance: list[dict[str, Any]]) -> list[str]:
    """Attach AI guidance to method steps; returns open items."""
    open_items: list[str] = []
    by_number = {item.get("number"): item for item in method}
    for entry in guidance:
        if not isinstance(entry, dict):
            continue
        number = entry.get("step_number")
        text = entry.get("guidance")
        if number not in by_number or not isinstance(text, str) \
                or not text.strip():
            open_items.append(
                f"Guidance for unknown step {number!r} ignored.")
            continue
        by_number[number]["ai_guidance"] = {
            "text": text.strip(),
            "provenance": "ai_estimated",
        }
    return open_items


def build_v2(v1: dict[str, Any],
             completion: dict[str, Any],
             completion_meta: dict[str, Any],
             generated_at: str) -> dict[str, Any]:
    """Merge v1 + completion into the proposed v2 snapshot."""
    import copy

    ingredients = copy.deepcopy(v1.get("ingredients") or [])
    method = copy.deepcopy(v1.get("method") or [])
    open_items: list[str] = []
    open_items.extend(apply_completion(
        ingredients, completion.get("quantity_proposals") or []))
    apply_roundings(ingredients)
    ingredients = exclude_flagged(ingredients)
    open_items.extend(attach_guidance(
        method, completion.get("method_guidance") or []))
    open_items.extend(
        str(item) for item in
        (completion.get("open_items") or []) if item)

    flags = copy.deepcopy(v1.get("review_flags") or [])
    for flag in flags:
        if flag.get("flag_id") == \
                "possible-mis-correction-ig_19421ce47de7":
            flag["v2_handling"] = (
                "Excluded from the v2 wheat-flour row; still "
                "awaiting human re-review.")
    limitations = list(v1.get("limitations") or [])
    limitations.append(
        "v2 draft values: bread flour 700 g and butter 113 g use "
        "the spoken baker's targets; the displayed 699 g and "
        "112 g are preserved in each row's draft_adjustment.")
    limitations.append(
        "AI-estimated quantities are culinary proposals, not "
        "measurements; they must be confirmed by the chef.")

    return {
        "snapshot_version": V2_VERSION,
        "snapshot_id": V2_SNAPSHOT_ID,
        "run_name": v1.get("run_name"),
        "generated_at": generated_at,
        "title": v1.get("title"),
        "title_basis": v1.get("title_basis"),
        "recipe_status": "draft",
        "review_summary": v1.get("review_summary"),
        "provenance": {
            "v1_snapshot": "runs_api/recipes/"
                           "bakery_demo_v1_ai_reviewed_v1/"
                           "snapshot.json",
            "completion_request_id": completion_meta.get(
                "request_id"),
            "completion_prompt_version": completion_meta.get(
                "prompt_version"),
            "completion_model": "muse-spark-1.3",
            "completion_actual_usd": completion_meta.get(
                "actual_usd"),
            "agent_f_note": (v1.get("provenance") or {}).get(
                "agent_f_note"),
        },
        "consolidation_aliases": v1.get("consolidation_aliases"),
        "ingredients": ingredients,
        "excluded_observations": v1.get("excluded_observations"),
        "method": method,
        "omitted_detail_steps": v1.get("omitted_detail_steps"),
        "review_flags": flags,
        "completion_open_items": open_items,
        "limitations": limitations,
    }
