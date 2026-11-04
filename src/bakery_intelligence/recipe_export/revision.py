"""AI recipe revisions: bounded call, protected fields, new version.

A revision never mutates its parent. The model receives the
latest SAVED recipe, the chef's instructions, selected
suggestions, and the target language; ingredient identity,
quantity, and unit are validated against the parent unless the
chef explicitly authorized ingredient changes. The result is a
new immutable version directory with parent linkage, cost, and
timestamps. Draft approval status always starts at draft.
"""

from __future__ import annotations

from typing import Any

from bakery_intelligence.phase2c import prompts

ESTIMATE_USD = 0.30
REVISION_SCHEMA = "recipe-revision/1"
# Revisions run on Baseten Spark Open, whose provider rates
# are unavailable; cost stays an explicit unknown.
REVISION_PROVIDER = "baseten"
REVISION_MODEL = "meta-models/muse-spark-open-1026"


class RevisionError(ValueError):
    """A revision request or model revision is invalid."""


def revision_bundle(recipe: dict[str, Any],
                    instructions: str,
                    suggestions: list[dict[str, Any]],
                    language: str,
                    allow_ingredient_changes: bool,
                    recipe_name: str,
                    unresolved_questions: list[str]
                    | None = None,
                    evidence: dict[str, Any] | None = None,
                    ) -> dict[str, Any]:
    """Assemble the model input from saved state plus chef input."""
    return {
        "recipe_name": recipe_name,
        "language": language,
        "allow_ingredient_changes": allow_ingredient_changes,
        "chef_instructions": instructions,
        "selected_suggestions": suggestions,
        "ingredients": [{
            "name": row.get("name"),
            "quantity": row.get("quantity"),
            "unit": row.get("unit"),
        } for row in recipe.get("ingredients") or []],
        "method": [{
            "number": item.get("number"),
            "instruction": item.get("instruction"),
        } for item in recipe.get("method") or []],
        "notes": recipe.get("limitations") or [],
        "unresolved_questions": list(unresolved_questions or []),
        "evidence": dict(evidence or {}),
    }


def _numbers_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return abs(float(left) - float(right)) < 1e-9
    except (TypeError, ValueError):
        return False


def validate_revision(recipe: dict[str, Any],
                      revised: Any,
                      allow_ingredient_changes: bool) -> dict[str, Any]:
    """Enforce shape, step stability, and ingredient protection."""
    violations = prompts.validate_output(prompts.RECIPE_REVISION,
                                         revised)
    if violations:
        raise RevisionError(
            f"Invalid revision payload: {'; '.join(violations)}")
    assert isinstance(revised, dict)
    rows = recipe.get("ingredients") or []
    new_rows = revised.get("ingredients")
    if not isinstance(new_rows, list) or len(new_rows) != len(rows):
        raise RevisionError(
            "Revision must return one ingredient row per input row.")
    if not allow_ingredient_changes:
        for position, (old, new) in enumerate(zip(rows, new_rows)):
            if not isinstance(new, dict):
                raise RevisionError(
                    f"Ingredient row {position} is not an object.")
            if (new.get("name") or "").strip() != \
                    (old.get("name") or "").strip() or \
                    not _numbers_equal(new.get("quantity"),
                                       old.get("quantity")) or \
                    (new.get("unit") or "") != (old.get("unit") or ""):
                raise RevisionError(
                    f"Ingredient row {position} changed without "
                    f"authorization; parent values preserved.")
    steps = recipe.get("method") or []
    new_steps = revised.get("method")
    if not isinstance(new_steps, list) or len(new_steps) != len(steps):
        raise RevisionError(
            "Revision must return the same method step count.")
    for old, new in zip(steps, new_steps):
        if not isinstance(new, dict) or \
                new.get("number") != old.get("number"):
            raise RevisionError(
                "Revision reordered or renumbered method steps.")
        text = new.get("instruction")
        if not isinstance(text, str) or not text.strip():
            raise RevisionError(
                f"Step {old.get('number')} has an empty instruction.")
    notes = revised.get("notes")
    if not isinstance(notes, list) or not notes or not all(
            isinstance(n, str) and n.strip() for n in notes):
        raise RevisionError(
            "Revision must return non-empty production notes.")
    title = revised.get("title")
    if not isinstance(title, str) or not title.strip():
        raise RevisionError("Revision must return a title.")
    return revised


def assemble_version(recipe: dict[str, Any],
                     revised: dict[str, Any],
                     meta: dict[str, Any]) -> dict[str, Any]:
    """Build the new immutable snapshot from parent + revision."""
    import copy

    snapshot = copy.deepcopy(recipe)
    snapshot.pop("improvement_suggestions", None)
    snapshot.pop("recipe_name", None)
    snapshot.pop("parent_version", None)
    snapshot["snapshot_id"] = meta["recipe_id"]
    snapshot["generated_at"] = meta["generated_at"]
    snapshot["title"] = revised["title"].strip()
    snapshot["language"] = meta["language"]
    snapshot["recipe_status"] = "draft"
    snapshot["status_note"] = None
    snapshot["status_updated_at"] = None
    snapshot["last_saved_at"] = None
    for row, new in zip(snapshot.get("ingredients") or [],
                        revised["ingredients"]):
        row["name"] = (new.get("name") or "").strip()
        row["quantity"] = new.get("quantity")
        row["unit"] = new.get("unit")
        row.pop("chef_edited", None)
        row.pop("original", None)
        if meta.get("allow_ingredient_changes"):
            row["quantity_status"] = "ai_estimated"
            row["quantity_basis"] = (
                "Revised per explicit chef authorization in "
                f"{meta['recipe_id']}.")
    for item, new in zip(snapshot.get("method") or [],
                         revised["method"]):
        item["instruction"] = (new.get("instruction") or "").strip()
        item.pop("chef_edited", None)
        item.pop("original", None)
    snapshot["limitations"] = [n.strip() for n in revised["notes"]]
    snapshot["revision"] = {
        "parent_version": meta["parent_id"],
        "chef_instructions": meta["instructions"],
        "selected_suggestions": meta["suggestions"],
        "allow_ingredient_changes":
        bool(meta.get("allow_ingredient_changes")),
        "changes_summary": revised.get("changes_summary"),
        "provider": meta.get("provider") or REVISION_PROVIDER,
        "model": meta.get("model") or REVISION_MODEL,
        "request_id": meta.get("request_id"),
        "actual_usd": meta.get("actual_usd"),
        "prompt_version": prompts.RECIPE_REVISION.version,
    }
    return snapshot
