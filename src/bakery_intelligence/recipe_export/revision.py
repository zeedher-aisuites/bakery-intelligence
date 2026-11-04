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

import re
import unicodedata
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


def _word_tokens(text: str) -> set[str]:
    """Lowercase alphanumeric tokens, accents stripped."""
    folded = unicodedata.normalize(
        "NFKD", (text or "").lower())
    ascii_text = "".join(
        char for char in folded
        if not unicodedata.combining(char))
    return {token for token in
            re.findall(r"[a-z0-9]+", ascii_text)
            if len(token) > 1}


def _leading_number(value: Any) -> float | None:
    """Leading numeric value of a chef correction, else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    match = re.match(r"\s*(-?\d+(?:\.\d+)?)", value)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def match_chef_suggestion(
        name: Any, quantity: Any,
        suggestions: list[dict[str, Any]] | None,
        labels: dict[str, str] | None = None,
        used: set[str] | None = None) -> dict[str, Any] | None:
    """Selected suggestion supporting this row, else None.

    A suggestion supports a row when the row's normalized
    name tokens overlap the suggestion id (plus its known
    label) and the row quantity equals the suggestion
    value's leading number. Units are not compared: the
    correction form captures values, not units. Each id in
    ``used`` is skipped so one correction cannot justify
    two rows.
    """
    row_tokens = _word_tokens(
        name if isinstance(name, str) else "")
    if not row_tokens:
        return None
    used = used if used is not None else set()
    labels = labels if labels is not None else {}
    for suggestion in suggestions or []:
        if not isinstance(suggestion, dict):
            continue
        key = suggestion.get("id")
        if not isinstance(key, str) or not key or key in used:
            continue
        number = _leading_number(suggestion.get("value"))
        if number is None or not _numbers_equal(
                quantity, number):
            continue
        keys = _word_tokens(key.replace("_", " "))
        keys |= _word_tokens(labels.get(key, ""))
        if row_tokens & keys:
            return suggestion
    return None


def _supported_selections(
        recipe: dict[str, Any],
        selected: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Selected suggestions known to this recipe + id labels.

    Only ids from the recipe's own suggestion list can
    justify an added row; anything else is ignored so a
    stale or forged id can never launder an addition.
    """
    labels = {
        item.get("id"): item.get("label") or ""
        for item in
        recipe.get("improvement_suggestions") or []
        if isinstance(item, dict)}
    kept = [
        item for item in selected or []
        if isinstance(item, dict)
        and item.get("id") in labels]
    return kept, labels


def validate_revision(
        recipe: dict[str, Any],
        revised: Any,
        allow_ingredient_changes: bool,
        selected: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Enforce shape, step stability, and ingredient protection.

    With ingredient changes authorized, the model may append
    rows for chef corrections that have no home row in the
    parent — but every added row must match a distinct
    selected suggestion by name and value. Anything else is
    rejected as an unsupported addition, and rows are never
    silently dropped.
    """
    violations = prompts.validate_output(prompts.RECIPE_REVISION,
                                         revised)
    if violations:
        raise RevisionError(
            f"Invalid revision payload: {'; '.join(violations)}")
    assert isinstance(revised, dict)
    rows = recipe.get("ingredients") or []
    new_rows = revised.get("ingredients")
    if not isinstance(new_rows, list):
        raise RevisionError(
            "Revision must return an ingredient list.")
    if not allow_ingredient_changes:
        if len(new_rows) != len(rows):
            raise RevisionError(
                "Revision must return one ingredient row per "
                "input row.")
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
    else:
        if len(new_rows) < len(rows):
            raise RevisionError(
                "Revision removed ingredient rows; deletions "
                "are not supported.")
        for position, new in enumerate(new_rows):
            if not isinstance(new, dict):
                raise RevisionError(
                    f"Ingredient row {position} is not an object.")
        supported, labels = _supported_selections(
            recipe, selected)
        used: set[str] = set()
        for position in range(len(rows), len(new_rows)):
            new = new_rows[position]
            match = match_chef_suggestion(
                new.get("name"), new.get("quantity"),
                supported, labels, used)
            if match is None:
                raise RevisionError(
                    f"Ingredient row {position} "
                    f"({(new.get('name') or '')[:60]!r}) is "
                    f"an unsupported addition: no matching "
                    f"selected chef correction.")
            used.add(match["id"])
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
    chosen, labels = _supported_selections(
        recipe, meta.get("suggestions"))
    merged_rows: list[dict[str, Any]] = []
    for position, new in enumerate(revised["ingredients"]):
        old_rows = snapshot.get("ingredients") or []
        old = old_rows[position] \
            if position < len(old_rows) else None
        if old is None:
            match = match_chef_suggestion(
                new.get("name"), new.get("quantity"),
                chosen, labels)
            basis = (
                f"Added from chef correction "
                f"{match['id']!r} ({match['value']}) in "
                f"{meta['recipe_id']}."
                if match is not None else
                "Added during authorized revision in "
                f"{meta['recipe_id']}.")
            merged_rows.append({
                "name": (new.get("name") or "").strip(),
                "quantity": new.get("quantity"),
                "unit": new.get("unit"),
                "quantity_status": "ai_estimated",
                "quantity_basis": basis,
                "evidence": [],
            })
            continue
        # Quantity status tracks the quantity: a pure
        # rename keeps the observed status, while a changed
        # quantity or unit is re-grounded below.
        quantity_changed = (
            not _numbers_equal(new.get("quantity"),
                               old.get("quantity"))
            or (new.get("unit") or "")
            != (old.get("unit") or ""))
        old["name"] = (new.get("name") or "").strip()
        old["quantity"] = new.get("quantity")
        old["unit"] = new.get("unit")
        old.pop("chef_edited", None)
        old.pop("original", None)
        if meta.get("allow_ingredient_changes") and quantity_changed:
            match = match_chef_suggestion(
                old["name"], old["quantity"],
                chosen, labels)
            old["quantity_status"] = "ai_estimated"
            if match is not None:
                old["quantity_basis"] = (
                    f"Set from chef correction "
                    f"{match['id']!r} ({match['value']}) in "
                    f"{meta['recipe_id']}.")
            else:
                old["quantity_basis"] = (
                    "Revised per explicit chef authorization in "
                    f"{meta['recipe_id']}.")
        merged_rows.append(old)
    snapshot["ingredients"] = merged_rows
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


def apply_deterministic_corrections(
        parsed: dict[str, Any],
        renames: list[tuple[str, str]],
        text_edits: list[tuple[str, int | None, str, str]],
        ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply audited corrections to parsed model output.

    ``renames`` maps an exact current ingredient name to its
    restored name; each must match exactly one row. ``text_edits``
    holds (section, index, old, new) tuples where section is one
    of "method" (index is the 1-based step number), "notes"
    (index is the 0-based note position), or "changes_summary"
    (index is None); each ``old`` string must occur exactly once
    in its target text. The input is never mutated; a corrected
    deep copy plus an applied-edit record is returned. Any
    mismatch raises RevisionError so persistence aborts instead
    of saving a partially corrected revision.
    """
    import copy

    corrected = copy.deepcopy(parsed)
    applied: list[dict[str, Any]] = []
    rows = corrected.get("ingredients")
    if not isinstance(rows, list):
        raise RevisionError(
            "Revision has no ingredient list to correct.")
    for current, restored in renames:
        hits = [row for row in rows
                if isinstance(row, dict)
                and row.get("name") == current]
        if len(hits) != 1:
            raise RevisionError(
                f"Rename {current!r} matches {len(hits)} rows, "
                f"expected exactly 1.")
        hits[0]["name"] = restored
        applied.append({"kind": "rename", "from": current,
                        "to": restored})
    steps = corrected.get("method")
    notes = corrected.get("notes")
    for section, index, old, new in text_edits:
        if section == "method":
            if not isinstance(steps, list) or not isinstance(
                    index, int) or index < 1 \
                    or index > len(steps):
                raise RevisionError(
                    f"Method step {index} is out of range.")
            target = steps[index - 1]
            field = "instruction"
        elif section == "notes":
            if not isinstance(notes, list) or not isinstance(
                    index, int) or index < 0 \
                    or index >= len(notes):
                raise RevisionError(
                    f"Note {index} is out of range.")
            target = None
            field = ""
        elif section == "changes_summary" and index is None:
            target = corrected
            field = "changes_summary"
        else:
            raise RevisionError(
                f"Unknown correction target: {section}.")
        if section == "notes":
            text = notes[index]  # type: ignore[index]
        else:
            text = (target or {}).get(field)  # type: ignore[union-attr]
        if not isinstance(text, str) or text.count(old) != 1:
            found = text.count(old) if isinstance(
                text, str) else 0
            raise RevisionError(
                f"Edit in {section} {index} matches {found} "
                f"text spans, expected exactly 1.")
        updated = text.replace(old, new)
        if section == "notes":
            notes[index] = updated  # type: ignore[index]
        else:
            target[field] = updated  # type: ignore[index]
        applied.append({"kind": "text", "section": section,
                        "index": index, "from": old, "to": new})
    return corrected, applied
