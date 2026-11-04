"""Editable recipe state over a frozen proposed snapshot.

The v2 snapshot is immutable. Chef edits live in ``editor.json``
as overrides keyed by ingredient index and method number, with
first-seen originals, timestamps, and a change history. Approval
status is a plain persisted field: draft, approved, or
needs_changes. Exports always render the merged state.
"""

from __future__ import annotations

import copy
import json
import math
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RECIPE_ID = "bakery_demo_v1_proposed_v2"
RECIPES_DIRNAME = "recipes"
SNAPSHOT_FILENAME = "snapshot.json"
EDITOR_FILENAME = "editor.json"
RECORD_FILENAME = "record.json"
EDITOR_SCHEMA = "recipe-editor-state/1"

STATUSES = ("draft", "approved", "needs_changes")
LANGUAGES = ("en", "es")

MAX_NAME_CHARS = 120
MAX_UNIT_CHARS = 20
MAX_TEXT_CHARS = 2000
MAX_NOTE_CHARS = 500
MAX_RECIPE_NAME_CHARS = 120
MAX_INSTRUCTIONS_CHARS = 2000

RECIPE_ID_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}"

# Selectable improvement suggestions derived from the recipe's
# actual open items. Values the chef supplies travel with the
# revision request; nothing is invented when left blank.
IMPROVEMENT_SUGGESTIONS: tuple[dict[str, str], ...] = (
    {"id": "microwave-settings",
     "label_en": "Add microwave settings",
     "label_es": "Agregar ajustes del microondas",
     "hint_en": "Power and heating intervals, e.g. 800 W, "
                "3 intervals of 30 seconds.",
     "hint_es": "Potencia e intervalos, p. ej. 800 W, "
                "3 intervalos de 30 segundos.",
     "placeholder_en": "800 W, 3 intervals of 30 seconds",
     "placeholder_es": "800 W, 3 intervalos de 30 segundos"},
    {"id": "chill-params",
     "label_en": "Add chilling temperature and duration",
     "label_es": "Agregar temperatura y tiempo de enfriado",
     "hint_en": "Fridge temperature and chill time for the paste.",
     "hint_es": "Temperatura y tiempo de enfriado de la pasta.",
     "placeholder_en": "4 C, 30 minutes",
     "placeholder_es": "4 C, 30 minutos"},
    {"id": "rest-time",
     "label_en": "Add yeast resting time",
     "label_es": "Agregar tiempo de reposo de la levadura",
     "hint_en": "How long the yeast mixture should rest.",
     "hint_es": "Cuánto debe reposar la mezcla de levadura.",
     "placeholder_en": "15 minutes until foamy",
     "placeholder_es": "15 minutos hasta que espume"},
    {"id": "mixer-params",
     "label_en": "Add mixer speed and kneading duration",
     "label_es": "Agregar velocidad y tiempo de amasado",
     "hint_en": "Stand-mixer speed and kneading time.",
     "hint_es": "Velocidad de la batidora y tiempo de amasado.",
     "placeholder_en": "Speed 2, 8 minutes",
     "placeholder_es": "Velocidad 2, 8 minutos"},
    {"id": "shaping-baking",
     "label_en": "Add shaping and baking instructions",
     "label_es": "Agregar formado y horneado",
     "hint_en": "Shaping, proofing, oven temperature and bake time.",
     "hint_es": "Formado, fermentado, temperatura y tiempo.",
     "placeholder_en": "Roll, cut 8 pieces, bake 180 C for 22 minutes",
     "placeholder_es": "Enrollar, cortar 8 piezas, hornear 180 C, "
                        "22 minutos"},
)


class RecipeEditorError(ValueError):
    """A recipe request or stored record is invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validated_id(recipe_id: Any) -> str:
    if not isinstance(recipe_id, str) or not re.fullmatch(
            RECIPE_ID_PATTERN, recipe_id):
        raise RecipeEditorError(f"Unknown recipe: {recipe_id!r}.")
    return recipe_id


def recipe_dir(root: Path, recipe_id: str) -> Path:
    return root / RECIPES_DIRNAME / _validated_id(recipe_id)


def read_record(root: Path, recipe_id: str) -> dict[str, Any] | None:
    path = recipe_dir(root, recipe_id) / RECORD_FILENAME
    if not path.is_file():
        return None
    stored = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(stored, dict):
        raise RecipeEditorError("Recipe record is not an object.")
    return stored


def write_record(root: Path, recipe_id: str,
                 record: dict[str, Any]) -> None:
    path = recipe_dir(root, recipe_id) / RECORD_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True)
                    + "\n", encoding="utf-8")


def create_record(root: Path, name: str, language: str,
                  source_run_id: str | None = None,
                  parent_version: str | None = None,
                  recipe_id: str | None = None) -> dict[str, Any]:
    """Persist a recipe creation request (no inference here)."""
    name = _clean_text(name, MAX_RECIPE_NAME_CHARS, "Recipe name")
    if language not in LANGUAGES:
        raise RecipeEditorError(
            f"Unsupported language: {language!r}.")
    if recipe_id is None:
        recipe_id = f"recipe_{secrets.token_hex(4)}"
    else:
        _validated_id(recipe_id)
    now = utc_now()
    record = {
        "recipe_id": recipe_id,
        "name": name,
        "language": language,
        "source_run_id": source_run_id,
        "parent_version": parent_version,
        "has_snapshot": False,
        "created_at": now,
        "updated_at": now,
    }
    write_record(root, recipe_id, record)
    return record


def touch_record(root: Path, recipe_id: str, **fields: Any) -> None:
    record = read_record(root, recipe_id) or {"recipe_id": recipe_id}
    record.update(fields)
    record["updated_at"] = utc_now()
    write_record(root, recipe_id, record)


def list_recipes(root: Path) -> list[dict[str, Any]]:
    """Newest-first recipe cards with live counts and status."""
    base = root / RECIPES_DIRNAME
    if not base.is_dir():
        return []
    cards: list[dict[str, Any]] = []
    for child in sorted(base.iterdir(), key=lambda p: p.name):
        record_path = child / RECORD_FILENAME
        snapshot_path = child / SNAPSHOT_FILENAME
        if not record_path.is_file() and not snapshot_path.is_file():
            continue
        record = json.loads(record_path.read_text(encoding="utf-8")) \
            if record_path.is_file() else {}
        if not isinstance(record, dict):
            continue
        card: dict[str, Any] = {
            "recipe_id": child.name,
            "name": record.get("name") or child.name,
            "language": record.get("language") or "en",
            "source_run_id": record.get("source_run_id"),
            "parent_version": record.get("parent_version"),
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
            "ingredients": 0,
            "steps": 0,
            "has_snapshot": snapshot_path.is_file(),
            "status": "processing",
            "build_error": None,
        }
        if not snapshot_path.is_file():
            for filename in ("unavailable.json",
                             "build_error.json"):
                marker = child / filename
                if marker.is_file():
                    try:
                        detail = json.loads(marker.read_text(
                            encoding="utf-8"))
                    except (OSError, ValueError):
                        detail = {}
                    card["status"] = "failed"
                    card["build_error"] = (
                        detail.get("reason")
                        or detail.get("error")
                        or "Recipe generation failed.")
                    break
        snap_title = None
        if snapshot_path.is_file():
            try:
                snapshot = json.loads(
                    snapshot_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                snapshot = {}
            if isinstance(snapshot, dict):
                snap_title = snapshot.get("title")
            card["ingredients"] = len(
                snapshot.get("ingredients") or [])
            card["steps"] = len(snapshot.get("method") or [])
            if not record.get("name"):
                card["name"] = snapshot.get("title") or child.name
            try:
                editor = read_editor(root, child.name)
                card["status"] = editor.get("recipe_status") or "draft"
                if editor.get("last_saved_at"):
                    card["updated_at"] = editor["last_saved_at"]
            except RecipeEditorError:
                card["status"] = "draft"
        display, custom = effective_display_name(
            record if isinstance(record, dict) else {},
            snap_title)
        if not display:
            display = child.name
        card["display_name"] = display
        card["has_custom_name"] = custom
        cards.append(card)
    cards.sort(key=lambda c: (c.get("updated_at") or "",
                              c.get("created_at") or "",
                              c["recipe_id"]), reverse=True)
    overlay_lineage(cards)
    return cards


def effective_display_name(
        record: dict[str, Any] | None,
        snapshot_title: Any = None) -> tuple[str, bool]:
    """User override, else request name, else snapshot title.

    Returns (name, has_custom_name). Snapshots are never
    consulted for writing; renames live in record.json only.
    """
    record = record if isinstance(record, dict) else {}
    custom = record.get("display_name")
    if isinstance(custom, str) and custom.strip():
        return custom.strip(), True
    name = record.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip(), False
    if isinstance(snapshot_title, str) and snapshot_title.strip():
        return snapshot_title.strip(), False
    recipe_id = record.get("recipe_id")
    if isinstance(recipe_id, str) and recipe_id:
        return recipe_id, False
    return "", False


def group_families(
        cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group cards by persisted lineage; number versions 1..N.

    Pure function over card dicts (recipe_id, parent_version,
    created_at). Missing parents and lineage cycles degrade to
    singleton families instead of merging unrelated recipes.
    Numbers follow creation order (created_at, recipe_id).
    """
    by_id = {c["recipe_id"]: c for c in cards}

    def root_of(start: str) -> str:
        seen: set[str] = set()
        current = start
        while True:
            if current in seen:
                return start
            seen.add(current)
            parent = by_id.get(current, {}).get(
                "parent_version")
            if not parent or parent not in by_id:
                return current
            current = parent

    members: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        members.setdefault(root_of(card["recipe_id"]),
                           []).append(card)
    families: list[dict[str, Any]] = []
    for root, group in members.items():
        ordered = sorted(
            group,
            key=lambda c: (c.get("created_at") or "",
                           c["recipe_id"]))
        numbers = {c["recipe_id"]: n + 1
                   for n, c in enumerate(ordered)}
        families.append({
            "family_id": root,
            "family_size": len(ordered),
            "members": [{
                "recipe_id": c["recipe_id"],
                "version_number": numbers[c["recipe_id"]],
                "parent_version": c.get("parent_version"),
                "parent_version_number": numbers.get(
                    c.get("parent_version") or ""),
            } for c in ordered],
        })
    families.sort(
        key=lambda f: (ordered_key(f, by_id)), reverse=True)
    return families


def ordered_key(family: dict[str, Any],
                by_id: dict[str, dict[str, Any]]) -> tuple:
    """Newest-member-first ordering for families."""
    latest = family["members"][-1]["recipe_id"]
    card = by_id.get(latest, {})
    return (card.get("updated_at") or "",
            card.get("created_at") or "", latest)


def overlay_lineage(cards: list[dict[str, Any]]) -> None:
    """Attach family/version/display fields to cards in place."""
    index: dict[str, dict[str, Any]] = {}
    for family in group_families(cards):
        for member in family["members"]:
            index[member["recipe_id"]] = {
                "family_id": family["family_id"],
                "family_size": family["family_size"],
                "version_number":
                member["version_number"],
                "parent_version_number":
                member["parent_version_number"],
                "is_latest": member["version_number"]
                == family["family_size"],
            }
    for card in cards:
        card.update(index.get(card["recipe_id"], {
            "family_id": card["recipe_id"],
            "family_size": 1,
            "version_number": 1,
            "parent_version_number": None,
            "is_latest": True,
        }))


def rename_recipe(root: Path, recipe_id: str,
                  name: Any) -> dict[str, Any]:
    """Set or clear a per-version display-name override.

    Metadata only: no new version, no inference, snapshots
    untouched. ``None`` clears the override; anything else
    must be a 1-120 character string after trimming.
    """
    _validated_id(recipe_id)
    record = read_record(root, recipe_id)
    if record is None:
        raise RecipeEditorError(
            f"Unknown recipe: {recipe_id!r}.")
    if name is None:
        record.pop("display_name", None)
        record["updated_at"] = utc_now()
        write_record(root, recipe_id, record)
        return record
    record["display_name"] = _clean_text(
        name, MAX_RECIPE_NAME_CHARS, "Recipe name")
    record["updated_at"] = utc_now()
    write_record(root, recipe_id, record)
    return record


def read_snapshot(root: Path, recipe_id: str) -> dict[str, Any]:
    path = recipe_dir(root, recipe_id) / SNAPSHOT_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing recipe snapshot: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _editor_path(root: Path, recipe_id: str) -> Path:
    return recipe_dir(root, recipe_id) / EDITOR_FILENAME


def read_editor(root: Path, recipe_id: str) -> dict[str, Any]:
    """Return editor state; a missing file means untouched draft."""
    path = _editor_path(root, recipe_id)
    if not path.is_file():
        return {
            "schema_version": EDITOR_SCHEMA,
            "recipe_status": "draft",
            "status_note": None,
            "status_updated_at": None,
            "last_saved_at": None,
            "ingredient_overrides": {},
            "method_overrides": {},
            "history": [],
        }
    stored = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(stored, dict):
        raise RecipeEditorError("Editor state is not an object.")
    return stored


def write_editor(root: Path, recipe_id: str,
                 state: dict[str, Any]) -> None:
    path = _editor_path(root, recipe_id)
    path.write_text(json.dumps(state, indent=2, sort_keys=True)
                    + "\n", encoding="utf-8")


def merged_recipe(root: Path, recipe_id: str) -> dict[str, Any]:
    """Snapshot plus chef overrides, export-ready shape."""
    snapshot = read_snapshot(root, recipe_id)
    editor = read_editor(root, recipe_id)
    recipe = copy.deepcopy(snapshot)
    language = (read_record(root, recipe_id) or {}).get(
        "language") or "en"
    if language not in LANGUAGES:
        language = "en"
    recipe["language"] = language
    localized = f"instruction_{language}"
    for item in recipe.get("method") or []:
        if isinstance(item, dict) and item.get(localized):
            item["instruction"] = item[localized]
    notes_key = f"notes_{language}"
    if recipe.get(notes_key):
        recipe["limitations"] = list(recipe[notes_key])
    ingredient_edits = editor.get("ingredient_overrides") or {}
    for position, row in enumerate(recipe.get("ingredients") or []):
        override = ingredient_edits.get(str(position))
        row["chef_edited"] = False
        if isinstance(override, dict):
            for field in ("name", "quantity", "unit"):
                if field in override:
                    row[field] = override[field]
            row["chef_edited"] = True
            row["original"] = override.get("original")
    method_edits = editor.get("method_overrides") or {}
    for item in recipe.get("method") or []:
        override = method_edits.get(str(item.get("number")))
        item["chef_edited"] = False
        if isinstance(override, dict) and "instruction" in override:
            item["instruction"] = override["instruction"]
            item["chef_edited"] = True
            item["original"] = override.get("original")
    recipe["recipe_status"] = editor.get("recipe_status") or "draft"
    recipe["status_note"] = editor.get("status_note")
    recipe["status_updated_at"] = editor.get("status_updated_at")
    recipe["last_saved_at"] = editor.get("last_saved_at")
    record = read_record(root, recipe_id) or {}
    recipe["recipe_name"] = record.get("name") or recipe.get("title")
    recipe["language"] = record.get("language") or "en"
    recipe["parent_version"] = record.get("parent_version")
    parent_name = None
    if record.get("parent_version"):
        try:
            parent = read_record(root, record["parent_version"])
        except RecipeEditorError:
            parent = None
        if parent:
            parent_name = parent.get("name")
    recipe["parent_name"] = parent_name
    if isinstance(snapshot.get("improvement_suggestions"), list) \
            and snapshot["improvement_suggestions"]:
        recipe["improvement_suggestions"] = copy.deepcopy(
            snapshot["improvement_suggestions"])
    else:
        recipe["improvement_suggestions"] = [
            dict(item) for item in IMPROVEMENT_SUGGESTIONS]
    display, custom = effective_display_name(
        record, snapshot.get("title"))
    recipe["display_name"] = display or recipe.get("title")
    recipe["has_custom_name"] = custom
    recipe["version_info"] = version_info_for(root, recipe_id)
    return recipe


def version_info_for(root: Path,
                     recipe_id: str) -> dict[str, Any]:
    """Family position and navigable history for one recipe."""
    cards = {c["recipe_id"]: c for c in list_recipes(root)}
    current = cards.get(recipe_id, {})
    family_id = current.get("family_id") or recipe_id
    members = sorted(
        (c for c in cards.values()
         if (c.get("family_id") or c["recipe_id"])
         == family_id),
        key=lambda c: (c.get("version_number") or 0,
                       c["recipe_id"]))
    return {
        "family_id": family_id,
        "family_size": current.get("family_size") or 1,
        "version_number": current.get("version_number") or 1,
        "is_latest": bool(current.get("is_latest", True)),
        "parent_version": current.get("parent_version"),
        "parent_version_number": current.get(
            "parent_version_number"),
        "members": [{
            "recipe_id": c["recipe_id"],
            "version_number": c.get("version_number") or 1,
            "display_name": c.get("display_name")
            or c["recipe_id"],
            "status": c.get("status") or "draft",
            "parent_version": c.get("parent_version"),
        } for c in members],
    }


def _clean_text(value: Any, maximum: int, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecipeEditorError(f"{label} must be a non-empty string.")
    text = value.strip()
    if len(text) > maximum:
        raise RecipeEditorError(
            f"{label} exceeds {maximum} characters.")
    return text


def _clean_quantity(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecipeEditorError("Quantity must be a number or null.")
    if not math.isfinite(value) or value < 0:
        raise RecipeEditorError(
            "Quantity must be a finite non-negative number or null.")
    return value


def save_edits(root: Path, recipe_id: str,
               ingredients: Any,
               method: Any) -> dict[str, Any]:
    """Validate and persist chef edits; returns the merged recipe."""
    snapshot = read_snapshot(root, recipe_id)
    rows = snapshot.get("ingredients") or []
    steps = {item.get("number"): item
             for item in snapshot.get("method") or []}
    if not isinstance(ingredients, list) or \
            not isinstance(method, list):
        raise RecipeEditorError(
            "Save payload needs ingredient and method lists.")
    editor = read_editor(root, recipe_id)
    ingredient_edits = editor.setdefault("ingredient_overrides", {})
    method_edits = editor.setdefault("method_overrides", {})
    history = editor.setdefault("history", [])
    now = utc_now()

    for entry in ingredients:
        if not isinstance(entry, dict):
            raise RecipeEditorError("Ingredient edit is not an object.")
        index = entry.get("index")
        if not isinstance(index, int) or \
                not 0 <= index < len(rows):
            raise RecipeEditorError(
                f"Unknown ingredient index: {index!r}.")
        row = rows[index]
        key = str(index)
        override = ingredient_edits.setdefault(key, {"original": {
            "name": row.get("name"), "quantity": row.get("quantity"),
            "unit": row.get("unit")}})
        changes: dict[str, Any] = {}
        if "name" in entry and entry["name"] is not None:
            changes["name"] = _clean_text(
                entry["name"], MAX_NAME_CHARS, "Ingredient name")
        if "quantity" in entry:
            changes["quantity"] = _clean_quantity(entry["quantity"])
        if "unit" in entry and entry["unit"] is not None:
            changes["unit"] = _clean_text(
                entry["unit"], MAX_UNIT_CHARS, "Unit")
        if not changes:
            raise RecipeEditorError(
                f"Ingredient edit {index} changes nothing.")
        before = {field: row.get(field)
                  if field not in override else override[field]
                  for field in changes}
        override.update(changes)
        history.append({"at": now, "kind": "ingredient",
                        "ref": index, "before": before,
                        "after": dict(changes)})

    for entry in method:
        if not isinstance(entry, dict):
            raise RecipeEditorError("Method edit is not an object.")
        number = entry.get("number")
        if number not in steps:
            raise RecipeEditorError(
                f"Unknown method step: {number!r}.")
        if "instruction" not in entry:
            raise RecipeEditorError(
                f"Method edit {number} changes nothing.")
        text = _clean_text(entry["instruction"], MAX_TEXT_CHARS,
                           "Instruction")
        key = str(number)
        override = method_edits.setdefault(key, {"original": {
            "instruction": steps[number].get("instruction")}})
        before = override.get("instruction",
                              steps[number].get("instruction"))
        language = (read_record(root, recipe_id) or {}).get(
            "language") or "en"
        if language not in LANGUAGES:
            language = "en"
        override["instruction"] = text
        override[f"instruction_{language}"] = text
        history.append({"at": now, "kind": "method", "ref": number,
                        "before": {"instruction": before},
                        "after": {"instruction": text}})

    editor["last_saved_at"] = now
    write_editor(root, recipe_id, editor)
    return merged_recipe(root, recipe_id)


def set_status(root: Path, recipe_id: str, status: Any,
               note: Any = None) -> dict[str, Any]:
    """Persist draft/approved/needs-changes status; no identities."""
    if status not in STATUSES:
        raise RecipeEditorError(
            f"Unknown recipe status: {status!r}.")
    if note is not None:
        if not isinstance(note, str):
            raise RecipeEditorError("Status note must be a string.")
        note = note.strip() or None
        if note is not None and len(note) > MAX_NOTE_CHARS:
            raise RecipeEditorError(
                f"Status note exceeds {MAX_NOTE_CHARS} characters.")
    read_snapshot(root, recipe_id)
    editor = read_editor(root, recipe_id)
    editor["recipe_status"] = status
    editor["status_note"] = note
    editor["status_updated_at"] = utc_now()
    history = editor.setdefault("history", [])
    history.append({"at": editor["status_updated_at"],
                    "kind": "status", "ref": status,
                    "before": None,
                    "after": {"status": status, "note": note}})
    write_editor(root, recipe_id, editor)
    return merged_recipe(root, recipe_id)
