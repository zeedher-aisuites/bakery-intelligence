"""Chef-readable export labels (English/Spanish).

Technical provenance stays in the structured snapshot; exports
and the editor show short human language. Rules derive only
from recorded fields: chef overrides, AI-estimate flags, and
the quantity basis text (Displayed = visual, spoken = audio).
"""

from __future__ import annotations

from typing import Any

VISUAL_HINTS = ("Displayed", "display", "observed", "Observed",
                "Counted", "counted", "readable", "legible")
AUDIO_HINTS = ("spoken", "Spoken", "mentioned", "Mentioned",
               "narration", "speaker", "audio", "mencion")
# Intermediate preparations are detected only by explicit
# recorded wording ("from previous step", "added back"):
# the model names a product of earlier recipe steps, not a
# fresh raw material. Anything else stays a raw row.
INTERMEDIATE_HINTS = ("from previous step", "added back",
                      "from the previous step")

TEXT: dict[str, dict[str, str]] = {
    "en": {
        "ingredients": "Ingredients",
        "method": "Method",
        "notes": "Production notes",
        "col_ingredient": "Ingredient",
        "col_quantity": "Quantity",
        "col_unit": "Unit",
        "col_source": "Source",
        "not_determined": "Not determined",
        "seen_video": "Seen in video",
        "heard_audio": "Mentioned in audio",
        "seen_and_heard": "Seen in video + Mentioned in audio",
        "ai_suggested": "Suggested by AI",
        "chef_edited": "Edited by chef",
        "unknown": "Unknown",
        "draft": "Draft",
        "approved": "Approved",
        "clip1": "Part 1 - Preparation",
        "clip2": "Part 2 - Mixing",
        "ai_guidance": "AI suggestion",
        "subtitle": ("AI-generated bakery recipe draft. Review "
                     "and approve before use."),
        "intermediate_note": ("Prepared during this recipe; not "
                              "an additional raw material."),
    },
    "es": {
        "ingredients": "Ingredientes",
        "method": "Preparación",
        "notes": "Notas de producción",
        "col_ingredient": "Ingrediente",
        "col_quantity": "Cantidad",
        "col_unit": "Unidad",
        "col_source": "Origen",
        "not_determined": "No determinado",
        "seen_video": "Visto en el video",
        "heard_audio": "Mencionado en el audio",
        "seen_and_heard": "Visto en el video + Mencionado en el audio",
        "ai_suggested": "Sugerido por IA",
        "chef_edited": "Editado por el chef",
        "unknown": "Desconocido",
        "draft": "Borrador",
        "approved": "Aprobada",
        "clip1": "Parte 1 - Preparación",
        "clip2": "Parte 2 - Mezclado",
        "ai_guidance": "Sugerencia de IA",
        "subtitle": ("Borrador de receta generado por IA. Revisa y "
                     "aprueba antes de usar."),
        "intermediate_note": ("Preparado durante esta receta; no es "
                              "materia prima adicional."),
    },
}


def lang_of(recipe: dict[str, Any]) -> str:
    """Export language from the recipe record, default English."""
    language = recipe.get("language")
    return language if language in TEXT else "en"


def _hint_kinds(row: dict[str, Any]) -> tuple[bool, bool]:
    """Visual/audio evidence hints from recorded fields."""
    basis = str(row.get("quantity_basis") or "").lower()
    kinds = " ".join(
        str(e.get("kind") or "")
        for e in (row.get("evidence") or [])).lower()
    visual = any(h.lower() in basis for h in VISUAL_HINTS) or \
        "scale_display" in kinds
    audio = any(h.lower() in basis for h in AUDIO_HINTS)
    return visual, audio


def row_source(row: dict[str, Any], language: str) -> str:
    """One short provenance label for an ingredient row."""
    labels = TEXT[language]
    if row.get("chef_edited"):
        return labels["chef_edited"]
    status = row.get("quantity_status")
    if status == "ai_estimated":
        visual, audio = _hint_kinds(row)
        if visual and audio:
            return labels["seen_and_heard"]
        if audio:
            return labels["heard_audio"]
        if visual:
            return labels["seen_video"]
        return labels["ai_suggested"]
    if status == "seen_video":
        return labels["seen_video"]
    if status == "mentioned_audio":
        return labels["heard_audio"]
    if status in ("chef_supplied", "observed_count"):
        visual, audio = _hint_kinds(row)
        if visual and audio:
            return labels["seen_and_heard"]
        if audio:
            return labels["heard_audio"]
        if visual:
            return labels["seen_video"]
        return labels["unknown"]
    return labels["unknown"]


def adjustment_note(row: dict[str, Any], language: str) -> str | None:
    """Contextual line for draft roundings, e.g. flour 699/700."""
    adjustment = row.get("draft_adjustment")
    if not isinstance(adjustment, dict):
        return None
    shown = adjustment.get("displayed_value")
    draft = adjustment.get("draft_value")
    unit = row.get("unit") or ""
    if shown is None or draft is None:
        return None
    if language == "es":
        return (f"La báscula mostró {shown} {unit}; se "
                f"mencionaron {draft} {unit} en el audio.")
    return (f"Scale showed {shown} {unit}; {draft} {unit} was "
            f"mentioned in the audio.")


def clip_label(clip_id: Any, language: str) -> str:
    """Recording part without technical identifiers.

    Internal single-clip ids ("uploaded_video") and empty
    values render as "": callers skip the group header
    instead of printing a technical label.
    """
    labels = TEXT[language]
    text = str(clip_id or "").strip()
    if text.endswith("preparation"):
        return labels["clip1"]
    if text.endswith("mixing"):
        return labels["clip2"]
    if text in ("", "uploaded_video", "unknown clip",
                "unknown"):
        return ""
    return text


def split_display_name(name: Any) -> tuple[str, str | None]:
    """Primary name plus parenthetical qualifier, if present.

    Descriptive fusion names carry the uncertainty note in
    trailing parentheses ("white powder (audio describes
    as flour)"): the split is purely presentational and
    never drops text.
    """
    text = str(name or "").strip()
    if not text.endswith(")"):
        return text, None
    depth = 0
    outer: int | None = None
    for pos in range(len(text) - 1, -1, -1):
        char = text[pos]
        if char == ")":
            depth += 1
        elif char == "(":
            depth -= 1
            if depth == 0:
                outer = pos
    if outer is None:
        return text, None
    primary = text[:outer].strip()
    qualifier = text[outer + 1:-1].strip()
    if primary and qualifier:
        return primary, qualifier
    return text, None


def ingredient_role(row: dict[str, Any]) -> str:
    """Raw material vs intermediate preparation (display only).

    Returns "intermediate" only when the recorded name or
    basis explicitly describes a product of earlier steps;
    otherwise "raw". Stored quantities are untouched: an
    intermediate with a null quantity cannot double-count.
    """
    text = (str(row.get("name") or "") + " "
            + str(row.get("quantity_basis") or "")).lower()
    if any(h in text for h in INTERMEDIATE_HINTS):
        return "intermediate"
    return "raw"
