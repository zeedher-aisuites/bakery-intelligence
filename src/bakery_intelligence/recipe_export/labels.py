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
    },
}


def lang_of(recipe: dict[str, Any]) -> str:
    """Export language from the recipe record, default English."""
    language = recipe.get("language")
    return language if language in TEXT else "en"


def row_source(row: dict[str, Any], language: str) -> str:
    """One short provenance label for an ingredient row."""
    labels = TEXT[language]
    if row.get("chef_edited"):
        return labels["chef_edited"]
    status = row.get("quantity_status")
    if status == "ai_estimated":
        return labels["ai_suggested"]
    if status == "seen_video":
        return labels["seen_video"]
    if status == "mentioned_audio":
        return labels["heard_audio"]
    if status in ("chef_supplied", "observed_count"):
        basis = str(row.get("quantity_basis") or "")
        kinds = " ".join(
            str(e.get("kind") or "")
            for e in (row.get("evidence") or []))
        visual = any(h in basis for h in VISUAL_HINTS) or \
            "scale_display" in kinds
        audio = any(h in basis for h in AUDIO_HINTS)
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
    """Recording part without technical identifiers."""
    labels = TEXT[language]
    text = str(clip_id or "")
    if text.endswith("preparation"):
        return labels["clip1"]
    if text.endswith("mixing"):
        return labels["clip2"]
    return text or labels["unknown"]
