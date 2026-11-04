"""Editable Word export of the merged recipe state (python-docx).

Same content contract as the PDF: title, status, ingredient
table, numbered method with AI suggestions, and production
notes. No evidence appendix. Fully editable in Microsoft Word.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

from bakery_intelligence.recipe_export.labels import (
    TEXT,
    adjustment_note,
    clip_label,
    lang_of,
    row_source,
)

INK = RGBColor(0x1A, 0x18, 0x15)
MUTED = RGBColor(0x6E, 0x66, 0x5C)
TERRA = RGBColor(0xA7, 0x5C, 0x43)


def _amount(value: Any, language: str) -> str:
    if value is None:
        return TEXT[language]["not_determined"]
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _heading(document: Document, text: str, level: int = 1) -> None:
    run = document.add_heading(level=level).add_run(text)
    run.font.color.rgb = TERRA if level == 1 else INK


def build_docx(recipe: dict[str, Any]) -> bytes:
    """Render the merged recipe as editable DOCX bytes."""
    language = lang_of(recipe)
    labels = TEXT[language]
    document = Document()
    style = document.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10)
    style.font.color.rgb = INK

    status_label = {"approved": labels["approved"],
                    "draft": labels["draft"]}.get(
                        recipe.get("recipe_status") or "draft",
                        recipe.get("recipe_status") or "draft")
    title = document.add_heading(level=0)
    title.add_run(str(recipe.get("title") or "Untitled recipe"))
    meta = document.add_paragraph()
    meta.add_run(
        f"{recipe.get('snapshot_id')}\n"
        f"Status: {status_label}"
        + (f"\nBased on: {recipe.get('parent_version')}"
           if recipe.get("parent_version") else "")
        + f"\n{labels['subtitle']}").italic = True

    _heading(document, labels["ingredients"])
    rows = recipe.get("ingredients") or []
    table = document.add_table(rows=1 + len(rows), cols=4)
    table.style = "Light Grid Accent 1"
    for cell, text in zip(table.rows[0].cells,
                          [labels["col_ingredient"],
                           labels["col_quantity"],
                           labels["col_unit"],
                           labels["col_source"]]):
        cell.text = text
    for table_row, row in zip(table.rows[1:], rows):
        values = [str(row.get("name") or ""),
                  _amount(row.get("quantity"), language),
                  str(row.get("unit") or "-"),
                  row_source(row, language)]
        for cell, value in zip(table_row.cells, values):
            cell.text = value
        detail = adjustment_note(row, language)
        if detail:
            note = document.add_paragraph()
            note.add_run(f"{row.get('name')}: ").bold = True
            note.add_run(detail).italic = True

    _heading(document, labels["method"])
    current_clip: str | None = None
    for item in recipe.get("method") or []:
        clip = item.get("clip_id") or ""
        if clip != current_clip:
            current_clip = clip
            _heading(document, clip_label(clip, language), level=2)
        paragraph = document.add_paragraph(
            style="List Number")
        run = paragraph.add_run(str(item.get("instruction") or ""))
        if item.get("chef_edited"):
            run.bold = True
        if item.get("chef_edited"):
            edited = document.add_paragraph()
            edited.add_run(labels["chef_edited"]).italic = True
        guidance = item.get("ai_guidance") or {}
        if guidance.get("text"):
            tip = document.add_paragraph()
            tip.add_run(f"{labels['ai_guidance']}: ").bold = True
            tip.add_run(str(guidance["text"])).italic = True

    notes = list(recipe.get("limitations") or [])
    if notes:
        _heading(document, labels["notes"])
        for text in notes:
            document.add_paragraph(text, style="List Bullet")

    for section in document.sections:
        section.top_margin = Pt(54)
        section.bottom_margin = Pt(54)
    footer = document.sections[0].footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run(str(recipe.get("snapshot_id"))).italic = True

    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()
