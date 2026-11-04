"""A4 editorial rendering of a recipe snapshot (fpdf2, core fonts).

Core Helvetica keeps the build reproducible on any host; text is
sanitized to latin-1 so diacritics survive and no glyph ever goes
missing. Warm neutrals and terracotta follow the product palette.
"""

from __future__ import annotations

from typing import Any

from fpdf import FPDF

from bakery_intelligence.recipe_export.labels import (
    TEXT,
    adjustment_note,
    clip_label,
    lang_of,
    row_source,
)

INK = (26, 24, 21)
MUTED = (110, 102, 92)
TERRA = (167, 92, 67)
LINE = (231, 225, 216)
CREAM = (245, 245, 243)
WHITE = (255, 255, 255)

_REPLACEMENTS = {
    "\u2013": "-",
    "\u2014": "-",
    "\u2022": "-",
    "\u2026": "...",
    "\u00a0": " ",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
}


def sanitize(text: Any) -> str:
    """Fold text into latin-1 without losing diacritics."""
    if text is None:
        return ""
    out = str(text)
    for old, new in _REPLACEMENTS.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def fmt_amount(value: Any, language: str = "en") -> str:
    """Compact number: 100.0 renders as 100, never 100.000."""
    if value is None:
        return TEXT[language]["not_determined"]
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


class RecipePDF(FPDF):
    """A4 recipe document with header/footer chrome."""

    def __init__(self, doc_id: str) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.doc_id = sanitize(doc_id)
        self.set_auto_page_break(True, margin=22)
        self.set_margins(22, 18, 22)

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 8, self.doc_id, align="R", new_x="LMARGIN",
                  new_y="NEXT")
        self.set_draw_color(*LINE)
        self.line(self.l_margin, self.get_y(),
                  self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self) -> None:
        self.set_y(-16)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 8, f"Page {self.page_no()} / {{nb}}",
                  align="C")

    def section_title(self, text: str) -> None:
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*TERRA)
        self.cell(0, 9, sanitize(text).upper(), new_x="LMARGIN",
                  new_y="NEXT")
        self.set_draw_color(*TERRA)
        self.set_line_width(0.5)
        self.line(self.l_margin, self.get_y(),
                  self.l_margin + 28, self.get_y())
        self.set_line_width(0.2)
        self.ln(4)

    def body(self, text: str, bold: bool = False,
             size: int = 10) -> None:
        self.set_font("Helvetica", "B" if bold else "", size)
        self.set_text_color(*INK)
        self.multi_cell(0, 5.2, sanitize(text), new_x="LMARGIN",
                        new_y="NEXT")
        self.ln(1)

    def muted(self, text: str, size: int = 9) -> None:
        self.set_font("Helvetica", "I", size)
        self.set_text_color(*MUTED)
        self.multi_cell(0, 4.6, sanitize(text), new_x="LMARGIN",
                        new_y="NEXT")
        self.ln(1)


def render_ingredients(pdf: RecipePDF,
                       ingredients: list[dict[str, Any]],
                       language: str) -> None:
    """Clean table: name, quantity, unit, source."""
    labels = TEXT[language]
    widths = [62, 34, 22, 48]
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(*INK)
    pdf.set_text_color(*WHITE)
    for title, width in zip(
            [labels["col_ingredient"], labels["col_quantity"],
             labels["col_unit"], labels["col_source"]], widths):
        pdf.cell(width, 7, title, border=0, fill=True)
    pdf.ln()
    pdf.set_text_color(*INK)
    for position, row in enumerate(ingredients):
        fill = position % 2 == 1
        if fill:
            pdf.set_fill_color(*CREAM)
        pdf.set_font("Helvetica", "B", 10)
        qty = fmt_amount(row.get("quantity"), language)
        detail = adjustment_note(row, language)
        cells = [sanitize(row.get("name")), qty,
                 sanitize(row.get("unit")) or "-",
                 sanitize(row_source(row, language))]
        height = 12 if detail else 7
        x_start = pdf.get_x()
        y_start = pdf.get_y()
        if y_start + height > pdf.page_break_trigger:
            pdf.add_page()
            y_start = pdf.get_y()
        for width, text in zip(widths, cells):
            pdf.set_xy(x_start, y_start)
            pdf.set_font("Helvetica",
                         "B" if width == widths[0] else "", 10)
            pdf.multi_cell(width, 7 if height == 7 else 6, text,
                           border=0, fill=fill)
            x_start += width
        if detail:
            pdf.set_x(pdf.l_margin + widths[0])
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(*MUTED)
            pdf.multi_cell(widths[1] + widths[2] + widths[3], 5,
                           sanitize(detail), fill=fill)
            pdf.set_text_color(*INK)
        pdf.set_xy(pdf.l_margin, max(pdf.get_y(), y_start + 7))
    pdf.ln(2)


def render_method(pdf: RecipePDF,
                  method: list[dict[str, Any]],
                  language: str) -> None:
    """Numbered instructions grouped by recording part."""
    labels = TEXT[language]
    current_clip: str | None = None
    for item in method:
        clip = item.get("clip_id") or "unknown clip"
        if clip != current_clip:
            current_clip = clip
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*TERRA)
            pdf.cell(0, 7, sanitize(clip_label(clip, language)),
                     new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 22)
        pdf.set_text_color(*LINE)
        number = str(item.get("number"))
        pdf.cell(12, 8, number, new_x="END")
        left = pdf.get_x()
        pdf.set_xy(left, pdf.get_y())
        pdf.set_text_color(*INK)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5.2, sanitize(item.get("instruction")),
                         new_x="LMARGIN", new_y="NEXT")
        edited = labels["chef_edited"] if item.get(
            "chef_edited") else ""
        if edited:
            pdf.set_x(left)
            pdf.set_font("Helvetica", "I", 8.5)
            pdf.set_text_color(*MUTED)
            pdf.multi_cell(0, 4.4, sanitize(edited),
                           new_x="LMARGIN", new_y="NEXT")
        guidance = item.get("ai_guidance") or {}
        if guidance.get("text"):
            pdf.set_x(left)
            pdf.set_font("Helvetica", "I", 8.5)
            pdf.set_text_color(*MUTED)
            pdf.multi_cell(
                0, 4.4,
                f"{labels['ai_guidance']}: "
                f"{sanitize(guidance['text'])}",
                new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(*INK)
        pdf.ln(2)


def render_bullets(pdf: RecipePDF, items: list[str]) -> None:
    """Compact bullet list for notes and limitations."""
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(*INK)
    for text in items:
        x = pdf.get_x()
        pdf.cell(6, 5, "-")
        pdf.multi_cell(0, 5, sanitize(text), new_x="LMARGIN",
                         new_y="NEXT")
        pdf.set_x(x)
    pdf.ln(2)


def build_pdf(snapshot: dict[str, Any]) -> bytes:
    """Render the full document; returns the PDF bytes."""
    language = lang_of(snapshot)
    labels = TEXT[language]
    pdf = RecipePDF(str(snapshot.get("snapshot_id")))
    pdf.alias_nb_pages("{nb}")
    pdf.add_page()

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 6, sanitize(str(snapshot.get("snapshot_id"))),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(*INK)
    pdf.multi_cell(0, 11, sanitize(snapshot.get("title")),
                     new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(0, 5.5, sanitize(labels["subtitle"]),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_fill_color(*CREAM)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*INK)
    status_label = {"approved": labels["approved"],
                    "draft": labels["draft"]}.get(
                        snapshot.get("recipe_status") or "draft",
                        snapshot.get("recipe_status") or "draft")
    meta = [
        f"{labels['col_source']}: {snapshot.get('snapshot_id')}",
        f"Status: {status_label}",
    ]
    if snapshot.get("parent_version"):
        meta.append(f"Based on: {snapshot.get('parent_version')}")
    y = pdf.get_y()
    pdf.set_xy(pdf.l_margin, y)
    for line in meta:
        pdf.cell(0, 5.5, sanitize(line), new_x="LMARGIN",
                 new_y="NEXT", fill=True)
    pdf.ln(4)

    pdf.section_title(labels["ingredients"])
    render_ingredients(pdf, snapshot.get("ingredients") or [],
                       language)

    pdf.section_title(labels["method"])
    render_method(pdf, snapshot.get("method") or [], language)

    notes = list(snapshot.get("limitations") or [])
    if notes:
        pdf.section_title(labels["notes"])
        render_bullets(pdf, notes)

    data = pdf.output()
    return bytes(data)
