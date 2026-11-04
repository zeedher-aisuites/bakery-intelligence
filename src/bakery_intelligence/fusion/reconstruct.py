"""Recipe draft reconstruction from reconciled evidence.

Builds an ordered, fully referenced draft. Quantities stay null:
Phase 2A performs no tare or continuity analysis, so no net
quantity can be derived. The draft is never approved here.
"""

from __future__ import annotations

from typing import Any

from bakery_intelligence.fusion.models import STEP_COVERAGE_GAP_THRESHOLD

QUANTITY_NULL_REASON = (
    "Net quantity derivation requires verified tare state and "
    "ingredient continuity analysis, which Phase 2A does not "
    "perform. Displayed scale values are evidence, not quantities."
)


def _pick_title(global_activity: Any) -> tuple[str | None, str]:
    if isinstance(global_activity, str) and global_activity.strip():
        return global_activity.strip(), "ai_suggested"
    return None, "unknown"


def identity_status_for(names: list[str],
                        labels: list[str]) -> str:
    """Bucket a group: model_proposed, described, or unresolved.

    'confirmed' is reserved for human verification, which no P04
    record carries; it is never emitted by this offline pass.
    """
    if names:
        return "model_proposed"
    if labels:
        return "described"
    return "unresolved"


def build_ingredient_entries(
        groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Draft ingredient rows with null quantities and full refs."""
    entries: list[dict[str, Any]] = []
    for group in groups:
        members = group.get("members") or []
        names = sorted({(m.get("normalized_name") or "").strip()
                        for m in members if m.get("normalized_name")})
        labels = sorted({(m.get("visible_label") or "").strip()
                         for m in members if m.get("visible_label")})
        used_flags = sorted({(m.get("used_or_visible") or "").strip()
                             for m in members
                             if m.get("used_or_visible")})
        ambiguities: list[str] = []
        for member in members:
            ambiguities.extend(member.get("ambiguities") or [])
        status = identity_status_for(names, labels)
        display = names[0] if names else (
            labels[0] if labels else "unidentified observation")
        entry = {
            "group_id": group.get("group_id"),
            "identity_status": status,
            "display_label": display,
            "member_count": len(members),
            "names": names,
            "visible_labels": labels,
            "used_or_visible": used_flags,
            "quantity": None,
            "unit": None,
            "quantity_reason": QUANTITY_NULL_REASON,
            "supporting_timestamps": group.get(
                "supporting_timestamps") or [],
            "sources": group.get("sources") or [],
            "source_count": group.get("source_count") or 0,
            "member_ids": group.get("member_ids") or [],
            "possibly_related": group.get("possibly_related") or [],
            "ambiguities": sorted(set(ambiguities)),
            "status": "ai_suggested",
        }
        if group.get("clip_id"):
            entry["clip_id"] = group["clip_id"]
        entries.append(entry)
    return entries


def build_ingredient_review(
        entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Chef-reviewable candidate list, grouped by identity status.

    Named candidates first, then described-but-unidentified
    observations, then fully unresolved records. Every row keeps
    its group reference; nothing is merged or renamed here.
    """
    order = ("model_proposed", "described", "unresolved")
    sections: dict[str, list[dict[str, Any]]] = {
        key: [] for key in order}
    for entry in entries:
        status = entry.get("identity_status") or "unresolved"
        sections.setdefault(status, sections["unresolved"]).append({
            "group_id": entry.get("group_id"),
            "display_label": entry.get("display_label"),
            "member_count": entry.get("member_count") or 0,
            "sources": entry.get("sources") or [],
            "supporting_timestamps": entry.get(
                "supporting_timestamps") or [],
            "ambiguities": entry.get("ambiguities") or [],
            "possibly_related": entry.get("possibly_related") or [],
        })
    for rows in sections.values():
        rows.sort(key=lambda r: (r["display_label"] is None,
                                 r["display_label"] or "",
                                 r["group_id"] or ""))
    return {
        "counts": {key: len(sections[key]) for key in order},
        "named_candidates": sections["model_proposed"],
        "described_unidentified": sections["described"],
        "unresolved": sections["unresolved"],
    }


def build_steps(groups: list[dict[str, Any]],
                clip_order: dict[str, int] | None = None,
                ) -> list[dict[str, Any]]:
    """One ordered step per event group, actions preserved verbatim.

    Multi-clip sessions order by declared clip, then start;
    single-video runs keep the original start ordering.
    """
    order = clip_order or {}
    steps: list[dict[str, Any]] = []

    def key(group: dict[str, Any]) -> tuple:
        clip = group.get("clip_id")
        rank = order.get(clip, 1_000_000) if clip else 0
        return (rank,
                clip or "",
                group.get("start_seconds") is None,
                group.get("start_seconds") or 0.0,
                group.get("group_id") or "")

    ordered = sorted(groups, key=key)
    for position, group in enumerate(ordered, start=1):
        members = group.get("members") or []
        actions = sorted({(m.get("action_type") or "").strip()
                          for m in members if m.get("action_type")})
        ingredients: list[str] = []
        for member in members:
            ingredients.extend(member.get("ingredients") or [])
        uncertainties = sorted({(m.get("uncertainty") or "").strip()
                                for m in members
                                if m.get("uncertainty")})
        detail = sorted(
            ({
                "member_id": m.get("event_id"),
                "source": m.get("source"),
                "source_ref": m.get("source_ref"),
                "start_seconds": m.get("start_seconds"),
                "end_seconds": m.get("end_seconds"),
                "action": (m.get("action_type") or "").strip() or None,
            } for m in members),
            key=lambda d: (d["start_seconds"] is None,
                           d["start_seconds"] or 0.0,
                           d["member_id"] or ""))
        gap_note = _coverage_gap_note(detail)
        if gap_note:
            uncertainties = sorted(set(uncertainties) | {gap_note})
        step = {
            "step_index": position,
            "group_id": group.get("group_id"),
            "start_seconds": group.get("start_seconds"),
            "end_seconds": group.get("end_seconds"),
            "actions": actions,
            "ingredients": sorted(set(ingredients)),
            "sources": group.get("sources") or [],
            "source_count": group.get("source_count") or 0,
            "member_ids": group.get("member_ids") or [],
            "members": detail,
            "preliminary_links": group.get("preliminary_links") or [],
            "uncertainties": uncertainties,
            "status": "ai_suggested",
        }
        if group.get("clip_id"):
            step["clip_id"] = group["clip_id"]
        if group.get("speech_links"):
            step["speech_links"] = group["speech_links"]
        steps.append(step)
    return steps


def _coverage_gap_note(detail: list[dict[str, Any]]) -> str | None:
    """Flag internal gaps so phase-like groups read honestly."""
    spans = sorted(
        (d["start_seconds"], d["end_seconds"])
        for d in detail
        if isinstance(d["start_seconds"], (int, float))
        and isinstance(d["end_seconds"], (int, float))
        and d["end_seconds"] > d["start_seconds"])
    if len(spans) < 2:
        return None
    total = max(end for _, end in spans) - min(
        start for start, _ in spans)
    if total > 0:
        detail_spans = [s for s in spans
                        if s[1] - s[0] < 0.9 * total]
        if len(detail_spans) >= 2:
            spans = sorted(detail_spans)
        elif len(detail_spans) < 2:
            return None
    coverage_end = spans[0][1]
    widest = 0.0
    for start, end in spans[1:]:
        if start > coverage_end:
            widest = max(widest, start - coverage_end)
        coverage_end = max(coverage_end, end)
    if widest >= STEP_COVERAGE_GAP_THRESHOLD:
        return (f"Member evidence covers this step unevenly; widest "
                f"internal gap is {widest:.1f}s. Review sub-steps "
                f"separately.")
    return None


def build_questions(conflicts: list[dict[str, Any]],
                    ingredients: list[dict[str, Any]],
                    readings: list[dict[str, Any]],
                    preliminary: list[dict[str, Any]]) -> list[str]:
    """Chef review questions; deterministic order, no duplicates."""
    questions: list[str] = []
    for conflict in conflicts:
        kind = conflict.get("kind")
        detail = conflict.get("detail")
        if kind == "action_conflict":
            questions.append(
                f"Conflicting actions need review: {detail}.")
        elif kind == "reading_conflict":
            questions.append(
                f"Conflicting scale displays need review: {detail}.")
        elif kind == "ingredient_conflict":
            questions.append(
                f"Conflicting ingredient identities need review: "
                f"{detail}.")
    unnamed = sum(1 for entry in ingredients
                  if not entry.get("names"))
    if unnamed:
        questions.append(
            f"{unnamed} ingredient group(s) have no normalized name; "
            f"confirm identity from the video before use.")
    illegible = sum(1 for reading in readings
                    if reading.get("displayed_value") is None)
    if illegible:
        questions.append(
            f"{illegible} scale observation(s) have illegible displays; "
            f"no quantity can be derived from them.")
    for note in preliminary:
        if note.get("note_kind") == "segment_observation":
            continue
        uncertainty = (note.get("uncertainty") or "").strip()
        if uncertainty:
            questions.append(
                f"Preliminary note {note.get('source_ref')}: "
                f"{uncertainty}")
    seen: set[str] = set()
    unique: list[str] = []
    for question in sorted(questions):
        if question not in seen:
            seen.add(question)
            unique.append(question)
    return unique


def build_draft(global_activity: Any,
                event_groups: list[dict[str, Any]],
                ingredient_groups: list[dict[str, Any]],
                readings: list[dict[str, Any]],
                conflicts: list[dict[str, Any]],
                corroborated_pairs: list[dict[str, Any]],
                preliminary: list[dict[str, Any]],
                provenance: dict[str, Any],
                quantity_assessment: dict[str, Any] | None = None,
                clip_order: dict[str, int] | None = None,
                ) -> dict[str, Any]:
    """Assemble the reviewable draft with explicit unknowns."""
    title, title_status = _pick_title(global_activity)
    ingredients = build_ingredient_entries(ingredient_groups)
    steps = build_steps(event_groups, clip_order)
    questions = build_questions(conflicts, ingredients, readings,
                                preliminary)
    return {
        "schema": "fusion-recipe-draft/v1",
        "status": "draft_needs_review",
        "working_title": title,
        "working_title_status": title_status,
        "ingredients": ingredients,
        "ingredient_review": build_ingredient_review(ingredients),
        "steps": steps,
        "quantity_assessment": quantity_assessment or {
            "status": "not_evaluated"},
        "conflicts": conflicts,
        "corroborated_reading_pairs": [
            {**pair,
             "note": "Independent-model agreement is recorded here "
                     "for review attention; it is not verification."}
            for pair in corroborated_pairs
        ],
        "unresolved_questions": questions,
        "provenance": provenance,
    }
