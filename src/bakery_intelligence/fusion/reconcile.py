"""Deterministic reconciliation of normalized evidence.

Groups overlapping observations, flags conflicts, and counts
corroboration. Agreement NEVER promotes a record's status: every
reconciled node keeps ai_suggested (or unknown) until a human
decides otherwise in the review workflow.
"""

from __future__ import annotations

from typing import Any

from bakery_intelligence.fusion.models import (
    CONFLICT_IOU_THRESHOLD,
    INGREDIENT_MATCH_TOLERANCE,
    NESTING_CONTAINMENT_THRESHOLD,
    OVERLAP_IOU_THRESHOLD,
    PRELIMINARY_LINK_TOLERANCE,
    READING_MATCH_TOLERANCE,
    SOURCE_PRELIMINARY,
    stable_id,
)
from bakery_intelligence.fusion.normalize import normalize_label


def interval_iou(start_a: float | None, end_a: float | None,
                 start_b: float | None, end_b: float | None) -> float:
    """Temporal intersection-over-union; 0 when unusable."""
    if start_a is None or end_a is None:
        return 0.0
    if start_b is None or end_b is None:
        return 0.0
    if end_a <= start_a or end_b <= start_b:
        return 0.0
    inter_start, inter_end = max(start_a, start_b), min(end_a, end_b)
    inter = max(0.0, inter_end - inter_start)
    union = max(end_a, end_b) - min(start_a, start_b)
    if union <= 0:
        return 0.0
    return inter / union


def actions_compatible(action_a: str | None,
                       action_b: str | None) -> bool:
    """Same canonical action, or either side unknown/absent."""
    if not action_a or not action_b:
        return True
    return action_a == action_b


def containment(start_a: float | None, end_a: float | None,
                start_b: float | None, end_b: float | None) -> float:
    """Fraction of the smaller interval covered by the overlap."""
    if start_a is None or end_a is None:
        return 0.0
    if start_b is None or end_b is None:
        return 0.0
    if end_a <= start_a or end_b <= start_b:
        return 0.0
    inter = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    smaller = min(end_a - start_a, end_b - start_b)
    if smaller <= 0:
        return 0.0
    return inter / smaller


def _interval_len(start: float | None, end: float | None) -> float | None:
    if start is None or end is None or end <= start:
        return None
    return end - start


def group_events(
        events: list[dict[str, Any]],
        clip_order: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Group cross-source overlapping events with compatible actions.

    Preliminary (approximate) records never join a group; they link
    as context afterwards. Single-pass greedy grouping over
    start-sorted events keeps the result deterministic.
    """
    concrete = sorted(
        (e for e in events if not e.get("approximate")),
        key=lambda e: (e.get("start_seconds") is None,
                       e.get("start_seconds") or 0.0,
                       e.get("source") or "", e.get("event_id") or ""))
    groups: list[dict[str, Any]] = []
    for event in concrete:
        placed = False
        for group in groups:
            if _group_accepts(group, event):
                group["members"].append(event)
                placed = True
                break
        if not placed:
            groups.append({"members": [event]})
    result: list[dict[str, Any]] = []
    for index, group in enumerate(grouped_sort(
            groups, clip_order)):
        members = group["members"]
        starts = [m["start_seconds"] for m in members
                  if m.get("start_seconds") is not None]
        ends = [m["end_seconds"] for m in members
                if m.get("end_seconds") is not None]
        sources = sorted({m["source"] for m in members})
        clips = sorted({m.get("clip_id") for m in members
                        if m.get("clip_id")})
        result.append({
            "group_id": f"eg_{index:03d}",
            "clip_id": clips[0] if len(clips) == 1 else None,
            "start_seconds": min(starts) if starts else None,
            "end_seconds": max(ends) if ends else None,
            "sources": sources,
            "source_count": len(sources),
            "member_ids": [m["event_id"] for m in members],
            "members": members,
            "preliminary_links": [],
            "speech_links": [],
            "status": "ai_suggested",
        })
    return result


def grouped_sort(
        groups: list[dict[str, Any]],
        clip_order: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Order groups by clip order, then earliest member start.

    Single-video runs (no clips) keep the original ordering.
    """
    order = clip_order or {}

    def key(group: dict[str, Any]) -> tuple:
        members = group["members"]
        clips = sorted({m.get("clip_id") for m in members
                        if m.get("clip_id")})
        rank = min([order.get(clip, 1_000_000) for clip in clips],
                   default=0)
        first_clip = min(clips) if clips else ""
        starts = [m.get("start_seconds") for m in members
                  if m.get("start_seconds") is not None]
        return (rank, first_clip,
                min(starts) if starts else float("inf"),
                members[0].get("event_id") or "")
    return sorted(groups, key=key)


def same_clip(first: dict[str, Any],
              second: dict[str, Any]) -> bool:
    """True unless both carry different non-null clip ids.

    Single-video runs (clip None everywhere) always match.
    """
    left, right = first.get("clip_id"), second.get("clip_id")
    if left is None or right is None:
        return True
    return left == right


def _group_accepts(group: dict[str, Any],
                   event: dict[str, Any]) -> bool:
    for member in group["members"]:
        if not same_clip(member, event):
            continue
        mstart, mend = (member.get("start_seconds"),
                        member.get("end_seconds"))
        estart, eend = (event.get("start_seconds"),
                        event.get("end_seconds"))
        iou = interval_iou(mstart, mend, estart, eend)
        if iou >= OVERLAP_IOU_THRESHOLD and actions_compatible(
                member.get("action_canonical"),
                event.get("action_canonical")):
            return True
        # Nested detail inside a global operation joins the group;
        # both verbatim actions are preserved on the step.
        # Scene-establishing surveys are context, not anchors: a long
        # survey spanning many distinct actions must not absorb them
        # into one pseudo-step.
        nested = containment(mstart, mend, estart, eend)
        if nested >= NESTING_CONTAINMENT_THRESHOLD:
            mlen = _interval_len(mstart, mend)
            elen = _interval_len(estart, eend)
            longer = member if (mlen or 0) >= (elen or 0) else event
            if longer.get("action_canonical") != "survey":
                return True
    return False


def link_preliminary(groups: list[dict[str, Any]],
                     preliminary: list[dict[str, Any]]) -> None:
    """Attach approximate human notes as context (never evidence)."""
    for note in preliminary:
        start = note.get("start_seconds")
        end = note.get("end_seconds")
        if start is None or end is None:
            continue
        for group in groups:
            if not same_clip(note, {"clip_id": group.get("clip_id")}):
                continue
            gstart = group.get("start_seconds")
            gend = group.get("end_seconds")
            if gstart is None or gend is None:
                continue
            if (gstart <= end + PRELIMINARY_LINK_TOLERANCE
                    and gend >= start - PRELIMINARY_LINK_TOLERANCE):
                group["preliminary_links"].append(note["event_id"])


def link_speech(groups: list[dict[str, Any]],
                speech: list[dict[str, Any]]) -> None:
    """Attach transcript turns overlapping a group (same clip only).

    Speech links are navigation aids: what was said near an
    event, never proof of the event itself.
    """
    for turn in speech:
        start = turn.get("start_seconds")
        end = turn.get("end_seconds")
        if start is None:
            continue
        stop = end if end is not None else start
        for group in groups:
            if not same_clip(turn, group):
                continue
            gstart = group.get("start_seconds")
            gend = group.get("end_seconds")
            if gstart is None or gend is None:
                continue
            if gstart <= stop and gend >= start:
                group["speech_links"].append(turn["speech_id"])


def ingredient_key(name: str | None) -> str | None:
    """Match key for normalized names; None stays unmatched."""
    return normalize_label(name)


def group_ingredients(
        records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group by normalized name, then by identical canonical label.

    Named records merge on identity. Unnamed records merge only on
    exact canonical-label match (repeated descriptions of one
    observation); records with neither stay separate. No identity
    is ever inferred from a description.
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    singles: list[dict[str, Any]] = []
    for record in sorted(records,
                         key=lambda r: (r.get("source") or "",
                                        r.get("record_id") or "")):
        key = ingredient_key(record.get("normalized_name"))
        if key is None:
            label = ingredient_key(record.get("visible_label"))
            if label is not None:
                label = f"label:{label}"
            key = label
        if key is None:
            singles.append(record)
        else:
            clip = record.get("clip_id")
            scoped = f"{clip}::{key}" if clip else key
            buckets.setdefault(scoped, []).append(record)
    groups: list[dict[str, Any]] = []
    for key in sorted(buckets):
        members = buckets[key]
        stamps: list[float] = []
        for member in members:
            stamps.extend(member.get("supporting_timestamps") or [])
        sources = sorted({m["source"] for m in members})
        clips = sorted({m.get("clip_id") for m in members
                        if m.get("clip_id")})
        groups.append({
            "group_id": f"ig_{stable_id(key)}",
            "match_key": key,
            "clip_id": clips[0] if len(clips) == 1 else None,
            "sources": sources,
            "source_count": len(sources),
            "member_ids": [m["record_id"] for m in members],
            "members": members,
            "supporting_timestamps": sorted(set(stamps)),
            "status": "ai_suggested",
        })
    for record in singles:
        groups.append({
            "group_id": f"ig_{record['record_id']}",
            "match_key": None,
            "clip_id": record.get("clip_id"),
            "sources": [record["source"]],
            "source_count": 1,
            "member_ids": [record["record_id"]],
            "members": [record],
            "supporting_timestamps": sorted(
                set(record.get("supporting_timestamps") or [])),
            "status": "ai_suggested",
        })
    _link_possible_relatives(groups)
    return sorted(groups, key=lambda g: (g["match_key"] is None,
                                         g["match_key"] or "",
                                         g["group_id"]))


_STOPWORDS = frozenset({
    "the", "and", "with", "from", "into", "onto", "near", "clear",
    "small", "large", "white", "brown", "beige", "granular",
})


def _label_tokens(label: str | None) -> set[str]:
    if not label:
        return set()
    cleaned = "".join(c if c.isalnum() else " " for c in label.lower())
    return {t for t in cleaned.split()
            if len(t) > 2 and t not in _STOPWORDS}


def _link_possible_relatives(groups: list[dict[str, Any]]) -> None:
    """Link unmerged groups sharing distinctive label tokens.

    Links are review hints, never merges: identity stays separate.
    """
    token_sets: list[set[str]] = []
    for group in groups:
        tokens: set[str] = set()
        for member in group.get("members") or []:
            tokens |= _label_tokens(member.get("visible_label"))
            tokens |= _label_tokens(member.get("normalized_name"))
        token_sets.append(tokens)
    for index, group in enumerate(groups):
        related: list[str] = []
        for other_index, other in enumerate(groups):
            if other_index == index:
                continue
            if len(token_sets[index] & token_sets[other_index]) >= 2:
                related.append(other["group_id"])
        group["possibly_related"] = sorted(related)


def corroborated_readings(
        readings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair same-value readings within tolerance (counts, not proof)."""
    pairs: list[dict[str, Any]] = []
    ordered = sorted(readings,
                     key=lambda r: (r.get("timestamp_seconds") is None,
                                    r.get("timestamp_seconds") or 0.0,
                                    r.get("reading_id") or ""))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1:]:
            if left["source"] == right["source"]:
                continue
            if not same_clip(left, right):
                continue
            lstamp, rstamp = (left.get("timestamp_seconds"),
                              right.get("timestamp_seconds"))
            if lstamp is None or rstamp is None:
                continue
            if abs(lstamp - rstamp) > READING_MATCH_TOLERANCE:
                continue
            lval, rval = left.get("displayed_value"), right.get(
                "displayed_value")
            if lval is None or rval is None:
                continue
            if str(lval).strip() != str(rval).strip():
                continue
            pairs.append({
                "pair_id": f"rp_{stable_id(left['reading_id'], right['reading_id'])}",  # noqa: E501
                "reading_ids": [left["reading_id"], right["reading_id"]],
                "sources": sorted([left["source"], right["source"]]),
                "displayed_value": str(lval).strip(),
                "status": "ai_suggested",
            })
    return pairs


def detect_conflicts(events: list[dict[str, Any]],
                     readings: list[dict[str, Any]],
                     ingredients: list[dict[str, Any]]
                     ) -> list[dict[str, Any]]:
    """Flag action, reading, and identity conflicts explicitly."""
    conflicts: list[dict[str, Any]] = []
    concrete = [e for e in events if not e.get("approximate")]
    for index, left in enumerate(concrete):
        for right in concrete[index + 1:]:
            if left["source"] == right["source"]:
                continue
            if not same_clip(left, right):
                continue
            iou = interval_iou(left.get("start_seconds"),
                               left.get("end_seconds"),
                               right.get("start_seconds"),
                               right.get("end_seconds"))
            if iou < CONFLICT_IOU_THRESHOLD:
                continue
            action_a, action_b = (left.get("action_canonical"),
                                  right.get("action_canonical"))
            if not (action_a and action_b and action_a != action_b):
                continue
            # Contradictions need comparable scope: nested detail at
            # a different granularity is not a disagreement.
            len_a = _interval_len(left.get("start_seconds"),
                                  left.get("end_seconds"))
            len_b = _interval_len(right.get("start_seconds"),
                                  right.get("end_seconds"))
            if len_a is None or len_b is None:
                continue
            if min(len_a, len_b) / max(len_a, len_b) < 0.5:
                continue
            conflicts.append({
                "conflict_id": f"cf_{stable_id(left['event_id'], right['event_id'])}",  # noqa: E501
                "kind": "action_conflict",
                "event_ids": [left["event_id"], right["event_id"]],
                "sources": sorted(
                    [left["source"], right["source"]]),
                "detail": f"{action_a} vs {action_b} "
                          f"(iou={iou:.2f})",
            })
    ordered = sorted(
        (r for r in readings if r.get("displayed_value") is not None
         and r.get("timestamp_seconds") is not None),
        key=lambda r: (r.get("timestamp_seconds") or 0.0,
                       r.get("reading_id") or ""))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1:]:
            if left["source"] == right["source"]:
                continue
            if not same_clip(left, right):
                continue
            if abs((left.get("timestamp_seconds") or 0.0)
                   - (right.get("timestamp_seconds") or 0.0)) > \
                    READING_MATCH_TOLERANCE:
                continue
            lval = str(left.get("displayed_value")).strip()
            rval = str(right.get("displayed_value")).strip()
            if lval != rval:
                conflicts.append({
                    "conflict_id": f"cf_{stable_id(left['reading_id'], right['reading_id'])}",  # noqa: E501
                    "kind": "reading_conflict",
                    "reading_ids": [left["reading_id"],
                                    right["reading_id"]],
                    "sources": sorted(
                        [left["source"], right["source"]]),
                    "detail": f"{lval} vs {rval}",
                })
    named = [i for i in ingredients if ingredient_key(
        i.get("normalized_name")) is not None]
    for index, left in enumerate(named):
        for right in named[index + 1:]:
            if left["source"] == right["source"]:
                continue
            if not same_clip(left, right):
                continue
            if ingredient_key(left.get("normalized_name")) == \
                    ingredient_key(right.get("normalized_name")):
                continue
            lstamps = left.get("supporting_timestamps") or []
            rstamps = right.get("supporting_timestamps") or []
            if not lstamps or not rstamps:
                continue
            closest = min(abs(a - b) for a in lstamps for b in rstamps)
            if closest > INGREDIENT_MATCH_TOLERANCE:
                continue
            conflicts.append({
                "conflict_id": f"cf_{stable_id(left['record_id'], right['record_id'])}",  # noqa: E501
                "kind": "ingredient_conflict",
                "ingredient_ids": [left["record_id"],
                                   right["record_id"]],
                "sources": sorted([left["source"], right["source"]]),
                "detail": f"{left.get('normalized_name')} vs "
                          f"{right.get('normalized_name')} "
                          f"(~{closest:.1f}s apart)",
            })
    return sorted(conflicts, key=lambda c: (c["kind"], c["conflict_id"]))
