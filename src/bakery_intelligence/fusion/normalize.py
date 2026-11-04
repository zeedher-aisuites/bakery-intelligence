"""Adapters: source schemas -> normalized fusion records.

Every adapter is a pure function over already-parsed dicts, so
tests use small synthetic fixtures and never touch evidence files.
"""

from __future__ import annotations

import re
from typing import Any

from bakery_intelligence.fusion.models import (
    ACTION_ALIASES,
    SOURCE_PRELIMINARY,
    SOURCE_SPARK,
    SOURCE_SPARK_OPEN,
    STATUS_AI_SUGGESTED,
    NormalizedEvent,
    NormalizedIngredient,
    NormalizedMeasurement,
    stable_id,
)


def normalize_confidence(raw: Any) -> str:
    """Map free-text confidence to a level; unknown on doubt."""
    if not isinstance(raw, str):
        return "unknown"
    lowered = raw.strip().lower()
    if "high" in lowered:
        return "high"
    if "medium" in lowered or "moderate" in lowered:
        return "medium"
    if "low" in lowered:
        return "low"
    return "unknown"


def canonical_action(raw: Any) -> str | None:
    """Map an action to its canonical alias, or keep the raw token."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    token = "_".join(raw.strip().lower().split())
    return ACTION_ALIASES.get(token, token)


def normalize_label(raw: Any) -> str | None:
    """Canonicalize a visible label for observation consolidation.

    Merges repeated descriptions of one observation across
    segments (exact canonical match only). Never infers identity:
    'white powder' stays a description, never becomes 'flour'.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    cleaned = "".join(c if c.isalnum() else " " for c in raw.lower())
    return " ".join(cleaned.split()) or None


def clip_parts(clip_id: str | None) -> list[Any]:
    """Id parts for clip scoping; empty for single-video runs.

    P04 calls pass clip_id=None, so their stable ids stay
    byte-identical to Phase 2A.
    """
    return [clip_id] if clip_id else []


def as_seconds(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def parse_clock_interval(raw: Any) -> tuple[float | None, float | None]:
    """Parse 'mm:ss<sep>mm:ss' with any separator; nulls on failure."""
    if not isinstance(raw, str):
        return None, None
    parts = [p for p in re.split(r"[^\d:]+", raw) if p]
    if len(parts) != 2:
        return None, None
    out: list[float | None] = []
    for part in parts:
        try:
            minutes, _, seconds = part.partition(":")
            out.append(float(minutes) * 60.0 + float(seconds or 0))
        except ValueError:
            out.append(None)
    start, end = out
    if start is None or end is None or end < start:
        return None, None
    return start, end


def spark_events(parsed_output: dict[str, Any],
                 response_id: str | None,
                 clip_id: str | None = None,
                 window: list[Any] | tuple[Any, ...] | None = None,
                 ) -> list[NormalizedEvent]:
    """Normalize Spark 1.3 operations (full video and segments).

    Full-video keys win; local segments use short aliases
    ('action'/'type', 'description'/'detail', 'ingredient',
    'tool'/'equipment'/'device'). Segment ops carry no event id
    or absolute timestamps: each keeps a distinct record id via
    its action and position, and the analyzed window is recorded
    as an uncertainty note instead of invented timestamps.
    """
    events: list[NormalizedEvent] = []
    operations = parsed_output.get("operations")
    if not isinstance(operations, list):
        return events
    win_start, win_end = _window_bounds(window)
    for index, item in enumerate(operations):
        if not isinstance(item, dict):
            continue
        raw_action = _first_str(
            item, ("action_type", "action", "type"))
        start = as_seconds(item.get("start_seconds"))
        end = as_seconds(item.get("end_seconds"))
        if (win_start is not None and win_end is not None
                and start is not None and end is not None
                and (start < win_start or end > win_end)):
            # Segment ops sometimes reuse absolute-looking keys
            # for window-relative stamps; values outside the
            # analyzed window cannot be clip-local. No offset
            # arithmetic: the window note below carries the span.
            start = end = None
        source_ref = str(item.get("event_id") or "unknown")
        ingredients = _ingredient_list(item)
        utensils = _utensil_list(item)
        parts: list[Any] = [SOURCE_SPARK, source_ref, start,
                            *clip_parts(clip_id)]
        if source_ref == "unknown" and start is None:
            # Degenerate segment ops would otherwise share one
            # stable id per response; action + position keep them
            # distinct without touching identified records.
            parts.extend([raw_action or "", index])
        uncertainty = item.get("uncertainty") \
            if isinstance(item.get("uncertainty"), str) else None
        if (start is None and end is None
                and win_start is not None and win_end is not None):
            note = ("Exact seconds not reported; analyzed segment "
                    f"window {win_start:.1f}-{win_end:.1f}s.")
            uncertainty = f"{uncertainty} {note}" if uncertainty \
                else note
        events.append({
            "event_id": f"fe_{stable_id(*parts)}",
            "source": SOURCE_SPARK,
            "clip_id": clip_id,
            "source_ref": f"{response_id}:{source_ref}" if response_id
            else source_ref,
            "action_type": raw_action if isinstance(raw_action, str)
            else None,
            "action_canonical": canonical_action(raw_action),
            "action_description": _first_str(
                item, ("action_description", "description",
                       "detail", "Detail")),
            "start_seconds": start,
            "end_seconds": end,
            "approximate": False,
            "ingredients": ingredients,
            "utensils": utensils,
            "containers": [],
            "evidence": item.get("evidence")
            if isinstance(item.get("evidence"), str) else None,
            "confidence": normalize_confidence(item.get("confidence")),
            "uncertainty": uncertainty,
            "status": STATUS_AI_SUGGESTED,
        })
    return events


def _first_str(item: dict[str, Any],
               keys: tuple[str, ...]) -> str | None:
    """First non-blank string under alias keys (schema fallback)."""
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _window_bounds(
        window: list[Any] | tuple[Any, ...] | None,
        ) -> tuple[float | None, float | None]:
    """Validate an analyzed segment window [start, end]."""
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        return None, None
    start, end = as_seconds(window[0]), as_seconds(window[1])
    if start is None or end is None or end <= start:
        return None, None
    return start, end


def _ingredient_list(item: dict[str, Any]) -> list[str]:
    """Full-video list, or one segment 'ingredient' name."""
    raw = item.get("ingredients")
    if isinstance(raw, list):
        return [str(v) for v in raw]
    single = _first_str(item, ("ingredient", "ingredient_name"))
    return [single] if single else []


def _utensil_list(item: dict[str, Any]) -> list[str]:
    """Full-video list, or segment tool/equipment/device names."""
    raw = item.get("containers_utensils")
    if isinstance(raw, list):
        return [str(v) for v in raw]
    found = [_first_str(item, ("tool", "equipment", "device"))]
    return [v for v in found if v]


def attach_claims(turns: list[dict[str, Any]],
                  claims: list[dict[str, Any]],
                  ) -> tuple[int, int]:
    """Join culinary claims to transcript turns by turn index.

    Returns (attached, orphaned) claim counts. Claims whose
    turn_ids fall outside the turn list attach nowhere; callers
    report the orphan count instead of dropping them silently.
    """
    attached = 0
    orphaned = 0
    for claim in claims:
        if not isinstance(claim, dict):
            orphaned += 1
            continue
        ids = claim.get("turn_ids")
        if not isinstance(ids, list):
            orphaned += 1
            continue
        linked = False
        for index in ids:
            if (isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < len(turns)):
                continue
            if not isinstance(turns[index], dict):
                continue
            bucket = turns[index].get("claims")
            if not isinstance(bucket, list):
                bucket = []
                turns[index]["claims"] = bucket
            seen = {c.get("claim_id") for c in bucket
                    if isinstance(c, dict)}
            if claim.get("claim_id") not in seen:
                bucket.append(claim)
            linked = True
        if linked:
            attached += 1
        else:
            orphaned += 1
    return attached, orphaned


def spark_ingredients(
        parsed_output: dict[str, Any],
        response_id: str | None,
        clip_id: str | None = None,
        ) -> list[NormalizedIngredient]:
    """Normalize Spark 1.3 ingredient candidates."""
    records: list[NormalizedIngredient] = []
    ingredients = parsed_output.get("ingredients")
    if not isinstance(ingredients, list):
        return records
    for index, item in enumerate(ingredients):
        if not isinstance(item, dict):
            continue
        # Global analyst keys win; local segments use short
        # aliases ('name', 'appearance'/'appearance_visible'/
        # 'state'). Without this fallback every local item loses
        # its identity and all records in one clip collapse to a
        # single stable id (duplicate review targets).
        name = item.get("normalized_name")
        if not (isinstance(name, str) and name.strip()):
            name = item.get("name")
        label = item.get("visible_label")
        if not (isinstance(label, str) and label.strip()):
            for alias in ("appearance", "appearance_visible",
                          "state"):
                candidate = item.get(alias)
                if isinstance(candidate, str) and candidate.strip():
                    label = candidate
                    break
        stamps = item.get("supporting_timestamps")
        ambiguities = item.get("ambiguities")
        source_ref = f"ingredient_{index}"
        parts = [SOURCE_SPARK, name, label,
                 *clip_parts(clip_id)]
        records.append({
            "record_id": f"fi_{stable_id(*parts)}",
            "source": SOURCE_SPARK,
            "clip_id": clip_id,
            "source_ref": f"{response_id}:{source_ref}" if response_id
            else source_ref,
            "normalized_name": name.strip().lower()
            if isinstance(name, str) and name.strip() else None,
            "visible_label": label if isinstance(label, str)
            else None,
            "supporting_timestamps": [
                s for s in ((as_seconds(v) for v in stamps)
                            if isinstance(stamps, list) else [])
                if s is not None],
            "used_or_visible": item.get("used_or_visible")
            if isinstance(item.get("used_or_visible"), str) else None,
            "evidence": item.get("evidence")
            if isinstance(item.get("evidence"), str) else None,
            "ambiguities": [str(v) for v in ambiguities]
            if isinstance(ambiguities, list) else [],
            "status": STATUS_AI_SUGGESTED,
        })
    return records


def spark_measurements(
        parsed_output: dict[str, Any],
        response_id: str | None,
        clip_id: str | None = None,
        ) -> list[NormalizedMeasurement]:
    """Normalize Spark 1.3 visible scale readings (displays, not nets).

    Accepts the P04 'scale_readings' key and the Phase 2C
    'measurements' key; P04 payloads take the original path.
    """
    records: list[NormalizedMeasurement] = []
    readings = parsed_output.get("scale_readings")
    if not isinstance(readings, list):
        readings = parsed_output.get("measurements")
    if not isinstance(readings, list):
        return records
    for index, item in enumerate(readings):
        if not isinstance(item, dict):
            continue
        value = item.get("displayed_value")
        unit = item.get("visible_unit")
        source_ref = f"scale_{index}"
        parts = [SOURCE_SPARK, value, index,
                 *clip_parts(clip_id)]
        records.append({
            "reading_id": f"fm_{stable_id(*parts)}",
            "source": SOURCE_SPARK,
            "clip_id": clip_id,
            "source_ref": f"{response_id}:{source_ref}" if response_id
            else source_ref,
            "timestamp_seconds": as_seconds(
                item.get("timestamp_seconds")),
            "displayed_value": str(value).strip()
            if value is not None else None,
            "visible_unit": str(unit).strip()
            if isinstance(unit, str) and unit.strip() else None,
            "tare_observable": item.get("tare_observable")
            if isinstance(item.get("tare_observable"), bool) else None,
            "uncertain": item.get("uncertain") is not False,
            "evidence": item.get("evidence")
            if isinstance(item.get("evidence"), str) else None,
            "status": STATUS_AI_SUGGESTED,
        })
    return records


def baseten_events(
        aggregated: dict[str, Any],
        clip_id: str | None = None) -> list[NormalizedEvent]:
    """Normalize Baseten segmented culinary events (absolute time)."""
    events: list[NormalizedEvent] = []
    items = aggregated.get("culinary_events")
    if not isinstance(items, list):
        return events
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_action = item.get("action_type")
        start = as_seconds(item.get("absolute_start_seconds"))
        if start is None:
            start = as_seconds(item.get("start_seconds"))
        end = as_seconds(item.get("absolute_end_seconds"))
        if end is None:
            end = as_seconds(item.get("end_seconds"))
        source_ref = str(item.get("event_id") or "unknown")
        ingredients = item.get("ingredient_candidates")
        utensils = item.get("utensils")
        containers = item.get("containers")
        parts = [SOURCE_SPARK_OPEN, source_ref,
                 *clip_parts(clip_id)]
        events.append({
            "event_id": f"fe_{stable_id(*parts)}",
            "source": SOURCE_SPARK_OPEN,
            "clip_id": clip_id,
            "source_ref": source_ref,
            "action_type": raw_action if isinstance(raw_action, str)
            else None,
            "action_canonical": canonical_action(raw_action),
            "action_description": item.get("action_description")
            if isinstance(item.get("action_description"), str) else None,
            "start_seconds": start,
            "end_seconds": end,
            "approximate": False,
            "ingredients": [str(v) for v in ingredients]
            if isinstance(ingredients, list) else [],
            "utensils": [str(v) for v in utensils]
            if isinstance(utensils, list) else [],
            "containers": [str(v) for v in containers]
            if isinstance(containers, list) else [],
            "evidence": item.get("evidence")
            if isinstance(item.get("evidence"), str) else None,
            "confidence": normalize_confidence(item.get("confidence")),
            "uncertainty": item.get("uncertainty")
            if isinstance(item.get("uncertainty"), str) else None,
            "status": STATUS_AI_SUGGESTED,
        })
    return events


def baseten_ingredients(
        aggregated: dict[str, Any],
        clip_id: str | None = None) -> list[NormalizedIngredient]:
    """Normalize Baseten ingredient records (absolute timestamps)."""
    records: list[NormalizedIngredient] = []
    items = aggregated.get("ingredients")
    if not isinstance(items, list):
        return records
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("normalized_name")
        label = item.get("visible_name_or_label")
        stamps = item.get("supporting_timestamps")
        offset = as_seconds(item.get("segment_source_start_seconds"))
        absolute: list[float] = []
        if isinstance(stamps, list):
            for value in stamps:
                seconds = as_seconds(value)
                if seconds is None:
                    continue
                absolute.append(seconds + offset if offset else seconds)
        ambiguities = item.get("ambiguities")
        parts = [SOURCE_SPARK_OPEN, label,
                 *clip_parts(clip_id)]
        records.append({
            "record_id": f"fi_{stable_id(*parts)}",
            "source": SOURCE_SPARK_OPEN,
            "clip_id": clip_id,
            "source_ref": f"seg{item.get('segment_index')}:{label}",
            "normalized_name": name.strip().lower()
            if isinstance(name, str) and name.strip() else None,
            "visible_label": label if isinstance(label, str)
            else None,
            "supporting_timestamps": absolute,
            "used_or_visible": item.get("actually_used_or_merely_visible")
            if isinstance(item.get("actually_used_or_merely_visible"), str)
            else None,
            "evidence": item.get("visual_evidence")
            if isinstance(item.get("visual_evidence"), str) else None,
            "ambiguities": [str(ambiguities)]
            if isinstance(ambiguities, str) and ambiguities.strip()
            else ([str(v) for v in ambiguities]
                  if isinstance(ambiguities, list) else []),
            "status": STATUS_AI_SUGGESTED,
        })
    return records


def baseten_measurements(
        aggregated: dict[str, Any],
        clip_id: str | None = None) -> list[NormalizedMeasurement]:
    """Normalize Baseten measurement records (absolute time)."""
    records: list[NormalizedMeasurement] = []
    items = aggregated.get("measurements")
    if not isinstance(items, list):
        return records
    for item in items:
        if not isinstance(item, dict):
            continue
        stamp = as_seconds(item.get("absolute_timestamp_seconds"))
        if stamp is None:
            stamp = as_seconds(item.get("timestamp_seconds"))
        value = item.get("displayed_value")
        unit = item.get("visible_unit")
        source_ref = str(item.get("reading_id") or "unknown")
        parts = [SOURCE_SPARK_OPEN, source_ref,
                 *clip_parts(clip_id)]
        records.append({
            "reading_id": f"fm_{stable_id(*parts)}",
            "source": SOURCE_SPARK_OPEN,
            "clip_id": clip_id,
            "source_ref": source_ref,
            "timestamp_seconds": stamp,
            "displayed_value": str(value).strip()
            if value is not None else None,
            "visible_unit": str(unit).strip()
            if isinstance(unit, str) and unit.strip() else None,
            "tare_observable": item.get("tare_operation_observable")
            if isinstance(item.get("tare_operation_observable"), bool)
            else None,
            "uncertain": item.get("is_uncertain") is not False,
            "evidence": item.get("evidence")
            if isinstance(item.get("evidence"), str) else None,
            "status": STATUS_AI_SUGGESTED,
        })
    return records


def preliminary_events(
        annotation: dict[str, Any]) -> list[NormalizedEvent]:
    """Normalize human preliminary notes (approximate, unverified).

    Ingests both the derived candidate_events summary and the full
    segments[].observations[] record (display reports, tare notes,
    artifact analysis). These are review notes, not observations:
    every record carries approximate=True and keeps its
    verification question as the uncertainty text. Candidates are
    global review questions; segment observations are step-linked
    context only.
    """
    events: list[NormalizedEvent] = []
    items = annotation.get("candidate_events")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                events.append(_preliminary_candidate(item))
    segments = annotation.get("segments")
    if isinstance(segments, list):
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            observations = segment.get("observations")
            if not isinstance(observations, list):
                continue
            for observation in observations:
                if isinstance(observation, dict):
                    events.append(_preliminary_observation(
                        observation, segment.get("segment_id")))
    return events


def _preliminary_candidate(item: dict[str, Any]) -> NormalizedEvent:
    source_ref = str(item.get("candidate_id") or "unknown")
    start, end = parse_clock_interval(item.get("reported_interval"))
    question = item.get("verification_question")
    note = _preliminary_note(
        source_ref=source_ref,
        action=item.get("candidate_type")
        if isinstance(item.get("candidate_type"), str) else None,
        start=start,
        end=end,
        context=None,
        question=str(question) if question is not None else None)
    note["note_kind"] = "candidate"
    return note


def _preliminary_observation(observation: dict[str, Any],
                             segment_id: Any) -> NormalizedEvent:
    source_ref = str(observation.get("observation_id") or "unknown")
    stamp = as_seconds(observation.get("approximate_timestamp_seconds"))
    action = observation.get("reported_action")
    display = observation.get("reported_display_or_state")
    need = observation.get("interpretation_and_verification_need")
    context_parts: list[str] = []
    if isinstance(display, str) and display.strip():
        context_parts.append(f"Reported display/state: {display.strip()}")
    segment = str(segment_id).strip() if isinstance(
        segment_id, str) and segment_id.strip() else None
    note = _preliminary_note(
        source_ref=f"{segment}:{source_ref}" if segment
        else source_ref,
        action=action if isinstance(action, str) else None,
        start=stamp,
        end=stamp,
        context=" ".join(context_parts) or None,
        question=str(need) if need is not None else None)
    note["note_kind"] = "segment_observation"
    return note


def _preliminary_note(source_ref: str,
                      action: str | None,
                      start: float | None,
                      end: float | None,
                      context: str | None,
                      question: str | None) -> NormalizedEvent:
    uncertainty = ". ".join(part for part in (context, question)
                            if part)
    return {
        "event_id": f"fe_{stable_id(SOURCE_PRELIMINARY, source_ref)}",
        "source": SOURCE_PRELIMINARY,
        "clip_id": None,
        "source_ref": source_ref,
        "action_type": action,
        "action_canonical": None,
        "action_description": None,
        "start_seconds": start,
        "end_seconds": end,
        "approximate": True,
        "ingredients": [],
        "utensils": [],
        "containers": [],
        "evidence": None,
        "confidence": "unknown",
        "uncertainty": uncertainty or None,
        "status": STATUS_AI_SUGGESTED,
    }


def speech_turns(turns: list[dict[str, Any]],
                 clip_id: str | None,
                 source: str = "voice-transcribe") -> list[dict[str, Any]]:
    """Normalize ASR transcript turns to linkable speech records.

    Speech is evidence of utterance, never of physical fact:
    records link to steps by clip-local time but never join
    visual groups. Timestamps keep provider granularity; missing
    ends stay null instead of invented.
    """
    records: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        text = turn.get("transcript")
        if not isinstance(text, str) or not text.strip():
            continue
        start = as_seconds(turn.get("start_seconds"))
        end = as_seconds(turn.get("end_seconds"))
        records.append({
            "speech_id": (
                f"fs_{stable_id(source, clip_id, index, text)}"),
            "source": source,
            "clip_id": clip_id,
            "turn_index": index,
            "speaker": turn.get("speaker")
            if isinstance(turn.get("speaker"), str) else None,
            "start_seconds": start,
            "end_seconds": end,
            "transcript": text.strip(),
            "claims": turn.get("claims")
            if isinstance(turn.get("claims"), list) else [],
            "status": STATUS_AI_SUGGESTED,
        })
    return records
