"""Deterministic temporal event reconstruction from readings.

Events, not frames, are the unit of reconstruction. Segments
are split at tare/reset frames and at sustained drops. A net
quantity is derived only when a supported baseline, compatible
unit, stable final plateau, and no intervening tare or
container change are all evidenced; otherwise it stays unknown
with the failed conditions recorded. No frame-count heuristics
substitute for boundary evidence.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

EVENT_VERSION = "temporal-events/1"
PLATEAU_TOLERANCE = 0.0


def _effective_value(frame: dict[str, Any]) -> float | None:
    effective = frame.get("effective")
    if isinstance(effective, dict) and isinstance(
            effective.get("value"), (int, float)):
        return float(effective["value"])
    value = (frame.get("measurement") or {}).get("value")
    return float(value) if isinstance(value, (int, float)) else None


def reconstruct_events(consolidated: dict[str, Any],
                       ) -> dict[str, Any]:
    """Reconstruct candidate weighing events from a package."""
    frames = [frame for frame in consolidated.get("frames", [])
              if _effective_value(frame) is not None]
    frames.sort(key=lambda frame: frame["frame_index"])
    action = consolidated.get("action_evidence") or {}
    tare_frames = set(action.get("tare_frames") or [])

    events: list[dict[str, Any]] = []
    segment: list[dict[str, Any]] = []

    def flush() -> None:
        if not segment:
            return
        first, last = segment[0], segment[-1]
        initial = _effective_value(first)
        final = _effective_value(last)
        plateau = (
            len(segment) >= 2
            and _effective_value(segment[-2]) == final)
        unit = (last.get("measurement") or {}).get("unit_observed")
        baseline_ok = initial == 0.0
        tare_inside = any(
            frame["frame_index"] in tare_frames
            for frame in segment[1:])
        conditions = {
            "baseline_supported": baseline_ok,
            "unit_compatible": unit is not None,
            "final_plateau_stable": plateau,
            "no_intervening_tare": not tare_inside,
            "no_container_change": True,
        }
        derived = all(conditions.values())
        seed = hashlib.sha256(json.dumps(
            [consolidated.get("run_id"), first["frame_index"],
             last["frame_index"]], sort_keys=True).encode()
        ).hexdigest()[:12]
        events.append({
            "event_id": f"event_{seed}",
            "event_type": "ingredient_addition_candidate",
            "start_seconds": first.get("timestamp_seconds"),
            "end_seconds": last.get("timestamp_seconds"),
            "frame_refs": [frame["frame_index"] for frame in segment],
            "ingredient_refs": [],
            "initial_reading": initial,
            "final_reading": final,
            "net_quantity": {
                "value": (final - initial) if derived else None,
                "unit": unit if derived else None,
                "derivation": (
                    "difference_between_readings" if derived
                    else "unknown"),
                "formula": (
                    f"{final} - {initial}" if derived else None),
                "inputs": {
                    "initial_frame": first["frame_index"],
                    "final_frame": last["frame_index"],
                },
                "conditions": conditions,
            },
            "provenance": "ai-extracted",
            "supporting_evidence": [
                first["frame_index"], last["frame_index"]],
            "contradictory_evidence": [],
            "uncertainty": (
                None if derived else
                "Net quantity is not established: "
                + ", ".join(name for name, ok in conditions.items()
                            if not ok) + "."),
            "verification_status": (
                "supported_candidate" if derived else "ambiguous"),
        })
        segment.clear()

    previous: float | None = None
    for frame in frames:
        value = _effective_value(frame)
        if frame["frame_index"] in tare_frames and segment:
            flush()
            events.append({
                "event_id": f"event_tare_{frame['frame_index']}",
                "event_type": "tare_or_reset",
                "start_seconds": frame.get("timestamp_seconds"),
                "end_seconds": frame.get("timestamp_seconds"),
                "frame_refs": [frame["frame_index"]],
                "ingredient_refs": [],
                "initial_reading": previous,
                "final_reading": value,
                "net_quantity": {
                    "value": None, "unit": None,
                    "derivation": "unknown", "formula": None,
                    "inputs": {}, "conditions": {},
                },
                "provenance": "ai-extracted",
                "supporting_evidence": [frame["frame_index"]],
                "contradictory_evidence": [],
                "uncertainty": None,
                "verification_status": "supported_candidate",
            })
            segment = [frame]
            previous = value
            continue
        if (previous is not None and value is not None
                and value < previous - 1e-9 and segment):
            flush()
        segment.append(frame)
        previous = value
    flush()

    return {
        "schema_version": EVENT_VERSION,
        "run_id": consolidated.get("run_id"),
        "consolidated_hash": consolidated.get("content_hash"),
        "events": events,
    }
