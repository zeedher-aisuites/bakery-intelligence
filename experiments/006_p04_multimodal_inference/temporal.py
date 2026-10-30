"""Deterministic offline temporal analysis (Exp. 006).

Groups frame-level multimodal observations into evidence-linked
candidate events: scale-visibility intervals, stable readings, abrupt
changes, spikes, and grouped actions. Pure stdlib, no network, no
model calls. Never invents quantities, units, ingredients, or steps.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ARTIFACT_VERSION = "1.1.0"
NUMERIC_PATTERN = re.compile(r"^-?\d+(\.\d+)?$")
SIGN_SPACE_PATTERN = re.compile(r"^-\s+\d+(\.\d+)?$")

# Machine-readable review reasons. Every event carries review_reasons
# (empty for clean candidates); uncertainty_notes stay human-readable.
REASON_TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
REASON_AMBIGUOUS_PARSING = "ambiguous_parsing"
REASON_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
REASON_LOW_CONFIDENCE = "low_confidence_observation"


class TemporalError(Exception):
    """Structured analysis failure with a machine-readable code."""

    def __init__(self, code: str, message: str, entity_id: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.entity_id = entity_id


@dataclass(frozen=True)
class TemporalParams:
    """Tunables; every default is documented, none is a measurement."""

    gap_tolerance_seconds: float = 0.5
    min_stable_run: int = 3
    abrupt_step: float = 100.0
    associate_window_seconds: float = 5.0


@dataclass
class Reading:
    raw: str | None
    value: float | None
    status: str  # ok | null_input | unparseable | normalized
    note: str | None = None


def parse_reading(raw: str | None) -> Reading:
    """Normalize a display string only when the reading is unambiguous.

    A detached leading sign ("- 750") admits exactly one numeric parse
    and is normalized. Internal spacing between digits ("5 15") admits
    several readings (concatenation, separator, misread glyph) and is
    therefore ambiguous: value None, raw string preserved.
    """
    if raw is None:
        return Reading(None, None, "null_input")
    text = raw.strip()
    if not text:
        return Reading(raw, None, "unparseable", "empty_string")
    if NUMERIC_PATTERN.match(text):
        return Reading(raw, float(text), "ok")
    if SIGN_SPACE_PATTERN.match(text):
        compact = "".join(text.split())
        return Reading(
            raw, float(compact), "normalized", "sign_space_removed"
        )
    if NUMERIC_PATTERN.match("".join(text.split())):
        return Reading(
            raw, None, "unparseable", "ambiguous_internal_spacing"
        )
    return Reading(raw, None, "unparseable", "non_numeric_characters")


def load_observations(obs_dir: Path) -> list[dict[str, Any]]:
    """Load, validate, and timestamp-order raw observation records."""
    if not obs_dir.is_dir():
        raise TemporalError(
            "OBS_INVALID", f"Not a directory: {obs_dir}", str(obs_dir)
        )
    files = sorted(obs_dir.glob("batch_*.json"))
    records: list[dict[str, Any]] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TemporalError(
                "OBS_INVALID", f"Unreadable observations file: {path.name}",
                path.name,
            ) from exc
        if not isinstance(payload, list):
            raise TemporalError(
                "OBS_INVALID", f"Expected a list: {path.name}", path.name
            )
        for position, item in enumerate(payload):
            records.append(_validate_record(item, path.name, position))
    records.sort(
        key=lambda item: (item["timestamp_seconds"], item["frame_index"])
    )
    indices = [item["frame_index"] for item in records]
    if len(set(indices)) != len(indices):
        raise TemporalError("OBS_INVALID", "Duplicate frame_index values.")
    hashes = {item["manifest_ref"]["manifest_sha256"] for item in records}
    if len(hashes) > 1:
        raise TemporalError(
            "OBS_INVALID", "Observations span multiple manifests."
        )
    return records


def _validate_record(item: Any, source: str, position: int) -> dict[str, Any]:
    entity = f"{source}#{position}"
    if not isinstance(item, dict):
        raise TemporalError("OBS_INVALID", "Record is not an object.", entity)
    for key in ("frame_index", "image_path", "timestamp_seconds",
                "source_timestamp_seconds", "requested_timestamp_seconds",
                "timestamp_basis", "manifest_ref", "batch_id", "pass_id",
                "model_fields", "parse_error"):
        if key not in item:
            raise TemporalError(
                "OBS_INVALID", f"Missing key: {key}.", entity
            )
    if not isinstance(item["frame_index"], int):
        raise TemporalError(
            "OBS_INVALID", "frame_index is not an int.", entity
        )
    for key in ("timestamp_seconds", "source_timestamp_seconds",
                "requested_timestamp_seconds"):
        if not isinstance(item[key], (int, float)):
            raise TemporalError(
                "OBS_INVALID", f"{key} is not numeric.", entity
            )
    fields = item["model_fields"]
    if not isinstance(fields, dict):
        raise TemporalError(
            "OBS_INVALID", "model_fields is not an object.", entity
        )
    for key in ("scale_display_verbatim", "scale_readable", "scale_state",
                "tare_action_visible", "addition_or_removal_visible",
                "container_description", "container_changed_hint",
                "confidence"):
        if key not in fields:
            raise TemporalError(
                "OBS_INVALID", f"Missing model field: {key}.", entity
            )
    ref = item["manifest_ref"]
    if not isinstance(ref, dict) or "manifest_sha256" not in ref:
        raise TemporalError(
            "OBS_INVALID", "manifest_ref lacks manifest_sha256.", entity
        )
    return item


def _event_id(kind: str, first: int, last: int) -> str:
    return f"{kind}_{first:08d}_{last:08d}"


def _ref(item: dict[str, Any]) -> dict[str, Any]:
    return {"batch_id": item["batch_id"], "frame_index": item["frame_index"]}


def find_gaps(
    records: list[dict[str, Any]], params: TemporalParams
) -> list[dict[str, Any]]:
    """Timestamp gaps beyond cadence + tolerance (1.0 s + 0.5 s)."""
    gaps = []
    for left, right in zip(records, records[1:]):
        step = right["timestamp_seconds"] - left["timestamp_seconds"]
        if step > 1.0 + params.gap_tolerance_seconds:
            gaps.append({
                "after_frame_index": left["frame_index"],
                "before_frame_index": right["frame_index"],
                "gap_seconds": round(step, 6),
            })
    return gaps


def split_runs(
    records: list[dict[str, Any]],
    key: Any,
    params: TemporalParams,
) -> list[list[dict[str, Any]]]:
    """Maximal consecutive runs sharing a key; gaps always split."""
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for item in records:
        breaks = (
            previous is not None
            and (
                key(item) != key(previous)
                or item["timestamp_seconds"] - previous["timestamp_seconds"]
                > 1.0 + params.gap_tolerance_seconds
            )
        )
        if breaks:
            runs.append(current)
            current = []
        current.append(item)
        previous = item
    if current:
        runs.append(current)
    return runs


def build_visibility_intervals(
    records: list[dict[str, Any]], params: TemporalParams
) -> list[dict[str, Any]]:
    """Candidate scale-visibility intervals with consistency analysis."""
    events = []
    readable = [
        item for item in records
        if item["model_fields"].get("scale_readable") is True
    ]
    for run in split_runs(readable, lambda item: True, params):
        readings = [
            parse_reading(item["model_fields"].get("scale_display_verbatim"))
            for item in run
        ]
        steps: list[dict[str, Any]] = []
        for left_item, left, right_item, right in zip(
            run, readings, run[1:], readings[1:]
        ):
            if left.value is None or right.value is None:
                continue
            delta = abs(right.value - left.value)
            steps.append({
                "from_frame_index": left_item["frame_index"],
                "to_frame_index": right_item["frame_index"],
                "delta": delta,
                "abrupt": delta > params.abrupt_step,
            })
        spike_positions = _find_spikes(run, readings, params)
        malformed = [
            (item, reading) for item, reading in zip(run, readings)
            if reading.status == "unparseable"
        ]
        low_confidence = [
            item for item in run
            if item["model_fields"].get("confidence") == "low"
        ]
        abrupt = sum(1 for step in steps if step["abrupt"])
        reasons: list[str] = []
        if malformed:
            reasons.append(REASON_AMBIGUOUS_PARSING)
        if low_confidence:
            reasons.append(REASON_LOW_CONFIDENCE)
        if abrupt or spike_positions:
            reasons.append(REASON_TEMPORAL_INCONSISTENCY)
        needs_review = bool(reasons)
        notes = []
        if malformed:
            notes.append(
                f"{len(malformed)} unparseable display string(s)"
            )
        if low_confidence:
            notes.append(
                f"{len(low_confidence)} low-confidence frame(s)"
            )
        if abrupt:
            notes.append(f"{abrupt} abrupt step(s) over {params.abrupt_step}")
        if spike_positions:
            notes.append(f"{len(spike_positions)} spike(s)")
        contradicting = [
            f"frame {index}: spike" for index in spike_positions
        ]
        contradicting.extend(
            f"frame {item['frame_index']}: unparseable display "
            f"'{reading.raw}'"
            for item, reading in malformed
        )
        contradicting.extend(
            f"frame {item['frame_index']}: low-confidence observation"
            for item in low_confidence
        )
        events.append({
            "event_id": _event_id(
                "scale_visibility", run[0]["frame_index"],
                run[-1]["frame_index"],
            ),
            "candidate_type": "scale_visibility",
            "start_timestamp_seconds": run[0]["timestamp_seconds"],
            "end_timestamp_seconds": run[-1]["timestamp_seconds"],
            "frame_indices": [item["frame_index"] for item in run],
            "observation_refs": [_ref(item) for item in run],
            "scale_verbatim": [reading.raw for reading in readings],
            "parsed_values": [
                {"value": reading.value, "status": reading.status,
                 "note": reading.note}
                for reading in readings
            ],
            "consistency": {
                "max_step": max(
                    (step["delta"] for step in steps), default=0.0
                ),
                "steps": steps,
                "spikes": spike_positions,
            },
            "supporting_evidence": [
                f"{len(run)} consecutive readable frames"
            ],
            "contradicting_evidence": contradicting,
            "verification_status": (
                "needs_review" if needs_review else "candidate"
            ),
            "review_reasons": reasons,
            "uncertainty_notes": notes or ["none noted"],
        })
    return events


def _find_spikes(
    run: list[dict[str, Any]],
    readings: list[Reading],
    params: TemporalParams,
) -> list[int]:
    """Frames differing from both neighbors while neighbors agree."""
    spikes = []
    for pos in range(1, len(run) - 1):
        left, current, right = (
            readings[pos - 1], readings[pos], readings[pos + 1]
        )
        if current.value is None or left.value is None or right.value is None:
            continue
        if (
            abs(current.value - left.value) > params.abrupt_step
            and abs(current.value - right.value) > params.abrupt_step
            and abs(right.value - left.value) <= params.abrupt_step
        ):
            spikes.append(run[pos]["frame_index"])
    return spikes


def build_stable_readings(
    records: list[dict[str, Any]],
    visibility: list[dict[str, Any]],
    params: TemporalParams,
) -> list[dict[str, Any]]:
    """Maximal runs of numerically equal parsed readings.

    Equality is exact: no numeric tolerance merges adjacent runs of
    distinct values (e.g. 69 vs 68), and runs never span timestamp
    gaps. Each run records its enclosing visibility interval(s); a
    run whose own frames or immediately bordering observations are
    uncertain or contradictory is needs_review rather than a
    verified-looking candidate.
    """
    events = []
    interval_by_frame: dict[int, dict[str, Any]] = {}
    for event in visibility:
        for index in event["frame_indices"]:
            interval_by_frame[index] = event
    position_by_frame = {
        item["frame_index"]: position
        for position, item in enumerate(records)
    }
    keyed = []
    for item in records:
        reading = parse_reading(
            item["model_fields"].get("scale_display_verbatim")
        )
        if item["model_fields"].get("scale_readable") is True \
                and reading.value is not None:
            keyed.append((item, reading))
    runs: list[list[tuple[dict[str, Any], Reading]]] = []
    current: list[tuple[dict[str, Any], Reading]] = []
    previous: tuple[dict[str, Any], Reading] | None = None
    for pair in keyed:
        breaks = (
            previous is not None
            and (
                pair[1].value != previous[1].value
                or pair[0]["timestamp_seconds"]
                - previous[0]["timestamp_seconds"]
                > 1.0 + params.gap_tolerance_seconds
            )
        )
        if breaks:
            runs.append(current)
            current = []
        current.append(pair)
        previous = pair
    if current:
        runs.append(current)
    for run in runs:
        if len(run) < params.min_stable_run:
            continue
        items = [pair[0] for pair in run]
        value = run[0][1].value
        assert value is not None  # keyed holds parsed readings only
        low = [
            item for item in items
            if item["model_fields"].get("confidence") == "low"
        ]
        enclosing: list[dict[str, Any]] = []
        for item in items:
            interval = interval_by_frame[item["frame_index"]]
            if not enclosing or enclosing[-1]["event_id"] != (
                    interval["event_id"]):
                enclosing.append(interval)
        reasons: list[str] = []
        contradicting: list[str] = []
        notes: list[str] = []
        if low:
            reasons.append(REASON_LOW_CONFIDENCE)
            notes.append(f"{len(low)} low-confidence frame(s)")
            contradicting.extend(
                f"frame {item['frame_index']}: low-confidence "
                "observation"
                for item in low
            )
        border_reasons, border_contra = _stable_border_issues(
            items, value, records, position_by_frame, params
        )
        for reason in border_reasons:
            if reason not in reasons:
                reasons.append(reason)
        contradicting.extend(border_contra)
        spikes = {
            frame for interval in enclosing
            for frame in interval["consistency"]["spikes"]
        }
        overlapped = sorted(
            set(pair[0]["frame_index"] for pair in run) & spikes
        )
        if overlapped:
            if REASON_TEMPORAL_INCONSISTENCY not in reasons:
                reasons.append(REASON_TEMPORAL_INCONSISTENCY)
            contradicting.extend(
                f"frame {index}: spike within stable run"
                for index in overlapped
            )
        flagged_enclosing = [
            interval for interval in enclosing
            if interval["verification_status"] == "needs_review"
        ]
        if flagged_enclosing and not reasons:
            notes.append(
                "enclosing interval(s) "
                + ", ".join(
                    interval["event_id"] for interval in flagged_enclosing
                )
                + " need review; run borders locally consistent"
            )
        elif not notes:
            notes.append("reading stability only; not a verified mass")
        events.append({
            "event_id": _event_id(
                "stable_reading", items[0]["frame_index"],
                items[-1]["frame_index"],
            ),
            "candidate_type": "stable_reading",
            "start_timestamp_seconds": items[0]["timestamp_seconds"],
            "end_timestamp_seconds": items[-1]["timestamp_seconds"],
            "frame_indices": [item["frame_index"] for item in items],
            "observation_refs": [_ref(item) for item in items],
            "scale_verbatim": [pair[1].raw for pair in run],
            "parsed_values": [
                {"value": pair[1].value, "status": pair[1].status,
                 "note": pair[1].note}
                for pair in run
            ],
            "consistency": {
                "run_length": len(run),
                "distinct_values": 1,
                "value": value,
                "enclosing_intervals": [
                    {"event_id": interval["event_id"],
                     "verification_status":
                         interval["verification_status"]}
                    for interval in enclosing
                ],
            },
            "supporting_evidence": [
                f"{len(run)} consecutive equal readings of {value}"
            ],
            "contradicting_evidence": contradicting,
            "verification_status": (
                "needs_review" if reasons else "candidate"
            ),
            "review_reasons": reasons,
            "uncertainty_notes": notes,
        })
    return events


def _stable_border_issues(
    items: list[dict[str, Any]],
    value: float,
    records: list[dict[str, Any]],
    position_by_frame: dict[int, int],
    params: TemporalParams,
) -> tuple[list[str], list[str]]:
    """Inconsistency in observations bordering a stable run.

    Only readable neighbors are examined: an unreadable neighbor is
    an ordinary segmentation boundary, not evidence against the run.
    """
    reasons: list[str] = []
    contradicting: list[str] = []
    neighbors = []
    first = position_by_frame[items[0]["frame_index"]]
    last = position_by_frame[items[-1]["frame_index"]]
    if first > 0:
        neighbors.append(("preceding", records[first - 1]))
    if last + 1 < len(records):
        neighbors.append(("following", records[last + 1]))
    for side, neighbor in neighbors:
        if neighbor["model_fields"].get("scale_readable") is not True:
            continue
        reading = parse_reading(
            neighbor["model_fields"].get("scale_display_verbatim")
        )
        frame = neighbor["frame_index"]
        if reading.value is None and reading.status == "unparseable":
            if REASON_AMBIGUOUS_PARSING not in reasons:
                reasons.append(REASON_AMBIGUOUS_PARSING)
            contradicting.append(
                f"frame {frame}: unparseable {side} display "
                f"'{reading.raw}'"
            )
        elif reading.value is not None and (
                abs(reading.value - value) > params.abrupt_step):
            if REASON_TEMPORAL_INCONSISTENCY not in reasons:
                reasons.append(REASON_TEMPORAL_INCONSISTENCY)
            contradicting.append(
                f"frame {frame}: abrupt {side} step "
                f"(delta {abs(reading.value - value)})"
            )
        if neighbor["model_fields"].get("confidence") == "low":
            if REASON_LOW_CONFIDENCE not in reasons:
                reasons.append(REASON_LOW_CONFIDENCE)
            contradicting.append(
                f"frame {frame}: low-confidence {side} observation"
            )
    return reasons, contradicting


def build_action_events(
    records: list[dict[str, Any]], params: TemporalParams
) -> list[dict[str, Any]]:
    """Grouped addition, removal, and tare candidates (no per-frame events)."""
    events = []

    def label(item: dict[str, Any]) -> str:
        return str(
            item["model_fields"].get("addition_or_removal_visible")
        )

    for run in split_runs(records, label, params):
        kind = label(run[0])
        if kind not in ("addition", "removal"):
            continue
        events.append(_action_event(run, f"action_{kind}", params))

    tare_runs = split_runs(
        [item for item in records
         if item["model_fields"].get("tare_action_visible") is True],
        lambda item: True, params,
    )
    for run in tare_runs:
        events.append(_action_event(run, "action_tare", params))
    return events


def _action_event(
    run: list[dict[str, Any]], kind: str, params: TemporalParams
) -> dict[str, Any]:
    low = [
        item for item in run
        if item["model_fields"].get("confidence") == "low"
    ]
    reasons = [REASON_LOW_CONFIDENCE] if low else []
    contradicting = [
        f"frame {item['frame_index']}: low-confidence observation"
        for item in low
    ]
    return {
        "event_id": _event_id(
            kind, run[0]["frame_index"], run[-1]["frame_index"]
        ),
        "candidate_type": kind,
        "start_timestamp_seconds": run[0]["timestamp_seconds"],
        "end_timestamp_seconds": run[-1]["timestamp_seconds"],
        "frame_indices": [item["frame_index"] for item in run],
        "observation_refs": [_ref(item) for item in run],
        "scale_verbatim": [],
        "parsed_values": [],
        "consistency": {"frame_count": len(run)},
        "supporting_evidence": [f"{len(run)} consecutive '{kind}' frame(s)"],
        "contradicting_evidence": contradicting,
        "verification_status": "needs_review" if low else "candidate",
        "review_reasons": reasons,
        "uncertainty_notes": (
            [f"{len(low)} low-confidence frame(s)"] if low
            else ["single grouped action; not confirmed"]
        ),
        "associated_scale_events": [],
    }


def associate_actions(
    actions: list[dict[str, Any]],
    scale_events: list[dict[str, Any]],
    params: TemporalParams,
) -> None:
    """Link actions to temporally overlapping or nearby scale events."""
    window = params.associate_window_seconds
    for action in actions:
        start = action["start_timestamp_seconds"] - window
        end = action["end_timestamp_seconds"] + window
        linked = [
            event["event_id"] for event in scale_events
            if event["start_timestamp_seconds"] <= end
            and event["end_timestamp_seconds"] >= start
        ]
        action["associated_scale_events"] = linked
        if not linked:
            action["verification_status"] = "needs_review"
            if REASON_INSUFFICIENT_EVIDENCE not in (
                    action["review_reasons"]):
                action["review_reasons"] = list(
                    action["review_reasons"]
                ) + [REASON_INSUFFICIENT_EVIDENCE]
            action["uncertainty_notes"] = list(
                action["uncertainty_notes"]
            ) + ["no nearby scale evidence"]
        else:
            action["supporting_evidence"] = list(
                action["supporting_evidence"]
            ) + [f"nearby scale event(s): {', '.join(linked)}"]


def analyze(
    obs_dir: Path,
    output_path: Path,
    params: TemporalParams | None = None,
) -> dict[str, Any]:
    """Run the full offline analysis and write the artifact."""
    active = params or TemporalParams()
    records = load_observations(obs_dir)
    visibility = build_visibility_intervals(records, active)
    stable = build_stable_readings(records, visibility, active)
    actions = build_action_events(records, active)
    associate_actions(actions, visibility, active)
    events = visibility + stable + actions
    events.sort(key=lambda event: (
        event["start_timestamp_seconds"], event["event_id"]
    ))
    by_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    for event in events:
        by_type[event["candidate_type"]] = (
            by_type.get(event["candidate_type"], 0) + 1
        )
        by_status[event["verification_status"]] = (
            by_status.get(event["verification_status"], 0) + 1
        )
        for reason in event["review_reasons"]:
            by_reason[reason] = by_reason.get(reason, 0) + 1
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "source": {
            "observations_dir": str(obs_dir),
            "observation_count": len(records),
            "manifest_sha256": (
                records[0]["manifest_ref"]["manifest_sha256"]
                if records else None
            ),
        },
        "parameters": {
            "gap_tolerance_seconds": active.gap_tolerance_seconds,
            "min_stable_run": active.min_stable_run,
            "abrupt_step": active.abrupt_step,
            "associate_window_seconds": active.associate_window_seconds,
        },
        "gaps": find_gaps(records, active),
        "events": events,
        "summary": {
            "event_count": len(events),
            "by_type": by_type,
            "by_status": by_status,
            # Reason occurrences across events; one event with two
            # reasons contributes to both counters.
            "by_review_reason": by_reason,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    return artifact


def main(argv: list[str] | None = None) -> int:
    """Minimal CLI: analyze OBS_DIR into OUTPUT_JSON."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("obs_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--gap-tolerance", type=float, default=0.5)
    parser.add_argument("--min-stable-run", type=int, default=3)
    parser.add_argument("--abrupt-step", type=float, default=100.0)
    parser.add_argument("--associate-window", type=float, default=5.0)
    args = parser.parse_args(argv)
    try:
        artifact = analyze(args.obs_dir, args.output_json, TemporalParams(
            gap_tolerance_seconds=args.gap_tolerance,
            min_stable_run=args.min_stable_run,
            abrupt_step=args.abrupt_step,
            associate_window_seconds=args.associate_window,
        ))
    except TemporalError as exc:
        print(f"ANALYSIS FAILED [{exc.code}]: {exc.message}")
        return 1
    print(json.dumps(artifact["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
