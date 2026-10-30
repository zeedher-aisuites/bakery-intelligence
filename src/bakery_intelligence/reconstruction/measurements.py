"""Timestamped measurement assembly from existing evidence.

Loads first-pass observations, temporal events, and verification
reports; assembles per-frame measurements preferring valid
verification results while preserving original links; computes
consecutive differences as plain arithmetic, never as quantities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bakery_intelligence.verification.numerics import (
    STATUS_OK,
    normalize_number,
)


class ReconstructionError(Exception):
    """Structured reconstruction failure with a code."""

    def __init__(self, code: str, message: str, entity_id: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.entity_id = entity_id


def load_observations_index(obs_dir: Path) -> dict[int, dict[str, Any]]:
    """Index first-pass observations by frame (read-only)."""
    if not obs_dir.is_dir():
        raise ReconstructionError("OBS_INVALID",
                                  f"Not a directory: {obs_dir}.",
                                  str(obs_dir))
    indexed: dict[int, dict[str, Any]] = {}
    for path in sorted(obs_dir.glob("batch_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReconstructionError(
                "OBS_INVALID", f"Unreadable: {path.name}.",
                path.name,
            ) from exc
        if not isinstance(payload, list):
            raise ReconstructionError(
                "OBS_INVALID", f"Expected a list: {path.name}.",
                path.name,
            )
        for item in payload:
            if not isinstance(item, dict) or "frame_index" not in item:
                raise ReconstructionError(
                    "OBS_INVALID",
                    f"Record without frame_index: {path.name}.",
                    path.name,
                )
            indexed[int(item["frame_index"])] = item
    return indexed


def load_temporal_artifact(path: Path) -> dict[str, Any]:
    """Load a temporal artifact with structural validation."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionError(
            "ARTIFACT_INVALID", f"Unreadable artifact: {path}.",
            str(path),
        ) from exc
    if not isinstance(payload, dict) or "events" not in payload:
        raise ReconstructionError(
            "ARTIFACT_INVALID", "Artifact lacks events.", str(path))
    return payload


def find_temporal_event(artifact: dict[str, Any],
                        event_id: str) -> dict[str, Any]:
    """Locate one temporal event by identifier."""
    for event in artifact["events"]:
        if isinstance(event, dict) and event.get("event_id") == event_id:
            return event
    raise ReconstructionError("UNKNOWN_EVENT",
                              f"Event not in artifact: {event_id}.",
                              event_id)


def load_execution_reports(
        exec_dir: Path | None) -> dict[str, dict[str, Any]]:
    """Index verification reports by event (missing dir yields none)."""
    if exec_dir is None or not exec_dir.is_dir():
        return {}
    reports = {}
    for path in sorted(exec_dir.glob("*/execution_report.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReconstructionError(
                "REPORT_INVALID", f"Unreadable report: {path}.",
                str(path),
            ) from exc
        if not isinstance(payload, dict) or "event_id" not in payload:
            raise ReconstructionError(
                "REPORT_INVALID", f"Report lacks event_id: {path}.",
                str(path),
            )
        reports[payload["event_id"]] = payload
    return reports


def verified_by_timestamp(
        report: dict[str, Any] | None) -> dict[float, dict[str, Any]]:
    """Selected verified readings keyed by timestamp."""
    if report is None or report.get("terminal_state") != (
            "valid_prediction"):
        return {}
    readings = report.get("selected_readings") or []
    mapped = {}
    for reading in readings:
        if isinstance(reading, dict) and isinstance(
                reading.get("timestamp_seconds"), (int, float)):
            mapped[float(reading["timestamp_seconds"])] = reading
    return mapped


def _original_fields(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    fields = record.get("model_fields", {})
    return {
        "verbatim": fields.get("scale_display_verbatim"),
        "readable": fields.get("scale_readable"),
        "confidence": fields.get("confidence"),
        "scale_state": fields.get("scale_state"),
        "tare_visible": fields.get("tare_action_visible"),
        "action_visible": fields.get("addition_or_removal_visible"),
        "batch_id": record.get("batch_id"),
        "timestamp_seconds": record.get("timestamp_seconds"),
    }


def assemble_measurements(
        event: dict[str, Any],
        obs_index: dict[int, dict[str, Any]],
        verified: dict[float, dict[str, Any]],
        start: float | None = None,
        end: float | None = None) -> list[dict[str, Any]]:
    """Build ordered measurements, preferring verified readings.

    Optional [start, end] bounds restrict to a sub-window of the
    event without modifying the event definition. Deltas are plain
    consecutive differences, never ingredient quantities.
    """
    frames = sorted(event.get("frame_indices", []))
    measurements = []
    previous_value: float | None = None
    for frame in frames:
        record = obs_index.get(int(frame))
        original = _original_fields(record)
        timestamp = None
        if record is not None:
            timestamp = record.get("timestamp_seconds")
        if timestamp is None:
            timestamp = float(frame) / 50.0
        if start is not None and timestamp < start:
            continue
        if end is not None and timestamp > end:
            continue
        verified_reading = verified.get(float(timestamp))
        value = None
        status = "missing"
        source = "none"
        if verified_reading is not None and (
                verified_reading.get("readable") is not False):
            candidate = verified_reading.get("numeric_value")
            if candidate is None:
                candidate = verified_reading.get("display_text")
            normalized = normalize_number(candidate)
            if normalized.status == STATUS_OK:
                value = normalized.value
                status = STATUS_OK
                source = "verification"
            else:
                status = normalized.status
                source = "verification_unusable"
        if value is None and original is not None:
            normalized = normalize_number(original["verbatim"])
            status = normalized.status
            source = "firstpass" if status == STATUS_OK else (
                "firstpass_unusable")
            value = normalized.value
        delta = None
        if value is not None and previous_value is not None:
            delta = value - previous_value
        if value is not None:
            previous_value = value
        measurements.append({
            "measurement_id": f"m_{int(frame):08d}",
            "frame_index": int(frame),
            "timestamp_seconds": float(timestamp),
            "original": original,
            "verified": (
                {key: verified_reading.get(key) for key in
                 ("display_text", "numeric_value", "readable",
                  "confidence", "uncertainty_notes")}
                if verified_reading is not None else None),
            "value_source": source,
            "value": value,
            "value_status": status,
            "delta_from_previous": delta,
            "unit_observed": None,
        })
    return measurements


def scan_action_evidence(
        event: dict[str, Any], obs_index: dict[int, dict[str, Any]],
        artifact: dict[str, Any]) -> dict[str, Any]:
    """Collect tare/addition/removal evidence overlapping the event."""
    start = event["start_timestamp_seconds"]
    end = event["end_timestamp_seconds"]
    tare_frames: list[int] = []
    addition_frames: list[int] = []
    removal_frames: list[int] = []
    for frame in sorted(event.get("frame_indices", [])):
        record = obs_index.get(int(frame))
        if record is None:
            continue
        fields = record.get("model_fields", {})
        if fields.get("tare_action_visible") is True:
            tare_frames.append(int(frame))
        visible = fields.get("addition_or_removal_visible")
        if visible == "addition":
            addition_frames.append(int(frame))
        elif visible == "removal":
            removal_frames.append(int(frame))
    temporal_actions = []
    for candidate in artifact.get("events", []):
        if not isinstance(candidate, dict):
            continue
        kind = candidate.get("candidate_type", "")
        if not kind.startswith("action_"):
            continue
        if candidate.get("start_timestamp_seconds", 0) <= end and (
                candidate.get("end_timestamp_seconds", 0) >= start):
            temporal_actions.append({
                "event_id": candidate.get("event_id"),
                "candidate_type": kind,
                "start_timestamp_seconds": candidate.get(
                    "start_timestamp_seconds"),
                "end_timestamp_seconds": candidate.get(
                    "end_timestamp_seconds"),
                "frame_indices": list(
                    candidate.get("frame_indices", [])),
                "verification_status": candidate.get(
                    "verification_status"),
            })
    temporal_actions.sort(key=lambda item: item["event_id"] or "")
    return {
        "tare_frames": tare_frames,
        "addition_frames": addition_frames,
        "removal_frames": removal_frames,
        "temporal_actions": temporal_actions,
    }
