"""Append-only human unit assignments with conflict control.

A sidecar file records human-confirmed measurement units separately
from visual evidence: observed units stay null unless pixels show
them, while human assignments carry scope, revision history, and
provenance. Scale scopes accept mass units only; volume units are
rejected because scale-to-volume conversion needs separately
justified evidence that this layer does not possess.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bakery_intelligence.reconstruction.measurements import (
    ReconstructionError,
)

SIDECAR_VERSION = "1.0.0"
MASS_UNITS = frozenset({"g", "kg", "mg", "oz", "lb"})
VOLUME_UNITS = frozenset({"ml", "l"})
SCOPE_EVENT_WINDOW = "event_window"
SCOPE_SESSION = "session"


def new_sidecar() -> dict[str, Any]:
    """Create an empty assignment sidecar."""
    return {"sidecar_version": SIDECAR_VERSION, "assignments": []}


def load_sidecar(path: Path | None) -> dict[str, Any]:
    """Load a sidecar; a missing path yields an empty one."""
    if path is None or not path.exists():
        return new_sidecar()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionError(
            "UNITS_INVALID", f"Unreadable sidecar: {path}.",
            str(path),
        ) from exc
    if not isinstance(payload, dict) or "assignments" not in payload:
        raise ReconstructionError(
            "UNITS_INVALID", "Sidecar lacks assignments.", str(path))
    if not isinstance(payload["assignments"], list):
        raise ReconstructionError(
            "UNITS_INVALID", "Sidecar assignments are not a list.",
            str(path))
    return payload


def save_sidecar(path: Path, sidecar: dict[str, Any]) -> None:
    """Write a sidecar deterministically (never touches evidence)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )


def _unit_kind(unit: str) -> str:
    if unit in MASS_UNITS:
        return "mass"
    if unit in VOLUME_UNITS:
        return "volume"
    raise ReconstructionError(
        "UNIT_UNKNOWN", f"Unsupported unit: {unit!r}.", unit)


def _overlaps(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return first["start_seconds"] <= second["end_seconds"] and (
        second["start_seconds"] <= first["end_seconds"])


def assign_unit(sidecar: dict[str, Any], unit: str, scope_type: str,
                event_id: str | None, start: float, end: float,
                reviewer: str, justification: str | None = None,
                note: str | None = None) -> dict[str, Any]:
    """Append a human unit assignment; returns the updated sidecar.

    Overlapping active scopes with different units are rejected as
    conflicts. Repeating an identical scope and unit supersedes the
    previous revision (history preserved). Session scope requires an
    explicit justification; volume units are rejected for scale data.
    """
    kind = _unit_kind(unit)
    if scope_type not in (SCOPE_EVENT_WINDOW, SCOPE_SESSION):
        raise ReconstructionError(
            "SCOPE_INVALID", f"Unknown scope type: {scope_type!r}.",
            scope_type)
    if scope_type == SCOPE_SESSION and not justification:
        raise ReconstructionError(
            "SCOPE_INVALID",
            "Session scope requires an explicit justification.",
            unit,
        )
    if kind != "mass":
        raise ReconstructionError(
            "VOLUME_FORBIDDEN",
            f"Scale measurements accept mass units only: {unit!r}.",
            unit,
        )
    if not (isinstance(start, (int, float))
            and isinstance(end, (int, float))):
        raise ReconstructionError(
            "SCOPE_INVALID", "Scope bounds must be numeric.", unit)
    if not start < end:
        raise ReconstructionError(
            "SCOPE_INVALID",
            f"Scope requires start < end: {start!r}, {end!r}.", unit)
    if not reviewer or not isinstance(reviewer, str):
        raise ReconstructionError(
            "REVIEWER_MISSING", "Assignment requires a reviewer id.",
            unit)
    candidate = {"start_seconds": float(start),
                 "end_seconds": float(end)}
    for existing in sidecar["assignments"]:
        if existing.get("status") != "active":
            continue
        if not _overlaps(candidate, existing):
            continue
        if existing.get("unit") != unit:
            raise ReconstructionError(
                "UNIT_CONFLICT",
                f"Overlapping assignment {existing['assignment_id']} "
                f"uses {existing.get('unit')!r}.", unit,
            )
    same_scope = [
        item for item in sidecar["assignments"]
        if item.get("status") == "active"
        and item.get("scope_type") == scope_type
        and item.get("event_id") == event_id
        and item.get("start_seconds") == float(start)
        and item.get("end_seconds") == float(end)
        and item.get("unit") == unit
    ]
    previous_id = None
    revision = 1
    for item in same_scope:
        item["status"] = "superseded"
        previous_id = item["assignment_id"]
        revision = max(revision, item.get("revision", 1) + 1)
    assignment = {
        "assignment_id": f"ua_{len(sidecar['assignments']) + 1:04d}",
        "unit": unit,
        "unit_kind": kind,
        "scope_type": scope_type,
        "event_id": event_id,
        "start_seconds": float(start),
        "end_seconds": float(end),
        "source": "human",
        "reviewer": reviewer,
        "justification": justification,
        "revision": revision,
        "previous_assignment_id": previous_id,
        "status": "active",
        "note": note,
    }
    sidecar["assignments"].append(assignment)
    return sidecar


def resolve_unit(timestamp: float, event_id: str | None,
                 sidecar: dict[str, Any]) -> dict[str, Any]:
    """Resolve the applicable unit for one measurement timestamp.

    Event-window assignments take precedence over session scope;
    among equal scopes the latest revision wins. No assignment
    yields undetermined, never a guessed unit.
    """
    candidates = [
        item for item in sidecar.get("assignments", [])
        if item.get("status") == "active"
        and item.get("start_seconds") <= timestamp <= item.get(
            "end_seconds")
        and (item.get("scope_type") == SCOPE_SESSION
             or item.get("event_id") == event_id)
    ]
    if not candidates:
        return {"unit": None, "provenance": "undetermined",
                "assignment_id": None}
    rank = {SCOPE_EVENT_WINDOW: 0, SCOPE_SESSION: 1}
    candidates.sort(key=lambda item: (
        rank.get(item.get("scope_type"), 2), -item.get("revision", 1)))
    chosen = candidates[0]
    return {"unit": chosen["unit"], "provenance": "human_assigned",
            "assignment_id": chosen["assignment_id"]}
