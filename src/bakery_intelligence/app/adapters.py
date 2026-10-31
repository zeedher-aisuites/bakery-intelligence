"""Read-only adapters over M2-04 reconstruction artifacts.

Loads the six JSON outputs of a reconstruction run without modifying
them. Missing files produce explicit issues instead of exceptions;
corrupt files produce explicit issues and empty defaults. All paths
are resolved relative to the repository root so the application
works from any working directory.

This module must never import Streamlit: it is exercised by plain
offline unit tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RECONSTRUCTION_RELDIR = Path("data/processed/experiment_008/reconstruction")
RUN_DIRNAME = "v01_full"
VIDEO_RELATIVE = Path("data/raw/epic_kitchens/P04_104.MP4")

MEASUREMENTS_FILENAME = "measurements.json"
RECIPE_DRAFT_FILENAME = "recipe_draft.json"
WEIGHING_EVENT_FILENAME = "weighing_event.json"
ACTION_EVENTS_FILENAME = "action_events.json"
UNITS_APPLIED_FILENAME = "units_applied.json"
VALIDATION_FILENAME = "validation.json"


@dataclass(frozen=True)
class LoadIssue:
    """One problem encountered while loading a reconstruction run."""

    path: str
    code: str
    message: str


@dataclass(frozen=True)
class ReconstructionBundle:
    """All artifacts of one reconstruction run plus load issues."""

    root: Path
    run_id: str
    measurements: dict[str, Any] = field(default_factory=dict)
    recipe_draft: dict[str, Any] = field(default_factory=dict)
    weighing_event: dict[str, Any] = field(default_factory=dict)
    action_events: tuple[dict[str, Any], ...] = ()
    units_applied: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    issues: tuple[LoadIssue, ...] = ()


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward until a directory containing pyproject.toml is found."""
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError("Repository root (pyproject.toml) not found.")


def run_directory(root: Path | None = None, run_id: str = RUN_DIRNAME) -> Path:
    """Resolve the reconstruction output directory for a run."""
    base = root or find_repo_root()
    return base / RECONSTRUCTION_RELDIR / run_id


def video_path(root: Path | None = None) -> Path:
    """Resolve the expected local source video path (may not exist)."""
    base = root or find_repo_root()
    return base / VIDEO_RELATIVE


def _read_json(path: Path, issues: list[LoadIssue]) -> Any:
    if not path.is_file():
        issues.append(LoadIssue(
            path=str(path), code="missing_file",
            message=f"Artifact not found: {path.name}.",
        ))
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        issues.append(LoadIssue(
            path=str(path), code="corrupt_file",
            message=f"Artifact is not valid JSON: {exc}.",
        ))
        return None


def load_bundle(root: Path | None = None,
               run_id: str = RUN_DIRNAME) -> ReconstructionBundle:
    """Load every artifact of a reconstruction run (read-only)."""
    base = root or find_repo_root()
    run_dir = run_directory(base, run_id)
    issues: list[LoadIssue] = []
    measurements = _read_json(run_dir / MEASUREMENTS_FILENAME, issues) or {}
    recipe_draft = _read_json(run_dir / RECIPE_DRAFT_FILENAME, issues) or {}
    weighing_event = _read_json(run_dir / WEIGHING_EVENT_FILENAME, issues) or {}
    action_events = _read_json(run_dir / ACTION_EVENTS_FILENAME, issues) or []
    units_applied = _read_json(run_dir / UNITS_APPLIED_FILENAME, issues) or {}
    validation = _read_json(run_dir / VALIDATION_FILENAME, issues) or {}
    return ReconstructionBundle(
        root=base,
        run_id=run_id,
        measurements=measurements if isinstance(measurements, dict) else {},
        recipe_draft=recipe_draft if isinstance(recipe_draft, dict) else {},
        weighing_event=weighing_event if isinstance(weighing_event, dict) else {},
        action_events=tuple(action_events) if isinstance(action_events, list) else (),
        units_applied=units_applied if isinstance(units_applied, dict) else {},
        validation=validation if isinstance(validation, dict) else {},
        issues=tuple(issues),
    )


def unit_for_measurement(bundle: ReconstructionBundle,
                         measurement_id: str) -> dict[str, Any] | None:
    """Return the unit resolution for a measurement, or None if absent."""
    resolutions = bundle.measurements.get("unit_resolutions", [])
    for entry in resolutions:
        if entry.get("measurement_id") == measurement_id:
            return dict(entry)
    return None


def measurement_rows(bundle: ReconstructionBundle) -> list[dict[str, Any]]:
    """Flatten measurements into display rows with unit provenance."""
    rows: list[dict[str, Any]] = []
    for item in bundle.measurements.get("measurements", []):
        original = item.get("original", {}) or {}
        verified = item.get("verified", {}) or {}
        resolution = unit_for_measurement(
            bundle, item.get("measurement_id", ""))
        rows.append({
            "measurement_id": item.get("measurement_id"),
            "timestamp_seconds": item.get("timestamp_seconds"),
            "frame_index": item.get("frame_index"),
            "original_verbatim": original.get("verbatim"),
            "original_readable": original.get("readable"),
            "original_confidence": original.get("confidence"),
            "value": item.get("value"),
            "value_source": item.get("value_source"),
            "value_status": item.get("value_status"),
            "verified_display": verified.get("display_text") if verified else None,
            "verified_numeric": verified.get("numeric_value") if verified else None,
            "verified_confidence": verified.get("confidence") if verified else None,
            "delta_from_previous": item.get("delta_from_previous"),
            "unit_observed": item.get("unit_observed"),
            "unit_assigned": resolution.get("unit") if resolution else None,
            "unit_provenance": resolution.get("provenance") if resolution else None,
            "unit_assignment_id": (
                resolution.get("assignment_id") if resolution else None),
            "has_disagreement": _has_disagreement(original, verified),
        })
    return rows


def _has_disagreement(original: dict[str, Any],
                      verified: dict[str, Any]) -> bool:
    """Detect first-pass/second-pass display-text disagreement."""
    if not verified:
        return False
    verbatim = original.get("verbatim")
    display = verified.get("display_text")
    if verbatim is None or display is None:
        return False
    return str(verbatim).strip() != str(display).strip()


def unresolved_questions(bundle: ReconstructionBundle) -> list[dict[str, Any]]:
    """Return the draft's unresolved questions (never invented)."""
    questions = bundle.recipe_draft.get("unresolved_questions", [])
    return [dict(question) for question in questions]


def event_window(bundle: ReconstructionBundle) -> dict[str, Any]:
    """Return the reconstructed event window with its identifier."""
    window = bundle.measurements.get("window", {}) or {}
    return {
        "event_id": bundle.measurements.get("event_id"),
        "start_seconds": window.get("start_seconds"),
        "end_seconds": window.get("end_seconds"),
    }
