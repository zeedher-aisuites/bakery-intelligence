"""Offline reconstruction CLI (M2-04).

Subcommands: ``reconstruct`` builds a validated RecipeDraft from
existing P04 artifacts plus an optional unit sidecar;
``assign-unit`` appends a human unit assignment revision. No
inference, no network, no history rewriting.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any

from bakery_intelligence.reconstruction.draft import (
    build_action_events,
    build_draft,
    build_unresolved,
    build_video_source,
    build_weighing_event,
    validate_bundle,
)
from bakery_intelligence.reconstruction.measurements import (
    ReconstructionError,
    assemble_measurements,
    find_temporal_event,
    load_execution_reports,
    load_observations_index,
    load_temporal_artifact,
    scan_action_evidence,
    verified_by_timestamp,
)
from bakery_intelligence.reconstruction.units import (
    assign_unit,
    load_sidecar,
    resolve_unit,
    save_sidecar,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )


def reconstruct(obs_dir: Path, temporal_path: Path, event_id: str,
                out_dir: Path, video_id: str,
                exec_dir: Path | None = None,
                units_path: Path | None = None,
                start: float | None = None, end: float | None = None,
                draft_version: int = 1,
                source_dataset: str | None = None) -> dict[str, Any]:
    """Run the full deterministic reconstruction for one event."""
    if out_dir.exists():
        raise ReconstructionError(
            "OUTPUT_EXISTS",
            f"Refusing to overwrite run: {out_dir}.", str(out_dir))
    obs_index = load_observations_index(obs_dir)
    artifact = load_temporal_artifact(temporal_path)
    event = find_temporal_event(artifact, event_id)
    if (start is None) != (end is None):
        raise ReconstructionError(
            "WINDOW_INVALID",
            "Sub-window requires both start and end.", event_id)
    if start is not None and end is not None:
        if not start < end:
            raise ReconstructionError(
                "WINDOW_INVALID",
                f"Sub-window requires start < end: {start}, {end}.",
                event_id)
        if start < event["start_timestamp_seconds"] or (
                end > event["end_timestamp_seconds"]):
            raise ReconstructionError(
                "WINDOW_OUT_OF_BOUNDS",
                "Sub-window extends outside the event; not clamped.",
                event_id)
    reports = load_execution_reports(exec_dir)
    verified = verified_by_timestamp(reports.get(event_id))
    measurements = assemble_measurements(
        event, obs_index, verified, start, end)
    if not measurements:
        raise ReconstructionError(
            "NO_MEASUREMENTS",
            "No measurements in the selected interval.", event_id)
    sidecar = load_sidecar(units_path)
    resolutions = []
    for item in measurements:
        resolution = resolve_unit(
            item["timestamp_seconds"], event_id, sidecar)
        resolutions.append({
            "measurement_id": item["measurement_id"],
            **resolution,
        })
    assigned = [resolution["unit"] for resolution in resolutions
                if resolution["unit"] is not None]
    human_unit = assigned[0] if assigned else None
    evidence = scan_action_evidence(event, obs_index, artifact)
    tare_observed = bool(evidence["tare_frames"])
    actions = build_action_events(
        evidence["temporal_actions"], measurements)
    weighing = build_weighing_event(
        event, measurements, tare_observed,
        [action.action_id for action in actions])
    unresolved = build_unresolved(event, tare_observed, human_unit)
    draft_id = f"draft_{video_id}_{event_id}_v{draft_version}"
    draft = build_draft(
        draft_id, draft_version, video_id, out_dir.name, event,
        measurements, actions, unresolved)
    video_source = build_video_source(
        video_id, obs_index, source_dataset)
    issues = validate_bundle(video_source, actions, [weighing])
    out_dir.mkdir(parents=True)
    _write_json(out_dir / "recipe_draft.json",
                draft.model_dump(mode="json"))
    _write_json(out_dir / "weighing_event.json",
                weighing.model_dump(mode="json"))
    _write_json(out_dir / "action_events.json",
                [action.model_dump(mode="json") for action in actions])
    _write_json(out_dir / "measurements.json", {
        "event_id": event_id,
        "window": {
            "start_seconds": (
                start if start is not None
                else event["start_timestamp_seconds"]),
            "end_seconds": (
                end if end is not None
                else event["end_timestamp_seconds"]),
        },
        "measurements": measurements,
        "unit_resolutions": resolutions,
        "action_evidence": evidence,
    })
    _write_json(out_dir / "validation.json", {
        "issue_count": len(issues),
        "issues": [issue.model_dump(mode="json") for issue in issues],
    })
    if units_path is not None and units_path.is_file():
        _write_json(out_dir / "units_applied.json", sidecar)
    return {
        "draft_id": draft_id,
        "measurements": len(measurements),
        "actions": len(actions),
        "unresolved": [question.code for question in unresolved],
        "issues": len(issues),
        "output_dir": str(out_dir),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI with reconstruct / assign-unit subcommands."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("reconstruct", help="build a recipe draft")
    rec.add_argument("--obs-dir", type=Path, required=True)
    rec.add_argument("--temporal", type=Path, required=True)
    rec.add_argument("--event-id", required=True)
    rec.add_argument("--out", type=Path, required=True)
    rec.add_argument("--video-id", default="P04_104")
    rec.add_argument("--executions", type=Path, default=None)
    rec.add_argument("--units", type=Path, default=None)
    rec.add_argument("--start", type=float, default=None)
    rec.add_argument("--end", type=float, default=None)
    rec.add_argument("--draft-version", type=int, default=1)
    rec.add_argument("--source-dataset", default=None)
    rec.add_argument("--run-id", default=None)
    assign = sub.add_parser("assign-unit", help="append a unit revision")
    assign.add_argument("--units-file", type=Path, required=True)
    assign.add_argument("--unit", required=True)
    assign.add_argument("--scope-type", required=True,
                        choices=["event_window", "session"])
    assign.add_argument("--event-id", default=None)
    assign.add_argument("--start", type=float, required=True)
    assign.add_argument("--end", type=float, required=True)
    assign.add_argument("--reviewer", required=True)
    assign.add_argument("--justification", default=None)
    assign.add_argument("--note", default=None)
    args = parser.parse_args(argv)
    try:
        if args.command == "assign-unit":
            sidecar = load_sidecar(args.units_file)
            updated = assign_unit(
                sidecar, args.unit, args.scope_type, args.event_id,
                args.start, args.end, args.reviewer,
                args.justification, args.note,
            )
            save_sidecar(args.units_file, updated)
            latest = updated["assignments"][-1]
            print(json.dumps({
                "assignment_id": latest["assignment_id"],
                "revision": latest["revision"],
                "status": latest["status"],
            }, indent=2, sort_keys=True))
            return 0
        stamp = datetime.datetime.now(datetime.timezone.utc)
        run_id = args.run_id or stamp.strftime(
            f"{args.event_id}_%Y%m%dT%H%M%SZ")
        summary = reconstruct(
            args.obs_dir, args.temporal, args.event_id,
            args.out / run_id, args.video_id, args.executions,
            args.units, args.start, args.end, args.draft_version,
            args.source_dataset,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        state_path = args.out / run_id / "validation.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        return 2 if state["issue_count"] else 0
    except ReconstructionError as exc:
        print(f"RECONSTRUCT FAILED [{exc.code}]: {exc.message}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
