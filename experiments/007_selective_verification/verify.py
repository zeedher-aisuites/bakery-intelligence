"""Selective verification planner (Exp. 007/010 integration).

Consumes first-pass temporal events plus a selection manifest and
emits a bounded verification plan: which needs_review events to
verify, which existing frames to reuse, and the request/cost
envelope. Planning only: no model calls, no frame extraction, no
paid operations. Original observations are never modified; any
future verified observations must be stored separately.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import manifest as selection_manifest  # noqa: E402

PLAN_VERSION = "1.0.0"
PLAN_BUDGETS = (1024, 2048, 3072)
MAX_ATTEMPTS_PER_EVENT = len(PLAN_BUDGETS)


class VerificationError(Exception):
    """Structured planning failure with a machine-readable code."""

    def __init__(self, code: str, message: str, entity_id: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.entity_id = entity_id


def load_selection_manifest(path: Path) -> dict[str, Any]:
    """Load a selection manifest with structural validation."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(
            "MANIFEST_INVALID", f"Unreadable manifest: {path}.",
            str(path),
        ) from exc
    if not isinstance(payload, dict) or "windows" not in payload:
        raise VerificationError(
            "MANIFEST_INVALID", "Selection manifest lacks windows.",
            str(path),
        )
    return payload


def plan(manifest: dict[str, Any], artifact: dict[str, Any],
         frame_map: dict[int, str] | None,
         frames_dir: Path | None) -> dict[str, Any]:
    """Build one plan entry per needs_review event in scope."""
    windows = {entry["window_id"]: entry
               for entry in manifest["windows"]}
    entries = []
    for event in artifact.get("events", []):
        if event.get("verification_status") != "needs_review":
            continue
        covering = [
            window_id for window_id, entry in windows.items()
            if event["start_timestamp_seconds"]
            <= entry["end_seconds"]
            and event["end_timestamp_seconds"]
            >= entry["start_seconds"]
        ]
        if not covering:
            continue
        reuse = []
        for index in event.get("frame_indices", []):
            available = (
                frame_map is not None and frames_dir is not None
                and index in frame_map
                and (frames_dir / frame_map[index]).is_file()
            )
            reuse.append({"frame_index": index,
                          "frame_available": available})
        entries.append({
            "window_ids": sorted(covering),
            "event_id": event["event_id"],
            "candidate_type": event.get("candidate_type"),
            "review_reasons": list(event.get("review_reasons", [])),
            "uncertainty_notes": list(
                event.get("uncertainty_notes", [])),
            "start_timestamp_seconds": event[
                "start_timestamp_seconds"],
            "end_timestamp_seconds": event[
                "end_timestamp_seconds"],
            "reuse_frames": reuse,
            "extraction_proposed": [],
            "extraction_rationale": (
                "Reuse only: additional extraction requires explicit "
                "human approval per event; never automatic."
            ),
            "token_budgets": list(PLAN_BUDGETS),
            "max_requests": MAX_ATTEMPTS_PER_EVENT,
            "cost_estimate_usd": None,
            "cost_note": (
                "Unknown until pricing is configured and per-request "
                "token usage is benchmarked for verification prompts."
            ),
            "separation_rule": (
                "Verified observations must be stored separately from "
                "first-pass observations; disagreement preserved, "
                "originals never overwritten."
            ),
        })
    entries.sort(key=lambda entry: entry["event_id"])
    reuse_total = sum(len(entry["reuse_frames"]) for entry in entries)
    available_total = sum(
        1 for entry in entries for frame in entry["reuse_frames"]
        if frame["frame_available"]
    )
    return {
        "plan_version": PLAN_VERSION,
        "windows_in_scope": sorted(windows),
        "entries": entries,
        "summary": {
            "event_count": len(entries),
            "reuse_frame_count": reuse_total,
            "reuse_frames_available": available_total,
            "extraction_proposed_count": 0,
            "max_total_requests": len(entries) * MAX_ATTEMPTS_PER_EVENT,
        },
    }


def write_plan(selection_path: Path, artifact_path: Path,
               extraction_manifest: Path | None,
               output_path: Path) -> dict[str, Any]:
    """Load inputs, plan, and write the plan deterministically."""
    manifest = load_selection_manifest(selection_path)
    try:
        artifact = selection_manifest.load_artifact(artifact_path)
    except selection_manifest.ManifestError as exc:
        raise VerificationError(
            "ARTIFACT_INVALID", exc.message, exc.entity_id
        ) from exc
    frame_map: dict[int, str] | None = None
    frames_dir: Path | None = None
    if extraction_manifest is not None:
        if not extraction_manifest.is_file():
            raise VerificationError(
                "MANIFEST_UNREADABLE",
                f"Extraction manifest not found: {extraction_manifest}.",
            )
        try:
            frame_map = selection_manifest.load_frame_map(
                extraction_manifest)
        except selection_manifest.ManifestError as exc:
            raise VerificationError(
                "MANIFEST_UNREADABLE", exc.message, exc.entity_id
            ) from exc
        frames_dir = extraction_manifest.parent
    bundle = plan(manifest, artifact, frame_map, frames_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    return bundle


def main(argv: list[str] | None = None) -> int:
    """Minimal CLI: verify SELECTION ARTIFACT OUTPUT."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--extraction-manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        bundle = write_plan(
            args.selection, args.artifact,
            args.extraction_manifest, args.output_json,
        )
    except VerificationError as exc:
        print(f"VERIFY FAILED [{exc.code}]: {exc.message}")
        return 1
    print(json.dumps(bundle["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
