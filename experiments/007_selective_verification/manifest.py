"""Deterministic review-window selection (Exp. 007).

Reads a temporal analysis artifact plus the underlying frame
observations and emits a selection manifest for four fixed review
windows. Offline, stdlib-only, no model calls. Event and observation
references are computed from the inputs; nothing is invented.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "006_p04_multimodal_inference"))

import temporal  # noqa: E402

MANIFEST_VERSION = "1.0.0"

WINDOWS: list[dict[str, Any]] = [
    {
        "window_id": "V01",
        "start_seconds": 142.0,
        "end_seconds": 153.0,
        "purpose": 'Ambiguous "5 15" reading and increasing values',
        "rationale": (
            "The t=142-153 visibility interval carries ambiguous "
            "parsing ('5 15' at frame 7500) together with an abrupt "
            "step; review establishes the true display text and "
            "whether the rise across the window is genuine."
        ),
    },
    {
        "window_id": "V02",
        "start_seconds": 368.0,
        "end_seconds": 378.0,
        "purpose": "Stable 1756 reading and subsequent zero",
        "rationale": (
            "The t=371-375 stable run of 1756 is needs_review with "
            "abrupt borders (1358->1756->0); review checks whether "
            "the plateau and the drop to zero are visually real."
        ),
    },
    {
        "window_id": "V03",
        "start_seconds": 396.0,
        "end_seconds": 420.0,
        "purpose": "Stable 238 and 315 readings with abrupt changes",
        "rationale": (
            "Contrasts an escalated stable run (238, abrupt border) "
            "with a locally consistent one (315, flagged enclosing "
            "interval) amid nearby addition candidates."
        ),
    },
    {
        "window_id": "V04",
        "start_seconds": 495.0,
        "end_seconds": 512.0,
        "purpose": "Highly variable readings and possible button interactions",
        "rationale": (
            "Thirteen abrupt steps plus a spike and a long tare run "
            "overlap here; review checks displays against possible "
            "button-press artifacts and obscured frames."
        ),
    },
]

MODEL_FIELD_KEYS = (
    "scale_display_verbatim",
    "scale_readable",
    "scale_state",
    "tare_action_visible",
    "addition_or_removal_visible",
    "confidence",
    "ambiguity_notes",
)


class ManifestError(Exception):
    """Structured selection failure with a machine-readable code."""

    def __init__(self, code: str, message: str, entity_id: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.entity_id = entity_id


def load_artifact(path: Path) -> dict[str, Any]:
    """Load and minimally validate a temporal artifact."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(
            "ARTIFACT_INVALID", f"Unreadable artifact: {path}", str(path)
        ) from exc
    if not isinstance(payload, dict):
        raise ManifestError(
            "ARTIFACT_INVALID", "Artifact root is not an object.",
            str(path),
        )
    for key in ("artifact_version", "events"):
        if key not in payload:
            raise ManifestError(
                "ARTIFACT_INVALID", f"Artifact lacks key: {key}.",
                str(path),
            )
    if not isinstance(payload["events"], list):
        raise ManifestError(
            "ARTIFACT_INVALID", "Artifact events are not a list.",
            str(path),
        )
    for position, event in enumerate(payload["events"]):
        for key in ("event_id", "candidate_type",
                    "start_timestamp_seconds", "end_timestamp_seconds",
                    "verification_status"):
            if not isinstance(event, dict) or key not in event:
                raise ManifestError(
                    "ARTIFACT_INVALID",
                    f"Event #{position} lacks key: {key}.",
                    str(position),
                )
    return payload


def load_frame_map(manifest_csv: Path) -> dict[int, str]:
    """Map frame_index to image filename from the extraction manifest."""
    try:
        text = manifest_csv.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(
            "MANIFEST_UNREADABLE", f"Unreadable manifest: {manifest_csv}",
            str(manifest_csv),
        ) from exc
    try:
        rows = list(csv.DictReader(text.splitlines()))
    except csv.Error as exc:
        raise ManifestError(
            "MANIFEST_UNREADABLE", f"Unparseable manifest: {manifest_csv}",
            str(manifest_csv),
        ) from exc
    if not rows or "frame_index" not in rows[0] or "image_path" not in rows[0]:
        raise ManifestError(
            "MANIFEST_UNREADABLE",
            "Manifest lacks frame_index/image_path columns.",
            str(manifest_csv),
        )
    mapping = {}
    for row in rows:
        try:
            mapping[int(row["frame_index"])] = row["image_path"]
        except (TypeError, ValueError) as exc:
            raise ManifestError(
                "MANIFEST_UNREADABLE",
                f"Bad frame_index value: {row.get('frame_index')!r}.",
                str(manifest_csv),
            ) from exc
    return mapping


def overlaps(
    event_start: float, event_end: float,
    window_start: float, window_end: float,
) -> bool:
    """True when an event touches a window (partial overlap counts)."""
    return event_start <= window_end and event_end >= window_start


def build_window(
    spec: dict[str, Any],
    events: list[dict[str, Any]],
    records: list[dict[str, Any]],
    frame_map: dict[int, str] | None,
    frames_dir: Path | None,
) -> dict[str, Any]:
    """Assemble one window entry with real computed references."""
    start = spec["start_seconds"]
    end = spec["end_seconds"]
    selected = [
        event for event in events
        if overlaps(
            event["start_timestamp_seconds"],
            event["end_timestamp_seconds"], start, end,
        )
    ]
    selected.sort(key=lambda event: (
        event["start_timestamp_seconds"], event["event_id"]
    ))
    event_refs = [{
        "event_id": event["event_id"],
        "candidate_type": event["candidate_type"],
        "verification_status": event["verification_status"],
        "fully_inside": (
            event["start_timestamp_seconds"] >= start
            and event["end_timestamp_seconds"] <= end
        ),
    } for event in selected]
    window_records = [
        item for item in records
        if start <= item["timestamp_seconds"] <= end
    ]
    window_records.sort(key=lambda item: (
        item["timestamp_seconds"], item["frame_index"]
    ))
    observations = []
    for item in window_records:
        frame_index = item["frame_index"]
        image_name = item.get("image_path")
        frame_path: str | None = None
        accessible = False
        if frame_map is not None and frames_dir is not None:
            mapped = frame_map.get(frame_index, image_name)
            if mapped:
                candidate = frames_dir / mapped
                frame_path = str(candidate)
                accessible = candidate.is_file()
        observations.append({
            "frame_index": frame_index,
            "timestamp_seconds": item["timestamp_seconds"],
            "batch_id": item["batch_id"],
            "image_path": image_name,
            "frame_path": frame_path,
            "frame_accessible": accessible,
            "model_fields": {
                key: item["model_fields"].get(key)
                for key in MODEL_FIELD_KEYS
            },
        })
    return {
        "window_id": spec["window_id"],
        "start_seconds": start,
        "end_seconds": end,
        "purpose": spec["purpose"],
        "rationale": spec["rationale"],
        "event_refs": event_refs,
        "observations": observations,
        "clip_range": {
            "start_frame_index": (
                window_records[0]["frame_index"] if window_records
                else None
            ),
            "end_frame_index": (
                window_records[-1]["frame_index"] if window_records
                else None
            ),
        },
        "review_status": "pending",
    }


def generate(
    artifact_path: Path,
    obs_dir: Path,
    extraction_manifest: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Generate the selection manifest and write it deterministically."""
    artifact = load_artifact(artifact_path)
    try:
        records = temporal.load_observations(obs_dir)
    except temporal.TemporalError as exc:
        raise ManifestError(
            "OBS_INVALID", exc.message, exc.entity_id
        ) from exc
    try:
        raw_artifact = artifact_path.read_bytes()
    except OSError as exc:
        raise ManifestError(
            "ARTIFACT_INVALID", f"Unreadable artifact: {artifact_path}",
            str(artifact_path),
        ) from exc
    missing: list[str] = []
    frame_map: dict[int, str] | None = None
    frames_dir: Path | None = extraction_manifest.parent
    if extraction_manifest.is_file():
        frame_map = load_frame_map(extraction_manifest)
    else:
        missing.append(f"extraction manifest not found: {extraction_manifest}")
        frames_dir = None
    windows = [
        build_window(spec, artifact["events"], records, frame_map,
                     frames_dir)
        for spec in WINDOWS
    ]
    absent = sum(
        1 for window in windows for obs in window["observations"]
        if not obs["frame_accessible"]
    )
    total_obs = sum(len(window["observations"]) for window in windows)
    if frame_map is not None and absent:
        missing.append(f"{absent} referenced frame file(s) not found")
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "source": {
            "temporal_artifact": str(artifact_path),
            "artifact_sha256": hashlib.sha256(raw_artifact).hexdigest(),
            "artifact_version": artifact["artifact_version"],
            "observations_dir": str(obs_dir),
            "observation_count": len(records),
            "extraction_manifest": (
                str(extraction_manifest) if frame_map is not None
                else None
            ),
            "frames_accessible": frame_map is not None and absent == 0,
            "missing": missing,
        },
        "windows": windows,
        "summary": {
            "window_count": len(windows),
            "event_ref_count": sum(
                len(window["event_refs"]) for window in windows
            ),
            "observation_count": total_obs,
            "frames_accessible": total_obs - absent,
            "frames_missing": absent,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Minimal CLI: manifest ARTIFACT OBS_DIR MANIFEST_CSV OUTPUT."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("obs_dir", type=Path)
    parser.add_argument("extraction_manifest", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = generate(
            args.artifact, args.obs_dir, args.extraction_manifest,
            args.output_json,
        )
    except ManifestError as exc:
        print(f"SELECTION FAILED [{exc.code}]: {exc.message}")
        return 1
    print(json.dumps(manifest["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
