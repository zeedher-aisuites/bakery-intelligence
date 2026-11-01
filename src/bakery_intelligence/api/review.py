"""Human review workspace data access (demo-first Phase 1).

Assembles read-only per-frame review bundles from retained work
files and persists human annotations separately from inference
artifacts. Original predictions are never modified: annotations
live in ``reviews/<run_id>.json`` under the storage root, outside
both the artifacts and work trees, so historical evidence stays
byte-identical.

All helpers take validated identifiers only; frame names must
match ``frame_<8 digits>.jpg`` exactly, which keeps every file
access inside the storage root by construction.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from bakery_intelligence.api.jobs import (
    JobRecord,
    utc_now_iso,
)
from bakery_intelligence.api.storage import (
    LocalStorageBackend,
    ObjectNotFoundError,
    validate_run_id,
)

REVIEW_SCHEMA_VERSION = "review-annotations/1"
REVIEWS_DIRNAME = "reviews"

DECISION_CONFIRMED = "confirmed"
DECISION_CORRECTED = "corrected"
DECISION_REJECTED = "rejected"
DECISION_UNCERTAIN = "uncertain"
DECISIONS = (
    DECISION_CONFIRMED,
    DECISION_CORRECTED,
    DECISION_REJECTED,
    DECISION_UNCERTAIN,
)

FRAME_NAME_PATTERN = re.compile(r"frame_[0-9]{8}\.jpg")

MAX_NOTES_CHARS = 1000
MAX_REVIEWER_CHARS = 64


class ReviewError(ValueError):
    """A review request or stored review record is invalid."""


def validate_frame_name(frame_name: str) -> str:
    """Accept an extracted-frame file name or raise."""
    if not isinstance(frame_name, str):
        raise ReviewError("Frame name must be a string.")
    if FRAME_NAME_PATTERN.fullmatch(frame_name) is None:
        raise ReviewError(f"Invalid frame name: {frame_name!r}.")
    return frame_name


def _reviews_path(root: Path, run_id: str) -> Path:
    validate_run_id(run_id)
    return root / REVIEWS_DIRNAME / f"{run_id}.json"


def read_reviews(root: Path, run_id: str) -> dict[str, Any]:
    """Return stored annotations keyed by frame index string.

    A missing file means no reviews yet, never an error.
    """
    path = _reviews_path(root, run_id)
    if not path.is_file():
        return {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "run_id": run_id,
            "annotations": {},
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReviewError(
            f"Stored reviews for {run_id} are unreadable: {exc}")
    if not isinstance(payload, dict) or not isinstance(
            payload.get("annotations"), dict):
        raise ReviewError(
            f"Stored reviews for {run_id} are malformed.")
    return {
        "schema_version": payload.get(
            "schema_version", REVIEW_SCHEMA_VERSION),
        "run_id": run_id,
        "annotations": payload["annotations"],
    }


def save_review(root: Path, run_id: str, frame_index: int,
               decision: str, value: Any, notes: Any,
               reviewer: Any) -> dict[str, Any]:
    """Validate and upsert one frame annotation, then persist it.

    One annotation exists per frame: saving replaces any prior
    record for that frame, so contradictory states cannot
    accumulate. Only ``corrected`` may carry a numeric value.
    """
    if isinstance(frame_index, bool) or not isinstance(
            frame_index, int) or frame_index < 0:
        raise ReviewError(
            f"frame_index must be a non-negative int, "
            f"got {frame_index!r}.")
    if decision not in DECISIONS:
        raise ReviewError(
            f"decision must be one of {', '.join(DECISIONS)}, "
            f"got {decision!r}.")
    if decision == DECISION_CORRECTED:
        if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ReviewError(
                "A corrected review requires a finite numeric "
                f"value, got {value!r}.")
        value = float(value)
    elif value is not None:
        raise ReviewError(
            f"Only a {DECISION_CORRECTED!r} review may carry a "
            f"value, got {value!r} with {decision!r}.")
    if notes is not None:
        if not isinstance(notes, str):
            raise ReviewError("Review notes must be a string.")
        if len(notes) > MAX_NOTES_CHARS:
            raise ReviewError(
                f"Review notes exceed {MAX_NOTES_CHARS} characters.")
    if reviewer is not None:
        if not isinstance(reviewer, str):
            raise ReviewError("Reviewer must be a string.")
        if len(reviewer) > MAX_REVIEWER_CHARS:
            raise ReviewError(
                f"Reviewer exceeds {MAX_REVIEWER_CHARS} characters.")
    stored = read_reviews(root, run_id)
    annotation = {
        "frame_index": frame_index,
        "decision": decision,
        "value": value,
        "notes": notes,
        "reviewer": reviewer,
        "updated_at": utc_now_iso(),
        "schema_version": REVIEW_SCHEMA_VERSION,
    }
    stored["annotations"][str(frame_index)] = annotation
    path = _reviews_path(root, run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(stored, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    tmp.replace(path)
    return annotation


def read_frame_bytes(root: Path, run_id: str,
                     frame_name: str) -> bytes:
    """Return retained frame pixels for a validated frame name."""
    validate_run_id(run_id)
    validate_frame_name(frame_name)
    path = root / "runs" / run_id / "work" / "frames" / frame_name
    if not path.is_file():
        raise ObjectNotFoundError(f"Frame not found: {frame_name!r}.")
    return path.read_bytes()


def frame_exists(root: Path, run_id: str, frame_index: int) -> bool:
    """Report whether retained observations cover a frame index."""
    validate_run_id(run_id)
    if isinstance(frame_index, bool) or not isinstance(
            frame_index, int):
        return False
    obs_dir = root / "runs" / run_id / "work" / "firstpass" / "observations"
    if not obs_dir.is_dir():
        return False
    for path in obs_dir.glob("*.json"):
        try:
            items = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            if (isinstance(item, dict)
                    and item.get("frame_index") == frame_index):
                return True
    return False


def _load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _observation_entry(item: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    frame_index = item.get("frame_index")
    if (isinstance(frame_index, bool)
            or not isinstance(frame_index, int)):
        return None
    fields = item.get("model_fields")
    if not isinstance(fields, dict):
        return None
    call = item.get("call")
    if not isinstance(call, dict):
        call = {}
    image_ref = item.get("image_ref")
    if not isinstance(image_ref, dict):
        image_ref = {}
    return {
        "frame_index": frame_index,
        "timestamp_seconds": item.get("timestamp_seconds"),
        "image_path": item.get("image_path"),
        "terminal_state": item.get("terminal_state"),
        "parse_error": item.get("parse_error"),
        "response_status": item.get("response_status"),
        "model_fields": {
            name: fields.get(name) for name in (
                "visible_objects",
                "scale_display_verbatim",
                "scale_readable",
                "scale_state",
                "tare_action_visible",
                "addition_or_removal_visible",
                "container_description",
                "container_changed_hint",
                "confidence",
                "ambiguity_notes",
            )
        },
        "request_id": call.get("request_id"),
        "latency_seconds": call.get("latency_seconds"),
        "input_tokens": call.get("input_tokens"),
        "output_tokens": call.get("output_tokens"),
        "reasoning_output_tokens": call.get(
            "reasoning_output_tokens"),
        "recorded_image_sha256": image_ref.get("sha256"),
    }


def build_review_bundle(storage: LocalStorageBackend,
                        record: JobRecord) -> dict[str, Any]:
    """Assemble the workspace payload for one completed run.

    Read-only: work files and artifacts are opened for reading
    only. Missing pieces surface as explicit nulls plus
    ``missing`` notes, never as invented values.
    """
    run_id = record.run_id
    root = storage.base_dir
    work = root / "runs" / run_id / "work"
    missing: list[str] = []
    observations: dict[int, dict[str, Any]] = {}
    obs_dir = work / "firstpass" / "observations"
    if obs_dir.is_dir():
        for path in sorted(obs_dir.glob("*.json")):
            try:
                items = _load_json_file(path)
            except (OSError, ValueError):
                missing.append(
                    f"Unreadable observation file: {path.name}.")
                continue
            if not isinstance(items, list):
                missing.append(
                    f"Malformed observation file: {path.name}.")
                continue
            for item in items:
                entry = _observation_entry(item)
                if entry is not None:
                    observations[entry["frame_index"]] = entry
    else:
        missing.append("No first-pass observations directory.")
    measurements: dict[int, dict[str, Any]] = {}
    try:
        bundle = json.loads(storage.read_artifact(
            run_id, "measurements.json").decode("utf-8"))
        for item in bundle.get("measurements", []):
            if isinstance(item, dict) and isinstance(
                    item.get("frame_index"), int):
                measurements[item["frame_index"]] = {
                    "measurement_id": item.get("measurement_id"),
                    "value": item.get("value"),
                    "value_source": item.get("value_source"),
                    "value_status": item.get("value_status"),
                    "unit_observed": item.get("unit_observed"),
                }
    except Exception:
        missing.append("measurements.json is unavailable.")
    try:
        reviews = read_reviews(root, run_id)["annotations"]
    except ReviewError as exc:
        missing.append(str(exc))
        reviews = {}
    frames: list[dict[str, Any]] = []
    for frame_index in sorted(observations):
        entry = observations[frame_index]
        image_name = entry["image_path"]
        image_available = False
        image_sha256: str | None = None
        if isinstance(image_name, str):
            try:
                validate_frame_name(image_name)
            except ReviewError:
                image_name = None
            else:
                image_path = (work / "frames" / image_name)
                if image_path.is_file():
                    image_available = True
                    digest = hashlib.sha256()
                    with image_path.open("rb") as handle:
                        while True:
                            chunk = handle.read(65536)
                            if not chunk:
                                break
                            digest.update(chunk)
                    image_sha256 = digest.hexdigest()
        recorded = entry["recorded_image_sha256"]
        frames.append({
            "frame_index": frame_index,
            "timestamp_seconds": entry["timestamp_seconds"],
            "image_available": image_available,
            "image_name": image_name,
            "image_sha256": image_sha256,
            "image_matches_record": (
                image_sha256 == recorded
                if image_sha256 and recorded else None),
            "observation": entry,
            "measurement": measurements.get(frame_index),
            "review": reviews.get(str(frame_index)),
        })
    if not frames:
        missing.append("No frame observations available.")
    return {
        "run_id": run_id,
        "workflow": record.workflow,
        "status": record.status,
        "source_object_id": record.source_object_id,
        "source_sha256": record.source_sha256,
        "frames": frames,
        "reviews": reviews,
        "missing": missing,
    }
