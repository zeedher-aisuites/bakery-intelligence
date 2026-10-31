"""Deterministic frame extraction for arbitrary videos (M2-06C.3).

Mirrors the Exp. 003 sampling semantics (half-open window,
index-derived timestamps, JPEG frames, manifest.csv) without
any fixed video length, frame count, or experiment paths.
Outputs convert directly to C.2 ``FrameRef``/``SourceRef``
inputs; extraction itself runs no inference.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from bakery_intelligence.inference.firstpass import (
    FrameRef,
    SourceRef,
)
from bakery_intelligence.video.intake import (
    VideoError,
    probe_video,
)

MANIFEST_FIELDS = (
    "image_path",
    "frame_index",
    "timestamp_seconds",
    "source_timestamp_seconds",
    "requested_timestamp_seconds",
)
MANIFEST_NAME = "manifest.csv"
REPORT_NAME = "extraction_report.json"

# Upper bound for one extraction job; arbitrary videos must not
# explode into unbounded frame counts silently.
DEFAULT_MAX_FRAMES = 100_000

_DURATION_TOLERANCE = 1e-7
_WINDOW_EPSILON = 1e-9


@dataclass(frozen=True)
class ExtractionConfig:
    """Complete extraction request (explicit, no hidden paths)."""

    video_path: Path
    out_dir: Path
    sample_fps: float = 1.0
    start: float = 0.0
    end: float | None = None
    source_offset: float = 0.0
    max_frames: int = DEFAULT_MAX_FRAMES


@dataclass(frozen=True)
class ExtractionResult:
    """Extracted frames plus their C.2-compatible references."""

    frames: list[FrameRef]
    source: SourceRef
    metadata: dict[str, Any]
    manifest_path: Path
    report_path: Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def extract_frames(config: ExtractionConfig) -> ExtractionResult:
    """Extract sampled frames; return refs plus a source anchor.

    The sampling window is half-open: [start, end). Requested
    times map to source indices via ``int(t * fps + 0.5)`` and
    reported timestamps derive from the chosen index
    (``index / fps``), never from container presentation
    timestamps. Raises VideoError on invalid windows, decode or
    write failures, duplicate schedules, and zero-frame results.
    Refuses to write into a directory that already holds a
    manifest.
    """
    if not config.video_path.is_file():
        raise VideoError(
            "VIDEO_NOT_FOUND",
            f"Video does not exist: {config.video_path}.",
            config.video_path.name,
        )
    if config.sample_fps <= 0:
        raise VideoError(
            "INVALID_FPS", "Sample FPS must be positive.")
    if config.source_offset < 0:
        raise VideoError(
            "INVALID_WINDOW", "Source offset cannot be negative.")
    if config.max_frames < 1:
        raise VideoError(
            "INVALID_CONFIG", "max_frames must be >= 1.")
    props = probe_video(config.video_path)
    fps = props["fps"]
    frame_count = props["frame_count"]
    duration = props["duration_seconds"]
    if config.sample_fps > fps:
        raise VideoError(
            "INVALID_FPS",
            f"Sample FPS {config.sample_fps} exceeds source "
            f"FPS {fps}.")
    end = duration if config.end is None else config.end
    if config.start < 0 or end <= config.start:
        raise VideoError(
            "INVALID_WINDOW", "Expected 0 <= start < end.")
    if end > duration + _DURATION_TOLERANCE:
        raise VideoError(
            "INVALID_WINDOW",
            f"End time {end} exceeds video duration {duration}.")
    out_dir = config.out_dir
    if (out_dir / MANIFEST_NAME).exists():
        raise VideoError(
            "OUTPUT_EXISTS",
            f"Refusing to overwrite extraction: {out_dir}.",
            out_dir.name,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(config.video_path))
    if not capture.isOpened():
        raise VideoError(
            "UNREADABLE_VIDEO",
            f"Cannot open video: {config.video_path.name}.",
            config.video_path.name,
        )
    rows: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    sample_number = 0
    try:
        while (config.start + sample_number / config.sample_fps
               < end - _WINDOW_EPSILON):
            if len(rows) >= config.max_frames:
                raise VideoError(
                    "TOO_MANY_FRAMES",
                    f"Extraction exceeds max_frames "
                    f"{config.max_frames}; narrow the window or rate.",
                )
            requested_time = (
                config.start + sample_number / config.sample_fps)
            frame_index = int(requested_time * fps + 0.5)
            if frame_index >= frame_count:
                break
            if frame_index in used_indices:
                raise VideoError(
                    "DUPLICATE_SCHEDULE",
                    f"Sampling schedule repeats index {frame_index}.",
                    str(frame_index),
                )
            used_indices.add(frame_index)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise VideoError(
                    "DECODE_FAILED",
                    f"Cannot decode frame {frame_index}.",
                    str(frame_index),
                )
            image_name = f"frame_{frame_index:08d}.jpg"
            if not cv2.imwrite(str(out_dir / image_name), frame):
                raise VideoError(
                    "WRITE_FAILED", f"Cannot write image: {image_name}.",
                    image_name,
                )
            actual_time = frame_index / fps
            rows.append({
                "image_path": image_name,
                "frame_index": frame_index,
                "timestamp_seconds": round(actual_time, 6),
                "source_timestamp_seconds": round(
                    actual_time + config.source_offset, 6),
                "requested_timestamp_seconds": round(
                    requested_time, 6),
            })
            sample_number += 1
    finally:
        capture.release()
    if not rows:
        raise VideoError(
            "ZERO_FRAMES",
            f"Window [{config.start}, {end}) at "
            f"{config.sample_fps} FPS selects no frames.",
        )
    manifest_path = out_dir / MANIFEST_NAME
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    video_sha = _sha256_file(config.video_path)
    source = SourceRef(
        name=(f"{config.video_path.stem}"
              f"@{config.sample_fps}fps"),
        sha256=video_sha,
    )
    frames = [
        FrameRef(
            frame_index=row["frame_index"],
            image_path=out_dir / row["image_path"],
            timestamp_seconds=row["timestamp_seconds"],
            source_timestamp_seconds=row[
                "source_timestamp_seconds"],
            requested_timestamp_seconds=row[
                "requested_timestamp_seconds"],
        )
        for row in rows
    ]
    report = {
        "video_name": config.video_path.name,
        "video_sha256": video_sha,
        "video_fps": fps,
        "video_frame_count": frame_count,
        "video_duration_seconds": duration,
        "sample_fps": config.sample_fps,
        "window": {"start": config.start, "end": end},
        "source_offset": config.source_offset,
        "frame_count": len(rows),
        "frame_indices": [row["frame_index"] for row in rows],
        "source_ref": {"name": source.name,
                       "sha256": source.sha256},
    }
    report_path = out_dir / REPORT_NAME
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    return ExtractionResult(
        frames=frames, source=source,
        metadata={"video": props, "report": report},
        manifest_path=manifest_path, report_path=report_path,
    )
