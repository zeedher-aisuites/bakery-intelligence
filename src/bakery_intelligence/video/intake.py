"""Local-first video intake and staging (M2-06C.3).

Streams an arbitrary supported video into content-addressed
staging with incremental SHA-256, size caps, a best-effort disk
check, read-back probing, and a provenance sidecar. No frame
is decoded here beyond the container probe; extraction lives in
``extract.py``. No network, no inference, no cloud.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

STAGING_MARKER = ".bakery-staging"
SIDECAR_NAME = "video_sidecar.json"

# Upper bound for one staged video (4 GiB; P04-class media is ~2 GiB).
DEFAULT_MAX_VIDEO_BYTES = 4 * 1024 * 1024 * 1024

SUPPORTED_SUFFIXES = frozenset({
    ".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg",
    ".webm",
})

_CHUNK_SIZE = 1024 * 1024


class VideoError(Exception):
    """Structured intake/extraction failure with a code."""

    def __init__(self, code: str, message: str,
                 entity_id: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.entity_id = entity_id


@dataclass(frozen=True)
class VideoMetadata:
    """Probed container properties plus integrity anchor."""

    file_name: str
    sha256: str
    bytes: int
    fps: float
    frame_count: int
    duration_seconds: float
    width: int
    height: int

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable sidecar mapping."""
        return {
            "file_name": self.file_name,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_seconds": self.duration_seconds,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class StagedVideo:
    """A staged video with its metadata and sidecar location."""

    video_path: Path
    staging_dir: Path
    metadata: VideoMetadata
    sidecar_path: Path


def sanitize_stem(name: str) -> str:
    """Derive a filesystem-safe stem from a client filename."""
    stem = (name or "video").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = stem.strip() or "video"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    cleaned = re.sub(r"\.{2,}", "_", cleaned).strip("._-")
    return cleaned[:48].rstrip("._-") or "video"


def probe_video(path: Path) -> dict[str, Any]:
    """Read container properties without decoding frames.

    Raises VideoError when the container cannot be opened or
    reports unusable properties.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoError(
            "UNREADABLE_VIDEO",
            f"Cannot open video container: {path.name}.",
            path.name,
        )
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0 or frame_count <= 0:
        raise VideoError(
            "UNREADABLE_VIDEO",
            f"Video reports unusable fps/frame count: {path.name}.",
            path.name,
        )
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps,
        "width": width,
        "height": height,
    }


def _check_disk_space(directory: Path, needed_bytes: int) -> None:
    """Best-effort free-space guard (checked, then trusted)."""
    try:
        free = shutil.disk_usage(directory).free
    except OSError:
        return
    if free < needed_bytes:
        raise VideoError(
            "INSUFFICIENT_DISK",
            f"Need {needed_bytes} free bytes, {free} available.",
            str(directory),
        )


def stage_video(source: Path,
                staging_root: Path,
                max_bytes: int = DEFAULT_MAX_VIDEO_BYTES,
                ) -> StagedVideo:
    """Copy a video into content-addressed staging (streamed).

    The file streams once through SHA-256 into a temp name, then
    renames to ``video_<sha16>_<stem><suffix>``. An identical
    previous staging is reused after hash verification; anything
    else already present fails instead of overwriting. A sidecar
    records metadata; only the basename (never host paths) is
    persisted.
    """
    if not source.is_file():
        raise VideoError(
            "VIDEO_NOT_FOUND", f"Video does not exist: {source}.",
            source.name,
        )
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise VideoError(
            "UNSUPPORTED_CONTAINER",
            f"Unsupported container {source.suffix!r}; expected one "
            f"of {sorted(SUPPORTED_SUFFIXES)}.",
            source.name,
        )
    size = source.stat().st_size
    if size > max_bytes:
        raise VideoError(
            "VIDEO_TOO_LARGE",
            f"Video is {size} bytes; limit is {max_bytes}.",
            source.name,
        )
    staging_root.mkdir(parents=True, exist_ok=True)
    _check_disk_space(staging_root, size)
    digest = hashlib.sha256()
    tmp_path = staging_root / f".staging-{source.name}.part"
    with source.open("rb") as origin, tmp_path.open("wb") as dest:
        while True:
            chunk = origin.read(_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            dest.write(chunk)
    sha = digest.hexdigest()
    stem = sanitize_stem(source.stem)
    final_name = f"video_{sha[:16]}_{stem}{suffix}"
    final_path = staging_root / final_name
    created_here = not final_path.exists()
    if not created_here:
        existing = hashlib.sha256(
            final_path.read_bytes()).hexdigest()
        tmp_path.unlink(missing_ok=True)
        if existing != sha:
            raise VideoError(
                "STAGING_MISMATCH",
                f"Staging slot collision with different content: "
                f"{final_name}.",
                final_name,
            )
    else:
        tmp_path.replace(final_path)
    try:
        props = probe_video(final_path)
    except VideoError:
        if created_here:
            final_path.unlink(missing_ok=True)
        raise
    metadata = VideoMetadata(
        file_name=source.name, sha256=sha, bytes=size,
        fps=props["fps"], frame_count=props["frame_count"],
        duration_seconds=props["duration_seconds"],
        width=props["width"], height=props["height"],
    )
    sidecar_path = staging_root / f"{final_path.stem}.json"
    if not sidecar_path.exists():
        sidecar_path.write_text(
            json.dumps({
                **metadata.to_dict(),
                "staged_at_utc": datetime.now(timezone.utc
                                             ).isoformat(),
                "staging_name": final_name,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n",
        )
    (staging_root / STAGING_MARKER).touch(exist_ok=True)
    return StagedVideo(
        video_path=final_path, staging_dir=staging_root,
        metadata=metadata, sidecar_path=sidecar_path,
    )


def purge_staging(staging_root: Path) -> int:
    """Delete a staging directory previously created here.

    Refuses paths without the staging marker, so evidence
    outside managed staging can never be removed by mistake.
    Returns the number of deleted files. Nothing is deleted
    automatically anywhere else in this package.
    """
    if not staging_root.is_dir():
        raise VideoError(
            "NOT_A_STAGING_DIR",
            f"Not a directory: {staging_root}.",
            staging_root.name,
        )
    if not (staging_root / STAGING_MARKER).is_file():
        raise VideoError(
            "NOT_A_STAGING_DIR",
            f"Refusing to purge unmanaged directory: {staging_root}.",
            staging_root.name,
        )
    removed = 0
    for path in sorted(staging_root.rglob("*"), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()
            removed += 1
    for path in sorted(staging_root.rglob("*"), reverse=True):
        if path.is_dir():
            path.rmdir()
    staging_root.rmdir()
    return removed
