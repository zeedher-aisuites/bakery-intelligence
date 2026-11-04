"""Phase 2C local media preparation (offline, deterministic).

FFmpeg helpers that never touch originals: WAV extraction for
transcription, time-based sub-clip cutting, and PTS-based frame
grabs. All seeking uses container timestamps, never frame-index
arithmetic, so variable-frame-rate sources stay correct.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


class MediaError(RuntimeError):
    """Local media preparation failed (no billing involved)."""


def run_ffmpeg(args: list[str], timeout: int = 600) -> None:
    try:
        completed = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error",
             "-y", *args],
            capture_output=True, text=True, timeout=timeout,
            check=False)
    except FileNotFoundError as exc:
        raise MediaError("ffmpeg is required.") from exc
    if completed.returncode != 0:
        raise MediaError(
            f"ffmpeg failed: {completed.stderr.strip()[:300]}")


def extract_wav(source: Path, dest: Path) -> Path:
    """Extract mono 16kHz WAV for transcription (new file only)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-i", str(source), "-vn", "-ac", "1", "-ar",
                "16000", "-c:a", "pcm_s16le", str(dest)])
    if not dest.is_file() or dest.stat().st_size == 0:
        raise MediaError(f"WAV extraction produced nothing: {dest}.")
    return dest


def cut_subclip(source: Path, start: float, duration: float,
                dest: Path) -> Path:
    """Cut a time-based sub-clip with re-encoded timestamps."""
    if duration <= 0:
        raise MediaError("Sub-clip duration must be positive.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-ss", f"{start:.3f}", "-i", str(source),
                "-t", f"{duration:.3f}", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "23", "-c:a",
                "aac", "-movflags", "+faststart", str(dest)])
    if not dest.is_file() or dest.stat().st_size == 0:
        raise MediaError(f"Sub-clip extraction failed: {dest}.")
    return dest


def split_wav(source: Path, start: float, duration: float,
              dest: Path) -> Path:
    """Cut one WAV segment with re-encoded timestamps."""
    if duration <= 0:
        raise MediaError("WAV segment duration must be positive.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-ss", f"{start:.3f}", "-i", str(source),
                "-t", f"{duration:.3f}", "-c:a", "pcm_s16le",
                str(dest)])
    if not dest.is_file() or dest.stat().st_size == 0:
        raise MediaError(f"WAV segment extraction failed: {dest}.")
    return dest


def grab_frame(source: Path, at_seconds: float,
               dest: Path) -> Path:
    """Grab one frame at a container timestamp (VFR-safe)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-ss", f"{at_seconds:.3f}", "-i", str(source),
                "-frames:v", "1", "-q:v", "3", str(dest)])
    if not dest.is_file() or dest.stat().st_size == 0:
        raise MediaError(f"Frame grab failed: {dest}.")
    return dest


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
