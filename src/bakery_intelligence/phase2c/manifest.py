"""Phase 2C production-session manifest (offline, deterministic).

Validates demo media with ffprobe (container truth, not decoder
guesses), records identity hashes, detects variable frame rate,
and declares the clip order with an explicit unknown inter-clip
gap. No inference, no network, no media modification.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

SESSION_ID = "bakery_demo_v1"
CLIP_PREPARATION = "clip_01_preparation"
CLIP_MIXING = "clip_02_mixing"
MANIFEST_VERSION = "phase2c-session-manifest/1"


class ManifestError(ValueError):
    """Session media failed validation."""


@dataclass
class ClipManifest:
    clip_id: str
    order: int
    path: str
    filename: str
    sha256: str
    size_bytes: int
    duration_seconds: float
    width: int
    height: int
    video_codec: str
    avg_frame_rate: float | None
    variable_frame_rate: bool
    audio_codec: str | None
    audio_sample_rate: int | None
    audio_channels: int | None
    publication_authorized: bool = True


@dataclass
class SessionManifest:
    manifest_version: str
    session_id: str
    clips: list[ClipManifest] = field(default_factory=list)
    inter_clip_gap: str = "unknown"
    ground_truth_recipe: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ffprobe_streams(path: Path) -> dict[str, Any]:
    """Return ffprobe format+streams JSON for one file."""
    try:
        completed = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries",
             "format=filename,size,duration:"
             "stream=index,codec_name,codec_type,width,height,"
             "avg_frame_rate,r_frame_rate,sample_rate,channels",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=120,
            check=False)
    except FileNotFoundError as exc:
        raise ManifestError(
            "ffprobe is required for media validation.") from exc
    if completed.returncode != 0:
        raise ManifestError(
            f"ffprobe failed for {path.name}: "
            f"{completed.stderr.strip()[:200]}")
    try:
        return json.loads(completed.stdout)
    except ValueError as exc:
        raise ManifestError(
            f"ffprobe output unparseable for {path.name}.") from exc


def _rate(value: Any) -> float | None:
    if not isinstance(value, str) or "/" not in value:
        return None
    try:
        rate = float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None
    return rate if rate and rate > 0 else None


def describe_clip(clip_id: str, order: int,
                  path: Path) -> ClipManifest:
    """Validate one clip file and describe it (read-only)."""
    if not path.is_file():
        raise ManifestError(f"Clip file missing: {path}.")
    if path.stat().st_size == 0:
        raise ManifestError(f"Clip file is empty: {path.name}.")
    probed = ffprobe_streams(path)
    try:
        duration = float(probed["format"]["duration"])
        size = int(probed["format"]["size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(
            f"Clip {path.name} reports no usable duration.") from exc
    if duration <= 0:
        raise ManifestError(f"Clip {path.name} has no duration.")
    video = next((s for s in probed.get("streams", [])
                  if s.get("codec_type") == "video"), None)
    audio = next((s for s in probed.get("streams", [])
                  if s.get("codec_type") == "audio"), None)
    if video is None:
        raise ManifestError(f"Clip {path.name} has no video stream.")
    avg = _rate(video.get("avg_frame_rate"))
    rframe = _rate(video.get("r_frame_rate"))
    vfr = (avg is not None and rframe is not None
           and abs(avg - rframe) / max(rframe, 1e-9) > 0.01)
    return ClipManifest(
        clip_id=clip_id,
        order=order,
        path=str(path),
        filename=path.name,
        sha256=sha256_file(path),
        size_bytes=size,
        duration_seconds=duration,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        video_codec=str(video.get("codec_name") or "unknown"),
        avg_frame_rate=avg,
        variable_frame_rate=vfr,
        audio_codec=(str(audio.get("codec_name"))
                     if audio else None),
        audio_sample_rate=(int(audio["sample_rate"])
                           if audio and audio.get("sample_rate")
                           else None),
        audio_channels=(int(audio["channels"])
                        if audio and audio.get("channels")
                        else None),
    )


def build_session(clips: list[tuple[str, int, Path]]) -> SessionManifest:
    """Build the ordered session manifest for validated clips."""
    ordered = sorted(clips, key=lambda item: item[1])
    if [order for _, order, _ in ordered] != list(
            range(1, len(ordered) + 1)):
        raise ManifestError(
            "Clip orders must be consecutive from 1.")
    if len({clip for clip, _, _ in ordered}) != len(ordered):
        raise ManifestError("Clip ids must be unique.")
    return SessionManifest(
        manifest_version=MANIFEST_VERSION,
        session_id=SESSION_ID,
        clips=[describe_clip(clip, order, path)
               for clip, order, path in ordered],
    )


def write_manifest(manifest: SessionManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
        + "\n", encoding="utf-8")
