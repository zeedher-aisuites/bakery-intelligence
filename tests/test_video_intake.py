"""Offline tests for video intake and extraction (M2-06C.3).

Synthetic MJPG videos are generated locally with OpenCV; no
licensed media, no network, no inference. Covers staging
integrity, sampling semantics, failure cases, and an end-to-end
pass into the C.2 first-pass runner with a fake client.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from bakery_intelligence.inference.firstpass import (
    FirstPassConfig,
    FirstPassRunner,
    InferenceResponse,
    InferenceUsage,
    SourceRef,
)
from bakery_intelligence.video.extract import (
    ExtractionConfig,
    extract_frames,
)
from bakery_intelligence.video.intake import (
    VideoError,
    probe_video,
    purge_staging,
    stage_video,
)

VALID_JSON = json.dumps({
    "visible_objects": ["scale"], "scale_display_verbatim": "120",
    "scale_readable": True, "scale_state": "stable",
    "tare_action_visible": False,
    "addition_or_removal_visible": "none",
    "container_description": "bowl", "container_changed_hint": False,
    "confidence": "high", "ambiguity_notes": None,
})


def _make_video(path: Path, fps: float, frame_count: int,
                size: tuple[int, int] = (64, 48)) -> Path:
    """Write a synthetic MJPG video; skip when encoding fails."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if not writer.isOpened():
        pytest.skip("MJPG video encoding is unavailable.")
    width, height = size
    for index in range(frame_count):
        shade = (index * 7) % 256
        writer.write(np.full((height, width, 3), shade,
                             dtype=np.uint8))
    writer.release()
    return path


def _ok_client(text: str = VALID_JSON):
    class _OkClient:
        def __init__(self) -> None:
            self.calls = 0

        def submit(self, prompt_text: str, image_paths: Any,
                   api_key: str,
                   max_output_tokens: int) -> InferenceResponse:
            self.calls += 1
            return InferenceResponse(
                model="fake-model", status="completed",
                output_text=text,
                usage=InferenceUsage(10, 20, 30),
                latency_seconds=0.0, request_id=f"req-{self.calls}",
                incomplete_reason=None,
            )

    return _OkClient()


def _read_manifest(out_dir: Path) -> list[dict[str, str]]:
    with (out_dir / "manifest.csv").open(
            newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


# Staging --------------------------------------------------------------


def test_stage_video_streams_hashes_and_probes(
        tmp_path: Path) -> None:
    source = _make_video(tmp_path / "clip.avi", 10.0, 30)
    expected_sha = hashlib.sha256(
        source.read_bytes()).hexdigest()
    staged = stage_video(source, tmp_path / "staging")
    assert staged.metadata.sha256 == expected_sha
    assert staged.metadata.bytes == source.stat().st_size
    assert staged.metadata.fps == pytest.approx(10.0)
    assert staged.metadata.frame_count == 30
    assert staged.metadata.duration_seconds == pytest.approx(3.0)
    assert staged.metadata.width == 64
    assert staged.metadata.height == 48
    assert staged.video_path.read_bytes() == source.read_bytes()
    assert staged.video_path.name.startswith(
        f"video_{expected_sha[:16]}_")
    sidecar = json.loads(staged.sidecar_path.read_text(
        encoding="utf-8"))
    assert sidecar["sha256"] == expected_sha
    assert sidecar["file_name"] == "clip.avi"
    assert "staged_at_utc" in sidecar
    assert (tmp_path / "staging" / ".bakery-staging").is_file()


def test_stage_video_reuses_identical_staging(
        tmp_path: Path) -> None:
    source = _make_video(tmp_path / "clip.avi", 10.0, 30)
    first = stage_video(source, tmp_path / "staging")
    second = stage_video(source, tmp_path / "staging")
    assert first.video_path == second.video_path
    assert first.metadata.sha256 == second.metadata.sha256


def test_stage_video_rejects_missing_unsupported_and_large(
        tmp_path: Path) -> None:
    with pytest.raises(VideoError) as caught:
        stage_video(tmp_path / "absent.mp4", tmp_path / "staging")
    assert caught.value.code == "VIDEO_NOT_FOUND"
    notes = tmp_path / "notes.txt"
    notes.write_text("not a video", encoding="utf-8")
    with pytest.raises(VideoError) as caught:
        stage_video(notes, tmp_path / "staging")
    assert caught.value.code == "UNSUPPORTED_CONTAINER"
    source = _make_video(tmp_path / "clip.avi", 10.0, 30)
    with pytest.raises(VideoError) as caught:
        stage_video(source, tmp_path / "staging", max_bytes=64)
    assert caught.value.code == "VIDEO_TOO_LARGE"


def test_stage_video_rejects_unreadable_container(
        tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.mp4"
    bogus.write_bytes(b"\x00" * 1024)
    with pytest.raises(VideoError) as caught:
        stage_video(bogus, tmp_path / "staging")
    assert caught.value.code == "UNREADABLE_VIDEO"
    leftovers = list((tmp_path / "staging").glob("video_*"))
    assert leftovers == []


def test_stage_video_sanitizes_filename(tmp_path: Path) -> None:
    source = _make_video(tmp_path / "my video (1).avi", 10.0, 5)
    staged = stage_video(source, tmp_path / "staging")
    assert " " not in staged.video_path.name
    assert "(" not in staged.video_path.name


def test_purge_staging_removes_managed_dir(tmp_path: Path) -> None:
    source = _make_video(tmp_path / "clip.avi", 10.0, 5)
    staged = stage_video(source, tmp_path / "staging")
    removed = purge_staging(staged.staging_dir)
    assert removed >= 3
    assert not staged.staging_dir.exists()


def test_purge_refuses_unmanaged_directory(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "keep.txt").write_text("evidence", encoding="utf-8")
    with pytest.raises(VideoError) as caught:
        purge_staging(plain)
    assert caught.value.code == "NOT_A_STAGING_DIR"
    assert (plain / "keep.txt").is_file()


# Sampling semantics ----------------------------------------------------


def test_extract_baseline_1fps(tmp_path: Path) -> None:
    source = _make_video(tmp_path / "clip.avi", 10.0, 30)
    result = extract_frames(ExtractionConfig(
        video_path=source, out_dir=tmp_path / "frames",
        sample_fps=1.0))
    assert [frame.frame_index for frame in result.frames] == [0, 10, 20]
    assert [frame.timestamp_seconds for frame in result.frames] == [
        0.0, 1.0, 2.0]
    assert all(frame.image_path.is_file() for frame in result.frames)
    assert [frame.image_path.name for frame in result.frames] == [
        "frame_00000000.jpg", "frame_00000010.jpg",
        "frame_00000020.jpg"]
    rows = _read_manifest(tmp_path / "frames")
    assert len(rows) == 3
    assert rows[1]["frame_index"] == "10"
    assert rows[1]["timestamp_seconds"] == "1.0"
    for frame in result.frames:
        assert cv2.imread(str(frame.image_path)) is not None
    assert result.source.sha256 == hashlib.sha256(
        source.read_bytes()).hexdigest()
    report = json.loads(result.report_path.read_text(
        encoding="utf-8"))
    assert report["frame_indices"] == [0, 10, 20]
    assert report["sample_fps"] == 1.0


def test_extract_subwindow_and_source_offset(tmp_path: Path) -> None:
    source = _make_video(tmp_path / "clip.avi", 10.0, 30)
    result = extract_frames(ExtractionConfig(
        video_path=source, out_dir=tmp_path / "frames",
        sample_fps=2.0, start=0.5, end=2.0, source_offset=225.0))
    assert [frame.frame_index for frame in result.frames] == [5, 10, 15]
    assert [frame.timestamp_seconds for frame in result.frames] == [
        0.5, 1.0, 1.5]
    assert [frame.source_timestamp_seconds
            for frame in result.frames] == [225.5, 226.0, 226.5]
    assert [frame.requested_timestamp_seconds
            for frame in result.frames] == [0.5, 1.0, 1.5]


def test_extract_non_integer_fps(tmp_path: Path) -> None:
    source = _make_video(tmp_path / "clip.avi", 7.5, 30)
    props = probe_video(source)
    if props["fps"] == int(props["fps"]):
        pytest.skip("Backend coerced FPS to an integer.")
    assert props["fps"] == pytest.approx(7.5)
    result = extract_frames(ExtractionConfig(
        video_path=source, out_dir=tmp_path / "frames",
        sample_fps=1.0))
    indices = [frame.frame_index for frame in result.frames]
    assert indices == sorted(indices)
    assert len(set(indices)) == len(indices)
    for frame in result.frames:
        assert frame.timestamp_seconds == pytest.approx(
            round(frame.frame_index / props["fps"], 6))


def test_extract_end_of_video_behavior(tmp_path: Path) -> None:
    source = _make_video(tmp_path / "clip.avi", 10.0, 30)
    full = extract_frames(ExtractionConfig(
        video_path=source, out_dir=tmp_path / "full",
        sample_fps=1.0, end=None))
    assert [frame.frame_index for frame in full.frames] == [0, 10, 20]
    explicit = extract_frames(ExtractionConfig(
        video_path=source, out_dir=tmp_path / "explicit",
        sample_fps=1.0, end=3.0))
    assert [frame.frame_index for frame in explicit.frames] == [0, 10, 20]
    with pytest.raises(VideoError) as caught:
        extract_frames(ExtractionConfig(
            video_path=source, out_dir=tmp_path / "beyond",
            sample_fps=1.0, end=99.0))
    assert caught.value.code == "INVALID_WINDOW"


def test_extract_rejects_bad_parameters(tmp_path: Path) -> None:
    source = _make_video(tmp_path / "clip.avi", 10.0, 30)
    with pytest.raises(VideoError) as caught:
        extract_frames(ExtractionConfig(
            video_path=source, out_dir=tmp_path / "r1",
            sample_fps=11.0))
    assert caught.value.code == "INVALID_FPS"
    with pytest.raises(VideoError) as caught:
        extract_frames(ExtractionConfig(
            video_path=source, out_dir=tmp_path / "r2",
            sample_fps=1.0, start=2.0, end=1.0))
    assert caught.value.code == "INVALID_WINDOW"
    with pytest.raises(VideoError) as caught:
        extract_frames(ExtractionConfig(
            video_path=source, out_dir=tmp_path / "r3",
            sample_fps=0.0))
    assert caught.value.code == "INVALID_FPS"
    with pytest.raises(VideoError) as caught:
        extract_frames(ExtractionConfig(
            video_path=tmp_path / "absent.avi",
            out_dir=tmp_path / "r4", sample_fps=1.0))
    assert caught.value.code == "VIDEO_NOT_FOUND"


def test_extract_rejects_zero_frames_and_frame_cap(
        tmp_path: Path) -> None:
    source = _make_video(tmp_path / "clip.avi", 10.0, 30)
    with pytest.raises(VideoError) as caught:
        extract_frames(ExtractionConfig(
            video_path=source, out_dir=tmp_path / "empty",
            sample_fps=1.0, start=2.999, end=3.0))
    assert caught.value.code == "ZERO_FRAMES"
    with pytest.raises(VideoError) as caught:
        extract_frames(ExtractionConfig(
            video_path=source, out_dir=tmp_path / "capped",
            sample_fps=10.0, max_frames=2))
    assert caught.value.code == "TOO_MANY_FRAMES"
    assert not (tmp_path / "capped" / "manifest.csv").exists()


def test_extract_refuses_overwrite(tmp_path: Path) -> None:
    source = _make_video(tmp_path / "clip.avi", 10.0, 30)
    out_dir = tmp_path / "frames"
    extract_frames(ExtractionConfig(
        video_path=source, out_dir=out_dir, sample_fps=1.0))
    with pytest.raises(VideoError) as caught:
        extract_frames(ExtractionConfig(
            video_path=source, out_dir=out_dir, sample_fps=2.0))
    assert caught.value.code == "OUTPUT_EXISTS"
    assert len(_read_manifest(out_dir)) == 3


# End-to-end into the C.2 runner -----------------------------------------


def test_end_to_end_video_to_observations(tmp_path: Path) -> None:
    source = _make_video(tmp_path / "clip.avi", 10.0, 30)
    staged = stage_video(source, tmp_path / "staging")
    result = extract_frames(ExtractionConfig(
        video_path=staged.video_path, out_dir=tmp_path / "frames",
        sample_fps=1.0))
    assert result.source.sha256 == staged.metadata.sha256
    fake = _ok_client()
    report = FirstPassRunner(FirstPassConfig(
        frames=result.frames, source=result.source,
        out_dir=tmp_path / "run", model="fake-model",
        prompt_text="TEST PROMPT", client_factory=lambda: fake,
        sleeper=lambda seconds: None,
    )).run()
    assert report["frames_total"] == 3
    assert report["frames_observed"] == 3
    assert report["frames_failed"] == 0
    assert fake.calls == 3
    obs_path = (tmp_path / "run" / "observations"
                / "batch_00000000-00000020.json")
    observations = json.loads(obs_path.read_text(encoding="utf-8"))
    assert len(observations) == 3
    first = observations[0]
    assert first["frame_index"] == 0
    assert first["terminal_state"] == "valid_prediction"
    assert first["model_fields"]["scale_display_verbatim"] == "120"
    assert first["manifest_ref"]["manifest_sha256"] == (
        staged.metadata.sha256)
    assert first["timestamp_seconds"] == 0.0
