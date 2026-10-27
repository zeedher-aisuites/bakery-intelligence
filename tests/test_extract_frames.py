"""Unit tests for deterministic frame extraction."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "003_video_to_recipe"
    / "extract_frames.py"
)
spec = importlib.util.spec_from_file_location("extract_frames_003", SCRIPT)
assert spec is not None and spec.loader is not None
extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractor)


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48)
    )
    if not writer.isOpened():
        pytest.skip("MJPG video encoding is unavailable in this OpenCV build.")
    for index in range(30):
        frame = np.full((48, 64, 3), index * 7, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_frame_indices_and_source_timestamps(
    synthetic_video: Path, tmp_path: Path
) -> None:
    output = tmp_path / "frames"
    rows = extractor.extract_frames(
        synthetic_video, output, start=0.0, end=3.0, sample_fps=2.0,
        source_offset=225.0,
    )
    assert [row["frame_index"] for row in rows] == [0, 5, 10, 15, 20, 25]
    assert [row["timestamp_seconds"] for row in rows] == [
        0.0, 0.5, 1.0, 1.5, 2.0, 2.5
    ]
    assert [row["source_timestamp_seconds"] for row in rows] == [
        225.0, 225.5, 226.0, 226.5, 227.0, 227.5
    ]
    with (output / "manifest.csv").open(newline="", encoding="utf-8") as file:
        manifest_rows = list(csv.DictReader(file))
    assert len(manifest_rows) == 6
    for row in rows:
        image = cv2.imread(str(output / str(row["image_path"])))
        assert image is not None
        assert abs(float(image.mean()) - int(row["frame_index"]) * 7) < 5


@pytest.mark.parametrize(
    "start,end,sample_fps,offset",
    [(-1, 1, 1, 0), (1, 1, 1, 0), (0, 1, 0, 0),
     (0, 1, 11, 0), (0, 4, 1, 0), (0, 1, 1, -1)],
)
def test_invalid_parameters(
    synthetic_video: Path, tmp_path: Path,
    start: float, end: float, sample_fps: float, offset: float,
) -> None:
    with pytest.raises(ValueError):
        extractor.extract_frames(
            synthetic_video, tmp_path / "frames", start, end, sample_fps, offset
        )


def test_missing_video(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        extractor.extract_frames(
            tmp_path / "missing.mp4", tmp_path / "frames", 0, 1, 1
        )
