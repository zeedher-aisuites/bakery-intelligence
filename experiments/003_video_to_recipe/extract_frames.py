"""Extract reproducible, timestamped frames from a local video."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


MANIFEST_FIELDS = (
    "image_path",
    "frame_index",
    "timestamp_seconds",
    "source_timestamp_seconds",
    "requested_timestamp_seconds",
)


def extract_frames(
    video_path: Path,
    output_dir: Path,
    start: float,
    end: float,
    sample_fps: float,
    source_offset: float = 0.0,
) -> list[dict[str, str | int | float]]:
    """Extract frames at regular requested times using source-frame indices.

    The interval is half-open: [start, end). Timestamps are derived from the
    source frame rate; they are not presentation timestamps for variable-FPS video.
    """
    if not video_path.is_file():
        raise ValueError(f"Video does not exist: {video_path}")
    if start < 0 or end <= start:
        raise ValueError("Expected 0 <= start < end.")
    if sample_fps <= 0:
        raise ValueError("Sample FPS must be positive.")
    if source_offset < 0:
        raise ValueError("Source offset cannot be negative.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or frame_count <= 0:
            raise RuntimeError("Video FPS or frame count is unavailable.")
        if sample_fps > fps:
            raise ValueError("Sample FPS cannot exceed source FPS.")
        if end > frame_count / fps + 1e-7:
            raise ValueError("End time exceeds video duration.")

        output_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, str | int | float]] = []
        used_indices: set[int] = set()
        sample_number = 0
        while start + sample_number / sample_fps < end - 1e-9:
            requested_time = start + sample_number / sample_fps
            frame_index = int(requested_time * fps + 0.5)
            if frame_index >= frame_count:
                break
            if frame_index in used_indices:
                raise RuntimeError("Duplicate frame index in sampling schedule.")
            used_indices.add(frame_index)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode frame {frame_index}.")
            image_name = f"frame_{frame_index:08d}.jpg"
            if not cv2.imwrite(str(output_dir / image_name), frame):
                raise RuntimeError(f"Cannot write image: {image_name}")
            actual_time = frame_index / fps
            rows.append(
                {
                    "image_path": image_name,
                    "frame_index": frame_index,
                    "timestamp_seconds": round(actual_time, 6),
                    "source_timestamp_seconds": round(actual_time + source_offset, 6),
                    "requested_timestamp_seconds": round(requested_time, 6),
                }
            )
            sample_number += 1

        manifest = output_dir / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return rows
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--source-offset", type=float, default=0.0)
    args = parser.parse_args()
    rows = extract_frames(
        args.video, args.output, args.start, args.end, args.fps, args.source_offset
    )
    print(f"Extracted {len(rows)} frames to {args.output}")
    print(f"Manifest: {args.output / 'manifest.csv'}")


if __name__ == "__main__":
    main()
