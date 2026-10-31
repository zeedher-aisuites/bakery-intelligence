# M2-06C.3 video intake and frame extraction

Status: implemented, awaiting host-side verification.
Offline only: local files, OpenCV decode, no network.

## 1. Components

`src/bakery_intelligence/video/intake.py` stages an arbitrary
supported video: streamed copy with incremental SHA-256,
container allowlist, size cap (default 4 GiB), best-effort
free-space check, content-addressed naming
(`video_<sha16>_<stem><suffix>`), read-back probing, and a
provenance sidecar. Identical re-staging reuses the verified
copy; anything else already present fails instead of
overwriting. Probe failures remove the just-copied file.

`src/bakery_intelligence/video/extract.py` samples frames with
Exp. 003 semantics and no fixed video assumptions: JPEG
frames, `manifest.csv` with the historical five columns,
`extraction_report.json`, and direct `FrameRef`/`SourceRef`
outputs for the C.2 runner. The source anchor is the staged
video's SHA-256; extraction parameters live in the report.

## 2. Timestamp and index semantics

- Probing reads container `fps` and `frame_count` once;
  `duration = frame_count / fps`.
- The window is half-open: `[start, end)`; `end=None` means
  exactly `duration`. `end` beyond duration fails (no silent
  clamping).
- Sample `n` requests `start + n / sample_fps` and maps to
  `index = int(t * fps + 0.5)`. Reported
  `timestamp_seconds = index / fps`; `source_timestamp`
  adds the configured offset; `requested_timestamp` keeps
  the pre-rounding request. All rounded to six decimals.
- A repeated index fails (`DUPLICATE_SCHEDULE`); indices at
  or past `frame_count` end sampling; a window selecting no
  frames fails (`ZERO_FRAMES`).
- Non-integer source FPS follows the same formula against
  the probed rate; behavior is covered by an offline test
  (skipped only if the backend coerces FPS to an integer).

## 3. Safety and retention

- Staging refuses unknown containers, oversized inputs, and
  low disk (best-effort: checked, then trusted).
- Extraction refuses `sample_fps` above source rate,
  non-positive rates, negative offsets, frame counts above
  `max_frames` (default 100,000), and any output directory
  that already holds a manifest. Failed runs write no
  manifest; partial JPEGs may remain and a rerun reuses the
  directory deterministically.
- Nothing is deleted automatically. `purge_staging`
  removes only directories carrying the staging marker and
  refuses anything else. Operator-owned retention beyond
  that is out of scope for C.3.
- Sidecars persist basenames only, never absolute host
  paths. No filename from a client is ever joined
  unsanitized.

## 4. C.2 compatibility

`extract_frames` returns `frames: list[FrameRef]` (absolute
staged JPEG paths, validated timestamps) and
`source: SourceRef(name, video_sha256)`, which feed
`FirstPassConfig` unchanged. Proven end to end by
`test_end_to_end_video_to_observations`: synthetic video →
stage → extract → C.2 runner with fake client → valid
observations whose `manifest_ref` carries the video SHA-256.

## 5. Verification status

Implemented and statically reviewed; Python is unavailable in
the agent sandbox, so the suite was authored but not executed
here. Host-side validation:

```powershell
uv run python -m pytest tests/test_video_intake.py -v
uv run python -m pytest -q
uv run python scripts/sync_contract_schemas.py --check
git diff --check
git status --short
```

The suite generates its own MJPG videos and skips only when
the backend cannot encode them or coerces FPS.

## 6. Limitations

- JPEG bytes depend on the OpenCV/libjpeg build; frame
  selection and metadata do not.
- No extraction resume: reruns restart (guarded by the
  manifest check). Workers, HTTP large upload, and cloud
  storage arrive in later milestones; the 16 MB API cap is
  unchanged and cannot carry real videos.
- Disk-space and duration checks trust container metadata;
  malformed files fail at decode with explicit errors.
- Supported suffixes are an allowlist (mp4/mov/mkv/avi/m4v/
  mpg/mpeg/webm, case-insensitive); anything else fails
  closed even if a decoder could open it.
