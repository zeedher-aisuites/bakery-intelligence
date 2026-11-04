"""Single source of truth for local video size limits.

Upload staging and pipeline execution must agree: a video the
upload endpoint accepts must not predictably fail later inside
the pipeline because of a smaller, undocumented limit. Both
layers resolve the same effective limit here.

Precedence (first valid positive value wins):
``BAKERY_PIPELINE_MAX_VIDEO_BYTES``, then
``BAKERY_MAX_VIDEO_UPLOAD_BYTES``, then the 2 GiB default.
"""

from __future__ import annotations

import os

DEFAULT_VIDEO_MAX_BYTES = 2 * 1024 * 1024 * 1024


def _positive_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def upload_max_bytes() -> int:
    """Cap enforced while streaming an upload to disk."""
    return (
        _positive_env("BAKERY_MAX_VIDEO_UPLOAD_BYTES")
        or _positive_env("BAKERY_PIPELINE_MAX_VIDEO_BYTES")
        or DEFAULT_VIDEO_MAX_BYTES
    )


def pipeline_max_bytes() -> int:
    """Cap enforced by pipeline execution for one video."""
    return (
        _positive_env("BAKERY_PIPELINE_MAX_VIDEO_BYTES")
        or _positive_env("BAKERY_MAX_VIDEO_UPLOAD_BYTES")
        or DEFAULT_VIDEO_MAX_BYTES
    )


def effective_video_max_bytes() -> int:
    """Largest video accepted end to end (the stricter cap)."""
    return min(upload_max_bytes(), pipeline_max_bytes())
