"""Deterministic consolidated evidence package.

Merges the immutable evidence package with human observation
corrections. A correction is authoritative only for its exact
field, frames, and interval; adjacent observations are never
expanded into. Original values are preserved alongside the
effective values, and the package carries a reproducible
content hash.
"""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

CONSOLIDATED_VERSION = "consolidated-evidence/1"


def _content_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, default=str).encode()).hexdigest()


def consolidate(package: dict[str, Any],
                corrections: dict[str, Any]) -> dict[str, Any]:
    """Merge corrections into a versioned consolidated package."""
    frames = copy.deepcopy(package.get("frames", []))
    by_index = {frame["frame_index"]: frame for frame in frames}
    applied: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for task_id in sorted(corrections):
        correction = corrections[task_id]
        applied.append(copy.deepcopy(correction))
        if correction.get("explicit_unknown"):
            unresolved.append({
                "code": "HUMAN_MARKED_UNKNOWN",
                "task_id": task_id,
                "frame_refs": correction.get("frame_refs"),
            })
            continue
        if correction.get("action") in ("supply", "correct", "confirm"):
            for frame_index in correction.get("frame_refs", []):
                frame = by_index.get(frame_index)
                if frame is None:
                    continue
                measurement = frame.setdefault("measurement", {})
                original = measurement.get("value")
                if correction.get("action") == "confirm":
                    effective = original
                else:
                    effective = correction.get("human_value")
                frame["effective"] = {
                    "value": effective,
                    "original_value": original,
                    "provenance": "human-corrected",
                    "task_id": task_id,
                    "revision": correction.get("revision"),
                }

    body = {
        "schema_version": CONSOLIDATED_VERSION,
        "run_id": package.get("run_id"),
        "source_schema_version": package.get("schema_version"),
        "video": copy.deepcopy(package.get("video")),
        "frames": frames,
        "action_evidence": copy.deepcopy(
            package.get("action_evidence")),
        "weighing_event": copy.deepcopy(
            package.get("weighing_event")),
        "corrections": applied,
        "unknowns": [
            *copy.deepcopy(package.get("unknowns", [])),
            *unresolved,
        ],
        "review_status": {
            "corrections": len(applied),
            "explicit_unknowns": len(unresolved),
        },
    }
    return {**body, "content_hash": _content_hash(body)}
