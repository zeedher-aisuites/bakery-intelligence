"""Normalized, provenance-preserving evidence package.

Built read-only from a completed run's existing artifacts:
pipeline summary, measurements, first-pass report, recipe
draft, weighing event, retained frame observations, and human
review annotations. Nothing is inferred here: ingredient
identity, units, and net quantities remain explicit unknowns
until evidence or a human supplies them.
"""

from __future__ import annotations

import json
from typing import Any

from bakery_intelligence.api.jobs import JobRecord
from bakery_intelligence.api.storage import LocalStorageBackend

PACKAGE_VERSION = "evidence-package/1"


def _read_json(storage: LocalStorageBackend, run_id: str,
               artifact_id: str) -> dict[str, Any] | None:
    try:
        return json.loads(
            storage.read_artifact(run_id, artifact_id).decode("utf-8"))
    except Exception:
        return None


def build_evidence_package(
        storage: LocalStorageBackend,
        record: JobRecord) -> dict[str, Any]:
    """Assemble the deterministic evidence package for one run."""
    run_id = record.run_id
    summary = _read_json(storage, run_id, "pipeline_summary.json") or {}
    measurements_doc = _read_json(
        storage, run_id, "measurements.json") or {}
    firstpass = _read_json(
        storage, run_id, "firstpass_report.json") or {}
    draft = _read_json(storage, run_id, "recipe_draft.json") or {}
    weighing = _read_json(storage, run_id, "weighing_event.json") or {}

    reviews: dict[str, Any] = {}
    try:
        from bakery_intelligence.api.review import read_reviews
        reviews = read_reviews(
            storage.base_dir, run_id).get("annotations", {})
    except Exception:
        reviews = {}

    observations: dict[int, dict[str, Any]] = {}
    try:
        from bakery_intelligence.api.review import build_review_bundle
        bundle = build_review_bundle(storage, record)
        for frame in bundle.get("frames", []):
            observations[frame["frame_index"]] = frame
    except Exception:
        observations = {}

    video = summary.get("video") if isinstance(
        summary.get("video"), dict) else {}
    duration = video.get("duration_seconds")
    if not isinstance(duration, (int, float)):
        window = measurements_doc.get("window")
        duration = (window.get("end_seconds")
                    if isinstance(window, dict) else None)

    frames: list[dict[str, Any]] = []
    for item in measurements_doc.get("measurements", []):
        if not isinstance(item, dict):
            continue
        frame_index = item.get("frame_index")
        if not isinstance(frame_index, int):
            continue
        observed = observations.get(frame_index, {})
        observation = observed.get("observation") or {}
        model_fields = observation.get("model_fields") or {}
        frames.append({
            "frame_index": frame_index,
            "timestamp_seconds": item.get("timestamp_seconds"),
            "image_name": observed.get("image_name")
            or f"frame_{frame_index:08d}.jpg",
            "image_available": bool(observed.get("image_available")),
            "measurement": {
                "measurement_id": item.get("measurement_id"),
                "value": item.get("value"),
                "value_status": item.get("value_status"),
                "value_source": item.get("value_source"),
                "unit_observed": item.get("unit_observed"),
            },
            "visible_objects": model_fields.get("visible_objects"),
            "scale": {
                "display_verbatim": model_fields.get(
                    "scale_display_verbatim"),
                "readable": model_fields.get("scale_readable"),
                "state": model_fields.get("scale_state"),
                "tare_action_visible": model_fields.get(
                    "tare_action_visible"),
                "addition_or_removal_visible": model_fields.get(
                    "addition_or_removal_visible"),
                "container_description": model_fields.get(
                    "container_description"),
                "confidence": model_fields.get("confidence"),
                "ambiguity_notes": model_fields.get("ambiguity_notes"),
            },
            "human_review": reviews.get(str(frame_index)),
        })
    frames.sort(key=lambda frame: frame["frame_index"])

    unknowns = [
        {"code": item.get("code"), "message": item.get("message")}
        for item in draft.get("unresolved_questions", [])
        if isinstance(item, dict)
    ]
    if weighing.get("ingredient_id") is None:
        unknowns.append({
            "code": "INGREDIENT_UNKNOWN",
            "message": "No ingredient identity is established by "
                       "the recorded evidence.",
        })

    return {
        "schema_version": PACKAGE_VERSION,
        "run_id": run_id,
        "workflow": record.workflow,
        "video": {
            "source_object_id": record.source_object_id,
            "source_sha256": record.source_sha256
            or summary.get("source_sha256"),
            "file_name": video.get("file_name"),
            "duration_seconds": duration,
            "fps": video.get("fps"),
            "width": video.get("width"),
            "height": video.get("height"),
        },
        "frames": frames,
        "action_evidence": measurements_doc.get("action_evidence"),
        "weighing_event": weighing or None,
        "firstpass": {
            "model": firstpass.get("model"),
            "provider": firstpass.get("provider"),
            "requests_sent": firstpass.get("requests_sent"),
            "input_tokens": firstpass.get("input_tokens"),
            "output_tokens": firstpass.get("output_tokens"),
            "cost_usd_estimated": firstpass.get("cost_usd_estimated"),
            "cost_status": firstpass.get("cost_status"),
        },
        "human_reviews": reviews,
        "audio_transcript": None,
        "unknowns": unknowns,
        "provenance": {
            "artifact_ids": list(record.artifact_ids),
            "artifact_sha256": dict(record.artifact_sha256),
            "draft_id": draft.get("draft_id"),
        },
    }
