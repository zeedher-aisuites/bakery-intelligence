"""Fixed prompts and structured observation records (Exp. 006).

Prompts are frozen constants: every request carries prompt text whose
SHA-256 is recorded, so outputs stay attributable. Predictions are
plain experiment JSON, never canonical ledger records.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

PRIMARY_PROMPT = """Analyze one cooking-video frame for a baking-evidence study.
Reply with exactly one JSON object and no other text, using these keys:
{
  "visible_objects": ["short noun phrases, empty list if unclear"],
  "scale_display_verbatim": "exact digits/segments on the scale display, or null",
  "scale_readable": true/false,
  "scale_state": "stable/changing/zero/off/occluded/unknown",
  "tare_action_visible": true/false,
  "addition_or_removal_visible": "addition/removal/none/unknown",
  "container_description": "short phrase or null",
  "container_changed_hint": true/false,
  "confidence": "high/medium/low",
  "ambiguity_notes": "short note or null"
}
Rules: report only what is visible; use null when evidence is
insufficient; never infer ingredient identity, units, or net mass;
never convert a display reading into a quantity; a negative display
stays a display string, never a negative quantity."""

REVIEW_PROMPT = """You review one primary frame analysis for errors.
Input: the frame image(s) plus the primary JSON.
Reply with exactly one JSON object and no other text, using these keys:
{
  "agree": true/false,
  "corrected_fields": {"field": "value"},
  "missed_evidence": "short note or null",
  "contradiction_flag": true/false,
  "confidence": "high/medium/low"
}
Rules: correct only against visible evidence; keep disagreements
explicit; never invent readings; never output net masses."""

TIMESTAMP_BASIS = "frame_index_over_fps"

REVIEW_OUTPUT_FIELDS = (
    "agree",
    "corrected_fields",
    "missed_evidence",
    "contradiction_flag",
    "confidence",
)

MODEL_OUTPUT_FIELDS = (
    "visible_objects",
    "scale_display_verbatim",
    "scale_readable",
    "scale_state",
    "tare_action_visible",
    "addition_or_removal_visible",
    "container_description",
    "container_changed_hint",
    "confidence",
    "ambiguity_notes",
)


def prompt_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def parse_model_json(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse strict model JSON; never invent fields on failure."""
    try:
        payload = json.loads(raw_text)
    except (json.JSONDecodeError, ValueError):
        return None, "unparseable"
    if not isinstance(payload, dict):
        return None, "not_an_object"
    return payload, None


def format_review_prompt(primary_json: str) -> str:
    """Append the primary observation JSON the review grounds on."""
    return REVIEW_PROMPT + "\nPRIMARY JSON:\n" + primary_json


def build_observation(
    frame_row: dict[str, Any],
    manifest_ref: dict[str, str],
    batch_id: str,
    pass_id: str,
    call_provenance: dict[str, Any],
    raw_text: str,
    fields: tuple[str, ...] = MODEL_OUTPUT_FIELDS,
    response_status: str | None = None,
    incomplete_reason: str | None = None,
    terminal_state: str | None = None,
    image_sha256: str | None = None,
    image_encoding: str | None = None,
) -> dict[str, Any]:
    """Assemble one timestamp-grounded prediction record.

    The trailing lifecycle arguments are optional evidence; older
    callers omit them and older artifacts lack the keys. Readers must
    treat absent keys as unknown, never as failure.
    """
    parsed, parse_error = parse_model_json(raw_text)
    model_fields: dict[str, Any] = {name: None for name in fields}
    if parsed is not None:
        for name in fields:
            if name in parsed:
                model_fields[name] = parsed[name]
    return {
        "frame_index": frame_row["frame_index"],
        "image_path": frame_row["image_path"],
        "timestamp_seconds": frame_row["timestamp_seconds"],
        "source_timestamp_seconds": frame_row["source_timestamp_seconds"],
        "requested_timestamp_seconds": frame_row[
            "requested_timestamp_seconds"
        ],
        "timestamp_basis": TIMESTAMP_BASIS,
        "manifest_ref": manifest_ref,
        "batch_id": batch_id,
        "pass_id": pass_id,
        "call": call_provenance,
        "model_fields": model_fields,
        "parse_error": parse_error,
        "raw_text_excerpt": raw_text[:500],
        "response_status": response_status,
        "incomplete_reason": incomplete_reason,
        "terminal_state": terminal_state,
        "image_ref": {
            "sha256": image_sha256,
            "encoding": image_encoding,
        },
    }


REVIEW_TRIGGER_CONFIDENCES = ("low",)


def select_review_candidates(
    observations: list[dict[str, Any]],
) -> list[int]:
    """Deterministic Wonder-review selection over primary outputs.

    Selects a frame when any trigger holds: unreadable scale display,
    visible tare action, container-change hint, low confidence, or an
    unparsed model reply. Output is sorted unique frame indices.
    """
    selected: set[int] = set()
    for obs in observations:
        fields = obs.get("model_fields", {})
        triggers = (
            obs.get("parse_error") is not None
            or fields.get("scale_readable") is False
            or fields.get("tare_action_visible") is True
            or fields.get("container_changed_hint") is True
            or fields.get("confidence") in REVIEW_TRIGGER_CONFIDENCES
        )
        if triggers:
            selected.add(int(obs["frame_index"]))
    return sorted(selected)
