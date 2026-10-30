"""Blank review templates and annotation validation (Exp. 007).

Templates freeze the model prediction beside blank human fields; a
human reviewer fills them in. Validation is structural only: it
checks contract conformance, never visual correctness. Offline,
stdlib-only, no model calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "006_p04_multimodal_inference"))

import temporal  # noqa: E402

TEMPLATE_VERSION = "1.0.0"

OUTCOMES = (
    "pending", "confirmed", "corrected", "unreadable",
    "insufficient_evidence",
)
ACTIONS = ("addition", "removal", "tare", "none")
VISIBILITY = ("clear", "partial", "obstructed")
UNITS = ("g", "kg", "mg", "oz", "lb", "ml", "l")
HUMAN_KEYS = (
    "display_text", "numeric_value", "unit", "action",
    "tare_visible", "visibility", "uncertainty_notes",
)


class ReviewError(Exception):
    """Structured review failure with a machine-readable code."""

    def __init__(self, code: str, message: str, entity_id: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.entity_id = entity_id


def annotation_id(window_id: str, frame_index: int) -> str:
    """Deterministic annotation identifier."""
    return f"{window_id}-{frame_index:08d}"


def build_templates(manifest: dict[str, Any]) -> dict[str, Any]:
    """Build one blank annotation per selected observation."""
    if not isinstance(manifest, dict) or "windows" not in manifest:
        raise ReviewError("MANIFEST_INVALID", "Manifest lacks windows.")
    annotations = []
    for window in manifest["windows"]:
        window_id = window["window_id"]
        for obs in window.get("observations", []):
            fields = obs.get("model_fields", {})
            verbatim = fields.get("scale_display_verbatim")
            parsed = temporal.parse_reading(verbatim)
            annotations.append({
                "annotation_id": annotation_id(
                    window_id, obs["frame_index"]
                ),
                "window_id": window_id,
                "frame_index": obs["frame_index"],
                "timestamp_seconds": obs["timestamp_seconds"],
                "model_prediction": {
                    "scale_display_verbatim": verbatim,
                    "model_numeric": parsed.value,
                    "model_numeric_status": parsed.status,
                    "scale_readable": fields.get("scale_readable"),
                    "scale_state": fields.get("scale_state"),
                    "tare_action_visible": fields.get(
                        "tare_action_visible"
                    ),
                    "addition_or_removal_visible": fields.get(
                        "addition_or_removal_visible"
                    ),
                    "confidence": fields.get("confidence"),
                    "ambiguity_notes": fields.get("ambiguity_notes"),
                    "batch_id": obs.get("batch_id"),
                },
                "human": {key: None for key in HUMAN_KEYS},
                "support": {
                    "frame_index": obs["frame_index"],
                    "timestamp_seconds": obs["timestamp_seconds"],
                },
                "outcome": "pending",
                "reviewer": None,
                "reviewed_at": None,
            })
    annotations.sort(key=lambda item: item["annotation_id"])
    return {
        "template_version": TEMPLATE_VERSION,
        "annotation_count": len(annotations),
        "annotations": annotations,
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_annotation(item: Any) -> list[dict[str, str]]:
    """Return structural issues for one annotation (empty when valid)."""
    issues: list[dict[str, str]] = []

    def report(code: str, message: str) -> None:
        entity = item.get("annotation_id", "?") if isinstance(
            item, dict) else "?"
        issues.append({
            "annotation_id": str(entity), "code": code,
            "message": message,
        })

    if not isinstance(item, dict):
        report("NOT_AN_OBJECT", "Annotation is not an object.")
        return issues
    for key in ("annotation_id", "window_id", "frame_index",
                "timestamp_seconds", "model_prediction", "human",
                "support", "outcome"):
        if key not in item:
            report("MISSING_KEY", f"Missing key: {key}.")
    if issues:
        return issues
    human = item["human"]
    if not isinstance(human, dict):
        report("HUMAN_NOT_AN_OBJECT", "human is not an object.")
        return issues
    for key in HUMAN_KEYS:
        if key not in human:
            report("MISSING_KEY", f"human lacks key: {key}.")
    outcome = item["outcome"]
    if outcome not in OUTCOMES:
        report("INVALID_OUTCOME", f"Unknown outcome: {outcome!r}.")
        return issues
    if outcome == "pending":
        for key in HUMAN_KEYS:
            if human.get(key) is not None:
                report("PENDING_NOT_BLANK",
                       f"Pending annotation sets human.{key}.")
        if item.get("reviewer") is not None:
            report("PENDING_NOT_BLANK",
                   "Pending annotation sets reviewer.")
        if item.get("reviewed_at") is not None:
            report("PENDING_NOT_BLANK",
                   "Pending annotation sets reviewed_at.")
        return issues
    if not item.get("reviewer") or not isinstance(
            item.get("reviewer"), str):
        report("MISSING_REVIEWER", "Reviewed annotation lacks reviewer.")
    reviewed_at = item.get("reviewed_at")
    if reviewed_at is not None and (
            not isinstance(reviewed_at, str) or not reviewed_at):
        report("INVALID_REVIEWED_AT", "reviewed_at must be a string.")
    model = item["model_prediction"]
    verbatim = model.get("scale_display_verbatim") if isinstance(
        model, dict) else None
    display = human.get("display_text")
    if outcome in ("confirmed", "corrected"):
        if not isinstance(display, str) or not display:
            report("MISSING_DISPLAY",
                   "Reviewed display requires human display_text.")
        elif outcome == "confirmed" and display != verbatim:
            report("MISMATCH_CONFIRMED",
                   "Confirmed display must exactly match prediction.")
        elif outcome == "corrected" and (
                verbatim is not None and display == verbatim):
            report("INVALID_CORRECTION",
                   "Corrected display must differ from prediction.")
    else:  # unreadable or insufficient_evidence
        for key in ("display_text", "numeric_value", "unit"):
            if human.get(key) is not None:
                report("NONBLANK_UNREVIEWABLE",
                       f"{outcome} annotation sets human.{key}.")
        if outcome == "insufficient_evidence":
            for key in ("action", "tare_visible", "visibility"):
                if human.get(key) is not None:
                    report("NONBLANK_UNREVIEWABLE",
                           f"{outcome} annotation sets human.{key}.")
    numeric = human.get("numeric_value")
    if numeric is not None and not _is_number(numeric):
        report("INVALID_NUMERIC", "human.numeric_value must be numeric.")
    unit = human.get("unit")
    if unit is not None:
        if unit not in UNITS:
            report("INVALID_UNIT", f"Unknown unit: {unit!r}.")
        if numeric is None:
            report("UNIT_WITHOUT_NUMERIC",
                   "human.unit requires human.numeric_value.")
    action = human.get("action")
    if action is not None and action not in ACTIONS:
        report("INVALID_ACTION", f"Unknown action: {action!r}.")
    tare = human.get("tare_visible")
    if tare is not None and not isinstance(tare, bool):
        report("INVALID_TARE", "human.tare_visible must be boolean.")
    visibility = human.get("visibility")
    if visibility is not None and visibility not in VISIBILITY:
        report("INVALID_VISIBILITY", f"Unknown visibility: {visibility!r}.")
    notes = human.get("uncertainty_notes")
    if notes is not None and not isinstance(notes, str):
        report("INVALID_NOTES", "human.uncertainty_notes must be text.")
    return issues


def validate_all(annotations: list[Any]) -> list[dict[str, str]]:
    """Validate a list of annotations; collect every issue."""
    issues: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in annotations:
        issues.extend(validate_annotation(item))
        if isinstance(item, dict) and "annotation_id" in item:
            entity = str(item["annotation_id"])
            if entity in seen:
                issues.append({
                    "annotation_id": entity,
                    "code": "DUPLICATE_ID",
                    "message": "Duplicate annotation_id.",
                })
            seen.add(entity)
    return issues


def write_templates(manifest_path: Path, output_path: Path) -> dict[str, Any]:
    """Load a manifest and write blank templates deterministically."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewError(
            "MANIFEST_INVALID", f"Unreadable manifest: {manifest_path}",
            str(manifest_path),
        ) from exc
    bundle = build_templates(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    return bundle


def load_annotations(path: Path) -> list[Any]:
    """Load an annotation bundle or bare list from disk."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewError(
            "ANNOTATIONS_INVALID", f"Unreadable annotations: {path}",
            str(path),
        ) from exc
    if isinstance(payload, dict) and "annotations" in payload:
        payload = payload["annotations"]
    if not isinstance(payload, list):
        raise ReviewError(
            "ANNOTATIONS_INVALID", "Annotations are not a list.",
            str(path),
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    """Minimal CLI: review template MANIFEST OUT | validate FILE."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("template", help="write blank templates")
    make.add_argument("manifest", type=Path)
    make.add_argument("output_json", type=Path)
    check = sub.add_parser("validate", help="validate annotations")
    check.add_argument("annotations", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "template":
            bundle = write_templates(args.manifest, args.output_json)
            print(json.dumps(
                {"annotation_count": bundle["annotation_count"]}))
            return 0
        annotations = load_annotations(args.annotations)
        issues = validate_all(annotations)
        print(json.dumps({
            "annotation_count": len(annotations),
            "issue_count": len(issues),
            "issues": issues,
        }, indent=2, sort_keys=True))
        return 1 if issues else 0
    except ReviewError as exc:
        print(f"REVIEW FAILED [{exc.code}]: {exc.message}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
