"""Prediction-vs-review comparison metrics (Exp. 007).

Every metric reports its numerator, denominator, and exclusions;
empty denominators yield null, never a fabricated score. Metrics
describe agreement between model predictions and human review, not
ground-truth accuracy of either side. Offline, stdlib-only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import review  # noqa: E402

METRICS_VERSION = "1.0.0"
ACTION_CLASSES = ("addition", "removal", "tare", "none")
REVIEWED_OUTCOMES = ("confirmed", "corrected")


class MetricsError(Exception):
    """Structured metrics failure with a machine-readable code."""

    def __init__(self, code: str, message: str, entity_id: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.entity_id = entity_id


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def model_action_label(prediction: dict[str, Any]) -> str | None:
    """Derive the model's action label, or None when it abstains."""
    if prediction.get("tare_action_visible") is True:
        return "tare"
    visible = prediction.get("addition_or_removal_visible")
    if visible in ACTION_CLASSES:
        return visible
    return None


def _metric(matches: int, denominator: int,
            excluded: dict[str, int]) -> dict[str, Any]:
    return {
        "matches": matches,
        "denominator": denominator,
        "value": (matches / denominator) if denominator else None,
        "excluded": excluded,
    }


def compare(annotations: list[Any]) -> dict[str, Any]:
    """Compare validated annotations against model predictions."""
    issues = review.validate_all(annotations)
    if issues:
        raise MetricsError(
            "INVALID_ANNOTATIONS",
            f"{len(issues)} annotation issue(s) block comparison.",
        )
    display_matches = 0
    display_total = 0
    display_excluded: dict[str, int] = {}
    numeric_matches = 0
    numeric_total = 0
    numeric_excluded: dict[str, int] = {}
    action_matches = 0
    action_total = 0
    action_excluded: dict[str, int] = {}
    per_class: dict[str, dict[str, int]] = {
        label: {"true_positives": 0, "model_positives": 0,
                "human_positives": 0}
        for label in ACTION_CLASSES
    }
    reviewed = 0
    abstained = 0

    def exclude(counter: dict[str, int], reason: str) -> None:
        counter[reason] = counter.get(reason, 0) + 1

    for item in annotations:
        assert isinstance(item, dict)  # validated above
        outcome = item["outcome"]
        human = item["human"]
        prediction = item["model_prediction"]
        if outcome != "pending":
            reviewed += 1
        if outcome in ("unreadable", "insufficient_evidence"):
            abstained += 1
        if outcome not in REVIEWED_OUTCOMES:
            exclude(display_excluded, "not_reviewed_display")
            exclude(numeric_excluded, "not_reviewed_numeric")
            exclude(action_excluded, "not_reviewed_action")
            continue
        verbatim = prediction.get("scale_display_verbatim")
        display = human.get("display_text")
        if not isinstance(verbatim, str):
            exclude(display_excluded, "model_verbatim_missing")
        elif not isinstance(display, str) or not display:
            exclude(display_excluded, "human_display_missing")
        else:
            display_total += 1
            if display == verbatim:
                display_matches += 1
        model_numeric = prediction.get("model_numeric")
        model_status = prediction.get("model_numeric_status")
        human_numeric = human.get("numeric_value")
        if model_status not in ("ok", "normalized") or not _is_number(
                model_numeric):
            exclude(numeric_excluded, "model_numeric_ambiguous")
        elif not _is_number(human_numeric):
            exclude(numeric_excluded, "human_numeric_missing")
        else:
            numeric_total += 1
            if human_numeric == model_numeric:
                numeric_matches += 1
        model_label = model_action_label(prediction)
        human_label = human.get("action")
        if model_label is None:
            exclude(action_excluded, "model_action_unknown")
        elif human_label is None:
            exclude(action_excluded, "human_action_missing")
        else:
            action_total += 1
            if human_label == model_label:
                action_matches += 1
            per_class[model_label]["model_positives"] += 1
            per_class[human_label]["human_positives"] += 1
            if human_label == model_label:
                per_class[human_label]["true_positives"] += 1
    action_classes = {}
    for label in ACTION_CLASSES:
        counts = per_class[label]
        true_positives = counts["true_positives"]
        model_positives = counts["model_positives"]
        human_positives = counts["human_positives"]
        action_classes[label] = {
            "true_positives": true_positives,
            "model_positives": model_positives,
            "human_positives": human_positives,
            "precision": (
                true_positives / model_positives
                if model_positives else None
            ),
            "recall": (
                true_positives / human_positives
                if human_positives else None
            ),
        }
    total = len(annotations)
    return {
        "metrics_version": METRICS_VERSION,
        "annotation_count": total,
        "reviewed_count": reviewed,
        "coverage": {
            "reviewed": reviewed,
            "denominator": total,
            "value": (reviewed / total) if total else None,
        },
        "abstention": {
            "abstained": abstained,
            "denominator": reviewed,
            "value": (abstained / reviewed) if reviewed else None,
        },
        "display_exact_match": _metric(
            display_matches, display_total, display_excluded
        ),
        "numeric_agreement": _metric(
            numeric_matches, numeric_total, numeric_excluded
        ),
        "action_agreement": _metric(
            action_matches, action_total, action_excluded
        ),
        "action_classes": action_classes,
    }


def write_metrics(annotations_path: Path,
                  output_path: Path) -> dict[str, Any]:
    """Load annotations, compare, and write metrics deterministically."""
    annotations = review.load_annotations(annotations_path)
    report = compare(annotations)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Minimal CLI: metrics ANNOTATIONS OUTPUT."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args(argv)
    try:
        report = write_metrics(args.annotations, args.output_json)
    except (review.ReviewError, MetricsError) as exc:
        print(f"METRICS FAILED [{exc.code}]: {exc.message}")
        return 1
    print(json.dumps({
        "annotation_count": report["annotation_count"],
        "reviewed_count": report["reviewed_count"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
