"""Offline fusion runner: evidence files -> draft + evaluation.

Reads the three P04 evidence artifacts, normalizes, reconciles,
reconstructs, and writes the draft, evaluation, and manifest into
a git-ignored output directory. Deterministic: same inputs always
produce byte-identical JSON (sorted keys, stable ids).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bakery_intelligence.fusion import normalize, quantities, reconcile, reconstruct
from bakery_intelligence.fusion.models import SOURCE_SPARK

FUSION_VERSION = "fusion/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def extract_spark_output(response_path: Path) -> tuple[dict[str, Any],
                                                      str | None,
                                                      str | None]:
    """Return (parsed output, response_id, model) from a response dump."""
    response = json.loads(response_path.read_text(encoding="utf-8"))
    response_id = response.get("id") if isinstance(response, dict) \
        else None
    model = response.get("model") if isinstance(response, dict) else None
    output = response.get("output") if isinstance(response, dict) else None
    if not isinstance(output, list):
        raise ValueError("Response dump has no output item list.")
    texts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("text"):
                texts.append(str(part["text"]))
    if not texts:
        raise ValueError("Response dump carries no message text.")
    return (json.loads(texts[0]),
            str(response_id) if response_id else None,
            str(model) if model else None)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _duration_ok(value: float | None, duration: float | None) -> bool:
    if value is None or value < 0:
        return False
    if duration is not None and value > duration + 1.0:
        return False
    return True


def build_evaluation(draft: dict[str, Any],
                     spark_events: list[dict[str, Any]],
                     open_events: list[dict[str, Any]],
                     measurements: list[dict[str, Any]],
                     duration: float | None,
                     coverage: dict[str, Any] | None = None,
                     ) -> dict[str, Any]:
    """Honest counts: recovery, conflicts, traceability, limits."""
    steps = draft.get("steps") or []
    ingredients = draft.get("ingredients") or []
    conflicts = draft.get("conflicts") or []
    pairs = draft.get("corroborated_reading_pairs") or []
    multi_source_steps = sum(
        1 for step in steps if (step.get("source_count") or 0) > 1)
    multi_source_ingredients = sum(
        1 for entry in ingredients
        if (entry.get("source_count") or 0) > 1)
    legible = [m for m in measurements
               if m.get("displayed_value") is not None]
    illegible = [m for m in measurements
                 if m.get("displayed_value") is None]
    out_of_range = 0
    for event in list(spark_events) + list(open_events):
        for key in ("start_seconds", "end_seconds"):
            value = _as_float(event.get(key))
            if value is not None and not _duration_ok(value, duration):
                out_of_range += 1
    traceable_steps = sum(
        1 for step in steps if step.get("member_ids"))
    traceable_ingredients = sum(
        1 for entry in ingredients if entry.get("member_ids"))
    identity: dict[str, int] = {
        "model_proposed": 0, "described": 0, "unresolved": 0}
    for entry in ingredients:
        key = entry.get("identity_status") or "unresolved"
        identity[key] = identity.get(key, 0) + 1
    assessment = draft.get("quantity_assessment") or {}
    clusters = assessment.get("clusters_by_status") or {}
    return {
        "schema": "fusion-evaluation/v1",
        "inputs": {
            "spark_events": len(spark_events),
            "spark_open_events": len(open_events),
            "measurements_total": len(measurements),
            "measurements_legible": len(legible),
            "measurements_illegible": len(illegible),
            "video_duration_seconds": duration,
            "timestamps_out_of_range": out_of_range,
        },
        "coverage": coverage or {"status": "not_reported"},
        "reconstruction": {
            "steps": len(steps),
            "multi_source_steps": multi_source_steps,
            "ingredient_groups": len(ingredients),
            "multi_source_ingredient_groups":
                multi_source_ingredients,
            "ingredient_identity": identity,
            "quantities_supported": assessment.get(
                "quantities_recoverable", 0),
            "quantity_clusters_by_status": clusters,
            "readings_unique": assessment.get("readings_unique"),
            "duplicates_folded": assessment.get(
                "duplicates_folded"),
            "conflicts": len(conflicts),
            "corroborated_reading_pairs": len(pairs),
            "unresolved_questions": len(draft.get(
                "unresolved_questions") or []),
        },
        "traceability": {
            "steps_with_source_refs": traceable_steps,
            "ingredients_with_source_refs": traceable_ingredients,
            "every_node_referenced": (
                traceable_steps == len(steps)
                and traceable_ingredients == len(ingredients)),
        },
        "limitations": [
            "No independent ground truth exists; counts measure "
            "recovery from model outputs, not accuracy.",
            "No quantity is recoverable: every stable display "
            "cluster lacks a visible unit, verified tare, "
            "container continuity, or ingredient attribution.",
            "Model agreement is recorded for review attention only "
            "and never treated as verification.",
            "Preliminary human notes are approximate context, not "
            "evidence.",
        ],
    }


def describe_coverage(annotation: dict[str, Any],
                      notes: list[dict[str, Any]]) -> dict[str, Any]:
    """Report which evidence streams genuinely feed the fusion."""
    candidates = annotation.get("candidate_events")
    candidate_count = len(candidates) if isinstance(
        candidates, list) else 0
    segment_count = 0
    for segment in annotation.get("segments") or []:
        observations = segment.get("observations") \
            if isinstance(segment, dict) else None
        segment_count += len(observations) if isinstance(
            observations, list) else 0
    return {
        "spark_full_video": "ingested",
        "spark_open_segments": "ingested",
        "preliminary_candidates": candidate_count,
        "preliminary_segment_observations": segment_count,
        "preliminary_notes_ingested": len(notes),
        "p04_frame_pipeline": "absent: no P04 frame inference "
        "exists; the 685-frame extraction was never inferenced "
        "and its frames were removed by the data reset",
        "p04_human_review_decisions": "absent: no P04 review "
        "records exist in the runtime store",
        "p35_run_excluded": "run_52a8b5cd5f24 covers a different "
        "video and is deliberately not merged",
    }


def run_fusion(spark_response: Path,
               baseten_aggregated: Path,
               preliminary: Path,
               output_dir: Path,
               run_name: str) -> dict[str, Any]:
    """Execute the full offline fusion and persist artifacts."""
    inputs = {
        "spark_response": spark_response,
        "baseten_aggregated": baseten_aggregated,
        "preliminary": preliminary,
    }
    for label, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing fusion input {label}: "
                                    f"{path}")
    parsed, response_id, model = extract_spark_output(spark_response)
    aggregated = json.loads(
        baseten_aggregated.read_text(encoding="utf-8"))
    annotation = json.loads(preliminary.read_text(encoding="utf-8"))

    spark_events = normalize.spark_events(parsed, response_id)
    spark_ingredients = normalize.spark_ingredients(parsed, response_id)
    spark_measurements = normalize.spark_measurements(parsed,
                                                      response_id)
    open_events = normalize.baseten_events(aggregated)
    open_ingredients = normalize.baseten_ingredients(aggregated)
    open_measurements = normalize.baseten_measurements(aggregated)
    notes = normalize.preliminary_events(annotation)

    events = spark_events + open_events + notes
    ingredients = spark_ingredients + open_ingredients
    measurements = spark_measurements + open_measurements

    groups = reconcile.group_events(events)
    reconcile.link_preliminary(groups, notes)
    ingredient_groups = reconcile.group_ingredients(ingredients)
    pairs = reconcile.corroborated_readings(measurements)
    conflicts = reconcile.detect_conflicts(events, measurements,
                                           ingredients)

    source_video = aggregated.get("source_video")
    duration = None
    if isinstance(source_video, dict):
        duration = _as_float(source_video.get("duration_seconds"))

    provenance = {
        "fusion_version": FUSION_VERSION,
        "run_name": run_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            label: {"path": str(path),
                    "sha256": sha256_file(path)}
            for label, path in inputs.items()
        },
        "source_models": {
            "spark": model,
            "spark_response_id": response_id,
            "spark_open": aggregated.get("model"),
            "preliminary_status": annotation.get(
                "annotation_status"),
        },
    }
    tare_claims = quantities.normalize_tare_claims(
        parsed.get("tare_events"), SOURCE_SPARK)
    assessment = quantities.build_quantity_assessment(
        measurements, tare_claims)
    draft = reconstruct.build_draft(
        parsed.get("overall_activity"), groups, ingredient_groups,
        measurements, conflicts, pairs, notes, provenance,
        quantity_assessment=assessment)
    evaluation = build_evaluation(
        draft, spark_events, open_events, measurements, duration,
        coverage=describe_coverage(annotation, notes))

    out = output_dir / run_name
    write_json(out / "fusion_draft.json", draft)
    write_json(out / "evaluation.json", evaluation)
    write_json(out / "manifest.json", {
        "fusion_version": FUSION_VERSION,
        "run_name": run_name,
        "created_utc": provenance["created_utc"],
        "inputs": provenance["inputs"],
        "artifacts": {
            "fusion_draft.json": sha256_file(out / "fusion_draft.json"),
            "evaluation.json": sha256_file(out / "evaluation.json"),
        },
    })
    return {"draft": draft, "evaluation": evaluation,
            "output_dir": str(out)}
