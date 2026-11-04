"""Automatic first-recipe proposals for new video analyses.

Builds a compact evidence bundle from a completed run's
persisted artifacts (no demo content, no re-inference),
validates a model's structured proposal against that bundle,
and assembles an editor-compatible snapshot. Seen-video
quantities must match recorded readings; everything else is
an AI estimate or stays null.
"""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any

from bakery_intelligence.phase2c import prompts

ESTIMATE_USD = 0.30
PROPOSAL_SCHEMA = "recipe-proposal/1"
SNAPSHOT_VERSION = "recipe-snapshot/4"

MAX_READINGS = 40
MAX_OBJECTS = 40
MAX_CONTAINERS = 20
MAX_TARE_EVENTS = 20
MAX_INGREDIENTS = 40
MAX_STEPS = 60
MAX_NOTES = 15
MAX_SUGGESTIONS = 8

QTY_TOLERANCE = 1.0

GENERIC_OBJECTS = {
    "kitchen scale", "scale", "digital scale", "table",
    "countertop", "kitchen counter",
}


class ProposalError(ValueError):
    """Evidence is missing or a proposal is invalid."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProposalError(f"Missing evidence file: {path}.") from exc
    except (OSError, ValueError) as exc:
        raise ProposalError(
            f"Unreadable evidence file: {path}.") from exc


def _clean(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def build_evidence_bundle(run_dir: Path, recipe_name: str,
                          language: str) -> dict[str, Any]:
    """Assemble the capped, JSON-serializable model input."""
    artifacts = run_dir / "artifacts"
    summary = _read_json(artifacts / "pipeline_summary.json")
    measurements = _read_json(artifacts / "measurements.json")
    draft = _read_json(artifacts / "recipe_draft.json")
    weighing = _read_json(artifacts / "weighing_event.json")
    if not isinstance(summary, dict) or not isinstance(
            measurements, dict):
        raise ProposalError("Pipeline artifacts are malformed.")

    readings: list[dict[str, Any]] = []
    for item in measurements.get("measurements") or []:
        if not isinstance(item, dict):
            continue
        if item.get("value_status") != "ok":
            continue
        value = item.get("value")
        if isinstance(value, bool) or not isinstance(
                value, (int, float)) or not math.isfinite(value):
            continue
        readings.append({
            "t": item.get("timestamp_seconds"),
            "value": float(value),
            "unit": item.get("unit_observed"),
        })
    readings.sort(key=lambda r: (
        r["t"] if isinstance(r["t"], (int, float)) else 1e12))
    usable_total = len(readings)
    readings = readings[:MAX_READINGS]

    objects: dict[str, dict[str, Any]] = {}
    containers: dict[str, float | None] = {}
    tare_events: list[float | None] = []
    obs_dir = run_dir / "work" / "firstpass" / "observations"
    if obs_dir.is_dir():
        for path in sorted(obs_dir.glob("*.json")):
            try:
                batch = json.loads(
                    path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(batch, list):
                continue
            for obs in batch:
                if not isinstance(obs, dict):
                    continue
                fields = obs.get("model_fields")
                if not isinstance(fields, dict):
                    continue
                stamp = obs.get("timestamp_seconds")
                for raw in fields.get("visible_objects") or []:
                    label = _clean(raw).lower()
                    if not label or label in GENERIC_OBJECTS:
                        continue
                    entry = objects.setdefault(
                        label, {"object": _clean(raw),
                                "first_t": stamp, "count": 0})
                    entry["count"] += 1
                container = _clean(
                    fields.get("container_description"))
                if container and container.lower() not in containers:
                    containers[container.lower()] = stamp
                if fields.get("tare_action_visible") is True:
                    tare_events.append(stamp)
    object_list = sorted(
        objects.values(),
        key=lambda e: (-e["count"], str(e["first_t"])),
    )[:MAX_OBJECTS]
    container_list = [{
        "description": desc, "t": stamp,
    } for desc, stamp in list(containers.items())
        [:MAX_CONTAINERS]]

    steps = []
    if isinstance(draft, dict):
        for item in draft.get("steps") or []:
            if isinstance(item, dict) and _clean(
                    item.get("description")):
                steps.append(_clean(item.get("description"))[:300])
    unresolved = []
    if isinstance(draft, dict):
        for item in draft.get("unresolved_questions") or []:
            if isinstance(item, dict) and _clean(item.get("code")):
                unresolved.append(_clean(item.get("code"))[:80])

    video = summary.get("video") if isinstance(summary, dict) \
        else None
    multimodal: dict[str, Any] = {}
    for name in ("multimodal_global.json",
                 "multimodal_speech_claims.json",
                 "multimodal_segments_index.json",
                 "multimodal_validation.json"):
        path = artifacts / name
        if not path.is_file():
            continue
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        key = name[len("multimodal_"):-len(".json")]
        multimodal[key] = stored
    segments: list[dict[str, Any]] = []
    for path in sorted(artifacts.glob("multimodal_segment_*.json")):
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        segments.append({
            "index": stored.get("index"),
            "start_seconds": stored.get("start_seconds"),
            "end_seconds": stored.get("end_seconds"),
            "request_id": stored.get("request_id"),
            "result": stored.get("result"),
        })
    if segments:
        multimodal["segments"] = segments
    return {
        "recipe_name": recipe_name,
        "language": language,
        "video": video if isinstance(video, dict) else {},
        "readings": readings,
        "readings_total": len(readings),
        "readings_usable_total": usable_total,
        "readings_truncated": usable_total > len(readings),
        "multimodal": multimodal,
        "weighing": {
            "tare_observed": (weighing.get("tare_observed")
                              if isinstance(weighing, dict)
                              else None),
            "notes": _clean((weighing or {}).get(
                "review_notes"))[:300]
            if isinstance(weighing, dict) else "",
        },
        "visible_objects": object_list,
        "containers": container_list,
        "tare_events": tare_events[:MAX_TARE_EVENTS],
        "draft_steps": steps[:10],
        "unresolved": unresolved[:10],
    }


def _numbers_close(value: float, target: float) -> bool:
    return abs(value - target) <= max(
        QTY_TOLERANCE, abs(target) * 0.02)


def validate_proposal(bundle: dict[str, Any],
                      parsed: Any) -> dict[str, Any]:
    """Enforce shape, provenance, and reading-grounded quantities."""
    violations = prompts.validate_output(prompts.RECIPE_PROPOSAL,
                                         parsed)
    if violations:
        raise ProposalError(
            f"Invalid proposal payload: {'; '.join(violations)}")
    assert isinstance(parsed, dict)
    meaningful = parsed.get("meaningful")
    if meaningful is not True and meaningful is not False:
        raise ProposalError("Proposal 'meaningful' must be boolean.")
    reason = _clean(parsed.get("reason"))
    if not reason:
        raise ProposalError("Proposal needs a reason.")
    ingredients = parsed.get("ingredients")
    method = parsed.get("method")
    if not isinstance(ingredients, list) or not isinstance(
            method, list):
        raise ProposalError(
            "Proposal ingredients and method must be lists.")
    if not meaningful:
        if ingredients or method:
            raise ProposalError(
                "A non-meaningful proposal must leave ingredients "
                "and method empty.")
        return parsed
    title = _clean(parsed.get("title"))
    if not title or len(title) > 120:
        raise ProposalError(
            "Proposal needs a title of 1-120 characters.")
    if not ingredients or len(ingredients) > MAX_INGREDIENTS:
        raise ProposalError(
            f"Proposal needs 1-{MAX_INGREDIENTS} ingredients.")
    readings = [r["value"] for r in bundle.get("readings") or []
                if isinstance(r.get("value"), (int, float))]
    seen_ids: set[str] = set()
    for position, row in enumerate(ingredients):
        if not isinstance(row, dict):
            raise ProposalError(
                f"Ingredient row {position} is not an object.")
        name = _clean(row.get("name"))
        if not name or len(name) > 120:
            raise ProposalError(
                f"Ingredient row {position} needs a name.")
        key = name.lower()
        if key in seen_ids:
            raise ProposalError(
                f"Duplicate ingredient name: {name!r}.")
        seen_ids.add(key)
        quantity = row.get("quantity")
        if quantity is not None and (
                isinstance(quantity, bool)
                or not isinstance(quantity, (int, float))
                or not math.isfinite(quantity) or quantity < 0):
            raise ProposalError(
                f"Ingredient {name!r} has an invalid quantity.")
        unit = row.get("unit")
        if unit is not None and (
                not isinstance(unit, str) or not unit.strip()
                or len(unit) > 20):
            raise ProposalError(
                f"Ingredient {name!r} has an invalid unit.")
        provenance = row.get("provenance")
        if provenance not in ("seen_video", "ai_estimated"):
            raise ProposalError(
                f"Ingredient {name!r} has invalid provenance.")
        basis = _clean(row.get("basis"))
        if not basis or len(basis) > 500:
            raise ProposalError(
                f"Ingredient {name!r} needs a basis.")
        if (provenance == "seen_video" and quantity is not None
                and not any(_numbers_close(quantity, target)
                            for target in readings)):
            raise ProposalError(
                f"Ingredient {name!r} claims a seen quantity "
                f"with no matching recorded reading.")
    if not method or len(method) > MAX_STEPS:
        raise ProposalError(
            f"Proposal needs 1-{MAX_STEPS} method steps.")
    for position, item in enumerate(method, start=1):
        if not isinstance(item, dict):
            raise ProposalError(
                f"Method step {position} is not an object.")
        if item.get("number") != position:
            raise ProposalError(
                "Method steps must be numbered sequentially "
                "from 1.")
        text = _clean(item.get("instruction"))
        if not text or len(text) > 2000:
            raise ProposalError(
                f"Step {position} needs an instruction.")
    notes = parsed.get("notes")
    if not isinstance(notes, list) or not notes or len(
            notes) > MAX_NOTES or not all(
            isinstance(n, str) and n.strip() and len(n) <= 500
            for n in notes):
        raise ProposalError(
            "Proposal needs 1-15 non-empty production notes.")
    suggestions = parsed.get("suggestions")
    if not isinstance(suggestions, list) or len(suggestions) > \
            MAX_SUGGESTIONS:
        raise ProposalError("Proposal suggestions are invalid.")
    seen_slugs: set[str] = set()
    for item in suggestions:
        if not isinstance(item, dict):
            raise ProposalError("A suggestion is not an object.")
        slug = _clean(item.get("id"))
        if not slug or len(slug) > 40 or not re.fullmatch(
                r"[a-z0-9][a-z0-9_-]*", slug):
            raise ProposalError(
                "Suggestion ids must be lowercase slugs.")
        if slug in seen_slugs:
            raise ProposalError(
                f"Duplicate suggestion id: {slug!r}.")
        seen_slugs.add(slug)
        for field in ("label", "hint", "placeholder"):
            text = _clean(item.get(field))
            if not text or len(text) > 200:
                raise ProposalError(
                    f"Suggestion {slug!r} needs {field}.")
    return parsed


def assemble_snapshot(bundle: dict[str, Any],
                      parsed: dict[str, Any],
                      meta: dict[str, Any]) -> dict[str, Any]:
    """Build the editor-compatible snapshot from a proposal."""
    ingredients = []
    for row in parsed["ingredients"]:
        ingredients.append({
            "name": _clean(row["name"]),
            "quantity": row.get("quantity"),
            "unit": row.get("unit"),
            "quantity_status": ("seen_video"
                                if row["provenance"] == "seen_video"
                                else "ai_estimated"),
            "quantity_basis": _clean(row["basis"]),
            "review_status": ["unknown"],
            "group_ids": [],
            "display_labels": [_clean(row["name"])],
            "evidence": [],
            "note": None,
        })
    method = []
    for item in parsed["method"]:
        method.append({
            "number": item["number"],
            "clip_id": "uploaded_video",
            "interval": "",
            "instruction": _clean(item["instruction"]),
            "step_group_ids": [],
            "review_status": ["unknown"],
            "caveats": [],
            "ai_guidance": None,
        })
    return {
        "snapshot_version": SNAPSHOT_VERSION,
        "snapshot_id": meta["recipe_id"],
        "run_name": meta["run_id"],
        "language": bundle.get("language") or "en",
        "generated_at": meta["generated_at"],
        "title": _clean(parsed["title"]),
        "title_basis": (
            f"Chef-supplied recipe name: {bundle.get('recipe_name')!r} "
            f"(context only)."),
        "recipe_status": "draft",
        "review_summary": {
            "reviewer": None,
            "ai_executed": True,
            "human_validated": False,
            "decisions": 0,
        },
        "provenance": {
            "builder": meta.get("builder") or "auto-proposal/1",
            "run_id": meta["run_id"],
            "proposal_request_id": meta.get("request_id"),
            "proposal_prompt_version": meta.get(
                "prompt_version")
            or prompts.RECIPE_PROPOSAL.version,
            "proposal_provider": meta.get("provider") or "meta",
            "proposal_model": meta.get("model")
            or prompts.RECIPE_PROPOSAL.model,
            "proposal_actual_usd": meta.get("actual_usd"),
            "readings_used": len(bundle.get("readings") or []),
            "readings_usable_total": bundle.get(
                "readings_usable_total",
                len(bundle.get("readings") or [])),
            "readings_truncated": bool(
                bundle.get("readings_truncated", False)),
        },
        "ingredients": ingredients,
        "excluded_observations": [],
        "method": method,
        "omitted_detail_steps": [],
        "review_flags": [],
        "limitations": [_clean(n) for n in parsed["notes"]],
        "technical_limitations": list(
            bundle.get("unresolved") or []),
        "improvement_suggestions": [{
            "id": s["id"],
            "label": _clean(s["label"]),
            "hint": _clean(s["hint"]),
            "placeholder": _clean(s["placeholder"]),
        } for s in parsed["suggestions"]],
        "open_questions": parsed.get("open_questions") or [],
        "completion_open_items": [],
    }
