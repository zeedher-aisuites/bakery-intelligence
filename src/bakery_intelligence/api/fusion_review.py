"""Chef review data access for offline fusion drafts (Phase 2B).

Serves read-only fusion drafts (``runs_api/fusion/<run>/``) and
persists chef decisions separately in
``runs_api/fusion_reviews/<run>.json``. Draft artifacts are never
modified: decisions reference draft node ids and live outside the
fusion output tree, so evidence stays byte-identical.

Run names must match ``p04_104_v<digits>`` exactly, which keeps
every file access inside the storage root by construction. The
reviewable video is one allowlisted repository encode, served
read-only; the endpoint reports 404 when the file is absent.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from bakery_intelligence.api.jobs import utc_now_iso

FUSION_RUN_PATTERN = re.compile(
    r"(p04_104_v[0-9]+|bakery_demo_v[0-9]+)")
FUSION_DIRNAME = "fusion"
FUSION_REVIEWS_DIRNAME = "fusion_reviews"
FUSION_REVIEW_SCHEMA = "fusion-review-decisions/1"

# Browser-playable H.264 review encodes, relative to the repo
# root, keyed by (run-pattern, clip or None for single-video).
REVIEW_VIDEOS: dict[tuple[str, str | None], str] = {
    ("p04_104", None): (
        "experiments/native_video_understanding/artifacts/"
        "p04_104_720p_crf23.mp4"),
    ("bakery_demo", "clip_01_preparation"): (
        "data/raw/bakery_demo/"
        "bakery_demo_01_preparation_with_audio.mp4"),
    ("bakery_demo", "clip_02_mixing"): (
        "data/raw/bakery_demo/"
        "bakery_demo_02_mixing_with_audio.mp4"),
}

TARGET_INGREDIENT = "ingredient"
TARGET_CLUSTER = "quantity_cluster"
TARGET_STEP = "step"
TARGET_CONFLICT = "conflict"
TARGET_QUESTION = "question"
TARGET_TYPES = (
    TARGET_INGREDIENT,
    TARGET_CLUSTER,
    TARGET_STEP,
    TARGET_CONFLICT,
    TARGET_QUESTION,
)

ACTION_CONFIRM = "confirm"
ACTION_CORRECT = "correct"
ACTION_SUPPLY = "supply"
ACTION_CANNOT_DETERMINE = "cannot_determine"
ACTION_REJECT = "reject"
ACTIONS = (
    ACTION_CONFIRM,
    ACTION_CORRECT,
    ACTION_SUPPLY,
    ACTION_CANNOT_DETERMINE,
    ACTION_REJECT,
)
# Actions that must carry a chef value; the rest must not.
VALUE_ACTIONS = (ACTION_CORRECT, ACTION_SUPPLY)

MAX_TEXT_CHARS = 500
MAX_REVIEWER_CHARS = 64
MAX_NOTES_CHARS = 1000


class FusionReviewError(ValueError):
    """A fusion review request or stored record is invalid."""


def validate_fusion_run(run_name: str) -> str:
    """Accept a fusion run name or raise."""
    if not isinstance(run_name, str):
        raise FusionReviewError("Run name must be a string.")
    if FUSION_RUN_PATTERN.fullmatch(run_name) is None:
        raise FusionReviewError(
            f"Unknown fusion run: {run_name!r}.")
    return run_name


def draft_dir(root: Path, run_name: str) -> Path:
    """Resolve the read-only draft directory inside the root."""
    validate_fusion_run(run_name)
    return root / FUSION_DIRNAME / run_name


def read_fusion_draft(root: Path,
                      run_name: str) -> dict[str, Any]:
    """Return the draft and evaluation payloads (read-only).

    Raises FusionReviewError when the run is unknown and
    FileNotFoundError when its artifacts are absent.
    """
    directory = draft_dir(root, run_name)
    try:
        draft = json.loads(
            (directory / "fusion_draft.json").read_text(
                encoding="utf-8"))
        evaluation = json.loads(
            (directory / "evaluation.json").read_text(
                encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Fusion run has no draft artifacts: "
            f"{run_name}.") from exc
    except (OSError, ValueError) as exc:
        raise FusionReviewError(
            f"Fusion draft for {run_name} is unreadable: "
            f"{exc}.")
    if not isinstance(draft, dict) or not isinstance(
            evaluation, dict):
        raise FusionReviewError(
            f"Fusion draft for {run_name} is malformed.")
    return {"run_name": run_name, "draft": draft,
            "evaluation": evaluation}


def _run_family(run_name: str) -> str:
    if run_name.startswith("p04_104_v"):
        return "p04_104"
    return "bakery_demo"


def review_video_path(root: Path, run_name: str,
                      clip_id: str | None = None) -> Path | None:
    """Resolve the allowlisted review video, or None if absent."""
    validate_fusion_run(run_name)
    family = _run_family(run_name)
    if family == "p04_104":
        clip_id = None
    rel = REVIEW_VIDEOS.get((family, clip_id))
    if rel is None:
        return None
    repo = root.resolve().parent
    candidate = (repo / rel).resolve()
    try:
        candidate.relative_to(repo)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def session_clips(root: Path,
                  run_name: str) -> list[dict[str, str]]:
    """List reviewable clips for a run (empty for P04)."""
    try:
        payload = read_fusion_draft(root, run_name)
    except (FusionReviewError, FileNotFoundError):
        return []
    session = payload["draft"].get("session") or {}
    clips = session.get("clips") or []
    return [{"clip_id": clip.get("clip_id"),
             "order": clip.get("order"),
             "duration_seconds": clip.get("duration_seconds")}
            for clip in clips if clip.get("clip_id")]


def _id_set(items: Any, key: str) -> set[str]:
    if not isinstance(items, list):
        return set()
    return {
        str(item.get(key)) for item in items
        if isinstance(item, dict) and item.get(key)}


def known_targets(draft: dict[str, Any]) -> dict[str, set[str]]:
    """Collect reviewable node ids from a draft payload."""
    assessment = draft.get("quantity_assessment")
    clusters = assessment.get("stable_clusters") \
        if isinstance(assessment, dict) else None
    questions = draft.get("unresolved_questions")
    return {
        TARGET_INGREDIENT: _id_set(
            draft.get("ingredients"), "group_id"),
        TARGET_STEP: _id_set(draft.get("steps"), "group_id"),
        TARGET_CONFLICT: _id_set(
            draft.get("conflicts"), "conflict_id"),
        TARGET_CLUSTER: _id_set(clusters, "cluster_id"),
        TARGET_QUESTION: {
            f"q{index}" for index in range(len(questions))}
        if isinstance(questions, list) else set(),
    }


def _reviews_path(root: Path, run_name: str) -> Path:
    validate_fusion_run(run_name)
    return root / FUSION_REVIEWS_DIRNAME / f"{run_name}.json"


def read_decisions(root: Path,
                   run_name: str) -> dict[str, Any]:
    """Return stored decisions keyed by 'type:id'.

    A missing file means no decisions yet, never an error.
    """
    path = _reviews_path(root, run_name)
    if not path.is_file():
        return {
            "schema_version": FUSION_REVIEW_SCHEMA,
            "run_name": run_name,
            "decisions": {},
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FusionReviewError(
            f"Stored decisions for {run_name} are unreadable: "
            f"{exc}.")
    if not isinstance(payload, dict) or not isinstance(
            payload.get("decisions"), dict):
        raise FusionReviewError(
            f"Stored decisions for {run_name} are malformed.")
    return {
        "schema_version": payload.get(
            "schema_version", FUSION_REVIEW_SCHEMA),
        "run_name": run_name,
        "decisions": payload["decisions"],
    }


def _validate_text(label: str, value: Any,
                   limit: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise FusionReviewError(f"{label} must be a string.")
    if len(value) > limit:
        raise FusionReviewError(
            f"{label} exceeds {limit} characters.")
    return value


def _validate_value(target_type: str, action: str,
                    value: Any) -> dict[str, Any] | None:
    if action not in VALUE_ACTIONS:
        if value is not None:
            raise FusionReviewError(
                f"Only {', '.join(VALUE_ACTIONS)} may carry a "
                f"value, got {value!r} with {action!r}.")
        return None
    if not isinstance(value, dict):
        raise FusionReviewError(
            f"A {action!r} decision requires a value object.")
    if target_type == TARGET_INGREDIENT:
        name = _validate_text("value.name", value.get("name"),
                              MAX_TEXT_CHARS)
        if not name or not name.strip():
            raise FusionReviewError(
                "An ingredient correction needs a "
                "non-empty value.name.")
        return {"name": name.strip()}
    if target_type == TARGET_CLUSTER:
        unit = _validate_text("value.unit", value.get("unit"),
                              MAX_TEXT_CHARS)
        amount = value.get("amount")
        if amount is not None and (
                isinstance(amount, bool)
                or not isinstance(amount, (int, float))
                or not math.isfinite(amount)):
            raise FusionReviewError(
                "value.amount must be a finite number.")
        if (unit is None or not unit.strip()) and amount is None:
            raise FusionReviewError(
                "A measurement decision needs a unit, an "
                "amount, or both.")
        result: dict[str, Any] = {}
        if unit is not None and unit.strip():
            result["unit"] = unit.strip()
        if amount is not None:
            result["amount"] = float(amount)
        return result
    raise FusionReviewError(
        f"Target {target_type!r} does not accept a value.")


def save_decision(root: Path, run_name: str,
                  target_type: str, target_id: str,
                  action: str, value: Any, reviewer: Any,
                  notes: Any,
                  known: dict[str, set[str]]) -> dict[str, Any]:
    """Validate and upsert one chef decision, then persist it.

    One decision exists per target: saving replaces any prior
    record for that target, so contradictory states cannot
    accumulate. Correction never implies verification; the
    provenance mapping lives with the caller.
    """
    if target_type not in TARGET_TYPES:
        raise FusionReviewError(
            f"target_type must be one of "
            f"{', '.join(TARGET_TYPES)}, got {target_type!r}.")
    if not isinstance(target_id, str) or not target_id:
        raise FusionReviewError("target_id must be a string.")
    if target_id not in known.get(target_type, set()):
        raise FusionReviewError(
            f"Unknown {target_type} target: {target_id!r}.")
    if action not in ACTIONS:
        raise FusionReviewError(
            f"action must be one of {', '.join(ACTIONS)}, got "
            f"{action!r}.")
    clean_value = _validate_value(target_type, action, value)
    reviewer_name = _validate_text(
        "reviewer", reviewer, MAX_REVIEWER_CHARS)
    if not reviewer_name or not reviewer_name.strip():
        raise FusionReviewError(
            "A decision requires a reviewer identity.")
    clean_notes = _validate_text("notes", notes, MAX_NOTES_CHARS)
    stored = read_decisions(root, run_name)
    decision = {
        "target_type": target_type,
        "target_id": target_id,
        "action": action,
        "value": clean_value,
        "reviewer": reviewer_name.strip(),
        "notes": clean_notes,
        "updated_at": utc_now_iso(),
        "schema_version": FUSION_REVIEW_SCHEMA,
    }
    stored["decisions"][f"{target_type}:{target_id}"] = decision
    path = _reviews_path(root, run_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(stored, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    tmp.replace(path)
    return decision
