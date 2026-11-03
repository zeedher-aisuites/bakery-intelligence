"""Durable chef recipe-review persistence.

State lives at ``recipe_reviews/<run_id>.json`` under the
storage root, outside run artifacts and work files, so original
model evidence stays byte-identical. Corrections create a new
draft version; approval appends an immutable snapshot (content
plus SHA-256) and a canonical ``ReviewDecision``. Snapshots are
never mutated by later edits.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from bakery_intelligence.api.jobs import utc_now_iso
from bakery_intelligence.api.storage import validate_run_id

RECIPE_REVIEW_VERSION = "recipe-review/1"
RECIPE_REVIEWS_DIRNAME = "recipe_reviews"
PROVENANCES = ("ai-extracted", "human-corrected", "chef-supplied")
DECISIONS = ("approved", "rejected", "needs_revision")


REVIEW_STATUSES = ("pending", "reviewed", "not_observed")


def _slug(value: str, fallback: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in "-_" else "_"
        for ch in value.lower()).strip("_")
    return cleaned or fallback


def normalize_ingredients(items: list[dict[str, Any]],
                          ) -> list[dict[str, Any]]:
    """Assign stable ids and temporal defaults (backward compatible)."""
    normalized = []
    for position, item in enumerate(items):
        row = copy.deepcopy(item)
        name = str(row.get("name") or row.get("ingredient_id") or "")
        row["ingredient_id"] = _slug(
            str(row.get("ingredient_id") or name),
            f"ingredient_{position + 1}")
        row.setdefault("name", name or row["ingredient_id"])
        row.setdefault("step_refs", [])
        row.setdefault("evidence_refs", [])
        row.setdefault("provenance", "chef-supplied")
        row.setdefault("review_status", "pending")
        normalized.append(row)
    return normalized


def normalize_steps(items: list[dict[str, Any]],
                    ) -> list[dict[str, Any]]:
    """Assign stable ids and explicit temporal defaults."""
    normalized = []
    for position, item in enumerate(items):
        row = copy.deepcopy(item)
        row["step_id"] = str(
            row.get("step_id") or f"step_{position + 1}")
        row["index"] = position
        row.setdefault("start_seconds", None)
        row.setdefault("end_seconds", None)
        row.setdefault("frame_refs", [])
        row.setdefault("ingredient_refs", [])
        row["usages"] = normalize_usages(row.get("usages"))
        row.setdefault("provenance", "chef-supplied")
        row.setdefault("review_status", "pending")
        row.setdefault("uncertainty", None)
        row.setdefault("not_observed", False)
        row.setdefault("correction", None)
        normalized.append(row)
    return normalized


def normalize_usages(items: Any) -> list[dict[str, Any]]:
    """Normalize per-step ingredient usages (backward compatible).

    Each usage references one summary ingredient and optionally
    carries a chef-supplied quantity and unit. Unknown quantities
    and units stay explicit nulls; nothing is invented. AI
    readings live in claims, never in usages: a missing
    provenance on this chef-edit path defaults to
    chef-supplied, and ai_observed is not an accepted value.
    """
    if items is None:
        return []
    if not isinstance(items, list):
        raise RecipeReviewError("Ingredient usages must be a list.")
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            raise RecipeReviewError("Ingredient usages must be objects.")
        row = copy.deepcopy(item)
        ingredient_id = str(row.get("ingredient_id") or "").strip()
        if not ingredient_id:
            raise RecipeReviewError(
                "Ingredient usages require an ingredient.")
        row["ingredient_id"] = ingredient_id
        quantity = row.get("quantity")
        if quantity is None:
            quantity = {}
        if not isinstance(quantity, dict):
            raise RecipeReviewError(
                "Ingredient usage quantities must be objects.")
        value = quantity.get("value")
        if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise RecipeReviewError(
                "Ingredient quantities must be non-negative numbers "
                "or unknown.")
        unit = quantity.get("unit")
        if unit is not None:
            if not isinstance(unit, str):
                raise RecipeReviewError(
                    "Ingredient units must be text or unknown.")
            unit = unit.strip() or None
        provenance = row.get("provenance") or "chef-supplied"
        if provenance not in PROVENANCES:
            raise RecipeReviewError(
                f"Invalid provenance: {provenance!r}.")
        row["quantity"] = {
            "value": value,
            "unit": unit,
            "derivation": quantity.get("derivation") or "unknown",
        }
        row["provenance"] = provenance
        normalized.append(row)
    return normalized


def validate_temporal(items: list[dict[str, Any]], label: str) -> None:
    """Validate intervals, references, and review states in steps."""
    for item in items:
        start, end = item.get("start_seconds"), item.get("end_seconds")
        for value in (start, end):
            if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or value < 0):
                raise RecipeReviewError(
                    f"{label} timestamps must be non-negative numbers.")
        if start is not None and end is not None and end < start:
            raise RecipeReviewError(
                f"{label} interval ends before it starts.")
        if item.get("review_status", "pending") not in REVIEW_STATUSES:
            raise RecipeReviewError(
                f"Invalid review_status: {item.get('review_status')!r}.")
        for ref in item.get("frame_refs", []):
            frame_index = ref.get("frame_index") if isinstance(
                ref, dict) else None
            if not isinstance(frame_index, int) or frame_index < 0:
                raise RecipeReviewError(
                    f"{label} frame references require a frame_index.")


class RecipeReviewError(ValueError):
    """A recipe review request or stored state is invalid."""


def _content_sha(payload: Any) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, default=str).encode()).hexdigest()


class RecipeReviewStore:
    """Atomic JSON persistence for one run's recipe review."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def _path(self, run_id: str) -> Path:
        validate_run_id(run_id)
        return self._root / RECIPE_REVIEWS_DIRNAME / f"{run_id}.json"

    def load(self, run_id: str) -> dict[str, Any] | None:
        path = self._path(run_id)
        if not path.is_file():
            return None
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RecipeReviewError(
                f"Recipe review for {run_id} is unreadable: {exc}")
        if not isinstance(state, dict):
            raise RecipeReviewError(
                f"Recipe review for {run_id} is malformed.")
        return state

    def save(self, state: dict[str, Any]) -> dict[str, Any]:
        path = self._path(str(state["run_id"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n")
        tmp.replace(path)
        return state

    def create(self, *, run_id: str, draft_id: str,
               claims: list[dict[str, Any]],
               verification: dict[str, Any],
               simulation_note: str) -> dict[str, Any]:
        """Create version 1 from debated claims, or return existing."""
        existing = self.load(run_id)
        if existing is not None:
            return existing
        now = utc_now_iso()
        state = {
            "schema_version": RECIPE_REVIEW_VERSION,
            "run_id": run_id,
            "draft_id": draft_id,
            "version": 1,
            "status": "pending_review",
            "simulation": True,
            "simulation_note": simulation_note,
            "ingredients": [],
            "steps": [],
            "claims": claims,
            "unresolved_questions": [
                claim for claim in claims
                if claim.get("status") != "supported"],
            "verification": verification,
            "decisions": [],
            "approved_snapshots": [],
            "history": [{
                "version": 1, "action": "created",
                "at": now, "actor": "deterministic-evidence",
            }],
            "created_at": now,
            "updated_at": now,
        }
        return self.save(state)

    def save_corrections(self, run_id: str,
                         ingredients: list[dict[str, Any]],
                         steps: list[dict[str, Any]],
                         reviewer: str) -> dict[str, Any]:
        """Persist chef edits as a new draft version."""
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise RecipeReviewError(
                "A reviewer identity is required for corrections.")
        state = self.load(run_id)
        if state is None:
            raise RecipeReviewError(f"Unknown recipe review: {run_id}.")
        for item in [*ingredients, *steps]:
            if not isinstance(item, dict):
                raise RecipeReviewError("Corrections must be objects.")
            if item.get("provenance", "chef-supplied") not in PROVENANCES:
                raise RecipeReviewError(
                    f"Invalid provenance: {item.get('provenance')!r}.")
        ingredients = normalize_ingredients(ingredients)
        steps = normalize_steps(steps)
        validate_temporal(steps, "Step")
        known = {item["ingredient_id"] for item in ingredients}
        for step in steps:
            refs = list(step.get("ingredient_refs", []))
            refs.extend(usage["ingredient_id"]
                        for usage in step.get("usages", []))
            missing = [ref for ref in refs if ref not in known]
            if missing:
                raise RecipeReviewError(
                    f"Step references unknown ingredient(s): {missing}.")
        # The summary stays consistent with step usage without
        # double-counting: quantities live only on usages, and each
        # ingredient lists the union of linked steps.
        using: dict[str, set[str]] = {}
        for step in steps:
            refs = set(step.get("ingredient_refs", []))
            refs.update(usage["ingredient_id"]
                        for usage in step.get("usages", []))
            for ref in refs:
                using.setdefault(ref, set()).add(step["step_id"])
        for item in ingredients:
            item["step_refs"] = sorted(
                set(item.get("step_refs", []))
                | using.get(item["ingredient_id"], set()))
        state["ingredients"] = ingredients
        state["steps"] = steps
        state["version"] = int(state["version"]) + 1
        state["status"] = "pending_review"
        state["updated_at"] = utc_now_iso()
        state["history"].append({
            "version": state["version"], "action": "corrected",
            "at": state["updated_at"], "actor": reviewer.strip(),
        })
        return self.save(state)

    def decide(self, run_id: str, decision: str, reviewer: str,
               notes: str | None = None) -> dict[str, Any]:
        """Record a canonical decision; approval snapshots content."""
        if decision not in DECISIONS:
            raise RecipeReviewError(
                f"decision must be one of {', '.join(DECISIONS)}.")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise RecipeReviewError(
                "A reviewer identity is required for a decision.")
        state = self.load(run_id)
        if state is None:
            raise RecipeReviewError(f"Unknown recipe review: {run_id}.")
        if decision == "approved":
            verdict = (state.get("verification") or {}).get("verdict")
            if verdict == "fail":
                raise RecipeReviewError(
                    "Approval requires a verification report that "
                    "does not fail (gate G3).")
        now = utc_now_iso()
        record = {
            "schema_version": "1.0.0",
            "decision_id": f"decision_{_content_sha([run_id, state['version'], decision, reviewer, now])[:12]}",
            "draft_id": state["draft_id"],
            "draft_version": state["version"],
            "reviewer_id": reviewer.strip(),
            "decision": decision,
            "notes": notes,
            "corrections_ref": f"recipe_reviews/{run_id}.json@v{state['version']}",
            "decided_at": now,
        }
        from bakery_intelligence.models.review_decision import (
            ReviewDecision,
        )
        ReviewDecision(**record)
        state["decisions"].append(record)
        if decision == "approved":
            snapshot_content = {
                "draft_id": state["draft_id"],
                "version": state["version"],
                "ingredients": copy.deepcopy(state["ingredients"]),
                "steps": copy.deepcopy(state["steps"]),
                "claims": copy.deepcopy(state["claims"]),
            }
            state["approved_snapshots"].append({
                **snapshot_content,
                "sha256": _content_sha(snapshot_content),
                "decision_id": record["decision_id"],
                "reviewer_id": record["reviewer_id"],
                "decided_at": now,
            })
            state["status"] = "approved"
        state["updated_at"] = now
        state["history"].append({
            "version": state["version"], "action": decision,
            "at": now, "actor": reviewer.strip(),
        })
        return self.save(state)
