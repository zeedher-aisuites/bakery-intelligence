"""Exception-driven observation review tasks and corrections.

Only fields that were actually evaluated can become review
tasks. For the historical measurement-only run the evaluated
field is the scale reading: ingredient, utensil, and action
identities were never evaluated, so they never generate tasks.

Field states: accepted, uncertain, contradictory,
missing_relevant, not_observable, not_applicable, not_evaluated.
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

OBSERVATION_REVIEW_VERSION = "observation-review/1"
OBSERVATION_REVIEWS_DIRNAME = "observation_reviews"

STATE_ACCEPTED = "accepted"
STATE_UNCERTAIN = "uncertain"
STATE_CONTRADICTORY = "contradictory"
STATE_MISSING_RELEVANT = "missing_relevant"
STATE_NOT_OBSERVABLE = "not_observable"
STATE_NOT_APPLICABLE = "not_applicable"
STATE_NOT_EVALUATED = "not_evaluated"

REVIEW_ACTIONS = (
    "supply", "correct", "confirm", "not_identifiable",
    "not_applicable", "skip_unknown", "review_later",
)
#: Actions that must carry a finite numeric reading.
VALUE_ACTIONS = ("supply", "correct")


def _check_human_value(action: str, human_value: Any) -> None:
    """Reject empty or non-numeric values for value-carrying actions."""
    if action in VALUE_ACTIONS:
        if isinstance(human_value, bool) or not isinstance(
                human_value, (int, float)):
            raise ObservationReviewError(
                "A correction needs a finite numeric value.")
        if not math.isfinite(float(human_value)):
            raise ObservationReviewError(
                "A correction needs a finite numeric value.")
    elif human_value is not None:
        raise ObservationReviewError(
            f"Action {action!r} does not take a value.")
RESOLVING_ACTIONS = (
    "supply", "correct", "confirm", "not_identifiable",
    "not_applicable", "skip_unknown",
)

# Configurable eligibility rules for the scale-reading field.
RULES: dict[str, Any] = {
    "field": "scale_reading",
    "group_max_gap_frames": 50,
    "accepted_requires": (
        "valid numeric reading", "scale readable",
        "no contradictory neighbor"),
}


class ObservationReviewError(ValueError):
    """An observation review request is invalid."""


def _is_relevant(frame: dict[str, Any],
                 action_frames: set[int]) -> bool:
    scale = frame.get("scale") or {}
    return (
        frame["frame_index"] in action_frames
        or scale.get("addition_or_removal_visible") in (
            "addition", "removal")
        or scale.get("tare_action_visible") is True
    )


def evaluate_frames(package: dict[str, Any],
                    ) -> list[dict[str, Any]]:
    """Classify every frame's scale-reading field deterministically."""
    action = package.get("action_evidence") or {}
    action_frames = set(action.get("addition_frames") or []) | set(
        action.get("tare_frames") or [])
    evaluated: list[dict[str, Any]] = []
    previous_value: float | None = None
    for frame in package.get("frames", []):
        measurement = frame.get("measurement") or {}
        scale = frame.get("scale") or {}
        value = measurement.get("value")
        relevant = _is_relevant(frame, action_frames)
        if value is not None and scale.get("readable") is not False:
            state = STATE_ACCEPTED
            if (previous_value is not None
                    and value < previous_value - 1e-9
                    and frame["frame_index"] not in (
                        action.get("tare_frames") or [])):
                state = STATE_CONTRADICTORY
            previous_value = float(value)
        elif value is not None:
            state = STATE_UNCERTAIN
            previous_value = float(value)
        elif relevant:
            state = STATE_MISSING_RELEVANT
        elif scale.get("readable") is False:
            state = STATE_NOT_OBSERVABLE
        else:
            state = STATE_NOT_EVALUATED
        evaluated.append({
            "frame_index": frame["frame_index"],
            "timestamp_seconds": frame.get("timestamp_seconds"),
            "field": RULES["field"],
            "state": state,
            "relevant": relevant,
            "value": value,
            # Acceptance is structural (parseable reading, readable
            # display, temporal consistency), not a visual-correctness
            # claim. Medium/low model confidence stays accepted to
            # minimize review, but is flagged for the UI.
            "limited_confidence": (
                value is not None
                and scale.get("confidence") in ("medium", "low")),
        })
    return evaluated


def _evaluated_task_id(package: dict[str, Any], frame_index: int,
                       ) -> str:
    seed = hashlib.sha256(json.dumps(
        [package.get("run_id"), frame_index], sort_keys=True).encode()
    ).hexdigest()[:12]
    return f"frame_{frame_index}_{seed}"


def _culinary_review_tasks(
        culinary: dict[str, Any] | None,
        corrections: dict[str, Any]) -> list[dict[str, Any]]:
    """One review task per problematic culinary event, if any.

    Events group related frames, so the chef reviews a handful of
    event decisions instead of every frame. Culinary states never
    appear in blocking_errors: they record uncertainty, they do
    not gate generation.
    """
    from bakery_intelligence.adversarial.culinary import (
        CULINARY_STATE_ACTION_CONFLICTING,
        CULINARY_STATE_INGREDIENT_UNCERTAIN,
        CULINARY_STATE_TARE_UNRESOLVED,
        CULINARY_STATE_TRANSITION_UNCERTAIN,
        CULINARY_STATE_UNIT_MISSING,
    )
    events = (culinary or {}).get("events") or []
    if not isinstance(events, list):
        return []
    severity = (
        CULINARY_STATE_ACTION_CONFLICTING,
        CULINARY_STATE_TRANSITION_UNCERTAIN,
        CULINARY_STATE_INGREDIENT_UNCERTAIN,
        CULINARY_STATE_UNIT_MISSING,
        CULINARY_STATE_TARE_UNRESOLVED,
    )
    code_to_state = {
        "ACTION_UNKNOWN": None,
        "INGREDIENT_UNKNOWN": CULINARY_STATE_INGREDIENT_UNCERTAIN,
        "INGREDIENT_AMBIGUOUS": CULINARY_STATE_INGREDIENT_UNCERTAIN,
        "TRANSITION_REPORTED": CULINARY_STATE_TRANSITION_UNCERTAIN,
        "TARE_UNRESOLVED": CULINARY_STATE_TARE_UNRESOLVED,
        "SCALE_GAP": None,
    }

    def issues_of(event: dict[str, Any]) -> list[str]:
        codes = [str(item.get("code")) for item in
                 event.get("unresolved_questions", [])
                 if isinstance(item, dict)]
        states = [code_to_state[code] for code in codes
                  if code_to_state.get(code) is not None]
        return sorted(set(states),
                      key=lambda state: severity.index(state))

    ordered = sorted(
        (event for event in events if isinstance(event, dict)),
        key=lambda event: event.get("start_frame", 0))
    previous_operation: str | None = None
    tasks: list[dict[str, Any]] = []
    for event in ordered:
        states = issues_of(event)
        operation = str(event.get("operation") or "unknown_action")
        if (previous_operation is not None
                and operation != previous_operation
                and operation not in ("no_action_visible",)
                and previous_operation not in ("no_action_visible",)):
            states = [CULINARY_STATE_ACTION_CONFLICTING] + [
                state for state in states
                if state != CULINARY_STATE_ACTION_CONFLICTING]
        previous_operation = operation
        scale_values = event.get("scale_values") or []
        if (any(value is not None for value in scale_values)
                and CULINARY_STATE_UNIT_MISSING not in states):
            states.append(CULINARY_STATE_UNIT_MISSING)
        if not states:
            continue
        state = states[0]
        task_id = str(event.get("event_id") or "culinary_unknown")
        correction = corrections.get(task_id)
        if correction is None:
            status = "open"
        elif correction.get("action") == "review_later":
            status = "deferred"
        else:
            status = "resolved"
        candidates = event.get("ingredient_candidates") or []
        names = ", ".join(
            str(item.get("name") or "unnamed")
            for item in candidates if isinstance(item, dict)) or "none"
        tasks.append({
            "task_id": task_id,
            "field": "culinary_event",
            "state": state,
            "frame_refs": list(event.get("frame_refs", [])),
            "start_seconds": event.get("start_seconds"),
            "end_seconds": event.get("end_seconds"),
            "summary": (f"{operation} · ingredients: {names} · "
                        f"uncertainty: {state}"),
            "issues": states,
            "operation": operation,
            "status": status,
            "correction": correction,
        })
    return tasks


def build_review_tasks(package: dict[str, Any],
                       corrections: dict[str, Any] | None = None,
                       classifications: dict[str, Any] | None = None,
                       culinary: dict[str, Any] | None = None,
                       ) -> dict[str, Any]:
    """Group actionable exceptions; accepted frames stay hidden.

    Culinary enrichment stays optional: analyses without culinary
    observations produce exactly the historical queue, and culinary
    uncertainties never block generation.
    """
    corrections = corrections or {}
    frame_by_index = {frame["frame_index"]: frame
                      for frame in package.get("frames", [])}
    classifications = classifications or {}
    evaluated = evaluate_frames(package)
    actionable = [
        item for item in evaluated
        if item["state"] in (STATE_UNCERTAIN, STATE_CONTRADICTORY,
                             STATE_MISSING_RELEVANT)
    ]
    action = package.get("action_evidence") or {}
    tare_frames = set(action.get("tare_frames") or [])
    tasks: list[dict[str, Any]] = []
    group: list[dict[str, Any]] = []

    def flush() -> None:
        if not group:
            return
        first, last = group[0], group[-1]
        seed = hashlib.sha256(json.dumps(
            [package.get("run_id"), first["frame_index"],
             last["frame_index"], first["state"]],
            sort_keys=True).encode()).hexdigest()[:12]
        task_id = f"task_{seed}"
        correction = corrections.get(task_id)
        if correction is None:
            status = "open"
        elif correction.get("action") == "review_later":
            status = "deferred"
        else:
            status = "resolved"
        tasks.append({
            "task_id": task_id,
            "field": RULES["field"],
            "state": first["state"],
            "frame_refs": [item["frame_index"] for item in group],
            "start_seconds": first["timestamp_seconds"],
            "end_seconds": last["timestamp_seconds"],
            "status": status,
            "correction": correction,
        })
        group.clear()

    for item in actionable:
        if group:
            gap = item["frame_index"] - group[-1]["frame_index"]
            crosses_tare = any(
                group[-1]["frame_index"] < tare <= item["frame_index"]
                for tare in tare_frames)
            if (gap > RULES["group_max_gap_frames"]
                    or item["state"] != group[-1]["state"]
                    or crosses_tare):
                flush()
        group.append(item)
    flush()

    for key in sorted(classifications):
        flagged = classifications[key]
        frame_index = flagged.get("frame_index")
        frame = frame_by_index.get(frame_index)
        if frame is None:
            continue
        synthetic_id = _evaluated_task_id(package, frame_index)
        if any(task["task_id"] == synthetic_id for task in tasks):
            continue
        tasks.append({
            "task_id": synthetic_id,
            "field": RULES["field"],
            "state": "flagged_for_review",
            "frame_refs": [frame_index],
            "start_seconds": frame.get("timestamp_seconds"),
            "end_seconds": frame.get("timestamp_seconds"),
            "status": ("resolved"
                       if flagged.get("action") in RESOLVING_ACTIONS
                       else "open"),
            "correction": flagged,
            "from_viewer": True,
        })

    tasks.extend(_culinary_review_tasks(
        culinary, corrections or {}))

    open_tasks = [task for task in tasks if task["status"] == "open"]
    deferred = [task for task in tasks if task["status"] == "deferred"]
    resolved = [task for task in tasks if task["status"] == "resolved"]
    frames_total = len(evaluated)
    usable = sum(1 for item in evaluated
                 if item.get("value") is not None)
    abstained = frames_total - usable
    blocked = [
        task["task_id"] for task in open_tasks + deferred
        if task["state"] == STATE_CONTRADICTORY
    ]
    counts = {
        "frames": frames_total,
        "usable_readings": usable,
        "abstentions": abstained,
        "accepted": sum(1 for item in evaluated
                        if item["state"] == STATE_ACCEPTED),
        "actionable": len(actionable),
        "tasks": len(tasks),
        "open": len(open_tasks),
        "deferred": len(deferred),
        "resolved": len(resolved),
        "explicit_unknowns": sum(
            1 for item in corrections.values()
            if item.get("explicit_unknown")),
        "not_evaluated": sum(1 for item in evaluated
                             if item["state"] == STATE_NOT_EVALUATED),
        "not_observable": sum(1 for item in evaluated
                              if item["state"] == STATE_NOT_OBSERVABLE),
        "blocking_errors": blocked,
    }
    return {"evaluated": evaluated, "tasks": tasks, "counts": counts}


class ObservationReviewStore:
    """Durable per-run observation corrections (outside artifacts)."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def _path(self, run_id: str) -> Path:
        validate_run_id(run_id)
        return self._root / OBSERVATION_REVIEWS_DIRNAME / f"{run_id}.json"

    def load(self, run_id: str) -> dict[str, Any]:
        path = self._path(run_id)
        if not path.is_file():
            return {"schema_version": OBSERVATION_REVIEW_VERSION,
                    "run_id": run_id, "corrections": {},
                    "classifications": {}, "correction_history": {}}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ObservationReviewError(
                f"Observation reviews for {run_id} are unreadable: {exc}")
        if not isinstance(payload, dict) or not isinstance(
                payload.get("corrections"), dict):
            raise ObservationReviewError(
                f"Observation reviews for {run_id} are malformed.")
        payload.setdefault("classifications", {})
        history = payload.setdefault("correction_history", {})
        if not isinstance(history, dict):
            raise ObservationReviewError(
                f"Observation reviews for {run_id} are malformed.")
        return payload

    def save_classification(self, run_id: str, package: dict[str, Any],
                            frame_index: int, action: str,
                            human_value: Any, reviewer: str,
                            ) -> dict[str, Any]:
        """Classify one frame directly from the evidence viewer."""
        if action not in (
                "confirm", "flag", "not_relevant", "cannot_determine",
                "review_later"):
            raise ObservationReviewError(
                "action must be one of confirm, flag, not_relevant, "
                "cannot_determine, review_later.")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ObservationReviewError(
                "A reviewer identity is required.")
        known = {frame["frame_index"]
                 for frame in package.get("frames", [])}
        if frame_index not in known:
            raise ObservationReviewError(
                f"Unknown frame: {frame_index!r}.")
        stored = self.load(run_id)
        existing = stored["classifications"].get(str(frame_index), {})
        revision = int(existing.get("revision", 0)) + 1
        record = {
            "frame_index": frame_index,
            "field": RULES["field"],
            "action": action,
            "human_value": human_value,
            "reviewer": reviewer.strip(),
            "corrected_at": utc_now_iso(),
            "provenance": "human-corrected",
            "revision": revision,
            "explicit_unknown": action in (
                "not_relevant", "cannot_determine"),
        }
        stored["classifications"][str(frame_index)] = record
        path = self._path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(stored, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n")
        tmp.replace(path)
        return copy.deepcopy(record)

    def save_correction(self, run_id: str, task_id: str,
                        package: dict[str, Any], action: str,
                        human_value: Any, reviewer: str,
                        culinary: dict[str, Any] | None = None,
                        ) -> dict[str, Any]:
        """Record one human decision without touching raw inference."""
        if action not in REVIEW_ACTIONS:
            raise ObservationReviewError(
                f"action must be one of {', '.join(REVIEW_ACTIONS)}.")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ObservationReviewError(
                "A reviewer identity is required.")
        _check_human_value(action, human_value)
        stored = self.load(run_id)
        queue = build_review_tasks(
            package, stored["corrections"],
            stored.get("classifications"), culinary)
        task = next((item for item in queue["tasks"]
                     if item["task_id"] == task_id), None)
        if task is None:
            raise ObservationReviewError(f"Unknown task: {task_id!r}.")
        stored = self.load(run_id)
        existing = stored["corrections"].get(task_id)
        revision = int((existing or {}).get("revision", 0)) + 1
        correction = {
            "task_id": task_id,
            "field": task["field"],
            "action": action,
            "original": None,
            "human_value": human_value,
            "reviewer": reviewer.strip(),
            "corrected_at": utc_now_iso(),
            "frame_refs": list(task["frame_refs"]),
            "start_seconds": task["start_seconds"],
            "end_seconds": task["end_seconds"],
            "provenance": "human-corrected",
            "revision": revision,
            "explicit_unknown": action in (
                "not_identifiable", "skip_unknown"),
        }
        stored["corrections"][task_id] = correction
        history = stored.setdefault("correction_history", {})
        entries = history.setdefault(task_id, [])
        entries.append(copy.deepcopy(correction))
        path = self._path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(stored, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n")
        tmp.replace(path)
        return copy.deepcopy(correction)

    def undo_correction(self, run_id: str, task_id: str,
                        package: dict[str, Any], reviewer: str,
                        culinary: dict[str, Any] | None = None,
                        ) -> dict[str, Any]:
        """Revert the latest decision without destroying the audit trail.

        The task returns to the decision that was effective immediately
        before the latest one (or to unresolved when there was none).
        Every undo is recorded in the per-task history, which only grows.
        """
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ObservationReviewError(
                "A reviewer identity is required.")
        stored = self.load(run_id)
        queue = build_review_tasks(
            package, stored["corrections"],
            stored.get("classifications"), culinary)
        task = next((item for item in queue["tasks"]
                     if item["task_id"] == task_id), None)
        if task is None:
            raise ObservationReviewError(f"Unknown task: {task_id!r}.")
        stored = self.load(run_id)
        history = stored.setdefault("correction_history", {})
        entries = history.get(task_id, [])
        marker = max(
            (index for index, entry in enumerate(entries)
             if entry.get("action") == "undo"),
            default=-1)
        chain = entries[marker + 1:]
        current = stored["corrections"].get(task_id)
        if not chain or current is None or current.get("revision") != chain[
                -1].get("revision"):
            raise ObservationReviewError(
                f"Nothing to undo for task: {task_id!r}.")
        undone = chain[-1]
        if len(chain) > 1:
            restored = copy.deepcopy(chain[-2])
            stored["corrections"][task_id] = restored
        else:
            restored = None
            stored["corrections"].pop(task_id, None)
        entries.append({
            "action": "undo",
            "reverted_action": undone.get("action"),
            "reverted_human_value": undone.get("human_value"),
            "reverted_revision": undone.get("revision"),
            "restored_action": restored.get("action")
            if restored is not None else None,
            "reviewer": reviewer.strip(),
            "corrected_at": utc_now_iso(),
            "task_id": task_id,
        })
        path = self._path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(stored, indent=2, sort_keys=True) + "\n",
            encoding="utf-8", newline="\n")
        tmp.replace(path)
        queue = build_review_tasks(
            package, stored["corrections"],
            stored.get("classifications"), culinary)
        current_task = next(item for item in queue["tasks"]
                            if item["task_id"] == task_id)
        return {"undone": copy.deepcopy(undone),
                "correction": copy.deepcopy(restored),
                "status": current_task["status"]}
