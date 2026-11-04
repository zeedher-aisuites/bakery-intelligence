"""Phase 2C multi-clip session fusion (offline, deterministic).

Combines per-clip agent artifacts into one session draft:
clip-scoped normalization, combined reconciliation with
same-clip gates, speech linking, quantity assessment, and an
explicit unknown inter-clip gap. Reuses the Phase 2A fusion
package; single-video behavior is unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bakery_intelligence.fusion import (
    normalize,
    quantities,
    reconcile,
    reconstruct,
)
from bakery_intelligence.fusion.run import (
    build_evaluation,
    describe_coverage,
    sha256_file,
    write_json,
)
from bakery_intelligence.phase2c import media
from bakery_intelligence.phase2c.manifest import SessionManifest

FUSION_VERSION = "fusion/v1-phase2c"

ROLE_GLOBAL = "muse-spark-1.3-global-video"
ROLE_LOCAL = "muse-spark-1.3-local-video"
ROLE_MEASURE = "muse-spark-1.3-measurement"


def scoped(target: list[dict[str, Any]],
           records: list[dict[str, Any]], role: str) -> None:
    """Append records tagged with their agent role as source.

    Roles share one model but run independent prompts, so
    cross-role agreement and disagreement are meaningful
    reconciliation signals (never verification).
    """
    for record in records:
        record["source"] = role
    target.extend(records)


def fuse_session(
    manifest: SessionManifest,
    clip_inputs: dict[str, dict[str, Any]],
    recipe_proposal: dict[str, Any] | None = None,
    critic_verdict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fuse normalized per-clip evidence into one session draft."""
    events: list[dict[str, Any]] = []
    ingredients: list[dict[str, Any]] = []
    measurements: list[dict[str, Any]] = []
    speech: list[dict[str, Any]] = []
    titles: list[str] = []
    claims_attached = 0
    claims_orphaned = 0
    for clip in manifest.clips:
        clip_id = clip.clip_id
        clip_in = clip_inputs.get(clip_id, {})
        agent_a = clip_in.get("agent_a") or {}
        parsed = agent_a.get("parsed") or {}
        response_id = agent_a.get("request_id")
        scoped(events, normalize.spark_events(
            parsed, response_id, clip_id), ROLE_GLOBAL)
        scoped(ingredients, normalize.spark_ingredients(
            parsed, response_id, clip_id), ROLE_GLOBAL)
        scoped(measurements, normalize.spark_measurements(
            parsed, response_id, clip_id), ROLE_GLOBAL)
        for segment in clip_in.get("agent_b") or []:
            parsed_b = segment.get("parsed") or {}
            response_b = segment.get("request_id")
            scoped(events, normalize.spark_events(
                parsed_b, response_b, clip_id,
                window=segment.get("window")), ROLE_LOCAL)
            scoped(ingredients, normalize.spark_ingredients(
                parsed_b, response_b, clip_id), ROLE_LOCAL)
            scoped(measurements, normalize.spark_measurements(
                parsed_b, response_b, clip_id), ROLE_LOCAL)
        agent_d = clip_in.get("agent_d") or {}
        scoped(measurements, [
            _agent_d_reading(
                reading, response_id, clip_id)
            for reading in (agent_d.get("parsed") or {}).get(
                "readings") or []], ROLE_MEASURE)
        turns = [dict(turn)
                 for turn in (clip_in.get("agent_c_turns") or [])
                 if isinstance(turn, dict)]
        claims_file = clip_in.get("agent_c_claims") or {}
        claims = (claims_file.get("parsed") or {}).get("claims") or []
        attached, orphaned = normalize.attach_claims(turns, claims)
        claims_attached += attached
        claims_orphaned += orphaned
        speech.extend(normalize.speech_turns(turns, clip_id))
        activity = parsed.get("overall_activity")
        if isinstance(activity, str) and activity.strip():
            titles.append(f"[{clip_id}] {activity.strip()}")
    order = {clip.clip_id: clip.order
             for clip in manifest.clips}
    groups = reconcile.group_events(events, order)
    reconcile.link_speech(groups, speech)
    ingredient_groups = reconcile.group_ingredients(ingredients)
    conflicts = reconcile.detect_conflicts(
        events, measurements, ingredients)
    pairs = reconcile.corroborated_readings(measurements)
    assessment = quantities.build_quantity_assessment(
        measurements, [])
    draft = reconstruct.build_draft(
        " / ".join(titles) if titles else None, groups,
        ingredient_groups, measurements, conflicts, pairs, [],
        {"session_id": manifest.session_id,
         "fusion_version": FUSION_VERSION},
        quantity_assessment=assessment,
        clip_order={clip.clip_id: clip.order
                    for clip in manifest.clips})
    draft["session"] = {
        "session_id": manifest.session_id,
        "clips": [{"clip_id": clip.clip_id, "order": clip.order,
                   "duration_seconds": clip.duration_seconds,
                   "sha256": clip.sha256}
                  for clip in manifest.clips],
        "inter_clip_gap": manifest.inter_clip_gap,
    }
    draft["speech"] = speech
    if recipe_proposal is not None:
        draft["recipe_proposal"] = recipe_proposal
    if critic_verdict is not None:
        draft["critic_verdict"] = critic_verdict
    evaluation = build_evaluation(
        draft, events, [], measurements, None,
        coverage={
            "session": manifest.session_id,
            "clips": [clip.clip_id for clip in manifest.clips],
            "speech_turns": len(speech),
            "speech_claims_attached": claims_attached,
            "speech_claims_orphaned": claims_orphaned,
            "inter_clip_gap": manifest.inter_clip_gap,
        })
    return {"draft": draft, "evaluation": evaluation}


def _agent_d_reading(reading: dict[str, Any],
                     response_id: str | None,
                     clip_id: str) -> dict[str, Any]:
    from bakery_intelligence.fusion.models import (
        SOURCE_SPARK,
        STATUS_AI_SUGGESTED,
        stable_id,
    )

    value = reading.get("displayed_value_verbatim")
    stamp = normalize.as_seconds(reading.get("clip_local_seconds"))
    ref = reading.get("frame_ref") or "frame"
    parts = [SOURCE_SPARK, "agent-d", ref, value,
             *normalize.clip_parts(clip_id)]
    return {
        "reading_id": f"fm_{stable_id(*parts)}",
        "source": SOURCE_SPARK,
        "clip_id": clip_id,
        "source_ref": f"{response_id}:agent-d:{ref}"
        if response_id else f"agent-d:{ref}",
        "timestamp_seconds": stamp,
        "displayed_value": str(value).strip()
        if isinstance(value, str) and value.strip()
        else (value if isinstance(value, (int, float)) else None),
        "visible_unit": reading.get("visible_unit")
        if isinstance(reading.get("visible_unit"), str) else None,
        "tare_observable": reading.get("tare_button_visible")
        if isinstance(reading.get("tare_button_visible"), bool)
        else None,
        "evidence": reading.get("evidence")
        if isinstance(reading.get("evidence"), str) else None,
        "uncertain": reading.get("uncertain", True) is not False,
        "status": STATUS_AI_SUGGESTED,
    }


def persist_session(result: dict[str, Any], output_dir: Path,
                    run_name: str) -> dict[str, str]:
    """Write draft, evaluation, and manifest for a session run."""
    out = output_dir / run_name
    write_json(out / "fusion_draft.json", result["draft"])
    write_json(out / "evaluation.json", result["evaluation"])
    media.write_json(out / "manifest.json", {
        "fusion_version": FUSION_VERSION,
        "run_name": run_name,
        "artifacts": {
            "fusion_draft.json": sha256_file(
                out / "fusion_draft.json"),
            "evaluation.json": sha256_file(
                out / "evaluation.json"),
        },
    })
    return {"draft": str(out / "fusion_draft.json"),
            "evaluation": str(out / "evaluation.json"),
            "output_dir": str(out)}
