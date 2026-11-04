"""Evaluatable recipe snapshots from reviewed fusion drafts.

Builds a versioned, machine-readable recipe snapshot from a
persisted fusion draft plus its chef-review decisions. The
snapshot is a frozen evaluation artifact: deterministic given
its inputs, separate from the pre-review AI draft, and explicit
about every unknown. It never invents quantities, ingredients,
or steps; editorial method text is keyed by stable evidence
group ids so each instruction stays traceable.
"""

from __future__ import annotations

from typing import Any

SNAPSHOT_VERSION = "recipe-snapshot/1"
SNAPSHOT_ID = "bakery_demo_v1_ai_reviewed_v1"

# Distinct model labels that describe one weighed addition. The
# mapping is recorded in the snapshot; group ids stay attached.
CONSOLIDATION_ALIASES = {
    "harina para pan": "bread flour",
    "white flour": "bread flour",
}

# Chef-supplied quantities with supported ingredient attribution.
# Each entry cites the display cluster plus the spoken claim and
# the weighing step that ties them to one ingredient.
QUANTITY_ATTRIBUTION: dict[str, dict[str, Any]] = {
    "qc_74119c003bb0": {
        "ingredient": "sugar",
        "amount": 100.0,
        "unit": "g",
        "basis": ("Displayed 100 at clip_02_mixing t=22s; spoken "
                  "'cien gramos de azucar' (t=7.3-19.8s); weighed "
                  "in step eg_021 (15-35s)."),
    },
    "qc_00e9f9487405": {
        "ingredient": "water",
        "amount": 200.0,
        "unit": "g",
        "basis": ("Displayed 200 at clip_01_preparation t=65s; spoken "
                  "'doscientos gramos de agua' (t=26.7-38.1s); poured "
                  "in step eg_003 (40-70s)."),
    },
    "qc_3f02618ecf84": {
        "ingredient": "milk powder",
        "amount": 24.0,
        "unit": "g",
        "basis": ("Displayed 24 at clip_02_mixing t=120s; spoken "
                  "'veinticuatro gramos de leche en polvo' "
                  "(t=60.1-72.5s); weighed in step eg_026 "
                  "(112-128s)."),
    },
    "qc_eb0bf29bd96e": {
        "ingredient": "bread flour",
        "amount": 699.0,
        "unit": "g",
        "basis": ("Displayed 699 at clip_02_mixing t=155s against a "
                  "spoken target of 700 ('setecientos gramos ... "
                  "harina ... para pan', t=79.8-89.3s); weighed in "
                  "step eg_027 (128-168s). Displayed value kept; "
                  "1 g short of the spoken target."),
    },
    "qc_44f6a1e410ed": {
        "ingredient": "salt",
        "amount": 3.0,
        "unit": "g",
        "basis": ("Displayed 3 at clip_02_mixing t=177s; spoken "
                  "'tres gramos de sal' (t=126.5-138.5s); weighed "
                  "in step eg_028 (168-183s)."),
    },
    "qc_7d58f1f10d28": {
        "ingredient": "dry yeast",
        "amount": 14.0,
        "unit": "g",
        "basis": ("Displayed 14 at clip_01_preparation t=205s; "
                  "spoken 'catorce gramos de levadura seca' "
                  "(t=204.7-213.5s); weighed in step eg_008 "
                  "(190-210s)."),
    },
    "qc_6be5de30e63e": {
        "ingredient": "butter",
        "amount": 112.0,
        "unit": "g",
        "basis": ("Displayed 112 at clip_02_mixing t=258s against "
                  "a spoken target of 113 ('ciento trece gramos de "
                  "mantequilla', t=176.2-180.3s; speaker reads back "
                  "'ciento, doce'); weighed in step eg_030 "
                  "(233-268s). Displayed value kept."),
    },
}

# Observed counts are not weighings; recorded separately.
COUNT_ATTRIBUTION: dict[str, dict[str, Any]] = {
    "egg": {
        "amount": 2,
        "unit": "whole",
        "basis": ("Two white eggs observed (group ig_a58bac6bee7c, "
                  "'two white eggs, then cracked yolks'); spoken "
                  "'dos piezas de huevo' (clip_02_mixing "
                  "t=36.7-44.2s). Counted, never weighed."),
    },
}

# Spoken amounts that stay unconfirmed (display never stable or
# never legible). Recorded as notes, never as quantities.
SPOKEN_UNCONFIRMED = {
    "wheat flour": ("Spoken 'cuarenta gramos de harina' for the "
                    "tangzhong (clip_01_preparation t=0.5-16.4s); "
                    "the display passed 30-40 transiently and was "
                    "never confirmed stable."),
    "milk": ("Spoken 'doscientos cuarenta gramos de leche' "
             "(clip_01_preparation t=176.5-188.1s); the display "
             "fluctuated around 230-239 mid-pour and was never "
             "confirmed stable."),
    "vanilla essence": ("Spoken 'una tapita ... igual a cuatro o "
                        "cinco gramos' (clip_02_mixing t=36.7-44.2s); "
                        "never weighed on camera."),
}

# Groups that are observations, not recipe ingredients.
EXCLUDED_GROUPS = {
    "ig_20210810ee14": ("Ciel water bottle verified visible in the "
                        "fridge; no evidence it was used."),
    "ig_3563e6df66e8": ("Unidentified translucent liquid; identity "
                         "could not be determined."),
    "ig_f40d90524946": ("Unidentified white liquid; identity could "
                         "not be determined."),
}

METHOD_PLAN: list[dict[str, Any]] = [
    {
        "clip_id": "clip_01_preparation",
        "interval": "0:00-0:35",
        "instruction": ("Tare the scale and add a flour portion to "
                        "the bowl for the tangzhong. The exact amount "
                        "was not determined: the display passed 30-40 "
                        "transiently and the spoken 40 g was never "
                        "confirmed stable."),
        "step_group_ids": ["eg_000", "eg_001", "eg_016", "eg_018"],
        "caveats": ["Flour amount not determined."],
    },
    {
        "clip_id": "clip_01_preparation",
        "interval": "0:35-1:10",
        "instruction": ("Tare the scale and pour in 200 g of water, "
                        "for a spoken tangzhong total of about 240 g."),
        "step_group_ids": ["eg_002", "eg_003"],
        "caveats": [],
    },
    {
        "clip_id": "clip_01_preparation",
        "interval": "1:10-1:25",
        "instruction": "Mix the flour and water together.",
        "step_group_ids": ["eg_004"],
        "caveats": ["Mixing completeness judged visually."],
    },
    {
        "clip_id": "clip_01_preparation",
        "interval": "1:25-2:15",
        "instruction": ("Microwave in intervals with stirring, per "
                        "the narration about 30-40 seconds each. "
                        "Exact time and power were not confirmed."),
        "step_group_ids": ["eg_005"],
        "caveats": ["Exact microwave time and power unconfirmed."],
    },
    {
        "clip_id": "clip_01_preparation",
        "interval": "2:15-2:45",
        "instruction": ("Cover the paste (plastic wrap is mentioned) "
                        "and chill it. Duration and temperature were "
                        "not observed."),
        "step_group_ids": ["eg_006", "eg_019"],
        "caveats": ["Chill duration and temperature unknown."],
    },
    {
        "clip_id": "clip_01_preparation",
        "interval": "2:45-3:10",
        "instruction": ("Tare the scale with the steel mixing bowl "
                        "and pour in milk from the red mug. The "
                        "display fluctuated around 230-239 mid-pour, "
                        "so the amount was not determined; the "
                        "narration mentions 240 g."),
        "step_group_ids": ["eg_007", "eg_011", "eg_015"],
        "caveats": ["Milk amount not determined. Unit not legible."],
    },
    {
        "clip_id": "clip_01_preparation",
        "interval": "3:10-3:30",
        "instruction": ("Tare the scale and add 14 g of dry yeast "
                        "from the small white dish."),
        "step_group_ids": ["eg_008", "eg_014", "eg_017"],
        "caveats": [],
    },
    {
        "clip_id": "clip_01_preparation",
        "interval": "3:30-4:16",
        "instruction": ("Rest the mixture so the yeast can activate; "
                        "bubbles were observed. Rest duration was not "
                        "observed and activation was not measured."),
        "step_group_ids": ["eg_009"],
        "caveats": ["Rest duration unknown; activation unmeasured."],
    },
    {
        "clip_id": "clip_02_mixing",
        "interval": "0:00-0:15",
        "instruction": ("Starting from a bubbly fermented mass in a "
                        "separate recording. Its exact composition "
                        "was not visually proven, and continuity with "
                        "clip 1 was not established: the gap between "
                        "recordings is unknown."),
        "step_group_ids": ["eg_020"],
        "caveats": ["Preferment composition unproven; gap unknown."],
    },
    {
        "clip_id": "clip_02_mixing",
        "interval": "0:15-0:35",
        "instruction": "Weigh 100 g of sugar and add it.",
        "step_group_ids": ["eg_021"],
        "caveats": [],
    },
    {
        "clip_id": "clip_02_mixing",
        "interval": "0:35-0:58",
        "instruction": ("Stir in the chilled tangzhong paste. Its "
                        "amount was not stated or measured at this "
                        "point."),
        "step_group_ids": ["eg_022"],
        "caveats": ["Tangzhong amount at this step unknown."],
    },
    {
        "clip_id": "clip_02_mixing",
        "interval": "0:58-1:52",
        "instruction": ("Crack 2 eggs, add a cap of vanilla essence "
                        "(about 4-5 g spoken, never weighed), whisk "
                        "with liquid, and pour the egg mixture into "
                        "the bowl. Some residue may remain in the "
                        "small bowl."),
        "step_group_ids": ["eg_023", "eg_024", "eg_025", "eg_035"],
        "caveats": ["Vanilla amount never weighed; eggs not weighed."],
    },
    {
        "clip_id": "clip_02_mixing",
        "interval": "1:52-2:08",
        "instruction": "Weigh 24 g of milk powder and add it.",
        "step_group_ids": ["eg_026", "eg_036", "eg_037", "eg_038"],
        "caveats": [],
    },
    {
        "clip_id": "clip_02_mixing",
        "interval": "2:08-2:48",
        "instruction": ("Weigh the bread flour and add it: 699 g "
                        "displayed against a spoken target of 700 g. "
                        "Flour type and protein content were not "
                        "proven by a label."),
        "step_group_ids": ["eg_027"],
        "caveats": ["Flour type unproven by label."],
    },
    {
        "clip_id": "clip_02_mixing",
        "interval": "2:48-3:03",
        "instruction": "Weigh 3 g of salt and add it last.",
        "step_group_ids": ["eg_028", "eg_034"],
        "caveats": [],
    },
    {
        "clip_id": "clip_02_mixing",
        "interval": "3:03-3:53",
        "instruction": ("Mix with the stand mixer until the "
                        "ingredients come together. Speed setting "
                        "and exact duration were not stated."),
        "step_group_ids": ["eg_029", "eg_039", "eg_040"],
        "caveats": ["Mixer speed and duration unstated."],
    },
    {
        "clip_id": "clip_02_mixing",
        "interval": "3:53-4:28",
        "instruction": ("Weigh 112 g of butter (spoken target 113 g) "
                        "and add it to the dough."),
        "step_group_ids": ["eg_030"],
        "caveats": ["Butter vs margarine unproven by label."],
    },
    {
        "clip_id": "clip_02_mixing",
        "interval": "4:28-5:20",
        "instruction": ("Machine-knead until the butter is "
                        "incorporated; a hand addition happens "
                        "off-camera. Exact duration unstated."),
        "step_group_ids": ["eg_031"],
        "caveats": ["Kneading duration unstated."],
    },
    {
        "clip_id": "clip_02_mixing",
        "interval": "5:20-5:48",
        "instruction": ("Check gluten development (windowpane); it was "
                        "described as still slightly lacking, a "
                        "subjective assessment."),
        "step_group_ids": ["eg_032"],
        "caveats": ["Gluten assessment subjective."],
    },
    {
        "clip_id": "clip_02_mixing",
        "interval": "5:48-6:28",
        "instruction": ("Finish kneading by hand. Shaping and "
                        "chilling were only a spoken plan; they were "
                        "not shown."),
        "step_group_ids": ["eg_033"],
        "caveats": ["Shaping and chilling not shown."],
    },
]

# Sub-actions without culinary instruction value. Listed for
# traceability, not rendered as recipe instructions.
OMITTED_DETAIL_STEPS = {
    "eg_010": "Local detail (weigh) inside the clip-1 window; no distinct instruction.",
    "eg_012": "Finger touching a liquid surface; no culinary instruction.",
    "eg_013": "Bowl handling sub-action; covered by the chilling instruction.",
    "eg_041": "Unlabeled local 'show' segment; could not be determined.",
}


def reviewed_name(group: dict[str, Any],
                  decision: dict[str, Any] | None) -> str | None:
    """Effective ingredient name, or None when unusable.

    A chef correction wins; a confirmation keeps the model label.
    Cannot-determine and missing decisions yield no recipe name.
    """
    if decision is None:
        return None
    action = decision.get("action")
    if action == "correct":
        value = decision.get("value") or {}
        name = value.get("name")
        return str(name).strip() or None \
            if isinstance(name, str) else None
    if action == "confirm":
        label = group.get("display_label")
        return str(label).strip() or None \
            if isinstance(label, str) else None
    return None


def decision_provenance(decision: dict[str, Any] | None) -> str:
    """Map a review action to its provenance badge value."""
    if decision is None:
        return "unknown"
    action = decision.get("action")
    if action == "confirm":
        return "human_verified"
    if action == "correct":
        return "human_corrected"
    if action == "supply":
        return "chef_supplied"
    return "unknown"


def consolidate_ingredients(
        draft: dict[str, Any],
        decisions: dict[tuple[str, str], dict[str, Any]],
        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge reviewed groups into recipe rows plus exclusions.

    Rows merge on reviewed name (case-insensitive) after alias
    resolution. Every row keeps all group ids and evidence
    timestamps; nothing merges silently.
    """
    rows: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    excluded: list[dict[str, Any]] = []
    for group in draft.get("ingredients") or []:
        group_id = group.get("group_id")
        decision = decisions.get(("ingredient", group_id))
        name = reviewed_name(group, decision)
        if group_id in EXCLUDED_GROUPS or name is None:
            excluded.append({
                "group_id": group_id,
                "display_label": group.get("display_label"),
                "reason": EXCLUDED_GROUPS.get(
                    group_id, "Identity could not be determined; "
                              "no usable recipe name."),
                "decision": (decision or {}).get("action"),
            })
            continue
        key = CONSOLIDATION_ALIASES.get(name.lower(), name).lower()
        canonical = CONSOLIDATION_ALIASES.get(name.lower(), name)
        row = rows.get(key)
        if row is None:
            row = {"name": canonical, "group_ids": [],
                   "provenance": set(), "evidence": [],
                   "display_labels": []}
            rows[key] = row
            order.append(key)
        row["group_ids"].append(group_id)
        row["provenance"].add(decision_provenance(decision))
        row["display_labels"].append(group.get("display_label"))
        for stamp in group.get("supporting_timestamps") or []:
            row["evidence"].append({
                "clip_id": group.get("clip_id"),
                "timestamp_seconds": stamp,
                "kind": "ingredient_observation",
            })
    ingredients: list[dict[str, Any]] = []
    for key in sorted(order):
        row = rows[key]
        ingredients.append({
            "name": row["name"],
            "quantity": None,
            "unit": None,
            "quantity_status": "not_determined",
            "quantity_basis": None,
            "review_status": sorted(row["provenance"]),
            "group_ids": sorted(row["group_ids"]),
            "display_labels": row["display_labels"],
            "evidence": sorted(
                row["evidence"],
                key=lambda e: (e.get("clip_id") or "",
                               e.get("timestamp_seconds")
                               if e.get("timestamp_seconds")
                               is not None else 1e12)),
            "note": SPOKEN_UNCONFIRMED.get(row["name"]),
        })
    excluded.sort(key=lambda e: e.get("group_id") or "")
    return ingredients, excluded


def attribute_quantities(
        ingredients: list[dict[str, Any]],
        draft: dict[str, Any],
        decisions: dict[tuple[str, str], dict[str, Any]],
        ) -> None:
    """Attach chef-supplied quantities with supported attribution.

    A quantity attaches only when the cluster decision is a
    supply whose value matches the recorded attribution and the
    named ingredient exists. Anything else stays not_determined.
    """
    by_name = {row["name"].lower(): row for row in ingredients}
    clusters = (draft.get("quantity_assessment") or {}).get(
        "stable_clusters") or []
    for cluster in clusters:
        cluster_id = cluster.get("cluster_id")
        spec = QUANTITY_ATTRIBUTION.get(cluster_id)
        if spec is None:
            continue
        decision = decisions.get(("quantity_cluster", cluster_id))
        if decision is None or decision.get("action") != "supply":
            continue
        value = decision.get("value") or {}
        if value.get("unit") != spec["unit"]:
            continue
        try:
            amount = float(value.get("amount"))
        except (TypeError, ValueError):
            continue
        if abs(amount - spec["amount"]) > 1e-9:
            continue
        row = by_name.get(spec["ingredient"].lower())
        if row is None:  # pragma: no cover - defensive
            continue
        row["quantity"] = amount
        row["unit"] = spec["unit"]
        row["quantity_status"] = "chef_supplied"
        row["quantity_basis"] = spec["basis"]
        row["evidence"].append({
            "clip_id": cluster.get("clip_id"),
            "timestamp_seconds": cluster.get("start_seconds"),
            "kind": "scale_display",
        })
        row["evidence"].sort(key=lambda e: (
            e.get("clip_id") or "",
            e.get("timestamp_seconds")
            if e.get("timestamp_seconds") is not None else 1e12))
    for name, spec in COUNT_ATTRIBUTION.items():
        row = by_name.get(name.lower())
        if row is None:  # pragma: no cover - defensive
            continue
        row["quantity"] = spec["amount"]
        row["unit"] = spec["unit"]
        row["quantity_status"] = "observed_count"
        row["quantity_basis"] = spec["basis"]


def _parse_interval(label: str) -> tuple[float, float]:
    """Parse a display 'm:ss-m:ss' label back to seconds."""
    bounds = []
    for part in label.split("-"):
        minutes, _, seconds = part.strip().partition(":")
        bounds.append(float(minutes) * 60.0 + float(seconds))
    if len(bounds) != 2:
        raise ValueError(f"Bad method interval: {label!r}")
    return bounds[0], bounds[1]


def build_method(
        draft: dict[str, Any],
        decisions: dict[tuple[str, str], dict[str, Any]]
        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Render method instructions with step coverage enforced.

    Every draft step lands in exactly one instruction or in the
    omitted-detail list; anything else raises instead of silently
    dropping evidence. Display intervals must contain their
    concrete step spans (1 s tolerance); local detail steps
    without timestamps are exempt.
    """
    steps = {s.get("group_id"): s for s in draft.get("steps") or []}
    covered: set[str] = set()
    method: list[dict[str, Any]] = []
    for position, item in enumerate(METHOD_PLAN, start=1):
        group_ids = list(item["step_group_ids"])
        span_start, span_end = _parse_interval(item["interval"])
        for group_id in group_ids:
            if group_id not in steps:
                raise KeyError(
                    f"Method plan references unknown step {group_id}")
            if group_id in covered:
                raise KeyError(
                    f"Step {group_id} used by two instructions")
            covered.add(group_id)
            step = steps[group_id]
            start, end = (step.get("start_seconds"),
                          step.get("end_seconds"))
            if isinstance(start, (int, float)) and isinstance(
                    end, (int, float)):
                if start < span_start - 1 or end > span_end + 1:
                    raise ValueError(
                        f"Step {group_id} span {start}-{end}s "
                        f"outside instruction interval "
                        f"{item['interval']}")
        states = sorted({
            decision_provenance(
                decisions.get(("step", group_id)))
            for group_id in group_ids})
        method.append({
            "number": position,
            "clip_id": item["clip_id"],
            "interval": item["interval"],
            "instruction": item["instruction"],
            "step_group_ids": group_ids,
            "review_status": states,
            "caveats": list(item["caveats"]),
        })
    omitted: list[dict[str, Any]] = []
    for group_id in sorted(set(steps) - covered):
        reason = OMITTED_DETAIL_STEPS.get(group_id)
        if reason is None:
            raise KeyError(
                f"Step {group_id} is neither planned nor omitted")
        step = steps[group_id]
        omitted.append({
            "group_id": group_id,
            "clip_id": step.get("clip_id"),
            "actions": step.get("actions") or [],
            "reason": reason,
            "decision": (decisions.get(("step", group_id)) or {}
                         ).get("action"),
        })
    return method, omitted


def build_snapshot(
        draft: dict[str, Any],
        decisions: dict[tuple[str, str], dict[str, Any]],
        agent_f_meta: dict[str, Any],
        generated_at: str,
        ) -> dict[str, Any]:
    """Assemble the frozen evaluation snapshot."""
    ingredients, excluded = consolidate_ingredients(draft, decisions)
    attribute_quantities(ingredients, draft, decisions)
    method, omitted = build_method(draft, decisions)
    counts = {"verified": 0, "corrected": 0, "supplied": 0,
              "cannot_determine": 0}
    for decision in decisions.values():
        action = decision.get("action")
        if action == "confirm":
            counts["verified"] += 1
        elif action == "correct":
            counts["corrected"] += 1
        elif action == "supply":
            counts["supplied"] += 1
        elif action == "cannot_determine":
            counts["cannot_determine"] += 1
    return {
        "snapshot_version": SNAPSHOT_VERSION,
        "snapshot_id": SNAPSHOT_ID,
        "run_name": "bakery_demo_v1",
        "generated_at": generated_at,
        "title": "Cinnamon Rolls (Roles de Canela)",
        "title_basis": ("Spoken goal: 'Vamos a hacer unos roles de "
                        "canela' (clip_01_preparation t=0.5-16.4s). "
                        "Enriched tangzhong dough with yeast, sugar, "
                        "eggs, milk powder, flour, salt, and butter."),
        "review_summary": {
            "reviewer": "ai-chef",
            "ai_executed": True,
            "human_validated": False,
            "decisions": len(decisions),
            **counts,
        },
        "provenance": {
            "pre_review_draft": "runs_api/fusion/bakery_demo_v1/"
                                "fusion_draft.json",
            "review_decisions": "runs_api/fusion_reviews/"
                                "bakery_demo_v1.json",
            "agent_f_request_id": agent_f_meta.get("request_id"),
            "agent_f_note": ("The persisted Agent F recipe proposal "
                             "and Agent G LLM verdict predate the "
                             "normalization fixes and are recorded "
                             "for provenance only; this snapshot is "
                             "built from the current draft plus "
                             "review decisions, not from them."),
        },
        "consolidation_aliases": dict(CONSOLIDATION_ALIASES),
        "ingredients": ingredients,
        "excluded_observations": excluded,
        "method": method,
        "omitted_detail_steps": omitted,
        "review_flags": [
            {
                "flag_id": "possible-mis-correction-ig_19421ce47de7",
                "severity": "needs_human_rereview",
                "detail": ("Group ig_19421ce47de7 (clip_02_mixing "
                           "local 'white powder / fine white powder "
                           "poured from small container') was "
                           "corrected to 'wheat flour', but its "
                           "segment evidence matches the 24 g "
                           "milk-powder weighing (spoken 'leche en "
                           "polvo', display 24). It is consolidated "
                           "under wheat flour faithfully to the "
                           "review, contributes no quantity, and "
                           "needs human re-review; the 24 g milk "
                           "powder attribution rests on group "
                           "ig_b8d55a07ebb0 instead."),
            },
        ],
        "limitations": [
            "No independently verified ground-truth recipe exists; "
            "accuracy cannot be scored.",
            "The gap between the two recordings is unknown; "
            "continuity of ingredients across clips (milk, yeast, "
            "tangzhong) was not established.",
            "Flour portion, milk, vanilla, tangzhong amount at use, "
            "activated-yeast amount, and egg-mixture amount were "
            "not determined; spoken hints are noted, not measured.",
            "Microwave time/power, chill duration/temperature, "
            "mixer speed, rest duration, and shaping/chilling were "
            "not observed or only spoken.",
            "Butter vs margarine and flour type/protein were not "
            "proven by labels.",
            "23 scale observations had illegible displays; 2 speech "
            "claims could not be attached to turns.",
            "The review was executed by an AI agent, not an "
            "independent human; chef-supplied values may reflect "
            "AI culinary judgment.",
        ],
    }
