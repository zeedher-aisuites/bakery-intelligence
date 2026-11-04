"""Deterministic quantity-derivation assessment (no inference).

Consolidates exact-duplicate display readings, clusters stable
same-value observations, and grades each cluster as recoverable,
ambiguous, or unavailable. A net quantity is recoverable only
with a legible value AND a visible unit AND verified tare AND
container continuity AND ingredient attribution. No P04 record
currently satisfies all five; the assessment records exactly
which pieces are missing instead of inventing quantities.
"""

from __future__ import annotations

from typing import Any

from bakery_intelligence.fusion.models import stable_id

# Same displayed value bridging gaps at or under this span reads
# as one stable observation window, not independent readings.
STABILITY_GAP_SECONDS = 10.0

STATUS_RECOVERABLE = "recoverable"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_UNAVAILABLE = "unavailable"


def _is_cross_reference(evidence: Any) -> bool:
    """Bare 'see elsewhere' pointers carry no observation of their own."""
    if not isinstance(evidence, str):
        return True
    text = evidence.strip().lower()
    if not text:
        return True
    return text.startswith("see ") and len(text) < 40


def consolidate_duplicates(
        readings: list[dict[str, Any]]) -> tuple[
            list[dict[str, Any]], list[dict[str, Any]]]:
    """Fold exact same-source duplicates; keep one representative.

    Two records are duplicates only when source, timestamp, and
    displayed value all match exactly. Illegible records never
    fold: two unreadable displays at one timestamp are
    independent non-observations with distinct evidence text.
    The survivor keeps the duplicate ids for audit; substantive
    evidence wins over bare cross-references.
    """
    ordered = sorted(readings,
                     key=lambda r: (r.get("source") or "",
                                    r.get("timestamp_seconds") is None,
                                    r.get("timestamp_seconds") or 0.0,
                                    str(r.get("displayed_value")),
                                    _is_cross_reference(
                                        r.get("evidence")),
                                    r.get("reading_id") or ""))
    unique: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    seen: dict[tuple[str, Any, Any], str] = {}
    survivor_ids: dict[str, list[str]] = {}
    for reading in ordered:
        if reading.get("displayed_value") is None:
            unique.append(reading)
            continue
        key = (reading.get("source") or "",
               reading.get("timestamp_seconds"),
               str(reading.get("displayed_value")).strip())
        survivor = seen.get(key)
        if survivor is None:
            seen[key] = reading.get("reading_id")
            survivor_ids[reading.get("reading_id")] = []
            unique.append(reading)
        else:
            duplicates.append({
                "reading_id": reading.get("reading_id"),
                "duplicate_of": survivor,
                "source": reading.get("source"),
                "source_ref": reading.get("source_ref"),
            })
            if reading.get("reading_id") not in survivor_ids[survivor]:
                survivor_ids[survivor].append(
                    reading.get("reading_id"))
    unique_with_refs = [
        {**reading, "duplicate_ids": survivor_ids[reading["reading_id"]]}
        if survivor_ids.get(reading.get("reading_id"))
        else reading
        for reading in unique
    ]
    return unique_with_refs, duplicates


def cluster_stable_readings(
        readings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chain same-value legible readings across short gaps."""
    legible = sorted(
        (r for r in readings if r.get("displayed_value") is not None
         and r.get("timestamp_seconds") is not None),
        key=lambda r: (r["timestamp_seconds"], r.get("reading_id") or ""))
    clusters: list[dict[str, Any]] = []
    for reading in legible:
        value = str(reading["displayed_value"]).strip()
        stamp = reading["timestamp_seconds"]
        if (clusters and clusters[-1]["displayed_value"] == value
                and stamp - clusters[-1]["end_seconds"]
                <= STABILITY_GAP_SECONDS):
            current = clusters[-1]
            current["end_seconds"] = stamp
            current["reading_ids"].append(reading["reading_id"])
            current["members"].append(reading)
        else:
            clusters.append({
                "cluster_id": f"qc_{stable_id(value, stamp)}",
                "displayed_value": value,
                "start_seconds": stamp,
                "end_seconds": stamp,
                "reading_ids": [reading["reading_id"]],
                "members": [reading],
            })
    for cluster in clusters:
        members = cluster.pop("members")
        clips = sorted({m.get("clip_id") for m in members
                        if m.get("clip_id")})
        cluster["clip_id"] = clips[0] if len(clips) == 1 else None
        cluster["sources"] = sorted({m.get("source") or "unknown"
                                     for m in members})
        cluster["units"] = sorted(
            {str(m.get("visible_unit")).strip() for m in members
             if m.get("visible_unit")})
        cluster["tare_flags"] = sorted(
            {m.get("tare_observable") for m in members
             if m.get("tare_observable") is not None},
            key=repr)
        cluster["all_uncertain"] = all(
            m.get("uncertain") for m in members)
    return clusters


def assess_cluster(cluster: dict[str, Any]) -> dict[str, Any]:
    """Grade one stable cluster; list every missing derivation piece.

    Tare verification has no offline mechanism in Phase 2A.5
    (tare claims are free text or illegible-display flags), so no
    cluster can reach recoverable here; the ladder is implemented
    so future verified tare plugs in without redesign.
    """
    missing: list[str] = []
    if not cluster.get("units"):
        missing.append("visible unit: no record shows a unit")
    missing.append(
        "verified tare: no confirmed tare baseline precedes this "
        "observation")
    missing.append(
        "container continuity: vessel on/off transitions are "
        "unverified")
    missing.append(
        "ingredient attribution: no record links this display to "
        "one ingredient")
    if cluster.get("all_uncertain"):
        missing.append(
            "reading certainty: every member is flagged uncertain")
    if len(cluster.get("sources") or []) < 2:
        missing.append(
            "independent corroboration: single-source observation")
    if not cluster.get("units"):
        status = STATUS_UNAVAILABLE
    elif len(missing) <= 3:
        status = STATUS_AMBIGUOUS
    else:
        status = STATUS_UNAVAILABLE
    assessed: dict[str, Any] = {
        "cluster_id": cluster.get("cluster_id"),
        "displayed_value": cluster.get("displayed_value"),
        "start_seconds": cluster.get("start_seconds"),
        "end_seconds": cluster.get("end_seconds"),
        "reading_ids": cluster.get("reading_ids") or [],
        "sources": cluster.get("sources") or [],
        "units": cluster.get("units") or [],
        "derivation_status": status,
        "missing_for_quantity": missing,
    }
    if cluster.get("clip_id"):
        assessed["clip_id"] = cluster["clip_id"]
    return assessed


def normalize_tare_claims(raw_claims: Any,
                          source: str) -> list[dict[str, Any]]:
    """Keep free-text tare claims as unverified context, never proof."""
    claims: list[dict[str, Any]] = []
    if not isinstance(raw_claims, list):
        return claims
    for index, item in enumerate(raw_claims):
        if not isinstance(item, str) or not item.strip():
            continue
        text = item.strip()
        claims.append({
            "claim_id": f"tc_{stable_id(source, text)}",
            "source": source,
            "source_ref": f"tare_{index}",
            "claim_text": text,
            "status": "unverified",
        })
    return claims


def build_quantity_assessment(
        readings: list[dict[str, Any]],
        tare_claims: list[dict[str, Any]]) -> dict[str, Any]:
    """Full assessment: duplicates, stable clusters, tare context."""
    unique, duplicates = consolidate_duplicates(readings)
    clusters = [assess_cluster(cluster)
                for cluster in cluster_stable_readings(unique)]
    by_status: dict[str, int] = {
        STATUS_RECOVERABLE: 0,
        STATUS_AMBIGUOUS: 0,
        STATUS_UNAVAILABLE: 0,
    }
    for cluster in clusters:
        by_status[cluster["derivation_status"]] += 1
    illegible = sum(1 for reading in unique
                    if reading.get("displayed_value") is None)
    return {
        "status": "evaluated",
        "readings_total": len(readings),
        "readings_unique": len(unique),
        "duplicates_folded": len(duplicates),
        "duplicates": duplicates,
        "illegible_unique": illegible,
        "stable_clusters": clusters,
        "clusters_by_status": by_status,
        "quantities_recoverable": by_status[STATUS_RECOVERABLE],
        "tare_claims": tare_claims,
        "tare_claims_verified": 0,
    }
