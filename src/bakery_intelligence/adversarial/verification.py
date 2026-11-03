"""Deterministic verification of recipe claims against evidence.

Returns a canonical ``VerificationReport``. Structural failures
(dangling references, out-of-bounds timestamps, unknown
derivations carrying values) fail the report; support gaps
(no evidence, ambiguity, conflicts) require human review.
A claim is never silently corrected here.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from bakery_intelligence.adversarial.claims import (
    STATUS_CONTRADICTED,
)
from bakery_intelligence.models import GateId, Verdict
from bakery_intelligence.models.verification import (
    VerificationIssue,
    VerificationReport,
)

VERIFIER_VERSION = "recipe-verifier/1"


def _issue(code: str, entity: str, gate: GateId,
           message: str) -> VerificationIssue:
    return VerificationIssue(
        code=code, entity_id=entity, gate=gate,
        message=message, severity="error")


def verify_claims(package: dict[str, Any],
                  claims: list[dict[str, Any]],
                  ) -> tuple[VerificationReport, dict[str, list[str]]]:
    """Verify claims; return (report, claim_id -> issue codes)."""
    frames = {frame["frame_index"]: frame
              for frame in package.get("frames", [])}
    duration = package.get("video", {}).get("duration_seconds")
    weighing = package.get("weighing_event") or {}
    tare_ambiguous = weighing.get("verification_status") == "ambiguous"

    issues: list[VerificationIssue] = []
    per_claim: dict[str, list[str]] = {}
    seen_ids: set[str] = set()
    quantities: dict[str, list[tuple[float, str, str]]] = {}

    def add(claim_id: str, issue: VerificationIssue) -> None:
        issues.append(issue)
        per_claim.setdefault(claim_id, []).append(issue.code)

    for claim in claims:
        claim_id = str(claim.get("claim_id"))
        if claim_id in seen_ids:
            add(claim_id, _issue(
                "DUPLICATE_ID", claim_id, GateId.G1,
                f"Duplicate claim id {claim_id!r}."))
        seen_ids.add(claim_id)

        refs = claim.get("evidence_refs") or []
        if claim.get("provenance") == "ai-extracted" and not refs:
            add(claim_id, _issue(
                "UNSUPPORTED_CLAIM", claim_id, GateId.G3,
                "AI-extracted claim has no evidence references; "
                "model agreement alone cannot support it."))
        for ref in refs:
            frame_index = ref.get("frame_index")
            if frame_index not in frames:
                add(claim_id, _issue(
                    "DANGLING_REFERENCE", claim_id, GateId.G1,
                    f"Evidence frame {frame_index!r} does not "
                    "exist in the evidence package."))
                continue
            timestamp = ref.get("timestamp_seconds")
            if timestamp is None:
                timestamp = frames[frame_index].get(
                    "timestamp_seconds")
            if (isinstance(timestamp, (int, float))
                    and isinstance(duration, (int, float))
                    and not 0 <= timestamp <= duration):
                add(claim_id, _issue(
                    "TIMESTAMP_OUT_OF_BOUNDS", claim_id, GateId.G1,
                    f"Timestamp {timestamp} is outside the "
                    f"0..{duration} video bounds."))
            if ref.get("run_id") not in (None, package.get("run_id")):
                add(claim_id, _issue(
                    "DANGLING_REFERENCE", claim_id, GateId.G1,
                    "Evidence reference belongs to another run."))

        quantity = claim.get("quantity") or {}
        value, unit = quantity.get("value"), quantity.get("unit")
        derivation = quantity.get("derivation", "unknown")
        if derivation == "unknown" and (
                value is not None or unit is not None):
            add(claim_id, _issue(
                "UNKNOWN_DERIVATION_WITH_VALUE", claim_id, GateId.G2,
                "Unknown derivation cannot carry a value or unit."))
        if value is not None and unit is None:
            add(claim_id, _issue(
                "DERIVATION_WITHOUT_UNIT", claim_id, GateId.G2,
                "A quantity value requires a unit."))
        if (claim.get("kind") == "quantity"
                and derivation != "unknown" and tare_ambiguous):
            add(claim_id, _issue(
                "AMBIGUOUS_TARE", claim_id, GateId.G2,
                "Weighing event is ambiguous (tare state not "
                "established); quantity cannot be supported "
                "without human review."))
        ingredient = claim.get("ingredient_id")
        if ingredient and value is not None and unit:
            quantities.setdefault(ingredient, []).append(
                (float(value), str(unit), claim_id))

    for ingredient, entries in quantities.items():
        distinct = {(value, unit) for value, unit, _ in entries}
        if len(distinct) > 1:
            for _, _, claim_id in entries:
                add(claim_id, _issue(
                    "CONFLICTING_CLAIMS", claim_id, GateId.G2,
                    f"Conflicting quantities for ingredient "
                    f"{ingredient!r}: {sorted(distinct)}."))

    has_error = any(issue.code in (
        "DANGLING_REFERENCE", "TIMESTAMP_OUT_OF_BOUNDS",
        "UNKNOWN_DERIVATION_WITH_VALUE", "DERIVATION_WITHOUT_UNIT",
        "DUPLICATE_ID") for issue in issues)
    verdict = (Verdict.FAIL if has_error
               else Verdict.REVIEW_NEEDED if issues
               else Verdict.PASS)
    seed = hashlib.sha256(json.dumps(
        [package.get("run_id"),
         [claim.get("claim_id") for claim in claims]],
        sort_keys=True, default=str).encode()).hexdigest()[:12]
    report = VerificationReport(
        schema_version="1.0.0",
        report_id=f"verification_{seed}",
        verdict=verdict,
        issues=issues,
        checked_ids=sorted(seen_ids),
        verifier_version=VERIFIER_VERSION,
        checked_at=datetime.now(timezone.utc),
    )
    return report, per_claim


def claim_supported(claim: dict[str, Any],
                    issue_codes: list[str]) -> bool:
    """True when a claim may be marked supported."""
    if issue_codes:
        return False
    if claim.get("provenance") == "chef-supplied":
        return False
    refs = claim.get("evidence_refs") or []
    if not refs:
        return False
    if claim.get("kind") == "ingredient" and not claim.get(
            "ingredient_id"):
        return False
    quantity = claim.get("quantity") or {}
    if claim.get("kind") == "quantity" and (
            quantity.get("value") is None
            or quantity.get("derivation", "unknown") == "unknown"):
        return False
    return claim.get("status") != STATUS_CONTRADICTED
