"""Bounded adversarial resolution over individual claims.

Round definition: one round is one challenger critique pass
over the claims still in dispute, followed by deterministic
verification and a status decision for those claims. Round 1
covers every proposed claim; rounds 2 and 3 cover only claims
whose objections remain unresolved. After round 3 no further
provider call is made: unresolved claims end as
``requires_human_review``.

The debate operates on the evidence package and claim dicts
only. It never re-reads video files or frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bakery_intelligence.adversarial.claims import (
    STATUS_CONTRADICTED,
    STATUS_REQUIRES_HUMAN,
    STATUS_SUPPORTED,
    STATUS_UNCERTAIN,
)
from bakery_intelligence.adversarial.providers import (
    AdversarialChallenger,
)
from bakery_intelligence.adversarial.verification import (
    claim_supported,
    verify_claims,
)
from bakery_intelligence.models.verification import VerificationReport

MAX_ROUNDS = 3


@dataclass
class DebateResult:
    """Outcome of one bounded debate."""

    claims: list[dict[str, Any]]
    verification: VerificationReport
    rounds_used: int
    termination_reason: str
    provider_calls: dict[str, int] = field(default_factory=dict)


def debate_claims(
        package: dict[str, Any],
        claims: list[dict[str, Any]],
        challenger: AdversarialChallenger,
        max_rounds: int = MAX_ROUNDS,
        ) -> DebateResult:
    """Debate claims individually for at most ``max_rounds`` rounds."""
    if not 1 <= max_rounds <= MAX_ROUNDS:
        raise ValueError(
            f"max_rounds must be between 1 and {MAX_ROUNDS}.")
    rounds_used = 0
    termination = "all_claims_resolved"
    disputed = list(claims)
    exhausted = False

    for round_number in range(1, max_rounds + 1):
        if not disputed:
            break
        rounds_used = round_number
        objections = challenger.critique(disputed, package)
        by_claim: dict[str, list[dict[str, Any]]] = {}
        for objection in objections:
            by_claim.setdefault(
                str(objection.get("claim_id")), []).append(objection)

        verification, per_claim = verify_claims(package, claims)
        remaining: list[dict[str, Any]] = []
        for claim in disputed:
            claim_id = str(claim.get("claim_id"))
            claim_objections = by_claim.get(claim_id, [])
            claim.setdefault("history", []).append({
                "round": round_number,
                "actor": "challenger",
                "objections": [item.get("objection")
                               for item in claim_objections],
            })
            claim["objections"] = [
                item.get("objection") for item in claim_objections]
            codes = per_claim.get(claim_id, [])

            if any(item.get("contradiction")
                   for item in claim_objections) or (
                    "CONFLICTING_CLAIMS" in codes):
                claim["status"] = STATUS_CONTRADICTED
            elif "DANGLING_REFERENCE" in codes or (
                    "AMBIGUOUS_TARE" in codes):
                claim["status"] = STATUS_REQUIRES_HUMAN
            elif claim_objections:
                if round_number == max_rounds:
                    claim["status"] = STATUS_REQUIRES_HUMAN
                    exhausted = True
                else:
                    claim["status"] = STATUS_UNCERTAIN
                    remaining.append(claim)
            elif claim_supported(claim, codes):
                claim["status"] = STATUS_SUPPORTED
            elif codes or not claim.get("evidence_refs"):
                explicit_unknown = (
                    claim.get("kind") == "ingredient"
                    and not claim.get("ingredient_id"))
                claim["status"] = (
                    STATUS_REQUIRES_HUMAN
                    if claim.get("provenance") == "chef-supplied"
                    else STATUS_UNCERTAIN)
                if claim.get("provenance") != "chef-supplied" and (
                        "UNSUPPORTED_CLAIM" in codes
                        and not explicit_unknown):
                    claim["status"] = STATUS_REQUIRES_HUMAN
            else:
                claim["status"] = STATUS_UNCERTAIN
            claim.setdefault("history", []).append({
                "round": round_number,
                "actor": "resolver",
                "status": claim["status"],
                "verification_codes": codes,
            })
        disputed = remaining
        if exhausted:
            termination = "max_rounds_reached"
        elif disputed:
            termination = "rounds_in_progress"

    if disputed:
        termination = "max_rounds_reached"
        for claim in disputed:
            claim["status"] = STATUS_REQUIRES_HUMAN
    verification, _ = verify_claims(package, claims)
    return DebateResult(
        claims=claims,
        verification=verification,
        rounds_used=rounds_used,
        termination_reason=termination,
        provider_calls={"challenger": getattr(challenger, "calls", 0)},
    )
