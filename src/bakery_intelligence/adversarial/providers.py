"""Provider-independent proposer/challenger interfaces and fakes.

Production roles (not wired to live calls here): Muse Spark
1.3 proposes recipe claims and Muse Open critiques them.
Neither provider's reasoning configuration is claimed: Spark's
existing client sends no reasoning parameter, and Muse Open's
API remains unverified in this repository.

The deterministic fakes below are the only providers used in
tests and in the local recipe-review bootstrap. They perform
no network access and are always labeled as simulated.
"""

from __future__ import annotations

from typing import Any, Protocol

from bakery_intelligence.adversarial.claims import (
    PROVENANCE_AI,
    evidence_ref,
    make_claim,
)


class RecipeProposer(Protocol):
    """Emits candidate recipe claims from an evidence package."""

    def propose(self, package: dict[str, Any]) -> list[dict[str, Any]]:
        ...


class AdversarialChallenger(Protocol):
    """Critiques claims against the evidence package."""

    def critique(self, claims: list[dict[str, Any]],
                 package: dict[str, Any],
                 ) -> list[dict[str, Any]]:
        ...


class EvidenceProposer:
    """Deterministic, model-free proposer over recorded evidence.

    Emits only what the evidence states: a measurement claim
    backed by usable readings, and an explicit unknown-ingredient
    claim. It never invents an ingredient, unit, or quantity.
    """

    provider_id = "deterministic-evidence"
    simulated = True

    def __init__(self) -> None:
        self.calls = 0
        self.packages_seen: list[str] = []

    def propose(self, package: dict[str, Any]) -> list[dict[str, Any]]:
        self.calls += 1
        self.packages_seen.append(str(package.get("run_id")))
        run_id = str(package.get("run_id"))
        usable = [
            frame for frame in package.get("frames", [])
            if (frame.get("measurement") or {}).get("value")
            is not None
        ]
        claims: list[dict[str, Any]] = []
        if usable:
            first, last = usable[0], usable[-1]
            measurement_claim = make_claim(
                kind="measurement",
                title=(
                    f"Scale displayed readings from "
                    f"{first['measurement']['value']} to "
                    f"{last['measurement']['value']} across "
                    f"{len(usable)} usable frames."),
                evidence_refs=[
                    evidence_ref(
                        run_id, frame["frame_index"],
                        (frame.get("measurement") or {}).get(
                            "measurement_id"),
                        frame.get("timestamp_seconds"))
                    for frame in (first, last)
                ],
                uncertainty=(
                    "Display readings only; not ingredient "
                    "quantities."),
            )
            # First and last usable frames are representative
            # evidence for a collection-level claim, never the
            # complete supporting set.
            measurement_claim["evidence_scope"] = "representative"
            measurement_claim["evidence_total"] = len(usable)
            claims.append(measurement_claim)
        claims.append(make_claim(
            kind="ingredient",
            title="Ingredient identity is not established.",
            ingredient_id=None,
            uncertainty=(
                "No ingredient identity is supported by the "
                "recorded evidence."),
        ))
        return claims


class DeterministicFakeProposer:
    """Configurable fake proposer for non-billable tests."""

    provider_id = "deterministic-fake-proposer"
    simulated = True

    def __init__(self, claims: list[dict[str, Any]] | None = None
                 ) -> None:
        self._claims = claims
        self.calls = 0

    def propose(self, package: dict[str, Any]) -> list[dict[str, Any]]:
        self.calls += 1
        if self._claims is not None:
            return [dict(claim) for claim in self._claims]
        return EvidenceProposer().propose(package)


class DeterministicFakeChallenger:
    """Configurable fake challenger; objections keyed by claim id.

    Each critique round consumes the next configured objection
    round, which lets tests model persistent disagreement and
    count exactly how many rounds occurred.
    """

    provider_id = "deterministic-fake-challenger"
    simulated = True

    def __init__(
            self,
            objection_rounds: list[dict[str, list[str]]] | None = None,
            ) -> None:
        self._rounds = list(objection_rounds or [])
        self.calls = 0
        self.claims_seen: list[list[str]] = []

    def critique(self, claims: list[dict[str, Any]],
                 package: dict[str, Any],
                 ) -> list[dict[str, Any]]:
        self.calls += 1
        self.claims_seen.append(
            [str(claim.get("claim_id")) for claim in claims])
        objections: list[dict[str, Any]] = []
        if self._rounds:
            configured = self._rounds.pop(0)
            for claim in claims:
                for text in configured.get(
                        str(claim.get("claim_id")), []):
                    objections.append({
                        "claim_id": claim.get("claim_id"),
                        "objection": text,
                        "contradiction": "contradict" in text.lower(),
                    })
        return objections
