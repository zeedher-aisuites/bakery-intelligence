"""Evidence-grounded adversarial recipe reconstruction (non-billable core).

Provider-independent orchestration over a normalized evidence
package: a proposer emits recipe claims, a challenger critiques
them, a deterministic verifier checks every claim against the
evidence, and a bounded resolver (at most three rounds) assigns
an explicit final status to each claim. Deterministic fakes are
the only providers exercised in tests; no live model is called
by this package.
"""

from bakery_intelligence.adversarial.claims import (
    CLAIM_STATUSES,
    PROVENANCE_AI,
    PROVENANCE_CHEF,
    PROVENANCE_HUMAN,
    STATUS_CONTRADICTED,
    STATUS_REQUIRES_HUMAN,
    STATUS_SUPPORTED,
    STATUS_UNCERTAIN,
    make_claim,
)
from bakery_intelligence.adversarial.evidence import (
    build_evidence_package,
)
from bakery_intelligence.adversarial.providers import (
    DeterministicFakeChallenger,
    DeterministicFakeProposer,
    EvidenceProposer,
)
from bakery_intelligence.adversarial.resolution import (
    MAX_ROUNDS,
    DebateResult,
    debate_claims,
)
from bakery_intelligence.adversarial.verification import (
    verify_claims,
)

__all__ = [
    "CLAIM_STATUSES",
    "MAX_ROUNDS",
    "PROVENANCE_AI",
    "PROVENANCE_CHEF",
    "PROVENANCE_HUMAN",
    "STATUS_CONTRADICTED",
    "STATUS_REQUIRES_HUMAN",
    "STATUS_SUPPORTED",
    "STATUS_UNCERTAIN",
    "DebateResult",
    "DeterministicFakeChallenger",
    "DeterministicFakeProposer",
    "EvidenceProposer",
    "build_evidence_package",
    "debate_claims",
    "make_claim",
    "verify_claims",
]
