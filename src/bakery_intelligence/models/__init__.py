"""Shared canonical contract types for ``schema_version 1.0.0`` records.

This package implements the authoritative M1 domain models specified by
``specs/video-to-recipe/plan.md`` §4 and ``specs/video-to-recipe/tasks.md``
M1-01. Every persisted top-level record carries ``schema_version`` and
inherits the major-version policy from :class:`SchemaVersionedModel`.

Verification semantics (plan.md §5.1, §7, §12): these models enforce
structural and semantic consistency (required fields, enums, intervals,
derivation coherence). A successfully parsed record is *not* proof that a
scale display or ingredient was interpreted correctly from pixels. Status
promotion rules (which component may assign ``confirmed``) live in the
adapter (M1-02) and the deterministic verifier (M2-03), not in these models.
"""

from __future__ import annotations

import re
from enum import Enum
from importlib import import_module
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

CANONICAL_SCHEMA_VERSION = "1.0.0"
CANONICAL_SCHEMA_MAJOR = 1

SHA256_HEX_PATTERN = r"^[0-9a-fA-F]{64}$"

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

NonBlankStr = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]
"""Non-empty identifier text. Surrounding whitespace is stripped."""


class TimestampBasis(str, Enum):
    """Provenance taxonomy for every timestamp (plan.md §4.2)."""

    FRAME_INDEX_OVER_FPS = "frame_index_over_fps"
    PRESENTATION_TIMESTAMP = "presentation_timestamp"
    APPROXIMATE_HUMAN_VIDEO_REVIEW = "approximate_human_video_review"
    SYNTHETIC_FIXTURE = "synthetic_fixture"


class VerificationStatus(str, Enum):
    """Observation/event verification state (plan.md §4.2, §5.1)."""

    CONFIRMED = "confirmed"
    AMBIGUOUS = "ambiguous"
    UNOBSERVABLE = "unobservable"


class SourceType(str, Enum):
    """Provenance source of a record (plan.md §4.2)."""

    HUMAN = "human"
    MODEL = "model"
    REVIEWER = "reviewer"
    ADAPTER = "adapter"


class Modality(str, Enum):
    """Observation modality. Audio is deferred to P2 (plan.md §4.2)."""

    VISUAL = "visual"


class ObservationType(str, Enum):
    """Observation subject (weighing_event_schema.md §Frame observations)."""

    SCALE_DISPLAY = "scale_display"
    INGREDIENT = "ingredient"
    ACTION = "action"
    CONTAINER = "container"
    OTHER = "other"


class DisplayState(str, Enum):
    """Scale display state (weighing_event_schema.md §Frame observations)."""

    CHANGING = "changing"
    STABLE = "stable"
    ZERO = "zero"
    OCCLUDED = "occluded"
    UNKNOWN = "unknown"


class ActionType(str, Enum):
    """Kitchen action type (weighing_event_schema.md §Action events)."""

    ADD = "add"
    REMOVE = "remove"
    TARE = "tare"
    PLACE_CONTAINER = "place_container"
    REMOVE_CONTAINER = "remove_container"
    MIX = "mix"
    OTHER = "other"
    UNKNOWN = "unknown"


class Derivation(str, Enum):
    """Net-quantity derivation (weighing_event_schema.md §Weighing events)."""

    DIRECT_AFTER_TARE = "direct_after_tare"
    DIFFERENCE_BETWEEN_READINGS = "difference_between_readings"
    INDEPENDENTLY_VERIFIED = "independently_verified"
    UNKNOWN = "unknown"


class MeasurementMode(str, Enum):
    """Weighing measurement mode (weighing_event_schema.md)."""

    INDIVIDUAL = "individual"
    CUMULATIVE = "cumulative"
    UNKNOWN = "unknown"


class DraftStatus(str, Enum):
    """Recipe draft lifecycle state. ``approved`` is set only through an
    explicit human ``ReviewDecision`` enforced by the review boundary and
    verification gate G3 (Constitution V, plan.md §8), never by a generator."""

    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"


class IngredientQuantityStatus(str, Enum):
    """Support state of a draft ingredient quantity (plan.md §4.2)."""

    SUPPORTED = "supported"
    UNKNOWN = "unknown"
    CONFLICTING = "conflicting"


class Verdict(str, Enum):
    """Verification gate outcome (plan.md §7)."""

    PASS = "pass"
    FAIL = "fail"
    REVIEW_NEEDED = "review_needed"


class GateId(str, Enum):
    """Verification gate identity (plan.md §7.1)."""

    G1 = "G1"
    G2 = "G2"
    G3 = "G3"


class RunStatus(str, Enum):
    """Terminal-or-current run state (plan.md §9.1)."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    AWAITING_REVIEW = "awaiting_review"


class ReviewDecisionValue(str, Enum):
    """Human review decision (plan.md §4.2, §8)."""

    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class SchemaVersionedModel(BaseModel):
    """Base class for every persisted canonical record.

    ``schema_version`` is required (no default). Any ``1.x.y`` version is
    accepted so additive minor revisions stay readable; unknown major
    versions and malformed values are rejected with a field-specific error
    (plan.md §4.1). Unknown fields are rejected: M1 models are the exact
    ``1.0.0`` contract, and forward-compatible extras are a deliberate
    future migration decision.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str

    @field_validator("schema_version")
    @classmethod
    def _check_schema_version(cls, value: str) -> str:
        match = _SEMVER_RE.match(value)
        if match is None or int(match.group(1)) != CANONICAL_SCHEMA_MAJOR:
            raise ValueError(
                "schema_version must use canonical major version "
                f"{CANONICAL_SCHEMA_MAJOR} (got {value!r})"
            )
        return value


class TimeInterval(BaseModel):
    """Half-open-style bounded interval with ``0 <= start < end``.

    The ``end <= duration`` bound needs bundle context (video duration) and
    is enforced at the bundle/reference-check layer (M1-03), except on
    :class:`VideoSource`, which carries its own duration.
    """

    model_config = ConfigDict(extra="forbid")

    start_seconds: float = Field(ge=0)
    end_seconds: float

    @model_validator(mode="after")
    def _check_end_after_start(self) -> "TimeInterval":
        if not self.start_seconds < self.end_seconds:
            raise ValueError(
                "interval must satisfy 0 <= start_seconds < end_seconds"
            )
        return self


class Provenance(BaseModel):
    """Origin of a record: which source type and which source artifact/run."""

    model_config = ConfigDict(extra="forbid")

    source_type: SourceType
    source_id: NonBlankStr


_LAZY_MODEL_MODULES = {
    "VideoSource": "video_source",
    "FrameEvidence": "frame_evidence",
    "Observation": "observation",
    "ActionEvent": "action_event",
    "WeighingEvent": "weighing_event",
    "RecipeDraft": "recipe_draft",
    "VerificationReport": "verification",
    "RunManifest": "run_manifest",
    "ReviewDecision": "review_decision",
}


def __getattr__(name: str) -> Any:
    """Lazily re-export the nine canonical models without import cycles."""
    module_name = _LAZY_MODEL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{module_name}")
    return getattr(module, name)


__all__ = [
    "CANONICAL_SCHEMA_MAJOR",
    "CANONICAL_SCHEMA_VERSION",
    "SHA256_HEX_PATTERN",
    "ActionEvent",
    "ActionType",
    "Derivation",
    "DisplayState",
    "DraftStatus",
    "FrameEvidence",
    "GateId",
    "IngredientQuantityStatus",
    "MeasurementMode",
    "Modality",
    "NonBlankStr",
    "Observation",
    "ObservationType",
    "Provenance",
    "RecipeDraft",
    "ReviewDecision",
    "ReviewDecisionValue",
    "RunManifest",
    "RunStatus",
    "SchemaVersionedModel",
    "SourceType",
    "TimeInterval",
    "TimestampBasis",
    "Verdict",
    "VerificationReport",
    "VerificationStatus",
    "VideoSource",
]
