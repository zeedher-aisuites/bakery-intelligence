"""Canonical ``FrameEvidence`` record (plan.md §4.2)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from bakery_intelligence.models import (
    SHA256_HEX_PATTERN,
    NonBlankStr,
    SchemaVersionedModel,
)


class Region(BaseModel):
    """Optional image crop in pixels (``x``, ``y`` origin, size)."""

    model_config = ConfigDict(extra="forbid")

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class Extraction(BaseModel):
    """Frame-extraction provenance (manifest-backed in M6)."""

    model_config = ConfigDict(extra="forbid")

    sample_fps: float = Field(gt=0)
    source_offset: float = Field(ge=0)
    manifest_sha256: str = Field(pattern=SHA256_HEX_PATTERN)


class FrameEvidence(SchemaVersionedModel):
    """One extracted frame addressable as observation evidence.

    ``frame_index`` is ``None`` for preliminary-sourced or non-frame
    fixtures. This record asserts that a frame was extracted and where to
    find it; it asserts nothing about what the frame depicts (plan.md §5.1
    state 1: frame-linked evidence is not verified interpretation).
    """

    evidence_id: NonBlankStr
    video_id: NonBlankStr
    frame_index: Optional[int] = Field(default=None, ge=0)
    timestamp_seconds: Optional[float] = Field(default=None, ge=0)
    source_timestamp_seconds: Optional[float] = Field(default=None, ge=0)
    requested_timestamp_seconds: Optional[float] = Field(default=None, ge=0)
    image_ref: Optional[NonBlankStr] = None
    region: Optional[Region] = None
    extraction: Optional[Extraction] = None
