"""Canonical ``VideoSource`` record (plan.md §4.2)."""

from __future__ import annotations

import ntpath
import posixpath
from typing import Optional

from pydantic import Field, field_validator, model_validator

from bakery_intelligence.models import (
    NonBlankStr,
    SchemaVersionedModel,
    TimeInterval,
    TimestampBasis,
)


class VideoSource(SchemaVersionedModel):
    """Identity, bounds, and timing basis of the interval under analysis.

    ``fps``, ``frame_count``, and ``duration_seconds`` are ``None`` when the
    corresponding source property is unknown. The interval always satisfies
    ``0 <= start < end`` (via :class:`TimeInterval`); ``end <= duration`` is
    additionally enforced here because this record carries its own duration.
    """

    video_id: NonBlankStr
    source_ref: NonBlankStr
    source_dataset: Optional[str] = None
    timestamp_basis: TimestampBasis
    fps: Optional[float] = Field(default=None, gt=0)
    frame_count: Optional[int] = Field(default=None, ge=0)
    duration_seconds: Optional[float] = Field(default=None, gt=0)
    interval: TimeInterval
    license: Optional[str] = None

    @field_validator("source_ref")
    @classmethod
    def _reject_absolute_source_ref(cls, value: str) -> str:
        if posixpath.isabs(value) or ntpath.isabs(value):
            raise ValueError(
                "source_ref must be a dataset-local filename or relative "
                "path, never an absolute path"
            )
        return value

    @model_validator(mode="after")
    def _check_interval_within_duration(self) -> "VideoSource":
        if (
            self.duration_seconds is not None
            and self.interval.end_seconds > self.duration_seconds
        ):
            raise ValueError(
                "interval.end_seconds must satisfy end <= duration_seconds "
                "when duration is known"
            )
        return self
