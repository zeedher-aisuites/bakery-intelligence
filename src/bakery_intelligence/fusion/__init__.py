"""Multimodal evidence fusion (offline, deterministic).

Combines previously collected P04 evidence (full-video Spark 1.3,
segmented Baseten Spark Open, preliminary human annotation) into
one traceable recipe draft. No inference, no network, no
auto-verification: corroboration counts are recorded but every
AI-derived node keeps an AI-suggested status.
"""

from bakery_intelligence.fusion import normalize, reconcile, reconstruct
from bakery_intelligence.fusion.models import (
    SOURCE_PRELIMINARY,
    SOURCE_SPARK,
    SOURCE_SPARK_OPEN,
    STATUS_AI_SUGGESTED,
    STATUS_UNKNOWN,
)

__all__ = [
    "normalize",
    "reconcile",
    "reconstruct",
    "SOURCE_SPARK",
    "SOURCE_SPARK_OPEN",
    "SOURCE_PRELIMINARY",
    "STATUS_AI_SUGGESTED",
    "STATUS_UNKNOWN",
]
