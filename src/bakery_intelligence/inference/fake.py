"""Deterministic fake inference client for offline pipelines (M2-06C.4).

Produces synthetic but strictly deterministic model replies
derived from frame indices, so offline pipeline runs are
reproducible and clearly distinguished from real provider
output. Every reply carries the fake model identity; nothing
here contacts any network, provider, or paid service.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

from bakery_intelligence.inference.firstpass import (
    InferenceError,
    InferenceResponse,
    InferenceUsage,
)

MODEL_ID = "fake-deterministic-v1"
PROVIDER_ID = "fake-offline"

# Fixed test vectors cycled by frame index; one ambiguous entry
# keeps honest uncertainty in default runs.
DEFAULT_VERBATIMS = ("120", "235", "5 15", "340", "410")

_FRAME_STEM = re.compile(r"frame_(\d+)")


class DeterministicFakeClient:
    """Frame-index-derived replies plus scripted overrides."""

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        verbatims: Sequence[str] = DEFAULT_VERBATIMS,
        overrides: dict[int, str | Exception] | None = None,
        usage: InferenceUsage | None = None,
    ) -> None:
        if not verbatims:
            raise ValueError("Fake verbatim table must be non-empty.")
        self._model_id = model_id
        self._verbatims = tuple(verbatims)
        self._overrides = dict(overrides or {})
        self._usage = usage or InferenceUsage(
            input_tokens=50, output_tokens=20, total_tokens=70)
        self.budgets: list[int] = []

    @property
    def model_id(self) -> str:
        """Return the fake model identity recorded in calls."""
        return self._model_id

    def submit(self, prompt_text: str, image_paths: Sequence[Path],
               api_key: str, max_output_tokens: int
               ) -> InferenceResponse:
        """Reply deterministically for the referenced frame."""
        _ = (prompt_text, api_key)
        self.budgets.append(max_output_tokens)
        stem = Path(str(image_paths[0])).stem
        match = _FRAME_STEM.fullmatch(stem)
        if match is None:
            raise ValueError(
                f"Fake client needs frame_<index> names, got: {stem}.")
        index = int(match.group(1))
        if index in self._overrides:
            action = self._overrides[index]
            if isinstance(action, Exception):
                raise action
            raw_text = action
        else:
            raw_text = json.dumps({
                "visible_objects": ["scale"],
                "scale_display_verbatim": self._verbatims[
                    index % len(self._verbatims)],
                "scale_readable": True,
                "scale_state": "stable",
                "tare_action_visible": False,
                "addition_or_removal_visible": "none",
                "container_description": "bowl",
                "container_changed_hint": False,
                "confidence": "high",
                "ambiguity_notes": None,
            })
        return InferenceResponse(
            model=self._model_id, status="completed",
            output_text=raw_text, usage=self._usage,
            latency_seconds=0.0,
            request_id=f"fake-{index:08d}",
            incomplete_reason=None,
        )
