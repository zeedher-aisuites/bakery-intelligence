"""Reusable local video intake and frame extraction (M2-06C.3).

Generalizes the Exp. 003 extraction pattern to arbitrary
supported videos without touching the frozen experiment.
Staging, hashing, probing, and sampling are deterministic and
offline; frame outputs feed the C.2 first-pass runner
unchanged.
"""

from __future__ import annotations
