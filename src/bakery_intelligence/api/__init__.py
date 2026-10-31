"""HTTP API layer for Bakery Intelligence (M2-06A spike).

FastAPI application exposing backend health and a truthful,
state-machine-driven analysis job interface. Long-running work
executes outside the HTTP request lifecycle behind a runner
boundary designed for a future Modal move.
"""

from __future__ import annotations
