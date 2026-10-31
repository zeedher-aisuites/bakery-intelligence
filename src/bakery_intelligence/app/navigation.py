"""Pure selection and navigation logic for the evidence demo.

Streamlit-free helpers that decide which view is active, which
measurement is selected, and which events sit near a timestamp.
The Streamlit layer stores the results in session state; these
functions are exercised by plain offline unit tests.
"""

from __future__ import annotations

from typing import Any

VIEW_WORKSPACE = "workspace"
VIEW_MEASUREMENTS = "measurements"
VIEW_EVENTS = "events"
VIEW_DRAFT = "draft"

VIEWS: tuple[str, ...] = (
    VIEW_WORKSPACE,
    VIEW_MEASUREMENTS,
    VIEW_EVENTS,
    VIEW_DRAFT,
)

VIEW_LABELS: dict[str, str] = {
    VIEW_WORKSPACE: "Video Explorer",
    VIEW_MEASUREMENTS: "Measurement Evidence",
    VIEW_EVENTS: "Temporal Events",
    VIEW_DRAFT: "Recipe Draft",
}


def validate_view(name: Any) -> str:
    """Return a known view name, defaulting to the workspace."""
    if isinstance(name, str) and name in VIEWS:
        return name
    return VIEW_WORKSPACE


def available_timestamps(rows: list[dict[str, Any]]) -> list[float]:
    """Return sorted numeric timestamps present in measurement rows."""
    stamps = [
        row["timestamp_seconds"] for row in rows
        if isinstance(row.get("timestamp_seconds"), (int, float))
    ]
    return sorted(float(stamp) for stamp in stamps)


def default_timestamp(rows: list[dict[str, Any]]) -> float | None:
    """Select the initial timestamp: first disagreement, else first row."""
    for row in rows:
        stamp = row.get("timestamp_seconds")
        if row.get("has_disagreement") and isinstance(stamp, (int, float)):
            return float(stamp)
    stamps = available_timestamps(rows)
    return stamps[0] if stamps else None


def row_at_timestamp(rows: list[dict[str, Any]],
                     timestamp: Any) -> dict[str, Any] | None:
    """Return the measurement row for an exact timestamp, if any."""
    if not isinstance(timestamp, (int, float)):
        return None
    for row in rows:
        if row.get("timestamp_seconds") == timestamp:
            return row
    return None


def nearest_timestamp(rows: list[dict[str, Any]],
                      timestamp: Any) -> float | None:
    """Snap an arbitrary timestamp to the closest available one."""
    if not isinstance(timestamp, (int, float)):
        return default_timestamp(rows)
    stamps = available_timestamps(rows)
    if not stamps:
        return None
    return min(stamps, key=lambda stamp: abs(stamp - float(timestamp)))


def events_near(actions: list[dict[str, Any]],
                timestamp: Any,
                tolerance_seconds: float = 1.0) -> list[dict[str, Any]]:
    """Return action events whose interval covers a timestamp."""
    if not isinstance(timestamp, (int, float)):
        return []
    nearby = []
    for action in actions:
        start = action.get("start_seconds")
        end = action.get("end_seconds")
        if not isinstance(start, (int, float)):
            continue
        if not isinstance(end, (int, float)):
            continue
        if float(start) - tolerance_seconds <= float(timestamp) <= float(end) + tolerance_seconds:
            nearby.append(action)
    return sorted(nearby, key=lambda item: item.get("start_seconds", 0))


def initial_state(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the default navigation state for a fresh session."""
    stamp = default_timestamp(rows)
    row = row_at_timestamp(rows, stamp) if stamp is not None else None
    return {
        "view": VIEW_WORKSPACE,
        "timestamp": stamp,
        "measurement_id": row.get("measurement_id") if row else None,
        "seek_nonce": 0,
    }


def video_slot_index(seek_nonce: Any) -> int:
    """Select which of two player slots renders the video.

    ``st.video`` accepts no ``key`` and updated ``start_time`` values
    do not reliably reseek a mounted player, so the UI alternates
    between two placeholder slots. Each chip click increments the
    nonce, which flips the active slot; the previously mounted
    element unmounts and the new one loads at the requested offset.
    """
    try:
        nonce = int(seek_nonce)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
    return abs(nonce) % 2
