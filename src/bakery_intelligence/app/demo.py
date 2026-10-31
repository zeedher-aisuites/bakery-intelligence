"""Bakery Intelligence interactive evidence demo (M2-05A).

Read-only Streamlit workspace over validated M2-04 reconstruction
artifacts. Launch from the repository root with:

    uv run streamlit run src/bakery_intelligence/app/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st

from bakery_intelligence.app.adapters import (
    VIDEO_RELATIVE,
    event_window,
    find_repo_root,
    load_bundle,
    measurement_rows,
    unresolved_questions,
    video_path,
)
from bakery_intelligence.app.navigation import (
    VIEW_DRAFT,
    VIEW_EVENTS,
    VIEW_LABELS,
    VIEW_MEASUREMENTS,
    VIEW_WORKSPACE,
    VIEWS,
    available_timestamps,
    events_near,
    initial_state,
    nearest_timestamp,
    row_at_timestamp,
    validate_view,
    video_slot_index,
)

THEME_CSS = """
<style>
.stApp { background-color: #FAF6EE; color: #352A25; }
section[data-testid="stSidebar"] { background-color: #EDE3D3; }
div[data-testid="stMetric"] {
    background-color: #FFFEFB;
    border: 1px solid #E2D5BE;
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
}
.ev-card {
    background-color: #FFFEFB;
    border: 1px solid #E2D5BE;
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.7rem;
}
.ev-card h4 { margin: 0 0 0.4rem 0; color: #352A25; font-size: 1rem; }
.ev-card p { margin: 0.15rem 0; font-size: 0.9rem; }
.ev-flag {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    border-radius: 6px;
    padding: 0.1rem 0.45rem;
    margin-right: 0.3rem;
}
.flag-observed { background-color: #EDE3D3; color: #5A4A3E; }
.flag-verified { background-color: #DCE5DC; color: #3E5642; }
.flag-human { background-color: #F3E3C8; color: #7A5220; }
.flag-derived { background-color: #E7E0F0; color: #4A4370; }
.flag-alert { background-color: #F5D9D2; color: #7A2E22; }
.stButton > button[kind="primary"] {
    background-color: #B77743;
    border-color: #B77743;
}
a { color: #8A5A2B; }
</style>
"""


@st.cache_data
def cached_bundle(root: str, run_id: str):  # type: ignore[no-untyped-def]
    return load_bundle(Path(root), run_id)


NAV_WIDGET_KEY = "nav_view"


def _ensure_state(rows: list[dict[str, Any]]) -> None:
    if "demo_state" not in st.session_state:
        st.session_state.demo_state = initial_state(rows)
    state = st.session_state.demo_state
    state.setdefault("seek_nonce", 0)
    # Single source of truth: the radio widget value in session state
    # owns the view. The widget is created without an index/default so
    # Streamlit never sees both a default and a session-state value.
    # This assignment runs before widget instantiation, so it is legal.
    st.session_state[NAV_WIDGET_KEY] = validate_view(
        st.session_state.get(NAV_WIDGET_KEY, state.get("view")))
    state["view"] = st.session_state[NAV_WIDGET_KEY]
    stamp = nearest_timestamp(rows, state.get("timestamp"))
    state["timestamp"] = stamp
    row = row_at_timestamp(rows, stamp)
    state["measurement_id"] = row.get("measurement_id") if row else None


def _go(view: str, timestamp: Any = None) -> None:
    """Navigate programmatically; always invoked as a button callback.

    Callbacks run before the script body, so updating the radio widget
    value here is legal and the destination view renders on the
    resulting rerun. Never call this after the sidebar is rendered.
    """
    st.session_state[NAV_WIDGET_KEY] = validate_view(view)
    st.session_state.demo_state["view"] = validate_view(view)
    if timestamp is not None:
        st.session_state.demo_state["timestamp"] = timestamp


def main() -> None:
    st.set_page_config(
        page_title="Bakery Intelligence Evidence Review",
        layout="wide",
    )
    st.markdown(THEME_CSS, unsafe_allow_html=True)

    try:
        root = find_repo_root(Path(__file__))
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    bundle = cached_bundle(str(root), "v01_full")
    rows = measurement_rows(bundle)
    _ensure_state(rows)
    state = st.session_state.demo_state

    for issue in bundle.issues:
        st.warning(f"{issue.code}: {issue.message} ({issue.path})")

    render_sidebar(bundle)
    view = state["view"]
    if view == VIEW_WORKSPACE:
        render_workspace(root, bundle, rows, state)
    elif view == VIEW_MEASUREMENTS:
        render_measurements(bundle, rows, state)
    elif view == VIEW_EVENTS:
        render_events(bundle, state)
    elif view == VIEW_DRAFT:
        render_draft(bundle)


def render_sidebar(bundle) -> None:  # type: ignore[no-untyped-def]
    window = event_window(bundle)
    st.sidebar.title("Bakery Intelligence")
    st.sidebar.caption("Evidence review (read-only)")
    state = st.session_state.demo_state
    st.sidebar.radio(
        "View",
        options=list(VIEWS),
        format_func=lambda name: VIEW_LABELS[name],
        key=NAV_WIDGET_KEY,
    )
    st.sidebar.divider()
    st.sidebar.write(f"Run: `{bundle.run_id}`")
    st.sidebar.write(f"Event: `{window['event_id']}`")
    st.sidebar.write(
        f"Window: {window['start_seconds']} s - {window['end_seconds']} s"
    )
    draft = bundle.recipe_draft or {}
    status = draft.get("status", "unknown")
    if status == "pending_review":
        st.sidebar.warning("Draft status: pending_review")
    else:
        st.sidebar.info(f"Draft status: {status}")
    validation = bundle.validation or {}
    st.sidebar.write(
        f"Validation issues: {validation.get('issue_count', 'unknown')}"
    )


def render_workspace(root: Path, bundle, rows: list[dict[str, Any]],  # type: ignore[no-untyped-def]
                     state: dict[str, Any]) -> None:
    window = event_window(bundle)
    st.header("Video Explorer")
    st.caption(
        f"Event `{window['event_id']}` spans {window['start_seconds']} s "
        f"to {window['end_seconds']} s. Select a timestamp to inspect its "
        "evidence beside the video."
    )
    stamps = available_timestamps(rows)
    if stamps:
        chips = st.columns(min(len(stamps), 6))
        for index, stamp in enumerate(stamps):
            with chips[index % len(chips)]:
                active = stamp == state["timestamp"]
                if st.button(
                    f"{stamp:g} s",
                    key=f"seek_{stamp:g}",
                    type="primary" if active else "secondary",
                    width="stretch",
                ):
                    state["timestamp"] = stamp
                    # Every click remounts the player, including repeats.
                    state["seek_nonce"] = state.get("seek_nonce", 0) + 1
                    st.rerun()
    video_col, context_col = st.columns([3, 2])
    with video_col:
        render_player(root)
        render_nearby(bundle, state)
    with context_col:
        render_context_card(bundle, rows, state)


def render_player(root: Path) -> None:
    path = video_path(root)
    if not path.is_file():
        st.warning(
            "Source video not found locally "
            f"(expected at `{VIDEO_RELATIVE.as_posix()}`). Evidence cards "
            "remain available."
        )
        return
    state = st.session_state.demo_state
    offset = state.get("timestamp") or 0
    # Two fixed slots; only the active one holds the player. Alternating
    # slots unmounts the previous element so the new one loads at the
    # requested offset (st.video accepts no key for this purpose).
    slots = (st.empty(), st.empty())
    active = video_slot_index(state.get("seek_nonce", 0))
    slots[1 - active].empty()
    slots[active].video(str(path), start_time=int(offset))
    st.caption(
        f"Playback requested at {offset:g} s (selected evidence "
        "timestamp). The browser keeps its own playback position while "
        "playing; each chip click reloads the player at the selection."
    )


def render_nearby(bundle, state: dict[str, Any]) -> None:  # type: ignore[no-untyped-def]
    actions = list(bundle.action_events)
    nearby = events_near(actions, state.get("timestamp"))
    st.subheader("Nearby events")
    if not nearby:
        st.write("No action events cover the selected timestamp.")
        return
    for action in nearby:
        st.markdown(
            "<div class='ev-card'>"
            f"<h4>{action.get('action_type')}: "
            f"{action.get('start_seconds')} s - "
            f"{action.get('end_seconds')} s</h4>"
            f"<p>Status: "
            f"<span class='ev-flag flag-observed'>"
            f"{action.get('verification_status')}</span></p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.button(
            "Inspect event",
            key=f"inspect_{action.get('action_id')}",
            width="stretch",
            on_click=_go,
            args=(VIEW_EVENTS,),
        )


def render_context_card(bundle, rows: list[dict[str, Any]],  # type: ignore[no-untyped-def]
                        state: dict[str, Any]) -> None:
    st.subheader("Selected evidence")
    row = row_at_timestamp(rows, state.get("timestamp"))
    if row is None:
        st.info("No measurement at the selected timestamp.")
        return
    verified = row["verified_display"] is not None
    flags = "<span class='ev-flag flag-observed'>OBSERVED</span>"
    if verified:
        flags += ("<span class='ev-flag flag-verified'>"
                  "SECOND-PASS MODEL READING</span>")
    if row["unit_assigned"]:
        flags += ("<span class='ev-flag flag-human'>"
                  "HUMAN-ASSIGNED UNIT</span>")
    if row["has_disagreement"]:
        flags += ("<span class='ev-flag flag-alert'>"
                  "READINGS DISAGREE</span>")
    st.markdown(
        "<div class='ev-card'>"
        f"<h4>{row['timestamp_seconds']:g} s "
        f"(frame {row['frame_index']})</h4>"
        f"<p>{flags}</p>"
        f"<p>First-pass display: <b>{row['original_verbatim']}</b> "
        f"(readable: {row['original_readable']}, "
        f"confidence: {row['original_confidence']})</p>"
        f"<p>Preferred value: <b>{row['value']}</b> "
        f"(source: {row['value_source']}, status: "
        f"{row['value_status']})</p>"
        f"<p>Second-pass display: <b>{row['verified_display']}</b> "
        f"(confidence: {row['verified_confidence']}). Model output, "
        "not independent ground truth.</p>"
        f"<p>Display delta: <b>{row['delta_from_previous']}</b> "
        "<span class='ev-flag flag-derived'>DERIVED</span> "
        "A delta is not an ingredient quantity.</p>"
        f"<p>Observed unit: <b>{row['unit_observed']}</b> "
        "(none recorded visually).</p>"
        f"<p>Assigned unit: <b>{row['unit_assigned']}</b> "
        f"(provenance: {row['unit_provenance']}, assignment: "
        f"{row['unit_assignment_id']}). Assigned units are never "
        "presented as observed.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.button(
        "View measurement details",
        width="stretch",
        on_click=_go,
        args=(VIEW_MEASUREMENTS,),
    )


def render_measurements(bundle, rows: list[dict[str, Any]],  # type: ignore[no-untyped-def]
                        state: dict[str, Any]) -> None:
    st.header("Measurement Evidence")
    st.caption(
        "Display deltas are arithmetic differences between readings, "
        "never confirmed ingredient quantities."
    )
    st.button(
        "Back to Video Explorer",
        on_click=_go,
        args=(VIEW_WORKSPACE,),
    )
    if not rows:
        st.info("No measurements loaded for this run.")
        return
    disagreements = sum(1 for row in rows if row["has_disagreement"])
    verified = sum(1 for row in rows if row["verified_display"] is not None)
    first, second, third = st.columns(3)
    first.metric("Measurements", len(rows))
    second.metric("Second-pass readings", verified)
    third.metric("Disagreements", disagreements)
    table = [{
        "time (s)": row["timestamp_seconds"],
        "frame": row["frame_index"],
        "original": row["original_verbatim"],
        "verified": row["verified_display"],
        "value": row["value"],
        "source": row["value_source"],
        "status": row["value_status"],
        "delta": row["delta_from_previous"],
        "unit seen": row["unit_observed"] or "not visible",
        "unit assigned": row["unit_assigned"] or "unresolved",
        "disagreement": "yes" if row["has_disagreement"] else "no",
    } for row in rows]
    st.dataframe(table, width="stretch")
    options = {
        f"{row['timestamp_seconds']:g} s ({row['measurement_id']})": row
        for row in rows
    }
    current = row_at_timestamp(rows, state.get("timestamp"))
    default_label = (
        f"{current['timestamp_seconds']:g} s "
        f"({current['measurement_id']})" if current else None
    )
    choice = st.selectbox(
        "Inspect one measurement",
        list(options),
        index=list(options).index(default_label) if default_label else 0,
    )
    picked = options[choice]
    if picked["timestamp_seconds"] != state.get("timestamp"):
        state["timestamp"] = picked["timestamp_seconds"]
        st.rerun()


def render_events(bundle, state: dict[str, Any]) -> None:  # type: ignore[no-untyped-def]
    st.header("Temporal Events")
    st.button(
        "Back to Video Explorer",
        on_click=_go,
        args=(VIEW_WORKSPACE,),
    )
    weighing = bundle.weighing_event or {}
    if weighing:
        st.subheader("Weighing event")
        st.markdown(
            "<div class='ev-card'>"
            f"<h4>{weighing.get('event_id')}</h4>"
            f"<p>Interval: {weighing.get('start_seconds')} s - "
            f"{weighing.get('end_seconds')} s</p>"
            f"<p>Status: <span class='ev-flag flag-observed'>"
            f"{weighing.get('verification_status')}</span></p>"
            f"<p>{weighing.get('review_notes')}</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        final = weighing.get("final_display_reading", {}) or {}
        st.write(
            f"Final display: `{final.get('value')}` "
            f"(observation `{final.get('observation_id')}`)."
        )
        net = weighing.get("net_ingredient_quantity", {}) or {}
        st.write(
            f"Net ingredient quantity: `{net.get('value')}` "
            f"(derivation: `{net.get('derivation')}`). Unresolved by design."
        )
    actions = list(bundle.action_events)
    st.subheader(f"Action events ({len(actions)})")
    for action in actions:
        with st.expander(
            f"{action.get('action_type')}: "
            f"{action.get('start_seconds')} s - {action.get('end_seconds')} s"
        ):
            st.write(f"Action ID: `{action.get('action_id')}`")
            st.write(f"Ingredient: `{action.get('ingredient_id')}`")
            st.write(
                "Evidence observations: "
                f"`{action.get('evidence_observation_ids')}`"
            )
            provenance = action.get("provenance", {}) or {}
            st.write(
                f"Provenance: `{provenance.get('source_type')}:"
                f"{provenance.get('source_id')}`"
            )
            st.write(
                "Verification status: "
                f"`{action.get('verification_status')}`"
            )


def render_draft(bundle) -> None:  # type: ignore[no-untyped-def]
    st.header("Recipe Draft")
    st.button(
        "Back to Video Explorer",
        on_click=_go,
        args=(VIEW_WORKSPACE,),
    )
    draft = bundle.recipe_draft or {}
    if not draft:
        st.info("No recipe draft loaded for this run.")
        return
    status = draft.get("status", "unknown")
    if status == "pending_review":
        st.warning(
            "Status: pending_review. This draft requires qualified human "
            "review before any operational use."
        )
    else:
        st.write(f"Status: `{status}`")
    st.write(
        f"Draft: `{draft.get('draft_id')}` "
        f"(version {draft.get('version')})"
    )
    st.write(f"Video: `{draft.get('video_id')}`")
    steps = draft.get("steps", [])
    st.subheader(f"Steps ({len(steps)})")
    for step in steps:
        st.markdown(
            "<div class='ev-card'>"
            f"<h4>Step {step.get('index')}</h4>"
            f"<p>{step.get('description')}</p>"
            f"<p>Actions: `{step.get('action_ids')}`</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    st.subheader("Unresolved questions")
    questions = unresolved_questions(bundle)
    if not questions:
        st.write("None recorded.")
    for question in questions:
        st.markdown(
            "<div class='ev-card'>"
            f"<p><span class='ev-flag flag-alert'>"
            f"{question.get('code')}</span></p>"
            f"<p>{question.get('message')}</p>"
            f"<p>Entities: `{question.get('related_entity_ids')}`</p>"
            "</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
