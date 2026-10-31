"""Headless Streamlit runtime tests for the evidence demo.

Uses ``streamlit.testing.v1.AppTest`` to execute the real application
script, so unsupported Streamlit API usage (for example the removed
``st.video(key=...)`` argument) fails here instead of only in a live
browser session. Pure navigation helpers remain covered separately
by ``tests/test_demo_navigation.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_SCRIPT = str(
    Path(__file__).resolve().parent.parent
    / "src/bakery_intelligence/app/demo.py"
)


def _run_app() -> AppTest:
    app = AppTest.from_file(APP_SCRIPT, default_timeout=120)
    app.run()
    return app


def test_app_starts_without_exception() -> None:
    """The workspace renders with no uncaught exception."""
    app = _run_app()
    assert not app.exception, (
        f"Unhandled app exception: {app.exception[0].stack_trace}"
        if app.exception else ""
    )
    titles = [header.value for header in app.header]
    assert any("Video Explorer" in title for title in titles)


def test_video_player_branch_renders_when_source_present() -> None:
    """The video-present path executes when the local MP4 exists.

    The installed AppTest API exposes no video accessor, so this
    asserts the player branch's observable effects instead: the
    requested-position caption (rendered only after the ``st.video``
    call) is present and the missing-video warning is absent.
    Actual seeking remains a manual browser acceptance test.
    """
    from bakery_intelligence.app.adapters import find_repo_root, video_path

    try:
        root = find_repo_root()
    except FileNotFoundError:
        pytest.skip("Repository root not resolvable.")
    if not video_path(root).is_file():
        pytest.skip("Local source video not present.")
    app = _run_app()
    assert not app.exception
    captions = [caption.value for caption in app.caption]
    assert any("Playback requested at" in text for text in captions)
    warnings = [warning.value for warning in app.warning]
    assert not any("Source video not found" in text for text in warnings)


def test_detail_navigation_preserves_selection() -> None:
    """Contextual navigation reaches the detail view with state intact."""
    app = _run_app()
    assert not app.exception
    matches = [
        button for button in app.button
        if button.label == "View measurement details"
    ]
    if not matches:
        pytest.skip("No measurement data loaded; detail button absent.")
    before = app.session_state["demo_state"]["timestamp"]
    matches[0].click().run()
    assert not app.exception
    assert app.session_state["demo_state"]["view"] == "measurements"
    assert app.session_state["demo_state"]["timestamp"] == before
    titles = [header.value for header in app.header]
    assert any("Measurement Evidence" in title for title in titles)


def test_navigation_round_trip_preserves_selection() -> None:
    """Workspace -> detail -> workspace keeps the selected timestamp."""
    app = _run_app()
    assert not app.exception
    assert app.session_state["demo_state"]["view"] == "workspace"
    details = [
        button for button in app.button
        if button.label == "View measurement details"
    ]
    if not details:
        pytest.skip("No measurement data loaded; detail button absent.")
    stamp = app.session_state["demo_state"]["timestamp"]
    details[0].click().run()
    assert not app.exception
    assert app.session_state["demo_state"]["view"] == "measurements"
    backs = [
        button for button in app.button
        if button.label == "Back to Video Explorer"
    ]
    assert len(backs) == 1
    backs[0].click().run()
    assert not app.exception
    assert app.session_state["demo_state"]["view"] == "workspace"
    assert app.session_state["demo_state"]["timestamp"] == stamp
    titles = [header.value for header in app.header]
    assert any("Video Explorer" in title for title in titles)
