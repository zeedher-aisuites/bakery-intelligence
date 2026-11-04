"""Offline Phase 2C tests: contracts, gates, resume, P04 identity.

All provider interactions use injected fakes; the only real
artifact read is the P04 v6 draft for byte-identity comparison
(non-billable, read-only). Media fixtures are synthetic FFmpeg
outputs in tmp dirs.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from bakery_intelligence.fusion import (
    normalize,
    reconcile,
)
from bakery_intelligence.phase2c import orchestrator, prompts
from bakery_intelligence.phase2c.clients import (
    BudgetExceeded,
    BudgetLedger,
)
from bakery_intelligence.phase2c.manifest import (
    CLIP_MIXING,
    CLIP_PREPARATION,
    ManifestError,
    build_session,
    describe_clip,
)


def _make_clip(path: Path, seconds: int = 2) -> Path:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i",
         f"testsrc=size=160x120:rate=10:duration={seconds}",
         "-f", "lavfi", "-i",
         f"sine=frequency=440:duration={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(path)],
        check=True, timeout=120)
    return path


def test_prompt_versions_pinned_and_validated():
    versions = prompts.prompt_versions()
    assert versions["agent-a-global-culinary-analyst"] == "v1.0"
    assert versions["culinary-completion-specialist"] == "v1.0"
    assert versions["recipe-revision-specialist"] == "v1.0"
    assert versions["recipe-proposal-specialist"] == "v1.1"
    assert versions["evidence-validator"] == "v1.0"
    assert len(versions) == 11
    assert prompts.validate_output(
        prompts.GLOBAL_ANALYST, {"overall_activity": None,
                                 "operations": [],
                                 "ingredients": [],
                                 "measurements": [],
                                 "uncertainties": [],
                                 "limitations": []}) == []
    assert prompts.validate_output(
        prompts.GLOBAL_ANALYST, {"operations": []})


def test_manifest_validates_synthetic_clip(tmp_path: Path):
    clip = _make_clip(tmp_path / "demo.mp4")
    described = describe_clip(CLIP_PREPARATION, 1, clip)
    assert described.audio_codec == "aac"
    assert described.duration_seconds == pytest.approx(2.0,
                                                       abs=0.2)
    assert len(described.sha256) == 64
    session = build_session(
        [(CLIP_PREPARATION, 1, clip),
         (CLIP_MIXING, 2, clip)])
    assert session.inter_clip_gap == "unknown"
    assert session.ground_truth_recipe is None
    with pytest.raises(ManifestError):
        describe_clip("x", 1, tmp_path / "missing.mp4")
    with pytest.raises(ManifestError):
        build_session([(CLIP_PREPARATION, 1, clip),
                       (CLIP_PREPARATION, 2, clip)])


def _event(source: str, clip: str | None, start: float,
           end: float, action: str) -> dict[str, Any]:
    return {"event_id": f"{source}-{clip}-{start}-{action}",
            "source": source, "clip_id": clip,
            "action_canonical": action,
            "start_seconds": start, "end_seconds": end,
            "approximate": False, "ingredients": [],
            "uncertainty": None}


def test_cross_clip_events_never_group():
    left = _event("s1", "clip-a", 0.0, 10.0, "pour")
    right = _event("s2", "clip-b", 0.0, 10.0, "pour")
    groups = reconcile.group_events([left, right])
    assert len(groups) == 2
    assert reconcile.detect_conflicts([left, right], [], []) == []
    same = _event("s2", "clip-a", 0.0, 10.0, "pour")
    assert len(reconcile.group_events([left, same])) == 1


def test_measurements_key_alias():
    one = normalize.spark_measurements(
        {"scale_readings": [{
            "timestamp_seconds": 5.0, "displayed_value": "10"}]},
        None, None)
    two = normalize.spark_measurements(
        {"measurements": [{
            "timestamp_seconds": 5.0, "displayed_value": "10"}]},
        None, None)
    assert len(one) == len(two) == 1
    assert one[0]["reading_id"] == two[0]["reading_id"]


def test_group_order_follows_clip_order():
    early = _event("s1", "clip-b", 0.0, 5.0, "pour")
    late = _event("s1", "clip-a", 100.0, 105.0, "stir")
    groups = reconcile.group_events(
        [early, late], {"clip-a": 1, "clip-b": 2})
    assert [group["clip_id"] for group in groups] == [
        "clip-a", "clip-b"]


def test_critic_accepts_speech_and_clip_refs():
    draft = {"steps": [{"group_id": "eg_000"}],
             "ingredients": [],
             "speech": [{"speech_id": "fs_1"}],
             "session": {"clips": [{"clip_id": "clip-a"}]},
             "quantity_assessment": {
                 "stable_clusters": [{"cluster_id": "qc_1"}]}}
    recipe = {"ingredients": [], "steps": [{
        "text": "mix", "evidence_refs": ["eg_000", "fs_1",
                                         "clip-a", "qc_1"]}],
        "open_questions": []}
    assert orchestrator.deterministic_critic(
        recipe, draft)["verdict"] == "needs_review"


def test_build_steps_orders_clips_before_starts():
    groups = [
        {"group_id": "eg_001", "clip_id": "clip-b",
         "start_seconds": 0.0, "end_seconds": 5.0,
         "sources": ["s"], "source_count": 1,
         "member_ids": ["e1"], "members": [],
         "preliminary_links": []},
        {"group_id": "eg_000", "clip_id": "clip-a",
         "start_seconds": 100.0, "end_seconds": 105.0,
         "sources": ["s"], "source_count": 1,
         "member_ids": ["e2"], "members": [],
         "preliminary_links": []},
    ]
    from bakery_intelligence.fusion import reconstruct

    steps = reconstruct.build_steps(
        groups, {"clip-a": 1, "clip-b": 2})
    assert [step["group_id"] for step in steps] == [
        "eg_000", "eg_001"]
    legacy = reconstruct.build_steps(
        [{**group, "clip_id": None} for group in groups])
    assert [step["group_id"] for step in legacy] == [
        "eg_001", "eg_000"]


def test_ingredient_buckets_are_clip_scoped():
    records = [
        {"record_id": "r1", "source": "s1", "clip_id": "a",
         "normalized_name": "flour", "visible_label": None,
         "used_or_visible": None, "ambiguities": [],
         "supporting_timestamps": [1.0]},
        {"record_id": "r2", "source": "s1", "clip_id": "b",
         "normalized_name": "flour", "visible_label": None,
         "used_or_visible": None, "ambiguities": [],
         "supporting_timestamps": [1.0]},
    ]
    groups = reconcile.group_ingredients(records)
    assert len(groups) == 2
    assert {group["clip_id"] for group in groups} == {"a", "b"}


def test_speech_links_same_clip_overlap_only():
    groups = [{"group_id": "eg_000", "clip_id": "a",
               "start_seconds": 10.0, "end_seconds": 20.0,
               "members": [], "speech_links": []}]
    turns = normalize.speech_turns(
        [{"transcript": "add the flour", "start_seconds": 12.0,
          "end_seconds": 14.0, "speaker": "S1"}], "a")
    other = normalize.speech_turns(
        [{"transcript": "add the flour", "start_seconds": 12.0,
          "end_seconds": 14.0}], "b")
    reconcile.link_speech(groups, turns + other)
    assert groups[0]["speech_links"] == [turns[0]["speech_id"]]
    assert turns[0]["speech_id"].startswith("fs_")
    assert normalize.speech_turns(
        [{"transcript": "  ", "start_seconds": 1.0}], "a") == []


def test_window_and_frame_selection_is_bounded():
    operations = [
        {"action_type": "pour", "action_description": "x",
         "evidence": "scale shows 100", "uncertainty": None,
         "start_seconds": 10.0, "end_seconds": 200.0},
        {"action_type": "stir", "action_description": "y",
         "evidence": "", "uncertainty": None,
         "start_seconds": 210.0, "end_seconds": 220.0},
    ]
    windows = orchestrator.select_windows(operations, 256.0)
    assert len(windows) <= orchestrator.MAX_WINDOWS_PER_CLIP
    assert windows[0][1] - windows[0][0] <= 60.0
    stamps = orchestrator.select_frames(
        operations, [{"timestamp_seconds": 50.0}], 256.0)
    assert 50.0 in stamps
    assert len(stamps) <= orchestrator.MAX_FRAMES_PER_CLIP


def test_deterministic_critic_requires_provenance():
    draft = {"steps": [{"group_id": "eg_000"}],
             "ingredients": [{"group_id": "ig_1"}]}
    good = {"ingredients": [{"name": "flour",
                             "evidence_refs": ["ig_1"]}],
            "steps": [{"text": "mix",
                       "evidence_refs": ["eg_000"]}],
            "open_questions": []}
    assert orchestrator.deterministic_critic(
        good, draft)["verdict"] == "needs_review"
    bad = {"ingredients": [{"name": "saffron",
                            "evidence_refs": []}],
           "steps": [], "open_questions": []}
    verdict = orchestrator.deterministic_critic(bad, draft)
    assert verdict["verdict"] == "unsupported"


def test_transient_retry_classification():
    assert orchestrator._transient("Responses call failed (504)")
    assert orchestrator._transient("request timed out")
    assert not orchestrator._transient("Schema violations: x")
    assert not orchestrator._transient("400 bad request")
    assert not orchestrator._transient("budget exceeded")


def test_budget_ledger_blocks_over_ceiling(tmp_path: Path):
    ledger = BudgetLedger(path=tmp_path / "ledger.json",
                          budget_usd=1.0,
                          max_single_call_usd=0.5)
    ledger.check(0.4, "ok-call")
    with pytest.raises(BudgetExceeded):
        ledger.check(0.6, "too-big")
    ledger.record({"agent": "A", "estimate_usd": 0.4,
                   "actual_usd": 0.4})
    assert ledger.remaining() == pytest.approx(0.6)
    with pytest.raises(BudgetExceeded):
        ledger.check(0.7, "over-remaining")


def test_p04_v6_byte_identity(tmp_path: Path):
    """Fusion changes must not alter the P04 v6 draft bytes."""
    from bakery_intelligence.fusion.run import run_fusion

    reference = Path("runs_api/fusion/p04_104_v6/fusion_draft.json")
    if not reference.is_file():
        pytest.skip("P04 v6 draft absent")
    expected = json.loads(reference.read_text(encoding="utf-8"))
    result = run_fusion(  # noqa: E501
        Path("experiments/native_video_understanding/"
             "p04_104_1080p_crf25_20261103T031547Z/response.json"),
        Path("experiments/native_video_baseten/segmented_runs/"
             "p04_104_720p_crf23_20261103T043654Z_269015/"
             "aggregated.json"),
        Path("experiments/003_video_to_recipe/annotations/"
             "preliminary/P04_104.json"),
        tmp_path, "identity")
    actual = json.loads(
        (tmp_path / "identity" / "fusion_draft.json").read_text(
            encoding="utf-8"))
    expected.pop("provenance", None)
    actual.pop("provenance", None)
    assert actual == expected


GLOBAL_FIXTURE = {
    "overall_activity": "mixing dough",
    "operations": [{
        "event_id": "op1", "start_seconds": 0.0,
        "end_seconds": 1.5, "action_type": "pour",
        "action_description": "pour flour near the scale",
        "ingredient_candidates": ["flour"],
        "containers_utensils": ["bowl"],
        "evidence": "hand pours", "confidence": "high",
        "uncertainty": None}],
    "ingredients": [{
        "visible_label": "flour bag", "normalized_name": "flour",
        "supporting_timestamps": [0.5],
        "visual_evidence": "bag visible",
        "ambiguities": [], "used_or_visible": "used"}],
    "measurements": [{
        "timestamp_seconds": 1.0, "displayed_value": "250",
        "visible_unit": None, "tare_observable": False,
        "uncertain": False, "evidence": "digits clear"}],
    "uncertainties": [], "limitations": [],
}


def _mock_provider(request: Any) -> Any:
    import httpx

    path = request.url.path
    assert path in ("/v1/files", "/v1/responses",
                    "/v1/asr/transcribe"), f"bad path: {path}"
    if path == "/v1/files":
        return httpx.Response(200, json={"id": "file_demo"})
    if path == "/v1/asr/transcribe":
        return httpx.Response(200, json={"turns": [{
            "turnId": 0, "speaker": "S1", "startMs": 100,
            "endMs": 1200, "transcript": "add two cups of flour"}]})
    body = json.loads(request.content.decode("utf-8"))
    text = json.dumps(body.get("input", ""))
    if "Analyze this complete" in text:
        parsed = GLOBAL_FIXTURE
    elif "short video segment" in text:
        parsed = {**GLOBAL_FIXTURE,
                  "segment_start_seconds": 0.0,
                  "segment_end_seconds": 1.5}
    elif "scale display" in text:
        parsed = {"readings": [{
            "frame_ref": "f0", "clip_local_seconds": 1.0,
            "displayed_value_verbatim": "250",
            "visible_unit": None,
            "tare_button_visible": False,
            "container_description": "bowl",
            "container_stable": True, "evidence": "clear",
            "uncertain": False}]}
    elif "Read the transcript" in text:
        parsed = {"claims": [{
            "claim_id": "c1", "turn_ids": [0],
            "claim_kind": "ingredient_quantity_utterance",
            "verbatim_quote": "add two cups of flour",
            "speaker": "S1", "clip_local_start": 0.1,
            "clip_local_end": 1.2,
            "interpretation": "flour mentioned",
            "uncertainty": None}]}
    elif "Draft a recipe" in text:
        parsed = {"title": "Demo dough", "ingredients": [],
                  "steps": [], "open_questions": ["q1"]}
    else:
        parsed = {"verdict": "needs_review", "findings": []}
    return httpx.Response(200, json={
        "id": "resp_demo",
        "output": [{"content": [{"text": json.dumps(parsed)}]}],
        "usage": {"input_tokens": 100, "output_tokens": 50}})


def test_orchestrator_dry_run_end_to_end(tmp_path: Path,
                                         monkeypatch: Any):
    import httpx

    from bakery_intelligence.phase2c.clients import (
        MetaResponsesClient,
        VoiceTranscribeClient,
    )
    from bakery_intelligence.phase2c.orchestrator import (
        Orchestrator,
    )

    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    clip = _make_clip(tmp_path / "demo.mp4")
    session = build_session(
        [(CLIP_PREPARATION, 1, clip),
         (CLIP_MIXING, 2, clip)])
    transport = httpx.MockTransport(_mock_provider)
    orch = Orchestrator(
        tmp_path,
        meta=MetaResponsesClient(transport=transport),
        asr=VoiceTranscribeClient(transport=transport))
    clips = {clip_id: orch.run_clip(session, clip_id)
             for clip_id in (CLIP_PREPARATION, CLIP_MIXING)}
    assert clips[CLIP_PREPARATION]["agent_c_turns"][0][
        "transcript"] == "add two cups of flour"
    out = orch.run_session(session, clips)
    draft = json.loads(
        (out / "fusion_draft.json").read_text(encoding="utf-8"))
    assert draft["session"]["inter_clip_gap"] == "unknown"
    assert len(draft["session"]["clips"]) == 2
    assert draft["speech"], "transcript must fuse into the draft"
    assert draft["recipe_proposal"]["parsed"]["title"] == \
        "Demo dough"
    assert draft["critic_verdict"]["deterministic"]["verdict"] \
        in ("supported", "needs_review", "unsupported")
    assert all(step.get("clip_id") in
               (CLIP_PREPARATION, CLIP_MIXING)
               for step in draft["steps"])
    # Resume: second run reuses artifacts without new billing.
    calls_before = len(orch.ledger.calls)
    orch.run_clip(session, CLIP_PREPARATION)
    assert len(orch.ledger.calls) == calls_before
    assert orch.status()["stages"][
        f"{CLIP_PREPARATION}/agent-a"]["state"] == "ok"
