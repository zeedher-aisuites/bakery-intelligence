"""Offline tests for the multimodal video pipeline.

All provider clients are fakes implementing the real call
signatures (Meta Responses payloads, Baseten chat dicts, ASR
transcripts). No network, no inference, no billable cost.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from bakery_intelligence.api import multimodal as mm
from bakery_intelligence.api.authorization import (
    GrantError,
    confirmation_sentence,
    parse_grant,
)
from bakery_intelligence.api.jobs import JobRecord
from bakery_intelligence.phase2c import prompts


class FakeMeta:
    """Fake MetaResponsesClient: queued Responses payloads."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.uploads: list[Path] = []
        self.calls: list[dict[str, Any]] = []

    def upload_file(self, path: Path,
                    purpose: str = "user_data") -> str:
        self.uploads.append(path)
        return "file_fake1"

    def create_response(self, model: str,
                        content: list[dict[str, Any]],
                        max_output_tokens: int = 16384,
                        ) -> dict[str, Any]:
        self.calls.append({
            "model": model, "content": content,
            "max_output_tokens": max_output_tokens})
        text = self._texts.pop(0)
        return {
            "id": f"resp_fake{len(self.calls)}",
            "output_text": text,
            "usage": {"input_tokens": 100,
                      "output_tokens": 50},
            "_elapsed_seconds": 0.5,
        }


class FakeBaseten:
    """Fake BasetenClient: queued chat-completion dicts."""

    model = mm.BASETEN_MODEL

    def __init__(self, contents: list[str]) -> None:
        self._contents = list(contents)
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, Any]],
             max_tokens: int) -> dict[str, Any]:
        self.calls.append({
            "messages": messages, "max_tokens": max_tokens})
        content = self._contents.pop(0)
        return {
            "id": f"chat_fake{len(self.calls)}",
            "model": mm.BASETEN_MODEL,
            "content": content,
            "usage": {"input_tokens": 200,
                      "output_tokens": 60,
                      "total_tokens": 260},
            "elapsed_seconds": 1.5,
        }


def _global_text(**overrides: Any) -> str:
    payload = {
        "overall_activity": "baking prep",
        "operations": [{"event_id": "op1",
                        "start_seconds": 0.0,
                        "end_seconds": 5.0,
                        "description": "weighing flour"}],
        "ingredients": [], "measurements": [],
        "uncertainties": [], "limitations": [],
    }
    payload.update(overrides)
    return json.dumps(payload)


def _segment_text(index: int) -> str:
    return json.dumps({
        "operations": [],
        "ingredients": [],
        "measurements": [],
        "segment_start_seconds": index * 30.0,
        "segment_end_seconds": index * 30.0 + 30.0,
        "events": [{"start_seconds": index * 30.0 + 1.0,
                    "note": f"event in {index}"}],
        "summary": f"segment {index} summary",
    })


def _validation_text() -> str:
    return json.dumps({
        "contradictions": [],
        "checks": [{"name": "readings sane",
                    "passed": True, "detail": "ok"}],
        "open_questions": ["unit unknown"],
    })


def _proposal(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "meaningful": True,
        "reason": "Readings and events support a draft.",
        "title": "Cinnamon Rolls",
        "ingredients": [
            {"name": "flour", "quantity": 700.0,
             "unit": "g", "provenance": "seen_video",
             "basis": "Display showed 700 at t=10s."},
            {"name": "cinnamon", "quantity": None,
             "unit": None, "provenance": "ai_estimated",
             "basis": "Culinary estimate; no reading."},
        ],
        "method": [
            {"number": 1, "instruction": "Weigh the flour."},
            {"number": 2, "instruction": "Mix and bake."},
        ],
        "notes": ["Confirm cinnamon to taste."],
        "suggestions": [],
        "open_questions": [],
    }
    base.update(overrides)
    return base


def _merged() -> dict[str, Any]:
    return {
        "video": {}, "readings_total": 1,
        "weighing": {}, "unresolved": [],
        "global_result": {}, "segment_count": 1,
        "transcript_turns": 0,
        "items": [
            {"source": "frames", "kind": "scale_reading",
             "t": 10.0, "value": 700.0, "unit": "g",
             "model": mm.SPARK_MODEL, "provider": "meta"},
        ],
    }


def _validation_record() -> dict[str, Any]:
    return {
        "model": mm.SPARK_MODEL, "provider": "meta",
        "prompt_version":
        prompts.EVIDENCE_VALIDATOR.version,
        "request_id": "resp_fake1", "usage": {},
        "result": json.loads(_validation_text()),
    }


def test_parse_json_strict_roundtrip() -> None:
    assert mm.parse_json_strict('{"a": 1}', "X") == {"a": 1}


def test_parse_json_strict_rejects_prose() -> None:
    with pytest.raises(mm.MultimodalError,
                       match="non-JSON output"):
        mm.parse_json_strict("not json at all", "Global")


def test_parse_json_tolerant_strips_fences() -> None:
    text = '```json\n{"a": 1}\n```'
    assert mm.parse_json_tolerant(text, "X") == {"a": 1}


def test_parse_json_tolerant_rejects_prose() -> None:
    with pytest.raises(mm.MultimodalError,
                       match="returned non-JSON output"):
        mm.parse_json_tolerant("```\nstill prose\n```",
                               "Segment 0")


def test_segment_windows_cover_timeline() -> None:
    assert mm.segment_windows(65.0) == [
        (0.0, 30.0), (30.0, 60.0), (60.0, 65.0)]


def test_segment_windows_reject_zero() -> None:
    with pytest.raises(mm.MultimodalError,
                       match="zero-length"):
        mm.segment_windows(0.0)


def test_asr_chunk_windows_split_long_audio() -> None:
    assert mm.asr_chunk_windows(644.57) == [
        (0.0, 330.0), (330.0, 644.57)]
    assert mm.asr_chunk_windows(700.0) == [
        (0.0, 330.0), (330.0, 660.0), (660.0, 700.0)]


def test_asr_chunk_windows_short_audio_single_call() -> None:
    assert mm.asr_chunk_windows(300.0) == [(0.0, 300.0)]
    assert mm.asr_chunk_windows(330.0) == [(0.0, 330.0)]


def test_asr_chunk_windows_reject_zero() -> None:
    with pytest.raises(mm.MultimodalError,
                       match="zero-length"):
        mm.asr_chunk_windows(0.0)


def test_non_frame_reserve_scales_with_chunks() -> None:
    assert mm.non_frame_reserve(300.0) == 4
    assert mm.non_frame_reserve(330.0) == 4
    assert mm.non_frame_reserve(644.57) == 5
    assert mm.non_frame_reserve(700.0) == 6


def _chunk_payload(text: str, turns: list[dict[str, Any]],
                   session: str) -> dict[str, Any]:
    return {"audioDurationMs": 330000, "sessionId": session,
            "transcript": text, "turns": turns,
            "_elapsed_seconds": 1.0}


def test_merge_transcripts_offsets_and_resequences() -> None:
    merged = mm.merge_transcripts(644.57, [
        (0.0, _chunk_payload("hello there", [
            {"turnId": 0, "speaker": "A", "startMs": 220,
             "endMs": 8060, "transcript": "hello there"}],
            "s0")),
        (330.0, _chunk_payload("bake now", [
            {"turnId": 0, "speaker": "B", "startMs": 500,
             "endMs": 1500, "transcript": "bake now"}],
            "s1")),
    ])
    assert merged["audioDurationMs"] == 644570
    assert merged["transcript"] == "hello there bake now"
    assert merged["chunked"] is True
    assert merged["chunk_session_ids"] == ["s0", "s1"]
    assert [t["turnId"] for t in merged["turns"]] == [0, 1]
    assert merged["turns"][0]["startMs"] == 220
    assert merged["turns"][1]["startMs"] == 330500
    assert merged["turns"][1]["endMs"] == 331500


def test_merge_transcripts_tolerates_empty_chunk() -> None:
    merged = mm.merge_transcripts(400.0, [
        (0.0, _chunk_payload("", [], "s0")),
        (330.0, _chunk_payload("only here", [
            {"turnId": 0, "speaker": "A", "startMs": 0,
             "endMs": 100, "transcript": "only here"}],
            "s1")),
    ])
    assert len(merged["turns"]) == 1
    assert merged["transcript"] == "only here"


def test_merge_transcripts_reject_bad_turns() -> None:
    with pytest.raises(mm.MultimodalError,
                       match="no turns list"):
        mm.merge_transcripts(
            330.0, [(0.0, {"turns": None})])
    with pytest.raises(mm.MultimodalError,
                       match="without numeric startMs"):
        mm.merge_transcripts(330.0, [(0.0, _chunk_payload(
            "x", [{"turnId": 0, "speaker": "A",
                   "endMs": 5, "transcript": "x"}],
            "s0"))])


class FakeASR:
    """Scripted VoiceTranscribeClient."""

    def __init__(self, payloads: list[Any]) -> None:
        self._payloads = list(payloads)
        self.paths: list[Path] = []

    def transcribe(self, audio_path: Path,
                   mode: str = "DIARIZATION") -> Any:
        self.paths.append(audio_path)
        assert mode == "DIARIZATION"
        next_item = self._payloads.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item


def _claims_text() -> str:
    return json.dumps({"claims": [{
        "claim_id": "c1", "turn_ids": [0],
        "claim_kind": "instruction", "speaker": "A",
        "clip_local_start": 0.2, "clip_local_end": 8.0,
        "interpretation": "mix", "uncertainty": "low"}]})


def _audio_harness(tmp_path: Path,
                   monkeypatch: pytest.MonkeyPatch,
                   duration: float) -> None:
    import bakery_intelligence.video.intake as intake

    monkeypatch.setattr(
        intake, "probe_video",
        lambda path: {"duration_seconds": duration})
    monkeypatch.setattr(mm, "has_audio_stream",
                        lambda path: True)
    monkeypatch.setattr(
        mm, "split_wav",
        lambda src, start, length, dest: dest.write_bytes(
            b"wav"))
    (tmp_path / "audio.wav").write_bytes(b"fake-wav")


def test_audio_stage_short_audio_single_call(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _audio_harness(tmp_path, monkeypatch, 300.0)
    asr = FakeASR([_chunk_payload("hello", [
        {"turnId": 0, "speaker": "A", "startMs": 0,
         "endMs": 1000, "transcript": "hello"}],
        "s0")])
    meta = FakeMeta([_claims_text()])
    transcript, claims, calls = mm.run_audio_stage(
        meta, asr, Path("clip.mp4"),  # type: ignore[arg-type]
        tmp_path)
    assert asr.paths == [
        tmp_path / "asr_chunks" / "chunk_000.wav"]
    assert transcript is not None
    assert transcript["chunked"] is False
    assert transcript["turns"][0]["startMs"] == 0
    assert claims is not None
    assert [c.stage for c in calls] == [
        "audio_transcribe_chunk_000", "speech_claims"]
    assert all(c.status == "ok" for c in calls)


def test_audio_stage_resumes_completed_chunk(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _audio_harness(tmp_path, monkeypatch, 400.0)
    chunks = tmp_path / "asr_chunks"
    chunks.mkdir()
    (chunks / "chunk_000.json").write_text(json.dumps({
        "index": 0, "start_seconds": 0.0,
        "end_seconds": 330.0,
        "payload": _chunk_payload("first part", [
            {"turnId": 0, "speaker": "A", "startMs": 0,
             "endMs": 1000, "transcript": "first part"}],
            "s0")}), encoding="utf-8")
    asr = FakeASR([_chunk_payload("second part", [
        {"turnId": 0, "speaker": "B", "startMs": 500,
         "endMs": 900, "transcript": "second part"}],
        "s1")])
    meta = FakeMeta([_claims_text()])
    transcript, claims, calls = mm.run_audio_stage(
        meta, asr, Path("clip.mp4"),  # type: ignore[arg-type]
        tmp_path)
    assert len(asr.paths) == 1
    assert asr.paths[0].name == "chunk_001.wav"
    assert transcript is not None
    assert transcript["chunked"] is True
    assert transcript["turns"][1]["startMs"] == 330500
    assert [c.stage for c in calls] == [
        "audio_transcribe_chunk_000",
        "audio_transcribe_chunk_001", "speech_claims"]
    assert calls[0].status == "reused"
    assert calls[0].estimate_usd == 0.0
    assert calls[1].status == "ok"


def test_audio_stage_chunk_failure_preserves_prior(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _audio_harness(tmp_path, monkeypatch, 400.0)
    asr = FakeASR([
        _chunk_payload("first part", [
            {"turnId": 0, "speaker": "A", "startMs": 0,
             "endMs": 1000, "transcript": "first part"}],
            "s0"),
        RuntimeError("provider 400"),
    ])
    meta = FakeMeta([_claims_text()])
    with pytest.raises(mm.MultimodalError,
                       match=r"ASR chunk 1 \(330\.0s\) failed"):
        mm.run_audio_stage(
            meta, asr, Path("clip.mp4"),  # type: ignore[arg-type]
            tmp_path)
    assert (tmp_path / "asr_chunks" / "chunk_000.json"
            ).is_file()
    failed = json.loads(
        (tmp_path / "asr_chunk_001_failed.json").read_text(
            encoding="utf-8"))
    assert failed["start_seconds"] == 330.0
    assert "RuntimeError" in failed["error"]


def test_audio_stage_reuses_transcript_and_claims(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _audio_harness(tmp_path, monkeypatch, 400.0)
    (tmp_path / "transcript.json").write_text(json.dumps({
        "audioDurationMs": 400000, "sessionId": "s0",
        "transcript": "all done", "turns": [
            {"turnId": 0, "speaker": "A", "startMs": 0,
             "endMs": 1000, "transcript": "all done"}],
        "chunked": True, "chunks": [
            {"index": 0, "start_seconds": 0.0,
             "end_seconds": 330.0, "request_id": None},
            {"index": 1, "start_seconds": 330.0,
             "end_seconds": 400.0, "request_id": None}],
    }), encoding="utf-8")
    (tmp_path / "speech_claims.json").write_text(json.dumps({
        "model": mm.SPARK_MODEL, "provider": "meta",
        "prompt_version": "v1.0", "request_id": "resp_1",
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "result": {"claims": []}}), encoding="utf-8")
    asr = FakeASR([])
    meta = FakeMeta([])
    transcript, claims, calls = mm.run_audio_stage(
        meta, asr, Path("clip.mp4"),  # type: ignore[arg-type]
        tmp_path)
    assert asr.paths == []
    assert meta.calls == []
    assert transcript is not None
    assert claims is not None
    assert [c.stage for c in calls] == [
        "audio_transcribe_chunk_000",
        "audio_transcribe_chunk_001", "speech_claims"]
    assert all(c.status == "reused" for c in calls)
    assert calls[2].usage["input_tokens"] == 10


def test_video_part_shape() -> None:
    part = mm.video_part(b"\x00\x01")
    assert part["type"] == "video_url"
    assert part["video_url"]["url"].startswith(
        "data:video/mp4;base64,")


def test_stage_call_rounds_elapsed() -> None:
    call = mm.StageCall(
        stage="global", provider="meta",
        model=mm.SPARK_MODEL, request_id="r1",
        status="ok", elapsed_seconds=1.23456)
    assert call.as_dict()["elapsed_seconds"] == 1.235


def test_frame_stage_calls_surface_report_totals() -> None:
    [call] = mm.frame_stage_calls({
        "requests_sent": 8, "batches_completed": 8,
        "input_tokens": 1000, "output_tokens": 500,
        "cost_usd_estimated": 0.003375})
    assert call.stage == "frames_firstpass"
    assert call.provider == "meta"
    assert call.usage["requests_sent"] == 8
    assert call.usage["input_tokens"] == 1000
    assert call.estimate_usd == 0.003375
    assert call.error is None


def test_frame_stage_calls_missing_estimate() -> None:
    [call] = mm.frame_stage_calls({
        "requests_sent": 2, "cost_usd_estimated": None})
    assert call.estimate_usd is None
    assert call.error == "usage in firstpass_report.json"


def test_baseten_model_id_default(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(mm.BASETEN_MODEL_VAR,
                       raising=False)
    assert mm.baseten_model_id() == mm.BASETEN_MODEL


def test_baseten_model_id_rejects_override(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(mm.BASETEN_MODEL_VAR, "other-model")
    with pytest.raises(mm.MultimodalError,
                       match="must be exactly"):
        mm.baseten_model_id()


def test_baseten_client_disables_sdk_retries(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """One explicit request: no hidden duplicate attempts."""
    monkeypatch.setenv(mm.BASETEN_KEY_VAR, "test-key")
    monkeypatch.delenv(mm.BASETEN_MODEL_VAR, raising=False)
    client = mm.BasetenClient(timeout_seconds=1.0)
    assert client._client.max_retries == 0  # noqa: SLF001


def test_get_validator_unknown() -> None:
    with pytest.raises(mm.MultimodalError,
                       match="Unknown validator adapter"):
        mm.get_validator("glimmer", None)  # type: ignore[arg-type]


def test_get_validator_default_is_spark() -> None:
    fake = FakeMeta([_validation_text()])
    validator = mm.get_validator(mm.DEFAULT_VALIDATOR, fake)  # type: ignore[arg-type]
    assert isinstance(validator, mm.SparkValidationAdapter)
    assert validator.model_id == mm.SPARK_MODEL
    assert validator.provider == "meta"


def test_spark_validator_records_true_identity() -> None:
    fake = FakeMeta([_validation_text()])
    validator = mm.SparkValidationAdapter(fake)  # type: ignore[arg-type]
    record = validator.validate(_merged())
    assert record["model"] == "muse-spark-1.3"
    assert record["provider"] == "meta"
    assert record["prompt_version"] == "v1.0"
    assert record["request_id"] == "resp_fake1"
    assert "glimmer" not in json.dumps(record).lower()
    assert fake.calls[0]["max_output_tokens"] == \
        mm.VALIDATION_MAX_TOKENS


def test_spark_validator_rejects_bad_schema() -> None:
    fake = FakeMeta([json.dumps({"contradictions": []})])
    validator = mm.SparkValidationAdapter(fake)  # type: ignore[arg-type]
    with pytest.raises(mm.MultimodalError,
                       match="Invalid validation payload"):
        validator.validate(_merged())


def test_spark_validator_rejects_prose() -> None:
    fake = FakeMeta(["looks fine to me"])
    validator = mm.SparkValidationAdapter(fake)  # type: ignore[arg-type]
    with pytest.raises(mm.MultimodalError,
                       match="non-JSON output"):
        validator.validate(_merged())


def test_global_stage_uses_native_video(
        tmp_path: Path) -> None:
    fake = FakeMeta([_global_text()])
    record, call = mm.run_global_stage(
        fake, Path("clip.mp4"), 65.0, tmp_path)  # type: ignore[arg-type]
    assert fake.uploads == [Path("clip.mp4")]
    sent = fake.calls[0]["content"]
    assert {"type": "input_file",
            "file_id": "file_fake1"} in sent
    assert "65.0s" in sent[0]["text"]
    assert fake.calls[0]["max_output_tokens"] == \
        mm.GLOBAL_MAX_TOKENS
    assert record["model"] == mm.SPARK_MODEL
    assert record["provider"] == "meta"
    assert record["request_id"] == "resp_fake1"
    assert record["result"]["overall_activity"] == \
        "baking prep"
    assert (tmp_path / "global.json").is_file()
    assert call.stage == "global"
    assert call.estimate_usd is not None
    assert call.estimate_usd > 0


def test_global_stage_rejects_bad_schema(
        tmp_path: Path) -> None:
    fake = FakeMeta([json.dumps({"operations": []})])
    with pytest.raises(mm.MultimodalError,
                       match="Invalid global payload"):
        mm.run_global_stage(
            fake, Path("clip.mp4"), 65.0,  # type: ignore[arg-type]
            tmp_path)


def test_audio_stage_skips_without_audio(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import bakery_intelligence.video.intake as intake

    monkeypatch.setattr(
        intake, "probe_video",
        lambda path: {"duration_seconds": 10.0})
    monkeypatch.setattr(mm, "has_audio_stream",
                        lambda path: False)
    transcript, claims, calls = mm.run_audio_stage(
        FakeMeta([]), object(),  # type: ignore[arg-type]
        Path("clip.mp4"), tmp_path)
    assert transcript is None
    assert claims is None
    assert calls == []
    stored = json.loads(
        (tmp_path / "transcript.json").read_text(
            encoding="utf-8"))
    assert stored["status"] == "skipped"


def test_segment_stage_absolute_windows_and_context(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cuts: list[tuple[float, float]] = []

    def fake_cut(src: Path, start: float, length: float,
                 dest: Path) -> None:
        cuts.append((start, length))
        dest.write_bytes(b"clip" + bytes([len(cuts)]))

    monkeypatch.setattr(mm, "cut_subclip", fake_cut)
    fake = FakeBaseten([_segment_text(0), _segment_text(1),
                        _segment_text(2)])
    index, calls = mm.run_segment_stage(
        fake, Path("clip.mp4"), 65.0,  # type: ignore[arg-type]
        "global summary", tmp_path)
    assert [c for c in cuts] == [
        (0.0, 30.0), (30.0, 30.0), (60.0, 5.0)]
    assert index["count"] == 3
    assert index["provider"] == "baseten"
    assert index["model"] == mm.BASETEN_MODEL
    assert [s["start_seconds"]
            for s in index["segments"]] == [0.0, 30.0, 60.0]
    assert len(calls) == 3
    assert all(c.provider == "baseten" for c in calls)
    assert all(c.estimate_usd is None for c in calls)
    first_text = fake.calls[0]["messages"][0][
        "content"][0]["text"]
    assert "start 0.0s" in first_text
    assert "global summary" in first_text
    second_text = fake.calls[1]["messages"][0][
        "content"][0]["text"]
    assert "segment 0 summary" in second_text
    parts = fake.calls[0]["messages"][0]["content"]
    assert parts[1]["type"] == "video_url"
    stored = json.loads(
        (tmp_path / "segment_000.json").read_text(
            encoding="utf-8"))
    assert stored["request_id"] == "chat_fake1"
    assert stored["clip_sha256"]
    assert fake.calls[0]["max_tokens"] == \
        mm.SEGMENT_MAX_TOKENS


def test_segment_stage_rejects_bad_schema(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mm, "cut_subclip",
        lambda src, start, length, dest: dest.write_bytes(
            b"clip"))
    fake = FakeBaseten([json.dumps({"operations": []})])
    with pytest.raises(mm.MultimodalError,
                       match="Invalid segment 0 payload"):
        mm.run_segment_stage(
            fake, Path("clip.mp4"), 10.0,  # type: ignore[arg-type]
            "", tmp_path)


def test_segment_parse_failure_persists_diagnostics(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mm, "cut_subclip",
        lambda src, start, length, dest: dest.write_bytes(
            b"clip"))
    fake = FakeBaseten(["{truncated mid-string"])
    with pytest.raises(mm.MultimodalError,
                       match="non-JSON output"):
        mm.run_segment_stage(
            fake, Path("clip.mp4"), 10.0,  # type: ignore[arg-type]
            "", tmp_path)
    stored = json.loads(
        (tmp_path / "segment_000_failed.json").read_text(
            encoding="utf-8"))
    assert stored["request_id"] == "chat_fake1"
    assert stored["provider"] == "baseten"
    assert stored["usage"]["output_tokens"] == 60
    assert stored["content_head"] == "{truncated mid-string"


def test_merged_evidence_preserves_provenance() -> None:
    bundle = {
        "video": {"duration_seconds": 65.0},
        "readings": [{"t": 10.0, "value": 700.0,
                      "unit": "g"}],
        "visible_objects": [{"first_t": 1.0,
                             "object": "flour pack",
                             "count": 2}],
        "containers": [{"t": 1.0,
                        "description": "steel bowl"}],
        "weighing": {"tare_observed": True},
        "unresolved": ["UNIT_UNKNOWN"],
    }
    global_record = {
        "model": mm.SPARK_MODEL, "provider": "meta",
        "request_id": "resp_g",
        "result": {"operations": [
            {"start_seconds": 5.0,
             "description": "mixing"}]},
    }
    segments = [{
        "index": 0, "model": mm.BASETEN_MODEL,
        "provider": "baseten", "request_id": "chat_0",
        "result": {"events": [{"start_seconds": 1.0,
                               "note": "pour"}]},
    }]
    claims = {
        "model": mm.SPARK_MODEL, "provider": "meta",
        "request_id": "resp_s",
        "result": {"claims": [{"start_seconds": 3.0,
                               "quote": "700 grams"}]},
    }
    merged = mm.build_merged_evidence(
        bundle, global_record, segments, None, claims)
    kinds = {(i["source"], i["kind"]) for i in merged["items"]}
    assert ("frames", "scale_reading") in kinds
    assert ("frames", "visible_object") in kinds
    assert ("frames", "container") in kinds
    assert ("global_video", "operation") in kinds
    assert ("segment_000", "culinary_event") in kinds
    assert ("audio", "speech_claim") in kinds
    by_source = {i["source"]: i for i in merged["items"]}
    assert by_source["segment_000"]["request_id"] == \
        "chat_0"
    assert by_source["segment_000"]["model"] == \
        mm.BASETEN_MODEL
    assert by_source["global_video"]["request_id"] == \
        "resp_g"
    assert merged["readings_total"] == 1
    assert merged["weighing"] == {"tare_observed": True}
    assert merged["unresolved"] == ["UNIT_UNKNOWN"]


def test_fusion_stage_validates_proposal_schema(
        tmp_path: Path) -> None:
    fake = FakeBaseten([json.dumps(_proposal())])
    record, call = mm.run_fusion_stage(
        fake, _merged(), _validation_record(),  # type: ignore[arg-type]
        "Roles XL", "es", tmp_path)
    assert record["provider"] == "baseten"
    assert record["model"] == mm.BASETEN_MODEL
    assert record["prompt_version"] == "v1.1"
    assert record["request_id"] == "chat_fake1"
    assert record["result"]["title"] == "Cinnamon Rolls"
    assert call.stage == "fusion"
    assert call.estimate_usd is None
    assert mm.BASEN_COST_UNKNOWN in (call.error or "")
    assert (tmp_path / "fusion_input.json").is_file()
    stored = json.loads(
        (tmp_path / "fusion_recipe.json").read_text(
            encoding="utf-8"))
    assert stored["result"]["ingredients"][0][
        "quantity"] == 700.0


def test_fusion_stage_rejects_ungrounded_quantity(
        tmp_path: Path) -> None:
    bad = _proposal(ingredients=[
        {"name": "flour", "quantity": 123.0, "unit": "g",
         "provenance": "seen_video",
         "basis": "Invented reading."}])
    fake = FakeBaseten([json.dumps(bad)])
    with pytest.raises(mm.MultimodalError,
                       match="failed proposal validation"):
        mm.run_fusion_stage(
            fake, _merged(), _validation_record(),  # type: ignore[arg-type]
            "Roles XL", "es", tmp_path)


def _revision_text() -> str:
    return json.dumps({
        "title": "Cinnamon Rolls",
        "ingredients": [
            {"name": "flour", "quantity": 700.0,
             "unit": "g"},
            {"name": "cinnamon", "quantity": None,
             "unit": None},
        ],
        "method": [
            {"number": 1, "instruction": "Weigh the flour."},
        ],
        "notes": ["Confirm cinnamon to taste."],
        "changes_summary": "Tightened the wording.",
    })


def test_revision_stage_record_shape(
        tmp_path: Path) -> None:
    fake = FakeBaseten([_revision_text()])
    record, call = mm.run_revision_stage(
        fake, {"recipe_name": "R"},  # type: ignore[arg-type]
        tmp_path)
    assert record["provider"] == "baseten"
    assert record["model"] == mm.BASETEN_MODEL
    assert record["prompt_version"] == "v1.0"
    assert record["request_id"] == "chat_fake1"
    assert record["parsed"]["title"] == "Cinnamon Rolls"
    assert call.stage == "revision"
    assert call.estimate_usd is None
    assert mm.BASEN_COST_UNKNOWN in (call.error or "")
    assert (tmp_path / "revision_raw.json").is_file()
    assert fake.calls[0]["max_tokens"] == \
        mm.REVISION_MAX_TOKENS


def test_revision_stage_rejects_prose(
        tmp_path: Path) -> None:
    fake = FakeBaseten(["looks good to me"])
    with pytest.raises(mm.MultimodalError,
                       match="non-JSON output"):
        mm.run_revision_stage(
            fake, {"recipe_name": "R"},  # type: ignore[arg-type]
            tmp_path)
    stored = json.loads(
        (tmp_path / "revision_raw_failed.json").read_text(
            encoding="utf-8"))
    assert stored["request_id"] == "chat_fake1"
    assert stored["content_head"] == "looks good to me"


def test_revision_stage_rejects_bad_schema(
        tmp_path: Path) -> None:
    fake = FakeBaseten([json.dumps({"title": "T"})])
    with pytest.raises(mm.MultimodalError,
                       match="Invalid revision payload"):
        mm.run_revision_stage(
            fake, {"recipe_name": "R"},  # type: ignore[arg-type]
            tmp_path)


class FakeBasetenMeta(FakeBaseten):
    """Fake Baseten with finish reason and usage control."""

    def __init__(self, content: str, finish_reason: Any,
                 output_tokens: int) -> None:
        super().__init__([content])
        self._finish = finish_reason
        self._out = output_tokens

    def chat(self, messages: list[dict[str, Any]],
             max_tokens: int) -> dict[str, Any]:
        result = super().chat(messages, max_tokens)
        result["finish_reason"] = self._finish
        result["usage"] = {"input_tokens": 6613,
                           "output_tokens": self._out,
                           "total_tokens": 6613 + self._out}
        return result


def test_revision_stage_detects_truncated_empty_output(
        tmp_path: Path) -> None:
    """finish=length with no content names budget exhaustion.

    Regression: the live chef revision failed with
    8192/8192 output tokens and zero content bytes; the
    error must say the budget was exhausted, not merely
    that the output was non-JSON.
    """
    fake = FakeBasetenMeta("", "length", 8192)
    with pytest.raises(mm.MultimodalError,
                       match="exhausted the token budget"):
        mm.run_revision_stage(
            fake, {"recipe_name": "R"},  # type: ignore[arg-type]
            tmp_path)
    stored = json.loads(
        (tmp_path / "revision_raw_failed.json").read_text(
            encoding="utf-8"))
    assert stored["failure_category"] == "truncated"
    assert stored["usage"]["output_tokens"] == 8192
    assert (tmp_path / "revision_raw_failed.txt").is_file()


def test_revision_stage_detects_empty_stop_output(
        tmp_path: Path) -> None:
    fake = FakeBasetenMeta("   ", "stop", 12)
    with pytest.raises(mm.MultimodalError,
                       match="empty output"):
        mm.run_revision_stage(
            fake, {"recipe_name": "R"},  # type: ignore[arg-type]
            tmp_path)
    stored = json.loads(
        (tmp_path / "revision_raw_failed.json").read_text(
            encoding="utf-8"))
    assert stored["failure_category"] == "empty"


def test_revision_stage_marks_truncated_malformed_output(
        tmp_path: Path) -> None:
    fake = FakeBasetenMeta('{"title": "cut', "length", 16384)
    with pytest.raises(mm.MultimodalError,
                       match="non-JSON output"):
        mm.run_revision_stage(
            fake, {"recipe_name": "R"},  # type: ignore[arg-type]
            tmp_path)
    stored = json.loads(
        (tmp_path / "revision_raw_failed.json").read_text(
            encoding="utf-8"))
    assert stored["failure_category"] == "truncated"
    assert (tmp_path / "revision_raw_failed.txt"
            ).read_text(encoding="utf-8") == '{"title": "cut'


def test_revision_stage_uses_fusion_scale_budget() -> None:
    """Revision shares the evidence-backed 16384 budget."""
    assert mm.REVISION_MAX_TOKENS == 16384
    assert mm.REVISION_MAX_TOKENS == mm.FUSION_MAX_TOKENS


def _grant_payload(workflow: str) -> dict[str, Any]:
    confirmation = confirmation_sentence(
        workflow, "muse-spark-1.3", 10, 4096, 6,
        "tester")
    return {
        "grant_format": "bakery-billable-grant/1",
        "workflow": workflow, "provider": "meta",
        "model": "muse-spark-1.3", "max_requests": 10,
        "token_budget": 4096, "max_frames": 6,
        "operator": "tester",
        "created_at": "2026-11-03T00:00:00Z",
        "confirmation": confirmation,
    }


def test_executor_requires_grant() -> None:
    record = JobRecord(
        run_id="run_0123456789ab",
        workflow=mm.WORKFLOW_VIDEO_PIPELINE_MULTIMODAL,
        source_object_id="clip.mp4", grant_json=None)
    with pytest.raises(GrantError, match="requires an"):
        mm.execute_video_pipeline_multimodal(None, record)  # type: ignore[arg-type]


def test_executor_rejects_foreign_workflow_grant() -> None:
    payload = _grant_payload("video_pipeline_meta")
    assert parse_grant(payload).workflow == \
        "video_pipeline_meta"
    record = JobRecord(
        run_id="run_0123456789ab",
        workflow=mm.WORKFLOW_VIDEO_PIPELINE_MULTIMODAL,
        source_object_id="clip.mp4",
        grant_json=json.dumps(payload))
    with pytest.raises(GrantError, match="does not match"):
        mm.execute_video_pipeline_multimodal(None, record)  # type: ignore[arg-type]


def test_executor_requires_source() -> None:
    payload = _grant_payload(
        mm.WORKFLOW_VIDEO_PIPELINE_MULTIMODAL)
    record = JobRecord(
        run_id="run_0123456789ab",
        workflow=mm.WORKFLOW_VIDEO_PIPELINE_MULTIMODAL,
        source_object_id=None,
        grant_json=json.dumps(payload))
    with pytest.raises(mm.MultimodalError,
                       match="no source"):
        mm.execute_video_pipeline_multimodal(None, record)  # type: ignore[arg-type]


def test_linked_recipe_meta_defaults() -> None:
    assert mm._linked_recipe_meta(object(), "run_x") == (  # noqa: SLF001 - unit seam
        "Untitled recipe", "en")


def test_workflow_registered_billable_and_dispatched() -> None:
    from bakery_intelligence.api import jobs
    from bakery_intelligence.api import worker as worker_mod

    workflow = mm.WORKFLOW_VIDEO_PIPELINE_MULTIMODAL
    assert workflow == "video_pipeline_multimodal"
    assert workflow in jobs.SUPPORTED_WORKFLOWS
    assert workflow in worker_mod.BILLABLE_WORKFLOWS
    assert worker_mod._EXECUTORS[  # noqa: SLF001 - registry seam
        workflow] is mm.execute_video_pipeline_multimodal


def test_executor_path_a_uses_pipeline_contract(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Path A must call run_pipeline_from_path with its real kwargs.

    Regression: the executor once passed positional args plus
    unknown kwargs (billable_config, max_output_tokens), which
    can never reach a provider. Global runs first (1 call);
    the stub then reports 8 frame requests against a
    10-request grant, so execution must stop at the authorize
    gate AFTER Path A ran (1 + 8 + 3 > 10).
    """
    import bakery_intelligence.api.pipeline as pipeline_mod
    import bakery_intelligence.video.intake as intake
    from bakery_intelligence.api.storage import (
        LocalStorageBackend,
    )

    monkeypatch.setenv("META_API_KEY", "test-key")
    monkeypatch.setenv(mm.BASETEN_KEY_VAR, "test-baseten-key")
    monkeypatch.setattr(
        intake, "probe_video",
        lambda path: {"duration_seconds": 5.0})
    captured: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> dict[str, bytes]:
        captured.update(kwargs)
        return {"firstpass_report.json":
                b'{"requests_sent": 8}'}

    monkeypatch.setattr(
        pipeline_mod, "run_pipeline_from_path", fake_run)
    storage = LocalStorageBackend(tmp_path)
    sources = tmp_path / "sources"
    sources.mkdir(exist_ok=True)
    (sources / "clip.mp4").write_bytes(b"fake-video")
    payload = _grant_payload(
        mm.WORKFLOW_VIDEO_PIPELINE_MULTIMODAL)
    record = JobRecord(
        run_id="run_0123456789ab",
        workflow=mm.WORKFLOW_VIDEO_PIPELINE_MULTIMODAL,
        source_object_id="clip.mp4",
        grant_json=json.dumps(payload))
    meta = FakeMeta([_global_text()])
    with pytest.raises(mm.MultimodalError,
                       match="Grant allows 10 Meta calls"):
        mm.execute_video_pipeline_multimodal(
            storage, record, clients={"meta": meta})
    assert meta.uploads == [sources / "clip.mp4"]
    assert captured["video_path"] == sources / "clip.mp4"
    assert captured["video_name"] == "clip.mp4"
    assert captured["run_id"] == "run_0123456789ab"
    assert captured["billable_token_budget"] == 4096
    assert captured["billable_max_requests"] == \
        10 - mm.EXTRA_META_CALLS
    assert captured["billable_live"] is True
    assert captured["billable_api_key"] == "test-key"
    assert captured["max_frames"] == 6


def _smoke_executor(tmp_path: Path,
                    monkeypatch: pytest.MonkeyPatch):
    """Tmp storage + job record for executor-order tests."""
    import bakery_intelligence.video.intake as intake
    from bakery_intelligence.api.storage import (
        LocalStorageBackend,
    )

    monkeypatch.setenv("META_API_KEY", "test-key")
    monkeypatch.setenv(mm.BASETEN_KEY_VAR, "test-baseten-key")
    monkeypatch.setattr(
        intake, "probe_video",
        lambda path: {"duration_seconds": 5.0})
    storage = LocalStorageBackend(tmp_path)
    sources = tmp_path / "sources"
    sources.mkdir(exist_ok=True)
    (sources / "clip.mp4").write_bytes(b"fake-video")
    payload = _grant_payload(
        mm.WORKFLOW_VIDEO_PIPELINE_MULTIMODAL)
    record = JobRecord(
        run_id="run_0123456789ab",
        workflow=mm.WORKFLOW_VIDEO_PIPELINE_MULTIMODAL,
        source_object_id="clip.mp4",
        grant_json=json.dumps(payload))
    return storage, record


def test_executor_runs_global_before_frames(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail-fast order: one global call precedes frame batches."""
    import bakery_intelligence.api.pipeline as pipeline_mod

    storage, record = _smoke_executor(tmp_path, monkeypatch)
    order: list[str] = []

    class TracedMeta(FakeMeta):
        def upload_file(self, path: Path,
                        purpose: str = "user_data") -> str:
            order.append("global")
            return super().upload_file(path, purpose)

    def fake_run(**kwargs: Any) -> dict[str, bytes]:
        order.append("frames")
        return {"firstpass_report.json":
                b'{"requests_sent": 8}'}

    monkeypatch.setattr(
        pipeline_mod, "run_pipeline_from_path", fake_run)
    with pytest.raises(mm.MultimodalError,
                       match="Grant allows 10 Meta calls"):
        mm.execute_video_pipeline_multimodal(
            storage, record,
            clients={"meta": TracedMeta([_global_text()])})
    assert order == ["global", "frames"]


def test_reuse_segments_requires_complete_set(
        tmp_path: Path) -> None:
    (tmp_path / "segments_index.json").write_text(json.dumps({
        "count": 2, "segments": [{"index": 0}, {"index": 1}],
    }), encoding="utf-8")
    (tmp_path / "segment_000.json").write_text(json.dumps({
        "index": 0, "request_id": "c0", "usage": {},
        "result": {"events": []}}), encoding="utf-8")
    assert mm._reuse_segments(tmp_path) is None  # noqa: SLF001
    (tmp_path / "segment_001.json").write_text(json.dumps({
        "index": 1, "request_id": "c1", "usage": {},
        "result": {"events": []}}), encoding="utf-8")
    index, records = mm._reuse_segments(  # noqa: SLF001
        tmp_path) or (None, None)
    assert index is not None
    assert [r["request_id"] for r in records] == ["c0", "c1"]


def test_executor_reuses_global_without_call(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A persisted global.json means zero new global calls."""
    import bakery_intelligence.api.pipeline as pipeline_mod

    storage, record = _smoke_executor(tmp_path, monkeypatch)
    mm_dir = (tmp_path / "runs" / "run_0123456789ab"
              / "work" / "multimodal")
    mm_dir.mkdir(parents=True)
    (mm_dir / "global.json").write_text(json.dumps({
        "model": mm.SPARK_MODEL, "provider": "meta",
        "prompt_version": "v1.0", "request_id": "resp_old",
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "result": json.loads(_global_text()),
    }), encoding="utf-8")

    class ExplodingMeta(FakeMeta):
        def upload_file(self, path: Path,
                        purpose: str = "user_data") -> str:
            raise AssertionError(
                "global must not re-upload on resume")

        def create_response(self, *args: Any,
                            **kwargs: Any) -> Any:
            raise AssertionError(
                "global must not re-call on resume")

    def explode_frames(**kwargs: Any) -> dict[str, bytes]:
        raise AssertionError("stop here: frames reached")

    monkeypatch.setattr(
        pipeline_mod, "run_pipeline_from_path",
        explode_frames)
    with pytest.raises(mm.MultimodalError,
                       match="Stage frames failed"):
        mm.execute_video_pipeline_multimodal(
            storage, record,
            clients={"meta": ExplodingMeta([])})
    report = json.loads(
        (mm_dir / "cost_report.json").read_text(
            encoding="utf-8"))
    assert [c["stage"] for c in report["calls"]] == [
        "global", "frames"]
    assert report["calls"][0]["status"] == "reused"
    assert report["calls"][0]["request_id"] == "resp_old"
    assert report["calls"][0]["estimate_usd"] == 0.0


def test_executor_global_failure_starts_nothing_else(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A global rejection means zero frame and Baseten calls."""
    import bakery_intelligence.api.pipeline as pipeline_mod

    storage, record = _smoke_executor(tmp_path, monkeypatch)

    def explode_frames(**kwargs: Any) -> dict[str, bytes]:
        raise AssertionError(
            "frames must not run after global failure")

    monkeypatch.setattr(
        pipeline_mod, "run_pipeline_from_path",
        explode_frames)
    with pytest.raises(mm.MultimodalError,
                       match="Stage global failed"):
        mm.execute_video_pipeline_multimodal(
            storage, record,
            clients={"meta": FakeMeta(["not json at all"]),
                     "baseten": FakeBaseten([])})
    work = (tmp_path / "runs" / "run_0123456789ab"
            / "work" / "multimodal")
    assert (work / "global_failed.json").is_file()
    assert not (work / "clips").exists()
    report = json.loads(
        (work / "cost_report.json").read_text(
            encoding="utf-8"))
    assert [c["stage"] for c in report["calls"]] == [
        "global"]
    assert report["meta_estimate_usd_total"] == 0.0


def test_parse_json_strict_strips_illegal_control_chars() -> None:
    """Interior control bytes must not kill an otherwise valid parse.

    Regression: live speech-claims output was complete head
    and tail at 6926/8192 tokens yet failed strict parsing;
    the cause was illegal control bytes inside a string value,
    not truncation.
    """
    raw = '{"claims": [{"text": "flour\x00sugar\x07x"}]}'
    parsed = mm.parse_json_strict(raw, "Speech claims")
    assert parsed["claims"][0]["text"] == "floursugarx"


def test_parse_json_strict_keeps_tab_newline_cr() -> None:
    """Tab, LF, and CR pass through the sanitizer untouched."""
    raw = '{"a": "x\x0by"}'
    assert mm.parse_json_strict(raw, "label") == {"a": "xy"}
    with pytest.raises(mm.MultimodalError,
                       match="non-JSON output"):
        mm.parse_json_strict('{"a": "x\ty"}', "label")


def test_parse_json_tolerant_strips_illegal_control_chars() -> None:
    text = '```json\n{"a": "x\x1bz"}\n```'
    assert mm.parse_json_tolerant(text, "X") == {"a": "xz"}


def test_merge_transcripts_preserves_provider_order() -> None:
    """Turns keep chunk, then provider, order (no startMs sort).

    Re-sorting by startMs would reorder provider turns and
    break the contiguous-chunk contract: each chunk's turns
    stay in the order the provider returned them.
    """
    merged = mm.merge_transcripts(660.0, [
        (0.0, _chunk_payload("b a", [
            {"turnId": 0, "speaker": "A", "startMs": 5000,
             "endMs": 6000, "transcript": "b"},
            {"turnId": 1, "speaker": "A", "startMs": 1000,
             "endMs": 2000, "transcript": "a"}],
            "s0")),
        (330.0, _chunk_payload("c", [
            {"turnId": 0, "speaker": "B", "startMs": 100,
             "endMs": 200, "transcript": "c"}],
            "s1")),
    ])
    assert [t["transcript"] for t in merged["turns"]] == [
        "b", "a", "c"]
    assert [t["turnId"] for t in merged["turns"]] == [0, 1, 2]
    assert merged["turns"][2]["startMs"] == 330100


def test_frame_stage_calls_resumed_records_zero_new_spend() -> None:
    """A resumed run re-verifies frames without new billing."""
    firstpass = {"input_tokens": 10, "output_tokens": 5,
                 "requests_sent": 645,
                 "batches_completed": 129,
                 "cost_usd_estimated": 4.4356}
    fresh = mm.frame_stage_calls(firstpass)[0]
    assert fresh.status == "ok"
    assert fresh.estimate_usd == 4.4356
    resumed = mm.frame_stage_calls(firstpass, resumed=True)[0]
    assert resumed.status == "reused"
    assert resumed.estimate_usd == 0.0
    assert resumed.usage["requests_sent"] == 645


def test_speech_claims_failure_persists_full_output(
        tmp_path: Path) -> None:
    """A failed speech parse keeps the complete model text."""
    bad = '{"claims": [broken ' + "x" * 3000
    with pytest.raises(mm.MultimodalError,
                       match="Speech claims"):
        mm.run_speech_claims(FakeMeta([bad]), [], tmp_path)
    assert (tmp_path / "speech_claims_failed.txt").read_text(
        encoding="utf-8") == bad
    failed = json.loads(
        (tmp_path / "speech_claims_failed.json").read_text(
            encoding="utf-8"))
    assert failed["raw_text_file"] == "speech_claims_failed.txt"
    assert failed["content_length"] == len(bad)


def _segment_record(index: int, start: float, end: float,
                    clip_bytes: bytes,
                    request_id: str = "chat_old0"
                    ) -> dict[str, Any]:
    return {
        "index": index, "start_seconds": start,
        "end_seconds": end,
        "clip_sha256": hashlib.sha256(
            clip_bytes).hexdigest(),
        "clip_bytes": len(clip_bytes),
        "model": mm.BASETEN_MODEL, "provider": "baseten",
        "prompt_version": prompts.LOCAL_ANALYST.version,
        "request_id": request_id,
        "finish_reason": "stop",
        "usage": {"input_tokens": 1, "output_tokens": 1,
                  "total_tokens": 2},
        "result": json.loads(_segment_text(index)),
    }


def test_segment_stage_reuses_verified_records(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Persisted segments are re-verified, never rebilled.

    Regression: a late segment failure once forced the
    whole stage to rerun, rebilling every segment that had
    already succeeded.
    """
    cuts: list[tuple[float, float]] = []

    def fake_cut(src: Path, start: float, length: float,
                 dest: Path) -> None:
        cuts.append((start, length))
        dest.write_bytes(b"clip" + bytes([len(cuts)]))

    monkeypatch.setattr(mm, "cut_subclip", fake_cut)
    clips = tmp_path / "clips"
    clips.mkdir()
    clip0 = b"persisted-clip-0"
    (clips / "seg_000.mp4").write_bytes(clip0)
    (tmp_path / "segment_000.json").write_text(
        json.dumps(_segment_record(0, 0.0, 30.0, clip0)),
        encoding="utf-8")
    fake = FakeBaseten([_segment_text(1), _segment_text(2)])
    index, calls = mm.run_segment_stage(
        fake, Path("clip.mp4"), 65.0,  # type: ignore[arg-type]
        "global summary", tmp_path)
    assert cuts == [(30.0, 30.0), (60.0, 5.0)]
    assert [c.status for c in calls] == [
        "reused", "ok", "ok"]
    assert calls[0].request_id == "chat_old0"
    assert calls[0].estimate_usd is None
    assert index["count"] == 3
    assert index["segments"][0]["request_id"] == "chat_old0"
    # The context chain rebuilds from the reused record.
    assert "segment 0 summary" in fake.calls[0][
        "messages"][0]["content"][0]["text"]


def test_segment_stage_redoes_changed_clip(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A clip that no longer matches its record is redone."""
    monkeypatch.setattr(
        mm, "cut_subclip",
        lambda src, start, length, dest: dest.write_bytes(
            b"fresh-clip"))
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "seg_000.mp4").write_bytes(b"stale-bytes")
    (tmp_path / "segment_000.json").write_text(
        json.dumps(_segment_record(
            0, 0.0, 30.0, b"other-bytes")),
        encoding="utf-8")
    fake = FakeBaseten([_segment_text(0)])
    _, calls = mm.run_segment_stage(
        fake, Path("clip.mp4"), 10.0,  # type: ignore[arg-type]
        "", tmp_path)
    assert len(fake.calls) == 1
    assert calls[0].status == "ok"
    assert calls[0].request_id == "chat_fake1"


def test_usable_segment_record_rejects_mismatch(
        tmp_path: Path) -> None:
    """Wrong window or invalid result means redo, not reuse."""
    clips = tmp_path / "clips"
    clips.mkdir()
    clip = b"clip-bytes"
    (clips / "seg_000.mp4").write_bytes(clip)
    record = _segment_record(0, 0.0, 30.0, clip)
    (tmp_path / "segment_000.json").write_text(
        json.dumps(record), encoding="utf-8")
    assert mm._usable_segment_record(  # noqa: SLF001
        tmp_path, 0, 0.0, 30.0) is not None
    assert mm._usable_segment_record(  # noqa: SLF001
        tmp_path, 0, 0.0, 60.0) is None
    record["result"] = {"operations": []}
    (tmp_path / "segment_000.json").write_text(
        json.dumps(record), encoding="utf-8")
    assert mm._usable_segment_record(  # noqa: SLF001
        tmp_path, 0, 0.0, 30.0) is None


def test_segment_parse_failure_persists_full_text(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed segment keeps its complete model output."""
    monkeypatch.setattr(
        mm, "cut_subclip",
        lambda src, start, length, dest: dest.write_bytes(
            b"clip"))
    bad = '{"operations": [broken ' + "y" * 3000
    fake = FakeBaseten([bad])
    with pytest.raises(mm.MultimodalError,
                       match="non-JSON output"):
        mm.run_segment_stage(
            fake, Path("clip.mp4"), 10.0,  # type: ignore[arg-type]
            "", tmp_path)
    assert (tmp_path / "segment_000_failed.txt").read_text(
        encoding="utf-8") == bad
    failed = json.loads(
        (tmp_path / "segment_000_failed.json").read_text(
            encoding="utf-8"))
    assert failed["raw_text_file"] == "segment_000_failed.txt"
    assert failed["content_length"] == len(bad)


def test_write_stage_failure_rotates_prior_attempt(
        tmp_path: Path) -> None:
    """A second failed attempt keeps the first one's files."""
    mm.write_stage_failure(tmp_path, "segment_007", {
        "request_id": "chat_first", "error": "boom1"})
    (tmp_path / "segment_007_failed.txt").write_text(
        "first-output", encoding="utf-8")
    mm.write_stage_failure(tmp_path, "segment_007", {
        "request_id": "chat_second", "error": "boom2"})
    latest = json.loads(
        (tmp_path / "segment_007_failed.json").read_text(
            encoding="utf-8"))
    assert latest["request_id"] == "chat_second"
    rotated = json.loads(
        (tmp_path / "segment_007_chat_first_failed.json")
        .read_text(encoding="utf-8"))
    assert rotated["request_id"] == "chat_first"
    assert (tmp_path / "segment_007_chat_first_failed.txt"
            ).read_text(encoding="utf-8") == "first-output"
    # Same attempt re-diagnosed: overwrite, no rotation.
    mm.write_stage_failure(tmp_path, "segment_007", {
        "request_id": "chat_second", "error": "boom2b"})
    assert not (tmp_path / "segment_007_chat_second_failed.json"
                ).exists()


def test_segment_reading_value_cases() -> None:
    unit = mm._segment_reading_value  # noqa: SLF001 - unit seam
    assert unit({"value": 238, "unit": "g"}) == 238.0
    assert unit({"value": 14.5}) == 14.5
    assert unit({"value": True}) is None
    assert unit({"value": float("inf")}) is None
    assert unit({"value": None,
                 "display_reading": "112, briefly 114"}) == 112.0
    assert unit({"value": None,
                 "display_reading": "0-1g"}) == 0.0
    assert unit({"value": None,
                 "display_reading": "blurred"}) is None
    assert unit({"value": None}) is None
    assert unit({}) is None


def test_merged_evidence_lifts_segment_measurements() -> None:
    """Segment displays join FirstPass readings with provenance.

    Regression: fusion grounding only saw Path A readings,
    so a quantity recorded by the segment path (e.g. a
    stable 238 on the TRUPER display) failed validation as
    ungrounded even though it was genuinely observed.
    """
    bundle = {"readings": [{"t": 1.0, "value": 40.0,
                            "unit": None}],
              "visible_objects": [], "containers": [],
              "video": {}, "weighing": {}, "unresolved": []}
    segments = [{
        "index": 6, "model": mm.BASETEN_MODEL,
        "provider": "baseten", "request_id": "chat_seg6",
        "result": {"measurements": [
            {"value": 238, "unit": "g",
             "timestamp_seconds": 345.0},
            {"value": None,
             "display_reading": "112, briefly 114",
             "unit": "g", "timestamp": 450.0},
            {"value": None, "display_reading": "blurred"},
            {"value": None},
            "not-a-dict",
        ]},
    }]
    merged = mm.build_merged_evidence(
        bundle, {"result": {"operations": []}},
        segments, None, None)
    scales = [item for item in merged["items"]
              if item["kind"] == "scale_reading"]
    assert [(item["value"], item["source"], item["t"])
            for item in scales] == [
                (40.0, "frames", 1.0),
                (238.0, "segment_006", 345.0),
                (112.0, "segment_006", 450.0),
            ]
    assert scales[1]["unit"] == "g"
    assert scales[1]["request_id"] == "chat_seg6"
    assert scales[1]["provider"] == "baseten"
    assert scales[2]["display_reading"] == "112, briefly 114"


def test_fusion_stage_accepts_segment_grounded_quantity(
        tmp_path: Path) -> None:
    """A seen quantity matching a segment reading validates."""
    merged = _merged()
    merged["items"] = [*merged["items"], {
        "source": "segment_006", "kind": "scale_reading",
        "t": 345.0, "value": 238.0, "unit": "g",
        "model": mm.BASETEN_MODEL, "provider": "baseten",
        "request_id": "chat_seg6",
    }]
    good = _proposal(ingredients=[
        {"name": "white liquid", "quantity": 238.0,
         "unit": None, "provenance": "seen_video",
         "basis": "Stable 238 after tare at t~345s."}])
    fake = FakeBaseten([json.dumps(good)])
    record, call = mm.run_fusion_stage(
        fake, merged, _validation_record(),  # type: ignore[arg-type]
        "Roles XL", "es", tmp_path)
    assert record["result"]["ingredients"][0][
        "quantity"] == 238.0
    assert call.status == "ok"
