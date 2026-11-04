"""Multimodal video pipeline: frames, global video, segments, audio.

Stages (all mandatory unless a source is verifiably absent):
  Path A  Meta Muse Spark 1.3 frame analysis at ~1 FPS (reuses the
          proven first-pass runner and reconstruction).
  Path B  Meta Muse Spark 1.3 global native-video understanding
          (Files upload + Responses ``input_file``).
  Audio   Meta Voice Transcribe (diarization) + Spark speech claims.
  Path C  Baseten ``meta-models/muse-spark-open-1026`` native-video
          segments (30 s MP4 clips, OpenAI-compatible
          ``video_url``) with bounded accumulated context.
  Check   Evidence validation behind a configurable model adapter.
          Spark 1.3 fills this role today; Glimmer is NOT
          available (no Meta API, not on our Baseten account,
          4 GB VRAM cannot self-host 30B) and is documented as
          unavailable, never labeled as the author.
  Fusion  Deterministic merge + Baseten Spark Open structured
          recipe reconstruction in the proposal schema.

A failed mandatory stage fails the job. Nothing is silently
skipped and no historical output is ever reused.

Execution order is global first, then frames: a provider
rejection of the full-length video terminates the job after
one call instead of after hundreds of frame calls.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from bakery_intelligence.api.authorization import (
    GrantError,
    parse_grant,
    resolve_api_key,
)
from bakery_intelligence.api.jobs import (
    WORKFLOW_VIDEO_PIPELINE_MULTIMODAL,
    JobRecord,
)
from bakery_intelligence.api.storage import (
    LocalStorageBackend,
    StorageBackend,
)
from bakery_intelligence.phase2c import prompts
from bakery_intelligence.phase2c.clients import (
    MetaResponsesClient,
    ProviderError,
    VoiceTranscribeClient,
    estimate_asr_usd,
    estimate_spark_usd,
    extract_output_text,
)
from bakery_intelligence.phase2c.media import (
    cut_subclip,
    extract_wav,
    read_json,
    split_wav,
    write_json,
)

SPARK_MODEL = "muse-spark-1.3"
BASETEN_BASE_URL = "https://inference.baseten.co/v1"
BASETEN_MODEL = "meta-models/muse-spark-open-1026"
BASETEN_MODEL_VAR = "BAKERY_BASETEN_MODEL"
BASETEN_KEY_VAR = "BASETEN_API_KEY"
ASR_MODEL = "muse-voice-transcribe-1.0"

SEGMENT_SECONDS = 30.0
# Voice Transcribe rejects audio over 600 s. Chunks stay
# comfortably below the limit; 330 s keeps a 10-11 minute
# video to two provider calls.
ASR_CHUNK_SECONDS = 330.0
ASR_PROVIDER_LIMIT_SECONDS = 600.0
# Non-frame Meta calls: global (1) + speech claims (1) +
# validation (1) + one transcribe call per audio chunk.
NON_FRAME_BASE_CALLS = 3
GLOBAL_MAX_TOKENS = 16384
# Verified live: the deployment consumes output budget on
# unexposed reasoning (finish=length, 4096/4096 tokens, empty
# content). 8192 leaves headroom for reasoning + JSON.
SEGMENT_MAX_TOKENS = 8192
SPEECH_MAX_TOKENS = 8192
# Verified live: 4096 truncated the validation JSON on a
# 10 s clip (reasoning + content exceed the budget).
VALIDATION_MAX_TOKENS = 8192
FUSION_MAX_TOKENS = 16384
# Chef-requested revision: one bounded call. Verified live:
# 8192 was exhausted by unexposed reasoning (finish=length,
# 8192/8192 output tokens, zero content bytes), while the
# same deployment completed full-recipe outputs twice at
# 16384 (fusion peaked at 8921 content tokens). Revision
# emits the same full-recipe shape, so it shares the fusion
# budget instead of the smaller segment one.
REVISION_MAX_TOKENS = 16384
# Meta calls beyond frames: global + speech claims + validation
# + one ASR request per audio chunk (chunking is mandatory
# past the provider duration limit).
EXTRA_META_CALLS = 4


def non_frame_reserve(duration_seconds: float) -> int:
    """Meta calls reserved beyond frame inference.

    4 for short audio (single transcribe call), plus one per
    extra 330 s audio chunk. The frontend mirrors this
    formula when it builds the grant.
    """
    return NON_FRAME_BASE_CALLS + len(
        asr_chunk_windows(duration_seconds))

# Per-1M-token rates are Meta-published; Baseten Spark Open
# pricing is provider-reported only, so Baseten cost stays an
# explicit unknown (usage is still recorded).
BASEN_COST_UNKNOWN = "provider rates unavailable; usage recorded"


class MultimodalError(ValueError):
    """A multimodal stage failed or produced invalid output."""


@dataclass
class StageCall:
    """One billable (or attempted) provider call, fully labeled."""

    stage: str
    provider: str
    model: str
    request_id: str | None
    status: str
    elapsed_seconds: float
    usage: dict[str, Any] = field(default_factory=dict)
    estimate_usd: float | None = None
    actual_usd: float | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "provider": self.provider,
            "model": self.model,
            "request_id": self.request_id,
            "status": self.status,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "usage": self.usage,
            "estimate_usd": self.estimate_usd,
            "actual_usd": self.actual_usd,
            "error": self.error,
        }


_ILLEGAL_JSON_CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _prep_json_text(text: str) -> str:
    """Strip fences and illegal literal control characters.

    Providers occasionally emit raw control bytes inside long
    string fields; JSON forbids them unescaped, so removing
    them recovers the intended document. Tab, LF, and CR are
    legal and preserved. Anything else still fails loudly.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    return _ILLEGAL_JSON_CONTROLS.sub("", cleaned)


def parse_json_strict(text: str, what: str) -> Any:
    try:
        return json.loads(_prep_json_text(text))
    except ValueError as exc:
        raise MultimodalError(
            f"{what} returned non-JSON output: "
            f"{text[:200]!r}") from exc


def parse_json_tolerant(text: str, what: str) -> Any:
    """Baseten path: prompt-enforced JSON, fences tolerated.

    ``response_format`` is unverified on this deployment, so the
    contract is prompt text plus fence-stripping. Anything else
    fails the stage loudly.
    """
    try:
        return json.loads(_prep_json_text(text))
    except ValueError as exc:
        raise MultimodalError(
            f"{what} returned non-JSON output: "
            f"{text[:200]!r}") from exc


def _reuse_json(work_dir: Path, name: str,
                required: tuple[str, ...]
                ) -> dict[str, Any] | None:
    """Load a persisted stage output, or None to run it.

    Recovery never repeats a billable stage whose output is
    already on disk and structurally valid. Anything missing
    or malformed reruns the stage instead of failing.
    """
    path = work_dir / name
    if not path.is_file():
        return None
    try:
        stored = read_json(path)
    except ValueError:
        return None
    if not isinstance(stored, dict):
        return None
    for key in required:
        if stored.get(key) is None:
            return None
    return stored


def _reuse_segments(mm_dir: Path) -> tuple[
        dict[str, Any], list[dict[str, Any]]] | None:
    """Load a complete persisted segment set, else None.

    All-or-nothing: a partial set reruns the whole segment
    stage rather than mixing fresh and stale records.
    """
    index = _reuse_json(
        mm_dir, "segments_index.json",
        ("segments", "count"))
    if index is None or not isinstance(
            index.get("segments"), list):
        return None
    records = []
    try:
        for entry in index["segments"]:
            record = _reuse_json(
                mm_dir,
                f"segment_{entry['index']:03d}.json",
                ("result",))
            if record is None:
                return None
            records.append(record)
    except (KeyError, TypeError, ValueError):
        return None
    if len(records) != index.get("count"):
        return None
    return index, records


def write_stage_failure(work_dir: Path, name: str,
                      detail: dict[str, Any]) -> None:
    """Persist nonsecret diagnostics for a failed billable call.

    A failed parse must never force a blind repeat: the
    request id, usage, finish state, and content head/tail
    stay on disk for inspection. When a new attempt fails
    after a previous one, the older file (and its full-text
    sidecar, if any) is renamed with its request id instead
    of being overwritten, so every billable attempt stays
    inspectable.
    """
    path = work_dir / f"{name}_failed.json"
    try:
        if path.is_file():
            old = json.loads(path.read_bytes())
            old_id = old.get("request_id") \
                if isinstance(old, dict) else None
            if (isinstance(old_id, str) and old_id
                    and old_id != detail.get("request_id")):
                safe = "".join(
                    c if c.isalnum() or c in "-_" else "_"
                    for c in old_id)[:64]
                path.rename(
                    work_dir / f"{name}_{safe}_failed.json")
                txt = work_dir / f"{name}_failed.txt"
                if txt.is_file():
                    txt.rename(
                        work_dir / f"{name}_{safe}_failed.txt")
    except (OSError, ValueError):
        pass
    write_json(path, detail)


def baseten_model_id() -> str:
    """Pinned deployment; an override must match exactly."""
    import os

    model = os.environ.get(BASETEN_MODEL_VAR, BASETEN_MODEL).strip()
    if model != BASETEN_MODEL:
        raise MultimodalError(
            f"{BASETEN_MODEL_VAR} must be exactly {BASETEN_MODEL}.")
    return model


class BasetenClient:
    """OpenAI-compatible chat client for the pinned deployment."""

    def __init__(self, timeout_seconds: float = 1200.0,
                 transport: Any = None) -> None:
        import os

        from openai import OpenAI

        key = os.environ.get(BASETEN_KEY_VAR, "").strip()
        if not key:
            raise MultimodalError(
                f"Missing {BASETEN_KEY_VAR}; refusing Baseten calls.")
        # No SDK retries: every Baseten call is an explicit,
        # operator-visible single request. A transport failure
        # surfaces immediately instead of burning hidden
        # duplicate billable attempts.
        self._client = OpenAI(
            base_url=BASETEN_BASE_URL, api_key=key,
            timeout=timeout_seconds, max_retries=0,
            http_client=None if transport is None else transport)
        self._model = baseten_model_id()

    @property
    def model(self) -> str:
        return self._model

    def chat(self, messages: list[dict[str, Any]],
             max_tokens: int) -> dict[str, Any]:
        started = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=self._model, messages=messages,
                max_tokens=max_tokens)
        except Exception as exc:
            raise ProviderError(
                f"Baseten call failed: {str(exc)[:300]}") from exc
        elapsed = time.monotonic() - started
        content = ""
        finish_reason = None
        try:
            choice = response.choices[0]
            content = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason",
                                    None)
        except (AttributeError, IndexError, TypeError):
            content = ""
        usage = {}
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            usage = {
                "input_tokens": getattr(
                    raw_usage, "prompt_tokens", None),
                "output_tokens": getattr(
                    raw_usage, "completion_tokens", None),
                "total_tokens": getattr(
                    raw_usage, "total_tokens", None),
            }
        return {
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", self._model),
            "content": content,
            "finish_reason": finish_reason,
            "usage": usage,
            "elapsed_seconds": elapsed,
        }


def video_part(data: bytes) -> dict[str, Any]:
    """Native-video chat part (base64 data URL, verified format)."""
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "type": "video_url",
        "video_url": {"url": f"data:video/mp4;base64,{encoded}"},
    }


class EvidenceValidator(Protocol):
    """Model-adapter seam for evidence consistency checking.

    Glimmer was specified for this role but has no usable
    hosting (not on Meta Model API, absent from our Baseten
    account, 30B weights exceed local VRAM). ``SparkValidation
    Adapter`` fills the role today and records its true
    identity; a future Glimmer adapter implements this same
    protocol with no pipeline redesign.
    """

    model_id: str
    provider: str

    def validate(self, merged: dict[str, Any],
                 work_dir: Path | None = None,
                 ) -> dict[str, Any]:
        """Return the validation record (identity included)."""
        ...


class SparkValidationAdapter:
    """Evidence validation via Meta Muse Spark 1.3."""

    model_id = SPARK_MODEL
    provider = "meta"

    def __init__(self, client: MetaResponsesClient) -> None:
        self._client = client

    def validate(self, merged: dict[str, Any],
                 work_dir: Path | None = None,
                 ) -> dict[str, Any]:
        started = time.monotonic()
        payload = self._client.create_response(
            SPARK_MODEL,
            [{"type": "input_text",
              "text": prompts.EVIDENCE_VALIDATOR.instruction
              + "\n\nMERGED EVIDENCE (JSON):\n"
              + json.dumps(merged)[:60000]}],
            max_output_tokens=VALIDATION_MAX_TOKENS)
        elapsed = time.monotonic() - started
        text = extract_output_text(payload)

        def failed(detail: dict[str, Any]) -> None:
            if work_dir is not None:
                write_stage_failure(
                    work_dir, "validation", detail)

        try:
            parsed = parse_json_strict(
                text, "Evidence validation")
            violations = prompts.validate_output(
                prompts.EVIDENCE_VALIDATOR, parsed)
        except MultimodalError as exc:
            failed({
                "model": self.model_id,
                "provider": self.provider,
                "request_id": payload.get("id"),
                "usage": _usage_of(payload),
                "content_length": len(text),
                "content_head": text[:2000],
                "content_tail": text[-2000:],
                "error": str(exc)[:300],
            })
            raise
        if violations:
            failed({
                "model": self.model_id,
                "provider": self.provider,
                "request_id": payload.get("id"),
                "usage": _usage_of(payload),
                "content_length": len(text),
                "violations": violations,
                "error": "Invalid validation payload",
            })
            raise MultimodalError(
                "Invalid validation payload: "
                + "; ".join(violations))
        usage = payload.get("usage") or {}
        return {
            "model": self.model_id,
            "provider": self.provider,
            "prompt_version": prompts.EVIDENCE_VALIDATOR.version,
            "request_id": payload.get("id"),
            "elapsed_seconds": round(elapsed, 3),
            "usage": usage,
            "result": parsed,
        }


VALIDATOR_ADAPTERS: dict[str, type[SparkValidationAdapter]] = {
    SPARK_MODEL: SparkValidationAdapter,
}
DEFAULT_VALIDATOR = SPARK_MODEL


def get_validator(name: str,
                  client: MetaResponsesClient) -> EvidenceValidator:
    try:
        adapter = VALIDATOR_ADAPTERS[name]
    except KeyError as exc:
        raise MultimodalError(
            f"Unknown validator adapter: {name!r}. Available: "
            f"{sorted(VALIDATOR_ADAPTERS)}.") from exc
    return adapter(client)


def _usage_of(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage")
    return dict(usage) if isinstance(usage, dict) else {}


def _spark_tokens(usage: dict[str, Any]) -> tuple[int, int]:
    def num(value: Any) -> int:
        return value if isinstance(value, int) and value >= 0 else 0

    return (num(usage.get("input_tokens")),
            num(usage.get("output_tokens")))


def frame_stage_calls(
        firstpass: dict[str, Any],
        resumed: bool = False) -> list[StageCall]:
    """Aggregate Path A usage into one cost-report call.

    The first-pass report already carries token totals and a
    rate-based estimate; surfacing them here keeps the
    multimodal cost report honest instead of hiding the
    dominant frame spend behind a pointer. A resumed run
    records 0.0 new spend: the pipeline core verified prior
    outputs without billing.
    """
    def num(value: Any) -> int:
        return value if isinstance(value, int) and value >= 0 else 0

    estimate = firstpass.get("cost_usd_estimated")
    status = "ok"
    new_estimate: float | None = None
    error: str | None = None
    if resumed:
        status, new_estimate, error = (
            "reused", 0.0, None)
    elif (isinstance(estimate, (int, float))
          and not isinstance(estimate, bool)):
        status, new_estimate, error = (
            "ok", float(estimate), None)
    else:
        status, new_estimate, error = (
            "ok", None, "usage in firstpass_report.json")
    return [StageCall(
        stage="frames_firstpass", provider="meta",
        model=SPARK_MODEL, request_id=None, status=status,
        elapsed_seconds=0.0,
        usage={
            "input_tokens": num(
                firstpass.get("input_tokens")),
            "output_tokens": num(
                firstpass.get("output_tokens")),
            "requests_sent": num(
                firstpass.get("requests_sent")),
            "batches_completed": num(
                firstpass.get("batches_completed")),
        },
        estimate_usd=new_estimate,
        error=error)]


def run_global_stage(meta: MetaResponsesClient,
                     video_path: Path,
                     duration_seconds: float,
                     work_dir: Path) -> tuple[dict[str, Any],
                                              StageCall]:
    """Path B: native-video global understanding (Spark 1.3)."""
    file_id = meta.upload_file(video_path, purpose="user_data")
    prompt = prompts.GLOBAL_ANALYST
    payload = meta.create_response(
        SPARK_MODEL,
        [{"type": "input_text",
          "text": prompt.instruction
          + f"\n\nVideo duration: {duration_seconds:.1f}s. "
            "The timeline may contain an edit between two "
            "recordings; do not assume continuous production. "
            "Return structured JSON only."},
         {"type": "input_file", "file_id": file_id}],
        max_output_tokens=GLOBAL_MAX_TOKENS)
    text = extract_output_text(payload)
    try:
        parsed = parse_json_strict(text, "Global video analysis")
        violations = prompts.validate_output(prompt, parsed)
    except MultimodalError as exc:
        write_stage_failure(work_dir, "global", {
            "model": SPARK_MODEL, "provider": "meta",
            "request_id": payload.get("id"),
            "usage": _usage_of(payload),
            "content_length": len(text),
            "content_head": text[:2000],
            "content_tail": text[-2000:],
            "error": str(exc)[:300],
        })
        raise
    if violations:
        write_stage_failure(work_dir, "global", {
            "model": SPARK_MODEL, "provider": "meta",
            "request_id": payload.get("id"),
            "usage": _usage_of(payload),
            "content_length": len(text),
            "violations": violations,
            "error": "Invalid global payload",
        })
        raise MultimodalError(
            "Invalid global payload: " + "; ".join(violations))
    usage = _usage_of(payload)
    in_tok, out_tok = _spark_tokens(usage)
    record = {
        "model": SPARK_MODEL,
        "provider": "meta",
        "prompt_version": prompt.version,
        "request_id": payload.get("id"),
        "file_id": file_id,
        "duration_seconds": duration_seconds,
        "usage": usage,
        "result": parsed,
    }
    write_json(work_dir / "global.json", record)
    call = StageCall(
        stage="global", provider="meta", model=SPARK_MODEL,
        request_id=payload.get("id"), status="ok",
        elapsed_seconds=float(
            payload.get("_elapsed_seconds") or 0.0),
        usage=usage,
        estimate_usd=estimate_spark_usd(in_tok, out_tok))
    return record, call


def has_audio_stream(video_path: Path) -> bool:
    """True when ffprobe reports an audio stream (no decoding)."""
    import subprocess

    try:
        completed = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index",
             "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, timeout=120)
    except (OSError, ValueError) as exc:
        raise MultimodalError(
            f"ffprobe unavailable for audio check: {exc}")
    if completed.returncode != 0:
        raise MultimodalError(
            "ffprobe could not read streams: "
            f"{completed.stderr[:200]}")
    return bool(completed.stdout.strip())


def asr_chunk_windows(duration_seconds: float,
                      window: float = ASR_CHUNK_SECONDS
                      ) -> list[tuple[float, float]]:
    """Contiguous [start, end) windows covering the audio.

    Contiguous (not overlapping) by design: no transcript
    segment can be duplicated, and no audio is skipped. A
    word straddling the exact boundary may be split across
    two chunk transcripts; chunks are transcribed
    independently and never reworded.
    """
    if duration_seconds <= 0:
        raise MultimodalError("Cannot chunk zero-length audio.")
    windows = []
    start = 0.0
    while start < duration_seconds:
        windows.append((start, min(start + window,
                                   duration_seconds)))
        start += window
    return windows


def merge_transcripts(
        duration_seconds: float,
        chunk_payloads: list[tuple[float, dict[str, Any]]]
        ) -> dict[str, Any]:
    """Merge chunk transcripts onto the absolute timeline.

    Turn timestamps shift by their chunk offset; turn ids are
    reassigned sequentially. The single-call schema
    (transcript text, turns, audioDurationMs, sessionId) is
    preserved, with chunk provenance added.
    """
    turns: list[dict[str, Any]] = []
    texts: list[str] = []
    sessions: list[Any] = []
    elapsed = 0.0
    turn_id = 0
    for start, payload in chunk_payloads:
        offset_ms = int(round(start * 1000))
        chunk_turns = payload.get("turns")
        if not isinstance(chunk_turns, list):
            raise MultimodalError(
                f"ASR chunk at {start:.1f}s has no turns list.")
        for turn in chunk_turns:
            if not isinstance(turn, dict):
                raise MultimodalError(
                    f"ASR chunk at {start:.1f}s has a "
                    "non-object turn.")
            merged_turn = dict(turn)
            for key in ("startMs", "endMs"):
                value = turn.get(key)
                if isinstance(value, bool) or not isinstance(
                        value, (int, float)):
                    raise MultimodalError(
                        f"ASR chunk at {start:.1f}s has a "
                        f"turn without numeric {key}.")
                merged_turn[key] = int(value) + offset_ms
            merged_turn["turnId"] = turn_id
            turn_id += 1
            turns.append(merged_turn)
        text = payload.get("transcript")
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
        sessions.append(payload.get("sessionId"))
        extra = payload.get("_elapsed_seconds")
        if isinstance(extra, (int, float)) and not isinstance(
                extra, bool):
            elapsed += float(extra)
    return {
        "audioDurationMs": int(round(duration_seconds * 1000)),
        "sessionId": sessions[0] if sessions else None,
        "transcript": " ".join(texts),
        "turns": turns,
        "chunked": len(chunk_payloads) > 1,
        "chunk_seconds": ASR_CHUNK_SECONDS,
        "chunk_session_ids": sessions,
        "_elapsed_seconds": round(elapsed, 3),
    }


def _valid_chunk_record(stored: Any) -> dict[str, Any] | None:
    """Return a persisted chunk payload, or None to redo it."""
    if not isinstance(stored, dict):
        return None
    payload = stored.get("payload")
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("turns"), list):
        return None
    if not isinstance(stored.get("start_seconds"),
                      (int, float)):
        return None
    return payload


def _reused_audio_calls(
        transcript: dict[str, Any],
        claims: dict[str, Any]) -> list[StageCall]:
    """Rebuilt cost calls for reused audio outputs.

    Reused stages record 0.0 new spend with their original
    usage and request ids preserved, so a resumed run never
    bills or double-counts them.
    """
    calls: list[StageCall] = []
    chunks = transcript.get("chunks")
    if isinstance(chunks, list) and chunks:
        for position, entry in enumerate(chunks):
            if not isinstance(entry, dict):
                continue
            index = entry.get("index")
            if not isinstance(index, int):
                index = position
            start = entry.get("start_seconds")
            end = entry.get("end_seconds")
            if not isinstance(start, (int, float)):
                start = 0.0
            if not isinstance(end, (int, float)):
                end = start
            calls.append(StageCall(
                stage="audio_transcribe_chunk_"
                      f"{index:03d}",
                provider="meta", model=ASR_MODEL,
                request_id=entry.get("request_id"),
                status="reused", elapsed_seconds=0.0,
                usage={"audio_seconds": round(
                    max(0.0, float(end) - float(start)), 3)},
                estimate_usd=0.0))
    else:
        calls.append(StageCall(
            stage="audio_transcribe", provider="meta",
            model=ASR_MODEL,
            request_id=transcript.get("request_id"),
            status="reused", elapsed_seconds=0.0,
            usage={}, estimate_usd=0.0))
    usage = _usage_of(claims)
    calls.append(StageCall(
        stage="speech_claims", provider="meta",
        model=SPARK_MODEL,
        request_id=claims.get("request_id"),
        status="reused", elapsed_seconds=0.0,
        usage=usage, estimate_usd=0.0))
    return calls


def run_audio_stage(meta: MetaResponsesClient,
                    asr: VoiceTranscribeClient,
                    video_path: Path,
                    work_dir: Path) -> tuple[
                        dict[str, Any] | None,
                        dict[str, Any] | None,
                        list[StageCall]]:
    """Transcribe audio, then extract structured speech claims."""
    from bakery_intelligence.video.intake import probe_video

    probe = probe_video(video_path)
    if not has_audio_stream(video_path):
        skipped = {"status": "skipped",
                   "reason": "no audio stream in source"}
        write_json(work_dir / "transcript.json", skipped)
        return None, None, []
    stored_transcript = _reuse_json(
        work_dir, "transcript.json", ("turns",))
    stored_claims = _reuse_json(
        work_dir, "speech_claims.json", ("result",))
    if (stored_transcript is not None
            and stored_claims is not None):
        return (stored_transcript, stored_claims,
                _reused_audio_calls(
                    stored_transcript, stored_claims))
    wav_path = work_dir / "audio.wav"
    if not (wav_path.is_file()
            and wav_path.stat().st_size > 0):
        extract_wav(video_path, wav_path)
    duration = float(probe.get("duration_seconds") or 0.0)
    if duration <= 0:
        raise MultimodalError(
            "Audio stage found no measurable duration.")
    chunks_dir = work_dir / "asr_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    windows = asr_chunk_windows(duration)
    chunk_payloads: list[tuple[float, dict[str, Any]]] = []
    calls: list[StageCall] = []
    for index, (start, end) in enumerate(windows):
        chunk_path = chunks_dir / f"chunk_{index:03d}.wav"
        record_path = chunks_dir / f"chunk_{index:03d}.json"
        if record_path.is_file():
            try:
                reused = _valid_chunk_record(
                    read_json(record_path))
            except ValueError:
                reused = None
            if reused is not None:
                chunk_payloads.append((start, reused))
                calls.append(StageCall(
                    stage=f"audio_transcribe_chunk_{index:03d}",
                    provider="meta", model=ASR_MODEL,
                    request_id=reused.get("request_id"),
                    status="reused", elapsed_seconds=0.0,
                    usage={"audio_seconds": round(
                        end - start, 3)},
                    estimate_usd=0.0))
                continue
        split_wav(wav_path, start, end - start, chunk_path)
        started = time.monotonic()
        try:
            payload = asr.transcribe(
                chunk_path, mode="DIARIZATION")
        except Exception as exc:
            write_stage_failure(
                work_dir, f"asr_chunk_{index:03d}", {
                    "model": ASR_MODEL, "provider": "meta",
                    "start_seconds": start,
                    "end_seconds": end,
                    "error": f"{type(exc).__name__}: "
                             f"{str(exc)[:300]}",
                })
            raise MultimodalError(
                f"ASR chunk {index} ({start:.1f}s) failed: "
                f"{exc}") from exc
        elapsed = time.monotonic() - started
        if not isinstance(payload, dict):
            raise MultimodalError(
                f"ASR chunk {index} returned a non-object "
                "payload.")
        payload["_elapsed_seconds"] = round(elapsed, 3)
        write_json(record_path, {
            "index": index,
            "start_seconds": start,
            "end_seconds": end,
            "payload": payload,
        })
        chunk_payloads.append((start, payload))
        calls.append(StageCall(
            stage=f"audio_transcribe_chunk_{index:03d}",
            provider="meta", model=ASR_MODEL,
            request_id=payload.get("request_id"),
            status="ok", elapsed_seconds=elapsed,
            usage={"audio_seconds": round(end - start, 3)},
            estimate_usd=estimate_asr_usd(end - start)))
    transcript = merge_transcripts(duration, chunk_payloads)
    transcript["chunks"] = [{
        "index": index,
        "start_seconds": start,
        "end_seconds": end,
        "request_id": payload.get("request_id"),
    } for index, ((start, end), (_, payload)) in enumerate(
        zip(windows, chunk_payloads))]
    write_json(work_dir / "transcript.json", transcript)
    turns = transcript.get("turns")
    if not isinstance(turns, list) or not turns:
        write_json(work_dir / "speech_claims.json",
                   {"status": "skipped",
                    "reason": "transcript has no turns"})
        return transcript, None, calls
    claims, call = run_speech_claims(meta, turns, work_dir)
    calls.append(call)
    return transcript, claims, calls


def run_speech_claims(meta: MetaResponsesClient,
                      turns: list[Any],
                      work_dir: Path) -> tuple[dict[str, Any],
                                             StageCall]:
    """Extract structured culinary claims from transcript turns."""
    prompt = prompts.SPEECH_ANALYST
    payload = meta.create_response(
        SPARK_MODEL,
        [{"type": "input_text",
          "text": prompt.instruction
          + "\n\nTRANSCRIPT TURNS (JSON, capped):\n"
          + json.dumps(turns[:60])[:40000]
          + "\nReturn structured JSON only."}],
        max_output_tokens=SPEECH_MAX_TOKENS)
    text = extract_output_text(payload)
    try:
        parsed = parse_json_strict(text, "Speech claims")
        violations = prompts.validate_output(prompt, parsed)
    except MultimodalError as exc:
        # Failure record first: it rotates any previous
        # attempt's files aside before the new text lands.
        write_stage_failure(work_dir, "speech_claims", {
            "model": SPARK_MODEL, "provider": "meta",
            "request_id": payload.get("id"),
            "usage": _usage_of(payload),
            "content_length": len(text),
            "content_head": text[:2000],
            "content_tail": text[-2000:],
            "raw_text_file": "speech_claims_failed.txt",
            "error": str(exc)[:300],
        })
        (work_dir / "speech_claims_failed.txt").write_text(
            text, encoding="utf-8", errors="backslashreplace")
        raise
    if violations:
        write_stage_failure(work_dir, "speech_claims", {
            "model": SPARK_MODEL, "provider": "meta",
            "request_id": payload.get("id"),
            "usage": _usage_of(payload),
            "content_length": len(text),
            "content_head": text[:2000],
            "content_tail": text[-2000:],
            "raw_text_file": "speech_claims_failed.txt",
            "violations": violations,
            "error": "Invalid speech payload",
        })
        (work_dir / "speech_claims_failed.txt").write_text(
            text, encoding="utf-8", errors="backslashreplace")
        raise MultimodalError(
            "Invalid speech payload: " + "; ".join(violations))
    usage = _usage_of(payload)
    in_tok, out_tok = _spark_tokens(usage)
    claims = {
        "model": SPARK_MODEL,
        "provider": "meta",
        "prompt_version": prompt.version,
        "request_id": payload.get("id"),
        "usage": usage,
        "result": parsed,
    }
    write_json(work_dir / "speech_claims.json", claims)
    call = StageCall(
        stage="speech_claims", provider="meta",
        model=SPARK_MODEL, request_id=payload.get("id"),
        status="ok",
        elapsed_seconds=float(
            payload.get("_elapsed_seconds") or 0.0),
        usage=usage,
        estimate_usd=estimate_spark_usd(in_tok, out_tok))
    return claims, call


def segment_windows(duration_seconds: float,
                    window: float = SEGMENT_SECONDS
                    ) -> list[tuple[float, float]]:
    """Absolute [start, end) windows covering the timeline."""
    if duration_seconds <= 0:
        raise MultimodalError("Cannot segment a zero-length video.")
    windows = []
    start = 0.0
    while start < duration_seconds:
        windows.append((start, min(start + window,
                                   duration_seconds)))
        start += window
    return windows


def _usable_segment_record(mm_dir: Path, index: int,
                           start: float, end: float,
                           ) -> dict[str, Any] | None:
    """Return a persisted segment record, or None to redo it.

    A record is reusable only when its window matches, its
    clip file is still on disk with the recorded hash, and
    its result still validates: a resumed run must never mix
    a fresh call chain with a stale or foreign record.
    """
    try:
        record = read_json(
            mm_dir / f"segment_{index:03d}.json")
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    if record.get("index") != index:
        return None
    for key, want in (("start_seconds", start),
                      ("end_seconds", end)):
        got = record.get(key)
        if not isinstance(got, (int, float)) or isinstance(
                got, bool) or abs(float(got) - want) > 0.01:
            return None
    clip_path = mm_dir / "clips" / f"seg_{index:03d}.mp4"
    want_sha = record.get("clip_sha256")
    if not clip_path.is_file() or not isinstance(
            want_sha, str):
        return None
    try:
        clip_bytes = clip_path.read_bytes()
    except OSError:
        return None
    if hashlib.sha256(clip_bytes).hexdigest() != want_sha:
        return None
    result = record.get("result")
    if not isinstance(result, dict):
        return None
    if prompts.validate_output(prompts.LOCAL_ANALYST,
                               result):
        return None
    return record


def run_segment_stage(baseten: BasetenClient,
                      video_path: Path,
                      duration_seconds: float,
                      global_summary: str,
                      work_dir: Path) -> tuple[dict[str, Any],
                                               list[StageCall]]:
    """Path C: Baseten Spark Open native-video segments.

    Segments resume individually: each persisted record is
    re-verified (window, clip hash, schema) before its call
    is skipped, so a late failure never rebills the segments
    that already succeeded.
    """
    prompt = prompts.LOCAL_ANALYST
    clips_dir = work_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    segments: list[dict[str, Any]] = []
    calls: list[StageCall] = []
    previous_summary = ""
    for index, (start, end) in enumerate(
            segment_windows(duration_seconds)):
        clip_path = clips_dir / f"seg_{index:03d}.mp4"
        reused = _usable_segment_record(
            work_dir, index, start, end)
        if reused is not None:
            segments.append(reused)
            calls.append(StageCall(
                stage=f"segment_{index:03d}",
                provider="baseten",
                model=reused.get("model")
                or baseten.model,
                request_id=reused.get("request_id"),
                status="reused", elapsed_seconds=0.0,
                usage=reused.get("usage") or {},
                estimate_usd=None,
                error=BASEN_COST_UNKNOWN))
            parsed = reused["result"]
            summary = parsed.get("summary") or parsed.get(
                "segment_summary") or ""
            previous_summary = str(summary)[:1500]
            continue
        cut_subclip(video_path, start, end - start, clip_path)
        clip_bytes = clip_path.read_bytes()
        context = global_summary[:1500]
        if previous_summary:
            context += ("\nPrevious segment summary: "
                        + previous_summary[:1500])
        instruction = (
            prompt.instruction
            + f"\n\nSegment absolute bounds: start {start:.1f}s, "
            f"end {end:.1f}s. Timestamps in the output must be "
            "absolute seconds on this timeline. "
            + (f"Context (untrusted): {context}\n" if context
               else "")
            + "Return structured JSON only.")
        result = baseten.chat(
            [{"role": "user",
              "content": [{"type": "text", "text": instruction},
                          video_part(clip_bytes)]}],
            max_tokens=SEGMENT_MAX_TOKENS)
        content = result["content"]
        if not isinstance(content, str):
            content = json.dumps(content, default=str)
        result["content"] = content
        try:
            parsed = parse_json_tolerant(
                content, f"Segment {index}")
            violations = prompts.validate_output(prompt, parsed)
        except MultimodalError as exc:
            write_stage_failure(
                work_dir, f"segment_{index:03d}", {
                    "model": result["model"],
                    "provider": "baseten",
                    "request_id": result["id"],
                    "finish_reason": result.get(
                        "finish_reason"),
                    "usage": result["usage"],
                    "content_length": len(
                        result["content"]),
                    "content_head": result["content"][:2000],
                    "content_tail": result["content"][-2000:],
                    "raw_text_file":
                        f"segment_{index:03d}_failed.txt",
                    "error": str(exc)[:300],
                })
            (work_dir
             / f"segment_{index:03d}_failed.txt").write_text(
                result["content"], encoding="utf-8",
                errors="backslashreplace")
            raise
        if violations:
            write_stage_failure(
                work_dir, f"segment_{index:03d}", {
                    "model": result["model"],
                    "provider": "baseten",
                    "request_id": result["id"],
                    "finish_reason": result.get(
                        "finish_reason"),
                    "usage": result["usage"],
                    "content_length": len(
                        result["content"]),
                    "content_head": result["content"][:2000],
                    "content_tail": result["content"][-2000:],
                    "raw_text_file":
                        f"segment_{index:03d}_failed.txt",
                    "violations": violations,
                    "error": ("Invalid segment "
                              f"{index} payload"),
                })
            (work_dir
             / f"segment_{index:03d}_failed.txt").write_text(
                result["content"], encoding="utf-8",
                errors="backslashreplace")
            raise MultimodalError(
                f"Invalid segment {index} payload: "
                + "; ".join(violations))
        record = {
            "index": index,
            "start_seconds": start,
            "end_seconds": end,
            "clip_sha256": hashlib.sha256(
                clip_bytes).hexdigest(),
            "clip_bytes": len(clip_bytes),
            "model": result["model"],
            "provider": "baseten",
            "prompt_version": prompt.version,
            "request_id": result["id"],
            "finish_reason": result.get("finish_reason"),
            "usage": result["usage"],
            "result": parsed,
        }
        write_json(work_dir / f"segment_{index:03d}.json",
                   record)
        segments.append(record)
        calls.append(StageCall(
            stage=f"segment_{index:03d}", provider="baseten",
            model=result["model"], request_id=result["id"],
            status="ok",
            elapsed_seconds=result["elapsed_seconds"],
            usage=result["usage"], estimate_usd=None,
            error=BASEN_COST_UNKNOWN))
        summary = parsed.get("summary") or parsed.get(
            "segment_summary") or ""
        previous_summary = str(summary)[:1500]
    index_record = {
        "model": baseten.model,
        "provider": "baseten",
        "segment_seconds": SEGMENT_SECONDS,
        "count": len(segments),
        "segments": [{
            "index": s["index"],
            "start_seconds": s["start_seconds"],
            "end_seconds": s["end_seconds"],
            "clip_sha256": s["clip_sha256"],
            "request_id": s["request_id"],
        } for s in segments],
    }
    write_json(work_dir / "segments_index.json", index_record)
    return index_record, calls


def _segment_reading_value(
        reading: dict[str, Any]) -> float | None:
    """Numeric display value of a segment measurement.

    Prefers the structured ``value``; falls back to the
    leading number of ``display_reading`` (e.g. "112,
    briefly 114" -> 112.0). Unreadable displays yield
    None and stay out of quantity grounding.
    """
    value = reading.get("value")
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(
            value):
        return float(value)
    text = reading.get("display_reading")
    if isinstance(text, str):
        match = re.match(r"\s*(-?\d+(?:\.\d+)?)", text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def build_merged_evidence(
        firstpass_bundle: dict[str, Any],
        global_record: dict[str, Any],
        segment_records: list[dict[str, Any]],
        transcript: dict[str, Any] | None,
        claims: dict[str, Any] | None) -> dict[str, Any]:
    """Deterministic cross-source merge with provenance.

    Segment scale measurements join the FirstPass readings
    as ``scale_reading`` items (with their own source and
    request id), so fusion grounding and quantity checks
    see every recorded display value, not just Path A.
    """
    items: list[dict[str, Any]] = []
    for reading in firstpass_bundle.get("readings") or []:
        items.append({
            "source": "frames", "kind": "scale_reading",
            "t": reading.get("t"), "value": reading.get("value"),
            "unit": reading.get("unit"),
            "model": SPARK_MODEL, "provider": "meta",
        })
    for entry in firstpass_bundle.get("visible_objects") or []:
        items.append({
            "source": "frames", "kind": "visible_object",
            "t": entry.get("first_t"),
            "text": entry.get("object"),
            "count": entry.get("count"),
            "model": SPARK_MODEL, "provider": "meta",
        })
    for container in firstpass_bundle.get("containers") or []:
        items.append({
            "source": "frames", "kind": "container",
            "t": container.get("t"),
            "text": container.get("description"),
            "model": SPARK_MODEL, "provider": "meta",
        })
    global_result = global_record.get("result") or {}
    for op in global_result.get("operations") or []:
        items.append({
            "source": "global_video", "kind": "operation",
            "t": (op.get("start_seconds")
                  if isinstance(op, dict) else None),
            "text": (op.get("description") or op.get("name")
                     if isinstance(op, dict) else str(op)),
            "model": global_record.get("model"),
            "provider": global_record.get("provider"),
            "request_id": global_record.get("request_id"),
        })
    for segment in segment_records:
        result = segment.get("result") or {}
        for event in result.get("events") or []:
            items.append({
                "source": f"segment_{segment['index']:03d}",
                "kind": "culinary_event",
                "t": (event.get("start_seconds")
                      if isinstance(event, dict) else None),
                "text": json.dumps(event)[:500]
                if isinstance(event, dict) else str(event),
                "model": segment.get("model"),
                "provider": segment.get("provider"),
                "request_id": segment.get("request_id"),
            })
        for reading in result.get("measurements") or []:
            if not isinstance(reading, dict):
                continue
            value = _segment_reading_value(reading)
            if value is None:
                continue
            stamp = reading.get("timestamp_seconds")
            if stamp is None:
                stamp = reading.get("timestamp")
            items.append({
                "source": f"segment_{segment['index']:03d}",
                "kind": "scale_reading",
                "t": stamp,
                "value": value,
                "unit": reading.get("unit"),
                "display_reading": reading.get(
                    "display_reading"),
                "model": segment.get("model"),
                "provider": segment.get("provider"),
                "request_id": segment.get("request_id"),
            })
    if claims is not None:
        for claim in (claims.get("result") or {}).get(
                "claims") or []:
            items.append({
                "source": "audio", "kind": "speech_claim",
                "t": (claim.get("start_seconds")
                      if isinstance(claim, dict) else None),
                "text": json.dumps(claim)[:500]
                if isinstance(claim, dict) else str(claim),
                "model": claims.get("model"),
                "provider": claims.get("provider"),
                "request_id": claims.get("request_id"),
            })
    return {
        "video": firstpass_bundle.get("video") or {},
        "readings_total": len(
            firstpass_bundle.get("readings") or []),
        "weighing": firstpass_bundle.get("weighing") or {},
        "unresolved": firstpass_bundle.get("unresolved") or [],
        "global_result": global_result,
        "segment_count": len(segment_records),
        "transcript_turns": len(
            (transcript or {}).get("turns") or [])
        if transcript else 0,
        "items": items,
    }


def run_fusion_stage(baseten: BasetenClient,
                     merged: dict[str, Any],
                     validation: dict[str, Any],
                     recipe_name: str,
                     language: str,
                     work_dir: Path) -> tuple[dict[str, Any],
                                              StageCall]:
    """Baseten Spark Open structured recipe reconstruction."""
    from bakery_intelligence.recipe_export import proposal as \
        proposal_mod

    prompt = prompts.RECIPE_PROPOSAL
    fusion_input = {
        "recipe_name": recipe_name,
        "language": language,
        "merged_evidence": merged,
        "validation": validation.get("result"),
        "validator": {
            "model": validation.get("model"),
            "provider": validation.get("provider"),
            "prompt_version": validation.get(
                "prompt_version"),
        },
    }
    write_json(work_dir / "fusion_input.json", fusion_input)
    instruction = (
        prompt.instruction
        + "\n\nEVIDENCE (JSON, capped):\n"
        + json.dumps(fusion_input)[:60000]
        + "\n\nAll top-level keys are REQUIRED, with "
        "non-empty notes (1-15 production-note strings) "
        "even when other fields are uncertain.")
    result = baseten.chat(
        [{"role": "user",
          "content": [{"type": "text", "text": instruction}]}],
        max_tokens=FUSION_MAX_TOKENS)
    if not isinstance(result["content"], str):
        result["content"] = json.dumps(
            result["content"], default=str)
    try:
        parsed = parse_json_tolerant(
            result["content"], "Fusion")
    except MultimodalError as exc:
        write_stage_failure(work_dir, "fusion_recipe", {
            "model": result["model"], "provider": "baseten",
            "request_id": result["id"],
            "finish_reason": result.get("finish_reason"),
            "usage": result["usage"],
            "content_length": len(result["content"]),
            "content_head": result["content"][:2000],
            "content_tail": result["content"][-2000:],
            "raw_text_file": "fusion_recipe_failed.txt",
            "error": str(exc)[:300],
        })
        (work_dir / "fusion_recipe_failed.txt").write_text(
            result["content"], encoding="utf-8",
            errors="backslashreplace")
        raise
    # Grounding covers every recorded display value: merged
    # scale_reading items come from both the FirstPass and
    # the segment paths.
    checks_bundle = {
        "readings": [{
            "value": item.get("value")}
            for item in merged.get("items") or []
            if item.get("kind") == "scale_reading"
            and isinstance(item.get("value"), (int, float))],
        "language": language,
        "recipe_name": recipe_name,
    }
    try:
        validated = proposal_mod.validate_proposal(
            checks_bundle, parsed)
    except proposal_mod.ProposalError as exc:
        write_stage_failure(work_dir, "fusion_recipe", {
            "model": result["model"], "provider": "baseten",
            "request_id": result["id"],
            "finish_reason": result.get("finish_reason"),
            "usage": result["usage"],
            "error": str(exc)[:300],
            "draft": parsed,
        })
        raise MultimodalError(
            f"Fusion draft failed proposal validation: "
            f"{exc}") from exc
    record = {
        "model": result["model"],
        "provider": "baseten",
        "prompt_version": prompt.version,
        "request_id": result["id"],
        "finish_reason": result.get("finish_reason"),
        "usage": result["usage"],
        "result": validated,
    }
    write_json(work_dir / "fusion_recipe.json", record)
    call = StageCall(
        stage="fusion", provider="baseten",
        model=result["model"], request_id=result["id"],
        status="ok", elapsed_seconds=result["elapsed_seconds"],
        usage=result["usage"], estimate_usd=None,
        error=BASEN_COST_UNKNOWN)
    return record, call


def run_revision_stage(baseten: BasetenClient,
                       bundle: dict[str, Any],
                       work_dir: Path) -> tuple[dict[str, Any],
                                              StageCall]:
    """One bounded Baseten revision of a saved recipe.

    No pipeline stage reruns here: the caller supplies the
    saved recipe, chef feedback, and capped evidence. The
    record keeps the ``revision_raw.json`` shape (request id,
    usage, parsed) plus provider identity.
    """
    prompt = prompts.RECIPE_REVISION
    instruction = (
        prompt.instruction
        + "\n\nREVISION REQUEST (JSON, capped):\n"
        + json.dumps(bundle)[:40000]
        + "\n\nAll top-level keys are REQUIRED: title, "
        "ingredients, method, notes, changes_summary.")
    result = baseten.chat(
        [{"role": "user",
          "content": [{"type": "text", "text": instruction}]}],
        max_tokens=REVISION_MAX_TOKENS)
    content = result["content"]
    if not isinstance(content, str):
        content = json.dumps(content, default=str)
    result["content"] = content
    finish = result.get("finish_reason")
    usage = result.get("usage") or {}
    out_tokens = usage.get("output_tokens")

    def failed(detail: dict[str, Any]) -> None:
        write_stage_failure(work_dir, "revision_raw", {
            "model": result["model"], "provider": "baseten",
            "request_id": result["id"],
            "finish_reason": finish,
            "usage": usage,
            "content_length": len(content),
            "content_head": content[:2000],
            "content_tail": content[-2000:],
            "raw_text_file": "revision_raw_failed.txt",
            **detail,
        })
        (work_dir / "revision_raw_failed.txt").write_text(
            content, encoding="utf-8",
            errors="backslashreplace")

    if not content.strip():
        if finish == "length":
            failed({
                "failure_category": "truncated",
                "error": "Revision output exhausted the "
                         "token budget.",
            })
            raise MultimodalError(
                f"Revision output exhausted the token "
                f"budget (finish=length, {out_tokens} "
                f"output tokens, request "
                f"{result['id']}); no content to parse.")
        failed({
            "failure_category": "empty",
            "error": "Revision returned empty output.",
        })
        raise MultimodalError(
            f"Revision returned empty output "
            f"(finish={finish}, request {result['id']}).")
    try:
        parsed = parse_json_tolerant(content, "Revision")
        violations = prompts.validate_output(prompt, parsed)
    except MultimodalError as exc:
        failed({
            "failure_category": ("truncated"
                                 if finish == "length"
                                 else "invalid"),
            "error": str(exc)[:300],
        })
        raise
    if violations:
        failed({
            "failure_category": ("truncated"
                                 if finish == "length"
                                 else "invalid"),
            "violations": violations,
            "error": "Invalid revision payload",
        })
        raise MultimodalError(
            "Invalid revision payload: "
            + "; ".join(violations))
    record = {
        "model": result["model"],
        "provider": "baseten",
        "prompt_version": prompt.version,
        "request_id": result["id"],
        "finish_reason": result.get("finish_reason"),
        "usage": result["usage"],
        "parsed": parsed,
    }
    write_json(work_dir / "revision_raw.json", record)
    call = StageCall(
        stage="revision", provider="baseten",
        model=result["model"], request_id=result["id"],
        status="ok",
        elapsed_seconds=result["elapsed_seconds"],
        usage=result["usage"], estimate_usd=None,
        error=BASEN_COST_UNKNOWN)
    return record, call


def _local_source_path(storage: StorageBackend,
                       run_id: str,
                       object_id: str) -> Path:
    if isinstance(storage, LocalStorageBackend):
        candidates = list(
            storage.base_dir.glob(f"sources/{object_id}"))
        if candidates:
            return candidates[0]
    raise MultimodalError(
        "Multimodal stages need a local source file; "
        f"{object_id!r} is not on local disk.")


def _linked_recipe_meta(storage: StorageBackend,
                        run_id: str) -> tuple[str, str]:
    """Recipe name + language from the linked record, if any."""
    if not isinstance(storage, LocalStorageBackend):
        return "Untitled recipe", "en"
    recipes_dir = storage.base_dir / "recipes"
    if not recipes_dir.is_dir():
        return "Untitled recipe", "en"
    for record_path in sorted(
            recipes_dir.glob("*/record.json")):
        try:
            record = json.loads(
                record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if record.get("source_run_id") == run_id:
            return (record.get("name") or "Untitled recipe",
                    record.get("language") or "en")
    return "Untitled recipe", "en"


def execute_video_pipeline_multimodal(
        storage: StorageBackend,
        record: JobRecord,
        clients: dict[str, Any] | None = None) -> dict[str, bytes]:
    """Run every mandatory multimodal stage (no silent skips)."""
    from bakery_intelligence.api.pipeline import (
        PIPELINE_STATE_FILENAME,
        STAGE_FIRSTPASS_DONE,
        run_pipeline_from_path,
    )
    from bakery_intelligence.inference.meta_spark import (
        MetaSparkClient,
    )
    from bakery_intelligence.recipe_export import proposal as \
        proposal_mod
    from bakery_intelligence.video.intake import probe_video

    clients = clients or {}
    if record.source_object_id is None:
        raise MultimodalError("Multimodal run has no source.")
    if record.grant_json is None:
        raise GrantError(
            "Multimodal workflow requires an authorization grant.")
    grant = parse_grant(json.loads(record.grant_json))
    if grant.workflow != WORKFLOW_VIDEO_PIPELINE_MULTIMODAL:
        raise GrantError(
            f"Grant workflow {grant.workflow!r} does not match "
            f"{WORKFLOW_VIDEO_PIPELINE_MULTIMODAL!r}.")
    run_id = record.run_id
    work_root = storage.run_work_dir(run_id)
    mm_dir = work_root / "multimodal"
    mm_dir.mkdir(parents=True, exist_ok=True)
    calls: list[StageCall] = []

    def fail(stage: str, exc: BaseException) -> MultimodalError:
        calls.append(StageCall(
            stage=stage, provider="none", model="none",
            request_id=None, status="failed",
            elapsed_seconds=0.0,
            error=f"{type(exc).__name__}: {str(exc)[:300]}"))
        write_json(mm_dir / "cost_report.json", cost_report())
        return MultimodalError(f"Stage {stage} failed: {exc}")

    def cost_report() -> dict[str, Any]:
        entries = [call.as_dict() for call in calls]
        meta_total = round(sum(
            call.estimate_usd or 0.0 for call in calls
            if call.provider == "meta"), 6)
        return {
            "run_id": run_id,
            "calls": entries,
            "meta_estimate_usd_total": meta_total,
            "meta_actual_usd_total": None,
            "baseten_cost": BASEN_COST_UNKNOWN,
            "note": "Meta estimates use provider per-token "
                    "rates; actuals are provider-billed.",
        }

    # Credentials resolve before any billable request.
    try:
        meta_key = resolve_api_key()
    except GrantError as exc:
        raise fail("authorize", exc)
    baseten_key_var = BASETEN_KEY_VAR
    import os

    if not os.environ.get(baseten_key_var, "").strip():
        raise fail("authorize", MultimodalError(
            f"Missing {baseten_key_var}; refusing Baseten calls."))
    meta = clients.get("meta") or MetaResponsesClient()
    asr = clients.get("asr") or VoiceTranscribeClient()
    baseten = clients.get("baseten") or BasetenClient()
    validator = clients.get("validator") or get_validator(
        DEFAULT_VALIDATOR, meta)

    source_path = _local_source_path(
        storage, run_id, record.source_object_id)
    probe = probe_video(source_path)
    duration = float(probe.get("duration_seconds") or 0.0)
    if duration <= 0:
        raise fail("probe", MultimodalError(
            "Source video has no measurable duration."))

    # Fail fast: the global native-video stage runs before
    # any frame inference, so a provider rejection of the
    # full-length video terminates the job after one call
    # instead of after hundreds of frame calls. The global
    # stage needs no frame artifacts (upload + duration only).
    reserve = non_frame_reserve(duration)
    frame_budget = grant.max_requests - reserve
    if frame_budget < 1:
        raise fail("authorize", MultimodalError(
            f"Grant allows {grant.max_requests} Meta calls; "
            f"{reserve} are reserved for the non-frame stages "
            f"({duration:.1f}s audio needs "
            f"{reserve - NON_FRAME_BASE_CALLS} ASR chunk "
            "calls)."))

    # Path B: global native video (reused when present).
    try:
        global_record = _reuse_json(
            mm_dir, "global.json", ("result", "request_id"))
        if global_record is None:
            global_record, call = run_global_stage(
                meta, source_path, duration, mm_dir)
            calls.append(call)
        else:
            usage = _usage_of(global_record)
            calls.append(StageCall(
                stage="global", provider="meta",
                model=global_record.get("model")
                or SPARK_MODEL,
                request_id=global_record.get("request_id"),
                status="reused", elapsed_seconds=0.0,
                usage=usage, estimate_usd=0.0))
    except Exception as exc:
        raise fail("global", exc)
    meta_used = 1

    # Path A: frames + first-pass + reconstruction.
    # Same contract as execute_video_pipeline_meta, except the
    # frame runner is capped below the grant so the audio and
    # validation stages keep their reserved calls (global
    # already ran). When the prior pipeline stage reached
    # firstpass_done, the core only re-verifies existing
    # outputs, so this run issues zero new frame calls.
    try:
        state_path = work_root / PIPELINE_STATE_FILENAME
        frames_resumed = False
        if state_path.is_file():
            try:
                prior = json.loads(state_path.read_bytes())
            except (OSError, ValueError):
                prior = {}
            frames_resumed = (isinstance(prior, dict) and prior.get(
                "stage") == STAGE_FIRSTPASS_DONE)
    except Exception as exc:
        raise fail("frames", exc)
    try:
        frame_client = clients.get("frames") or MetaSparkClient(
            model=grant.model,
            timeout_seconds=grant.timeout_seconds,
            store=grant.store)
        frame_artifacts = run_pipeline_from_path(
            video_path=source_path,
            video_name=record.source_object_id,
            run_id=run_id,
            source_object_id=record.source_object_id,
            work_dir=work_root,
            billable_client=frame_client,
            billable_token_budget=grant.token_budget,
            billable_max_requests=frame_budget,
            billable_cost_ceiling_usd=grant.cost_ceiling_usd,
            billable_rates_usd_per_m=grant.rates_usd_per_m,
            billable_strict_cost=grant.strict_cost,
            billable_api_key=meta_key,
            billable_live=True,
            max_frames=grant.max_frames)
        firstpass = json.loads(frame_artifacts[
            "firstpass_report.json"].decode("utf-8"))
        frames_used = int(firstpass.get("requests_sent") or 0)
    except Exception as exc:
        raise fail("frames", exc)
    calls.extend(frame_stage_calls(
        firstpass, resumed=frames_resumed))
    meta_used += frames_used
    remaining = reserve - 1  # global already ran
    if meta_used + remaining > grant.max_requests:
        raise fail("authorize", MultimodalError(
            f"Grant allows {grant.max_requests} Meta calls; "
            f"global + frames used {meta_used} and {remaining} "
            "more are required."))

    # Audio + speech claims.
    try:
        transcript, claims, audio_calls = run_audio_stage(
            meta, asr, source_path, mm_dir)
        calls.extend(audio_calls)
        meta_used += sum(
            1 for call in audio_calls
            if call.provider == "meta")
    except Exception as exc:
        raise fail("audio", exc)

    # Path C: Baseten segments (reused when complete).
    try:
        global_summary = json.dumps(
            global_record.get("result") or {})[:2000]
        reused_segments = _reuse_segments(mm_dir)
        if reused_segments is None:
            segments_index, segment_calls = (
                run_segment_stage(
                    baseten, source_path, duration,
                    global_summary, mm_dir))
            calls.extend(segment_calls)
        else:
            segments_index, segment_records = reused_segments
            for record in segment_records:
                calls.append(StageCall(
                    stage=f"segment_{record.get('index', 0):03d}"
                    if isinstance(record.get("index"), int)
                    else "segment_unk",
                    provider="baseten",
                    model=record.get("model")
                    or BASETEN_MODEL,
                    request_id=record.get("request_id"),
                    status="reused", elapsed_seconds=0.0,
                    usage=record.get("usage") or {},
                    estimate_usd=None,
                    error=BASEN_COST_UNKNOWN))
    except Exception as exc:
        raise fail("segments", exc)

    # Deterministic merge. Path A artifacts exist only as
    # bytes until the worker persists them, so the bundle
    # inputs are materialized first (the worker re-stores
    # identical bytes afterwards).
    try:
        recipe_name, language = _linked_recipe_meta(
            storage, run_id)
        run_artifacts = work_root.parent / "artifacts"
        run_artifacts.mkdir(parents=True, exist_ok=True)
        for name in ("pipeline_summary.json",
                     "measurements.json", "recipe_draft.json",
                     "weighing_event.json"):
            if name in frame_artifacts:
                (run_artifacts / name).write_bytes(
                    frame_artifacts[name])
        firstpass_bundle = (
            proposal_mod.build_evidence_bundle(
                work_root.parent, recipe_name, language))
        segment_records = [
            read_json(mm_dir / f"segment_{s['index']:03d}.json")
            for s in segments_index["segments"]]
        merged = build_merged_evidence(
            firstpass_bundle, global_record, segment_records,
            transcript, claims)
        write_json(mm_dir / "merged.json", merged)
    except Exception as exc:
        raise fail("merge", exc)

    # Evidence validation (reused when present).
    try:
        validation = _reuse_json(
            mm_dir, "validation.json",
            ("result", "model", "provider"))
        if validation is None:
            validation = validator.validate(merged, mm_dir)
            write_json(mm_dir / "validation.json",
                       validation)
            status = "ok"
            usage = validation.get("usage") or {}
            in_tok, out_tok = _spark_tokens(usage)
            estimate = estimate_spark_usd(in_tok, out_tok) \
                if validation.get("provider") == "meta" \
                else None
        else:
            status = "reused"
            usage = _usage_of(validation)
            estimate = 0.0
        calls.append(StageCall(
            stage="validation",
            provider=validation.get("provider") or "meta",
            model=validation.get("model") or SPARK_MODEL,
            request_id=validation.get("request_id"),
            status=status,
            elapsed_seconds=float(
                validation.get("elapsed_seconds") or 0.0),
            usage=usage,
            estimate_usd=estimate))
        if validation.get("provider") == "meta":
            meta_used += 1
    except Exception as exc:
        raise fail("validation", exc)

    # Fusion → proposal-schema recipe draft (reused if present).
    try:
        fusion_record = _reuse_json(
            mm_dir, "fusion_recipe.json", ("result",))
        if fusion_record is None:
            fusion_record, fusion_call = run_fusion_stage(
                baseten, merged, validation, recipe_name,
                language, mm_dir)
            calls.append(fusion_call)
        else:
            calls.append(StageCall(
                stage="fusion", provider="baseten",
                model=fusion_record.get("model")
                or BASETEN_MODEL,
                request_id=fusion_record.get("request_id"),
                status="reused", elapsed_seconds=0.0,
                usage=fusion_record.get("usage") or {},
                estimate_usd=None,
                error=BASEN_COST_UNKNOWN))
    except Exception as exc:
        raise fail("fusion", exc)

    if meta_used > grant.max_requests:
        raise fail("authorize", MultimodalError(
            f"Meta calls used ({meta_used}) exceeded grant "
            f"({grant.max_requests})."))

    artifacts = dict(frame_artifacts)
    for name in ("global.json", "transcript.json",
                 "speech_claims.json", "segments_index.json",
                 "merged.json", "validation.json",
                 "fusion_input.json", "fusion_recipe.json"):
        path = mm_dir / name
        if path.is_file():
            artifacts[f"multimodal_{name}"] = path.read_bytes()
    for segment in segments_index["segments"]:
        path = mm_dir / f"segment_{segment['index']:03d}.json"
        artifacts[f"multimodal_segment_{segment['index']:03d}"
                  f".json"] = path.read_bytes()
    write_json(mm_dir / "pipeline_summary_multimodal.json", {
        "run_id": run_id,
        "workflow": WORKFLOW_VIDEO_PIPELINE_MULTIMODAL,
        "duration_seconds": duration,
        "frames_used": frames_used,
        "meta_calls_used": meta_used,
        "segments": segments_index["count"],
        "recipe_name": recipe_name,
        "language": language,
        "stages": [call.as_dict() for call in calls],
    })
    artifacts["multimodal_pipeline_summary.json"] = (
        mm_dir / "pipeline_summary_multimodal.json"
    ).read_bytes()
    write_json(mm_dir / "cost_report.json", cost_report())
    artifacts["multimodal_cost_report.json"] = (
        mm_dir / "cost_report.json").read_bytes()
    return artifacts
