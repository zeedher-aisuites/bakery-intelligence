"""Phase 2C specialized agents (A-D, F, G; E is deterministic).

Each agent separates offline-pure helpers (prompt building,
response parsing, validation) from one budgeted live execution.
Live paths record prompt versions, request ids, usage, and cost;
failures are explicit and resumable. Agent E lives in the
existing fusion package; Agent G runs deterministic checks
first and calls the LLM critic only when they pass.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from bakery_intelligence.phase2c import media, prompts
from bakery_intelligence.phase2c.clients import (
    BudgetLedger,
    MetaResponsesClient,
    ProviderError,
    VoiceTranscribeClient,
    estimate_asr_usd,
    estimate_spark_usd,
    extract_output_text,
)


@dataclass
class AgentResult:
    agent: str
    clip_id: str
    status: str  # ok | failed | skipped
    prompt_version: str | None = None
    request_id: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    estimate_usd: float = 0.0
    actual_usd: float | None = None
    elapsed_seconds: float = 0.0
    payload: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent, "clip_id": self.clip_id,
            "status": self.status,
            "prompt_version": self.prompt_version,
            "request_id": self.request_id, "usage": self.usage,
            "estimate_usd": self.estimate_usd,
            "actual_usd": self.actual_usd,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
        }


def parse_json_output(text: str) -> Any:
    """Parse model JSON, tolerating fenced code blocks."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    return json.loads(cleaned)


def build_video_content(file_id: str,
                        instruction: str) -> list[dict[str, Any]]:
    return [{"type": "input_text", "text": instruction},
            {"type": "input_video", "file_id": file_id}]


def build_image_content(image_paths: list[Path],
                        instruction: str) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [
        {"type": "input_text", "text": instruction}]
    for path in image_paths:
        encoded = base64.b64encode(
            path.read_bytes()).decode("ascii")
        parts.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{encoded}",
        })
    return parts


def usage_of(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage")
    return dict(usage) if isinstance(usage, dict) else {}


def actual_spark_usd(usage: dict[str, Any]) -> float | None:
    try:
        return estimate_spark_usd(
            int(usage.get("input_tokens", 0)),
            int(usage.get("output_tokens", 0)))
    except (TypeError, ValueError):
        return None


def run_agent_a_global(client: MetaResponsesClient,
                       ledger: BudgetLedger,
                       clip_id: str, video_path: Path,
                       work_dir: Path,
                       estimate_usd: float = 0.30) -> AgentResult:
    """Global native-video analysis of one full clip."""
    prompt = prompts.GLOBAL_ANALYST
    label = f"agent-a/{clip_id}"
    ledger.check(estimate_usd, label)
    try:
        file_id = client.upload_file(video_path)
        payload = client.create_response(
            prompt.model,
            build_video_content(file_id, prompt.instruction))
        text = extract_output_text(payload)
        parsed = parse_json_output(text)
        violations = prompts.validate_output(prompt, parsed)
        if violations:
            raise ProviderError(
                "Schema violations: " + "; ".join(violations))
        usage = usage_of(payload)
        actual = actual_spark_usd(usage)
        media.write_json(work_dir / f"{clip_id}.agent-a.json",
                         {"file_id": file_id, "parsed": parsed,
                          "request_id": payload.get("id"),
                          "usage": usage})
        result = AgentResult(
            agent="A", clip_id=clip_id, status="ok",
            prompt_version=prompt.version,
            request_id=payload.get("id"), usage=usage,
            estimate_usd=estimate_usd, actual_usd=actual,
            elapsed_seconds=float(
                payload.get("_elapsed_seconds", 0.0)),
            payload=parsed)
    except Exception as exc:  # noqa: BLE001 - recorded, resumable
        result = AgentResult(
            agent="A", clip_id=clip_id, status="failed",
            prompt_version=prompt.version,
            estimate_usd=estimate_usd, error=str(exc)[:500])
    ledger.record({"agent": "A", "clip_id": clip_id,
                   "label": label,
                   "estimate_usd": result.estimate_usd,
                   "actual_usd": result.actual_usd,
                   "request_id": result.request_id,
                   "usage": result.usage,
                   "error": result.error})
    return result


def run_agent_b_local(
        client: MetaResponsesClient, ledger: BudgetLedger,
        clip_id: str, windows: list[tuple[float, float]],
        cut: Callable[[float, float, Path], Path],
        work_dir: Path,
        estimate_per_window_usd: float = 0.10) -> AgentResult:
    """Targeted native-video analysis of short windows."""
    prompt = prompts.LOCAL_ANALYST
    segments: list[dict[str, Any]] = []
    total_actual = 0.0
    request_ids: list[str] = []
    for index, (start, end) in enumerate(windows):
        label = f"agent-b/{clip_id}/w{index}"
        ledger.check(estimate_per_window_usd, label)
        clip_path = work_dir / f"{clip_id}.w{index}.mp4"
        try:
            cut(start, end - start, clip_path)
            file_id = client.upload_file(clip_path)
            payload = client.create_response(
                prompt.model,
                build_video_content(file_id, prompt.instruction))
            parsed = parse_json_output(extract_output_text(payload))
            violations = prompts.validate_output(prompt, parsed)
            if violations:
                raise ProviderError(
                    "Schema violations: " + "; ".join(violations))
            usage = usage_of(payload)
            actual = actual_spark_usd(usage) or 0.0
            total_actual += actual
            request_ids.append(str(payload.get("id")))
            segments.append(
                {"window": [start, end], "file_id": file_id,
                 "request_id": payload.get("id"), "usage": usage,
                 "parsed": parsed})
            ledger.record(
                {"agent": "B", "clip_id": clip_id, "label": label,
                 "estimate_usd": estimate_per_window_usd,
                 "actual_usd": actual,
                 "request_id": payload.get("id"), "usage": usage,
                 "error": None})
        except Exception as exc:  # noqa: BLE001 - recorded
            ledger.record(
                {"agent": "B", "clip_id": clip_id, "label": label,
                 "estimate_usd": estimate_per_window_usd,
                 "actual_usd": None, "request_id": None,
                 "usage": {}, "error": str(exc)[:300]})
            return AgentResult(
                agent="B", clip_id=clip_id, status="failed",
                prompt_version=prompt.version,
                estimate_usd=estimate_per_window_usd * len(
                    windows),
                error=f"window {index}: {exc}"[:500],
                payload={"segments": segments})
    media.write_json(work_dir / f"{clip_id}.agent-b.json",
                     {"segments": segments})
    return AgentResult(
        agent="B", clip_id=clip_id, status="ok",
        prompt_version=prompt.version,
        request_id=",".join(request_ids),
        estimate_usd=estimate_per_window_usd * len(windows),
        actual_usd=total_actual, payload={"segments": segments})


def normalize_transcript_turns(
        payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize ASR turns to clip-local culinary transcript rows."""
    turns = payload.get("turns")
    if not isinstance(turns, list):
        raise ProviderError("ASR response carries no turns array.")
    rows: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        text = turn.get("transcript") or turn.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        rows.append({
            "turn_index": index,
            "turn_id": turn.get("turnId", index),
            "speaker": turn.get("speaker"),
            "start_seconds": _ms_to_s(
                turn.get("startMs", turn.get("audioStartMs"))),
            "end_seconds": _ms_to_s(
                turn.get("endMs", turn.get("audioProcessedMs"))),
            "transcript": text.strip(),
            "status": "ai_suggested",
        })
    return rows


def _ms_to_s(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) / 1000.0
    return None


def run_agent_c_transcribe(
        client: VoiceTranscribeClient, ledger: BudgetLedger,
        clip_id: str, wav_path: Path, audio_seconds: float,
        work_dir: Path) -> AgentResult:
    """Transcribe one clip's audio with turn timestamps."""
    label = f"agent-c/{clip_id}"
    estimate = round(estimate_asr_usd(audio_seconds) + 0.02, 4)
    ledger.check(estimate, label)
    try:
        payload = client.transcribe(wav_path)
        turns = normalize_transcript_turns(payload)
        media.write_json(work_dir / f"{clip_id}.agent-c.json",
                         {"turns": turns,
                          "raw_request_elapsed": payload.get(
                              "_elapsed_seconds")})
        result = AgentResult(
            agent="C", clip_id=clip_id, status="ok",
            prompt_version=None, request_id=None,
            estimate_usd=estimate, actual_usd=estimate,
            elapsed_seconds=float(
                payload.get("_elapsed_seconds", 0.0)),
            payload={"turns": turns})
    except Exception as exc:  # noqa: BLE001 - recorded
        result = AgentResult(
            agent="C", clip_id=clip_id, status="failed",
            estimate_usd=estimate, error=str(exc)[:500])
    ledger.record({"agent": "C", "clip_id": clip_id,
                   "label": label,
                   "estimate_usd": result.estimate_usd,
                   "actual_usd": result.actual_usd,
                   "request_id": result.request_id,
                   "usage": {"audio_seconds": audio_seconds},
                   "error": result.error})
    return result


D_BATCH_SIZE = 4


def run_agent_d_measure(
        client: MetaResponsesClient, ledger: BudgetLedger,
        clip_id: str, frames: list[tuple[Path, float]],
        work_dir: Path,
        estimate_per_frame_usd: float = 0.02) -> AgentResult:
    """Read scale displays in selected frames (small batches).

    Frames go out at most D_BATCH_SIZE per call: oversized
    image batches time out provider-side (observed 504).
    """
    prompt = prompts.MEASUREMENT_ANALYST
    batches = [frames[index:index + D_BATCH_SIZE]
               for index in range(0, len(frames), D_BATCH_SIZE)]
    readings: list[dict[str, Any]] = []
    total_actual = 0.0
    request_ids: list[str] = []
    for number, batch in enumerate(batches):
        label = f"agent-d/{clip_id}/b{number}"
        estimate = round(estimate_per_frame_usd * len(batch), 4)
        try:
            ledger.check(estimate, label)
            payload = client.create_response(
                prompt.model,
                build_image_content(
                    [path for path, _ in batch],
                    prompt.instruction),
                max_output_tokens=8192)
            parsed = parse_json_output(
                extract_output_text(payload))
            violations = prompts.validate_output(prompt, parsed)
            if violations:
                raise ProviderError(
                    "Schema violations: " + "; ".join(violations))
            usage = usage_of(payload)
            actual = actual_spark_usd(usage) or 0.0
            total_actual += actual
            request_ids.append(str(payload.get("id")))
            readings.extend(parsed.get("readings") or [])
            ledger.record(
                {"agent": "D", "clip_id": clip_id,
                 "label": label, "estimate_usd": estimate,
                 "actual_usd": actual,
                 "request_id": payload.get("id"), "usage": usage,
                 "error": None})
        except Exception as exc:  # noqa: BLE001 - recorded
            ledger.record(
                {"agent": "D", "clip_id": clip_id,
                 "label": label, "estimate_usd": estimate,
                 "actual_usd": None, "request_id": None,
                 "usage": {}, "error": str(exc)[:300]})
            return AgentResult(
                agent="D", clip_id=clip_id, status="failed",
                prompt_version=prompt.version,
                estimate_usd=estimate, error=str(exc)[:500])
    parsed_all = {"readings": readings}
    media.write_json(work_dir / f"{clip_id}.agent-d.json", {
        "frames": [{"path": path.name, "clip_local_seconds": sec}
                   for path, sec in frames],
        "request_ids": request_ids,
        "batches": len(batches),
        "parsed": parsed_all})
    return AgentResult(
        agent="D", clip_id=clip_id, status="ok",
        prompt_version=prompt.version,
        request_id=",".join(request_ids),
        estimate_usd=round(
            estimate_per_frame_usd * len(frames), 4),
        actual_usd=total_actual, payload=parsed_all)


def run_text_agent(client: MetaResponsesClient,
                   ledger: BudgetLedger, prompt: prompts.RolePrompt,
                   agent: str, clip_id: str, bundle: dict[str, Any],
                   work_dir: Path, filename: str,
                   estimate_usd: float) -> AgentResult:
    """Run a text-only role (speech claims, recipe, critic)."""
    label = f"agent-{agent.lower()}/{clip_id}/{filename}"
    ledger.check(estimate_usd, label)
    content = [{"type": "input_text",
                "text": prompt.instruction
                + "\n\nEVIDENCE BUNDLE (untrusted):\n"
                + json.dumps(bundle)[:60000]}]
    try:
        payload = client.create_response(
            prompt.model, content, max_output_tokens=16384)
        parsed = parse_json_output(extract_output_text(payload))
        violations = prompts.validate_output(prompt, parsed)
        if violations:
            raise ProviderError(
                "Schema violations: " + "; ".join(violations))
        usage = usage_of(payload)
        actual = actual_spark_usd(usage)
        media.write_json(work_dir / filename,
                         {"request_id": payload.get("id"),
                          "usage": usage, "parsed": parsed})
        result = AgentResult(
            agent=agent, clip_id=clip_id, status="ok",
            prompt_version=prompt.version,
            request_id=payload.get("id"), usage=usage,
            estimate_usd=estimate_usd, actual_usd=actual,
            elapsed_seconds=float(
                payload.get("_elapsed_seconds", 0.0)),
            payload=parsed)
    except Exception as exc:  # noqa: BLE001 - recorded
        result = AgentResult(
            agent=agent, clip_id=clip_id, status="failed",
            prompt_version=prompt.version,
            estimate_usd=estimate_usd, error=str(exc)[:500])
    ledger.record({"agent": agent, "clip_id": clip_id,
                   "label": label,
                   "estimate_usd": result.estimate_usd,
                   "actual_usd": result.actual_usd,
                   "request_id": result.request_id,
                   "usage": result.usage,
                   "error": result.error})
    return result
