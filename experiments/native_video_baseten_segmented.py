"""Sequential, evidence-preserving native-video benchmark for Baseten Muse Spark Open.

Processes one source MP4 in fixed-duration clips with bounded textual context
from earlier clips. A separate response and evaluation are saved for each clip.
No frame-based fallback is used. The original single-request experiment remains
unchanged. This is a segmented-context architecture, not a single-video call.

Requires the existing experiments/native_video_baseten.py and
experiments/native_video_understanding.py modules, ffmpeg, ffprobe, and openai.

Examples (from repository root):
    uv run --env-file .env python experiments/native_video_baseten_segmented.py VIDEO.mp4
    uv run --env-file .env python experiments/native_video_baseten_segmented.py VIDEO.mp4 --execute --max-segments 2
    uv run --env-file .env python experiments/native_video_baseten_segmented.py VIDEO.mp4 --execute --resume RUN_DIRECTORY
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_video_baseten as original

base = original.base
MODEL = original.AUTHORIZED_MODEL
BASE_URL = original.BASE_URL
OUT_DIR = Path(__file__).resolve().parent / "native_video_baseten" / "segmented_runs"
VERSION = "segmented-v1"


def save_json(path: Path, value: object) -> None:
    """Atomically persist metadata without request video bytes or API secrets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def parse_model_json(content: str) -> dict:
    """Accept pure JSON or a single fenced JSON object, without schema invention."""
    value = content.strip().lstrip("\ufeff")
    match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\s*```", value, re.DOTALL | re.IGNORECASE)
    if match:
        value = match.group(1).strip()
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Model response must be a JSON object")
    events = parsed.get("culinary_events", parsed.get("operations"))
    if not isinstance(events, list):
        raise ValueError("Missing culinary_events or operations array")
    for key in ("ingredients", "measurements", "scale_readings"):
        if key in parsed and not isinstance(parsed[key], list):
            raise ValueError(f"{key} must be an array")
    return parsed


def events_from(parsed: dict) -> list:
    return parsed.get("culinary_events", parsed.get("operations", []))


def safe_str(value: object, limit: int) -> str:
    if isinstance(value, list):
        value = ", ".join(str(x) for x in value[:3])
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def segment_digest(parsed: dict, index: int, start: float, end: float,
                   max_chars: int = 1100) -> str:
    """Compact, explicitly unverified event notes for the next clip."""
    lines = [f"Segment {index:02d} ({start:.2f}-{end:.2f}s source timeline):"]
    for event in events_from(parsed):
        if not isinstance(event, dict):
            continue
        local_start = event.get("start_timestamp_seconds", event.get("start_seconds"))
        local_end = event.get("end_timestamp_seconds", event.get("end_seconds"))
        t = "time unknown"
        if isinstance(local_start, (int, float)) and not isinstance(local_start, bool):
            t = f"{start + local_start:.1f}s"
            if isinstance(local_end, (int, float)) and not isinstance(local_end, bool):
                t += f"-{start + local_end:.1f}s"
        action = safe_str(event.get("action_type", event.get("operation_type")), 48)
        description = safe_str(event.get("action_description", event.get("description")), 135)
        uncertainty = safe_str(event.get("uncertainties", []), 95)
        lines.append(f"- {t} {action}: {description}" + (f" [uncertain: {uncertainty}]" if uncertainty else ""))
    if len(lines) == 1:
        lines.append("- No structured events were identified.")
    digest = "\n".join(lines)
    if len(digest) > max_chars:
        digest = digest[:max_chars - 27].rsplit("\n", 1)[0] + "\n[Digest shortened for context]"
    return digest


def prior_context(run_dir: Path, current_index: int, max_chars: int) -> tuple[str, list[int]]:
    """Include previous summaries and explicit failed-segment gaps."""
    lines = []
    gaps = []
    for i in range(1, current_index):
        result_path = run_dir / "segments" / f"segment_{i:02d}" / "result.json"
        if not result_path.exists():
            gaps.append(i)
            lines.append(f"Segment {i:02d}: MISSING; do not infer its contents.")
            continue
        result = load_json(result_path)
        if result.get("status") == "completed":
            lines.append(result.get("context_digest", f"Segment {i:02d}: summary unavailable"))
        else:
            gaps.append(i)
            lines.append(f"Segment {i:02d}: FAILED; no trustworthy observation available.")
    context = "\n".join(lines)
    if len(context) > max_chars:
        # Keep the latest evidence; the complete history remains on disk.
        context = "[Earlier summaries omitted due to context size limit.]\n" + context[-(max_chars - 70):]
    return context, gaps


def make_prompt(base_prompt: str, index: int, total: int, start: float,
                duration: float, context: str) -> str:
    return (
        base_prompt + "\n\n"
        "SEGMENTED-VIDEO EVALUATION INSTRUCTIONS:\n"
        f"You are viewing segment {index} of {total}. The current clip covers "
        f"source-video time {start:.3f}s through {start + duration:.3f}s "
        f"(clip duration approximately {duration:.3f}s).\n"
        "Report ONLY events visually or audibly evidenced in the CURRENT clip. "
        "All event timestamps must be LOCAL to this clip, starting at 0 seconds. "
        "Do not repeat events from earlier clips or claim to have observed missing clips. "
        "Earlier summaries below are unverified model-generated context, NOT ground truth. "
        "Use them only for continuity of object references; do not treat them "
        "as visual evidence of the current clip. Do not invent ingredient identities, "
        "scale digits, units, quantities, or missing steps.\n"
        "Return ONLY one valid JSON object (no Markdown code fence, no explanation). "
        "Preserve the original output schema requested above, especially "
        "operations, ingredients, scale_readings, and temporal_coverage. "
        "operations must be an array with event_id, start_seconds, end_seconds, "
        "action_type, action_description, evidence, and uncertainty. "
        "If no events are visible, return an empty operations array.\n"
        "PRIOR CONTEXT (may contain gaps):\n"
        + (context if context else "No earlier clips.")
    )


def source_segments(duration: float, seconds: float) -> list[dict]:
    count = math.ceil((duration - 1e-6) / seconds)
    return [
        {"index": i + 1, "start_seconds": round(i * seconds, 6),
         "duration_seconds": round(min(seconds, duration - i * seconds), 6)}
        for i in range(count)
    ]


def clip_path(run_dir: Path, index: int) -> Path:
    return run_dir / "clips" / f"segment_{index:02d}.mp4"


def make_clip(video: Path, clip: Path, start: float, duration: float) -> None:
    if clip.exists() and clip.stat().st_size > 1024:
        return
    clip.parent.mkdir(parents=True, exist_ok=True)
    tmp = clip.with_name(clip.stem + ".partial.mp4")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-ss", f"{start:.6f}", "-i", str(video), "-t", f"{duration:.6f}",
           "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "25", "-c:a", "aac",
           "-b:a", "128k", "-movflags", "+faststart", str(tmp)]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed: {completed.stderr[-500:]}")
    if not tmp.exists() or tmp.stat().st_size <= 1024:
        raise RuntimeError("ffmpeg produced an empty or missing clip")
    os.replace(tmp, clip)


def scrub_error(exc: Exception, key: str) -> str:
    message = str(exc)
    if key:
        message = message.replace(key, "[REDACTED]")
    message = re.sub(r"data:video/[^\s\"']+", "[REDACTED_VIDEO_DATA]", message)
    return message[:500]


def recover_saved_response(path: Path) -> tuple[dict, dict, str, str]:
    response = load_json(path)
    choice = response["choices"][0]
    content = choice["message"]["content"]
    parsed = parse_model_json(content)
    return response, parsed, content, choice.get("finish_reason")


def usage_from(response: dict) -> dict:
    usage = response.get("usage") or {}
    return {
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def estimated_cost(usage: dict, rate_in: float, rate_out: float) -> float | None:
    a, b = usage.get("input_tokens"), usage.get("output_tokens")
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    return round((a * rate_in + b * rate_out) / 1_000_000, 8)


def request_segment(client, model: str, clip: Path, prompt: str, max_tokens: int):
    raw = clip.read_bytes()
    data_url = "data:video/mp4;base64," + base64.b64encode(raw).decode("ascii")
    del raw
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": data_url}},
        ]}],
        max_tokens=max_tokens,
    )
    del data_url
    return response.model_dump() if hasattr(response, "model_dump") else response


def process_segment(client, model: str, source: Path, run_dir: Path, seg: dict,
                    total: int, args: argparse.Namespace, base_prompt: str) -> dict:
    index, start, duration = seg["index"], seg["start_seconds"], seg["duration_seconds"]
    seg_dir = run_dir / "segments" / f"segment_{index:02d}"
    seg_dir.mkdir(parents=True, exist_ok=True)
    context, gaps = prior_context(run_dir, index, args.context_max_chars)
    prompt = make_prompt(base_prompt, index, total, start, duration, context)
    (seg_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    clip = clip_path(run_dir, index)
    attempts = []
    parsed = None
    response = None
    finish_reason = None
    last_error = None
    status = "failed"
    try:
        make_clip(source, clip, start, duration)
        measured_duration = base.probe_duration_seconds(clip).get("duration_seconds")
        if not isinstance(measured_duration, (int, float)) or abs(measured_duration - duration) > 0.5:
            raise ValueError(f"Clip duration mismatch: expected {duration}s, got {measured_duration}s")
        request_info = {
            "model": model, "base_url": BASE_URL, "segment_index": index,
            "clip_measured_duration_seconds": measured_duration,
            "source_start_seconds": start, "requested_duration_seconds": duration,
            "clip_size_bytes": clip.stat().st_size,
            "clip_sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prior_context_characters": len(context), "prior_failed_segments": gaps,
            "max_tokens": args.max_tokens, "max_attempts": args.max_attempts,
            "video_content_part": "inline video_url data URL (payload not saved)",
        }
        save_json(seg_dir / "request.json", request_info)
        for attempt in range(1, args.max_attempts + 1):
            response_path = seg_dir / f"response_attempt_{attempt}.json"
            start_time = time.perf_counter()
            try:
                if response_path.exists():
                    response, parsed, _, finish_reason = recover_saved_response(response_path)
                    source_kind = "saved_response"
                else:
                    response = request_segment(client, model, clip, prompt, args.max_tokens)
                    save_json(response_path, response)  # Save before parsing; avoid double billing on resume.
                    _, parsed, _, finish_reason = recover_saved_response(response_path)
                    source_kind = "api"
                if finish_reason == "length":
                    raise ValueError("Response finished at max token limit")
                attempts.append({"attempt": attempt, "status": "completed", "source": source_kind,
                                 "response_id": response.get("id"),
                                 "latency_seconds": round(time.perf_counter() - start_time, 3),
                                 "usage": usage_from(response),
                                 "estimated_cost_usd": estimated_cost(usage_from(response), args.rate_input_per_m,
                                                                      args.rate_output_per_m)})
                status = "completed"
                break
            except Exception as exc:
                last_error = scrub_error(exc, args.api_key)
                error_status = getattr(exc, "status_code", None)
                attempts.append({"attempt": attempt, "status": "failed",
                                 "error_type": type(exc).__name__, "http_status": error_status,
                                 "error_detail": last_error,
                                 "latency_seconds": round(time.perf_counter() - start_time, 3),
                                 "usage": usage_from(response) if isinstance(response, dict) else None,
                                 "estimated_cost_usd": estimated_cost(usage_from(response), args.rate_input_per_m,
                                                                      args.rate_output_per_m) if isinstance(response, dict) else None})
                # A saved malformed response must not be re-submitted in the same attempt.
                if error_status is not None and error_status not in (408, 409, 429) and error_status < 500:
                    break
                if attempt < args.max_attempts:
                    time.sleep(min(3 * attempt, 10))
                parsed, response = None, None
    except Exception as exc:
        last_error = scrub_error(exc, args.api_key)

    result = {
        **seg, "status": status, "model": model,
        "context_gap_indices": gaps, "attempts": attempts,
        "error": last_error if status != "completed" else None,
        "finish_reason": finish_reason,
        "event_count": len(events_from(parsed)) if parsed else None,
        "ingredient_record_count": len(parsed.get("ingredients", [])) if parsed else None,
        "measurement_record_count": len(parsed.get("measurements", parsed.get("scale_readings", []))) if parsed else None,
        "context_digest": segment_digest(parsed, index, start, start + duration) if parsed else None,
    }
    if parsed:
        save_json(seg_dir / "parsed_output.json", parsed)
    save_json(seg_dir / "result.json", result)
    return result


def aggregate(run_dir: Path, manifest: dict, args: argparse.Namespace) -> dict:
    events, ingredients, measurements, outcomes, missing = [], [], [], [], []
    for seg in manifest["segments"]:
        i, start = seg["index"], seg["start_seconds"]
        path = run_dir / "segments" / f"segment_{i:02d}" / "result.json"
        if not path.exists():
            missing.append(i)
            continue
        result = load_json(path)
        outcomes.append(result)
        if result["status"] != "completed":
            continue
        parsed = load_json(path.parent / "parsed_output.json")
        for n, event in enumerate(events_from(parsed), 1):
            if not isinstance(event, dict):
                continue
            item = copy.deepcopy(event)
            item["source_event_id"] = item.get("event_id")
            item["event_id"] = f"S{i:02d}_E{n:02d}"
            item["segment_index"] = i
            item["segment_source_start_seconds"] = start
            for key in ("start_timestamp_seconds", "end_timestamp_seconds", "start_seconds", "end_seconds"):
                val = item.get(key)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    item["absolute_" + key] = round(start + val, 4)
            events.append(item)
        for key, dest in (("ingredients", ingredients), ("measurements", measurements)):
            for record in parsed.get(key, []):
                if isinstance(record, dict):
                    item = copy.deepcopy(record)
                    item["segment_index"] = i
                    item["segment_source_start_seconds"] = start
                    val = item.get("timestamp_seconds")
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        item["absolute_timestamp_seconds"] = round(start + val, 4)
                    dest.append(item)
        for record in parsed.get("scale_readings", []):
            if isinstance(record, dict):
                item = copy.deepcopy(record)
                item["segment_index"] = i
                item["segment_source_start_seconds"] = start
                val = item.get("timestamp_seconds")
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    item["absolute_timestamp_seconds"] = round(start + val, 4)
                measurements.append(item)
    completed = [x for x in outcomes if x["status"] == "completed"]
    failed = [x["index"] for x in outcomes if x["status"] != "completed"]
    usages = [a for x in outcomes for a in x.get("attempts", [])]
    cost = sum(a["estimated_cost_usd"] for a in usages if a.get("estimated_cost_usd") is not None)
    total_input = sum((a.get("usage") or {}).get("input_tokens") or 0 for a in usages)
    total_output = sum((a.get("usage") or {}).get("output_tokens") or 0 for a in usages)
    status = "completed" if not failed and not missing else "incomplete"
    result = {
        "experiment_version": VERSION,
        "architecture": "sequential native-video clips with bounded accumulated textual context",
        "source_video": manifest["video"],
        "model": manifest["model"],
        "status": status,
        "segments_planned": len(manifest["segments"]),
        "segments_completed": len(completed),
        "segments_failed": failed,
        "segments_pending": missing,
        "event_count": len(events),
        "ingredient_record_count": len(ingredients),
        "measurement_record_count": len(measurements),
        "usage": {"input_tokens": total_input, "output_tokens": total_output,
                  "total_tokens": total_input + total_output},
        "estimated_cost_usd": round(cost, 8),
        "cost_note": "Estimate from configured rates; deployment billing may differ. "
                     "Failed requests without usage are not included.",
        "limitations": [
            "Segmented processing with model-generated context is not equivalent to one native full-video inference.",
            "Event timestamps and labels are model-generated and require visual ground-truth validation.",
            "Context digests may omit details; full original per-segment responses are preserved.",
            "Coverage indicates processed segments, not verified semantic accuracy.",
            "Ingredient records and measurements are not necessarily unique or numerically verified.",
        ],
        "culinary_events": events,
        "ingredients": ingredients,
        "measurements": measurements,
        "segment_results": [{k: x.get(k) for k in ("index", "status", "event_count", "context_gap_indices")}
                            for x in outcomes],
    }
    save_json(run_dir / "aggregated.json", result)
    save_json(run_dir / "evaluation.json", {k: v for k, v in result.items()
                                            if k not in ("culinary_events", "ingredients", "measurements")})
    return result


def get_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", type=Path)
    p.add_argument("--execute", action="store_true", help="Run billable Baseten requests.")
    p.add_argument("--resume", type=Path, help="Continue an interrupted run without repeating completed clips.")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--segment-seconds", type=float, default=30.0)
    p.add_argument("--max-segments", type=int, default=None, help="Optional first-N-segment smoke test.")
    p.add_argument("--max-attempts", type=int, default=2, help="Bounded attempts per clip; default 2.")
    p.add_argument("--max-tokens", type=int, default=original.DEFAULT_MAX_TOKENS)
    p.add_argument("--timeout-seconds", type=float, default=original.DEFAULT_TIMEOUT_SECONDS)
    p.add_argument("--context-max-chars", type=int, default=26000)
    p.add_argument("--rate-input-per-m", type=float, default=original.DEFAULT_RATE_INPUT_PER_M)
    p.add_argument("--rate-output-per-m", type=float, default=original.DEFAULT_RATE_OUTPUT_PER_M)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = get_args(argv)
    if args.segment_seconds <= 0 or args.segment_seconds > 300:
        raise SystemExit("--segment-seconds must be between 0 and 300.")
    if args.max_attempts < 1 or args.max_attempts > 5:
        raise SystemExit("--max-attempts must be between 1 and 5.")
    if args.max_tokens <= 0 or args.context_max_chars < 2000:
        raise SystemExit("--max-tokens must be positive and --context-max-chars >= 2000.")
    if args.max_segments is not None and args.max_segments < 1:
        raise SystemExit("--max-segments must be positive.")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise SystemExit("ffmpeg and ffprobe must be installed and available on PATH.")

    source = args.video.resolve()
    video = base.validate_video(source)
    duration = base.probe_duration_seconds(source)
    seconds = duration.get("duration_seconds")
    if not isinstance(seconds, (float, int)) or seconds <= 0:
        raise SystemExit("Source duration could not be measured.")
    segments = source_segments(float(seconds), args.segment_seconds)
    if args.resume:
        run_dir = args.resume.resolve()
        manifest = load_json(run_dir / "manifest.json")
        if (manifest.get("experiment_version") != VERSION or
            manifest["video"]["sha256"] != video["sha256"] or
            manifest["segment_seconds"] != args.segment_seconds or
            manifest["model"] != MODEL):
            raise SystemExit("Resume settings/source mismatch; existing run left unchanged.")
    else:
        run_dir = args.out_dir.resolve() / f"{source.stem}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:6]}"
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = {"experiment_version": VERSION, "video": {**video, **duration},
                    "model": MODEL, "segment_seconds": args.segment_seconds,
                    "segments": segments, "created_at_utc": datetime.now(timezone.utc).isoformat()}
        save_json(run_dir / "manifest.json", manifest)

    print(f"source: {source.name} ({seconds:.3f}s); segments: {len(segments)}")
    print(f"run: {run_dir}")
    if not args.execute:
        save_json(run_dir / "plan.json", {"mode": "dry-run", "billable": False,
                                          "segments": segments, "model": MODEL,
                                          "notes": "Use --execute for inference; --resume to continue a run."})
        print("dry-run complete; no Baseten requests sent.")
        return 0

    args.api_key, model = original.read_config()
    from openai import OpenAI
    client = OpenAI(base_url=BASE_URL, api_key=args.api_key,
                    timeout=args.timeout_seconds, max_retries=0)
    base_prompt = base.build_prompt()
    count = min(len(segments), args.max_segments or len(segments))
    for seg in segments[:count]:
        index = seg["index"]
        result_path = run_dir / "segments" / f"segment_{index:02d}" / "result.json"
        if result_path.exists():
            existing = load_json(result_path)
            # Never overwrite an existing attempt. Failed clips remain explicit gaps.
            print(f"[{index:02d}/{len(segments):02d}] existing {existing['status']}; skipped")
            continue
        print(f"[{index:02d}/{len(segments):02d}] processing {seg['start_seconds']:.1f}s "
              f"+{seg['duration_seconds']:.1f}s", flush=True)
        result = process_segment(client, model, source, run_dir, seg, len(segments), args, base_prompt)
        print(f"[{index:02d}/{len(segments):02d}] {result['status']}; "
              f"events={result['event_count']}; attempts={len(result['attempts'])}", flush=True)
        aggregate(run_dir, manifest, args)  # Checkpoint after every clip.
    final = aggregate(run_dir, manifest, args)
    print(f"status: {final['status']}; completed={final['segments_completed']}/{final['segments_planned']}; "
          f"events={final['event_count']}; failed={final['segments_failed']}; "
          f"estimated USD={final['estimated_cost_usd']:.4f}")
    print(f"results: {run_dir}")
    return 0 if final["status"] == "completed" or args.max_segments is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
