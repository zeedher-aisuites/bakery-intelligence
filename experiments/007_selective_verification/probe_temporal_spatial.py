"""Disposable temporal/spatial fidelity probe (Exp. 010).

Tests whether temporal context, denser sampling, and PNG fidelity
change Muse Spark 1.3's reading of the scale display at t=150 s in
P04_104. Three modes: prepare (extract + manifest, no API calls),
dry-run (validate + display plan, no API calls), run (adaptive
inference with at most three token budgets per condition,
user-invoked only; at most 15 requests total).

EVAL-ONLY REFERENCE: the human value below is used exclusively for
post-inference comparison. It is never placed in prompts, requests,
image labels, or model-visible metadata, and it never triggers a
retry.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "006_p04_multimodal_inference"))

import meta_client  # noqa: E402

TARGET_TIMESTAMP = 150.0
DENSE_TIMESTAMPS = (
    149.000, 149.333, 149.667, 150.000, 150.333, 150.667, 151.000,
)
CONTEXT_TIMESTAMPS = (148.0, 149.0, 150.0, 151.0, 152.0)

# Human reference. POST-INFERENCE EVALUATION ONLY. Never sent to the
# model and never consulted by the retry policy.
REFERENCE_DISPLAY = "515"
REFERENCE_NUMERIC = 515.0

MODEL = meta_client.MODEL_SPARK
TIMEOUT_SECONDS = meta_client.DEFAULT_TIMEOUT_SECONDS
TOKEN_BUDGETS = (1024, 2048, 3072)
MAX_ATTEMPTS = len(TOKEN_BUDGETS)
MAX_TOTAL_REQUESTS = 5 * MAX_ATTEMPTS
# store=false: run_manual_01 proved store=true cannot persist the ~27 MB
# condition-E request (HTTP 413 recommends exactly this setting).
REQUEST_STORE = False
# Standard-tier rates, REPORTED_NOT_DIRECTLY_VERIFIED (see Exp. 005
# cost report): official pricing page never directly retrieved.
INPUT_USD_PER_1M = 1.25
OUTPUT_USD_PER_1M = 4.25

MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

PROBE_PROMPT = """You are analyzing frames from a cooking video for a display-reading study.
The TARGET timestamp is {target:.3f} seconds. The images below are listed in chronological order with their timestamps:
{image_table}
Identify the digital weighing scale associated with the target frame and read its visible display specifically at the target timestamp. Neighboring frames are supporting temporal evidence only; do not assume the reading stays constant. Distinguish directly visible digits from inferred digits. Abstain when the evidence is insufficient. Never invent units, ingredients, or actions.
Reply with exactly one JSON object and no other text, using these keys:
{{
  "target_timestamp_seconds": number,
  "display_text": "exact digits/segments on the display at the target timestamp, or null",
  "numeric_value": "number only when the display text is an unambiguous plain number, else null",
  "readable": true/false,
  "confidence": "high/medium/low",
  "evidence_timestamps": ["timestamps (seconds) of the frames actually used as evidence"],
  "direct_visual_evidence": "which digits were directly visible at the target timestamp",
  "temporal_inference_used": true/false,
  "uncertainty_notes": "short note or null"
}}"""

CONDITION_SPECS = (
    {"id": "A", "kind": "pipeline_jpg",
     "timestamps": (150.0,),
     "purpose": "Baseline: existing pipeline JPG at target."},
    {"id": "B", "kind": "pipeline_jpg",
     "timestamps": CONTEXT_TIMESTAMPS,
     "purpose": "Temporal context: five pipeline JPGs."},
    {"id": "C", "kind": "fresh_jpg",
     "timestamps": DENSE_TIMESTAMPS,
     "purpose": "Denser sampling: seven fresh JPGs at 3 FPS."},
    {"id": "D", "kind": "fresh_png",
     "timestamps": (150.0,),
     "purpose": "Image fidelity: one fresh PNG at target."},
    {"id": "E", "kind": "fresh_png",
     "timestamps": DENSE_TIMESTAMPS,
     "purpose": "Temporal context plus fidelity: seven fresh PNGs."},
)

COMPARISONS = (
    {"pair": "B_vs_A",
     "changed": ["image_count", "temporal_context"]},
    {"pair": "C_vs_A",
     "changed": ["image_count", "sampling_density", "fresh_extraction"]},
    {"pair": "D_vs_A",
     "changed": ["encoding"]},
    {"pair": "E_vs_C",
     "changed": ["encoding"]},
    {"pair": "E_vs_B",
     "changed": ["image_count", "sampling_density", "encoding"]},
)


class ProbeError(Exception):
    """Structured probe failure with a machine-readable code."""

    def __init__(self, code: str, message: str, entity_id: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.entity_id = entity_id


def sha256_file(path: Path) -> str:
    """SHA-256 over exact file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_index_for(requested: float, fps: float) -> int:
    """Source frame index, mirroring extract_frames.py rounding."""
    return int(requested * fps + 0.5)


def decode_video_props(video: Path) -> dict[str, Any]:
    """Read container-reported video properties via OpenCV."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ProbeError("VIDEO_UNREADABLE",
                         f"Cannot open video: {video}", str(video))
    try:
        return {
            "cv2_version": cv2.__version__,
            "fps": capture.get(cv2.CAP_PROP_FPS),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()


def extract_dense_frames(video: Path, timestamps: tuple[float, ...],
                         fps: float) -> list[dict[str, Any]]:
    """Decode each requested frame once; record timing evidence."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ProbeError("VIDEO_UNREADABLE",
                         f"Cannot open video: {video}", str(video))
    decoded = []
    seen: set[int] = set()
    try:
        for requested in timestamps:
            index = frame_index_for(requested, fps)
            if index in seen:
                raise ProbeError(
                    "DUPLICATE_FRAME",
                    f"Schedule maps twice to frame {index}.",
                    str(index),
                )
            seen.add(index)
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ProbeError(
                    "DECODE_FAILED",
                    f"Cannot decode frame {index}.", str(index),
                )
            decoded.append({
                "requested_timestamp": requested,
                "frame_index": index,
                "actual_timestamp": round(index / fps, 6),
                "decoded_pos_msec": capture.get(cv2.CAP_PROP_POS_MSEC),
                "array": frame,
            })
    finally:
        capture.release()
    return decoded


def write_variants(decoded: list[dict[str, Any]],
                   images_dir: Path) -> dict[int, dict[str, Any]]:
    """Write controlled JPG (pipeline defaults) and PNG per frame."""
    import cv2

    images_dir.mkdir(parents=True, exist_ok=True)
    variants: dict[int, dict[str, Any]] = {}
    for item in decoded:
        index = item["frame_index"]
        jpg_name = f"probe_{index:08d}.jpg"
        png_name = f"probe_{index:08d}.png"
        jpg_path = images_dir / jpg_name
        png_path = images_dir / png_name
        # Pipeline parity: cv2.imwrite defaults (quality 95), no resize.
        if not cv2.imwrite(str(jpg_path), item["array"]):
            raise ProbeError("WRITE_FAILED",
                             f"Cannot write {jpg_name}.", jpg_name)
        # Direct from decoded frame; never converted from JPG.
        if not cv2.imwrite(str(png_path), item["array"]):
            raise ProbeError("WRITE_FAILED",
                             f"Cannot write {png_name}.", png_name)
        height, width = item["array"].shape[:2]
        variants[index] = {
            "requested_timestamp": item["requested_timestamp"],
            "actual_timestamp": item["actual_timestamp"],
            "decoded_pos_msec": item["decoded_pos_msec"],
            "width": width,
            "height": height,
            "jpg": {"path": str(jpg_path),
                    "bytes": jpg_path.stat().st_size,
                    "sha256": sha256_file(jpg_path),
                    "params": "cv2.imwrite defaults (quality 95)"},
            "png": {"path": str(png_path),
                    "bytes": png_path.stat().st_size,
                    "sha256": sha256_file(png_path),
                    "params": "cv2.imwrite defaults (compression 3)"},
        }
    return variants


def pixel_diff_stats(first: Path, second: Path) -> dict[str, Any]:
    """Decoded-pixel difference summary (no enhancement anywhere)."""
    import cv2
    import numpy as np

    left = cv2.imread(str(first), cv2.IMREAD_UNCHANGED)
    right = cv2.imread(str(second), cv2.IMREAD_UNCHANGED)
    if left is None or right is None:
        raise ProbeError("DECODE_FAILED",
                         f"Cannot decode {first.name} or {second.name}.")
    if left.shape != right.shape:
        return {"shapes_equal": False,
                "left_shape": list(left.shape),
                "right_shape": list(right.shape)}
    diff = cv2.absdiff(left, right)
    return {"shapes_equal": True,
            "max_abs_diff": int(np.max(diff)),
            "mean_abs_diff": round(float(np.mean(diff)), 6)}


def prepare(video: Path, frames_dir: Path, out_root: Path,
            hash_video: bool = False) -> dict[str, Any]:
    """Resolve assets, extract variants, write the input manifest."""
    manifest_path = out_root / "input_manifest.json"
    if manifest_path.exists():
        raise ProbeError("OUTPUT_EXISTS",
                         f"Refusing to overwrite: {manifest_path}.")
    if not video.is_file():
        raise ProbeError("VIDEO_UNREADABLE",
                         f"Video not found: {video}", str(video))
    props = decode_video_props(video)
    fps = props["fps"]
    if fps <= 0:
        raise ProbeError("VIDEO_UNREADABLE",
                         "Video FPS unavailable.", str(video))
    images_dir = out_root / "prepared_images"
    decoded = extract_dense_frames(video, DENSE_TIMESTAMPS, fps)
    variants = write_variants(decoded, images_dir)
    by_index = {item["frame_index"]: item for item in decoded}
    conditions: list[dict[str, Any]] = []
    for spec in CONDITION_SPECS:
        images = []
        for requested in spec["timestamps"]:
            index = frame_index_for(requested, fps)
            if spec["kind"] == "pipeline_jpg":
                path = frames_dir / f"frame_{index:08d}.jpg"
                if not path.is_file():
                    raise ProbeError(
                        "PIPELINE_FRAME_MISSING",
                        f"Pipeline frame missing: {path.name}.",
                        spec["id"],
                    )
                images.append({
                    "requested_timestamp": requested,
                    "frame_index": index,
                    "actual_timestamp": round(index / fps, 6),
                    "path": str(path),
                    "variant": "pipeline_jpg",
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
            else:
                if index not in variants:
                    raise ProbeError(
                        "VARIANT_MISSING",
                        f"No prepared variant for frame {index}.",
                        spec["id"],
                    )
                entry = variants[index]["jpg" if spec["kind"]
                                          == "fresh_jpg" else "png"]
                images.append({
                    "requested_timestamp": requested,
                    "frame_index": index,
                    "actual_timestamp": variants[index][
                        "actual_timestamp"],
                    "decoded_pos_msec": variants[index][
                        "decoded_pos_msec"],
                    "path": entry["path"],
                    "variant": spec["kind"],
                    "bytes": entry["bytes"],
                    "sha256": entry["sha256"],
                })
        conditions.append({
            "id": spec["id"],
            "purpose": spec["purpose"],
            "images": images,
        })
    redundancy = analyze_redundancy(conditions, variants, frames_dir,
                                    fps)
    manifest = {
        "manifest_version": "1.0.0",
        "target_timestamp_seconds": TARGET_TIMESTAMP,
        "source": {
            "video": str(video),
            "video_bytes": video.stat().st_size,
            "video_sha256": (
                sha256_file(video) if hash_video
                else "not_computed (use --hash-video)"),
            "frames_dir": str(frames_dir),
            **props,
        },
        "resolution_finding": redundancy["resolution_finding"],
        "conditions": [
            {**condition, "eligible": redundancy["eligible"][condition["id"]],
             "eligibility_notes": redundancy["notes"][condition["id"]]}
            for condition in conditions
        ],
        "redundancy": redundancy["report"],
        "shared_pipeline_indices": sorted(
            index for index in by_index
            if (frames_dir / f"frame_{index:08d}.jpg").is_file()
        ),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    return manifest


def analyze_redundancy(conditions: list[dict[str, Any]],
                       variants: dict[int, dict[str, Any]],
                       frames_dir: Path, fps: float) -> dict[str, Any]:
    """Eligibility, resolution, and equivalence analysis (no API)."""
    eligible = {spec["id"]: True for spec in CONDITION_SPECS}
    notes: dict[str, list[str]] = {spec["id"]: [] for spec in CONDITION_SPECS}
    report: dict[str, Any] = {}
    widths = {variants[index]["width"] for index in variants}
    heights = {variants[index]["height"] for index in variants}
    if widths == {1920} and heights == {1080}:
        resolution_finding = (
            "NATIVE_RESOLUTION_CONFIRMED: fresh variants are 1920x1080, "
            "matching the pipeline JPG dimensions and the reported "
            "native source resolution; no upscaling comparison created."
        )
    else:
        resolution_finding = (
            f"RESOLUTION_DIFFERS: fresh variants are {widths}x{heights}; "
            "resolution is a confounded variable, see report."
        )
    report["resolution_finding"] = resolution_finding
    pair_stats = {}
    for index, entry in variants.items():
        stats = pixel_diff_stats(
            Path(entry["jpg"]["path"]), Path(entry["png"]["path"]))
        pair_stats[str(index)] = stats
    report["fresh_jpg_vs_png_pixels"] = pair_stats
    all_identical = all(
        stats.get("max_abs_diff") == 0 for stats in pair_stats.values()
    )
    if all_identical:
        eligible["E"] = False
        notes["E"].append(
            "Redundant: every fresh PNG is pixel-identical to its JPG.")
        report["c_e_equivalence"] = (
            "ESTABLISHED at decoded-pixel level; E skipped.")
    else:
        report["c_e_equivalence"] = (
            "NOT established: JPEG compression differs from PNG; "
            "C-vs-E comparison retained.")
    by_id = {condition["id"]: condition for condition in conditions}
    target_index = frame_index_for(TARGET_TIMESTAMP, fps)
    target_matches = [
        image for image in by_id["A"]["images"]
        if image["frame_index"] == target_index
    ]
    target_jpg = (
        target_matches[0] if target_matches else by_id["A"]["images"][0]
    )
    target_index = target_jpg["frame_index"]
    if target_index in variants:
        stats = pixel_diff_stats(
            Path(target_jpg["path"]),
            Path(variants[target_index]["png"]["path"]))
        report["a_vs_d_pixels"] = stats
        fresh_jpg_sha = variants[target_index]["jpg"]["sha256"]
        if fresh_jpg_sha == target_jpg["sha256"]:
            report["extraction_reproducibility"] = (
                "Fresh JPG at target is byte-identical to the pipeline "
                "JPG: extraction reproduces; A-vs-D differs by encoding.")
        else:
            report["extraction_reproducibility"] = (
                "Fresh JPG at target differs from the pipeline JPG; "
                "A-vs-D changes encoding AND extraction instance.")
    signatures = {
        condition["id"]: [image["sha256"] for image in condition["images"]]
        for condition in conditions
    }
    seen: dict[tuple[str, ...], str] = {}
    for cid, sig in signatures.items():
        key = tuple(sig)
        if key in seen:
            eligible[cid] = False
            notes[cid].append(
                f"Redundant: image set byte-identical to {seen[key]}.")
        else:
            seen[key] = cid
    report["condition_signatures"] = {
        cid: {"images": len(sig), "sha256": sig}
        for cid, sig in signatures.items()
    }
    for cid in eligible:
        if eligible[cid] and not notes[cid]:
            notes[cid].append("Eligible: distinct inputs.")
    return {"eligible": eligible, "notes": notes, "report": report,
            "resolution_finding": resolution_finding}


def load_manifest(out_root: Path) -> dict[str, Any]:
    """Load the prepared input manifest."""
    path = out_root / "input_manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeError(
            "MANIFEST_INVALID", f"Unreadable manifest: {path}."
        ) from exc
    if not isinstance(payload, dict):
        raise ProbeError("MANIFEST_INVALID",
                         "Manifest root is not an object.")
    for key in ("target_timestamp_seconds", "conditions",
                "resolution_finding"):
        if key not in payload:
            raise ProbeError("MANIFEST_INVALID",
                             f"Manifest lacks key: {key}.")
    return payload


def build_prompt(target: float,
                 images: list[dict[str, Any]]) -> str:
    """Frozen prompt with timestamp labels only (no reference)."""
    lines = [
        f"Image {position}: timestamp {image['actual_timestamp']:.3f} s "
        f"(source frame {image['frame_index']})"
        for position, image in enumerate(images, start=1)
    ]
    return PROBE_PROMPT.format(target=target,
                               image_table="\n".join(lines))


def dry_run(out_root: Path) -> dict[str, Any]:
    """Validate prepared inputs and display the plan (no API calls)."""
    manifest = load_manifest(out_root)
    conditions = []
    for condition in manifest["conditions"]:
        images = []
        for image in condition["images"]:
            path = Path(image["path"])
            suffix = path.suffix.lower()
            images.append({
                "path": image["path"],
                "exists": path.is_file(),
                "timestamp": image["actual_timestamp"],
                "frame_index": image["frame_index"],
                "mime": MIME_BY_SUFFIX.get(suffix, "UNSUPPORTED"),
                "sha256": image["sha256"][:16],
            })
        conditions.append({
            "id": condition["id"],
            "purpose": condition["purpose"],
            "eligible": condition["eligible"],
            "notes": condition["eligibility_notes"],
            "image_count": len(images),
            "images": images,
        })
    plan = {
        "model": MODEL,
        "endpoint": meta_client.BASE_URL,
        "timeout_seconds": TIMEOUT_SECONDS,
        "token_budgets": list(TOKEN_BUDGETS),
        "max_attempts_per_condition": MAX_ATTEMPTS,
        "max_total_requests": MAX_TOTAL_REQUESTS,
        "request_store": REQUEST_STORE,
        "retry_policy": (
            "Retry only when status=incomplete with an explicit "
            "output-token-limit reason; escalate through budgets "
            "1024/2048/3072; never retry on content, agreement, "
            "transport, or rate-limit failures."
        ),
        "request_structure": (
            "Responses API: one input_text block plus one input_image "
            "block per image (base64 data URL with per-file MIME), "
            "store=false."
        ),
        "format_support": {
            "image/jpeg": "VERIFIED: 685 prior successful requests.",
            "image/png": (
                "UNVERIFIED in this repository: the production client "
                "sends image/jpeg only; this probe sends image/png "
                "for .png inputs and records acceptance per response."
            ),
        },
        "prompt_sha256": hashlib.sha256(
            build_prompt(manifest["target_timestamp_seconds"],
                         manifest["conditions"][0]["images"]
                         ).encode("utf-8")).hexdigest(),
        "conditions": conditions,
        "resolution_finding": manifest["resolution_finding"],
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    return plan


def _default_client_factory(api_key: str,
                            timeout_seconds: float) -> Any:
    from openai import OpenAI

    return OpenAI(base_url=meta_client.BASE_URL, api_key=api_key,
                  timeout=timeout_seconds, max_retries=0)


def _incomplete_reason(response: Any) -> str | None:
    """Extract incomplete_details.reason from object or dict shape."""
    details = getattr(response, "incomplete_details", None)
    if details is None:
        return None
    if isinstance(details, dict):
        reason = details.get("reason")
    else:
        reason = getattr(details, "reason", None)
    return str(reason) if reason is not None else None


def _dump_response(response: Any) -> Any:
    """Best-effort raw payload dump for diagnosis."""
    dumper = getattr(response, "model_dump", None)
    if callable(dumper):
        try:
            return dumper(mode="json")
        except TypeError:
            pass
        except Exception:
            pass
        if dumper is not None:
            try:
                return dumper()
            except Exception:
                pass
    return {"status": getattr(response, "status", None),
            "output_text": getattr(response, "output_text", None)}


def send_request(prompt_text: str, image_paths: list[Path],
                 api_key: str, max_output_tokens: int,
                 client_factory: Any = None) -> dict[str, Any]:
    """One Responses call with per-file MIME and store=false.

    Single attempt, no retries. The production client hardcodes
    image/jpeg and is frozen by scope, so the probe sends requests
    itself. client_factory is an injectable seam for offline tests.
    """
    if len(image_paths) > meta_client.MAX_IMAGES_PER_REQUEST:
        raise ProbeError(
            "TOO_MANY_IMAGES",
            f"{len(image_paths)} images exceed the verified maximum.",
        )
    chunks: list[dict[str, Any]] = [
        {"type": "input_text", "text": prompt_text}
    ]
    mimes = []
    for path in image_paths:
        mime = MIME_BY_SUFFIX.get(path.suffix.lower())
        if mime is None:
            raise ProbeError("UNSUPPORTED_FORMAT",
                             f"Unsupported image suffix: {path.name}.",
                             path.name)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ProbeError("IMAGE_MISSING",
                             f"Cannot read image bytes: {path.name}.",
                             path.name) from exc
        encoded = base64.b64encode(raw).decode("ascii")
        chunks.append({"type": "input_image",
                       "image_url": f"data:{mime};base64,{encoded}"})
        mimes.append(mime)
    factory = client_factory or _default_client_factory
    client = factory(api_key, TIMEOUT_SECONDS)
    probe_client = meta_client.MetaClient(model=MODEL)
    start = probe_client.clock()
    try:
        response = client.responses.create(
            model=MODEL,
            input=[{"role": "user", "content": chunks}],
            max_output_tokens=max_output_tokens,
            store=REQUEST_STORE,
        )
    except Exception as exc:  # Single attempt; errors are recorded.
        latency = probe_client.clock() - start
        mapped = probe_client._map_exception(exc)
        return {"ok": False, "code": mapped.code,
                "message": mapped.message,
                "status_code": mapped.status_code,
                "latency_seconds": latency, "mimes": mimes,
                "store": REQUEST_STORE,
                "max_output_tokens": max_output_tokens}
    latency = probe_client.clock() - start
    reason = _incomplete_reason(response)
    raw_payload = _dump_response(response)
    try:
        result = probe_client._normalize(response, latency)
    except meta_client.ProviderError as exc:
        return {"ok": False, "code": exc.code, "message": exc.message,
                "status_code": exc.status_code,
                "latency_seconds": latency, "mimes": mimes,
                "store": REQUEST_STORE,
                "max_output_tokens": max_output_tokens,
                "incomplete_reason": reason,
                "raw_response": raw_payload}
    return {"ok": True, "model": result.model, "status": result.status,
            "output_text": result.output_text,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "total_tokens": result.total_tokens,
            "reasoning_output_tokens": result.reasoning_output_tokens,
            "latency_seconds": result.latency_seconds,
            "request_id": result.request_id, "mimes": mimes,
            "store": REQUEST_STORE,
            "max_output_tokens": max_output_tokens,
            "incomplete_reason": reason,
            "raw_response": raw_payload}


def _is_token_limit_reason(reason: Any) -> bool:
    """True only for an explicit output-token-limit reason."""
    if reason is None:
        return False
    text = str(reason).lower()
    return "max_output_tokens" in text or (
        "output_token" in text
        and ("limit" in text or "max" in text or "exhaust" in text)
    )


def parse_prediction(text: Any) -> tuple[bool, dict[str, Any] | None]:
    """Parse structured output; no repair attempts."""
    if not isinstance(text, str):
        return False, None
    try:
        parsed = json.loads(text.strip())
    except (json.JSONDecodeError, AttributeError):
        return False, None
    if not isinstance(parsed, dict):
        return False, None
    return True, parsed


def decide_next(record: dict[str, Any], parse_ok: bool,
                attempts_made: int) -> dict[str, Any]:
    """Adaptive policy: retry, stop with terminal state, or abort.

    The human reference is never consulted here.
    """
    if not record.get("ok"):
        code = record.get("code")
        if code in ("AUTH_FAILED", "CREDENTIAL_MISSING"):
            return {"action": "abort", "terminal": "request_failed",
                    "reason": f"auth/config error ({code}); run stops"}
        return {"action": "stop", "terminal": "request_failed",
                "reason": f"transport failure ({code}); no escalation"}
    status = record.get("status")
    reason = record.get("incomplete_reason")
    budgets_left = attempts_made < MAX_ATTEMPTS
    if status == "incomplete":
        if _is_token_limit_reason(reason):
            if budgets_left:
                return {"action": "retry", "terminal": None,
                        "reason": "incomplete: token budget exhausted; "
                                  "escalating to next budget"}
            return {"action": "stop",
                    "terminal": "exhausted_token_budget",
                    "reason": "all token budgets exhausted"}
        return {"action": "stop", "terminal": "unsupported_response",
                "reason": f"incomplete with non-token reason {reason!r}; "
                          "cause not inferred"}
    if status != "completed":
        return {"action": "stop", "terminal": "unsupported_response",
                "reason": f"unrecognized status {status!r}"}
    text = record.get("output_text") or ""
    if not text.strip():
        if _is_token_limit_reason(reason):
            if budgets_left:
                return {"action": "retry", "terminal": None,
                        "reason": "empty output with explicit token-limit "
                                  "reason; escalating"}
            return {"action": "stop",
                    "terminal": "exhausted_token_budget",
                    "reason": "all token budgets exhausted"}
        return {"action": "stop", "terminal": "empty_output",
                "reason": "completed with empty output; no token-limit "
                          "reason present"}
    if parse_ok:
        return {"action": "stop", "terminal": "valid_prediction",
                "reason": "completed valid structured response accepted"}
    return {"action": "stop", "terminal": "invalid_json",
            "reason": "completed but output is not valid JSON; content "
                      "alone never triggers a retry"}


def evaluate_prediction(parsed: dict[str, Any],
                        record: dict[str, Any]) -> dict[str, Any]:
    """Post-inference comparison against the human reference."""
    from bakery_intelligence.verification.numerics import numbers_match

    display = parsed.get("display_text")
    numeric = parsed.get("numeric_value")
    numeric_match, numeric_note = numbers_match(numeric,
                                                REFERENCE_NUMERIC)
    return {
        "status": "evaluated",
        "display_text": display,
        "display_exact_match": display == REFERENCE_DISPLAY,
        "numeric_value": numeric,
        "numeric_match": numeric_match,
        "numeric_match_note": numeric_note,
        "abstained": (parsed.get("readable") is False) or display is None,
        "readable": parsed.get("readable"),
        "confidence": parsed.get("confidence"),
        "evidence_timestamps": parsed.get("evidence_timestamps"),
        "direct_visual_evidence": parsed.get("direct_visual_evidence"),
        "temporal_inference_used": parsed.get(
            "temporal_inference_used"),
        "uncertainty_notes": parsed.get("uncertainty_notes"),
        "tokens": {key: record.get(key) for key in
                   ("input_tokens", "output_tokens", "total_tokens",
                    "reasoning_output_tokens")},
        "cost_usd_estimate": estimate_cost(
            record.get("input_tokens"), record.get("output_tokens")),
    }


def estimate_cost(input_tokens: Any, output_tokens: Any) -> float | None:
    """Cost from reported usage and reported standard-tier rates."""
    if not isinstance(input_tokens, (int, float)) or not isinstance(
            output_tokens, (int, float)):
        return None
    if isinstance(input_tokens, bool) or isinstance(output_tokens, bool):
        return None
    return round(input_tokens / 1e6 * INPUT_USD_PER_1M
                 + output_tokens / 1e6 * OUTPUT_USD_PER_1M, 6)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )


def run_conditions(out_root: Path, run_id: str,
                   client_factory: Any = None) -> dict[str, Any]:
    """Rerun every condition from the first budget (paid, max 15)."""
    try:
        api_key = meta_client.resolve_api_key()
    except meta_client.ProviderError as exc:
        raise ProbeError("CREDENTIAL_MISSING", exc.message) from exc
    manifest = load_manifest(out_root)
    run_dir = out_root / "runs" / run_id
    if run_dir.exists():
        raise ProbeError("OUTPUT_EXISTS",
                         f"Refusing to overwrite run: {run_dir}.")
    run_dir.mkdir(parents=True)
    attempts_dir = run_dir / "attempts"
    attempts_dir.mkdir()
    results: dict[str, Any] = {}
    attempt_log: list[dict[str, Any]] = []
    request_count = 0
    aborted = False
    abort_reason = ""
    started = time.perf_counter()
    for condition in manifest["conditions"]:
        cid = condition["id"]
        if aborted:
            results[cid] = {"status": "not_executed",
                            "reason": abort_reason}
            continue
        if not condition.get("eligible", False):
            results[cid] = {
                "status": "not_executed",
                "reason": "; ".join(condition.get(
                    "eligibility_notes", ["ineligible"])),
            }
            continue
        images = condition["images"]
        prompt_text = build_prompt(
            manifest["target_timestamp_seconds"], images)
        prompt_sha = hashlib.sha256(
            prompt_text.encode("utf-8")).hexdigest()
        image_paths = [Path(image["path"]) for image in images]
        try:
            runtime_hashes = [sha256_file(path) for path in image_paths]
        except OSError as exc:
            results[cid] = {"status": "not_executed",
                            "reason": f"input unreadable: {exc}"}
            continue
        manifest_hashes = [image["sha256"] for image in images]
        if runtime_hashes != manifest_hashes:
            results[cid] = {
                "status": "not_executed",
                "reason": "input hash mismatch vs manifest; inputs changed",
            }
            continue
        terminal: str | None = None
        selected_attempt: int | None = None
        selected_evaluation: dict[str, Any] | None = None
        attempt_files: list[str] = []
        made = 0
        for number, budget in enumerate(TOKEN_BUDGETS, start=1):
            try:
                record = send_request(
                    prompt_text, image_paths, api_key, budget,
                    client_factory,
                )
            except ProbeError as exc:
                results[cid] = {"status": "not_executed",
                                "reason": f"local failure: [{exc.code}] "
                                          f"{exc.message}"}
                terminal = "not_executed"
                break
            request_count += 1
            if request_count > MAX_TOTAL_REQUESTS:
                raise ProbeError(
                    "BUDGET_EXCEEDED",
                    f"Refusing request {request_count} beyond "
                    f"authorized {MAX_TOTAL_REQUESTS}.",
                )
            made += 1
            parse_ok, parsed = parse_prediction(record.get("output_text"))
            decision = decide_next(record, parse_ok, made)
            action = decision["action"]
            attempt = {
                "condition_id": cid,
                "attempt_number": made,
                "max_output_tokens": budget,
                "model": MODEL,
                "prompt_hash": prompt_sha,
                "prompt_text": prompt_text,
                "input_image_hashes": runtime_hashes,
                "store": record.get("store"),
                "request_id": record.get("request_id"),
                "status": record.get("status"),
                "incomplete_reason": record.get("incomplete_reason"),
                "raw_response_text": record.get("output_text"),
                "normalized_response_text": (
                    record.get("output_text") or "").strip() or None,
                "json_parse_ok": parse_ok,
                "parsed_prediction": parsed,
                "input_tokens": record.get("input_tokens"),
                "output_tokens": record.get("output_tokens"),
                "reasoning_output_tokens": record.get(
                    "reasoning_output_tokens"),
                "total_tokens": record.get("total_tokens"),
                "latency_seconds": record.get("latency_seconds"),
                "error": (
                    None if record.get("ok") else
                    {"type": record.get("code"),
                     "message": record.get("message")}),
                "retry_triggered": action == "retry",
                "retry_reason": decision["reason"],
                "raw_response": record.get("raw_response"),
            }
            attempt_path = attempts_dir / f"{cid}_attempt_{made:02d}.json"
            _write_json(attempt_path, attempt)
            attempt_files.append(str(attempt_path))
            attempt_log.append({**attempt, "condition_id": cid})
            if action == "abort":
                terminal = "request_failed"
                aborted = True
                abort_reason = decision["reason"]
                break
            if action == "retry":
                continue
            terminal = decision["terminal"]
            if terminal == "valid_prediction":
                selected_attempt = made
                assert parsed is not None
                selected_evaluation = evaluate_prediction(parsed, record)
            break
        if terminal is None:
            terminal = "unsupported_response"
        if terminal == "not_executed":
            continue  # result already recorded above
        results[cid] = {
            "status": terminal,
            "attempts_made": made,
            "attempt_files": attempt_files,
            "selected_attempt": selected_attempt,
            "evaluation": (
                selected_evaluation
                if selected_evaluation is not None
                else {"status": terminal}),
        }
    elapsed = time.perf_counter() - started
    recovery = summarize_recovery(results, attempt_log, elapsed)
    report = {
        "report_version": "1.1.0",
        "run_id": run_id,
        "model": MODEL,
        "token_budgets": list(TOKEN_BUDGETS),
        "request_store": REQUEST_STORE,
        "max_total_requests": MAX_TOTAL_REQUESTS,
        "reference": {
            "display": REFERENCE_DISPLAY,
            "numeric": REFERENCE_NUMERIC,
            "basis": ("Independent human reference, used exclusively "
                      "for post-inference evaluation; never consulted "
                      "by the retry policy."),
        },
        "pricing_basis": {
            "input_usd_per_1m": INPUT_USD_PER_1M,
            "output_usd_per_1m": OUTPUT_USD_PER_1M,
            "status": "REPORTED_NOT_DIRECTLY_VERIFIED",
        },
        "conditions": results,
        "comparisons": [
            {**item, "multi_variable": len(item["changed"]) > 1}
            for item in COMPARISONS
        ],
        "recovery": recovery,
        "caveats": {
            "verified": [
                "Raw attempt records per condition and budget.",
                "Input manifest hashes, dimensions, timestamps.",
            ],
            "interpretation": [
                "Display/numeric agreement computed post hoc.",
            ],
            "hypothesis": [
                "Any fidelity/context effect suggested here needs "
                "replication across timestamps before generalizing.",
            ],
            "limitation": [
                "Single target timestamp; no general OCR accuracy claim.",
                "Comparisons with multiple changed variables cannot "
                "attribute effects to one factor.",
            ],
        },
    }
    _write_json(run_dir / "comparison_report.json", report)
    summary = render_summary(report)
    (run_dir / "summary.txt").write_text(
        summary, encoding="utf-8", newline="\n",
    )
    print(summary)
    return report


def summarize_recovery(results: dict[str, Any],
                       attempt_log: list[dict[str, Any]],
                       elapsed_seconds: float) -> dict[str, Any]:
    """Aggregate cost and recovery metrics across all attempts."""
    per_condition = {}
    retries = 0
    recovered = 0
    exhausted = 0
    for cid, result in results.items():
        made = result.get("attempts_made", 0)
        per_condition[cid] = made
        if made > 1:
            retries += made - 1
        if result.get("status") == "valid_prediction" and (
                (result.get("selected_attempt") or 1) > 1):
            recovered += 1
        if result.get("status") == "exhausted_token_budget":
            exhausted += 1
    input_tokens = sum(
        attempt.get("input_tokens") or 0 for attempt in attempt_log
    )
    output_tokens = sum(
        attempt.get("output_tokens") or 0 for attempt in attempt_log
    )
    total_tokens = sum(
        attempt.get("total_tokens") or 0 for attempt in attempt_log
    )
    selected = {
        (cid, result.get("selected_attempt"))
        for cid, result in results.items()
        if result.get("status") == "valid_prediction"
    }
    cost_total = 0.0
    cost_wasted = 0.0
    cost_unknown = 0
    for attempt in attempt_log:
        cost = estimate_cost(attempt.get("input_tokens"),
                             attempt.get("output_tokens"))
        if cost is None:
            cost_unknown += 1
            continue
        cost_total += cost
        key = (attempt.get("condition_id"),
               attempt.get("attempt_number"))
        if key not in selected:
            cost_wasted += cost
    return {
        "total_requests": len(attempt_log),
        "requests_per_condition": per_condition,
        "adaptive_retries": retries,
        "conditions_recovered": recovered,
        "conditions_exhausted": exhausted,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd_estimate": round(cost_total, 6),
        "cost_usd_wasted_estimate": round(cost_wasted, 6),
        "cost_unknown_attempts": cost_unknown,
        "latency_per_attempt": [
            {"condition_id": attempt.get("condition_id"),
             "attempt_number": attempt.get("attempt_number"),
             "latency_seconds": attempt.get("latency_seconds")}
            for attempt in attempt_log
        ],
        "elapsed_inference_seconds": round(elapsed_seconds, 3),
    }


def render_summary(report: dict[str, Any]) -> str:
    """Concise human-readable outcome summary."""
    lines = [
        f"Temporal/spatial probe run {report['run_id']} "
        f"(model {report['model']}, store={report['request_store']}, "
        f"budgets={report['token_budgets']})",
        f"Reference (post-hoc only): {REFERENCE_DISPLAY}",
        "",
    ]
    for cid, result in sorted(report["conditions"].items()):
        status = result.get("status")
        if status == "not_executed":
            lines.append(f"[{cid}] NOT_EXECUTED: {result.get('reason')}")
            continue
        evaluation = result.get("evaluation", {})
        if status == "valid_prediction":
            lines.append(
                f"[{cid}] valid_prediction (attempt "
                f"{result.get('selected_attempt')}): "
                f"display={evaluation.get('display_text')!r} "
                f"exact={evaluation.get('display_exact_match')} "
                f"numeric={evaluation.get('numeric_value')!r} "
                f"match={evaluation.get('numeric_match')} "
                f"abstained={evaluation.get('abstained')} "
                f"confidence={evaluation.get('confidence')}")
        else:
            lines.append(
                f"[{cid}] {status} after "
                f"{result.get('attempts_made')} attempt(s)")
    recovery = report["recovery"]
    lines += [
        "",
        f"Requests: {recovery['total_requests']} total, "
        f"{recovery['adaptive_retries']} retries, "
        f"{recovery['conditions_recovered']} recovered, "
        f"{recovery['conditions_exhausted']} exhausted; "
        f"tokens in/out/total={recovery['input_tokens']}/"
        f"{recovery['output_tokens']}/{recovery['total_tokens']}; "
        f"cost_usd~{recovery['cost_usd_estimate']} "
        f"(wasted~{recovery['cost_usd_wasted_estimate']}); "
        f"elapsed_s={recovery['elapsed_inference_seconds']}",
        "",
        "Comparisons (changed variables):",
    ]
    for item in report["comparisons"]:
        lines.append(
            f"  {item['pair']}: {', '.join(item['changed'])} "
            f"(multi_variable={item['multi_variable']})")
    lines += ["",
              "Single timestamp only: no general OCR accuracy claim."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI with prepare / dry-run / run modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="extract variants, no API calls")
    prep.add_argument("--video", type=Path, default=Path(
        "data/raw/epic_kitchens/P04_104.MP4"))
    prep.add_argument("--frames-dir", type=Path, default=Path(
        "data/processed/experiment_003/global_1fps_rerun"))
    prep.add_argument("--out", type=Path, default=Path(
        "data/processed/experiment_007/temporal_spatial_probe"))
    prep.add_argument("--hash-video", action="store_true")
    dry = sub.add_parser("dry-run", help="validate plan, no API calls")
    dry.add_argument("--out", type=Path, default=Path(
        "data/processed/experiment_007/temporal_spatial_probe"))
    run = sub.add_parser("run", help="paid inference, user-invoked only")
    run.add_argument("--out", type=Path, default=Path(
        "data/processed/experiment_007/temporal_spatial_probe"))
    run.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            manifest = prepare(args.video, args.frames_dir, args.out,
                               args.hash_video)
            print(json.dumps(
                {"conditions": [
                    {"id": item["id"],
                     "eligible": item["eligible"]}
                    for item in manifest["conditions"]]},
                indent=2, sort_keys=True))
            return 0
        if args.command == "dry-run":
            dry_run(args.out)
            return 0
        stamp = datetime.datetime.now(datetime.timezone.utc)
        run_id = args.run_id or stamp.strftime("run_%Y%m%dT%H%M%SZ")
        run_conditions(args.out, run_id)
        return 0
    except ProbeError as exc:
        print(f"PROBE FAILED [{exc.code}]: {exc.message}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
