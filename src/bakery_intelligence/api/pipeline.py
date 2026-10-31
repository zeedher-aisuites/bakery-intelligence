"""Offline video pipeline orchestration (M2-06C.4).

Workflow ``video_pipeline_offline`` coordinates staged video
intake, frame extraction, first-pass inference with a
deterministic fake client, evidence transformation, and
offline reconstruction. Every stage runs locally; no model is
queried and no network is touched. Fake provenance is labeled
in the summary, the first-pass report, and the workflow name.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bakery_intelligence.api.jobs import WORKFLOW_VIDEO_PIPELINE_OFFLINE
from bakery_intelligence.api.offline import (
    OFFLINE_SUMMARY_ARTIFACT,
    parse_offline_bundle,
    run_bundle_reconstruction,
)
from bakery_intelligence.inference.fake import (
    PROVIDER_ID as FAKE_PROVIDER_ID,
    DeterministicFakeClient,
)
from bakery_intelligence.inference.firstpass import (
    FirstPassConfig,
    FirstPassRunner,
)
from bakery_intelligence.video.extract import (
    ExtractionConfig,
    extract_frames,
)
from bakery_intelligence.video.intake import (
    VideoError,
    sanitize_stem,
    stage_video,
)

if TYPE_CHECKING:
    from bakery_intelligence.api.jobs import JobRecord
    from bakery_intelligence.api.storage import StorageBackend
    from bakery_intelligence.inference.firstpass import InferenceClient

PIPELINE_EVIDENCE_VERSION = "pipeline-evidence/1"
PIPELINE_SUMMARY_ARTIFACT = "pipeline_summary.json"
FIRSTPASS_REPORT_ARTIFACT = "firstpass_report.json"

PIPELINE_PROMPT = "PIPELINE TEST PROMPT (fake client ignores content)."
PIPELINE_PASS_ID = "pipeline_firstpass"
PIPELINE_SAMPLE_FPS = 1.0
PIPELINE_BATCH_SIZE = 5
PIPELINE_MAX_VIDEO_BYTES = 16 * 1024 * 1024
PIPELINE_MAX_FRAMES = 1000

TEMPORAL_STUB_REASON = "pipeline_synthetic_stub"


def build_pipeline_temporal(event_id: str,
                            frame_indices: list[int],
                            start: float, end: float) -> dict[str, Any]:
    """Build the window-spanning temporal placeholder (v1).

    C.4 performs no temporal analysis, so the single candidate
    event honestly marks itself as pipeline scaffolding awaiting
    real analysis. Shape follows temporal artifact 1.1.0 so the
    unmodified reconstruction loaders consume it.
    """
    return {
        "artifact_version": "1.1.0",
        "pipeline_evidence": PIPELINE_EVIDENCE_VERSION,
        "events": [{
            "event_id": event_id,
            "candidate_type": "scale_visibility",
            "start_timestamp_seconds": start,
            "end_timestamp_seconds": end,
            "verification_status": "candidate",
            "frame_indices": sorted(frame_indices),
            "review_reasons": [TEMPORAL_STUB_REASON],
            "uncertainty_notes": [
                "No temporal analysis performed; window-spanning "
                "placeholder pending real analysis.",
            ],
        }],
    }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def client_model_id(client: InferenceClient) -> str:
    """Return the client's model identity, fake-labeled by default."""
    model_id = getattr(client, "model_id", None)
    if isinstance(model_id, str) and model_id:
        return model_id
    return "unknown-offline-model"


def run_pipeline_offline(
    *,
    video_bytes: bytes,
    video_name: str,
    run_id: str,
    source_object_id: str,
    source_sha256: str,
    work_dir: Path,
    fake: InferenceClient | None = None,
    sample_fps: float = PIPELINE_SAMPLE_FPS,
    max_frames: int = PIPELINE_MAX_FRAMES,
) -> dict[str, bytes]:
    """Run all pipeline stages; return id -> bytes artifacts.

    Pure orchestration over caller-provided bytes and directory:
    stage, extract, infer (fake), transform, reconstruct. Raises
    VideoError for intake/extraction failures and ValueError when
    inference yields no usable observations.
    """
    if len(video_bytes) > PIPELINE_MAX_VIDEO_BYTES:
        raise VideoError(
            "VIDEO_TOO_LARGE",
            f"Pipeline video is {len(video_bytes)} bytes; limit is "
            f"{PIPELINE_MAX_VIDEO_BYTES}.",
            video_name,
        )
    client: InferenceClient = fake or DeterministicFakeClient()
    input_dir = work_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / f"upload_{sanitize_stem(video_name)}"
    input_path.write_bytes(video_bytes)
    staged = stage_video(input_path, work_dir / "staging")
    extraction = extract_frames(ExtractionConfig(
        video_path=staged.video_path,
        out_dir=work_dir / "frames",
        sample_fps=sample_fps, max_frames=max_frames,
    ))
    firstpass_dir = work_dir / "firstpass"
    firstpass_report = FirstPassRunner(FirstPassConfig(
        frames=extraction.frames, source=extraction.source,
        out_dir=firstpass_dir, model=client_model_id(client),
        prompt_text=PIPELINE_PROMPT, pass_id=PIPELINE_PASS_ID,
        provider=FAKE_PROVIDER_ID, batch_size=PIPELINE_BATCH_SIZE,
        client_factory=lambda: client,
    )).run()
    observations: list[dict[str, Any]] = []
    for batch_path in sorted(
            (firstpass_dir / "observations").glob("*.json")):
        observations.extend(json.loads(batch_path.read_text(
            encoding="utf-8")))
    if not observations:
        raise ValueError(
            f"Pipeline produced no observations "
            f"({firstpass_report['frames_failed']} failed of "
            f"{firstpass_report['frames_total']}); failing instead "
            f"of reconstructing from nothing.")
    indices = [frame.frame_index for frame in extraction.frames]
    stamps = [frame.timestamp_seconds for frame in extraction.frames]
    event_id = f"pipeline_window_{stamps[0]:g}_{stamps[-1]:g}"
    video_id = f"upload_{staged.metadata.sha256[:12]}"
    temporal = build_pipeline_temporal(
        event_id, indices, stamps[0], stamps[-1])
    bundle_raw = (json.dumps({
        "bundle_format": "bakery-offline-bundle/1",
        "video_id": video_id,
        "event_id": event_id,
        "source_fps": staged.metadata.fps,
        "observations": observations,
        "temporal": temporal,
        "execution_reports": [],
        "units": None,
    }, indent=2, sort_keys=True) + "\n").encode("utf-8")
    bundle = parse_offline_bundle(bundle_raw)
    summary, artifacts = run_bundle_reconstruction(bundle, run_id)
    firstpass_report_raw = (
        firstpass_dir / "reports" / "run_report.json").read_bytes()
    manifest_sha = _sha256_bytes(
        extraction.manifest_path.read_bytes())
    payload = {
        "workflow": WORKFLOW_VIDEO_PIPELINE_OFFLINE,
        "run_id": run_id,
        "source_object_id": source_object_id,
        "source_sha256": source_sha256,
        "evidence_version": PIPELINE_EVIDENCE_VERSION,
        "inference_performed": False,
        "network_accessed": False,
        "paid_services_used": False,
        "synthetic_inference": True,
        "fake_model_id": client_model_id(client),
        "fake_provider_id": FAKE_PROVIDER_ID,
        "video": staged.metadata.to_dict(),
        "extraction": {
            "sample_fps": sample_fps,
            "frame_indices": indices,
            "manifest_sha256": manifest_sha,
        },
        "firstpass": {
            "model": client_model_id(client),
            "provider": FAKE_PROVIDER_ID,
            "pass_id": PIPELINE_PASS_ID,
            "frames_total": firstpass_report["frames_total"],
            "frames_observed": firstpass_report["frames_observed"],
            "frames_failed": firstpass_report["frames_failed"],
            "terminal_states": firstpass_report["lifecycle"][
                "terminal_states"],
        },
        "bundle_sha256": _sha256_bytes(bundle_raw),
        "draft_id": summary["draft_id"],
        "measurements": summary["measurements"],
        "actions": summary["actions"],
        "unresolved": summary["unresolved"],
        "issues": summary["issues"],
        "artifacts": sorted([*artifacts, PIPELINE_SUMMARY_ARTIFACT,
                             FIRSTPASS_REPORT_ARTIFACT]),
        "note": ("Offline pipeline run over a staged upload with "
                 "deterministic fake inference. No model was "
                 "queried; all readings are synthetic test vectors "
                 "and the draft requires human review before "
                 "operational use."),
    }
    artifacts[PIPELINE_SUMMARY_ARTIFACT] = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    artifacts[FIRSTPASS_REPORT_ARTIFACT] = firstpass_report_raw
    assert OFFLINE_SUMMARY_ARTIFACT not in artifacts
    return artifacts


def execute_video_pipeline_offline(
        storage: StorageBackend,
        record: JobRecord) -> dict[str, bytes]:
    """Worker entry point: pipeline a small uploaded video offline."""
    if record.source_object_id is None:
        raise ValueError("video_pipeline_offline requires a source.")
    if not storage.source_exists(record.source_object_id):
        raise ValueError(
            f"Unknown source: {record.source_object_id!r}.")
    with storage.open_source(record.source_object_id) as handle:
        raw = handle.read()
    with tempfile.TemporaryDirectory(
            prefix="video_pipeline_") as staging:
        return run_pipeline_offline(
            video_bytes=raw,
            video_name=record.source_object_id,
            run_id=record.run_id,
            source_object_id=record.source_object_id,
            source_sha256=hashlib.sha256(raw).hexdigest(),
            work_dir=Path(staging),
        )
