"""Offline video pipeline orchestration (M2-06C.4).

Workflow ``video_pipeline_offline`` coordinates staged video
intake, frame extraction, first-pass inference with a
deterministic fake client, evidence transformation, and
offline reconstruction. Every stage runs locally; no model is
queried and no network is touched. Fake provenance is labeled
in the summary, the first-pass report, and the workflow name.

Durable recovery (S2): the worker runs the pipeline inside a
run-scoped durable work directory (never a temp directory),
with stage transitions recorded in an atomic state file.
A restarted worker verifies the state fingerprint and resumes
instead of restarting: extraction outputs are verified, the
first-pass runner resumes from its own checkpoints (skipping
completed requests), and reconstruction re-runs deterministically.

Billable seam (S4-B): ``run_pipeline_offline`` accepts an
explicit caller-supplied billable-capable client, executed
under S3 single-submit invariants with provider, model, and
execution policy fingerprinted. No registered workflow
supplies the seam; S4-B exercises it with test-double SDKs
only, so no provider is contacted.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bakery_intelligence.api.authorization import (
    GrantError,
    parse_grant,
    redact_key,
    resolve_api_key,
)
from bakery_intelligence.api.jobs import (
    WORKFLOW_VIDEO_PIPELINE_META,
    WORKFLOW_VIDEO_PIPELINE_OFFLINE,
    utc_now_iso,
)
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
    DEFAULT_BUDGETS,
    DEFAULT_MAX_ATTEMPTS,
    UNCERTAIN_REFUSE,
    UNCERTAIN_RERUN,
    FirstPassConfig,
    FirstPassRunner,
    FrameRef,
    SourceRef,
    prompt_hash,
)
from bakery_intelligence.inference.meta_spark import (
    MetaSparkClient,
    billable_runner_config,
)
from bakery_intelligence.video.extract import (
    MANIFEST_NAME,
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

# Provider label for the caller-supplied billable path (S4-B).
# Only the deterministic fake runs through registered
# workflows; the billable seam takes effect solely when a
# caller explicitly injects a client instance.
META_PROVIDER_ID = "meta"

TEMPORAL_STUB_REASON = "pipeline_synthetic_stub"

PIPELINE_STATE_FILENAME = "pipeline_state.json"
PIPELINE_STATE_VERSION = "pipeline-workdir/1"

STAGE_EXTRACTED = "extracted"
STAGE_FIRSTPASS_RUNNING = "firstpass_running"
STAGE_FIRSTPASS_DONE = "firstpass_done"

# Stage directories wiped for a clean start when no durable
# state exists. The firstpass directory is never wiped: billed
# inference progress without a state file fails closed instead
# of being silently discarded.
STAGE_DIRNAMES = ("input", "staging", "frames")

VIDEO_METADATA_KEYS = (
    "file_name", "sha256", "bytes", "fps", "frame_count",
    "duration_seconds", "width", "height",
)


class PipelineResumeError(ValueError):
    """Durable pipeline state is missing, corrupt, or mismatched."""


def _validate_billable_bounds(
        token_budget: int, max_requests: int | None,
        cost_ceiling_usd: float | None,
        rates_usd_per_m: tuple[float, float] | None,
        strict_cost: bool, api_key: str) -> None:
    """Reject malformed billing bounds before any state write.

    The key travels opaquely and is never inspected beyond
    its type: emptiness is legal for test-double clients,
    while the authorized executor always supplies a key it
    resolved from the worker environment.
    """
    if not isinstance(api_key, str):
        raise ValueError(
            "billable_api_key must be a str, "
            f"got {type(api_key).__name__}.")
    if (isinstance(token_budget, bool)
            or not isinstance(token_budget, int)
            or token_budget <= 0):
        raise ValueError(
            "billable_token_budget must be a positive int, "
            f"got {token_budget!r}.")
    if (max_requests is not None and (
            isinstance(max_requests, bool)
            or not isinstance(max_requests, int)
            or max_requests < 0)):
        raise ValueError(
            "billable_max_requests must be a non-negative int "
            f"or None, got {max_requests!r}.")
    if (cost_ceiling_usd is not None and (
            isinstance(cost_ceiling_usd, bool)
            or not isinstance(cost_ceiling_usd, (int, float))
            or not math.isfinite(cost_ceiling_usd)
            or cost_ceiling_usd < 0)):
        raise ValueError(
            "billable_cost_ceiling_usd must be a non-negative "
            f"finite number or None, got {cost_ceiling_usd!r}.")
    if rates_usd_per_m is not None and (
            not isinstance(rates_usd_per_m, (list, tuple))
            or len(rates_usd_per_m) != 2
            or any(isinstance(rate, bool)
                   or not isinstance(rate, (int, float))
                   or not math.isfinite(rate) or rate < 0
                   for rate in rates_usd_per_m)):
        raise ValueError(
            "billable_rates_usd_per_m must be None or a pair "
            "of non-negative finite numbers, "
            f"got {rates_usd_per_m!r}.")
    if not isinstance(strict_cost, bool):
        raise ValueError(
            "billable_strict_cost must be a bool, "
            f"got {strict_cost!r}.")


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


def _write_state_atomic(work_dir: Path,
                        state: dict[str, Any]) -> None:
    """Persist pipeline state via temp file plus atomic replace."""
    payload = {**state, "updated_at_utc": utc_now_iso()}
    path = work_dir / PIPELINE_STATE_FILENAME
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    os.replace(tmp, path)


def _read_state(work_dir: Path) -> dict[str, Any] | None:
    """Load durable state, or None when this work dir is fresh.

    Unreadable, malformed, or version-unknown states fail
    closed: resume must never guess.
    """
    path = work_dir / PIPELINE_STATE_FILENAME
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PipelineResumeError(
            f"Pipeline state is unreadable: {path.name}."
        ) from exc
    if (not isinstance(state, dict)
            or state.get("state_version")
            != PIPELINE_STATE_VERSION
            or state.get("stage") not in (
                STAGE_EXTRACTED, STAGE_FIRSTPASS_RUNNING,
                STAGE_FIRSTPASS_DONE)):
        raise PipelineResumeError(
            "Pipeline state is malformed or from an unknown "
            f"version: {path.name}.")
    return state


def _reset_stage_dirs(work_dir: Path) -> None:
    """Wipe stage inputs for a clean start (containment-guarded).

    Only the fixed stage names directly under the work dir are
    removed; anything else refuses instead of deleting. The
    firstpass directory is never wiped here.
    """
    root = work_dir.resolve()
    for name in STAGE_DIRNAMES:
        target = (work_dir / name).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise PipelineResumeError(
                f"Refusing to reset outside work dir: {name}."
            ) from exc
        if target.is_dir() or target.is_symlink():
            if target.is_symlink() or target.is_file():
                raise PipelineResumeError(
                    f"Refusing to reset non-directory: {name}.")
            shutil.rmtree(target)


def _diff_json(expected: Any, actual: Any, path: str) -> list[str]:
    """List JSON paths where actual differs from expected."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        diffs = []
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                diffs.append(f"{path}.{key}: unexpected")
            elif key not in actual:
                diffs.append(f"{path}.{key}: missing")
            else:
                diffs.extend(_diff_json(
                    expected[key], actual[key], f"{path}.{key}"))
        return diffs
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: length {len(actual)} != "
                    f"{len(expected)}"]
        diffs = []
        for position, (want, got) in enumerate(
                zip(expected, actual)):
            diffs.extend(_diff_json(
                want, got, f"{path}[{position}]"))
        return diffs
    if expected != actual or (
            isinstance(expected, bool)
            != isinstance(actual, bool)):
        return [f"{path}: {actual!r} != {expected!r}"]
    return []


def _staged_names(video_name: str, source_sha256: str
                  ) -> tuple[str, str]:
    """Recompute the deterministic staged video/sidecar names.

    Mirrors ``intake.stage_video``: the input copy keeps the
    client name (sanitized), and staging prefixes the content
    hash. Returns (video_filename, sidecar_filename).
    """
    input_name = f"upload_{sanitize_stem(video_name)}"
    suffix = Path(input_name).suffix.lower()
    stem = sanitize_stem(Path(input_name).stem)
    final = f"video_{source_sha256[:16]}_{stem}{suffix}"
    return final, f"{Path(final).stem}.json"


def _verify_extraction_outputs(work_dir: Path,
                               state: dict[str, Any]) -> bytes:
    """Verify staged and extracted bytes; return manifest bytes.

    Every frame file is hash-checked against the state record,
    so a resumed run cannot silently analyze substituted
    frames. Anything unexpected fails closed.
    """
    run_id = state.get("run_id", "?")
    extraction = state.get("extraction")
    if not isinstance(extraction, dict):
        raise PipelineResumeError(
            f"Run {run_id}: state lacks extraction details; "
            f"refusing resume.")
    frames_dir = work_dir / "frames"
    manifest_path = frames_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise PipelineResumeError(
            f"Run {run_id}: extraction manifest is missing; "
            f"refusing resume.")
    manifest_bytes = manifest_path.read_bytes()
    if _sha256_bytes(manifest_bytes) != extraction.get(
            "manifest_sha256"):
        raise PipelineResumeError(
            f"Run {run_id}: extraction manifest changed; "
            f"refusing resume.")
    indices = extraction.get("frame_indices")
    frame_hashes = extraction.get("frame_hashes")
    if (not isinstance(indices, list)
            or not isinstance(frame_hashes, dict)):
        raise PipelineResumeError(
            f"Run {run_id}: extraction record is malformed; "
            f"refusing resume.")
    rows = _read_manifest_rows(manifest_bytes, run_id)
    if [int(row["frame_index"]) for row in rows] != indices:
        raise PipelineResumeError(
            f"Run {run_id}: extraction manifest disagrees with "
            f"state; refusing resume.")
    for index in indices:
        if not isinstance(index, int) or isinstance(index, bool):
            raise PipelineResumeError(
                f"Run {run_id}: extraction record is "
                f"malformed; refusing resume.")
        frame_path = frames_dir / f"frame_{index:08d}.jpg"
        if not frame_path.is_file():
            raise PipelineResumeError(
                f"Run {run_id}: extracted frame {index} is "
                f"missing; refusing resume.")
        if _sha256_bytes(frame_path.read_bytes()) != (
                frame_hashes.get(str(index))):
            raise PipelineResumeError(
                f"Run {run_id}: extracted frame {index} "
                f"changed; refusing resume.")
    sidecar_path = work_dir / "staging" / str(
        extraction.get("sidecar_name"))
    if (not sidecar_path.is_file() or _sha256_bytes(
            sidecar_path.read_bytes())
            != extraction.get("sidecar_sha256")):
        raise PipelineResumeError(
            f"Run {run_id}: staged video sidecar is missing "
            f"or changed; refusing resume.")
    try:
        sidecar = json.loads(
            sidecar_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise PipelineResumeError(
            f"Run {run_id}: staged video sidecar is "
            f"unreadable; refusing resume.") from exc
    video = state.get("video")
    if not isinstance(sidecar, dict) or not isinstance(
            video, dict):
        raise PipelineResumeError(
            f"Run {run_id}: staged video metadata is "
            f"malformed; refusing resume.")
    for key in VIDEO_METADATA_KEYS:
        if sidecar.get(key) != video.get(key):
            raise PipelineResumeError(
                f"Run {run_id}: staged video metadata "
                f"disagrees with state ({key}); refusing resume.")
    if sidecar.get("sha256") != state.get("source_sha256"):
        raise PipelineResumeError(
            f"Run {run_id}: staged video is not the "
            f"fingerprinted source; refusing resume.")
    return manifest_bytes


def _read_manifest_rows(manifest_bytes: bytes, run_id: str
                        ) -> list[dict[str, str]]:
    """Parse verified manifest bytes into row dicts."""
    try:
        text = manifest_bytes.decode("utf-8")
        rows = list(csv.DictReader(text.splitlines()))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise PipelineResumeError(
            f"Run {run_id}: extraction manifest is "
            f"unreadable; refusing resume.") from exc
    for row in rows:
        try:
            int(row["frame_index"])
            float(row["timestamp_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PipelineResumeError(
                f"Run {run_id}: extraction manifest is "
                f"malformed; refusing resume.") from exc
    return rows


def _rebuild_frame_refs(frames_dir: Path,
                        rows: list[dict[str, str]],
                        run_id: str) -> list[FrameRef]:
    """Rebuild frame references from verified manifest rows."""
    refs = []
    for row in rows:
        name = row["image_path"]
        if Path(name).name != name:
            raise PipelineResumeError(
                f"Run {run_id}: manifest image path escapes "
                f"the frames directory; refusing resume.")

        def _optional(key: str) -> float | None:
            text = row.get(key)
            if text is None or text == "":
                return None
            try:
                return float(text)
            except (TypeError, ValueError) as exc:
                raise PipelineResumeError(
                    f"Run {run_id}: extraction manifest is "
                    f"malformed; refusing resume.") from exc

        refs.append(FrameRef(
            frame_index=int(row["frame_index"]),
            image_path=frames_dir / name,
            timestamp_seconds=float(row["timestamp_seconds"]),
            source_timestamp_seconds=_optional(
                "source_timestamp_seconds"),
            requested_timestamp_seconds=_optional(
                "requested_timestamp_seconds"),
        ))
    return refs


def _collect_observations(firstpass_dir: Path
                          ) -> list[dict[str, Any]]:
    """Read every batch observation file, validating shape."""
    observations: list[dict[str, Any]] = []
    pattern = sorted(
        (firstpass_dir / "observations").glob("*.json"))
    for batch_path in pattern:
        try:
            payload = json.loads(
                batch_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PipelineResumeError(
                "Observation file is unreadable: "
                f"{batch_path.name}.") from exc
        if not isinstance(payload, list):
            raise PipelineResumeError(
                "Observation file is malformed: "
                f"{batch_path.name}.")
        for item in payload:
            if (not isinstance(item, dict)
                    or not isinstance(
                        item.get("frame_index"), int)
                    or isinstance(item.get("frame_index"),
                                  bool)):
                raise PipelineResumeError(
                    "Observation file is malformed: "
                    f"{batch_path.name}.")
        observations.extend(payload)
    return observations


def _hash_firstpass_files(firstpass_dir: Path
                          ) -> dict[str, str]:
    """Hash every first-pass output file by relative path.

    Transient ``*.tmp`` debris from crashed atomic writes is
    excluded: it is never read and is overwritten on retry.
    """
    hashed: dict[str, str] = {}
    for sub in ("raw_responses", "observations", "checkpoints",
                "reports"):
        subdir = firstpass_dir / sub
        if not subdir.is_dir():
            continue
        for path in sorted(subdir.iterdir()):
            if path.is_file() and path.suffix != ".tmp":
                hashed[path.relative_to(
                    firstpass_dir).as_posix()] = _sha256_bytes(
                        path.read_bytes())
    return hashed


def _verify_firstpass_outputs(
        firstpass_dir: Path, state: dict[str, Any],
        indices: list[int]) -> dict[str, Any]:
    """Verify a completed first-pass run; return its report.

    The report bytes, the observation coverage, and the full
    output file set must all match the state record; anything
    else fails closed instead of reconstructing from doubt.
    """
    run_id = state.get("run_id", "?")
    report_path = (firstpass_dir / "reports" / "run_report.json")
    if not report_path.is_file():
        raise PipelineResumeError(
            f"Run {run_id}: first-pass report is missing; "
            f"refusing resume.")
    report_bytes = report_path.read_bytes()
    if _sha256_bytes(report_bytes) != state.get(
            "firstpass_report_sha256"):
        raise PipelineResumeError(
            f"Run {run_id}: first-pass report changed; "
            f"refusing resume.")
    try:
        report = json.loads(report_bytes.decode("utf-8"))
    except ValueError as exc:
        raise PipelineResumeError(
            f"Run {run_id}: first-pass report is unreadable; "
            f"refusing resume.") from exc
    if not isinstance(report, dict):
        raise PipelineResumeError(
            f"Run {run_id}: first-pass report is malformed; "
            f"refusing resume.")
    for key in ("frames_total", "frames_observed",
                "frames_failed"):
        value = report.get(key)
        if (not isinstance(value, int)
                or isinstance(value, bool)):
            raise PipelineResumeError(
                f"Run {run_id}: first-pass report is "
                f"malformed; refusing resume.")
    if _hash_firstpass_files(firstpass_dir) != state.get(
            "firstpass_files_sha256"):
        raise PipelineResumeError(
            f"Run {run_id}: first-pass outputs changed; "
            f"refusing resume.")
    observations = _collect_observations(firstpass_dir)
    if report["frames_total"] != len(indices):
        raise PipelineResumeError(
            f"Run {run_id}: first-pass report covers the "
            f"wrong frame set; refusing resume.")
    if len(observations) != report["frames_observed"]:
        raise PipelineResumeError(
            f"Run {run_id}: observation coverage disagrees "
            f"with the first-pass report; refusing resume.")
    if {item["frame_index"] for item in observations
            } - set(indices):
        raise PipelineResumeError(
            f"Run {run_id}: observations cover unknown "
            f"frames; refusing resume.")
    return report


def _lifetime_firstpass_stats(firstpass_dir: Path,
                              total_frames: int
                              ) -> dict[str, Any]:
    """Derive lifetime observation stats from durable files.

    Each frame takes its observation terminal when observed,
    else the terminal of its latest raw record; repeated
    attempts keep only the latest terminal per frame. Failed
    means observed-nowhere: this pipeline never budget-halts,
    so no frame is left unattempted.
    """
    terminals: dict[int, str] = {}
    pattern = sorted(
        (firstpass_dir / "raw_responses").glob("*.json"))
    for raw_path in pattern:
        try:
            records = json.loads(
                raw_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PipelineResumeError(
                f"Raw response file is unreadable: "
                f"{raw_path.name}.") from exc
        if not isinstance(records, list):
            raise PipelineResumeError(
                f"Raw response file is malformed: "
                f"{raw_path.name}.")
        for record in records:
            frames = record.get("frame_indices") if isinstance(
                record, dict) else None
            terminal = record.get("terminal_state") if (
                isinstance(record, dict)) else None
            if (not isinstance(frames, list)
                    or len(frames) != 1
                    or not isinstance(frames[0], int)
                    or isinstance(frames[0], bool)
                    or not isinstance(terminal, str)):
                raise PipelineResumeError(
                    f"Raw response file is malformed: "
                    f"{raw_path.name}.")
            terminals[frames[0]] = terminal
    observations = _collect_observations(firstpass_dir)
    for item in observations:
        terminal = item.get("terminal_state")
        if not isinstance(terminal, str):
            raise PipelineResumeError(
                "Observation file is malformed: terminal "
                "state is missing.")
        terminals[item["frame_index"]] = terminal
    observed = len(observations)
    states: dict[str, int] = {}
    for terminal in terminals.values():
        states[terminal] = states.get(terminal, 0) + 1
    return {
        "observed": observed,
        "failed": total_frames - observed,
        "terminal_states": states,
    }


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
    billable_client: InferenceClient | None = None,
    billable_token_budget: int = 1024,
    billable_max_requests: int | None = None,
    billable_cost_ceiling_usd: float | None = None,
    billable_rates_usd_per_m: tuple[float, float] | None = None,
    billable_strict_cost: bool = False,
    billable_api_key: str = "",
) -> dict[str, bytes]:
    """Run all pipeline stages; return id -> bytes artifacts.

    Orchestration over caller-provided bytes and directory:
    stage, extract, infer, transform, reconstruct. Inference
    uses the deterministic fake unless ``billable_client``
    supplies an explicit billable-capable client (S4-B:
    MetaSparkClient with a test-double SDK only), which
    runs under S3 single-submit invariants via
    ``billable_runner_config``. At most one client override
    may be given. No registered workflow supplies the
    billable seam. The work directory is durable and
    run-scoped: a fresh directory runs every stage, while a
    directory holding a verified state file resumes instead
    of restarting (see the module docstring). Raises
    VideoError for intake/extraction failures,
    PipelineResumeError for incompatible prior state, and
    ValueError when inference yields no usable
    observations.
    """
    if len(video_bytes) > PIPELINE_MAX_VIDEO_BYTES:
        raise VideoError(
            "VIDEO_TOO_LARGE",
            f"Pipeline video is {len(video_bytes)} bytes; limit is "
            f"{PIPELINE_MAX_VIDEO_BYTES}.",
            video_name,
        )
    actual_sha = _sha256_bytes(video_bytes)
    if actual_sha != source_sha256:
        raise ValueError(
            f"Pipeline source bytes do not match declared "
            f"sha256 for {video_name!r}.")
    meta_backed = billable_client is not None
    if meta_backed and fake is not None:
        raise ValueError(
            "Pipeline accepts at most one inference client "
            "override (fake or billable_client).")
    if not meta_backed and (
            billable_token_budget != 1024
            or billable_max_requests is not None
            or billable_cost_ceiling_usd is not None
            or billable_rates_usd_per_m is not None
            or billable_strict_cost is not False
            or billable_api_key != ""):
        raise ValueError(
            "billable_* parameters require billable_client; "
            "refusing to silently ignore billing bounds.")
    if meta_backed:
        _validate_billable_bounds(
            billable_token_budget, billable_max_requests,
            billable_cost_ceiling_usd, billable_rates_usd_per_m,
            billable_strict_cost, billable_api_key)
    client: InferenceClient = (
        billable_client if meta_backed
        else fake or DeterministicFakeClient())
    provider_id = META_PROVIDER_ID if meta_backed else FAKE_PROVIDER_ID
    fingerprinted_rates = (
        list(billable_rates_usd_per_m)
        if meta_backed and billable_rates_usd_per_m is not None
        else None)
    staged_name, sidecar_name = _staged_names(
        video_name, actual_sha)
    firstpass_config = {
        "model": client_model_id(client),
        "provider": provider_id,
        "pass_id": PIPELINE_PASS_ID,
        "batch_size": PIPELINE_BATCH_SIZE,
        "token_budgets": (
            [billable_token_budget] if meta_backed
            else list(DEFAULT_BUDGETS)),
        "prompt_sha256": prompt_hash(PIPELINE_PROMPT),
        "max_requests": (
            billable_max_requests if meta_backed else None),
        "cost_ceiling_usd": (
            billable_cost_ceiling_usd if meta_backed else None),
        "rates_usd_per_m": fingerprinted_rates,
        "strict_cost": (
            billable_strict_cost if meta_backed else False),
        # S3/S4-B/S5-A: billing posture and execution policy
        # are fingerprinted. Any mismatch (provider, model,
        # budgets, attempts, billing, uncertain policy,
        # request/cost guards) refuses resume with a diff
        # instead of resuming onto a different policy.
        "max_attempts": 1 if meta_backed else DEFAULT_MAX_ATTEMPTS,
        "billable": meta_backed,
        "uncertain_policy": (
            UNCERTAIN_REFUSE if meta_backed else UNCERTAIN_RERUN),
    }
    state = _read_state(work_dir)
    resumed = state is not None
    if state is None:
        firstpass_dir = work_dir / "firstpass"
        if firstpass_dir.exists() and any(
                firstpass_dir.iterdir()):
            raise PipelineResumeError(
                f"Run {run_id}: first-pass outputs exist "
                f"without pipeline state; refusing to discard "
                f"billed work.")
        _reset_stage_dirs(work_dir)
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
        sidecar_path = (work_dir / "staging" / sidecar_name)
        state = {
            "state_version": PIPELINE_STATE_VERSION,
            "workflow": WORKFLOW_VIDEO_PIPELINE_OFFLINE,
            "run_id": run_id,
            "source_object_id": source_object_id,
            "source_sha256": actual_sha,
            "stage": STAGE_EXTRACTED,
            "video": staged.metadata.to_dict(),
            "extraction": {
                "sample_fps": sample_fps,
                "max_frames": max_frames,
                "frame_indices": [frame.frame_index
                                  for frame in extraction.frames],
                "manifest_sha256": _sha256_bytes(
                    extraction.manifest_path.read_bytes()),
                "sidecar_name": sidecar_name,
                "sidecar_sha256": _sha256_bytes(
                    sidecar_path.read_bytes()),
                "frame_hashes": {
                    str(frame.frame_index): _sha256_bytes(
                        frame.image_path.read_bytes())
                    for frame in extraction.frames
                },
                "source_name": extraction.source.name,
                "source_sha256": extraction.source.sha256,
            },
            "firstpass": firstpass_config,
            "firstpass_report_sha256": None,
            "firstpass_files_sha256": None,
        }
        _write_state_atomic(work_dir, state)
        video_meta = staged.metadata.to_dict()
        frame_refs = extraction.frames
        source_ref = extraction.source
        manifest_bytes = extraction.manifest_path.read_bytes()
    else:
        expected = {
            "workflow": WORKFLOW_VIDEO_PIPELINE_OFFLINE,
            "run_id": run_id,
            "source_object_id": source_object_id,
            "source_sha256": actual_sha,
            "extraction": {
                "sample_fps": sample_fps,
                "max_frames": max_frames,
                "source_name": (f"{Path(staged_name).stem}@"
                                f"{sample_fps}fps"),
                "source_sha256": actual_sha,
                "sidecar_name": sidecar_name,
            },
            "firstpass": firstpass_config,
        }
        actual = {
            key: state.get(key)
            for key in ("workflow", "run_id",
                        "source_object_id", "source_sha256")
        }
        prior_extraction = state.get("extraction")
        if not isinstance(prior_extraction, dict):
            prior_extraction = {}
        actual["extraction"] = {
            key: prior_extraction.get(key)
            for key in ("sample_fps", "max_frames",
                        "source_name", "source_sha256",
                        "sidecar_name")
        }
        actual["firstpass"] = state.get("firstpass")
        diffs = _diff_json(expected, actual, "state")
        if diffs:
            raise PipelineResumeError(
                f"Run {run_id}: work dir state does not match "
                f"this run ({len(diffs)} difference(s)): "
                + "; ".join(diffs) + ".")
        manifest_bytes = _verify_extraction_outputs(
            work_dir, state)
        video_meta = {
            key: state["video"][key]
            for key in VIDEO_METADATA_KEYS
        }
        rows = _read_manifest_rows(
            manifest_bytes, state.get("run_id", "?"))
        frame_refs = _rebuild_frame_refs(
            work_dir / "frames", rows,
            state.get("run_id", "?"))
        source_ref = SourceRef(
            name=state["extraction"]["source_name"],
            sha256=state["extraction"]["source_sha256"],
        )
    firstpass_dir = work_dir / "firstpass"
    if state["stage"] == STAGE_FIRSTPASS_DONE:
        firstpass_report = _verify_firstpass_outputs(
            firstpass_dir, state,
            [frame.frame_index for frame in frame_refs])
    else:
        state["stage"] = STAGE_FIRSTPASS_RUNNING
        _write_state_atomic(work_dir, state)
        if meta_backed:
            runner_config = billable_runner_config(
                frames=frame_refs, source=source_ref,
                out_dir=firstpass_dir,
                prompt_text=PIPELINE_PROMPT,
                client_factory=lambda: client,
                model=firstpass_config["model"],
                token_budget=billable_token_budget,
                pass_id=firstpass_config["pass_id"],
                batch_size=firstpass_config["batch_size"],
                max_requests=billable_max_requests,
                rates_usd_per_m=(
                    tuple(billable_rates_usd_per_m)
                    if billable_rates_usd_per_m is not None
                    else None),
                cost_ceiling_usd=billable_cost_ceiling_usd,
                strict_cost=billable_strict_cost,
                api_key=billable_api_key,
            )
        else:
            runner_config = FirstPassConfig(
                frames=frame_refs, source=source_ref,
                out_dir=firstpass_dir,
                model=firstpass_config["model"],
                prompt_text=PIPELINE_PROMPT,
                pass_id=firstpass_config["pass_id"],
                provider=firstpass_config["provider"],
                batch_size=firstpass_config["batch_size"],
                client_factory=lambda: client,
                billable=False,
                uncertain_policy=UNCERTAIN_RERUN,
            )
        firstpass_report = FirstPassRunner(runner_config).run()
        report_path = (firstpass_dir / "reports"
                       / "run_report.json")
        try:
            report_bytes = report_path.read_bytes()
        except OSError as exc:
            raise PipelineResumeError(
                f"Run {run_id}: first-pass report is missing "
                f"after a successful run.") from exc
        state["stage"] = STAGE_FIRSTPASS_DONE
        state["firstpass_report_sha256"] = _sha256_bytes(
            report_bytes)
        state["firstpass_files_sha256"] = _hash_firstpass_files(
            firstpass_dir)
        _write_state_atomic(work_dir, state)
    observations = _collect_observations(firstpass_dir)
    indices = [frame.frame_index for frame in frame_refs]
    lifetime = _lifetime_firstpass_stats(
        firstpass_dir, len(indices))
    if not observations:
        raise ValueError(
            f"Pipeline produced no observations "
            f"({lifetime['failed']} failed of {len(indices)}); "
            f"failing instead of reconstructing from nothing.")
    stamps = [frame.timestamp_seconds for frame in frame_refs]
    event_id = f"pipeline_window_{stamps[0]:g}_{stamps[-1]:g}"
    video_id = f"upload_{video_meta['sha256'][:12]}"
    temporal = build_pipeline_temporal(
        event_id, indices, stamps[0], stamps[-1])
    bundle_raw = (json.dumps({
        "bundle_format": "bakery-offline-bundle/1",
        "video_id": video_id,
        "event_id": event_id,
        "source_fps": video_meta["fps"],
        "observations": observations,
        "temporal": temporal,
        "execution_reports": [],
        "units": None,
    }, indent=2, sort_keys=True) + "\n").encode("utf-8")
    bundle = parse_offline_bundle(bundle_raw)
    summary, artifacts = run_bundle_reconstruction(bundle, run_id)
    firstpass_report_raw = (
        firstpass_dir / "reports" / "run_report.json").read_bytes()
    manifest_sha = _sha256_bytes(manifest_bytes)
    if meta_backed:
        note = ("Meta-backed pipeline run with a caller-supplied "
                "client under single-submit billable policy. S4-B "
                "supports test-double SDKs only: no provider was "
                "contacted, all readings are synthetic, and the "
                "draft requires human review before operational "
                "use.")
    else:
        note = ("Offline pipeline run over a staged upload with "
                "deterministic fake inference. No model was "
                "queried; all readings are synthetic test vectors "
                "and the draft requires human review before "
                "operational use.")
    payload = {
        "workflow": WORKFLOW_VIDEO_PIPELINE_OFFLINE,
        "run_id": run_id,
        "resumed": resumed,
        "source_object_id": source_object_id,
        "source_sha256": source_sha256,
        "evidence_version": PIPELINE_EVIDENCE_VERSION,
        "inference_performed": False,
        "network_accessed": False,
        "paid_services_used": False,
        "synthetic_inference": True,
        "fake_model_id": client_model_id(client),
        "fake_provider_id": provider_id,
        "video": video_meta,
        "extraction": {
            "sample_fps": sample_fps,
            "frame_indices": indices,
            "manifest_sha256": manifest_sha,
        },
        "firstpass": {
            "model": client_model_id(client),
            "provider": provider_id,
            "pass_id": PIPELINE_PASS_ID,
            "frames_total": len(indices),
            "frames_observed": lifetime["observed"],
            "frames_failed": lifetime["failed"],
            "terminal_states": lifetime["terminal_states"],
        },
        "bundle_sha256": _sha256_bytes(bundle_raw),
        "draft_id": summary["draft_id"],
        "measurements": summary["measurements"],
        "actions": summary["actions"],
        "unresolved": summary["unresolved"],
        "issues": summary["issues"],
        "artifacts": sorted([*artifacts, PIPELINE_SUMMARY_ARTIFACT,
                             FIRSTPASS_REPORT_ARTIFACT]),
        "note": note,
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
    """Worker entry point: pipeline a small uploaded video offline.

    Runs inside the durable run-scoped work directory, so a
    restarted worker resumes verified progress instead of
    restarting. Incompatible prior state raises
    PipelineResumeError, an explicit job failure that is never
    silently retried.
    """
    if record.source_object_id is None:
        raise ValueError("video_pipeline_offline requires a source.")
    if not storage.source_exists(record.source_object_id):
        raise ValueError(
            f"Unknown source: {record.source_object_id!r}.")
    with storage.open_source(record.source_object_id) as handle:
        raw = handle.read()
    return run_pipeline_offline(
        video_bytes=raw,
        video_name=record.source_object_id,
        run_id=record.run_id,
        source_object_id=record.source_object_id,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        work_dir=storage.run_work_dir(record.run_id),
    )


def _build_meta_client(*, model: str, timeout_seconds: float,
                       store: bool) -> InferenceClient:
    """Construct the billable Meta client (live-capable SDK).

    Tests monkeypatch this seam to inject test-double SDKs.
    Production always returns a real-factory client, so the
    seam is reachable only through the authorized executor
    below: no registered fake workflow touches it.
    """
    return MetaSparkClient(
        model=model, timeout_seconds=timeout_seconds,
        store=store)


def execute_video_pipeline_meta(
        storage: StorageBackend,
        record: JobRecord) -> dict[str, bytes]:
    """Worker entry point: authorized billable Meta pipeline.

    Authorization first: the stored grant must validate and
    match this job, and a provider key must be present in
    the worker environment. Only then is the client built
    and the billable seam executed with the granted bounds.
    Any failure before or during execution fails the job
    explicitly without automatic retry; the key never
    reaches persisted errors or artifacts.
    """
    if record.grant_json is None:
        raise GrantError(
            f"Workflow {record.workflow!r} requires an "
            "authorization grant; refusing unauthorized "
            "billable execution.")
    try:
        grant = parse_grant(json.loads(record.grant_json))
    except (json.JSONDecodeError, ValueError) as exc:
        raise GrantError(
            f"Stored grant for {record.run_id} is invalid: "
            f"{exc}") from exc
    if grant.workflow != record.workflow:
        raise GrantError(
            f"Grant workflow {grant.workflow!r} does not match "
            f"job workflow {record.workflow!r}.")
    api_key = resolve_api_key()
    if record.source_object_id is None:
        raise ValueError(
            f"{WORKFLOW_VIDEO_PIPELINE_META} requires a source.")
    if not storage.source_exists(record.source_object_id):
        raise ValueError(
            f"Unknown source: {record.source_object_id!r}.")
    with storage.open_source(record.source_object_id) as handle:
        raw = handle.read()
    client = _build_meta_client(
        model=grant.model, timeout_seconds=grant.timeout_seconds,
        store=grant.store)
    try:
        return run_pipeline_offline(
            video_bytes=raw,
            video_name=record.source_object_id,
            run_id=record.run_id,
            source_object_id=record.source_object_id,
            source_sha256=hashlib.sha256(raw).hexdigest(),
            work_dir=storage.run_work_dir(record.run_id),
            billable_client=client,
            billable_token_budget=grant.token_budget,
            billable_max_requests=grant.max_requests,
            billable_cost_ceiling_usd=grant.cost_ceiling_usd,
            billable_rates_usd_per_m=grant.rates_usd_per_m,
            billable_strict_cost=grant.strict_cost,
            billable_api_key=api_key,
            max_frames=grant.max_frames,
        )
    except Exception as exc:
        message = str(exc)
        cleaned = redact_key(message, api_key)
        if cleaned != message:
            raise RuntimeError(cleaned) from None
        raise
