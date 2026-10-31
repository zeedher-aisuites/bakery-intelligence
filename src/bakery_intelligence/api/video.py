"""Recorded-evidence assembly for video reconstruction (M2-06C.1).

Workflow ``video_reconstruct`` resolves a small uploaded manifest
against an explicit allowlist of recorded evidence sets, reads the
recorded files strictly read-only, assembles the equivalent offline
bundle, and executes the shared reconstruction core
(``offline.run_bundle_reconstruction``). No inference runs, no
network is touched, and no paid service is involved in C.1.

Recorded sets live outside the API storage (default
``<cwd>/data/processed``) and are never written: every consumed
file is hashed for the provenance record and left untouched. New
sets join the allowlist only through an explicit code change,
never through user-supplied paths.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bakery_intelligence.api.jobs import WORKFLOW_VIDEO_RECONSTRUCT
from bakery_intelligence.api.offline import (
    OFFLINE_SUMMARY_ARTIFACT,
    OfflineBundle,
    parse_offline_bundle,
    run_bundle_reconstruction,
)

if TYPE_CHECKING:
    from bakery_intelligence.api.jobs import JobRecord
    from bakery_intelligence.api.storage import StorageBackend

VIDEO_MANIFEST_FORMAT = "bakery-video-manifest/1"
VIDEO_SUMMARY_ARTIFACT = "video_summary.json"

DEFAULT_EVIDENCE_DIRNAME = "data/processed"

BATCH_FILENAME = re.compile(r"batch_[A-Za-z0-9._-]+\.json")

# Allowlisted recorded evidence sets: name -> paths relative to the
# evidence root. Observations and temporal are required; executions
# and units are optional and recorded as absent when missing.
EVIDENCE_SETS = {
    "p04_full_4096": {
        "observations": (
            "experiment_006/full_run/p04_full_4096/observations"),
        "temporal": (
            "experiment_006/temporal/p04_full_4096/"
            "temporal_events_1.1.0.json"),
        "executions": (
            "experiment_007/selective_verification/executions"),
        "units": "experiment_008/reconstruction/units.json",
    },
}


class VideoManifestError(ValueError):
    """A video manifest failed structural validation."""


class EvidenceUnavailableError(ValueError):
    """Recorded evidence is missing or unreadable (read-only)."""


@dataclass
class VideoManifest:
    """Validated job manifest selecting recorded evidence."""

    evidence_set: str
    event_id: str
    video_id: str
    draft_version: int = 1
    source_dataset: str | None = None
    window: tuple[float, float] | None = None
    manifest_sha256: str = ""


@dataclass
class ResolvedEvidence:
    """Evidence-set files resolved under one root (read-only)."""

    evidence_set: str
    root: Path
    obs_dir: Path
    temporal_path: Path
    exec_dir: Path | None
    units_path: Path | None
    provenance: list[dict[str, Any]] = field(default_factory=list)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(
        value, bool)


def parse_video_manifest(data: bytes) -> VideoManifest:
    """Validate manifest bytes; raise VideoManifestError when invalid."""
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VideoManifestError(
            f"Manifest is not valid UTF-8 JSON: {exc}.") from exc
    if not isinstance(payload, dict):
        raise VideoManifestError("Manifest must be a JSON object.")
    if payload.get("manifest_format") != VIDEO_MANIFEST_FORMAT:
        raise VideoManifestError(
            f"Unsupported manifest_format: "
            f"{payload.get('manifest_format')!r} "
            f"(expected {VIDEO_MANIFEST_FORMAT!r}).")
    evidence_set = payload.get("evidence_set")
    if not isinstance(evidence_set, str) or not evidence_set:
        raise VideoManifestError(
            "Manifest needs an evidence_set string.")
    if evidence_set not in EVIDENCE_SETS:
        raise VideoManifestError(
            f"Unknown evidence_set: {evidence_set!r}.")
    event_id = payload.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        raise VideoManifestError("Manifest needs an event_id string.")
    video_id = payload.get("video_id")
    if not isinstance(video_id, str) or not video_id:
        raise VideoManifestError("Manifest needs a video_id string.")
    draft_version = payload.get("draft_version", 1)
    if (not isinstance(draft_version, int)
            or isinstance(draft_version, bool)
            or draft_version < 1):
        raise VideoManifestError(
            "Manifest draft_version must be an integer >= 1.")
    source_dataset = payload.get("source_dataset")
    if source_dataset is not None and not isinstance(
            source_dataset, str):
        raise VideoManifestError(
            "Manifest source_dataset must be a string or null.")
    window: tuple[float, float] | None = None
    raw_window = payload.get("window")
    if raw_window is not None:
        if not isinstance(raw_window, dict):
            raise VideoManifestError(
                "Manifest window must be an object or null.")
        start = raw_window.get("start")
        end = raw_window.get("end")
        if not _is_number(start) or not _is_number(end):
            raise VideoManifestError(
                "Manifest window needs numeric start and end.")
        window = (float(start), float(end))
    return VideoManifest(
        evidence_set=evidence_set, event_id=event_id,
        video_id=video_id, draft_version=draft_version,
        source_dataset=source_dataset, window=window,
        manifest_sha256=hashlib.sha256(data).hexdigest(),
    )


def default_evidence_root() -> Path:
    """Return the default recorded-evidence root under cwd."""
    return Path.cwd() / DEFAULT_EVIDENCE_DIRNAME


def _hash_file(path: Path) -> tuple[str, int]:
    """Stream one file's SHA-256 and size (read-only)."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _record(root: Path, role: str, path: Path | None,
            provenance: list[dict[str, Any]]) -> None:
    """Append one file's provenance entry (or its absence)."""
    if path is None or not path.is_file():
        provenance.append({
            "role": role,
            "present": False,
            "path": (path.relative_to(root).as_posix()
                     if path is not None else None),
            "sha256": None,
            "bytes": None,
        })
        return
    digest, size = _hash_file(path)
    provenance.append({
        "role": role,
        "present": True,
        "path": path.relative_to(root).as_posix(),
        "sha256": digest,
        "bytes": size,
    })


def resolve_evidence_set(root: Path | None,
                         evidence_set: str) -> ResolvedEvidence:
    """Resolve an allowlisted set under root (read-only checks).

    Raises EvidenceUnavailableError when required inputs are
    missing. Optional inputs resolve to None and are recorded as
    absent in the provenance list.
    """
    if evidence_set not in EVIDENCE_SETS:
        raise EvidenceUnavailableError(
            f"Unknown evidence_set: {evidence_set!r}.")
    base = (root or default_evidence_root()).resolve()
    spec = EVIDENCE_SETS[evidence_set]
    obs_dir = (base / spec["observations"]).resolve()
    temporal_path = (base / spec["temporal"]).resolve()
    for candidate in (obs_dir, temporal_path):
        try:
            candidate.relative_to(base)
        except ValueError as exc:
            raise EvidenceUnavailableError(
                f"Evidence path escapes root: {candidate}.") from exc
    if not obs_dir.is_dir():
        raise EvidenceUnavailableError(
            f"Recorded observations not found: {obs_dir}.")
    if not temporal_path.is_file():
        raise EvidenceUnavailableError(
            f"Recorded temporal artifact not found: {temporal_path}.")
    exec_dir = (base / spec["executions"]).resolve()
    if not exec_dir.is_dir():
        exec_dir = None
    units_path = (base / spec["units"]).resolve()
    if not units_path.is_file():
        units_path = None
    resolved = ResolvedEvidence(
        evidence_set=evidence_set, root=base, obs_dir=obs_dir,
        temporal_path=temporal_path, exec_dir=exec_dir,
        units_path=units_path,
    )
    return resolved


def _read_json(path: Path, role: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceUnavailableError(
            f"Unreadable recorded {role}: {path.name}.") from exc


def build_recorded_bundle(resolved: ResolvedEvidence,
                          manifest: VideoManifest) -> OfflineBundle:
    """Assemble the equivalent offline bundle from recorded files.

    Batch files concatenate in sorted filename order; every consumed
    file is hashed into the resolved provenance list. Recorded files
    are only read, never modified, moved, or renamed.
    """
    batch_paths = sorted(resolved.obs_dir.glob("batch_*.json"))
    if not batch_paths:
        raise EvidenceUnavailableError(
            f"No recorded batch files in: {resolved.obs_dir}.")
    observations: list[dict[str, Any]] = []
    for path in batch_paths:
        if BATCH_FILENAME.fullmatch(path.name) is None:
            raise EvidenceUnavailableError(
                f"Refusing unexpected batch name: {path.name}.")
        payload = _read_json(path, "observations")
        if not isinstance(payload, list):
            raise EvidenceUnavailableError(
                f"Recorded batch is not a list: {path.name}.")
        observations.extend(payload)
        _record(resolved.root, "observations", path,
                resolved.provenance)
    temporal = _read_json(resolved.temporal_path, "temporal")
    _record(resolved.root, "temporal", resolved.temporal_path,
            resolved.provenance)
    reports: list[dict[str, Any]] = []
    if resolved.exec_dir is not None:
        for path in sorted(
                resolved.exec_dir.glob("*/execution_report.json")):
            reports.append(_read_json(path, "execution report"))
            _record(resolved.root, "execution_report", path,
                    resolved.provenance)
    else:
        resolved.provenance.append({
            "role": "execution_reports", "present": False,
            "path": None, "sha256": None, "bytes": None,
        })
    units: dict[str, Any] | None = None
    if resolved.units_path is not None:
        units = _read_json(resolved.units_path, "units sidecar")
        _record(resolved.root, "units", resolved.units_path,
                resolved.provenance)
    else:
        resolved.provenance.append({
            "role": "units", "present": False,
            "path": None, "sha256": None, "bytes": None,
        })
    bundle_payload = {
        "bundle_format": "bakery-offline-bundle/1",
        "video_id": manifest.video_id,
        "event_id": manifest.event_id,
        "draft_version": manifest.draft_version,
        "source_dataset": manifest.source_dataset,
        "window": (
            {"start": manifest.window[0], "end": manifest.window[1]}
            if manifest.window else None),
        "observations": observations,
        "temporal": temporal,
        "execution_reports": reports,
        "units": units,
    }
    raw = (json.dumps(bundle_payload, indent=2, sort_keys=True) + "\n"
           ).encode("utf-8")
    return parse_offline_bundle(raw)


def execute_video_reconstruct(
        storage: StorageBackend,
        record: JobRecord,
        evidence_root: Path | None = None) -> dict[str, bytes]:
    """Assemble recorded evidence and reconstruct; id -> bytes map.

    Raises VideoManifestError for malformed manifests,
    EvidenceUnavailableError for missing evidence, and propagates
    ReconstructionError for domain failures. The worker records any
    of these as a structured job failure.
    """
    if record.source_object_id is None:
        raise ValueError("video_reconstruct requires a source.")
    if not storage.source_exists(record.source_object_id):
        raise ValueError(
            f"Unknown source: {record.source_object_id!r}.")
    with storage.open_source(record.source_object_id) as handle:
        raw = handle.read()
    manifest = parse_video_manifest(raw)
    resolved = resolve_evidence_set(evidence_root,
                                    manifest.evidence_set)
    bundle = build_recorded_bundle(resolved, manifest)
    summary, artifacts = run_bundle_reconstruction(
        bundle, record.run_id)
    payload = {
        "workflow": WORKFLOW_VIDEO_RECONSTRUCT,
        "run_id": record.run_id,
        "video_id": manifest.video_id,
        "event_id": manifest.event_id,
        "evidence_set": manifest.evidence_set,
        "manifest_sha256": manifest.manifest_sha256,
        "source_object_id": record.source_object_id,
        "inference_performed": False,
        "network_accessed": False,
        "paid_services_used": False,
        "draft_id": summary["draft_id"],
        "measurements": summary["measurements"],
        "actions": summary["actions"],
        "unresolved": summary["unresolved"],
        "issues": summary["issues"],
        "inputs": resolved.provenance,
        "artifacts": sorted([*artifacts,
                             VIDEO_SUMMARY_ARTIFACT]),
        "note": ("Deterministic reconstruction from read-only "
                 "recorded evidence. This run performed no "
                 "inference and queried no model; the draft "
                 "requires human review before operational use."),
    }
    artifacts[VIDEO_SUMMARY_ARTIFACT] = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    # The shared core never writes the offline receipt; guard the
    # namespace so the two workflows cannot be confused.
    assert OFFLINE_SUMMARY_ARTIFACT not in artifacts
    return artifacts
