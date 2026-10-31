"""Offline pipeline bridge for durable job execution (M2-06B.1).

Workflow ``offline_reconstruct`` consumes a self-contained JSON
bundle of existing analysis artifacts (first-pass observations,
temporal events, optional verification reports and unit sidecar)
and executes the deterministic reconstruction stage
(``reconstruction.cli.reconstruct``). No inference runs, no network
is touched, and no paid service is involved: stages that need a
model or the network are out of scope for this workflow.

The bundle travels as the job's uploaded source object, so every
offline run is fully reproducible from its stored source bytes.
Outputs mirror the reconstruction CLI files plus an explicit
``offline_summary.json`` that states what ran and what did not.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bakery_intelligence.api.jobs import WORKFLOW_OFFLINE_RECONSTRUCT
from bakery_intelligence.reconstruction.cli import reconstruct

if TYPE_CHECKING:
    from bakery_intelligence.api.jobs import JobRecord
    from bakery_intelligence.api.storage import StorageBackend

BUNDLE_FORMAT = "bakery-offline-bundle/1"

RECIPE_DRAFT_ARTIFACT = "recipe_draft.json"
WEIGHING_EVENT_ARTIFACT = "weighing_event.json"
ACTION_EVENTS_ARTIFACT = "action_events.json"
MEASUREMENTS_ARTIFACT = "measurements.json"
VALIDATION_ARTIFACT = "validation.json"
UNITS_APPLIED_ARTIFACT = "units_applied.json"
OFFLINE_SUMMARY_ARTIFACT = "offline_summary.json"

_ALWAYS_WRITTEN = (
    RECIPE_DRAFT_ARTIFACT,
    WEIGHING_EVENT_ARTIFACT,
    ACTION_EVENTS_ARTIFACT,
    MEASUREMENTS_ARTIFACT,
    VALIDATION_ARTIFACT,
)


class OfflineBundleError(ValueError):
    """An offline bundle failed structural validation."""


@dataclass
class OfflineBundle:
    """Validated offline inputs staged for reconstruction."""

    video_id: str
    event_id: str
    draft_version: int
    source_dataset: str | None
    window: tuple[float, float] | None
    observations: list[dict[str, Any]]
    temporal: dict[str, Any]
    execution_reports: list[dict[str, Any]]
    units: dict[str, Any] | None
    bundle_sha256: str


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(
        value, bool)


def parse_offline_bundle(data: bytes) -> OfflineBundle:
    """Validate bundle bytes; raise OfflineBundleError when invalid.

    Checks stay structural (shape, format tag, scalar types).
    Domain validation belongs to the reconstruction loaders, whose
    structured ReconstructionError codes surface unchanged.
    """
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OfflineBundleError(
            f"Bundle is not valid UTF-8 JSON: {exc}.") from exc
    if not isinstance(payload, dict):
        raise OfflineBundleError("Bundle must be a JSON object.")
    if payload.get("bundle_format") != BUNDLE_FORMAT:
        raise OfflineBundleError(
            f"Unsupported bundle_format: "
            f"{payload.get('bundle_format')!r} "
            f"(expected {BUNDLE_FORMAT!r}).")
    video_id = payload.get("video_id")
    if not isinstance(video_id, str) or not video_id:
        raise OfflineBundleError("Bundle needs a video_id string.")
    event_id = payload.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        raise OfflineBundleError("Bundle needs an event_id string.")
    draft_version = payload.get("draft_version", 1)
    if (not isinstance(draft_version, int)
            or isinstance(draft_version, bool)
            or draft_version < 1):
        raise OfflineBundleError(
            "Bundle draft_version must be an integer >= 1.")
    source_dataset = payload.get("source_dataset")
    if source_dataset is not None and not isinstance(
            source_dataset, str):
        raise OfflineBundleError(
            "Bundle source_dataset must be a string or null.")
    window: tuple[float, float] | None = None
    raw_window = payload.get("window")
    if raw_window is not None:
        if not isinstance(raw_window, dict):
            raise OfflineBundleError(
                "Bundle window must be an object or null.")
        start = raw_window.get("start")
        end = raw_window.get("end")
        if not _is_number(start) or not _is_number(end):
            raise OfflineBundleError(
                "Bundle window needs numeric start and end.")
        window = (float(start), float(end))
    observations = payload.get("observations")
    if not isinstance(observations, list) or not observations:
        raise OfflineBundleError(
            "Bundle observations must be a non-empty list.")
    temporal = payload.get("temporal")
    if not isinstance(temporal, dict):
        raise OfflineBundleError("Bundle temporal must be an object.")
    reports = payload.get("execution_reports", [])
    if not isinstance(reports, list):
        raise OfflineBundleError(
            "Bundle execution_reports must be a list.")
    for report in reports:
        if not isinstance(report, dict) or not isinstance(
                report.get("event_id"), str):
            raise OfflineBundleError(
                "Each execution report needs an event_id string.")
    units = payload.get("units")
    if units is not None:
        if not isinstance(units, dict) or not isinstance(
                units.get("assignments"), list):
            raise OfflineBundleError(
                "Bundle units must be an object with assignments.")
    return OfflineBundle(
        video_id=video_id, event_id=event_id,
        draft_version=draft_version, source_dataset=source_dataset,
        window=window, observations=observations, temporal=temporal,
        execution_reports=reports, units=units,
        bundle_sha256=hashlib.sha256(data).hexdigest(),
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )


def _stage_bundle(bundle: OfflineBundle, root: Path) -> dict[str, Any]:
    """Write bundle parts in the layout reconstruct() expects."""
    obs_dir = root / "obs"
    obs_dir.mkdir()
    _write_json(obs_dir / "batch_bundle.json", bundle.observations)
    temporal_path = root / "temporal.json"
    _write_json(temporal_path, bundle.temporal)
    units_path: Path | None = None
    if bundle.units is not None:
        units_path = root / "units.json"
        _write_json(units_path, bundle.units)
    exec_dir: Path | None = None
    if bundle.execution_reports:
        exec_dir = root / "executions"
        for index, report in enumerate(bundle.execution_reports):
            report_dir = exec_dir / f"report_{index:03d}"
            report_dir.mkdir(parents=True)
            _write_json(report_dir / "execution_report.json", report)
    return {"obs_dir": obs_dir, "temporal_path": temporal_path,
            "units_path": units_path, "exec_dir": exec_dir}


def run_bundle_reconstruction(
        bundle: OfflineBundle,
        run_dir_name: str) -> tuple[dict[str, Any],
                                    dict[str, bytes]]:
    """Stage one validated bundle and run reconstruction on it.

    Shared core for every bundle-shaped workflow: identical inputs
    always produce identical staged files and identical
    reconstruction outputs. Returns the reconstruct summary and the
    output artifacts (without any workflow summary receipt, which
    each caller builds from its own provenance).
    """
    with tempfile.TemporaryDirectory(
            prefix="bundle_reconstruct_") as staging:
        root = Path(staging)
        staged = _stage_bundle(bundle, root)
        out_dir = root / run_dir_name
        start, end = bundle.window if bundle.window else (None, None)
        summary = reconstruct(
            staged["obs_dir"], staged["temporal_path"],
            bundle.event_id, out_dir, bundle.video_id,
            staged["exec_dir"], staged["units_path"], start, end,
            bundle.draft_version, bundle.source_dataset,
        )
        artifacts = {name: (out_dir / name).read_bytes()
                     for name in _ALWAYS_WRITTEN}
        if staged["units_path"] is not None:
            artifacts[UNITS_APPLIED_ARTIFACT] = (
                out_dir / UNITS_APPLIED_ARTIFACT).read_bytes()
    return summary, artifacts


def execute_offline_reconstruct(
        storage: StorageBackend,
        record: JobRecord) -> dict[str, bytes]:
    """Run deterministic reconstruction; return id -> bytes mapping.

    Raises OfflineBundleError for malformed bundles and propagates
    ReconstructionError for domain failures (unknown event, missing
    measurements, invalid window). The worker records either as a
    structured job failure; nothing is silently substituted.
    """
    if record.source_object_id is None:
        raise ValueError("offline_reconstruct requires a source.")
    if not storage.source_exists(record.source_object_id):
        raise ValueError(
            f"Unknown source: {record.source_object_id!r}.")
    with storage.open_source(record.source_object_id) as handle:
        raw = handle.read()
    bundle = parse_offline_bundle(raw)
    summary, artifacts = run_bundle_reconstruction(
        bundle, record.run_id)
    payload = {
        "workflow": WORKFLOW_OFFLINE_RECONSTRUCT,
        "run_id": record.run_id,
        "video_id": bundle.video_id,
        "event_id": bundle.event_id,
        "bundle_format": BUNDLE_FORMAT,
        "bundle_sha256": bundle.bundle_sha256,
        "source_object_id": record.source_object_id,
        "inference_performed": False,
        "network_accessed": False,
        "paid_services_used": False,
        "draft_id": summary["draft_id"],
        "measurements": summary["measurements"],
        "actions": summary["actions"],
        "unresolved": summary["unresolved"],
        "issues": summary["issues"],
        "artifacts": sorted([*artifacts,
                             OFFLINE_SUMMARY_ARTIFACT]),
        "note": ("Deterministic offline reconstruction from "
                 "uploaded analysis artifacts. No video was "
                 "analyzed and no model was queried; the draft "
                 "requires human review before operational use."),
    }
    artifacts[OFFLINE_SUMMARY_ARTIFACT] = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return artifacts
