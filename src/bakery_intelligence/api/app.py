"""FastAPI application factory for the M2-06A spike.

Endpoints:

- ``GET /api/health``: real backend health response.
- ``POST /api/analyses``: store a small uploaded source and queue a
  run (multipart ``file`` field plus an explicit ``workflow`` form
  field; ``storage_smoke`` when omitted). ``offline_reconstruct``
  expects a JSON offline bundle as the source (see
  ``docs/m206/offline-integration.md``). ``video_pipeline_meta``
  additionally requires a ``grant`` form field carrying the
  authorization document (see ``docs/m206/live-authorization.md``).
- ``GET /api/analyses/{run_id}``: truthful job status record.
- ``GET /api/analyses/{run_id}/artifacts/{artifact_id}``: download a
  stored result artifact.
- ``GET /api/analyses``: list completed runs, newest first.
- ``GET /api/analyses/{run_id}/review``: per-frame review bundle
  assembled from retained work files (read-only).
- ``GET /api/analyses/{run_id}/frames/{frame}``: retained frame
  pixels (``frame_<8 digits>.jpg`` only).
- ``GET``/``POST /api/analyses/{run_id}/reviews``: durable human
  annotations stored separately from inference artifacts.

Run with ``uv run uvicorn bakery_intelligence.api.app:build_app
--factory`` (see docs/m206/spike-setup.md).
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import json
import re
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from bakery_intelligence.api.authorization import GrantError
from bakery_intelligence.adversarial.service import (
    RecipeReviewService,
)
from bakery_intelligence.adversarial.live import (
    GenerateRecipeRequest,
    LiveConfigError,
    _load_local_env,
)
from bakery_intelligence.adversarial.observations import (
    ObservationReviewError,
)
from bakery_intelligence.inference.firstpass import InferenceError
from bakery_intelligence.adversarial.store import (
    RecipeReviewError,
)
from bakery_intelligence.api.jobs import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    RetryError,
    JobService,
    UnknownWorkflowError,
    WORKFLOW_STORAGE_SMOKE,
    WORKFLOW_VIDEO_PIPELINE_META,
    WORKFLOW_VIDEO_PIPELINE_OFFLINE,
)
from bakery_intelligence.api.review import (
    ReviewError,
    build_review_bundle,
    frame_exists,
    read_frame_bytes,
    read_reviews,
    save_review,
)
from bakery_intelligence.api.storage import (
    InvalidObjectIdError,
    LocalStorageBackend,
    ObjectNotFoundError,
    StorageBackend,
    validate_object_id,
    validate_run_id,
)
from bakery_intelligence.api.limits import (
    effective_video_max_bytes,
    pipeline_max_bytes,
    upload_max_bytes,
)
from bakery_intelligence.api.store import SqliteJobStore
from bakery_intelligence.video.intake import (
    VideoError,
    probe_video,
    sanitize_stem,
)

SERVICE_NAME = "bakery-intelligence-api"
SERVICE_VERSION = "0.1.0-spike"
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
DEFAULT_STORAGE_DIRNAME = "runs_api"
UPLOADS_DIRNAME = "uploads"
UPLOAD_TMP_DIRNAME = "uploads_tmp"
DEMO_RUN_ID = "run_387662c3eaab"
FIRSTPASS_REPORT_ARTIFACT = "firstpass_report.json"
PIPELINE_FRAME_CAP = 1000
META_RATES_USD_PER_M = (1.25, 4.25)


def max_video_upload_bytes() -> int:
    """Configurable cap for the real-video upload endpoint.

    Delegates to the shared limits module so the upload and
    pipeline caps cannot drift apart.
    """
    return upload_max_bytes()

DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


class AnalysisCreated(BaseModel):
    """Response for a successfully queued analysis run."""

    run_id: str
    workflow: str
    status: str
    source_object_id: str


class HealthResponse(BaseModel):
    """Backend health response with storage backend identity."""

    status: str
    service: str
    version: str
    storage_backend: str
    storage_ready: bool


class UploadAnalysisRequest(BaseModel):
    """Start an analysis for a previously staged upload."""

    workflow: str
    grant: dict[str, Any] | None = None


class RecipeCorrections(BaseModel):
    """Chef edits to the editable recipe draft."""

    ingredients: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    reviewer: str


class ObservationCorrection(BaseModel):
    """One human decision on an observation review task."""

    task_id: str
    action: str
    human_value: Any = None
    reviewer: str


class ObservationUndo(BaseModel):
    """Revert the latest decision on a task without erasing history."""

    task_id: str
    reviewer: str


class ObservationClassification(BaseModel):
    """Human classification of one frame, from the viewer."""

    frame_index: int
    action: str
    human_value: Any = None
    reviewer: str


class RecipeDecisionRequest(BaseModel):
    """Explicit human decision on one draft version."""

    decision: str
    reviewer: str
    notes: str | None = None


class CulinaryEnrichRequest(BaseModel):
    """Explicitly trigger one culinary enrichment batch."""

    max_frames: int = 5
    confirmed: bool = False
    frame_indices: list[int] | None = None


class CulinaryRequeueRequest(BaseModel):
    """Explicitly authorize reprocessing of terminal-failed frames."""

    frame_indices: list[int]
    reason: str
    reviewer: str


class ReviewSubmission(BaseModel):
    """One human review decision for a single frame."""

    frame_index: int
    decision: str
    value: float | None = None
    notes: str | None = None
    reviewer: str | None = None


def sanitize_filename(filename: str | None) -> str:
    """Derive a safe storage object id from a client filename."""
    stem = (filename or "upload").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = stem.strip() or "upload"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    cleaned = re.sub(r"\.{2,}", "_", cleaned).strip("._-")
    candidate = f"upload_{cleaned}"[:64].rstrip("._-") or "upload"
    return validate_object_id(candidate)


def default_storage_dir() -> Path:
    """Resolve the default API storage root under the repository."""
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "pyproject.toml").is_file():
            return parent / DEFAULT_STORAGE_DIRNAME
    return Path.cwd() / DEFAULT_STORAGE_DIRNAME


def build_app(storage: StorageBackend | None = None,
              storage_dir: Path | str | None = None,
              db_path: Path | str | None = None) -> FastAPI:
    """Create the FastAPI application with injected dependencies."""
    _load_local_env()
    backend = storage or LocalStorageBackend(
        storage_dir or default_storage_dir())
    if db_path is None and isinstance(backend, LocalStorageBackend):
        db_path = backend.base_dir / "jobs.sqlite3"
    if db_path is None:
        raise ValueError("A job database path is required.")
    store = SqliteJobStore(db_path)
    service = JobService(store)
    app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.state.storage = backend
    app.state.jobs = service

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> dict[str, Any]:
        ready = backend.base_dir.is_dir() if isinstance(
            backend, LocalStorageBackend) else True
        return {
            "status": "ok",
            "service": SERVICE_NAME,
            "version": SERVICE_VERSION,
            "storage_backend": type(backend).__name__,
            "storage_ready": ready,
        }

    @app.post("/api/analyses", response_model=AnalysisCreated,
              status_code=202)
    async def create_analysis(
        file: Annotated[UploadFile, File()],
        workflow: Annotated[str, Form()] = WORKFLOW_STORAGE_SMOKE,
        grant: Annotated[str | None, Form()] = None,
    ) -> dict[str, Any]:
        grant_payload: dict[str, Any] | None = None
        if grant is not None:
            try:
                parsed = json.loads(grant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Grant is not valid JSON: {exc}",
                ) from exc
            if not isinstance(parsed, dict):
                raise HTTPException(
                    status_code=422,
                    detail="Grant must be a JSON object.")
            grant_payload = parsed
        content = await file.read()
        cap = (effective_video_max_bytes()
               if workflow in (WORKFLOW_VIDEO_PIPELINE_OFFLINE,
                               WORKFLOW_VIDEO_PIPELINE_META)
               else MAX_UPLOAD_BYTES)
        if len(content) > cap:
            raise HTTPException(
                status_code=413,
                detail=(f"Upload is {len(content)} bytes; "
                        f"the limit for this workflow is "
                        f"{cap} bytes. No job was created."),
            )
        object_id = sanitize_filename(file.filename)
        backend.store_source(object_id, io.BytesIO(content))
        try:
            record = service.submit(
                workflow, source_object_id=object_id,
                grant=grant_payload)
        except UnknownWorkflowError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except GrantError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {
            "run_id": record.run_id,
            "workflow": record.workflow,
            "status": record.status,
            "source_object_id": object_id,
        }

    @app.get("/api/analyses/{run_id}")
    def get_analysis(run_id: str) -> dict[str, Any]:
        try:
            validate_run_id(run_id)
        except InvalidObjectIdError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        record = service.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Unknown run id.")
        return record.to_dict()

    @app.get("/api/analyses/{run_id}/artifacts/{artifact_id}")
    def get_artifact(run_id: str, artifact_id: str) -> Response:
        try:
            validate_run_id(run_id)
            validate_object_id(artifact_id)
        except InvalidObjectIdError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        record = service.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Unknown run id.")
        if record.status == STATUS_FAILED:
            raise HTTPException(
                status_code=409,
                detail=f"Run failed: {record.error or 'unknown error'}.")
        try:
            data = backend.read_artifact(run_id, artifact_id)
        except ObjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return Response(content=data, media_type="application/json")

    def _require_completed(run_id: str) -> Any:
        try:
            validate_run_id(run_id)
        except InvalidObjectIdError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        record = service.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Unknown run id.")
        if record.status != STATUS_COMPLETED:
            raise HTTPException(
                status_code=409,
                detail=(f"Run is {record.status}; review requires "
                        "a completed run."))
        return record

    def _review_root() -> Path:
        if not isinstance(backend, LocalStorageBackend):
            raise HTTPException(
                status_code=501,
                detail="Review workspace requires local storage.")
        return backend.base_dir

    @app.get("/api/jobs")
    def list_jobs(status: str | None = None,
                  limit: int = 50) -> list[dict[str, Any]]:
        """Persisted processing queue across all statuses."""
        if status is not None and status not in (
                STATUS_QUEUED, STATUS_RUNNING,
                STATUS_COMPLETED, STATUS_FAILED):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid status filter: {status!r}.")
        if isinstance(limit, bool) or not 1 <= limit <= 200:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid list limit: {limit!r}.")
        try:
            records = service.list_jobs(status, limit)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return [record.to_dict() for record in records]

    @app.post("/api/jobs/{run_id}/retry")
    def retry_job(run_id: str) -> dict[str, Any]:
        """Requeue one failed non-billable job in place.

        Never invokes inference directly and never retries a
        billable workflow; the worker claims the job through
        the normal atomic path.
        """
        try:
            validate_run_id(run_id)
        except InvalidObjectIdError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        if service.get(run_id) is None:
            raise HTTPException(
                status_code=404, detail="Unknown run id.")
        try:
            record = service.retry(run_id)
        except RetryError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return record.to_dict()

    @app.get("/api/analyses")
    def list_analyses(limit: int = 50) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not 1 <= limit <= 200:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid list limit: {limit!r}.")
        try:
            records = service.list_recent(limit)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return [record.to_dict() for record in records]

    @app.get("/api/analyses/{run_id}/review")
    def get_review_bundle(run_id: str) -> dict[str, Any]:
        record = _require_completed(run_id)
        _review_root()
        assert isinstance(backend, LocalStorageBackend)
        try:
            return build_review_bundle(backend, record)
        except ReviewError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/analyses/{run_id}/frames/{frame_name}")
    def get_frame(run_id: str, frame_name: str) -> Response:
        _require_completed(run_id)
        root = _review_root()
        try:
            data = read_frame_bytes(root, run_id, frame_name)
        except InvalidObjectIdError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except ReviewError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except ObjectNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return Response(content=data, media_type="image/jpeg")

    @app.get("/api/analyses/{run_id}/observation-review")
    def get_observation_review(run_id: str) -> dict[str, Any]:
        record = _require_completed(run_id)
        _review_root()
        assert isinstance(backend, LocalStorageBackend)
        return RecipeReviewService(backend).observation_queue(record)

    @app.post("/api/analyses/{run_id}/observation-review/corrections")
    def save_observation_correction(
            run_id: str,
            request: ObservationCorrection) -> dict[str, Any]:
        record = _require_completed(run_id)
        _review_root()
        assert isinstance(backend, LocalStorageBackend)
        try:
            return RecipeReviewService(backend).save_observation_correction(
                record, request.task_id, request.action,
                request.human_value, request.reviewer)
        except ObservationReviewError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/api/analyses/{run_id}/observation-review/corrections/undo")
    def undo_observation_correction(
            run_id: str,
            request: ObservationUndo) -> dict[str, Any]:
        record = _require_completed(run_id)
        _review_root()
        assert isinstance(backend, LocalStorageBackend)
        try:
            return RecipeReviewService(backend).undo_observation_correction(
                record, request.task_id, request.reviewer)
        except ObservationReviewError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/api/analyses/{run_id}/observation-review/classifications")
    def save_observation_classification(
            run_id: str,
            request: ObservationClassification) -> dict[str, Any]:
        record = _require_completed(run_id)
        _review_root()
        assert isinstance(backend, LocalStorageBackend)
        try:
            return RecipeReviewService(backend).save_frame_classification(
                record, request.frame_index, request.action,
                request.human_value, request.reviewer)
        except ObservationReviewError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.get("/api/analyses/{run_id}/consolidated-evidence")
    def get_consolidated_evidence(run_id: str) -> dict[str, Any]:
        record = _require_completed(run_id)
        _review_root()
        assert isinstance(backend, LocalStorageBackend)
        return RecipeReviewService(backend).consolidated(record)

    @app.get("/api/analyses/{run_id}/culinary/observations")
    def get_culinary_observations(run_id: str) -> dict[str, Any]:
        record = _require_completed(run_id)
        _review_root()
        assert isinstance(backend, LocalStorageBackend)
        return RecipeReviewService(backend).culinary_status(record)

    @app.post("/api/analyses/{run_id}/culinary/enrich")
    def enrich_culinary_observations(
            run_id: str,
            request: CulinaryEnrichRequest) -> dict[str, Any]:
        from bakery_intelligence.adversarial.culinary import (
            CulinaryError,
        )
        from bakery_intelligence.adversarial.store import (
            RecipeReviewError,
        )
        record = _require_completed(run_id)
        _review_root()
        assert isinstance(backend, LocalStorageBackend)
        if (isinstance(request.max_frames, bool)
                or not 1 <= request.max_frames <= 51):
            raise HTTPException(
                status_code=422,
                detail=f"max_frames must be between 1 and 51, "
                       f"got {request.max_frames!r}.")
        if request.frame_indices is not None and (
                not isinstance(request.frame_indices, list)
                or len(request.frame_indices) > 51
                or any(isinstance(index, bool)
                       or not isinstance(index, int) or index < 0
                       for index in request.frame_indices)):
            raise HTTPException(
                status_code=422,
                detail="frame_indices must be a list of up to 51 "
                       "non-negative frame indices.")
        try:
            return RecipeReviewService(backend).enrich_culinary(
                record, max_frames=request.max_frames,
                confirmed=request.confirmed,
                frame_indices=request.frame_indices)
        except (CulinaryError, RecipeReviewError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/api/analyses/{run_id}/culinary/requeue")
    def requeue_culinary_frames(
            run_id: str,
            request: CulinaryRequeueRequest) -> dict[str, Any]:
        from bakery_intelligence.adversarial.culinary import (
            CulinaryError,
        )
        record = _require_completed(run_id)
        _review_root()
        assert isinstance(backend, LocalStorageBackend)
        try:
            return RecipeReviewService(backend).requeue_culinary_frames(
                record, frame_indices=request.frame_indices,
                reason=request.reason, reviewer=request.reviewer)
        except CulinaryError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.get("/api/analyses/{run_id}/temporal-events")
    def get_temporal_events(run_id: str) -> dict[str, Any]:
        record = _require_completed(run_id)
        _review_root()
        assert isinstance(backend, LocalStorageBackend)
        return RecipeReviewService(backend).temporal_events(record)

    @app.get("/api/analyses/{run_id}/recipe/readiness")
    def get_recipe_readiness(run_id: str) -> dict[str, Any]:
        record = _require_completed(run_id)
        _review_root()
        assert isinstance(backend, LocalStorageBackend)
        return RecipeReviewService(backend).readiness(record)

    @app.post("/api/analyses/{run_id}/recipe/generate")
    def generate_recipe(
            run_id: str,
            request: GenerateRecipeRequest | None = None,
    ) -> dict[str, Any]:
        record = _require_completed(run_id)
        _review_root()
        assert isinstance(backend, LocalStorageBackend)
        try:
            return RecipeReviewService(backend).generate_recipe(
                record, request or GenerateRecipeRequest())
        except LiveConfigError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except InferenceError as exc:
            raise HTTPException(status_code=502, detail=(
                f"Live provider error ({exc.code}); no simulated "
                "fallback was used."))

    @app.get("/api/analyses/{run_id}/recipe")
    def get_recipe_review(run_id: str) -> dict[str, Any]:
        record = _require_completed(run_id)
        root = _review_root()
        assert isinstance(backend, LocalStorageBackend)
        return RecipeReviewService(backend).get_or_create(record)

    @app.get("/api/analyses/{run_id}/recipe/evidence")
    def get_recipe_evidence(run_id: str) -> dict[str, Any]:
        record = _require_completed(run_id)
        _review_root()
        assert isinstance(backend, LocalStorageBackend)
        return RecipeReviewService(backend).evidence(record)

    @app.post("/api/analyses/{run_id}/recipe/corrections")
    def save_recipe_corrections(
            run_id: str,
            request: RecipeCorrections) -> dict[str, Any]:
        record = _require_completed(run_id)
        _review_root()
        assert isinstance(backend, LocalStorageBackend)
        service = RecipeReviewService(backend)
        service.get_or_create(record)
        try:
            return service.save_corrections(
                record,
                request.ingredients, request.steps,
                request.reviewer)
        except RecipeReviewError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/api/analyses/{run_id}/recipe/decisions")
    def save_recipe_decision(
            run_id: str,
            request: RecipeDecisionRequest) -> dict[str, Any]:
        record = _require_completed(run_id)
        _review_root()
        assert isinstance(backend, LocalStorageBackend)
        service = RecipeReviewService(backend)
        service.get_or_create(record)
        try:
            return service.store.decide(
                run_id, request.decision, request.reviewer,
                request.notes)
        except RecipeReviewError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.get("/api/analyses/{run_id}/reviews")
    def get_reviews(run_id: str) -> dict[str, Any]:
        _require_completed(run_id)
        root = _review_root()
        try:
            stored = read_reviews(root, run_id)
        except ReviewError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        try:
            ordered = [stored["annotations"][key]
                       for key in sorted(stored["annotations"],
                                         key=int)]
        except ValueError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Stored reviews are malformed: {exc}")
        return {
            "run_id": run_id,
            "schema_version": stored["schema_version"],
            "annotations": ordered,
        }

    @app.post("/api/analyses/{run_id}/reviews", status_code=200)
    def save_review_annotation(
            run_id: str, submission: ReviewSubmission) -> dict[str, Any]:
        _require_completed(run_id)
        root = _review_root()
        try:
            known = frame_exists(root, run_id, submission.frame_index)
        except InvalidObjectIdError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        if not known:
            raise HTTPException(
                status_code=404,
                detail=(f"Unknown frame index: "
                        f"{submission.frame_index}."))
        try:
            return save_review(
                root, run_id, submission.frame_index,
                submission.decision, submission.value,
                submission.notes, submission.reviewer)
        except ReviewError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    def _upload_root() -> Path:
        if not isinstance(backend, LocalStorageBackend):
            raise HTTPException(
                status_code=501,
                detail="Video uploads require local storage.")
        return backend.base_dir

    def _cost_estimate(frames: int) -> dict[str, Any] | None:
        if frames <= 0:
            return None
        try:
            raw = backend.read_artifact(
                DEMO_RUN_ID, FIRSTPASS_REPORT_ARTIFACT)
            report = json.loads(raw)
        except Exception:
            return None
        requests = (report.get("requests_submitted")
                    or report.get("requests_sent"))
        in_tok = report.get("input_tokens")
        out_tok = report.get("output_tokens")
        if (not isinstance(requests, (int, float)) or requests <= 0
                or not isinstance(in_tok, (int, float))
                or not isinstance(out_tok, (int, float))):
            return None
        per_request = (
            (in_tok / requests) * META_RATES_USD_PER_M[0]
            + (out_tok / requests) * META_RATES_USD_PER_M[1]) / 1e6
        return {
            "estimated_cost_usd": round(per_request * frames, 2),
            "frames": frames,
            "nonbinding": True,
            "basis": (
                f"Measured average of {int(requests)} completed "
                "Meta requests from the retained S5-C analysis at "
                "verified rates of USD 1.25/M input and "
                "USD 4.25/M output tokens. Actual billing may "
                "differ."),
        }

    def _read_upload(object_id: str) -> dict[str, Any]:
        root = _upload_root()
        try:
            validate_object_id(object_id)
        except InvalidObjectIdError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        sidecar = root / UPLOADS_DIRNAME / f"{object_id}.json"
        if not sidecar.is_file() or not backend.source_exists(
                object_id):
            raise HTTPException(
                status_code=404, detail="Unknown upload id.")
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Upload metadata unreadable: {exc}")

    @app.post("/api/uploads", status_code=201)
    async def create_upload(
            file: Annotated[UploadFile, File()]) -> dict[str, Any]:
        root = _upload_root()
        filename = (file.filename or "video.mp4")
        if Path(filename).suffix.lower() != ".mp4":
            raise HTTPException(
                status_code=422,
                detail="Only MP4 videos are supported here.")
        cap = effective_video_max_bytes()
        tmp_dir = root / UPLOAD_TMP_DIRNAME
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"upload-{os.getpid()}-{id(file)}.part"
        digest = hashlib.sha256()
        size = 0
        try:
            with tmp_path.open("wb") as handle:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > cap:
                        raise HTTPException(
                            status_code=413,
                            detail=(f"Video exceeds the "
                                    f"{cap}-byte upload limit."))
                    digest.update(chunk)
                    handle.write(chunk)
            try:
                props = probe_video(tmp_path)
            except VideoError as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"{exc.code}: {exc.message}")
            sha = digest.hexdigest()
            stem = sanitize_stem(Path(filename).stem)
            object_id = validate_object_id(
                f"upload_{sha[:16]}_{stem}.mp4")
            deduplicated = backend.source_exists(object_id)
            if not deduplicated:
                with tmp_path.open("rb") as handle:
                    stored_sha = backend.store_source(
                        object_id, handle)
                if stored_sha != sha:
                    raise HTTPException(
                        status_code=500,
                        detail="Stored upload failed verification.")
            duration = float(props["duration_seconds"])
            estimated = int(math.ceil(duration))
            metadata = {
                "upload_id": object_id,
                "file_name": Path(filename).name,
                "bytes": size,
                "sha256": sha,
                "duration_seconds": duration,
                "fps": float(props["fps"]),
                "frame_count": int(props["frame_count"]),
                "width": int(props["width"]),
                "height": int(props["height"]),
                "sample_fps": 1.0,
                "estimated_frames": estimated,
                "exceeds_frame_cap": estimated > PIPELINE_FRAME_CAP,
                "deduplicated": deduplicated,
                "status": "ready",
            }
            uploads_dir = root / UPLOADS_DIRNAME
            uploads_dir.mkdir(parents=True, exist_ok=True)
            (uploads_dir / f"{object_id}.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8", newline="\n")
            return {
                **metadata,
                "estimate": _cost_estimate(
                    min(estimated, PIPELINE_FRAME_CAP)),
            }
        finally:
            tmp_path.unlink(missing_ok=True)

    @app.get("/api/uploads/{object_id}")
    def get_upload(object_id: str) -> dict[str, Any]:
        metadata = _read_upload(object_id)
        return {
            **metadata,
            "estimate": _cost_estimate(min(
                int(metadata.get("estimated_frames", 0)),
                PIPELINE_FRAME_CAP)),
        }

    @app.post("/api/uploads/{object_id}/analyses", status_code=202)
    def create_upload_analysis(
            object_id: str,
            request: UploadAnalysisRequest) -> dict[str, Any]:
        metadata = _read_upload(object_id)
        size = metadata.get("bytes")
        limit = pipeline_max_bytes()
        if isinstance(size, int) and size > limit:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Video is {size} bytes; the pipeline "
                    f"limit is {limit} bytes. No job was "
                    "created."))
        if request.workflow not in (
                WORKFLOW_VIDEO_PIPELINE_OFFLINE,
                WORKFLOW_VIDEO_PIPELINE_META):
            raise HTTPException(
                status_code=422,
                detail=("Uploads start either the offline preview "
                        "or the Meta pipeline workflow."))
        try:
            record = service.submit(
                request.workflow, source_object_id=object_id,
                grant=request.grant)
        except UnknownWorkflowError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except GrantError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {
            "run_id": record.run_id,
            "workflow": record.workflow,
            "status": record.status,
            "source_object_id": object_id,
        }

    @app.get("/api/analyses/{run_id}/progress")
    def get_progress(run_id: str) -> dict[str, Any]:
        try:
            validate_run_id(run_id)
        except InvalidObjectIdError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        record = service.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Unknown run id.")
        stage = record.status
        frames_total: int | None = None
        frames_processed: int | None = None
        updated = record.updated_at
        if isinstance(backend, LocalStorageBackend):
            work = (backend.base_dir / "runs" / run_id / "work")
            state_path = work / "pipeline_state.json"
            if state_path.is_file():
                try:
                    state = json.loads(
                        state_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    state = None
                if isinstance(state, dict):
                    stage = str(state.get("stage") or stage)
                    updated = str(
                        state.get("updated_at_utc") or updated)
                    extraction = state.get("extraction")
                    if isinstance(extraction, dict):
                        indices = extraction.get("frame_indices")
                        if isinstance(indices, list):
                            frames_total = len(indices)
            obs_dir = work / "firstpass" / "observations"
            if obs_dir.is_dir():
                frames_processed = 0
                for obs_path in obs_dir.glob("*.json"):
                    try:
                        payload = json.loads(
                            obs_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    if isinstance(payload, list):
                        frames_processed += len(payload)
        percent = None
        if frames_total and frames_processed is not None:
            percent = round(
                100.0 * frames_processed / frames_total, 1)
        if record.status == STATUS_COMPLETED:
            percent = 100.0
        return {
            "run_id": record.run_id,
            "workflow": record.workflow,
            "status": record.status,
            "stage": stage,
            "frames_total": frames_total,
            "frames_processed": frames_processed,
            "percent": percent,
            "error": record.error,
            "updated_at": updated,
        }

    return app
