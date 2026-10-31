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

Run with ``uv run uvicorn bakery_intelligence.api.app:build_app
--factory`` (see docs/m206/spike-setup.md).
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from bakery_intelligence.api.authorization import GrantError
from bakery_intelligence.api.jobs import (
    STATUS_FAILED,
    JobService,
    UnknownWorkflowError,
    WORKFLOW_STORAGE_SMOKE,
)
from bakery_intelligence.api.storage import (
    InvalidObjectIdError,
    LocalStorageBackend,
    ObjectNotFoundError,
    StorageBackend,
    validate_object_id,
    validate_run_id,
)
from bakery_intelligence.api.store import SqliteJobStore

SERVICE_NAME = "bakery-intelligence-api"
SERVICE_VERSION = "0.1.0-spike"
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
DEFAULT_STORAGE_DIRNAME = "runs_api"

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
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(f"Upload exceeds {MAX_UPLOAD_BYTES} bytes; "
                        "large-file uploads are out of spike scope."),
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

    return app
