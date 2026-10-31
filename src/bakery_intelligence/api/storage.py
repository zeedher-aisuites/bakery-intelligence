"""Storage abstraction with local filesystem backend.

The API layer addresses every object by stable identifier, never by
filesystem path. Identifiers are restricted to a safe character set
and resolved strictly inside the configured storage root, which
eliminates path traversal by construction.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO

OBJECT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
RUN_ID_PATTERN = re.compile(r"run_[0-9a-f]{12}")
SHA256_HEX_PATTERN = re.compile(r"[0-9a-f]{64}")

SOURCES_DIRNAME = "sources"
RUNS_DIRNAME = "runs"
ARTIFACTS_DIRNAME = "artifacts"


class StorageError(Exception):
    """Base error for storage failures."""


class InvalidObjectIdError(StorageError):
    """An object, run, or artifact identifier failed validation."""


class ObjectNotFoundError(StorageError):
    """The requested stored object does not exist."""


def validate_object_id(object_id: str) -> str:
    """Accept a safe storage identifier or raise."""
    if not isinstance(object_id, str):
        raise InvalidObjectIdError("Object id must be a string.")
    if OBJECT_ID_PATTERN.fullmatch(object_id) is None:
        raise InvalidObjectIdError(
            f"Invalid object id: {object_id!r}.")
    if ".." in object_id or object_id.startswith((".", "-", "_")):
        raise InvalidObjectIdError(
            f"Invalid object id: {object_id!r}.")
    return object_id


def validate_run_id(run_id: str) -> str:
    """Accept a run identifier of the expected shape or raise."""
    if not isinstance(run_id, str):
        raise InvalidObjectIdError("Run id must be a string.")
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise InvalidObjectIdError(f"Invalid run id: {run_id!r}.")
    return run_id


def sha256_hex(data: bytes) -> str:
    """Return the SHA-256 hex digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 65536) -> str:
    """Return the SHA-256 hex digest of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class StorageBackend(ABC):
    """Contract every storage adapter must satisfy."""

    @abstractmethod
    def store_source(self, object_id: str, stream: BinaryIO) -> str:
        """Persist an uploaded source file; return its SHA-256 hex."""

    @abstractmethod
    def source_exists(self, object_id: str) -> bool:
        """Report whether a source object exists."""

    @abstractmethod
    def open_source(self, object_id: str) -> BinaryIO:
        """Open a source object for reading (caller closes)."""

    @abstractmethod
    def store_artifact(self, run_id: str, artifact_id: str,
                       data: bytes) -> str:
        """Persist a run artifact; return its SHA-256 hex."""

    @abstractmethod
    def artifact_exists(self, run_id: str, artifact_id: str) -> bool:
        """Report whether a run artifact exists."""

    @abstractmethod
    def read_artifact(self, run_id: str, artifact_id: str) -> bytes:
        """Read a run artifact fully into memory."""


class LocalStorageBackend(StorageBackend):
    """Filesystem storage rooted at a dedicated base directory."""

    def __init__(self, base_dir: Path | str) -> None:
        self._base = Path(base_dir).resolve()
        (self._base / SOURCES_DIRNAME).mkdir(parents=True, exist_ok=True)
        (self._base / RUNS_DIRNAME).mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        """Return the resolved storage root."""
        return self._base

    def _source_path(self, object_id: str) -> Path:
        validate_object_id(object_id)
        return self._base / SOURCES_DIRNAME / object_id

    def _artifact_path(self, run_id: str, artifact_id: str) -> Path:
        validate_run_id(run_id)
        validate_object_id(artifact_id)
        directory = self._base / RUNS_DIRNAME / run_id / ARTIFACTS_DIRNAME
        directory.mkdir(parents=True, exist_ok=True)
        return directory / artifact_id

    def store_source(self, object_id: str, stream: BinaryIO) -> str:
        path = self._source_path(object_id)
        digest = hashlib.sha256()
        with path.open("wb") as handle:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                digest.update(chunk)
                handle.write(chunk)
        return digest.hexdigest()

    def source_exists(self, object_id: str) -> bool:
        return self._source_path(object_id).is_file()

    def open_source(self, object_id: str) -> BinaryIO:
        path = self._source_path(object_id)
        if not path.is_file():
            raise ObjectNotFoundError(f"Source not found: {object_id!r}.")
        return path.open("rb")

    def store_artifact(self, run_id: str, artifact_id: str,
                       data: bytes) -> str:
        path = self._artifact_path(run_id, artifact_id)
        path.write_bytes(data)
        return sha256_hex(data)

    def artifact_exists(self, run_id: str, artifact_id: str) -> bool:
        return self._artifact_path(run_id, artifact_id).is_file()

    def read_artifact(self, run_id: str, artifact_id: str) -> bytes:
        path = self._artifact_path(run_id, artifact_id)
        if not path.is_file():
            raise ObjectNotFoundError(
                f"Artifact not found: {run_id}/{artifact_id}.")
        return path.read_bytes()
