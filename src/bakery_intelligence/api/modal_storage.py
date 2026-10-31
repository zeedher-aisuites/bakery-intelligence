"""Modal Volumes v2 storage adapter boundary (not yet implemented).

This module defines where a Modal-backed implementation of
``StorageBackend`` will live. It deliberately performs no Modal
import, no credential lookup, and no network access: the official
Volumes v2 API must be verified against
https://modal.com/docs/guide/volumes before any implementation or
invocation. See ``experiments/008_modal_volumes_spike/PROCEDURE.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from bakery_intelligence.api.storage import StorageBackend, StorageError


class ModalNotConfiguredError(StorageError):
    """Modal storage was requested but is not implemented/configured."""


class ModalStorageBackend(StorageBackend):
    """Placeholder adapter reserved for a verified Volumes v2 client.

    Every method raises ``ModalNotConfiguredError`` until the adapter
    is implemented against the verified official API. Constructing
    the placeholder is safe offline: it never imports ``modal``.
    """

    def __init__(self, volume_name: str) -> None:
        self.volume_name = volume_name

    def _unavailable(self) -> ModalNotConfiguredError:
        return ModalNotConfiguredError(
            "Modal storage is not implemented. Verify the official "
            "Volumes v2 API, then implement this adapter. Volume: "
            f"{self.volume_name!r}.")

    def store_source(self, object_id: str, stream: BinaryIO) -> str:
        raise self._unavailable()

    def source_exists(self, object_id: str) -> bool:
        raise self._unavailable()

    def open_source(self, object_id: str) -> BinaryIO:
        raise self._unavailable()

    def store_artifact(self, run_id: str, artifact_id: str,
                       data: bytes) -> str:
        raise self._unavailable()

    def artifact_exists(self, run_id: str, artifact_id: str) -> bool:
        raise self._unavailable()

    def read_artifact(self, run_id: str, artifact_id: str) -> bytes:
        raise self._unavailable()

    def run_work_dir(self, run_id: str) -> Path:
        raise self._unavailable()
