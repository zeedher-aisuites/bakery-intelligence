"""Offline tests for the storage abstraction and local backend."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from bakery_intelligence.api.modal_storage import (
    ModalNotConfiguredError,
    ModalStorageBackend,
)
from bakery_intelligence.api.storage import (
    InvalidObjectIdError,
    LocalStorageBackend,
    ObjectNotFoundError,
    sha256_hex,
    validate_object_id,
    validate_run_id,
)


def test_validate_object_id_accepts_safe_names() -> None:
    assert validate_object_id("clip01.mp4") == "clip01.mp4"
    assert validate_object_id("a") == "a"


def test_validate_object_id_rejects_traversal() -> None:
    for bad in ("", "../x", "..", ".hidden", "-dash", "_under",
                "a/b", "a\\b", "a..b", "x" * 200, "sp ace", "semi;colon"):
        with pytest.raises(InvalidObjectIdError):
            validate_object_id(bad)


def test_validate_run_id_shape() -> None:
    assert validate_run_id("run_0123456789ab") == "run_0123456789ab"
    for bad in ("", "run_123", "RUN_0123456789ab", "run_0123456789ag",
                "../run_0123456789ab"):
        with pytest.raises(InvalidObjectIdError):
            validate_run_id(bad)


def test_local_storage_source_roundtrip(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    digest = backend.store_source("clip01.mp4", io.BytesIO(b"fake-video"))
    assert digest == sha256_hex(b"fake-video")
    assert backend.source_exists("clip01.mp4")
    assert not backend.source_exists("missing.mp4")
    with backend.open_source("clip01.mp4") as handle:
        assert handle.read() == b"fake-video"
    with pytest.raises(ObjectNotFoundError):
        backend.open_source("missing.mp4")


def test_local_storage_artifact_roundtrip(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    run_id = "run_0123456789ab"
    digest = backend.store_artifact(run_id, "result.json", b"{}")
    assert digest == sha256_hex(b"{}")
    assert backend.artifact_exists(run_id, "result.json")
    assert backend.read_artifact(run_id, "result.json") == b"{}"
    with pytest.raises(ObjectNotFoundError):
        backend.read_artifact(run_id, "absent.json")


def test_local_storage_rejects_bad_ids(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    with pytest.raises(InvalidObjectIdError):
        backend.store_source("../evil", io.BytesIO(b"x"))
    with pytest.raises(InvalidObjectIdError):
        backend.read_artifact("bogus", "result.json")


def test_modal_placeholder_never_touches_network(tmp_path: Path) -> None:
    backend = ModalStorageBackend("test-volume")
    assert backend.volume_name == "test-volume"
    with pytest.raises(ModalNotConfiguredError):
        backend.store_source("x.mp4", io.BytesIO(b"x"))
    with pytest.raises(ModalNotConfiguredError):
        backend.source_exists("x.mp4")
    with pytest.raises(ModalNotConfiguredError):
        backend.open_source("x.mp4")
    with pytest.raises(ModalNotConfiguredError):
        backend.store_artifact("run_0123456789ab", "r.json", b"{}")
    with pytest.raises(ModalNotConfiguredError):
        backend.artifact_exists("run_0123456789ab", "r.json")
    with pytest.raises(ModalNotConfiguredError):
        backend.read_artifact("run_0123456789ab", "r.json")
    with pytest.raises(ModalNotConfiguredError):
        backend.run_work_dir("run_0123456789ab")


def test_run_work_dir_created_and_namespaced(tmp_path: Path) -> None:
    backend = LocalStorageBackend(tmp_path)
    run_id = "run_0123456789ab"
    path = backend.run_work_dir(run_id)
    assert path == tmp_path / "runs" / run_id / "work"
    assert path.is_dir()
    assert backend.run_work_dir(run_id) == path
    (path / "progress.json").write_bytes(b"{}")
    assert not backend.artifact_exists(run_id, "progress.json")
    with pytest.raises(ObjectNotFoundError):
        backend.read_artifact(run_id, "progress.json")
    assert LocalStorageBackend(tmp_path).run_work_dir(
        run_id).is_dir()
    with pytest.raises(InvalidObjectIdError):
        backend.run_work_dir("bogus")
