"""Isolated tests for the local backup utility (scripts/backup_data.py).

All data is synthetic and lives in temporary directories; real
analysis artifacts are never touched.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "backup_data.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "backup_data", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


backup = _load_module()


def _make_source(root: Path) -> Path:
    source = root / "runs_api"
    (source / "sources").mkdir(parents=True)
    (source / "sources" / "video.mp4").write_bytes(
        b"video-bytes" * 10000)
    (source / "uploads").mkdir()
    (source / "uploads" / "video.json").write_text(
        json.dumps({"sha256": "abc"}), encoding="utf-8")
    run_dir = source / "runs" / "run_0123456789ab"
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "artifacts" / "report.json").write_text(
        json.dumps({"ok": True}), encoding="utf-8")
    (run_dir / "work" / "frames").mkdir(parents=True)
    (run_dir / "work" / "frames" / "frame_00000000.jpg").write_bytes(
        b"jpeg" * 100)
    (source / "reviews").mkdir()
    (source / "reviews" / "run_0123456789ab.json").write_text(
        json.dumps({"annotations": {}}), encoding="utf-8")
    (source / ".env").write_text("SECRET=1", encoding="utf-8")
    tmp_dir = source / "uploads_tmp"
    tmp_dir.mkdir()
    (tmp_dir / "leftover.part").write_bytes(b"partial")
    conn = sqlite3.connect(str(source / "jobs.sqlite3"))
    conn.execute(
        "CREATE TABLE jobs (run_id TEXT, workflow TEXT, "
        "status TEXT, created_at TEXT)")
    conn.execute(
        "INSERT INTO jobs VALUES ('run_0123456789ab', "
        "'video_pipeline_meta', 'completed', '2026-01-01')")
    conn.commit()
    conn.close()
    return source


def _blob_for(dest: Path, manifest: dict, rel: str) -> Path:
    entry = next(e for e in manifest["files"] if e["path"] == rel)
    return dest / entry["blob"]


def test_initial_full_backup(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    dest = tmp_path / "backups"
    manifest = backup.create_backup(source, dest)
    assert manifest["status"] == "completed"
    paths = {e["path"] for e in manifest["files"]}
    assert "sources/video.mp4" in paths
    assert "uploads/video.json" in paths
    assert "runs/run_0123456789ab/artifacts/report.json" in paths
    assert "runs/run_0123456789ab/work/frames/frame_00000000.jpg" \
        in paths
    assert "reviews/run_0123456789ab.json" in paths
    assert "jobs.sqlite3" in paths
    db_entry = next(e for e in manifest["files"]
                    if e["path"] == "jobs.sqlite3")
    assert db_entry["kind"] == "sqlite-snapshot"
    assert not any(".env" in p or p.endswith(".part") for p in paths)
    assert backup.verify_snapshot(dest,
                                  manifest["snapshot_id"]) == []


def test_sqlite_snapshot_is_consistent(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    dest = tmp_path / "backups"
    manifest = backup.create_backup(source, dest)
    blob = _blob_for(dest, manifest, "jobs.sqlite3")
    conn = sqlite3.connect(f"file:{blob}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT run_id, status FROM jobs").fetchall()
    finally:
        conn.close()
    assert rows == [("run_0123456789ab", "completed")]


def test_incremental_dedupes_immutable_and_versions_mutable(
        tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    dest = tmp_path / "backups"
    first = backup.create_backup(source, dest)
    big_blob = _blob_for(dest, first, "sources/video.mp4")
    (source / "reviews" / "run_0123456789ab.json").write_text(
        json.dumps({"annotations": {"0": {"decision": "corrected"}}}),
        encoding="utf-8")
    second = backup.create_backup(source, dest, reason="test-2")
    assert second["snapshot_id"] != first["snapshot_id"]
    # The multi-KB video blob is reused, not recopied.
    assert _blob_for(dest, second, "sources/video.mp4") == big_blob
    assert second["totals"]["reused_blobs"] >= 1
    # Only the changed review is new; even the SQLite snapshot
    # deduplicates when the database content did not change.
    assert second["totals"]["new_blobs"] == 1
    first_review = _blob_for(
        dest, first, "reviews/run_0123456789ab.json")
    second_review = _blob_for(
        dest, second, "reviews/run_0123456789ab.json")
    assert first_review != second_review
    assert first_review.is_file() and second_review.is_file()


def test_incomplete_snapshot_is_reported_and_unusable(
        tmp_path: Path) -> None:
    dest = tmp_path / "backups"
    partial = dest / "snapshots" / "20990101T000000Z"
    partial.mkdir(parents=True)
    (partial / backup.PARTIAL_MARKER).write_text(
        "in progress", encoding="utf-8")
    rows = backup.list_snapshots(dest)
    assert rows[0]["status"] == "incomplete"
    with pytest.raises(backup.BackupError):
        backup.load_manifest(dest, "20990101T000000Z")
    with pytest.raises(backup.BackupError):
        backup.restore_snapshot(
            dest, "20990101T000000Z", tmp_path / "out")


def test_verify_detects_tampered_blob(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    dest = tmp_path / "backups"
    manifest = backup.create_backup(source, dest)
    blob = _blob_for(dest, manifest, "sources/video.mp4")
    blob.write_bytes(b"tampered")
    failures = backup.verify_snapshot(dest, manifest["snapshot_id"])
    assert failures and "video.mp4" in failures[0]


def test_restore_into_separate_directory(tmp_path: Path) -> None:
    source = _make_source(tmp_path)
    dest = tmp_path / "backups"
    manifest = backup.create_backup(source, dest)
    target = tmp_path / "restored"
    count = backup.restore_snapshot(
        dest, manifest["snapshot_id"], target)
    assert count == manifest["totals"]["files"]
    assert (target / "sources" / "video.mp4").read_bytes() == \
        b"video-bytes" * 10000
    expected = hashlib.sha256(
        (target / "sources" / "video.mp4").read_bytes()).hexdigest()
    entry = next(e for e in manifest["files"]
                 if e["path"] == "sources/video.mp4")
    assert expected == entry["sha256"]
    assert (target / "jobs.sqlite3").is_file()
    assert not (target / ".env").exists()
    # Non-empty target requires explicit overwrite + confirmation.
    with pytest.raises(backup.BackupError):
        backup.restore_snapshot(
            dest, manifest["snapshot_id"], target)
    with pytest.raises(backup.BackupError):
        backup.restore_snapshot(
            dest, manifest["snapshot_id"], target, overwrite=True)
    assert backup.restore_snapshot(
        dest, manifest["snapshot_id"], target,
        overwrite=True, confirmed=True) == count
