"""Local backup system for Bakery Intelligence application data.

Creates content-addressed, verifiable snapshots of the API storage
tree (``runs_api`` by default) into a sibling backup directory
outside the Git repository. Standard library only.

Layout under the backup destination:

- ``objects/<sha256[:2]>/<sha256>``: immutable file blobs, stored
  once per content hash (large videos are never duplicated).
- ``snapshots/<snapshot_id>/manifest.json``: the snapshot record,
  written atomically as the final step. A snapshot directory
  without a manifest is an interrupted attempt and is reported
  as incomplete, never as a usable backup.

SQLite databases are captured through the SQLite backup API, so a
snapshot never contains a torn copy of a live database. Secrets
(``.env`` files), temporary upload parts, and caches are excluded.

Usage (from the repository root):

    .venv\\Scripts\\python.exe scripts/backup_data.py run
    .venv\\Scripts\\python.exe scripts/backup_data.py list
    .venv\\Scripts\\python.exe scripts/backup_data.py verify
    .venv\\Scripts\\python.exe scripts/backup_data.py trigger
    .venv\\Scripts\\python.exe scripts/backup_data.py restore \\
        --snapshot <id> --target <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

MANIFEST_FORMAT = "bakery-backup-manifest/1"
DEFAULT_SOURCE_DIRNAME = "runs_api"
BACKUP_DIRNAME = "bakery-intelligence-backups"
PARTIAL_MARKER = "snapshot.partial"
SKIP_DIRNAMES = {"uploads_tmp", "__pycache__", ".git", "node_modules"}
SKIP_SUFFIXES = {".part", ".pyc"}
DATABASE_NAME = "jobs.sqlite3"
_CHUNK = 4 * 1024 * 1024


class BackupError(RuntimeError):
    """A backup or restore operation failed safely."""


def default_source() -> Path:
    """API storage root under the repository."""
    return Path(__file__).resolve().parents[1] / DEFAULT_SOURCE_DIRNAME


def default_destination() -> Path:
    """Sibling backup directory outside the Git repository."""
    repo = Path(__file__).resolve().parents[1]
    return repo.parent / BACKUP_DIRNAME


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _snapshot_id(now: datetime) -> str:
    return now.strftime("%Y%m%dT%H%M%SZ")


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return digest.hexdigest(), size


def _is_secret_name(name: str) -> bool:
    return name == ".env" or name.startswith(".env.")


def iter_source_files(source: Path) -> Iterable[Path]:
    """Yield backup-worthy files under the storage root."""
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(source)
        if any(part in SKIP_DIRNAMES for part in rel.parts):
            continue
        if _is_secret_name(path.name):
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        if path.name in (DATABASE_NAME, f"{DATABASE_NAME}-wal",
                         f"{DATABASE_NAME}-shm"):
            continue
        yield path


def _active_jobs(source: Path) -> list[dict[str, Any]]:
    db_path = source / DATABASE_NAME
    if not db_path.is_file():
        return []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT run_id, workflow, status FROM jobs "
                "WHERE status IN ('queued', 'running')").fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return []
    return [{"run_id": r[0], "workflow": r[1], "status": r[2]}
            for r in rows]


def _completed_runs(source: Path) -> list[str]:
    db_path = source / DATABASE_NAME
    if not db_path.is_file():
        return []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT run_id FROM jobs WHERE status = 'completed' "
                "ORDER BY run_id").fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return []
    return [r[0] for r in rows]


def _snapshot_sqlite(source: Path, snapshot_dir: Path) -> Path:
    """Consistent database copy via the SQLite backup API."""
    db_path = source / DATABASE_NAME
    target_dir = snapshot_dir / "databases"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / DATABASE_NAME
    src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(str(target))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    return target


def _store_blob(dest: Path, sha: str, data_path: Path,
                size: int) -> bool:
    """Store one blob atomically; return True when newly written."""
    blob = dest / "objects" / sha[:2] / sha
    if blob.is_file() and blob.stat().st_size == size:
        return False
    blob.parent.mkdir(parents=True, exist_ok=True)
    tmp = blob.with_name(f".{sha}.tmp-{os.getpid()}")
    with data_path.open("rb") as src, tmp.open("wb") as out:
        shutil.copyfileobj(src, out, _CHUNK)
    os.replace(tmp, blob)
    return True


def create_backup(source: Path, dest: Path,
                  reason: str = "manual") -> dict[str, Any]:
    """Create one verified snapshot; return its manifest.

    Original data is only read. Any failure leaves the source
    untouched and the snapshot directory marked partial.
    """
    source = Path(source)
    dest = Path(dest)
    if not source.is_dir():
        raise BackupError(f"Source storage not found: {source}")
    now = _utc_now()
    snapshot_id = _snapshot_id(now)
    snapshots_dir = dest / "snapshots"
    snapshot_dir = snapshots_dir / snapshot_id
    counter = 1
    while snapshot_dir.exists():
        counter += 1
        snapshot_dir = snapshots_dir / f"{snapshot_id}-{counter}"
        snapshot_id = snapshot_dir.name
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / PARTIAL_MARKER).write_text(
        "backup in progress\n", encoding="utf-8")
    active = _active_jobs(source)
    entries: list[dict[str, Any]] = []
    new_blobs = 0
    reused_blobs = 0
    total_bytes = 0
    try:
        db_snapshot: Path | None = None
        if (source / DATABASE_NAME).is_file():
            db_snapshot = _snapshot_sqlite(source, snapshot_dir)
        paths = list(iter_source_files(source))
        if db_snapshot is not None:
            paths.append(db_snapshot)
        for path in paths:
            if db_snapshot is not None and path == db_snapshot:
                rel = DATABASE_NAME
                kind = "sqlite-snapshot"
            else:
                rel = path.relative_to(source).as_posix()
                kind = "file"
            sha, size = _sha256_file(path)
            written = _store_blob(dest, sha, path, size)
            if written:
                new_blobs += 1
            else:
                reused_blobs += 1
            total_bytes += size
            entries.append({
                "path": rel, "size": size, "sha256": sha,
                "blob": f"objects/{sha[:2]}/{sha}", "kind": kind,
                "status": "stored",
            })
        if db_snapshot is not None:
            db_snapshot.unlink(missing_ok=True)
            (snapshot_dir / "databases").rmdir()
    except Exception as exc:
        raise BackupError(
            f"Backup failed during copy: {exc}. Snapshot "
            f"{snapshot_id} is incomplete; original data is "
            "unchanged.") from exc
    manifest: dict[str, Any] = {
        "format": MANIFEST_FORMAT,
        "snapshot_id": snapshot_id,
        "created_at": now.isoformat(),
        "reason": reason,
        "status": "completed",
        "source_dirname": source.name,
        "active_jobs_at_backup": active,
        "completed_runs": _completed_runs(source),
        "files": sorted(entries, key=lambda e: e["path"]),
        "totals": {
            "files": len(entries),
            "bytes": total_bytes,
            "new_blobs": new_blobs,
            "reused_blobs": reused_blobs,
        },
    }
    tmp_manifest = snapshot_dir / "manifest.json.tmp"
    tmp_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")
    os.replace(tmp_manifest, snapshot_dir / "manifest.json")
    (snapshot_dir / PARTIAL_MARKER).unlink(missing_ok=True)
    return manifest


def list_snapshots(dest: Path) -> list[dict[str, Any]]:
    """List snapshots with completed/incomplete status."""
    snapshots_dir = Path(dest) / "snapshots"
    result: list[dict[str, Any]] = []
    if not snapshots_dir.is_dir():
        return result
    for entry in sorted(snapshots_dir.iterdir()):
        if not entry.is_dir():
            continue
        manifest_path = entry / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8"))
                result.append({
                    "snapshot_id": entry.name,
                    "status": manifest.get("status", "completed"),
                    "created_at": manifest.get("created_at"),
                    "files": manifest.get("totals", {}).get("files"),
                    "bytes": manifest.get("totals", {}).get("bytes"),
                })
                continue
            except (OSError, ValueError):
                pass
        result.append({
            "snapshot_id": entry.name, "status": "incomplete",
            "created_at": None, "files": None, "bytes": None,
        })
    return result


def load_manifest(dest: Path, snapshot_id: str) -> dict[str, Any]:
    """Load a completed snapshot manifest or fail safely."""
    path = Path(dest) / "snapshots" / snapshot_id / "manifest.json"
    if not path.is_file():
        raise BackupError(
            f"Snapshot {snapshot_id!r} has no manifest; it is "
            "incomplete or unknown and cannot be used.")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BackupError(f"Manifest unreadable: {exc}") from exc
    if manifest.get("format") != MANIFEST_FORMAT:
        raise BackupError("Manifest format is not recognized.")
    if manifest.get("status") != "completed":
        raise BackupError("Snapshot is not marked completed.")
    return manifest


def latest_snapshot_id(dest: Path) -> str | None:
    """Newest completed snapshot id, or None."""
    done = [row for row in list_snapshots(dest)
            if row["status"] == "completed"]
    return done[-1]["snapshot_id"] if done else None


def verify_snapshot(dest: Path,
                    snapshot_id: str) -> list[str]:
    """Rehash every referenced blob; return failure strings."""
    dest = Path(dest)
    manifest = load_manifest(dest, snapshot_id)
    failures: list[str] = []
    for entry in manifest["files"]:
        blob = dest / entry["blob"]
        if not blob.is_file():
            failures.append(f"missing blob for {entry['path']}")
            continue
        sha, size = _sha256_file(blob)
        if size != entry["size"] or sha != entry["sha256"]:
            failures.append(f"hash mismatch for {entry['path']}")
    return failures


def restore_snapshot(dest: Path, snapshot_id: str, target: Path,
                     overwrite: bool = False,
                     confirmed: bool = False) -> int:
    """Materialize a snapshot into target; return file count.

    Refuses incomplete snapshots and non-empty targets unless
    overwrite was explicitly confirmed. Every restored file is
    hash-verified after writing.
    """
    dest = Path(dest)
    target = Path(target)
    manifest = load_manifest(dest, snapshot_id)
    if target.exists() and any(target.iterdir()):
        if not overwrite:
            raise BackupError(
                f"Restore target is not empty: {target}. Choose "
                "an empty directory or pass --overwrite.")
        if not confirmed:
            raise BackupError(
                "Overwriting existing data requires explicit "
                "confirmation (--yes).")
    restored = 0
    for entry in manifest["files"]:
        blob = dest / entry["blob"]
        if not blob.is_file():
            raise BackupError(
                f"Missing blob for {entry['path']}; aborting "
                "restore without further changes.")
        out_path = target / entry["path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(f".{out_path.name}.restore-tmp")
        with blob.open("rb") as src, tmp.open("wb") as out:
            shutil.copyfileobj(src, out, _CHUNK)
        sha, size = _sha256_file(tmp)
        if sha != entry["sha256"] or size != entry["size"]:
            tmp.unlink(missing_ok=True)
            raise BackupError(
                f"Restored file failed verification: "
                f"{entry['path']}")
        os.replace(tmp, out_path)
        restored += 1
    return restored


def _fmt_bytes(size: int | None) -> str:
    if size is None:
        return "?"
    return f"{size / 1e9:.2f} GB ({size} bytes)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bakery Intelligence local backup utility.")
    parser.add_argument("--source", type=Path,
                        default=default_source(),
                        help="API storage root (default: runs_api).")
    parser.add_argument("--dest", type=Path,
                        default=default_destination(),
                        help="Backup destination outside the repo.")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="Create a backup now.")
    run_p.add_argument("--reason", default="manual")
    sub.add_parser("list", help="List snapshots and status.")
    verify_p = sub.add_parser("verify", help="Rehash a snapshot.")
    verify_p.add_argument("--snapshot", default=None,
                          help="Snapshot id (default: latest).")
    sub.add_parser(
        "trigger",
        help="Back up only when a new analysis completed since "
        "the last completed snapshot (safe post-analysis hook).")
    restore_p = sub.add_parser("restore", help="Restore a snapshot.")
    restore_p.add_argument("--snapshot", required=True)
    restore_p.add_argument("--target", type=Path, required=True)
    restore_p.add_argument("--overwrite", action="store_true")
    restore_p.add_argument("--yes", action="store_true",
                           help="Confirm overwriting existing data.")
    args = parser.parse_args(argv)
    source: Path = args.source
    dest: Path = args.dest
    try:
        if args.command == "run":
            manifest = create_backup(source, dest,
                                     reason=args.reason)
            print(f"snapshot {manifest['snapshot_id']} completed: "
                  f"{manifest['totals']['files']} files, "
                  f"{_fmt_bytes(manifest['totals']['bytes'])}, "
                  f"{manifest['totals']['new_blobs']} new blobs, "
                  f"{manifest['totals']['reused_blobs']} reused")
            if manifest["active_jobs_at_backup"]:
                print("warning: jobs were active during backup: "
                      + ", ".join(
                          j["run_id"] for j in
                          manifest["active_jobs_at_backup"]))
            failures = verify_snapshot(dest,
                                       manifest["snapshot_id"])
            if failures:
                print("verification FAILED: "
                      + "; ".join(failures), file=sys.stderr)
                return 1
            print("verification: all blob hashes match manifest")
            return 0
        if args.command == "list":
            rows = list_snapshots(dest)
            if not rows:
                print("no snapshots found")
            for row in rows:
                print(f"{row['snapshot_id']}  {row['status']}  "
                      f"files={row['files']}  "
                      f"size={_fmt_bytes(row['bytes'])}")
            return 0
        if args.command == "verify":
            snapshot_id = args.snapshot or latest_snapshot_id(dest)
            if snapshot_id is None:
                print("no completed snapshot to verify")
                return 1
            failures = verify_snapshot(dest, snapshot_id)
            if failures:
                print("verification FAILED: "
                      + "; ".join(failures), file=sys.stderr)
                return 1
            print(f"snapshot {snapshot_id}: all hashes verified")
            return 0
        if args.command == "trigger":
            latest = latest_snapshot_id(dest)
            known: set[str] = set()
            if latest is not None:
                known = set(load_manifest(
                    dest, latest).get("completed_runs", []))
            current = set(_completed_runs(source))
            if latest is not None and not (current - known):
                print("no new completed analyses; backup skipped")
                return 0
            manifest = create_backup(source, dest,
                                     reason="after-analysis")
            print(f"snapshot {manifest['snapshot_id']} completed "
                  "(trigger: completed analysis)")
            return 0
        if args.command == "restore":
            count = restore_snapshot(
                dest, args.snapshot, args.target,
                overwrite=args.overwrite, confirmed=args.yes)
            print(f"restored {count} files from snapshot "
                  f"{args.snapshot} into {args.target}")
            return 0
    except BackupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

