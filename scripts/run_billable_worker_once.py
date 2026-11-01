"""Run exactly one worker pass with .env-loaded credentials.

Ops launcher for the authorized billable smoke test. It fills
only the known credential names from a dotenv file into the
process environment (never overriding already-exported
values), reports credential *presence by name only*, and —
only with ``--run`` — executes the existing worker CLI in
``--once`` mode. Default invocation validates and exits
without launching anything.

Security properties:

- Credentials travel process-memory-only: file bytes to
  ``os.environ`` to the worker's existing lookup. They are
  never printed, never passed via argv, never persisted,
  and never stored in the report object (names only).
- Precedence is explicit and tested: within one name the
  process environment wins over the file; across names
  ``META_API_KEY`` wins over ``MODEL_API_KEY`` (the
  worker's documented lookup order).
- Missing credentials fail closed: without a key the
  launcher exits non-zero and the worker is never
  started, so no queued job can be failed spuriously.
- Importing this module has no side effects beyond the
  ``sys.path`` entry shared by every script in this
  directory: no environment reads, no worker imports.

Usage (from the repository root; never run unattended):

    uv run python scripts/run_billable_worker_once.py --check
    uv run python scripts/run_billable_worker_once.py --run
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, MutableMapping

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT / "src"))

from bakery_intelligence.api.authorization import ENV_KEYS


class LauncherError(ValueError):
    """Launcher usage or credential-availability failure."""


@dataclass(frozen=True)
class CredentialReport:
    """Name-only credential audit record (never holds values)."""

    env_file: str
    file_found: bool
    selected_name: str | None
    present_names: tuple[str, ...] = ()
    notices: tuple[str, ...] = field(default_factory=tuple)


def load_credentials(
    env_file: Path | None,
    environ: MutableMapping[str, str],
) -> CredentialReport:
    """Fill known credential names from a dotenv file.

    Only ``ENV_KEYS`` are read; every other variable in the
    file is ignored. Already-exported values always win over
    file values. Returns a name-only report.
    """
    path = env_file if env_file is not None else Path(".env")
    if env_file is not None and not path.is_file():
        raise LauncherError(f"Env file not found: {path}.")
    file_found = path.is_file()
    file_values: dict[str, str | None] = (
        dict(dotenv_values(path)) if file_found else {})
    origins: dict[str, str] = {}
    for name in ENV_KEYS:
        if environ.get(name):
            origins[name] = "process environment"
        elif file_values.get(name):
            environ[name] = str(file_values[name])
            origins[name] = str(path)
    present = tuple(name for name in ENV_KEYS if environ.get(name))
    selected = next(
        (name for name in ENV_KEYS if environ.get(name)), None)
    notices: list[str] = []
    if file_found:
        notices.append(f"env file found: {path}")
    else:
        notices.append(
            f"env file not found: {path}; process environment only")
    for name in present:
        notices.append(f"{name}: present (from {origins[name]})")
    if len(present) > 1 and selected is not None:
        shadowed = ", ".join(
            name for name in present if name != selected)
        notices.append(
            f"{selected} takes precedence; shadowed: {shadowed}")
    if selected is None:
        notices.append(
            "no credential found; refusing to start the worker")
    return CredentialReport(
        env_file=str(path), file_found=file_found,
        selected_name=selected, present_names=present,
        notices=tuple(notices))


def _worker_argv(args: argparse.Namespace) -> list[str]:
    """Build the exact worker CLI invocation (key-free)."""
    argv = ["--once", "--worker-id", args.worker_id]
    if args.storage_dir is not None:
        argv += ["--storage-dir", args.storage_dir]
    if args.db_path is not None:
        argv += ["--db-path", args.db_path]
    return argv


def main(
    argv: list[str] | None = None,
    *,
    environ: MutableMapping[str, str] | None = None,
    worker_main: Callable[[list[str]], int] | None = None,
) -> int:
    """Validate credentials; launch one worker pass on --run."""
    parser = argparse.ArgumentParser(
        description="Run one worker pass with .env credentials.")
    parser.add_argument("--env-file", type=Path, default=None,
                        help="Dotenv file (default: ./.env).")
    parser.add_argument("--check", action="store_true",
                        help="Validate only; never start a worker.")
    parser.add_argument("--run", action="store_true",
                        help="Required to start the worker pass.")
    parser.add_argument("--worker-id", default="worker-1")
    parser.add_argument("--storage-dir", default=None)
    parser.add_argument("--db-path", default=None)
    args = parser.parse_args(argv)
    if args.check and args.run:
        print("error: --check and --run are mutually exclusive",
              file=sys.stderr)
        return 2
    env = environ if environ is not None else os.environ
    try:
        report = load_credentials(args.env_file, env)
    except LauncherError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for notice in report.notices:
        print(notice)
    if report.selected_name is None:
        return 2
    if not args.run:
        return 0
    run_worker = worker_main
    if run_worker is None:
        from bakery_intelligence.api.worker import main as run_worker
    return run_worker(_worker_argv(args))


if __name__ == "__main__":
    raise SystemExit(main())
