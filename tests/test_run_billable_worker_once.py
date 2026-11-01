"""Offline security tests for the billable worker launcher.

Loads ``scripts/run_billable_worker_once.py`` in isolation and
proves credential handling with synthetic keys only: explicit
precedence, fail-closed absence, name-only output, no worker
launch on import or validation, and exact worker argv. No
network, no SDK construction, no paid calls.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from bakery_intelligence.api.authorization import resolve_api_key

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = REPO_ROOT / "scripts" / "run_billable_worker_once.py"


def _load_launcher(name: str = "billable_launcher"):
    spec = importlib.util.spec_from_file_location(name, LAUNCHER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The import machinery normally registers the module before
    # execution; dataclass processing requires it (it resolves
    # sys.modules[__module__]). Register first, and remove the
    # entry if execution fails so no half-initialized module
    # lingers.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


launcher = _load_launcher()

META_KEY = "sk-test-launcher-meta-aaa"
MODEL_KEY = "sk-test-launcher-model-bbb"


def _write_env(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_valid_env_reaches_lookup(tmp_path: Path) -> None:
    env_path = _write_env(
        tmp_path / ".env",
        f"META_API_KEY={META_KEY}\nUNRELATED=1\nEMPTY=\n")
    environ: dict[str, str] = {}
    report = launcher.load_credentials(env_path, environ)
    assert environ == {"META_API_KEY": META_KEY}
    assert report.selected_name == "META_API_KEY"
    assert report.present_names == ("META_API_KEY",)
    assert resolve_api_key(environ) == META_KEY
    assert META_KEY not in repr(report)


def test_explicit_missing_file_fails(tmp_path: Path) -> None:
    with pytest.raises(launcher.LauncherError, match="not found"):
        launcher.load_credentials(tmp_path / "absent.env", {})


def test_missing_credential_reports_without_values(
        tmp_path: Path) -> None:
    env_path = _write_env(tmp_path / ".env", "UNRELATED=1\n")
    environ: dict[str, str] = {}
    report = launcher.load_credentials(env_path, environ)
    assert report.selected_name is None
    assert report.present_names == ()
    assert environ == {}
    assert any("no credential" in notice
               for notice in report.notices)


def test_precedence_file_both_names(tmp_path: Path) -> None:
    env_path = _write_env(
        tmp_path / ".env",
        f"META_API_KEY={META_KEY}\nMODEL_API_KEY={MODEL_KEY}\n")
    environ: dict[str, str] = {}
    report = launcher.load_credentials(env_path, environ)
    assert report.selected_name == "META_API_KEY"
    assert resolve_api_key(environ) == META_KEY
    assert any("shadowed" in notice for notice in report.notices)


def test_precedence_env_model_yields_to_file_meta(
        tmp_path: Path) -> None:
    env_path = _write_env(
        tmp_path / ".env", f"META_API_KEY={META_KEY}\n")
    environ: dict[str, str] = {"MODEL_API_KEY": MODEL_KEY}
    report = launcher.load_credentials(env_path, environ)
    assert report.selected_name == "META_API_KEY"
    assert resolve_api_key(environ) == META_KEY


def test_precedence_env_wins_within_name(tmp_path: Path) -> None:
    env_path = _write_env(
        tmp_path / ".env", "META_API_KEY=sk-test-file-shadowed\n")
    environ: dict[str, str] = {"META_API_KEY": META_KEY}
    report = launcher.load_credentials(env_path, environ)
    assert environ["META_API_KEY"] == META_KEY
    assert report.selected_name == "META_API_KEY"
    assert resolve_api_key(environ) == META_KEY


def test_fallback_model_key_selected(tmp_path: Path) -> None:
    env_path = _write_env(
        tmp_path / ".env", f"MODEL_API_KEY={MODEL_KEY}\n")
    environ: dict[str, str] = {}
    report = launcher.load_credentials(env_path, environ)
    assert report.selected_name == "MODEL_API_KEY"
    assert resolve_api_key(environ) == MODEL_KEY


def test_default_is_check_only(
        tmp_path: Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    _write_env(tmp_path / ".env", f"META_API_KEY={META_KEY}\n")
    calls: list[list[str]] = []
    monkeypatch.chdir(tmp_path)
    code = launcher.main(
        [], environ={},
        worker_main=lambda argv: calls.append(argv) or 0)
    assert code == 0
    assert calls == []
    out, err = capsys.readouterr()
    assert "META_API_KEY" in out
    assert META_KEY not in out and META_KEY not in err


def test_check_mode_never_launches(
        tmp_path: Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    _write_env(tmp_path / ".env", f"META_API_KEY={META_KEY}\n")
    calls: list[list[str]] = []
    monkeypatch.chdir(tmp_path)
    code = launcher.main(
        ["--check"], environ={},
        worker_main=lambda argv: calls.append(argv) or 0)
    assert code == 0
    assert calls == []
    out, err = capsys.readouterr()
    assert META_KEY not in out and META_KEY not in err


def test_run_without_credential_fails_closed(
        tmp_path: Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    _write_env(tmp_path / ".env", "UNRELATED=1\n")
    calls: list[list[str]] = []
    monkeypatch.chdir(tmp_path)
    code = launcher.main(
        ["--run"], environ={},
        worker_main=lambda argv: calls.append(argv) or 0)
    assert code == 2
    assert calls == []


def test_run_invokes_worker_once_exactly(
        tmp_path: Path, capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    _write_env(tmp_path / ".env", f"META_API_KEY={META_KEY}\n")
    calls: list[list[str]] = []

    def fake_worker(argv: list[str]) -> int:
        calls.append(argv)
        return 7

    monkeypatch.chdir(tmp_path)
    code = launcher.main(
        ["--run", "--worker-id", "w-9",
         "--storage-dir", "s", "--db-path", "d"],
        environ={}, worker_main=fake_worker)
    assert code == 7
    assert calls == [["--once", "--worker-id", "w-9",
                      "--allow-billable",
                      "--storage-dir", "s", "--db-path", "d"]]
    assert META_KEY not in " ".join(calls[0])
    out, err = capsys.readouterr()
    assert META_KEY not in out and META_KEY not in err


def test_run_uses_worker_defaults(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_env(tmp_path / ".env", f"META_API_KEY={META_KEY}\n")
    calls: list[list[str]] = []
    monkeypatch.chdir(tmp_path)
    code = launcher.main(
        ["--run"], environ={},
        worker_main=lambda argv: calls.append(argv) or 0)
    assert code == 0
    assert calls == [["--once", "--worker-id", "worker-1",
                      "--allow-billable"]]


def test_check_and_run_rejected(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.chdir(tmp_path)
    code = launcher.main(
        ["--check", "--run"], environ={},
        worker_main=lambda argv: calls.append(argv) or 0)
    assert code == 2
    assert calls == []


def test_import_has_no_side_effects(
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAUNCHER_CANARY", "intact")
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    import os

    before = dict(os.environ)
    fresh = _load_launcher("billable_launcher_fresh")
    assert os.environ.get("LAUNCHER_CANARY") == "intact"
    assert "META_API_KEY" not in os.environ
    assert "MODEL_API_KEY" not in os.environ
    assert dict(os.environ) == before
    out, err = capsys.readouterr()
    assert out == "" and err == ""
    assert hasattr(fresh, "main")
