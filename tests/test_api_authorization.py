"""Offline tests for billable authorization (M2-06C.5/S5-A).

Covers grant validation, submit gating, store migration,
the authorized meta executor through the worker with
test-double SDKs only, hard request caps, credential
redaction, and refusal paths. No network, no keys
provisioned for real, no paid calls.
"""

from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

import bakery_intelligence.api.pipeline as pipeline_module
from bakery_intelligence.api.app import build_app
from bakery_intelligence.api.authorization import (
    GRANT_FORMAT,
    GrantError,
    confirmation_sentence,
    grant_to_json,
    parse_grant,
    redact_key,
    resolve_api_key,
)
from bakery_intelligence.api.jobs import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    JobRecord,
    JobService,
    WORKFLOW_STORAGE_SMOKE,
    WORKFLOW_VIDEO_PIPELINE_META,
    new_run_id,
    utc_now_iso,
)
from bakery_intelligence.api.storage import LocalStorageBackend
from bakery_intelligence.api.store import SqliteJobStore
from bakery_intelligence.api.worker import run_once
from bakery_intelligence.inference.firstpass import STOP_REQUEST_LIMIT
from bakery_intelligence.inference.meta_spark import (
    MODEL_SPARK,
    MetaSparkClient,
    _default_client_factory,
)

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

KEY = "sk-test-s5a-secret"
OPERATOR = "operator-1"
CREATED = "2026-10-31T12:00:00+00:00"

VALID_OBS = json.dumps({
    "visible_objects": ["scale"],
    "scale_display_verbatim": "120",
    "scale_readable": True,
    "scale_state": "stable",
    "tare_action_visible": False,
    "addition_or_removal_visible": "none",
    "container_description": "bowl",
    "container_changed_hint": False,
    "confidence": "high",
    "ambiguity_notes": None,
})


def _grant(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "grant_format": GRANT_FORMAT,
        "workflow": WORKFLOW_VIDEO_PIPELINE_META,
        "provider": "meta",
        "model": MODEL_SPARK,
        "max_requests": 10,
        "token_budget": 1024,
        "max_frames": 100,
        "operator": OPERATOR,
        "created_at": CREATED,
    }
    base.update(overrides)
    if "confirmation" not in overrides:
        base["confirmation"] = confirmation_sentence(
            base["workflow"], base["model"],
            base["max_requests"], base["token_budget"],
            base["max_frames"], base["operator"])
    return base


def _service(tmp_path: Path) -> JobService:
    return JobService(SqliteJobStore(tmp_path / "jobs.sqlite3"))


def _stack(tmp_path: Path):
    storage = LocalStorageBackend(tmp_path / "storage")
    store = SqliteJobStore(tmp_path / "jobs.sqlite3")
    return storage, store, JobService(store)


def _make_video(path: Path, fps: float = 10.0,
                frame_count: int = 30) -> Path:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    if not writer.isOpened():
        pytest.skip("MJPG video encoding is unavailable.")
    for index in range(frame_count):
        shade = (index * 7) % 256
        writer.write(np.full((48, 64, 3), shade, dtype=np.uint8))
    writer.release()
    return path


class _FakeSDK:
    """Recording stand-in for the OpenAI SDK client."""

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    @property
    def responses(self) -> _FakeSDK:
        return self

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        action = self._script.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def _ok_response(request_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        status="completed", output_text=VALID_OBS,
        usage={"input_tokens": 100, "output_tokens": 50,
               "total_tokens": 150},
        model=MODEL_SPARK, id=request_id,
        incomplete_details=None)


def _patch_builder(monkeypatch: pytest.MonkeyPatch,
                   sdk: _FakeSDK,
                   factory_keys: list[str] | None = None
                   ) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def fake_builder(*, model: str, timeout_seconds: float,
                     store: bool) -> MetaSparkClient:
        seen.update(model=model, timeout_seconds=timeout_seconds,
                    store=store)

        def fake_factory(key: str, timeout: float) -> _FakeSDK:
            if factory_keys is not None:
                factory_keys.append(key)
            _ = timeout
            return sdk

        return MetaSparkClient(
            model=model, timeout_seconds=timeout_seconds,
            store=store, client_factory=fake_factory)

    monkeypatch.setattr(pipeline_module, "_build_meta_client",
                        fake_builder)
    return seen


def _set_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("META_API_KEY", KEY)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)


def _clear_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)


# Grant validation -------------------------------------------------


def test_valid_grant_parses_with_defaults() -> None:
    grant = parse_grant(_grant())
    assert grant.workflow == WORKFLOW_VIDEO_PIPELINE_META
    assert grant.provider == "meta"
    assert grant.model == MODEL_SPARK
    assert grant.max_requests == 10
    assert grant.token_budget == 1024
    assert grant.max_frames == 100
    assert grant.timeout_seconds == 120.0
    assert grant.store is True
    assert grant.cost_ceiling_usd is None
    assert grant.rates_usd_per_m is None
    assert grant.strict_cost is False
    assert grant.operator == OPERATOR


def test_grant_with_cost_controls_parses() -> None:
    grant = parse_grant(_grant(cost_ceiling_usd=5.0,
                               rates_usd_per_m=[1.5, 6.0],
                               strict_cost=True))
    assert grant.cost_ceiling_usd == 5.0
    assert grant.rates_usd_per_m == (1.5, 6.0)
    assert grant.strict_cost is True


def _invalid_grants() -> list[Any]:
    cases: list[Any] = [
        None, [], "grant",
        _grant(grant_format="bakery-billable-grant/9"),
        _grant(provider="baseten"), _grant(provider=""),
        _grant(model="muse-wonder-1.0"),
        _grant(model="gpt-x"),
        _grant(workflow=""),
        _grant(operator=""), _grant(operator=None),
        _grant(created_at=""), _grant(created_at="yesterday"),
        _grant(confirmation="authorize everything"),
        _grant(confirmation=""),
        _grant(api_key="sk-live-should-never-be-here"),
        _grant(max_cost_usd=5.0),
    ]
    for field in ("max_requests", "token_budget", "max_frames"):
        for bad in (0, -1, True, "3", None, 2.5):
            payload = _grant()
            payload[field] = bad
            cases.append(payload)
    for bad in (0, -1.0, True, "x", float("inf"), float("nan")):
        cases.append(_grant(timeout_seconds=bad))
    for bad in ("yes", 1, None):
        cases.append(_grant(store=bad))
        cases.append(_grant(strict_cost=bad))
    for bad in (-1.0, True, "x", float("inf")):
        cases.append(_grant(cost_ceiling_usd=bad))
    for bad in ("x", [1.0], [1.0, 2.0, 3.0], [-1.0, 2.0],
                [True, 2.0], [1.0, float("nan")]):
        cases.append(_grant(rates_usd_per_m=bad))
    missing = _grant()
    del missing["max_requests"]
    cases.append(missing)
    return cases


@pytest.mark.parametrize("payload", _invalid_grants())
def test_invalid_grants_rejected(payload: Any) -> None:
    with pytest.raises(GrantError):
        parse_grant(payload)


# Note: "model" is omitted here because a tampered model id
# is caught earlier by the allowlist, not by confirmation.
@pytest.mark.parametrize("field", [
    "workflow", "max_requests", "token_budget",
    "max_frames", "operator",
])
def test_confirmation_binds_every_field(field: str) -> None:
    payload = _grant()
    original = payload["confirmation"]
    if isinstance(payload[field], int):
        payload[field] = payload[field] + 1
    else:
        payload[field] = payload[field] + "-tampered"
    payload["confirmation"] = original
    with pytest.raises(GrantError, match="confirmation"):
        parse_grant(payload)


def test_grant_json_round_trip() -> None:
    grant = parse_grant(_grant(cost_ceiling_usd=5.0,
                               rates_usd_per_m=[1.5, 6.0]))
    assert parse_grant(json.loads(grant_to_json(grant))) == grant


def test_smoke_grant_template_parses() -> None:
    path = (Path(__file__).resolve().parents[1] / "docs" / "m206"
            / "smoke-grant.json")
    grant = parse_grant(json.loads(path.read_text(encoding="utf-8")))
    assert grant.max_requests == 3
    assert grant.max_frames == 3
    assert grant.token_budget == 512


def test_resolve_api_key_precedence(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("META_API_KEY", "meta-key")
    monkeypatch.setenv("MODEL_API_KEY", "model-key")
    assert resolve_api_key() == "meta-key"
    monkeypatch.delenv("META_API_KEY")
    assert resolve_api_key() == "model-key"
    monkeypatch.setenv("META_API_KEY", "")
    assert resolve_api_key() == "model-key"
    monkeypatch.delenv("META_API_KEY")
    monkeypatch.delenv("MODEL_API_KEY")
    with pytest.raises(GrantError, match="CREDENTIAL_MISSING"):
        resolve_api_key()


def test_redact_key() -> None:
    assert redact_key("boom sk-abc-123 here", "sk-abc-123") == (
        "boom *** here")
    assert redact_key("clean message", "sk-abc-123") == (
        "clean message")
    assert redact_key("anything", "") == "anything"


# Submit gating ----------------------------------------------------


def test_submit_meta_requires_grant(tmp_path: Path) -> None:
    service = _service(tmp_path)
    with pytest.raises(GrantError, match="requires an"):
        service.submit(WORKFLOW_VIDEO_PIPELINE_META,
                       source_object_id="clip.avi")
    assert service.store.claim_next(
        "worker-1", utc_now_iso()) is None


def test_submit_meta_rejects_workflow_mismatch(
        tmp_path: Path) -> None:
    service = _service(tmp_path)
    payload = _grant()
    payload["workflow"] = WORKFLOW_STORAGE_SMOKE
    with pytest.raises(GrantError, match="does not match"):
        service.submit(WORKFLOW_VIDEO_PIPELINE_META,
                       source_object_id="clip.avi",
                       grant=payload)


def test_submit_meta_rejects_invalid_grant(
        tmp_path: Path) -> None:
    service = _service(tmp_path)
    with pytest.raises(GrantError, match="confirmation"):
        service.submit(WORKFLOW_VIDEO_PIPELINE_META,
                       source_object_id="clip.avi",
                       grant=_grant(confirmation="nope"))
    assert service.store.claim_next(
        "worker-1", utc_now_iso()) is None


def test_submit_fake_rejects_grant(tmp_path: Path) -> None:
    service = _service(tmp_path)
    with pytest.raises(GrantError, match="does not accept"):
        service.submit(WORKFLOW_STORAGE_SMOKE,
                       source_object_id="clip.mp4",
                       grant=_grant())


def test_submit_meta_persists_grant(tmp_path: Path) -> None:
    service = _service(tmp_path)
    record = service.submit(WORKFLOW_VIDEO_PIPELINE_META,
                            source_object_id="clip.avi",
                            grant=_grant(max_requests=3))
    assert record.grant_json is not None
    assert KEY not in record.grant_json
    reread = service.get(record.run_id)
    assert reread is not None
    assert parse_grant(json.loads(reread.grant_json)) == (
        parse_grant(_grant(max_requests=3)))
    assert "grant_json" not in reread.to_dict()


def test_migration_adds_grant_column(tmp_path: Path) -> None:
    service = _service(tmp_path)
    old = service.submit(WORKFLOW_STORAGE_SMOKE)
    db_path = tmp_path / "jobs.sqlite3"
    connection = sqlite3.connect(str(db_path))
    connection.execute("ALTER TABLE jobs DROP COLUMN grant_json;")
    connection.commit()
    connection.close()
    reopened = JobService(SqliteJobStore(db_path))
    reread = reopened.get(old.run_id)
    assert reread is not None
    assert reread.status == "queued"
    assert reread.grant_json is None
    meta = reopened.submit(WORKFLOW_VIDEO_PIPELINE_META,
                           source_object_id="clip.avi",
                           grant=_grant())
    assert meta.grant_json is not None


# Authorized execution (fake SDK only) ------------------------------


def _submit_meta_video(storage: LocalStorageBackend,
                       service: JobService, raw: bytes,
                       grant: dict[str, Any]):
    storage.store_source("clip.avi", io.BytesIO(raw))
    return service.submit(WORKFLOW_VIDEO_PIPELINE_META,
                          source_object_id="clip.avi",
                          grant=grant)


def test_meta_executor_completes_with_fake_sdk(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage, store, service = _stack(tmp_path)
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    record = _submit_meta_video(storage, service, raw, _grant())
    sdk = _FakeSDK([_ok_response(f"resp_{index}")
                    for index in range(3)])
    seen = _patch_builder(monkeypatch, sdk)
    _set_key(monkeypatch)
    assert run_once(store, storage, "worker-1") == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_COMPLETED
    assert final.error is None
    assert len(sdk.calls) == 3
    assert seen["model"] == MODEL_SPARK
    assert seen["timeout_seconds"] == 120.0
    assert seen["store"] is True
    report = json.loads(storage.read_artifact(
        record.run_id, "firstpass_report.json"))
    assert report["billable"] is True
    assert report["budget"]["requests_submitted"] == 3
    haystacks = [
        storage.read_artifact(record.run_id, artifact_id
                              ).decode("utf-8", errors="replace")
        for artifact_id in final.artifact_ids
    ]
    haystacks.append(json.dumps(final.to_dict(), sort_keys=True))
    haystacks.append(final.grant_json or "")
    assert all(KEY not in haystack for haystack in haystacks)


def test_meta_request_cap_halts_before_dispatch(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage, store, service = _stack(tmp_path)
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    record = _submit_meta_video(
        storage, service, raw, _grant(max_requests=2))
    sdk = _FakeSDK([_ok_response(f"resp_{index}")
                    for index in range(3)])
    _patch_builder(monkeypatch, sdk)
    _set_key(monkeypatch)
    assert run_once(store, storage, "worker-1") == record.run_id
    assert service.get(record.run_id).status == STATUS_COMPLETED
    assert len(sdk.calls) == 2
    report = json.loads(storage.read_artifact(
        record.run_id, "firstpass_report.json"))
    assert report["budget"]["stopped_reason"] == STOP_REQUEST_LIMIT
    assert report["budget"]["remaining_frame_indices"] == [20]


def _observation_indices(storage: LocalStorageBackend,
                       run_id: str) -> list[int]:
    found: list[int] = []
    obs_dir = (storage.run_work_dir(run_id) / "firstpass"
               / "observations")
    for path in sorted(obs_dir.glob("*.json")):
        found.extend(item["frame_index"] for item in json.loads(
            path.read_text(encoding="utf-8")))
    return sorted(found)


def test_meta_smoke_grant_bounds_frames(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A 15-frame clip schedules exactly [0, 10] at 1 fps, so
    # it fits max_frames=2 and dispatches exactly 2 billed
    # requests with the granted 512-token budget.
    storage, store, service = _stack(tmp_path)
    raw = _make_video(tmp_path / "clip.avi",
                      frame_count=15).read_bytes()
    record = _submit_meta_video(
        storage, service, raw,
        _grant(max_requests=2, max_frames=2, token_budget=512))
    sdk = _FakeSDK([_ok_response("resp_a"),
                    _ok_response("resp_b")])
    _patch_builder(monkeypatch, sdk)
    _set_key(monkeypatch)
    assert run_once(store, storage, "worker-1") == record.run_id
    assert service.get(record.run_id).status == STATUS_COMPLETED
    assert len(sdk.calls) == 2
    assert {call["max_output_tokens"]
            for call in sdk.calls} == {512}
    assert _observation_indices(storage, record.run_id) == [0, 10]


def test_meta_frame_bound_refuses_oversized_schedule(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # max_frames is a hard extraction bound, not silent
    # truncation: a 3-frame schedule under max_frames=2
    # fails before any inference is dispatched.
    storage, store, service = _stack(tmp_path)
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    record = _submit_meta_video(
        storage, service, raw,
        _grant(max_requests=10, max_frames=2))
    sdk = _FakeSDK([])
    _patch_builder(monkeypatch, sdk)
    _set_key(monkeypatch)
    assert run_once(store, storage, "worker-1") == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_FAILED
    assert "exceeds max_frames" in (final.error or "")
    assert sdk.calls == []


def test_meta_executor_missing_key(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage, store, service = _stack(tmp_path)
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    record = _submit_meta_video(storage, service, raw, _grant())

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("client builder must not run")

    monkeypatch.setattr(pipeline_module, "_build_meta_client",
                        explode)
    _clear_keys(monkeypatch)
    assert run_once(store, storage, "worker-1") == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_FAILED
    assert "CREDENTIAL_MISSING" in (final.error or "")
    assert "must not run" not in (final.error or "")


def test_meta_executor_missing_grant(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage, store, _ = _stack(tmp_path)
    probe = JobRecord(run_id=new_run_id(),
                      workflow=WORKFLOW_VIDEO_PIPELINE_META)
    store.create(probe)

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("client builder must not run")

    monkeypatch.setattr(pipeline_module, "_build_meta_client",
                        explode)
    _set_key(monkeypatch)
    assert run_once(store, storage, "worker-1") == probe.run_id
    final = store.get(probe.run_id)
    assert final is not None
    assert final.status == STATUS_FAILED
    assert "requires an" in (final.error or "")
    assert "must not run" not in (final.error or "")


def test_meta_executor_tampered_grant(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage, store, service = _stack(tmp_path)
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    record = _submit_meta_video(storage, service, raw, _grant())
    tampered = _grant()
    tampered["workflow"] = WORKFLOW_STORAGE_SMOKE
    tampered["confirmation"] = confirmation_sentence(
        tampered["workflow"], tampered["model"],
        tampered["max_requests"], tampered["token_budget"],
        tampered["max_frames"], tampered["operator"])
    connection = sqlite3.connect(str(tmp_path / "jobs.sqlite3"))
    connection.execute(
        "UPDATE jobs SET grant_json = ? WHERE run_id = ?;",
        (json.dumps(tampered, sort_keys=True), record.run_id))
    connection.commit()
    connection.close()

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("client builder must not run")

    monkeypatch.setattr(pipeline_module, "_build_meta_client",
                        explode)
    _set_key(monkeypatch)
    assert run_once(store, storage, "worker-1") == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_FAILED
    assert "does not match" in (final.error or "")
    assert "must not run" not in (final.error or "")


def test_meta_redacts_key_from_errors(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage, store, service = _stack(tmp_path)
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    record = _submit_meta_video(storage, service, raw, _grant())
    sdk = _FakeSDK([_ok_response("resp_0")])

    def fail_clock() -> float:
        raise RuntimeError(f"clock exploded {KEY}")

    def fake_builder(*, model: str, timeout_seconds: float,
                     store: bool) -> MetaSparkClient:
        _ = (model, timeout_seconds, store)
        return MetaSparkClient(
            client_factory=lambda key, timeout: sdk,
            clock=fail_clock)

    monkeypatch.setattr(pipeline_module, "_build_meta_client",
                        fake_builder)
    _set_key(monkeypatch)
    assert run_once(store, storage, "worker-1") == record.run_id
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_FAILED
    assert KEY not in (final.error or "")
    assert "***" in (final.error or "")


def test_meta_stale_job_never_requeues(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    storage, store, service = _stack(tmp_path)
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    record = _submit_meta_video(storage, service, raw, _grant())
    claimed = store.claim_next(
        "crashed-worker", "2020-01-01T00:00:00+00:00")
    assert claimed is not None

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("client builder must not run")

    monkeypatch.setattr(pipeline_module, "_build_meta_client",
                        explode)
    _set_key(monkeypatch)
    assert run_once(store, storage, "worker-2") is None
    final = service.get(record.run_id)
    assert final is not None
    assert final.status == STATUS_FAILED
    assert final.attempts == 1
    assert "automatic retry refused" in (final.error or "")
    assert "must not run" not in (final.error or "")


def test_sdk_factory_receives_env_key(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression test for the R6 handoff break: the executor
    # resolves the key, and the SDK factory must receive that
    # exact key (previously it received "").
    storage, store, service = _stack(tmp_path)
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    record = _submit_meta_video(storage, service, raw, _grant())
    sdk = _FakeSDK([_ok_response(f"resp_{index}")
                    for index in range(3)])
    factory_keys: list[str] = []
    _patch_builder(monkeypatch, sdk, factory_keys)
    _set_key(monkeypatch)
    assert run_once(store, storage, "worker-1") == record.run_id
    assert service.get(record.run_id).status == STATUS_COMPLETED
    assert factory_keys == [KEY]


def test_work_dir_files_contain_no_key(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # State, checkpoints, reports, observations, raw
    # records, manifests, and staged inputs must all be
    # key-free after an authorized run.
    storage, store, service = _stack(tmp_path)
    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    record = _submit_meta_video(storage, service, raw, _grant())
    sdk = _FakeSDK([_ok_response(f"resp_{index}")
                    for index in range(3)])
    _patch_builder(monkeypatch, sdk)
    _set_key(monkeypatch)
    assert run_once(store, storage, "worker-1") == record.run_id
    assert service.get(record.run_id).status == STATUS_COMPLETED
    work_dir = storage.run_work_dir(record.run_id)
    checked = 0
    for path in sorted(work_dir.rglob("*")):
        if not path.is_file():
            continue
        checked += 1
        assert KEY.encode("utf-8") not in path.read_bytes(), path
    assert checked > 10


def test_config_repr_hides_api_key(tmp_path: Path) -> None:
    from bakery_intelligence.inference.firstpass import (
        FirstPassConfig,
        SourceRef,
    )

    config = FirstPassConfig(
        frames=[], source=SourceRef(name="s", sha256="0" * 64),
        out_dir=tmp_path, model="m", prompt_text="p",
        api_key=KEY)
    assert KEY not in repr(config)


def test_fake_path_rejects_api_key(tmp_path: Path) -> None:
    from bakery_intelligence.api.pipeline import (
        run_pipeline_offline,
    )
    from bakery_intelligence.api.storage import sha256_hex

    raw = _make_video(tmp_path / "clip.avi").read_bytes()
    work_dir = tmp_path / "work"
    with pytest.raises(ValueError, match="billable_client"):
        run_pipeline_offline(
            video_bytes=raw, video_name="clip.avi",
            run_id="run_0123456789ab",
            source_object_id="clip.avi",
            source_sha256=sha256_hex(raw),
            work_dir=work_dir, billable_api_key=KEY)
    assert not (work_dir / "pipeline_state.json").exists()


def test_production_builder_is_live_capable() -> None:
    client = pipeline_module._build_meta_client(
        model=MODEL_SPARK, timeout_seconds=45.0, store=False)
    assert isinstance(client, MetaSparkClient)
    assert client.model_id == MODEL_SPARK
    assert client.timeout_seconds == 45.0
    assert client.store is False
    assert client.client_factory is _default_client_factory


# HTTP layer -------------------------------------------------------


def _http_stack(tmp_path: Path):
    storage = LocalStorageBackend(tmp_path)
    db_path = tmp_path / "jobs.sqlite3"
    client = TestClient(build_app(storage=storage, db_path=db_path))
    return client, SqliteJobStore(db_path), storage


def test_api_meta_requires_grant(tmp_path: Path) -> None:
    client, _, _ = _http_stack(tmp_path)
    response = client.post(
        "/api/analyses",
        files={"file": ("clip.avi", b"fake-video")},
        data={"workflow": WORKFLOW_VIDEO_PIPELINE_META},
    )
    assert response.status_code == 422
    assert "requires an" in response.json()["detail"]


def test_api_meta_rejects_malformed_grant(tmp_path: Path) -> None:
    client, _, _ = _http_stack(tmp_path)
    response = client.post(
        "/api/analyses",
        files={"file": ("clip.avi", b"fake-video")},
        data={"workflow": WORKFLOW_VIDEO_PIPELINE_META,
              "grant": "{not json"},
    )
    assert response.status_code == 422


def test_api_meta_accepts_valid_grant(tmp_path: Path) -> None:
    client, _, _ = _http_stack(tmp_path)
    response = client.post(
        "/api/analyses",
        files={"file": ("clip.avi", b"fake-video")},
        data={"workflow": WORKFLOW_VIDEO_PIPELINE_META,
              "grant": json.dumps(_grant())},
    )
    assert response.status_code == 202
    created = response.json()
    assert created["workflow"] == WORKFLOW_VIDEO_PIPELINE_META
    status = client.get(f"/api/analyses/{created['run_id']}")
    assert status.status_code == 200
    assert "grant_json" not in status.json()


def test_api_fake_rejects_grant(tmp_path: Path) -> None:
    client, _, _ = _http_stack(tmp_path)
    response = client.post(
        "/api/analyses",
        files={"file": ("clip.mp4", b"fake-video")},
        data={"workflow": WORKFLOW_STORAGE_SMOKE,
              "grant": json.dumps(_grant())},
    )
    assert response.status_code == 422
    assert "does not accept" in response.json()["detail"]
