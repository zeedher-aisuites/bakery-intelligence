"""Contract tests for the canonical schema_version 1.0.0 domain models (M1-01).

Covers tasks.md M1-01 acceptance criteria: every record type is
constructable and JSON-serializable, ``schema_version`` is required and
rejected on unknown majors, hard model-layer invariants hold, and exported
JSON Schemas are present and well-formed.

These tests assert structural and semantic consistency only. A passing
contract suite is not evidence that any visual interpretation is factually
correct (plan.md §12).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

import pytest
from pydantic import ValidationError

from bakery_intelligence.models import (
    ActionEvent,
    DraftStatus,
    FrameEvidence,
    Observation,
    RecipeDraft,
    ReviewDecision,
    RunManifest,
    SchemaVersionedModel,
    VerificationReport,
    VerificationStatus,
    VideoSource,
    WeighingEvent,
)
from bakery_intelligence.models.recipe_draft import DraftQuantity
from bakery_intelligence.models.weighing_event import NetIngredientQuantity

SCHEMAS_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "bakery_intelligence"
    / "models"
    / "schemas"
)

SCHEMA_FILES = {
    "VideoSource": "video_source.json",
    "FrameEvidence": "frame_evidence.json",
    "Observation": "observation.json",
    "ActionEvent": "action_event.json",
    "WeighingEvent": "weighing_event.json",
    "RecipeDraft": "recipe_draft.json",
    "VerificationReport": "verification.json",
    "RunManifest": "run_manifest.json",
    "ReviewDecision": "review_decision.json",
}


def _video_source_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "video_id": "SYNTH.FIXTURE.001",
        "source_ref": "synthetic/nominal.mp4",
        "source_dataset": "synthetic-fixture",
        "timestamp_basis": "synthetic_fixture",
        "fps": 25.0,
        "frame_count": 250,
        "duration_seconds": 10.0,
        "interval": {"start_seconds": 0.0, "end_seconds": 10.0},
        "license": "CC-BY-NC-4.0",
    }


def _frame_evidence_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "evidence_id": "frm_00000120",
        "video_id": "SYNTH.FIXTURE.001",
        "frame_index": 120,
        "timestamp_seconds": 4.8,
        "source_timestamp_seconds": 4.8,
        "requested_timestamp_seconds": 4.8,
        "image_ref": "runs/demo/frames/frame_00000120.jpg",
        "region": {"x": 10, "y": 20, "width": 100, "height": 60},
        "extraction": {
            "sample_fps": 2.0,
            "source_offset": 0.0,
            "manifest_sha256": "a" * 64,
        },
    }


def _observation_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "observation_id": "obs_001",
        "modality": "visual",
        "timestamp_seconds": 4.8,
        "timestamp_basis": "synthetic_fixture",
        "observation_type": "scale_display",
        "scale_value": 120.0,
        "scale_unit": "g",
        "scale_unit_hypothesis": None,
        "display_state": "stable",
        "evidence_ids": ["frm_00000120"],
        "raw_reported_value": None,
        "description": "Display reads 120 g in the sampled frame.",
        "provenance": {"source_type": "human", "source_id": "annotator_01"},
        "verification_status": "ambiguous",
    }


def _action_event_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "action_id": "act_001",
        "start_seconds": 2.0,
        "end_seconds": 4.0,
        "action_type": "tare",
        "ingredient_id": None,
        "evidence_observation_ids": ["obs_001"],
        "provenance": {"source_type": "human", "source_id": "annotator_01"},
        "verification_status": "ambiguous",
    }


def _weighing_event_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "event_id": "weigh_001",
        "start_seconds": 2.0,
        "end_seconds": 6.0,
        "ingredient_id": "flour",
        "action_ids": ["act_001"],
        "observation_ids": ["obs_001"],
        "final_display_reading": {
            "value": 120.0,
            "unit": "g",
            "observation_id": "obs_001",
        },
        "net_ingredient_quantity": {
            "value": 120.0,
            "unit": "g",
            "derivation": "direct_after_tare",
        },
        "measurement_mode": "individual",
        "tare_observed": True,
        "verification_status": "ambiguous",
        "review_notes": "Tare observed before the addition.",
    }


def _recipe_draft_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "draft_id": "draft_001",
        "version": 1,
        "video_id": "SYNTH.FIXTURE.001",
        "steps": [
            {
                "index": 0,
                "description": "Tare the scale.",
                "action_ids": ["act_001"],
                "observation_ids": ["obs_001"],
                "evidence_summary": "Tare action with supporting observation.",
            }
        ],
        "ingredients": [
            {
                "ingredient_id": "flour",
                "quantity": {
                    "value": 120.0,
                    "unit": "g",
                    "derivation": "direct_after_tare",
                    "source_event_id": "weigh_001",
                },
                "status": "supported",
            }
        ],
        "unresolved_questions": [],
        "evidence_refs": {
            "observation_ids": ["obs_001"],
            "frame_evidence_ids": ["frm_00000120"],
        },
        "verification_ref": "ver_001",
        "run_id": "run_001",
        "status": "pending_review",
    }


def _verification_report_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "report_id": "ver_001",
        "verdict": "pass",
        "issues": [],
        "checked_ids": ["obs_001", "weigh_001"],
        "verifier_version": "1.0.0",
        "checked_at": "2026-10-28T12:00:00+00:00",
    }


def _run_manifest_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "run_id": "run_001",
        "code_revision": "8200e2e7248c12028adff645179a5e12f1930ac2",
        "started_at": "2026-10-28T12:00:00+00:00",
        "finished_at": "2026-10-28T12:00:05+00:00",
        "elapsed_ms": 5000,
        "video_source_snapshot": _video_source_payload(),
        "run_config_snapshot": {
            "sample_fps": 2.0,
            "interval": {"start_seconds": 0.0, "end_seconds": 10.0},
            "frame_budget": 100,
            "model_budget": 0,
            "retry_limit": 1,
            "cost_budget_usd": 0.0,
        },
        "timestamp_assumption": "constant_fps_25",
        "budget_state": {
            "frames_used": 20,
            "model_calls_used": 0,
            "cost_usd_spent": 0.0,
            "exhausted": False,
            "exhausted_at": None,
        },
        "outputs": {
            "ledger_path": "runs/run_001/ledger.jsonl",
            "events_path": "runs/run_001/events.json",
            "draft_path": "runs/run_001/draft.json",
            "verification_path": "runs/run_001/verification.json",
        },
        "attempts": [{"stage": "normalization", "verdict": "pass"}],
        "provider_calls": [],
        "errors": [],
        "status": "completed",
        "status_reason": "All stages completed.",
        "cost_summary": {"total_cost_usd": 0.0, "currency": "USD"},
    }


def _review_decision_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "decision_id": "dec_001",
        "draft_id": "draft_001",
        "draft_version": 1,
        "reviewer_id": "reviewer_01",
        "decision": "approved",
        "notes": "Verified against frame evidence.",
        "corrections_ref": None,
        "decided_at": "2026-10-28T13:00:00+00:00",
    }


PayloadFactory = Callable[[], dict[str, Any]]

VALID_PAYLOADS: dict[str, tuple[type[SchemaVersionedModel], PayloadFactory]] = {
    "VideoSource": (VideoSource, _video_source_payload),
    "FrameEvidence": (FrameEvidence, _frame_evidence_payload),
    "Observation": (Observation, _observation_payload),
    "ActionEvent": (ActionEvent, _action_event_payload),
    "WeighingEvent": (WeighingEvent, _weighing_event_payload),
    "RecipeDraft": (RecipeDraft, _recipe_draft_payload),
    "VerificationReport": (VerificationReport, _verification_report_payload),
    "RunManifest": (RunManifest, _run_manifest_payload),
    "ReviewDecision": (ReviewDecision, _review_decision_payload),
}


def _error_locations(exc: ValidationError) -> list[tuple[Any, ...]]:
    return [tuple(error["loc"]) for error in exc.errors()]


@pytest.mark.parametrize("model_name", sorted(VALID_PAYLOADS))
def test_schema_version_required(model_name: str) -> None:
    model_cls, factory = VALID_PAYLOADS[model_name]
    payload = factory()
    del payload["schema_version"]
    with pytest.raises(ValidationError) as exc_info:
        model_cls.model_validate(payload)
    assert ("schema_version",) in _error_locations(exc_info.value)


@pytest.mark.parametrize("model_name", sorted(VALID_PAYLOADS))
@pytest.mark.parametrize(
    "bad_version", ["0.1.0", "2.0.0", "10.3.2", "1.0", "v1", ""]
)
def test_schema_version_unknown_major_rejected(
    model_name: str, bad_version: str
) -> None:
    model_cls, factory = VALID_PAYLOADS[model_name]
    payload = factory()
    payload["schema_version"] = bad_version
    with pytest.raises(ValidationError) as exc_info:
        model_cls.model_validate(payload)
    assert ("schema_version",) in _error_locations(exc_info.value)


@pytest.mark.parametrize("model_name", sorted(VALID_PAYLOADS))
@pytest.mark.parametrize("good_version", ["1.0.0", "1.1.0"])
def test_schema_version_accepts_canonical_major(
    model_name: str, good_version: str
) -> None:
    """Additive minor revisions stay readable; only majors are gated."""
    model_cls, factory = VALID_PAYLOADS[model_name]
    payload = factory()
    payload["schema_version"] = good_version
    assert model_cls.model_validate(payload).schema_version == good_version


@pytest.mark.parametrize("model_name", sorted(VALID_PAYLOADS))
def test_model_roundtrip(model_name: str) -> None:
    model_cls, factory = VALID_PAYLOADS[model_name]
    original = model_cls.model_validate(factory())
    dumped = original.model_dump(mode="json")
    json.dumps(dumped)  # Must be JSON-serializable.
    assert model_cls.model_validate(dumped).model_dump() == (
        original.model_dump()
    )
    assert model_cls.model_validate_json(
        original.model_dump_json()
    ).model_dump() == original.model_dump()


@pytest.mark.parametrize(
    ("model_name", "required_field"),
    [
        ("VideoSource", "video_id"),
        ("FrameEvidence", "evidence_id"),
        ("Observation", "observation_id"),
        ("ActionEvent", "action_id"),
        ("WeighingEvent", "event_id"),
        ("RecipeDraft", "draft_id"),
        ("VerificationReport", "report_id"),
        ("RunManifest", "run_id"),
        ("ReviewDecision", "decision_id"),
    ],
)
def test_model_rejects_missing_required_field(
    model_name: str, required_field: str
) -> None:
    model_cls, factory = VALID_PAYLOADS[model_name]
    payload = factory()
    del payload[required_field]
    with pytest.raises(ValidationError) as exc_info:
        model_cls.model_validate(payload)
    assert (required_field,) in _error_locations(exc_info.value)


@pytest.mark.parametrize(
    ("model_name", "field", "bad_value"),
    [
        ("VideoSource", "timestamp_basis", "sometime"),
        ("Observation", "observation_type", "smell"),
        ("Observation", "display_state", "blurry"),
        ("ActionEvent", "action_type", "taste"),
        ("WeighingEvent", "measurement_mode", "sometimes"),
        ("RecipeDraft", "status", "auto_approved"),
        ("VerificationReport", "verdict", "maybe"),
        ("RunManifest", "status", "done"),
        ("ReviewDecision", "decision", "maybe"),
    ],
)
def test_model_rejects_unknown_enum_value(
    model_name: str, field: str, bad_value: str
) -> None:
    model_cls, factory = VALID_PAYLOADS[model_name]
    payload = factory()
    payload[field] = bad_value
    with pytest.raises(ValidationError) as exc_info:
        model_cls.model_validate(payload)
    assert (field,) in _error_locations(exc_info.value)


@pytest.mark.parametrize("model_name", ["VideoSource", "Observation"])
def test_model_rejects_unknown_field(model_name: str) -> None:
    model_cls, factory = VALID_PAYLOADS[model_name]
    payload = factory()
    payload["typo_field"] = "oops"
    with pytest.raises(ValidationError):
        model_cls.model_validate(payload)


def test_video_source_interval_bounds() -> None:
    base = _video_source_payload()
    VideoSource.model_validate(base)

    degenerate = dict(base)
    degenerate["interval"] = {"start_seconds": 5.0, "end_seconds": 5.0}
    with pytest.raises(ValidationError):
        VideoSource.model_validate(degenerate)

    negative = dict(base)
    negative["interval"] = {"start_seconds": -1.0, "end_seconds": 5.0}
    with pytest.raises(ValidationError):
        VideoSource.model_validate(negative)

    beyond_duration = dict(base)
    beyond_duration["interval"] = {"start_seconds": 0.0, "end_seconds": 10.5}
    with pytest.raises(ValidationError):
        VideoSource.model_validate(beyond_duration)

    at_duration = dict(base)
    at_duration["interval"] = {"start_seconds": 0.0, "end_seconds": 10.0}
    VideoSource.model_validate(at_duration)

    unknown_duration = dict(base)
    unknown_duration["duration_seconds"] = None
    unknown_duration["interval"] = {"start_seconds": 0.0, "end_seconds": 99.0}
    VideoSource.model_validate(unknown_duration)


@pytest.mark.parametrize(
    "absolute_ref", ["/abs/vid.mp4", "C:\\vids\\x.mp4", "C:/vids/x.mp4"]
)
def test_video_source_rejects_absolute_source_ref(absolute_ref: str) -> None:
    payload = _video_source_payload()
    payload["source_ref"] = absolute_ref
    with pytest.raises(ValidationError) as exc_info:
        VideoSource.model_validate(payload)
    assert ("source_ref",) in _error_locations(exc_info.value)


def test_observation_time_forms() -> None:
    base = _observation_payload()
    Observation.model_validate(base)

    both = dict(base)
    both["interval"] = {"start_seconds": 4.0, "end_seconds": 5.0}
    with pytest.raises(ValidationError):
        Observation.model_validate(both)

    neither = dict(base)
    del neither["timestamp_seconds"]
    with pytest.raises(ValidationError):
        Observation.model_validate(neither)

    interval_only = dict(base)
    del interval_only["timestamp_seconds"]
    interval_only["interval"] = {"start_seconds": 4.0, "end_seconds": 5.0}
    parsed = Observation.model_validate(interval_only)
    assert parsed.timestamp_seconds is None
    assert parsed.interval is not None


def test_observation_rejects_negative_timestamp() -> None:
    payload = _observation_payload()
    payload["timestamp_seconds"] = -0.5
    with pytest.raises(ValidationError) as exc_info:
        Observation.model_validate(payload)
    assert ("timestamp_seconds",) in _error_locations(exc_info.value)


@pytest.mark.parametrize(
    "factory_name",
    ["ActionEvent", "WeighingEvent"],
)
@pytest.mark.parametrize(
    ("start", "end"), [(4.0, 4.0), (5.0, 4.0), (-1.0, 2.0)]
)
def test_event_intervals_reject_bad_bounds(
    factory_name: str, start: float, end: float
) -> None:
    model_cls, factory = VALID_PAYLOADS[factory_name]
    payload = factory()
    payload["start_seconds"] = start
    payload["end_seconds"] = end
    with pytest.raises(ValidationError):
        model_cls.model_validate(payload)


@pytest.mark.parametrize("quantity_cls", [NetIngredientQuantity, DraftQuantity])
def test_unknown_derivation_requires_null_quantity(
    quantity_cls: type[NetIngredientQuantity] | type[DraftQuantity],
) -> None:
    quantity_cls.model_validate(
        {"value": None, "unit": None, "derivation": "unknown"}
    )
    with pytest.raises(ValidationError):
        quantity_cls.model_validate(
            {"value": 120.0, "unit": None, "derivation": "unknown"}
        )
    with pytest.raises(ValidationError):
        quantity_cls.model_validate(
            {"value": None, "unit": "g", "derivation": "unknown"}
        )
    quantity_cls.model_validate(
        {"value": 120.0, "unit": "g", "derivation": "direct_after_tare"}
    )


def test_scale_display_kept_distinct_from_net_mass() -> None:
    assert "net_ingredient_quantity" not in Observation.model_fields
    assert "scale_value" not in WeighingEvent.model_fields
    payload = _weighing_event_payload()
    payload["final_display_reading"] = {
        "value": 150.0,
        "unit": "g",
        "observation_id": "obs_001",
    }
    payload["net_ingredient_quantity"] = {
        "value": 120.0,
        "unit": "g",
        "derivation": "difference_between_readings",
    }
    event = WeighingEvent.model_validate(payload)
    assert event.final_display_reading.value == 150.0
    assert event.net_ingredient_quantity.value == 120.0


def test_no_automatic_status_change() -> None:
    payload = _recipe_draft_payload()
    del payload["status"]
    assert RecipeDraft.model_validate(payload).status is DraftStatus.PENDING_REVIEW

    payload = _observation_payload()
    payload["provenance"] = {
        "source_type": "adapter",
        "source_id": "P04_104.json@deadbeef",
    }
    payload["verification_status"] = "ambiguous"
    observed = Observation.model_validate(payload)
    assert observed.verification_status is VerificationStatus.AMBIGUOUS


@pytest.mark.parametrize("model_name", sorted(SCHEMA_FILES))
def test_json_schemas_present_and_wellformed(model_name: str) -> None:
    path = SCHEMAS_DIR / SCHEMA_FILES[model_name]
    assert path.is_file(), f"missing exported schema: {path}"
    schema = json.loads(path.read_text(encoding="utf-8"))
    assert schema["$schema"].startswith("https://json-schema.org/")
    assert schema["title"] == model_name
    assert "schema_version" in schema["required"]
    assert schema.get("additionalProperties") is False


SYNC_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "sync_contract_schemas.py"
)


def _load_sync_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "sync_contract_schemas", SYNC_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)
    return sync


def test_json_schemas_match_model_exports() -> None:
    """Committed schemas must byte-match the canonical model export.

    If this fails, regenerate with:
        python scripts/sync_contract_schemas.py
    inspect the diff, and only then accept the updated artifacts.
    """
    sync = _load_sync_module()
    for filename, expected in sorted(sync.render_all().items()):
        path = SCHEMAS_DIR / filename
        assert path.is_file(), f"missing exported schema: {path}"
        assert path.read_text(encoding="utf-8") == expected, (
            f"schema drift in {filename}; run "
            "python scripts/sync_contract_schemas.py to resynchronize"
        )


def test_generated_schema_version_pattern() -> None:
    """Every generated schema restricts schema_version to major 1."""
    sync = _load_sync_module()
    for filename, text in sorted(sync.render_all().items()):
        schema = json.loads(text)
        pattern = schema["properties"]["schema_version"].get("pattern")
        assert pattern == r"^1\.\d+\.\d+$", filename


def test_generated_observation_time_rule() -> None:
    """Generated observation schema requires exactly one time form."""
    sync = _load_sync_module()
    schema = json.loads(sync.render_all()["observation.json"])
    ones = [
        entry["oneOf"] for entry in schema.get("allOf", []) if "oneOf" in entry
    ]
    assert len(ones) == 1
    required_keys = sorted(branch["required"] for branch in ones[0])
    assert required_keys == [["interval"], ["timestamp_seconds"]]


@pytest.mark.parametrize(
    ("filename", "def_name"),
    [
        ("weighing_event.json", "NetIngredientQuantity"),
        ("recipe_draft.json", "DraftQuantity"),
    ],
)
def test_generated_unknown_derivation_rules(
    filename: str, def_name: str
) -> None:
    """Generated quantity definitions forbid value/unit unless supported."""
    sync = _load_sync_module()
    schema = json.loads(sync.render_all()[filename])
    node = schema["$defs"][def_name]
    conditionals = [
        entry for entry in node.get("allOf", []) if "if" in entry
    ]
    assert len(conditionals) == 1
    then = conditionals[0]["then"]
    assert then["properties"]["value"] == {"type": "null"}
    assert then["properties"]["unit"] == {"type": "null"}


def test_generated_source_ref_rule() -> None:
    """Generated video source schema forbids absolute source_ref paths."""
    sync = _load_sync_module()
    schema = json.loads(sync.render_all()["video_source.json"])
    not_rule = schema["properties"]["source_ref"].get("not", {})
    assert not_rule.get("pattern") == r"^([/\\]|[A-Za-z]:[/\\])"


def _collect_refs(node: Any, refs: list[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                refs.append(value)
            else:
                _collect_refs(value, refs)
    elif isinstance(node, list):
        for item in node:
            _collect_refs(item, refs)


def test_generated_schemas_self_contained() -> None:
    """Every generated $ref resolves inside its own file (no cross-file)."""
    sync = _load_sync_module()
    for filename, text in sorted(sync.render_all().items()):
        schema = json.loads(text)
        refs: list[str] = []
        _collect_refs(schema, refs)
        for ref in refs:
            assert ref.startswith("#"), (filename, ref)
            if ref.startswith("#/$defs/"):
                target = ref[len("#/$defs/"):]
                assert target in schema.get("$defs", {}), (filename, ref)


@pytest.mark.parametrize("quantity_cls", [NetIngredientQuantity, DraftQuantity])
def test_omitted_derivation_defaults_to_unknown(
    quantity_cls: type[NetIngredientQuantity] | type[DraftQuantity],
) -> None:
    """An omitted derivation behaves as unknown at the model layer."""
    quantity_cls.model_validate({"value": None, "unit": None})
    quantity_cls.model_validate({})
    with pytest.raises(ValidationError):
        quantity_cls.model_validate({"value": 120.0, "unit": "g"})


@pytest.mark.parametrize("model_name", sorted(VALID_PAYLOADS))
def test_generated_schemas_accept_contract_fixtures(model_name: str) -> None:
    """Valid contract payloads validate against the generated schemas.

    Requires a JSON Schema validator library; skipped when absent.
    """
    jsonschema = pytest.importorskip("jsonschema")
    sync = _load_sync_module()
    model_cls, factory = VALID_PAYLOADS[model_name]
    instance = model_cls.model_validate(factory()).model_dump(mode="json")
    schema = json.loads(sync.render_all()[SCHEMA_FILES[model_name]])
    jsonschema.validate(instance=instance, schema=schema)


def _negative_bad_version() -> tuple[str, dict[str, Any]]:
    payload = _video_source_payload()
    payload["schema_version"] = "2.0.0"
    return "video_source.json", payload


def _negative_both_time_forms() -> tuple[str, dict[str, Any]]:
    payload = _observation_payload()
    payload["interval"] = {"start_seconds": 4.0, "end_seconds": 5.0}
    return "observation.json", payload


def _negative_no_time_form() -> tuple[str, dict[str, Any]]:
    payload = _observation_payload()
    del payload["timestamp_seconds"]
    return "observation.json", payload


def _negative_unknown_derivation_with_value() -> tuple[str, dict[str, Any]]:
    payload = _weighing_event_payload()
    payload["net_ingredient_quantity"] = {
        "value": 50.0,
        "unit": "g",
        "derivation": "unknown",
    }
    return "weighing_event.json", payload


def _negative_omitted_derivation_with_value() -> tuple[str, dict[str, Any]]:
    payload = _weighing_event_payload()
    payload["net_ingredient_quantity"] = {"value": 50.0, "unit": "g"}
    return "weighing_event.json", payload


def _negative_absolute_source_ref() -> tuple[str, dict[str, Any]]:
    payload = _video_source_payload()
    payload["source_ref"] = "/abs/x.mp4"
    return "video_source.json", payload


NEGATIVE_SCHEMA_CASES = {
    "bad_schema_version": _negative_bad_version,
    "observation_both_time_forms": _negative_both_time_forms,
    "observation_no_time_form": _negative_no_time_form,
    "unknown_derivation_with_value": _negative_unknown_derivation_with_value,
    "omitted_derivation_with_value": _negative_omitted_derivation_with_value,
    "absolute_source_ref": _negative_absolute_source_ref,
}


@pytest.mark.parametrize("case", sorted(NEGATIVE_SCHEMA_CASES))
def test_generated_schemas_reject_invalid_payloads(case: str) -> None:
    """Invalid payloads fail against the generated schemas.

    Requires a JSON Schema validator library; skipped when absent.
    """
    jsonschema = pytest.importorskip("jsonschema")
    sync = _load_sync_module()
    filename, instance = NEGATIVE_SCHEMA_CASES[case]()
    schema = json.loads(sync.render_all()[filename])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=instance, schema=schema)
