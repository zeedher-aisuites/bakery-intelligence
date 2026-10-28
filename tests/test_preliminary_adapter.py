"""Adapter tests: preliminary 0.1.0 human review to canonical 1.0.0 (M1-02).

Covers tasks.md M1-02 acceptance: successful adaptation of P04_104.json,
canonical Pydantic validation, JSON Schema validation, golden snapshot
equality, source integrity, no promotion to confirmed, no fabricated
evidence, approximate timestamp basis, adapter provenance, hypothesis
isolation, conservative display parsing, structured errors, and
determinism.

To regenerate the golden snapshot after an approved adapter change:
    python -c "import json; from pathlib import Path;
    from bakery_intelligence.adapters.preliminary_adapter import
    adapt_preliminary, result_to_snapshot;
    p = Path('tests/data/canonical/P04_104.canonical.json');
    p.write_text(json.dumps(result_to_snapshot(adapt_preliminary(Path(
    'experiments/003_video_to_recipe/annotations/preliminary/P04_104.json'
    ))), indent=2, sort_keys=True) + chr(10), encoding='utf-8')"
then inspect the diff before accepting it.

These tests assert structural validity of the translation only. A passing
suite is not verification that any reported reading is factually correct
(plan.md §12).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from bakery_intelligence.adapters.preliminary_adapter import (
    AdapterError,
    adapt_preliminary,
    result_to_snapshot,
)
from bakery_intelligence.models import Observation

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    REPO_ROOT
    / "experiments"
    / "003_video_to_recipe"
    / "annotations"
    / "preliminary"
    / "P04_104.json"
)
GOLDEN = REPO_ROOT / "tests" / "data" / "canonical" / "P04_104.canonical.json"
SCHEMAS_DIR = (
    REPO_ROOT / "src" / "bakery_intelligence" / "models" / "schemas"
)


def _by_id(result: Any) -> dict[str, Observation]:
    return {item.observation_id: item for item in result.observations}


def test_adapt_preliminary_counts() -> None:
    """52 source records yield 51 observations plus one timestamp issue."""
    result = adapt_preliminary(SOURCE)
    assert len(result.observations) == 51
    assert result.frame_evidence == []
    assert len(result.issues) == 1
    issue = result.issues[0]
    assert issue.code == "missing_timestamp"
    assert issue.entity_id == "pre_s4_34"


def test_observations_validate_canonically() -> None:
    result = adapt_preliminary(SOURCE)
    for item in result.observations:
        assert isinstance(item, Observation)
        assert item.schema_version == "1.0.0"
        assert Observation.model_validate(item.model_dump()) == item


def test_observations_validate_against_json_schema() -> None:
    """Emitted records validate against the committed Observation schema."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((SCHEMAS_DIR / "observation.json").read_text())
    result = adapt_preliminary(SOURCE)
    for item in result.observations:
        jsonschema.validate(
            instance=item.model_dump(mode="json"), schema=schema
        )


def test_golden_snapshot_equality() -> None:
    assert GOLDEN.is_file(), f"missing golden snapshot: {GOLDEN}"
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert result_to_snapshot(adapt_preliminary(SOURCE)) == expected


def test_source_bytes_unchanged() -> None:
    before = SOURCE.read_bytes()
    result = adapt_preliminary(SOURCE)
    assert SOURCE.read_bytes() == before
    assert result.source_hash == hashlib.sha256(before).hexdigest()


def test_no_confirmed_status() -> None:
    result = adapt_preliminary(SOURCE)
    for item in result.observations:
        assert item.verification_status.value in (
            "ambiguous",
            "unobservable",
        )


def test_status_mapping() -> None:
    """Absence-asserting notes are unobservable; reports are ambiguous."""
    found = _by_id(adapt_preliminary(SOURCE))
    assert found["pre_s4_06"].verification_status.value == "unobservable"
    assert found["pre_s2_01"].verification_status.value == "unobservable"
    assert found["pre_s2_02"].verification_status.value == "ambiguous"
    assert found["pre_s1_03"].verification_status.value == "ambiguous"


def test_no_frame_evidence() -> None:
    result = adapt_preliminary(SOURCE)
    assert result.frame_evidence == []
    for item in result.observations:
        assert item.evidence_ids == []


def test_timestamp_basis_approximate() -> None:
    result = adapt_preliminary(SOURCE)
    for item in result.observations:
        assert item.timestamp_basis.value == "approximate_human_video_review"
        assert item.timestamp_seconds is not None
        assert item.interval is None


def test_adapter_provenance() -> None:
    before = SOURCE.read_bytes()
    expected = f"P04_104.json@{hashlib.sha256(before).hexdigest()}"
    result = adapt_preliminary(SOURCE)
    for item in result.observations:
        assert item.provenance.source_type.value == "adapter"
        assert item.provenance.source_id == expected


def test_unit_hypothesis_not_promoted() -> None:
    result = adapt_preliminary(SOURCE)
    for item in result.observations:
        assert item.scale_unit is None
        assert item.scale_unit_hypothesis == "g"


def test_ambiguous_range_preserved() -> None:
    found = _by_id(adapt_preliminary(SOURCE))
    item = found["pre_s3_03"]
    assert item.scale_value is None
    assert item.raw_reported_value == "54–55"


def test_negative_reading_preserved_as_display_value() -> None:
    """A negative display is kept as a reading, never as a net mass."""
    found = _by_id(adapt_preliminary(SOURCE))
    item = found["pre_s1_03"]
    assert item.scale_value == -750.0
    assert item.raw_reported_value is None
    assert "net_ingredient_quantity" not in Observation.model_fields


def test_occluded_and_noisy_readings_refused() -> None:
    found = _by_id(adapt_preliminary(SOURCE))
    occluded = found["pre_s3_08"]
    assert occluded.scale_value is None
    assert occluded.raw_reported_value == "54"
    assert occluded.display_state is not None
    assert occluded.display_state.value == "occluded"
    noisy = found["pre_s4_03"]
    assert noisy.scale_value is None
    assert noisy.raw_reported_value == "0"


def test_timer_is_not_a_scale_reading() -> None:
    found = _by_id(adapt_preliminary(SOURCE))
    item = found["pre_s4_33"]
    assert item.scale_value is None
    assert item.raw_reported_value == "0:30:00"
    assert item.observation_type.value == "other"


def test_multi_span_readings_refused() -> None:
    found = _by_id(adapt_preliminary(SOURCE))
    stirring = found["pre_s3_05"]
    assert stirring.scale_value is None
    assert stirring.raw_reported_value == (
        "`60` before stirring; peaks near `600` during stirring."
    )
    battery = found["pre_s4_05"]
    assert battery.scale_value is None
    assert battery.raw_reported_value == (
        "`448` through glass, followed by `Lo`."
    )


def test_spurious_context_refused() -> None:
    found = _by_id(adapt_preliminary(SOURCE))
    item = found["pre_s3_10"]
    assert item.scale_value is None
    assert item.raw_reported_value == "68"


def test_display_state_conservative() -> None:
    """Only directly reported zero or occlusion set a display state."""
    found = _by_id(adapt_preliminary(SOURCE))
    assert found["pre_s1_05"].display_state is not None
    assert found["pre_s1_05"].display_state.value == "zero"
    assert found["pre_s2_02"].display_state is None
    assert found["pre_s4_22"].display_state is None


def test_verified_frame_path_with_synthetic_fixture(tmp_path: Path) -> None:
    """A non-null source evidence entry yields one linked FrameEvidence."""
    payload = {
        "preliminary_schema_version": "0.1.0",
        "video_id": "SYNTH.001",
        "scale_unit_hypothesis": None,
        "segments": [
            {
                "segment_id": "S1",
                "observations": [
                    {
                        "observation_id": "syn_01",
                        "time_label": "00:01",
                        "approximate_timestamp_seconds": 1,
                        "reported_action": "Scale visible.",
                        "reported_display_or_state": "`42`.",
                        "interpretation_and_verification_need": "None.",
                        "evidence": {
                            "frame_index": 25,
                            "image_ref": "frames/f000025.jpg",
                            "verification_status": "frame_verified",
                        },
                        "provenance": {
                            "source_type": "human",
                            "source_id": "notes.md",
                        },
                    }
                ],
            }
        ],
    }
    path = tmp_path / "synthetic.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = adapt_preliminary(path)
    assert len(result.observations) == 1
    assert len(result.frame_evidence) == 1
    assert result.frame_evidence[0].frame_index == 25
    assert result.observations[0].evidence_ids == ["syn_01:frm"]


def test_malformed_json_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(AdapterError) as exc_info:
        adapt_preliminary(path)
    assert exc_info.value.code == "invalid_preliminary"


def test_missing_segments_rejected(tmp_path: Path) -> None:
    path = tmp_path / "no_segments.json"
    path.write_text(
        json.dumps({"preliminary_schema_version": "0.1.0"}), encoding="utf-8"
    )
    with pytest.raises(AdapterError) as exc_info:
        adapt_preliminary(path)
    assert exc_info.value.code == "invalid_preliminary"


def test_wrong_schema_version_rejected(tmp_path: Path) -> None:
    path = tmp_path / "wrong_version.json"
    path.write_text(
        json.dumps(
            {"preliminary_schema_version": "0.2.0", "segments": []}
        ),
        encoding="utf-8",
    )
    with pytest.raises(AdapterError) as exc_info:
        adapt_preliminary(path)
    assert exc_info.value.code == "invalid_preliminary"


def test_segment_without_observations_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad_segment.json"
    path.write_text(
        json.dumps(
            {
                "preliminary_schema_version": "0.1.0",
                "segments": [{"segment_id": "S9"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(AdapterError) as exc_info:
        adapt_preliminary(path)
    assert exc_info.value.code == "invalid_preliminary"
    assert exc_info.value.entity_id == "S9"


def test_empty_segments_yield_empty_result(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(
        json.dumps(
            {"preliminary_schema_version": "0.1.0", "segments": []}
        ),
        encoding="utf-8",
    )
    result = adapt_preliminary(path)
    assert result.observations == []
    assert result.frame_evidence == []
    assert result.issues == []
    assert len(result.source_hash) == 64


def test_deterministic_repeat() -> None:
    first = adapt_preliminary(SOURCE)
    second = adapt_preliminary(SOURCE)
    assert [item.model_dump() for item in first.observations] == [
        item.model_dump() for item in second.observations
    ]
    assert result_to_snapshot(first) == result_to_snapshot(second)
