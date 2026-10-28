"""M2-01 synthetic fixture bundle tests.

Covers tasks.md M2-01 acceptance: the five offline fixtures under
``fixtures/synthetic/`` load through a strict test-harness envelope into
canonical M1-01 models, carry honest synthetic provenance, and produce the
frozen M1-03 structural outcomes (four pass bundles plus one
dangling-reference bundle). Tare history, preserved contradictions, and
budget-exhausted execution are asserted structurally only; downstream
M2-03/M2-04/M2-06 behavior is NOT EVALUATED here.

Oracle sources: every test case and expected outcome comes from the frozen
M2-01 decisions (tasks.md M2-01, spec.md, constitution). Literal ids,
counts, timestamps, and tag strings were pinned by reading the committed
fixture files, never by running checks or the loader.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from bakery_intelligence.models import (
    FrameEvidence,
    GateId,
    Observation,
    VideoSource,
)
from bakery_intelligence.models.verification import VerificationIssue
from bakery_intelligence.verification.reference_checks import (
    check_interval_bounds,
    check_measurement_derivation,
    check_referential_integrity,
    check_timestamps,
    check_unique_ids,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REPO_ROOT / "fixtures" / "synthetic"
FIXTURES_README = REPO_ROOT / "fixtures" / "README.md"
SCHEMAS_DIR = (
    REPO_ROOT / "src" / "bakery_intelligence" / "models" / "schemas"
)

FIXTURE_CODE = "FIXTURE_INVALID"
ENVELOPE_VERSION = "1.0.0"
ENVELOPE_KEYS = (
    "schema_version",
    "fixture_id",
    "description",
    "video_source",
    "frame_evidence",
    "observations",
)
ENVELOPE_KEY_SET = frozenset(ENVELOPE_KEYS)


class FixtureError(Exception):
    """Load-time fixture failure with a machine-readable code."""

    def __init__(self, code: str, entity_id: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.entity_id = entity_id
        self.message = message


def _record_id(
    payload: Any, field: str, fallback: str
) -> str:
    if isinstance(payload, dict) and isinstance(
        payload.get(field), str
    ):
        value = payload[field]
        assert isinstance(value, str)
        if value:
            return value
    return fallback


def load_fixture(path: Path) -> dict[str, Any]:
    """Load and validate one fixture envelope into M1-01 models.

    Returns the validated bundle (envelope scalars plus model
    instances). Raises :class:`FixtureError` with code
    ``FIXTURE_INVALID`` on any load-time failure: unreadable file,
    malformed JSON, non-object envelope, unknown or missing envelope
    keys, a non-``"1.0.0"`` envelope ``schema_version``, a
    ``fixture_id`` that does not equal the filename stem, an empty
    description, mistyped sections, or a nested ``ValidationError``
    (re-raised with the original error preserved as ``__cause__``).
    A structurally invalid bundle (dangling reference) still parses
    successfully here; it fails only at check time.
    """
    stem = path.stem
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FixtureError(
            FIXTURE_CODE, stem, f"cannot read {path.name}: {exc}"
        ) from exc
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FixtureError(
            FIXTURE_CODE, stem, f"malformed JSON in {path.name}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise FixtureError(
            FIXTURE_CODE,
            stem,
            f"fixture {path.name} must be a JSON object",
        )
    unknown = sorted(set(payload) - ENVELOPE_KEY_SET)
    if unknown:
        raise FixtureError(
            FIXTURE_CODE,
            stem,
            f"unknown envelope keys {unknown} in {path.name}",
        )
    missing = [key for key in ENVELOPE_KEYS if key not in payload]
    if missing:
        raise FixtureError(
            FIXTURE_CODE,
            stem,
            f"missing envelope keys {missing} in {path.name}",
        )
    if payload["schema_version"] != ENVELOPE_VERSION:
        raise FixtureError(
            FIXTURE_CODE,
            stem,
            "envelope schema_version must be exactly "
            f"{ENVELOPE_VERSION!r} in {path.name}",
        )
    fixture_id = payload["fixture_id"]
    if not isinstance(fixture_id, str) or fixture_id != stem:
        raise FixtureError(
            FIXTURE_CODE,
            stem,
            f"fixture_id {fixture_id!r} must equal filename stem "
            f"{stem!r} in {path.name}",
        )
    description = payload["description"]
    if not isinstance(description, str) or not description.strip():
        raise FixtureError(
            FIXTURE_CODE,
            fixture_id,
            f"description must be a non-empty string in {path.name}",
        )
    video_payload = payload["video_source"]
    if not isinstance(video_payload, dict):
        raise FixtureError(
            FIXTURE_CODE, fixture_id, "video_source must be an object"
        )
    try:
        video_source = VideoSource.model_validate(video_payload)
    except ValidationError as exc:
        raise FixtureError(
            FIXTURE_CODE,
            _record_id(video_payload, "video_id", fixture_id),
            f"invalid video_source in {path.name}: "
            f"{exc.error_count()} error(s)",
        ) from exc
    frame_payloads = payload["frame_evidence"]
    if not isinstance(frame_payloads, list):
        raise FixtureError(
            FIXTURE_CODE, fixture_id, "frame_evidence must be an array"
        )
    frames: list[FrameEvidence] = []
    for index, item in enumerate(frame_payloads):
        if not isinstance(item, dict):
            raise FixtureError(
                FIXTURE_CODE,
                f"{fixture_id}#frame_evidence[{index}]",
                "frame_evidence entries must be objects",
            )
        try:
            frames.append(FrameEvidence.model_validate(item))
        except ValidationError as exc:
            raise FixtureError(
                FIXTURE_CODE,
                _record_id(
                    item,
                    "evidence_id",
                    f"{fixture_id}#frame_evidence[{index}]",
                ),
                f"invalid frame_evidence[{index}] in {path.name}: "
                f"{exc.error_count()} error(s)",
            ) from exc
    observation_payloads = payload["observations"]
    if not isinstance(observation_payloads, list):
        raise FixtureError(
            FIXTURE_CODE, fixture_id, "observations must be an array"
        )
    observations: list[Observation] = []
    for index, item in enumerate(observation_payloads):
        if not isinstance(item, dict):
            raise FixtureError(
                FIXTURE_CODE,
                f"{fixture_id}#observations[{index}]",
                "observation entries must be objects",
            )
        try:
            observations.append(Observation.model_validate(item))
        except ValidationError as exc:
            raise FixtureError(
                FIXTURE_CODE,
                _record_id(
                    item,
                    "observation_id",
                    f"{fixture_id}#observations[{index}]",
                ),
                f"invalid observations[{index}] in {path.name}: "
                f"{exc.error_count()} error(s)",
            ) from exc
    return {
        "schema_version": ENVELOPE_VERSION,
        "fixture_id": fixture_id,
        "description": description,
        "video_source": video_source,
        "frame_evidence": frames,
        "observations": observations,
    }


FIXTURE_STEMS = [
    "synthetic_interval_conflicting",
    "synthetic_interval_dangling_ref",
    "synthetic_interval_nominal",
    "synthetic_interval_partial_budget",
    "synthetic_interval_unknown_tare",
]

DANGLING_STEM = "synthetic_interval_dangling_ref"
NOMINAL_STEM = "synthetic_interval_nominal"
UNKNOWN_TARE_STEM = "synthetic_interval_unknown_tare"
CONFLICTING_STEM = "synthetic_interval_conflicting"
PARTIAL_BUDGET_STEM = "synthetic_interval_partial_budget"

PASS_STEMS = [
    stem for stem in FIXTURE_STEMS if stem != DANGLING_STEM
]

# (class tag, expected-m1-03 tag, expected-m2 tag) pinned from each file.
EXPECTED_TAGS = {
    NOMINAL_STEM: ("nominal", "pass", "completed-pass"),
    UNKNOWN_TARE_STEM: ("b", "pass", "unknown-mass-UNKNOWN_TARE_HISTORY"),
    CONFLICTING_STEM: (
        "b",
        "pass",
        "review-needed-contradiction-preserved",
    ),
    DANGLING_STEM: ("a", "fail", "fail-DANGLING_REFERENCE-no-draft"),
    PARTIAL_BUDGET_STEM: ("c", "pass", "partial-BUDGET_EXHAUSTED"),
}

EXPECTED_VIDEO_IDS = {
    NOMINAL_STEM: "SYNTH.NOM.001",
    UNKNOWN_TARE_STEM: "SYNTH.UNK.001",
    CONFLICTING_STEM: "SYNTH.CON.001",
    DANGLING_STEM: "SYNTH.DAN.001",
    PARTIAL_BUDGET_STEM: "SYNTH.BUD.001",
}

EXPECTED_FRAME_IDS = {
    NOMINAL_STEM: ["frm_nom_001", "frm_nom_002", "frm_nom_003"],
    UNKNOWN_TARE_STEM: ["frm_unk_001", "frm_unk_002"],
    CONFLICTING_STEM: ["frm_con_001", "frm_con_002"],
    DANGLING_STEM: ["frm_dan_001"],
    PARTIAL_BUDGET_STEM: ["frm_bud_001", "frm_bud_002", "frm_bud_003"],
}

EXPECTED_OBSERVATION_IDS = {
    NOMINAL_STEM: ["obs_nom_tare", "obs_nom_mass"],
    UNKNOWN_TARE_STEM: ["obs_unk_001"],
    CONFLICTING_STEM: ["obs_con_001", "obs_con_002"],
    DANGLING_STEM: ["obs_dan_001"],
    PARTIAL_BUDGET_STEM: ["obs_bud_001", "obs_bud_002", "obs_bud_003"],
}

# D9: runtime-result vocabulary forbidden anywhere in fixture JSON.
FORBIDDEN_TOKENS = (
    "net_ingredient_quantity",
    "derivation",
    "measurement_mode",
    "quantity",
    "action_ids",
    "consensus",
)

FRAME_ID_RE = re.compile(r"^frm_(nom|unk|con|dan|bud)_[0-9]{3}$")
CLASS_TAG_RE = re.compile(r"\[class: (nominal|a|b|c)\]")
M103_TAG_RE = re.compile(r"\[expected-m1-03: (pass|fail)\]")
M2_TAG_RE = re.compile(r"\[expected-m2: ([^\]]+)\]")


def _fixture_path(stem: str) -> Path:
    return FIXTURES_DIR / f"{stem}.json"


def _all_fixture_paths() -> list[Path]:
    return sorted(FIXTURES_DIR.glob("synthetic_interval_*.json"))


def _raw_text(stem: str) -> str:
    return _fixture_path(stem).read_text(encoding="utf-8")


def _raw_payload(stem: str) -> dict[str, Any]:
    payload = json.loads(_raw_text(stem))
    assert isinstance(payload, dict)
    return payload


def _bundle(stem: str) -> dict[str, Any]:
    return load_fixture(_fixture_path(stem))


def _dump_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    video = bundle["video_source"]
    assert isinstance(video, VideoSource)
    frames = bundle["frame_evidence"]
    observations = bundle["observations"]
    assert all(isinstance(item, FrameEvidence) for item in frames)
    assert all(isinstance(item, Observation) for item in observations)
    return {
        "schema_version": bundle["schema_version"],
        "fixture_id": bundle["fixture_id"],
        "description": bundle["description"],
        "video_source": video.model_dump(mode="json"),
        "frame_evidence": [
            item.model_dump(mode="json") for item in frames
        ],
        "observations": [
            item.model_dump(mode="json") for item in observations
        ],
    }


def _run_all_checks(bundle: dict[str, Any]) -> list[list[VerificationIssue]]:
    video = bundle["video_source"]
    frames = bundle["frame_evidence"]
    observations = bundle["observations"]
    return [
        check_unique_ids(
            frame_evidence=frames, observations=observations
        ),
        check_timestamps(
            video_source=video,
            observations=observations,
            frame_evidence=frames,
        ),
        check_referential_integrity(
            frame_evidence=frames, observations=observations
        ),
        check_interval_bounds(
            video_source=video, action_events=[], weighing_events=[]
        ),
        check_measurement_derivation(weighing_events=[]),
    ]


def _assert_fixture_invalid(error: FixtureError) -> None:
    assert error.code == FIXTURE_CODE
    assert isinstance(error.entity_id, str) and error.entity_id
    assert isinstance(error.message, str) and error.message


def test_fixture_inventory_exactly_five_files() -> None:
    paths = _all_fixture_paths()
    assert [path.name for path in paths] == [
        f"{stem}.json" for stem in FIXTURE_STEMS
    ]
    assert FIXTURE_STEMS == sorted(FIXTURE_STEMS)


def test_readme_documents_every_fixture_and_keywords() -> None:
    assert FIXTURES_README.is_file(), (
        f"missing fixtures README: {FIXTURES_README}"
    )
    text = FIXTURES_README.read_text(encoding="utf-8")
    for stem in FIXTURE_STEMS:
        assert stem in text
    for keyword in (
        "M2-01",
        "synthetic_fixture",
        "FIXTURE_INVALID",
        "DANGLING_REFERENCE",
        "BUDGET_EXHAUSTED",
        "UNKNOWN_TARE_HISTORY",
        "NOT EVALUATED",
    ):
        assert keyword in text


def test_readme_states_envelope_is_not_a_domain_model() -> None:
    text = FIXTURES_README.read_text(encoding="utf-8")
    assert "new canonical domain model" in text
    assert "ActionEvent" in text
    assert "WeighingEvent" in text


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_envelope_exact_key_set_and_order(stem: str) -> None:
    payload = _raw_payload(stem)
    assert set(payload) == set(ENVELOPE_KEYS)
    assert list(payload) == list(ENVELOPE_KEYS)


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_envelope_schema_version_pinned_exactly(stem: str) -> None:
    payload = _raw_payload(stem)
    assert payload["schema_version"] == "1.0.0"
    assert isinstance(payload["schema_version"], str)


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_fixture_id_equals_filename_stem(stem: str) -> None:
    payload = _raw_payload(stem)
    assert payload["fixture_id"] == stem
    bundle = _bundle(stem)
    assert bundle["fixture_id"] == stem


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_description_tags_exact(stem: str) -> None:
    description = _raw_payload(stem)["description"]
    assert isinstance(description, str) and description.strip()
    expected_class, expected_m103, expected_m2 = EXPECTED_TAGS[stem]
    class_match = CLASS_TAG_RE.search(description)
    m103_match = M103_TAG_RE.search(description)
    m2_match = M2_TAG_RE.search(description)
    assert class_match is not None
    assert m103_match is not None
    assert m2_match is not None
    assert class_match.group(1) == expected_class
    assert m103_match.group(1) == expected_m103
    assert m2_match.group(1) == expected_m2


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_nested_schema_versions_pinned(stem: str) -> None:
    payload = _raw_payload(stem)
    assert payload["video_source"]["schema_version"] == "1.0.0"
    for record in payload["frame_evidence"]:
        assert record["schema_version"] == "1.0.0"
    for record in payload["observations"]:
        assert record["schema_version"] == "1.0.0"


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_nested_records_validate_and_roundtrip(stem: str) -> None:
    bundle = _bundle(stem)
    video = bundle["video_source"]
    assert isinstance(video, VideoSource)
    assert VideoSource.model_validate(video.model_dump()) == video
    for item in bundle["frame_evidence"]:
        assert isinstance(item, FrameEvidence)
        assert FrameEvidence.model_validate(item.model_dump()) == item
    for item in bundle["observations"]:
        assert isinstance(item, Observation)
        assert Observation.model_validate(item.model_dump()) == item


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_nested_records_validate_against_committed_schemas(
    stem: str,
) -> None:
    """Bundle records validate against the committed M1-01 schemas."""
    jsonschema = pytest.importorskip("jsonschema")
    schemas = {
        key: json.loads(
            (SCHEMAS_DIR / filename).read_text(encoding="utf-8")
        )
        for key, filename in (
            ("video_source", "video_source.json"),
            ("frame_evidence", "frame_evidence.json"),
            ("observation", "observation.json"),
        )
    }
    bundle = _bundle(stem)
    video = bundle["video_source"]
    assert isinstance(video, VideoSource)
    jsonschema.validate(
        instance=video.model_dump(mode="json"),
        schema=schemas["video_source"],
    )
    for item in bundle["frame_evidence"]:
        assert isinstance(item, FrameEvidence)
        jsonschema.validate(
            instance=item.model_dump(mode="json"),
            schema=schemas["frame_evidence"],
        )
    for item in bundle["observations"]:
        assert isinstance(item, Observation)
        jsonschema.validate(
            instance=item.model_dump(mode="json"),
            schema=schemas["observation"],
        )


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_no_event_keys_in_raw_text(stem: str) -> None:
    text = _raw_text(stem)
    assert "ActionEvent" not in text
    assert "WeighingEvent" not in text


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_shared_video_source_shape(stem: str) -> None:
    bundle = _bundle(stem)
    video = bundle["video_source"]
    assert isinstance(video, VideoSource)
    assert video.video_id == EXPECTED_VIDEO_IDS[stem]
    assert video.interval.start_seconds == 0.0
    assert video.interval.end_seconds == 10.0
    assert video.duration_seconds == 10.0
    assert video.fps == 25.0
    assert video.frame_count == 250
    assert video.source_dataset == "synthetic-fixture"


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_deterministic_record_ids(stem: str) -> None:
    bundle = _bundle(stem)
    frame_ids = [item.evidence_id for item in bundle["frame_evidence"]]
    observation_ids = [
        item.observation_id for item in bundle["observations"]
    ]
    assert frame_ids == EXPECTED_FRAME_IDS[stem]
    assert observation_ids == EXPECTED_OBSERVATION_IDS[stem]
    for value in frame_ids:
        assert FRAME_ID_RE.match(value) is not None


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_observation_shape_single_time_form(stem: str) -> None:
    bundle = _bundle(stem)
    assert len(bundle["observations"]) > 0
    for item in bundle["observations"]:
        assert isinstance(item, Observation)
        assert item.observation_type.value == "scale_display"
        assert item.modality.value == "visual"
        assert item.timestamp_seconds is not None
        assert item.interval is None
        assert item.scale_unit_hypothesis is None
        assert item.raw_reported_value is None
        assert isinstance(item.description, str) and item.description


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_observation_provenance_and_ambiguous_status(
    stem: str,
) -> None:
    bundle = _bundle(stem)
    for item in bundle["observations"]:
        assert isinstance(item, Observation)
        assert item.provenance.source_type.value == "human"
        assert item.provenance.source_id == f"{stem}.json"
        assert item.verification_status.value == "ambiguous"


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_timestamp_basis_synthetic_fixture_everywhere(
    stem: str,
) -> None:
    bundle = _bundle(stem)
    video = bundle["video_source"]
    assert isinstance(video, VideoSource)
    assert video.timestamp_basis.value == "synthetic_fixture"
    assert len(bundle["observations"]) > 0
    for item in bundle["observations"]:
        assert isinstance(item, Observation)
        assert item.timestamp_basis.value == "synthetic_fixture"


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_frame_index_consistent_with_fps(stem: str) -> None:
    bundle = _bundle(stem)
    video = bundle["video_source"]
    assert isinstance(video, VideoSource)
    assert video.fps is not None
    for item in bundle["frame_evidence"]:
        assert isinstance(item, FrameEvidence)
        assert item.video_id == video.video_id
        assert item.timestamp_seconds is not None
        assert item.frame_index == round(
            item.timestamp_seconds * video.fps
        )


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_check_timestamps_clean_with_nonempty_guards(
    stem: str,
) -> None:
    bundle = _bundle(stem)
    assert len(bundle["frame_evidence"]) > 0
    assert len(bundle["observations"]) > 0
    assert (
        check_timestamps(
            video_source=bundle["video_source"],
            observations=bundle["observations"],
            frame_evidence=bundle["frame_evidence"],
        )
        == []
    )


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_check_unique_ids_clean(stem: str) -> None:
    bundle = _bundle(stem)
    assert (
        check_unique_ids(
            frame_evidence=bundle["frame_evidence"],
            observations=bundle["observations"],
        )
        == []
    )


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_load_twice_id_stability(stem: str) -> None:
    first = _bundle(stem)
    second = _bundle(stem)
    assert [
        item.evidence_id for item in second["frame_evidence"]
    ] == [item.evidence_id for item in first["frame_evidence"]]
    assert [
        item.observation_id for item in second["observations"]
    ] == [item.observation_id for item in first["observations"]]


@pytest.mark.parametrize("stem", PASS_STEMS)
def test_pass_fixtures_all_five_checks_clean(stem: str) -> None:
    bundle = _bundle(stem)
    for issues in _run_all_checks(bundle):
        assert issues == []


def test_dangling_loader_accepts_structurally_invalid_bundle() -> None:
    bundle = _bundle(DANGLING_STEM)
    assert bundle["fixture_id"] == DANGLING_STEM
    assert len(bundle["frame_evidence"]) == 1
    assert len(bundle["observations"]) == 1


def test_dangling_ids_pinned_from_file() -> None:
    payload = _raw_payload(DANGLING_STEM)
    assert payload["observations"][0]["observation_id"] == "obs_dan_001"
    assert payload["observations"][0]["evidence_ids"] == [
        "frm_missing_001"
    ]
    assert "frm_missing_001" not in [
        record["evidence_id"] for record in payload["frame_evidence"]
    ]
    assert "obs_dan_001" in payload["description"]
    assert "frm_missing_001" in payload["description"]


def test_dangling_exactly_one_referential_issue() -> None:
    bundle = _bundle(DANGLING_STEM)
    assert (
        check_unique_ids(
            frame_evidence=bundle["frame_evidence"],
            observations=bundle["observations"],
        )
        == []
    )
    assert (
        check_timestamps(
            video_source=bundle["video_source"],
            observations=bundle["observations"],
            frame_evidence=bundle["frame_evidence"],
        )
        == []
    )
    issues = check_referential_integrity(
        frame_evidence=bundle["frame_evidence"],
        observations=bundle["observations"],
    )
    assert len(issues) == 1
    issue = issues[0]
    assert isinstance(issue, VerificationIssue)
    assert issue.code == "DANGLING_REFERENCE"
    assert issue.entity_id == "obs_dan_001"
    assert issue.gate == GateId.G1
    assert issue.severity == "error"
    assert "frm_missing_001" in issue.message
    assert (
        check_interval_bounds(
            video_source=bundle["video_source"],
            action_events=[],
            weighing_events=[],
        )
        == []
    )
    assert check_measurement_derivation(weighing_events=[]) == []


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_no_forbidden_runtime_tokens_in_raw_text(stem: str) -> None:
    text = _raw_text(stem)
    for token in FORBIDDEN_TOKENS:
        assert token not in text


def test_unknown_tare_description_states_absence() -> None:
    description = _raw_payload(UNKNOWN_TARE_STEM)["description"]
    assert "No tare observation; tare history unknown" in description


def test_unknown_tare_has_no_tare_history() -> None:
    bundle = _bundle(UNKNOWN_TARE_STEM)
    assert len(bundle["observations"]) == 1
    item = bundle["observations"][0]
    assert isinstance(item, Observation)
    assert item.observation_id == "obs_unk_001"
    assert item.timestamp_seconds == 4.8
    assert item.scale_value == 150.0
    assert item.scale_unit == "g"
    assert item.display_state is not None
    assert item.display_state.value == "stable"
    assert item.evidence_ids == ["frm_unk_001", "frm_unk_002"]
    for candidate in bundle["observations"]:
        assert candidate.display_state is None or (
            candidate.display_state.value != "zero"
        )
        assert candidate.scale_value != 0.0


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_no_confirmed_status_in_raw_text(stem: str) -> None:
    assert "confirmed" not in _raw_text(stem)


def test_nominal_tare_and_mass_readings() -> None:
    bundle = _bundle(NOMINAL_STEM)
    frames = bundle["frame_evidence"]
    assert [(item.timestamp_seconds, item.frame_index) for item in frames] == [
        (2.0, 50),
        (4.8, 120),
        (5.0, 125),
    ]
    by_id = {item.observation_id: item for item in bundle["observations"]}
    assert sorted(by_id) == ["obs_nom_mass", "obs_nom_tare"]
    tare = by_id["obs_nom_tare"]
    assert tare.timestamp_seconds == 2.0
    assert tare.scale_value == 0.0
    assert tare.scale_unit == "g"
    assert tare.display_state is not None
    assert tare.display_state.value == "zero"
    assert tare.evidence_ids == ["frm_nom_001"]
    mass = by_id["obs_nom_mass"]
    assert mass.timestamp_seconds == 4.8
    assert mass.scale_value == 120.0
    assert mass.scale_unit == "g"
    assert mass.display_state is not None
    assert mass.display_state.value == "stable"
    assert mass.evidence_ids == ["frm_nom_002", "frm_nom_003"]


def test_nominal_stable_reading_has_two_evidence_ids() -> None:
    bundle = _bundle(NOMINAL_STEM)
    by_id = {item.observation_id: item for item in bundle["observations"]}
    assert len(by_id["obs_nom_mass"].evidence_ids) >= 2


def test_conflicting_readings_preserved() -> None:
    bundle = _bundle(CONFLICTING_STEM)
    frames = bundle["frame_evidence"]
    assert [item.evidence_id for item in frames] == [
        "frm_con_001",
        "frm_con_002",
    ]
    assert frames[0].timestamp_seconds == 4.8
    assert frames[1].timestamp_seconds == 4.8
    assert frames[0].frame_index == 120
    assert frames[1].frame_index == 120
    by_id = {item.observation_id: item for item in bundle["observations"]}
    assert sorted(by_id) == ["obs_con_001", "obs_con_002"]
    first = by_id["obs_con_001"]
    second = by_id["obs_con_002"]
    assert first.scale_value == 54.0
    assert second.scale_value == 58.0
    assert first.scale_value != second.scale_value
    assert first.scale_unit == "g"
    assert second.scale_unit == "g"
    assert first.evidence_ids == ["frm_con_001"]
    assert second.evidence_ids == ["frm_con_002"]


def test_conflicting_no_resolution_tokens() -> None:
    text = _raw_text(CONFLICTING_STEM)
    assert "consensus" not in text
    assert "56.0" not in text


def test_partial_budget_shape_counts_and_order() -> None:
    bundle = _bundle(PARTIAL_BUDGET_STEM)
    frames = bundle["frame_evidence"]
    observations = bundle["observations"]
    assert len(frames) == 3
    assert len(observations) == 3
    assert [(item.timestamp_seconds, item.frame_index) for item in frames] == [
        (1.0, 25),
        (4.8, 120),
        (7.0, 175),
    ]
    assert [item.timestamp_seconds for item in observations] == [
        1.0,
        4.8,
        7.0,
    ]
    assert sum(len(item.evidence_ids) for item in observations) == 3
    assert observations[0].scale_value == 0.0
    assert observations[0].display_state is not None
    assert observations[0].display_state.value == "zero"
    assert observations[1].scale_value == 120.0
    assert observations[2].scale_value == 120.0


def test_partial_budget_records_carry_no_runtime_tokens() -> None:
    payload = _raw_payload(PARTIAL_BUDGET_STEM)
    assert "BUDGET_EXHAUSTED" in payload["description"]
    assert "partial-BUDGET_EXHAUSTED" in payload["description"]
    without_description = {
        key: value
        for key, value in payload.items()
        if key != "description"
    }
    scanned = json.dumps(without_description, sort_keys=True)
    for token in ("BUDGET_EXHAUSTED", "completed", "COMPLETED", "PARTIAL"):
        assert token not in scanned


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_load_twice_equal_bundles(stem: str) -> None:
    assert _dump_bundle(_bundle(stem)) == _dump_bundle(_bundle(stem))


def test_sorted_fixture_processing() -> None:
    paths = _all_fixture_paths()
    assert paths == sorted(paths)
    assert [path.stem for path in paths] == FIXTURE_STEMS


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_fixture_bytes_unchanged_by_load_and_checks(
    stem: str,
) -> None:
    path = _fixture_path(stem)
    before = path.read_bytes()
    bundle = load_fixture(path)
    _run_all_checks(bundle)
    assert path.read_bytes() == before


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_checks_do_not_mutate_bundles(stem: str) -> None:
    bundle = _bundle(stem)
    before = _dump_bundle(bundle)
    _run_all_checks(bundle)
    assert _dump_bundle(bundle) == before


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_file_formatting_lf_two_space_indent(stem: str) -> None:
    raw = _fixture_path(stem).read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    assert raw.split(b"\n")[1].startswith(b'  "')


def _write_drift_copy(
    stem: str, tmp_path: Path, mutate: Any
) -> Path:
    source = _fixture_path(stem)
    target = tmp_path / source.name
    payload = json.loads(source.read_text(encoding="utf-8"))
    mutate(payload)
    target.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return target


def test_malformed_json_rejected(tmp_path: Path) -> None:
    path = tmp_path / f"{NOMINAL_STEM}.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(FixtureError) as exc_info:
        load_fixture(path)
    _assert_fixture_invalid(exc_info.value)
    assert exc_info.value.entity_id == NOMINAL_STEM


def test_missing_video_source_rejected(tmp_path: Path) -> None:
    def _mutate(payload: dict[str, Any]) -> None:
        del payload["video_source"]

    path = _write_drift_copy(NOMINAL_STEM, tmp_path, _mutate)
    with pytest.raises(FixtureError) as exc_info:
        load_fixture(path)
    _assert_fixture_invalid(exc_info.value)


def test_wrong_envelope_schema_version_rejected(
    tmp_path: Path,
) -> None:
    def _mutate(payload: dict[str, Any]) -> None:
        payload["schema_version"] = "2.0.0"

    path = _write_drift_copy(NOMINAL_STEM, tmp_path, _mutate)
    with pytest.raises(FixtureError) as exc_info:
        load_fixture(path)
    _assert_fixture_invalid(exc_info.value)


def test_nested_negative_timestamp_rejected_with_cause(
    tmp_path: Path,
) -> None:
    def _mutate(payload: dict[str, Any]) -> None:
        payload["observations"][0]["timestamp_seconds"] = -0.5

    path = _write_drift_copy(UNKNOWN_TARE_STEM, tmp_path, _mutate)
    with pytest.raises(FixtureError) as exc_info:
        load_fixture(path)
    _assert_fixture_invalid(exc_info.value)
    assert exc_info.value.entity_id == "obs_unk_001"
    assert isinstance(exc_info.value.__cause__, ValidationError)


def test_unknown_envelope_key_rejected(tmp_path: Path) -> None:
    def _mutate(payload: dict[str, Any]) -> None:
        payload["extra_key"] = 1

    path = _write_drift_copy(NOMINAL_STEM, tmp_path, _mutate)
    with pytest.raises(FixtureError) as exc_info:
        load_fixture(path)
    _assert_fixture_invalid(exc_info.value)


@pytest.mark.parametrize(
    "section",
    ["video_source", "frame_evidence", "observations"],
)
def test_direct_model_rejects_unexpected_field(section: str) -> None:
    """Unknown nested fields fail direct model validation.

    Via the loader every failure (including nested ones) surfaces as
    ``FixtureError``; only direct ``model_validate`` calls surface the
    underlying ``ValidationError``.
    """
    payload = _raw_payload(NOMINAL_STEM)
    if section == "video_source":
        record = dict(payload["video_source"])
        model_cls: Any = VideoSource
    elif section == "frame_evidence":
        record = dict(payload["frame_evidence"][0])
        model_cls = FrameEvidence
    else:
        record = dict(payload["observations"][0])
        model_cls = Observation
    record["typo_field"] = "oops"
    with pytest.raises(ValidationError):
        model_cls.model_validate(record)


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_source_ref_relative_synthetic_no_real_media(
    stem: str,
) -> None:
    bundle = _bundle(stem)
    video = bundle["video_source"]
    assert isinstance(video, VideoSource)
    assert video.source_ref == f"synthetic/{stem}.mp4"
    assert "\\" not in video.source_ref
    assert not video.source_ref.startswith("/")
    assert not re.match(r"^[A-Za-z]:", video.source_ref)
    text = _raw_text(stem)
    for token in ("data/raw/", "EPIC-KITCHENS", "epic-kitchens"):
        assert token not in text
    assert "http://" not in text
    assert "https://" not in text


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_no_secrets_or_provider_markers_in_raw_text(
    stem: str,
) -> None:
    lowered = _raw_text(stem).lower()
    for token in (
        "api_key",
        "apikey",
        "api-key",
        "secret",
        "bearer",
        "password",
        "openai",
        "anthropic",
        "baseten",
    ):
        assert token not in lowered


@pytest.mark.parametrize("stem", FIXTURE_STEMS)
def test_image_ref_null_everywhere(stem: str) -> None:
    bundle = _bundle(stem)
    assert len(bundle["frame_evidence"]) > 0
    for item in bundle["frame_evidence"]:
        assert isinstance(item, FrameEvidence)
        assert item.image_ref is None


def test_imports_limited_to_allowlist() -> None:
    allowed = frozenset(
        {
            "__future__",
            "json",
            "re",
            "pathlib",
            "typing",
            "pytest",
            "pydantic",
            "bakery_intelligence",
        }
    )
    roots: set[str] = set()
    text = Path(__file__).read_text(encoding="utf-8")
    for line in text.splitlines():
        match = re.match(r"\s*import\s+([A-Za-z0-9_.]+)", line)
        if match is not None:
            roots.add(match.group(1).split(".")[0])
            continue
        match = re.match(
            r"\s*from\s+([A-Za-z0-9_.]+)\s+import\s+", line
        )
        if match is not None:
            roots.add(match.group(1).split(".")[0])
    assert roots
    assert roots <= allowed, (
        f"unexpected import roots: {sorted(roots - allowed)}"
    )
