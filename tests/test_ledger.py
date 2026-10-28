"""M2-02 append-only observation ledger tests.

Covers the frozen M2-02 decisions: 0-based ``ledger_seq`` assigned only by
:func:`append`, strict positional ``read_ledger`` validation, link rules
for ``supersedes`` / ``contradiction_of``, verbatim observation
preservation, snapshot derivation that never drops entries, and the
closed ``LedgerError`` code contract.

Oracle sources: every test case and expected outcome comes from the
frozen M2-02 orchestrator decisions (F1-F13), never from running the
ledger code. Literal ids, scale values, and timestamps were pinned by
reading the committed ``fixtures/synthetic/*.json`` files. Fixture
observations are loaded minimally via ``Observation.model_validate``;
nothing is imported from ``test_fixtures``.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pytest
from pydantic import ValidationError

import bakery_intelligence.ledger as ledger_pkg
from bakery_intelligence.ledger import (
    LedgerEntry,
    LedgerError,
    LedgerSnapshot,
    SnapshotEntry,
    append,
    derive_snapshot,
    read_ledger,
    utcnow,
    write_snapshot,
)
from bakery_intelligence.models import Observation

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REPO_ROOT / "fixtures" / "synthetic"
LEDGER_DIR = REPO_ROOT / "src" / "bakery_intelligence" / "ledger"

CONFLICTING_STEM = "synthetic_interval_conflicting"
NOMINAL_STEM = "synthetic_interval_nominal"

# Pinned by reading fixtures/synthetic/synthetic_interval_conflicting.json.
CON_001_ID = "obs_con_001"
CON_002_ID = "obs_con_002"
CON_001_VALUE = 54.0
CON_002_VALUE = 58.0
CON_TIMESTAMP = 4.8

# Pinned by reading fixtures/synthetic/synthetic_interval_nominal.json.
NOM_TARE_ID = "obs_nom_tare"
NOM_MASS_ID = "obs_nom_mass"
NOM_TARE_VALUE = 0.0
NOM_MASS_VALUE = 120.0
NOM_TARE_TIMESTAMP = 2.0
NOM_MASS_TIMESTAMP = 4.8

FIXED_APPENDED_AT = datetime(2026, 3, 4, 5, 6, 7, tzinfo=timezone.utc)
FIXED_GENERATED_AT = datetime(2026, 3, 5, 6, 7, 8, tzinfo=timezone.utc)

# Runtime-result vocabulary forbidden anywhere in stored ledger lines.
FORBIDDEN_LEDGER_TOKENS = (
    "net_ingredient_quantity",
    "derivation",
    "measurement_mode",
    "action_ids",
    "consensus",
    "confirmed",
)


def _clock(value: datetime) -> Callable[[], datetime]:
    return lambda: value


def _fixture_observations(stem: str) -> list[Observation]:
    payload = json.loads(
        (FIXTURES_DIR / f"{stem}.json").read_text(encoding="utf-8")
    )
    assert isinstance(payload, dict)
    return [
        Observation.model_validate(item)
        for item in payload["observations"]
    ]


def _fixture_observation(stem: str, observation_id: str) -> Observation:
    for item in _fixture_observations(stem):
        if item.observation_id == observation_id:
            return item
    raise AssertionError(f"{observation_id} not found in {stem}")


def _make_observation(
    observation_id: str,
    *,
    timestamp_seconds: float | None = 4.8,
    interval: dict[str, float] | None = None,
    scale_value: float | None = 10.0,
    scale_unit: str | None = "g",
    scale_unit_hypothesis: str | None = None,
    source_type: str = "adapter",
    source_id: str = "test-ledger",
    verification_status: str = "ambiguous",
    display_state: str | None = "stable",
) -> Observation:
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "observation_id": observation_id,
        "modality": "visual",
        "timestamp_basis": "synthetic_fixture",
        "observation_type": "scale_display",
        "scale_value": scale_value,
        "scale_unit": scale_unit,
        "scale_unit_hypothesis": scale_unit_hypothesis,
        "display_state": display_state,
        "evidence_ids": [],
        "raw_reported_value": None,
        "description": f"Ledger test observation {observation_id}.",
        "provenance": {
            "source_type": source_type,
            "source_id": source_id,
        },
        "verification_status": verification_status,
    }
    if interval is None:
        payload["timestamp_seconds"] = timestamp_seconds
    else:
        payload["interval"] = interval
    return Observation.model_validate(payload)


def _raw_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _raw_lines(path: Path) -> list[str]:
    return _raw_text(path).split("\n")[:-1]


def _parsed_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in _raw_lines(path)]


def _snapshot_by_id(snapshot: LedgerSnapshot) -> dict[str, SnapshotEntry]:
    return {item.observation.observation_id: item for item in snapshot.entries}


def _assert_ledger_error(
    error: LedgerError,
    *,
    code: str,
    entity_id: str,
    line_no: int | None = None,
) -> None:
    assert error.code == code
    assert error.entity_id == entity_id
    assert error.line_no == line_no
    assert isinstance(error.message, str) and error.message


def _write_drifted_ledger(
    path: Path, lines: list[str], *, trailing_newline: bool = True
) -> None:
    text = "\n".join(lines)
    if trailing_newline:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _mutate_line_payload(
    path: Path, index: int, mutate: Callable[[dict[str, Any]], None]
) -> None:
    lines = _raw_lines(path)
    payload = json.loads(lines[index])
    mutate(payload)
    lines[index] = json.dumps(payload)
    _write_drifted_ledger(path, lines)


def _seed_three_line_ledger(path: Path) -> list[LedgerEntry]:
    clock = _clock(FIXED_APPENDED_AT)
    return [
        append(
            path,
            _fixture_observation(NOMINAL_STEM, NOM_TARE_ID),
            clock=clock,
        ),
        append(
            path,
            _fixture_observation(NOMINAL_STEM, NOM_MASS_ID),
            clock=clock,
        ),
        append(
            path,
            _fixture_observation(CONFLICTING_STEM, CON_001_ID),
            clock=clock,
        ),
    ]


def test_three_appends_persist_three_parseable_jsonl_lines(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    clock = _clock(FIXED_APPENDED_AT)
    first = append(
        path, _fixture_observation(NOMINAL_STEM, NOM_TARE_ID), clock=clock
    )
    second = append(
        path, _fixture_observation(NOMINAL_STEM, NOM_MASS_ID), clock=clock
    )
    third = append(
        path, _fixture_observation(CONFLICTING_STEM, CON_001_ID), clock=clock
    )
    assert (first.ledger_seq, second.ledger_seq, third.ledger_seq) == (0, 1, 2)
    assert _raw_text(path).endswith("\n")
    lines = _raw_lines(path)
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    assert [item["ledger_seq"] for item in parsed] == [0, 1, 2]
    assert [item["observation"]["observation_id"] for item in parsed] == [
        NOM_TARE_ID,
        NOM_MASS_ID,
        CON_001_ID,
    ]


def test_reload_round_trip_preserves_entries_and_nulls(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    stored = _seed_three_line_ledger(path)
    reloaded = read_ledger(path)
    assert len(reloaded) == 3
    for expected, actual in zip(stored, reloaded):
        assert actual.model_dump(mode="json") == expected.model_dump(
            mode="json"
        )
    for payload in _parsed_lines(path):
        observation = payload["observation"]
        assert observation["scale_unit_hypothesis"] is None
        assert observation["raw_reported_value"] is None
        assert payload["supersedes"] is None
        assert payload["contradiction_of"] is None
        assert payload["reason"] is None


def test_out_of_video_order_appends_keep_append_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    clock = _clock(FIXED_APPENDED_AT)
    mass = _fixture_observation(NOMINAL_STEM, NOM_MASS_ID)
    tare = _fixture_observation(NOMINAL_STEM, NOM_TARE_ID)
    assert mass.timestamp_seconds is not None
    assert tare.timestamp_seconds is not None
    assert mass.timestamp_seconds > tare.timestamp_seconds
    append(path, mass, clock=clock)
    append(path, tare, clock=clock)
    reloaded = read_ledger(path)
    assert [item.ledger_seq for item in reloaded] == [0, 1]
    assert [item.observation.observation_id for item in reloaded] == [
        NOM_MASS_ID,
        NOM_TARE_ID,
    ]
    snapshot = derive_snapshot(reloaded, clock=_clock(FIXED_GENERATED_AT))
    assert [item.observation.observation_id for item in snapshot.entries] == [
        NOM_MASS_ID,
        NOM_TARE_ID,
    ]


def test_later_appends_preserve_existing_bytes_prefix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    clock = _clock(FIXED_APPENDED_AT)
    append(path, _fixture_observation(NOMINAL_STEM, NOM_TARE_ID), clock=clock)
    append(path, _fixture_observation(NOMINAL_STEM, NOM_MASS_ID), clock=clock)
    before = path.read_bytes()
    first_two_lines = _raw_lines(path)
    append(path, _fixture_observation(CONFLICTING_STEM, CON_001_ID), clock=clock)
    after = path.read_bytes()
    assert after.startswith(before)
    assert _raw_lines(path)[:2] == first_two_lines
    assert len(_raw_lines(path)) == 3


def test_duplicate_id_rejected_with_history_intact(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    stored = _seed_three_line_ledger(path)
    before = path.read_bytes()
    with pytest.raises(LedgerError) as exc_info:
        append(
            path,
            _fixture_observation(NOMINAL_STEM, NOM_MASS_ID),
            clock=_clock(FIXED_APPENDED_AT),
        )
    _assert_ledger_error(exc_info.value, code="duplicate_id", entity_id=NOM_MASS_ID)
    assert path.read_bytes() == before
    reloaded = read_ledger(path)
    assert [item.observation.observation_id for item in reloaded] == [
        item.observation.observation_id for item in stored
    ]
    assert [item.model_dump(mode="json") for item in reloaded] == [
        item.model_dump(mode="json") for item in stored
    ]


def test_correction_supersedes_chain_preserves_both_lines(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    clock = _clock(FIXED_APPENDED_AT)
    original = _fixture_observation(CONFLICTING_STEM, CON_001_ID)
    append(path, original, clock=clock)
    first_line_before = _raw_lines(path)[0]
    reason = "Re-read display as 54.5 g after zoom; keeps original visible."
    reread = _make_observation("obs_cor_002", scale_value=54.5)
    stored = append(
        path, reread, supersedes=CON_001_ID, reason=reason, clock=clock
    )
    assert stored.ledger_seq == 1
    assert stored.supersedes == CON_001_ID
    assert stored.reason == reason
    entries = read_ledger(path)
    assert [item.observation.observation_id for item in entries] == [
        CON_001_ID,
        "obs_cor_002",
    ]
    assert _raw_lines(path)[0] == first_line_before
    assert entries[0].observation.scale_value == CON_001_VALUE
    assert entries[1].observation.scale_value == 54.5
    snapshot = derive_snapshot(entries, clock=_clock(FIXED_GENERATED_AT))
    by_id = _snapshot_by_id(snapshot)
    assert by_id["obs_cor_002"].supersedes == CON_001_ID
    assert by_id[CON_001_ID].superseded_by == "obs_cor_002"
    assert by_id["obs_cor_002"].reason == reason
    assert by_id[CON_001_ID].is_current is False
    assert by_id["obs_cor_002"].is_current is True
    assert by_id[CON_001_ID].observation.verification_status.value == "ambiguous"
    assert (
        by_id["obs_cor_002"].observation.verification_status.value == "ambiguous"
    )


def test_contradiction_pair_preserved_without_resolution(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    clock = _clock(FIXED_APPENDED_AT)
    append(path, _fixture_observation(CONFLICTING_STEM, CON_001_ID), clock=clock)
    append(
        path,
        _fixture_observation(CONFLICTING_STEM, CON_002_ID),
        contradiction_of=CON_001_ID,
        clock=clock,
    )
    entries = read_ledger(path)
    assert [item.observation.observation_id for item in entries] == [
        CON_001_ID,
        CON_002_ID,
    ]
    assert entries[1].contradiction_of == CON_001_ID
    assert entries[0].observation.scale_value == CON_001_VALUE
    assert entries[1].observation.scale_value == CON_002_VALUE
    snapshot = derive_snapshot(entries, clock=_clock(FIXED_GENERATED_AT))
    assert [item.observation.observation_id for item in snapshot.entries] == [
        CON_001_ID,
        CON_002_ID,
    ]
    text = _raw_text(path)
    assert "56.0" not in text
    assert "consensus" not in text


def test_observation_provenance_and_status_preserved_verbatim(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    clock = _clock(FIXED_APPENDED_AT)
    human = _fixture_observation(CONFLICTING_STEM, CON_001_ID)
    model = _make_observation(
        "obs_mod_001",
        source_type="model",
        source_id="run-7",
        verification_status="ambiguous",
    )
    adapter = _make_observation(
        "obs_adp_001",
        source_type="adapter",
        source_id="adapter-1.0",
        verification_status="unobservable",
    )
    for item in (human, model, adapter):
        append(path, item, clock=clock)
    entries = read_ledger(path)
    for expected, actual in zip((human, model, adapter), entries):
        assert actual.observation.model_dump(mode="json") == (
            expected.model_dump(mode="json")
        )
    assert entries[0].observation.provenance.source_type.value == "human"
    assert entries[1].observation.provenance.source_type.value == "model"
    assert entries[1].observation.provenance.source_id == "run-7"
    assert entries[2].observation.provenance.source_type.value == "adapter"
    assert entries[2].observation.verification_status.value == "unobservable"


def test_none_scale_unit_and_hypothesis_never_promoted(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    candidate = _make_observation(
        "obs_hyp_001", scale_unit=None, scale_unit_hypothesis="g"
    )
    append(path, candidate, clock=_clock(FIXED_APPENDED_AT))
    (entry,) = read_ledger(path)
    assert entry.observation.model_dump(mode="json") == candidate.model_dump(
        mode="json"
    )
    assert entry.observation.scale_unit is None
    assert entry.observation.scale_unit_hypothesis == "g"
    assert _parsed_lines(path)[0]["observation"]["scale_unit"] is None


def test_incoming_confirmed_passes_through_unpoliced(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    candidate = _make_observation(
        "obs_cfm_001", verification_status="confirmed"
    )
    append(path, candidate, clock=_clock(FIXED_APPENDED_AT))
    (entry,) = read_ledger(path)
    assert entry.observation.verification_status.value == "confirmed"
    assert entry.observation.model_dump(mode="json") == candidate.model_dump(
        mode="json"
    )


def test_unknown_supersedes_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append(
        path,
        _fixture_observation(NOMINAL_STEM, NOM_TARE_ID),
        clock=_clock(FIXED_APPENDED_AT),
    )
    before = path.read_bytes()
    with pytest.raises(LedgerError) as exc_info:
        append(
            path,
            _make_observation("obs_new_001"),
            supersedes="obs_missing_001",
            reason="targets nothing",
            clock=_clock(FIXED_APPENDED_AT),
        )
    _assert_ledger_error(
        exc_info.value, code="unknown_supersedes", entity_id="obs_new_001"
    )
    assert path.read_bytes() == before
    assert [item.observation.observation_id for item in read_ledger(path)] == [
        NOM_TARE_ID
    ]


def test_self_link_via_supersedes_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append(
        path,
        _fixture_observation(NOMINAL_STEM, NOM_TARE_ID),
        clock=_clock(FIXED_APPENDED_AT),
    )
    before = path.read_bytes()
    with pytest.raises(LedgerError) as exc_info:
        append(
            path,
            _make_observation("obs_self_001"),
            supersedes="obs_self_001",
            reason="self target",
            clock=_clock(FIXED_APPENDED_AT),
        )
    _assert_ledger_error(
        exc_info.value, code="self_link", entity_id="obs_self_001"
    )
    assert path.read_bytes() == before
    assert [item.observation.observation_id for item in read_ledger(path)] == [
        NOM_TARE_ID
    ]


def test_self_link_via_contradiction_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append(
        path,
        _fixture_observation(NOMINAL_STEM, NOM_TARE_ID),
        clock=_clock(FIXED_APPENDED_AT),
    )
    before = path.read_bytes()
    with pytest.raises(LedgerError) as exc_info:
        append(
            path,
            _make_observation("obs_self_002"),
            contradiction_of="obs_self_002",
            clock=_clock(FIXED_APPENDED_AT),
        )
    _assert_ledger_error(
        exc_info.value, code="self_link", entity_id="obs_self_002"
    )
    assert path.read_bytes() == before
    assert [item.observation.observation_id for item in read_ledger(path)] == [
        NOM_TARE_ID
    ]


def test_unknown_contradiction_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append(
        path,
        _fixture_observation(NOMINAL_STEM, NOM_TARE_ID),
        clock=_clock(FIXED_APPENDED_AT),
    )
    before = path.read_bytes()
    with pytest.raises(LedgerError) as exc_info:
        append(
            path,
            _make_observation("obs_new_002"),
            contradiction_of="obs_missing_002",
            clock=_clock(FIXED_APPENDED_AT),
        )
    _assert_ledger_error(
        exc_info.value, code="unknown_contradiction", entity_id="obs_new_002"
    )
    assert path.read_bytes() == before
    assert [item.observation.observation_id for item in read_ledger(path)] == [
        NOM_TARE_ID
    ]


def test_already_superseded_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    clock = _clock(FIXED_APPENDED_AT)
    append(path, _fixture_observation(NOMINAL_STEM, NOM_TARE_ID), clock=clock)
    append(
        path,
        _make_observation("obs_rev_001"),
        supersedes=NOM_TARE_ID,
        reason="first refinement",
        clock=clock,
    )
    before = path.read_bytes()
    with pytest.raises(LedgerError) as exc_info:
        append(
            path,
            _make_observation("obs_rev_002"),
            supersedes=NOM_TARE_ID,
            reason="second refinement of the same target",
            clock=clock,
        )
    _assert_ledger_error(
        exc_info.value, code="already_superseded", entity_id="obs_rev_002"
    )
    assert path.read_bytes() == before
    assert [item.observation.observation_id for item in read_ledger(path)] == [
        NOM_TARE_ID,
        "obs_rev_001",
    ]


def test_missing_reason_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append(
        path,
        _fixture_observation(NOMINAL_STEM, NOM_TARE_ID),
        clock=_clock(FIXED_APPENDED_AT),
    )
    for observation_id, reason in (
        ("obs_nor_001", None),
        ("obs_nor_002", "   "),
    ):
        before = path.read_bytes()
        with pytest.raises(LedgerError) as exc_info:
            append(
                path,
                _make_observation(observation_id),
                supersedes=NOM_TARE_ID,
                reason=reason,
                clock=_clock(FIXED_APPENDED_AT),
            )
        _assert_ledger_error(
            exc_info.value, code="missing_reason", entity_id=observation_id
        )
        assert path.read_bytes() == before
    assert [item.observation.observation_id for item in read_ledger(path)] == [
        NOM_TARE_ID
    ]


def test_reason_without_supersedes_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    append(
        path,
        _fixture_observation(NOMINAL_STEM, NOM_TARE_ID),
        clock=_clock(FIXED_APPENDED_AT),
    )
    before = path.read_bytes()
    with pytest.raises(LedgerError) as exc_info:
        append(
            path,
            _make_observation("obs_rws_001"),
            reason="orphan reason",
            clock=_clock(FIXED_APPENDED_AT),
        )
    _assert_ledger_error(
        exc_info.value,
        code="reason_without_supersedes",
        entity_id="obs_rws_001",
    )
    assert path.read_bytes() == before
    assert [item.observation.observation_id for item in read_ledger(path)] == [
        NOM_TARE_ID
    ]


def test_supersedes_edges_point_backward_and_chain_snapshot(
    tmp_path: Path,
) -> None:
    """Three-link refinement chain: edges point to strictly earlier seq.

    Cycles are unconstructable because targets must pre-exist, so no
    cycle-construction case exists; the backward-edge invariant and the
    chained reverse index are asserted instead.
    """
    path = tmp_path / "ledger.jsonl"
    clock = _clock(FIXED_APPENDED_AT)
    append(path, _make_observation("obs_chain_001"), clock=clock)
    append(
        path,
        _make_observation("obs_chain_002"),
        supersedes="obs_chain_001",
        reason="first refinement",
        clock=clock,
    )
    append(
        path,
        _make_observation("obs_chain_003"),
        supersedes="obs_chain_002",
        reason="second refinement",
        clock=clock,
    )
    entries = read_ledger(path)
    seq_by_id = {
        item.observation.observation_id: item.ledger_seq for item in entries
    }
    for item in entries:
        if item.supersedes is not None:
            assert seq_by_id[item.supersedes] < item.ledger_seq
    snapshot = derive_snapshot(entries, clock=_clock(FIXED_GENERATED_AT))
    by_id = _snapshot_by_id(snapshot)
    assert by_id["obs_chain_001"].superseded_by == "obs_chain_002"
    assert by_id["obs_chain_002"].superseded_by == "obs_chain_003"
    assert by_id["obs_chain_003"].superseded_by is None
    assert [item.is_current for item in snapshot.entries] == [
        False,
        False,
        True,
    ]
    assert by_id["obs_chain_002"].reason == "first refinement"
    assert by_id["obs_chain_003"].reason == "second refinement"
    assert [item.observation.observation_id for item in snapshot.entries] == [
        "obs_chain_001",
        "obs_chain_002",
        "obs_chain_003",
    ]


def test_malformed_json_line_two_is_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "drift.jsonl"
    _seed_three_line_ledger(path)
    lines = _raw_lines(path)
    lines[1] = "{not json"
    _write_drifted_ledger(path, lines)
    with pytest.raises(LedgerError) as exc_info:
        read_ledger(path)
    _assert_ledger_error(
        exc_info.value,
        code="corrupt_line",
        entity_id="drift.jsonl:2",
        line_no=2,
    )
    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)


def test_invalid_point_observation_line_two_is_corrupt(
    tmp_path: Path,
) -> None:
    path = tmp_path / "drift.jsonl"
    _seed_three_line_ledger(path)

    def _break_timestamp(payload: dict[str, Any]) -> None:
        payload["observation"]["timestamp_seconds"] = -1.0

    _mutate_line_payload(path, 1, _break_timestamp)
    with pytest.raises(LedgerError) as exc_info:
        read_ledger(path)
    _assert_ledger_error(
        exc_info.value,
        code="corrupt_line",
        entity_id="drift.jsonl:2",
        line_no=2,
    )
    assert isinstance(exc_info.value.__cause__, ValidationError)


def test_invalid_interval_observation_line_two_is_corrupt(
    tmp_path: Path,
) -> None:
    path = tmp_path / "drift.jsonl"
    clock = _clock(FIXED_APPENDED_AT)
    append(path, _fixture_observation(NOMINAL_STEM, NOM_TARE_ID), clock=clock)
    append(
        path,
        _make_observation(
            "obs_int_001",
            timestamp_seconds=None,
            interval={"start_seconds": 1.0, "end_seconds": 2.0},
        ),
        clock=clock,
    )
    append(path, _fixture_observation(NOMINAL_STEM, NOM_MASS_ID), clock=clock)

    def _break_interval(payload: dict[str, Any]) -> None:
        payload["observation"]["interval"]["end_seconds"] = 1.0

    _mutate_line_payload(path, 1, _break_interval)
    with pytest.raises(LedgerError) as exc_info:
        read_ledger(path)
    _assert_ledger_error(
        exc_info.value,
        code="corrupt_line",
        entity_id="drift.jsonl:2",
        line_no=2,
    )
    assert isinstance(exc_info.value.__cause__, ValidationError)


@pytest.mark.parametrize("drifted_seq", [1, 7])
def test_bad_seq_line_three_is_corrupt(
    tmp_path: Path, drifted_seq: int
) -> None:
    path = tmp_path / "drift.jsonl"
    _seed_three_line_ledger(path)

    def _break_seq(payload: dict[str, Any]) -> None:
        payload["ledger_seq"] = drifted_seq

    _mutate_line_payload(path, 2, _break_seq)
    assert json.loads(_raw_lines(path)[2])["ledger_seq"] == drifted_seq
    with pytest.raises(LedgerError) as exc_info:
        read_ledger(path)
    _assert_ledger_error(
        exc_info.value,
        code="corrupt_line",
        entity_id="drift.jsonl:3",
        line_no=3,
    )


def test_blank_middle_line_is_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "drift.jsonl"
    _seed_three_line_ledger(path)
    lines = _raw_lines(path)
    lines[1] = ""
    _write_drifted_ledger(path, lines)
    with pytest.raises(LedgerError) as exc_info:
        read_ledger(path)
    _assert_ledger_error(
        exc_info.value,
        code="corrupt_line",
        entity_id="drift.jsonl:2",
        line_no=2,
    )


def test_dangling_supersedes_target_is_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "drift.jsonl"
    _seed_three_line_ledger(path)

    def _break_target(payload: dict[str, Any]) -> None:
        payload["supersedes"] = "obs_missing_999"
        payload["reason"] = "dangling link"

    _mutate_line_payload(path, 1, _break_target)
    with pytest.raises(LedgerError) as exc_info:
        read_ledger(path)
    _assert_ledger_error(
        exc_info.value,
        code="corrupt_line",
        entity_id="drift.jsonl:2",
        line_no=2,
    )


def test_doubly_superseded_target_is_corrupt(tmp_path: Path) -> None:
    path = tmp_path / "drift.jsonl"
    _seed_three_line_ledger(path)

    def _link_to_first(payload: dict[str, Any]) -> None:
        payload["supersedes"] = NOM_TARE_ID
        payload["reason"] = "competing refinement"

    _mutate_line_payload(path, 1, _link_to_first)
    _mutate_line_payload(path, 2, _link_to_first)
    with pytest.raises(LedgerError) as exc_info:
        read_ledger(path)
    _assert_ledger_error(
        exc_info.value,
        code="corrupt_line",
        entity_id="drift.jsonl:3",
        line_no=3,
    )


def test_empty_file_yields_empty_ledger_and_snapshot(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_bytes(b"")
    assert read_ledger(path) == []
    snapshot = derive_snapshot([], clock=_clock(FIXED_GENERATED_AT))
    assert snapshot.schema_version == "1.0.0"
    assert snapshot.generated_at == FIXED_GENERATED_AT
    assert snapshot.entries == []


def test_missing_file_is_corrupt_line_and_read_creates_nothing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing.jsonl"
    assert not path.exists()
    with pytest.raises(LedgerError) as exc_info:
        read_ledger(path)
    _assert_ledger_error(
        exc_info.value,
        code="corrupt_line",
        entity_id="missing.jsonl",
        line_no=None,
    )
    assert not path.exists()


def test_snapshot_regeneration_identical_across_fresh_reads(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    first_clock = _clock(datetime(2026, 1, 1, tzinfo=timezone.utc))
    second_clock = _clock(datetime(2026, 6, 1, tzinfo=timezone.utc))
    for path, clock in ((first_path, first_clock), (second_path, second_clock)):
        append(
            path, _fixture_observation(NOMINAL_STEM, NOM_TARE_ID), clock=clock
        )
        append(
            path, _fixture_observation(NOMINAL_STEM, NOM_MASS_ID), clock=clock
        )
    generated = _clock(FIXED_GENERATED_AT)
    first = derive_snapshot(read_ledger(first_path), clock=generated)
    second = derive_snapshot(read_ledger(first_path), clock=generated)
    assert second.model_dump(mode="json") == first.model_dump(mode="json")
    third = derive_snapshot(read_ledger(second_path), clock=generated)

    def _without_appended_at(snapshot: LedgerSnapshot) -> list[dict[str, Any]]:
        dumped = snapshot.model_dump(mode="json")
        for item in dumped["entries"]:
            del item["appended_at"]
        return dumped["entries"]

    assert _without_appended_at(third) == _without_appended_at(first)


def test_write_snapshot_is_deterministic_json(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    _seed_three_line_ledger(ledger_path)
    snapshot = derive_snapshot(
        read_ledger(ledger_path), clock=_clock(FIXED_GENERATED_AT)
    )
    snapshot_path = tmp_path / "nested" / "dir" / "observations.json"
    write_snapshot(snapshot_path, snapshot)
    raw = snapshot_path.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    assert raw.split(b"\n")[1].startswith(b'  "')
    assert json.loads(raw.decode("utf-8")) == snapshot.model_dump(mode="json")
    rerun_path = tmp_path / "rerun.json"
    write_snapshot(rerun_path, snapshot)
    assert rerun_path.read_bytes() == raw


def test_snapshot_never_drops_superseded_or_contradicted(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    clock = _clock(FIXED_APPENDED_AT)
    append(path, _make_observation("obs_keep_001"), clock=clock)
    append(
        path,
        _make_observation("obs_keep_002"),
        supersedes="obs_keep_001",
        reason="refinement",
        clock=clock,
    )
    append(path, _make_observation("obs_keep_003"), clock=clock)
    append(
        path,
        _make_observation("obs_keep_004"),
        contradiction_of="obs_keep_003",
        clock=clock,
    )
    append(path, _make_observation("obs_keep_005"), clock=clock)
    entries = read_ledger(path)
    assert len(entries) == 5
    snapshot = derive_snapshot(entries, clock=_clock(FIXED_GENERATED_AT))
    assert [item.observation.observation_id for item in snapshot.entries] == [
        "obs_keep_001",
        "obs_keep_002",
        "obs_keep_003",
        "obs_keep_004",
        "obs_keep_005",
    ]
    by_id = _snapshot_by_id(snapshot)
    assert by_id["obs_keep_001"].superseded_by == "obs_keep_002"
    assert by_id["obs_keep_001"].is_current is False
    assert by_id["obs_keep_004"].contradiction_of == "obs_keep_003"
    assert by_id["obs_keep_003"].is_current is True
    assert by_id["obs_keep_004"].is_current is True


def test_ledger_package_has_no_mutation_helpers() -> None:
    mutation = re.compile(r"(update|edit|delete)", re.IGNORECASE)
    for name in dir(ledger_pkg):
        if name.startswith("_"):
            continue
        assert mutation.search(name) is None, name
    for filename in ("__init__.py", "ledger.py"):
        text = (LEDGER_DIR / filename).read_text(encoding="utf-8")
        for match in re.finditer(
            r"^def\s+([A-Za-z_][A-Za-z0-9_]*)", text, re.MULTILINE
        ):
            assert mutation.search(match.group(1)) is None, match.group(1)


def test_ledger_package_imports_only_allowlisted_modules() -> None:
    stdlib = set(sys.stdlib_module_names)
    allowed = {"pydantic", "bakery_intelligence"}
    pattern = re.compile(r"^(?:import|from)\s+([A-Za-z_][A-Za-z0-9_.]*)")
    for filename in ("__init__.py", "ledger.py"):
        text = (LEDGER_DIR / filename).read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                top = pattern.match(stripped).group(1).split(".")[0]  # type: ignore[union-attr]
                assert top in stdlib or top in allowed, stripped


def test_export_surface_pinned() -> None:
    assert set(ledger_pkg.__all__) == {
        "LedgerEntry",
        "LedgerSnapshot",
        "SnapshotEntry",
        "LedgerError",
        "append",
        "read_ledger",
        "derive_snapshot",
        "write_snapshot",
        "utcnow",
    }
    assert ledger_pkg.LedgerEntry is LedgerEntry
    assert ledger_pkg.LedgerSnapshot is LedgerSnapshot
    assert ledger_pkg.SnapshotEntry is SnapshotEntry
    assert ledger_pkg.LedgerError is LedgerError
    assert ledger_pkg.append is append
    assert ledger_pkg.read_ledger is read_ledger
    assert ledger_pkg.derive_snapshot is derive_snapshot
    assert ledger_pkg.write_snapshot is write_snapshot
    assert ledger_pkg.utcnow is utcnow


def test_entry_has_no_superseded_by_field() -> None:
    """plan.md L267 wins over tasks.md L120: no stored reverse index.

    Recording ``superseded_by`` on the superseded line would force
    rewriting a prior line, violating append-only; the reverse index
    lives only on the derived :class:`SnapshotEntry`.
    """
    assert "superseded_by" not in LedgerEntry.model_fields
    assert "superseded_by" in SnapshotEntry.model_fields


def test_utcnow_returns_tz_aware_utc() -> None:
    now = utcnow()
    assert now.tzinfo is not None
    assert now.utcoffset() == timezone.utc.utcoffset(None)


def test_conflicting_fixture_integration_preserves_both(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    clock = _clock(FIXED_APPENDED_AT)
    first = _fixture_observation(CONFLICTING_STEM, CON_001_ID)
    second = _fixture_observation(CONFLICTING_STEM, CON_002_ID)
    assert first.scale_value == CON_001_VALUE
    assert second.scale_value == CON_002_VALUE
    assert first.timestamp_seconds == CON_TIMESTAMP
    assert second.timestamp_seconds == CON_TIMESTAMP
    append(path, first, clock=clock)
    append(path, second, contradiction_of=CON_001_ID, clock=clock)
    entries = read_ledger(path)
    assert [item.observation.observation_id for item in entries] == [
        CON_001_ID,
        CON_002_ID,
    ]
    for entry, expected in zip(entries, (first, second)):
        assert entry.observation.model_dump(mode="json") == (
            expected.model_dump(mode="json")
        )
        assert entry.observation.provenance.source_type.value == "human"
        assert (
            entry.observation.provenance.source_id
            == "synthetic_interval_conflicting.json"
        )
        assert entry.observation.verification_status.value == "ambiguous"


def test_nominal_fixture_order_and_null_hygiene(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    clock = _clock(FIXED_APPENDED_AT)
    tare = _fixture_observation(NOMINAL_STEM, NOM_TARE_ID)
    mass = _fixture_observation(NOMINAL_STEM, NOM_MASS_ID)
    assert tare.scale_value == NOM_TARE_VALUE
    assert mass.scale_value == NOM_MASS_VALUE
    assert tare.timestamp_seconds == NOM_TARE_TIMESTAMP
    assert mass.timestamp_seconds == NOM_MASS_TIMESTAMP
    append(path, tare, clock=clock)
    append(path, mass, clock=clock)
    entries = read_ledger(path)
    assert [item.ledger_seq for item in entries] == [0, 1]
    assert [item.observation.observation_id for item in entries] == [
        NOM_TARE_ID,
        NOM_MASS_ID,
    ]
    assert {item.observation.observation_id for item in entries} == {
        NOM_TARE_ID,
        NOM_MASS_ID,
    }
    for entry, expected in zip(entries, (tare, mass)):
        assert entry.observation.model_dump(mode="json") == (
            expected.model_dump(mode="json")
        )
    for payload in _parsed_lines(path):
        assert payload["observation"]["scale_unit_hypothesis"] is None
        assert payload["observation"]["raw_reported_value"] is None


def test_ledger_lines_carry_no_derivation_keys(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    clock = _clock(FIXED_APPENDED_AT)
    append(path, _fixture_observation(NOMINAL_STEM, NOM_TARE_ID), clock=clock)
    append(path, _fixture_observation(NOMINAL_STEM, NOM_MASS_ID), clock=clock)
    reread = _make_observation("obs_der_003", scale_value=121.0)
    append(
        path,
        reread,
        supersedes=NOM_MASS_ID,
        reason="re-read keeps mass history",
        clock=clock,
    )
    for line in _raw_lines(path):
        for token in FORBIDDEN_LEDGER_TOKENS:
            assert token not in line
        payload = json.loads(line)
        assert "net_ingredient_quantity" not in payload["observation"]
        assert "derivation" not in payload["observation"]


def test_append_on_corrupt_ledger_raises_without_writing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "drift.jsonl"
    _seed_three_line_ledger(path)
    lines = _raw_lines(path)
    lines[1] = "{not json"
    _write_drifted_ledger(path, lines)
    before = path.read_bytes()
    with pytest.raises(LedgerError) as exc_info:
        append(
            path,
            _make_observation("obs_new_003"),
            clock=_clock(FIXED_APPENDED_AT),
        )
    _assert_ledger_error(
        exc_info.value,
        code="corrupt_line",
        entity_id="drift.jsonl:2",
        line_no=2,
    )
    assert path.read_bytes() == before


def test_append_write_failure_reports_write_error(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"not a directory")
    path = blocker / "ledger.jsonl"
    with pytest.raises(LedgerError) as exc_info:
        append(
            path,
            _make_observation("obs_werr_001"),
            clock=_clock(FIXED_APPENDED_AT),
        )
    _assert_ledger_error(
        exc_info.value, code="write_error", entity_id="obs_werr_001"
    )
    assert isinstance(exc_info.value.__cause__, OSError)
    assert blocker.read_bytes() == b"not a directory"


def test_missing_final_newline_is_tolerated_on_read(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    _seed_three_line_ledger(path)
    _write_drifted_ledger(path, _raw_lines(path), trailing_newline=False)
    assert not _raw_text(path).endswith("\n")
    entries = read_ledger(path)
    assert [item.observation.observation_id for item in entries] == [
        NOM_TARE_ID,
        NOM_MASS_ID,
        CON_001_ID,
    ]
    assert [item.ledger_seq for item in entries] == [0, 1, 2]


def test_mutual_contradiction_pair_accepted_on_read(
    tmp_path: Path,
) -> None:
    """Mutual links are append-unconstructible but read-accepted.

    Appended edges always point backward, so no append sequence can
    build a mutual pair; a hand-stored mutual pair is still accepted
    by the whole-file target check, pinning "mutual allowed".
    """
    path = tmp_path / "drift.jsonl"
    clock = _clock(FIXED_APPENDED_AT)
    append(path, _make_observation("obs_mut_001"), clock=clock)
    append(
        path,
        _make_observation("obs_mut_002"),
        contradiction_of="obs_mut_001",
        clock=clock,
    )

    def _link_forward(payload: dict[str, Any]) -> None:
        payload["contradiction_of"] = "obs_mut_002"

    _mutate_line_payload(path, 0, _link_forward)
    entries = read_ledger(path)
    assert [item.contradiction_of for item in entries] == [
        "obs_mut_002",
        "obs_mut_001",
    ]
    snapshot = derive_snapshot(entries, clock=_clock(FIXED_GENERATED_AT))
    assert [item.observation.observation_id for item in snapshot.entries] == [
        "obs_mut_001",
        "obs_mut_002",
    ]


def test_append_check_order_prefers_earlier_violation(
    tmp_path: Path,
) -> None:
    """Overlapping violations report the earliest F4 check that fails."""
    path = tmp_path / "ledger.jsonl"
    clock = _clock(FIXED_APPENDED_AT)
    append(path, _fixture_observation(NOMINAL_STEM, NOM_TARE_ID), clock=clock)
    with pytest.raises(LedgerError) as exc_info:
        append(
            path,
            _fixture_observation(NOMINAL_STEM, NOM_TARE_ID),
            supersedes="obs_missing_010",
            reason="duplicate id and unknown target",
            clock=clock,
        )
    _assert_ledger_error(
        exc_info.value, code="duplicate_id", entity_id=NOM_TARE_ID
    )
    with pytest.raises(LedgerError) as exc_info:
        append(
            path,
            _make_observation(NOM_TARE_ID),
            supersedes=NOM_TARE_ID,
            reason="self link on a stored id",
            clock=clock,
        )
    _assert_ledger_error(
        exc_info.value, code="self_link", entity_id=NOM_TARE_ID
    )
    assert [item.observation.observation_id for item in read_ledger(path)] == [
        NOM_TARE_ID
    ]
