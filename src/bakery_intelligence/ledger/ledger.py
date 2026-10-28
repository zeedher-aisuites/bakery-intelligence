"""Append-only observation ledger: JSONL storage and snapshots (M2-02).

A ledger is a single-writer JSONL file with one :class:`LedgerEntry` per
line. Lines are content-addressed by position: ``ledger_seq`` is the
0-based line index, assigned only by :func:`append`, and
:func:`read_ledger` rejects any file where line ``i`` does not carry
``ledger_seq == i``. History is never rewritten: corrections and
refinements are new lines linked with ``supersedes`` /
``contradiction_of``, and :func:`derive_snapshot` renders the current
view (including the ``superseded_by`` reverse index) without dropping
any entry.

Design notes pinned by the M2-02 frozen decisions:

- ``tasks.md`` L120 suggested a ``superseded_by`` field on the entry;
  it is deliberately absent from :class:`LedgerEntry` (``plan.md`` L267
  wins). Recording ``superseded_by`` on the superseded line would force
  rewriting an already stored line, violating append-only; the reverse
  index therefore lives only on :class:`SnapshotEntry`, which is
  derived, never stored.
- :func:`append` takes ``(ledger_path, observation, *, supersedes,
  contradiction_of, reason, clock)`` rather than a prebuilt entry, so
  ``ledger_seq`` and ``appended_at`` authority stays inside ``append``
  and callers cannot forge positions or timestamps.
- A missing ledger file is ``LedgerError`` with code ``"corrupt_line"``
  and ``line_no=None`` from :func:`read_ledger`. There is no separate
  missing-file code: the two-code file contract stays closed, and
  ``None`` is the only ``line_no`` that references no physical line.
  :func:`append` treats a missing file as an empty ledger to create.
- The supersedes graph is a DAG by construction: targets must already
  exist in the ledger, so every edge points to a strictly earlier
  ``ledger_seq`` and cycles are unconstructable. Each observation may
  be superseded at most once, so refinements chain forward and
  ``superseded_by`` stays singular. ``contradiction_of`` only requires
  an existing, non-self target; mutual contradiction is allowed.
- Reference isolation (Constitution VII): this module never ingests
  ``annotations/reference/``. Adjudicated references stay out of the
  pipeline; the ledger stores pipeline observations verbatim.
- There is intentionally no update, edit, or delete helper in this
  package. The only writer is :func:`append`, and the only snapshot
  writer is :func:`write_snapshot`.

The ledger preserves observations byte-for-byte: provenance,
``verification_status``, and every other observation field pass
through untouched. The ledger assigns no ``observation_id``, infers no
net quantities or events, and performs no draft logic.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from pydantic import Field, ValidationError, model_validator

from bakery_intelligence.models import (
    CANONICAL_SCHEMA_VERSION,
    Observation,
    SchemaVersionedModel,
)

__all__ = [
    "LedgerEntry",
    "LedgerError",
    "LedgerSnapshot",
    "SnapshotEntry",
    "append",
    "derive_snapshot",
    "read_ledger",
    "utcnow",
    "write_snapshot",
]


class LedgerError(Exception):
    """Structured ledger failure with a machine-readable code.

    Codes raised by :func:`append`: ``duplicate_id``,
    ``unknown_supersedes``, ``unknown_contradiction``, ``self_link``,
    ``already_superseded``, ``missing_reason``,
    ``reason_without_supersedes``, ``write_error``. Code raised by
    :func:`read_ledger`: ``corrupt_line`` (including the missing-file
    case, where ``line_no`` is ``None`` and ``entity_id`` is the file
    name). ``entity_id`` is the incoming ``observation_id`` for
    append-time relationship, duplicate, reason, and write errors, and
    ``"<name>:<line_no>"`` for corrupt stored lines. The original
    ``ValidationError`` / ``JSONDecodeError`` / ``OSError`` is preserved
    as ``__cause__`` where applicable. Unreadable-file ``OSError``
    failures from :func:`read_ledger` propagate unwrapped.
    """

    def __init__(
        self,
        code: str,
        entity_id: str,
        message: str,
        line_no: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.entity_id = entity_id
        self.message = message
        self.line_no = line_no


class LedgerEntry(SchemaVersionedModel):
    """One append-only ledger line: an observation plus link metadata.

    ``observation`` is embedded verbatim and never mutated.
    ``ledger_seq`` is the 0-based JSONL line index, assigned only by
    :func:`append`. ``appended_at`` records the injection clock, not an
    ordering authority (``ledger_seq`` orders). ``supersedes`` names a
    strictly earlier observation this line refines and requires a
    non-empty ``reason``; ``contradiction_of`` names an existing
    observation this line contradicts and needs no reason. Neither link
    may target the entry's own ``observation_id``.
    """

    observation: Observation
    ledger_seq: int = Field(ge=0)
    appended_at: datetime
    supersedes: str | None = None
    contradiction_of: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _check_links_and_reason(self) -> "LedgerEntry":
        own = self.observation.observation_id
        if self.supersedes is not None and self.supersedes == own:
            raise ValueError(
                "supersedes must not target the entry's own "
                "observation_id"
            )
        if self.contradiction_of is not None and (
            self.contradiction_of == own
        ):
            raise ValueError(
                "contradiction_of must not target the entry's own "
                "observation_id"
            )
        if self.supersedes is not None:
            if self.reason is None or not self.reason.strip():
                raise ValueError(
                    "reason must be non-empty when supersedes is set"
                )
        elif self.reason is not None:
            raise ValueError("reason requires supersedes to be set")
        return self


class SnapshotEntry(LedgerEntry):
    """A ledger line rendered into a snapshot with derived state.

    Adds the ``superseded_by`` reverse index (the ``observation_id``
    whose entry supersedes this one, else ``None``) and ``is_current``
    (``False`` exactly when superseded). All other fields are copied
    from the stored :class:`LedgerEntry`.
    """

    superseded_by: str | None = None
    is_current: bool


class LedgerSnapshot(SchemaVersionedModel):
    """Point-in-time rendered view over ledger entries in seq order."""

    schema_version: str = CANONICAL_SCHEMA_VERSION
    generated_at: datetime
    entries: list[SnapshotEntry] = Field(default_factory=list)


def utcnow() -> datetime:
    """Return the current tz-aware UTC time (default ledger clock)."""
    return datetime.now(timezone.utc)


def read_ledger(ledger_path: Path) -> list[LedgerEntry]:
    """Read and strictly validate a ledger file in stored order.

    Pure: never creates or modifies files. A missing file raises
    ``LedgerError`` with code ``"corrupt_line"``, ``line_no=None``,
    and ``entity_id`` equal to the file name. An empty (0-byte) file
    yields ``[]``. Any blank line, malformed JSON, entry validation
    failure, ``ledger_seq``/position mismatch, dangling
    ``supersedes``/``contradiction_of`` target, or doubly superseded
    target raises ``corrupt_line`` with the 1-based ``line_no``; partial
    results are never returned. A missing final newline is tolerated.
    """
    if not ledger_path.exists():
        raise LedgerError(
            "corrupt_line",
            ledger_path.name,
            f"missing ledger file: {ledger_path.name}",
        )
    text = ledger_path.read_text(encoding="utf-8")
    if text == "":
        return []
    raw_lines = text.split("\n")
    if text.endswith("\n"):
        raw_lines.pop()
    entries: list[LedgerEntry] = []
    for index, raw in enumerate(raw_lines):
        line_no = index + 1
        entity = f"{ledger_path.name}:{line_no}"
        if raw.strip() == "":
            raise LedgerError(
                "corrupt_line",
                entity,
                f"blank line {line_no} in {ledger_path.name}",
                line_no=line_no,
            )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LedgerError(
                "corrupt_line",
                entity,
                f"malformed JSON at line {line_no} in "
                f"{ledger_path.name}: {exc}",
                line_no=line_no,
            ) from exc
        try:
            entry = LedgerEntry.model_validate(payload)
        except ValidationError as exc:
            raise LedgerError(
                "corrupt_line",
                entity,
                f"invalid ledger entry at line {line_no} in "
                f"{ledger_path.name}: {exc.error_count()} error(s)",
                line_no=line_no,
            ) from exc
        if entry.ledger_seq != index:
            raise LedgerError(
                "corrupt_line",
                entity,
                f"ledger_seq {entry.ledger_seq} does not match "
                f"0-based line index {index} in {ledger_path.name}",
                line_no=line_no,
            )
        entries.append(entry)
    known = {entry.observation.observation_id for entry in entries}
    superseded: set[str] = set()
    for index, entry in enumerate(entries):
        line_no = index + 1
        entity = f"{ledger_path.name}:{line_no}"
        if entry.supersedes is not None and entry.supersedes not in known:
            raise LedgerError(
                "corrupt_line",
                entity,
                f"unknown supersedes target "
                f"{entry.supersedes!r} at line {line_no} in "
                f"{ledger_path.name}",
                line_no=line_no,
            )
        if entry.contradiction_of is not None and (
            entry.contradiction_of not in known
        ):
            raise LedgerError(
                "corrupt_line",
                entity,
                f"unknown contradiction_of target "
                f"{entry.contradiction_of!r} at line {line_no} in "
                f"{ledger_path.name}",
                line_no=line_no,
            )
        if entry.supersedes is not None:
            if entry.supersedes in superseded:
                raise LedgerError(
                    "corrupt_line",
                    entity,
                    f"observation {entry.supersedes!r} already "
                    f"superseded before line {line_no} in "
                    f"{ledger_path.name}",
                    line_no=line_no,
                )
            superseded.add(entry.supersedes)
    return entries


def append(
    ledger_path: Path,
    observation: Observation,
    *,
    supersedes: str | None = None,
    contradiction_of: str | None = None,
    reason: str | None = None,
    clock: Callable[[], datetime] = utcnow,
) -> LedgerEntry:
    """Validate and append one observation as the next ledger line.

    The existing ledger is validated first: a corrupt file raises
    ``LedgerError`` and nothing is written. A missing file is treated
    as an empty ledger. Relationship checks run in fixed order:
    ``self_link``, ``duplicate_id``, unknown targets
    (``unknown_supersedes``, then ``unknown_contradiction``),
    ``already_superseded``, then the reason rules (``missing_reason``,
    ``reason_without_supersedes``). On success, parent directories are
    created, exactly one ``model_dump_json()`` line plus ``"\\n"`` is
    appended, and the stored entry (with ``ledger_seq`` equal to the
    prior entry count and ``appended_at`` from ``clock``) is returned.
    The observation is embedded verbatim: no identifier is invented
    and no field is altered. ``OSError`` during the write is wrapped
    as ``write_error``.
    """
    if ledger_path.exists():
        existing = read_ledger(ledger_path)
    else:
        existing = []
    new_id = observation.observation_id
    if supersedes == new_id or contradiction_of == new_id:
        raise LedgerError(
            "self_link",
            new_id,
            f"observation {new_id!r} must not link to itself",
        )
    known = {entry.observation.observation_id for entry in existing}
    if new_id in known:
        raise LedgerError(
            "duplicate_id",
            new_id,
            f"observation {new_id!r} is already stored in "
            f"{ledger_path.name}",
        )
    if supersedes is not None and supersedes not in known:
        raise LedgerError(
            "unknown_supersedes",
            new_id,
            f"supersedes target {supersedes!r} is not stored in "
            f"{ledger_path.name}",
        )
    if contradiction_of is not None and contradiction_of not in known:
        raise LedgerError(
            "unknown_contradiction",
            new_id,
            f"contradiction_of target {contradiction_of!r} is not "
            f"stored in {ledger_path.name}",
        )
    if supersedes is not None and any(
        entry.supersedes == supersedes for entry in existing
    ):
        raise LedgerError(
            "already_superseded",
            new_id,
            f"supersedes target {supersedes!r} is already superseded "
            f"in {ledger_path.name}",
        )
    if supersedes is not None:
        if reason is None or not reason.strip():
            raise LedgerError(
                "missing_reason",
                new_id,
                f"observation {new_id!r} supersedes "
                f"{supersedes!r} without a non-empty reason",
            )
    elif reason is not None:
        raise LedgerError(
            "reason_without_supersedes",
            new_id,
            f"observation {new_id!r} carries a reason without "
            "supersedes",
        )
    entry = LedgerEntry(
        schema_version=CANONICAL_SCHEMA_VERSION,
        observation=observation,
        ledger_seq=len(existing),
        appended_at=clock(),
        supersedes=supersedes,
        contradiction_of=contradiction_of,
        reason=reason,
    )
    try:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(entry.model_dump_json() + "\n")
    except OSError as exc:
        raise LedgerError(
            "write_error",
            new_id,
            f"cannot append observation {new_id!r} to "
            f"{ledger_path.name}: {exc}",
        ) from exc
    return entry


def derive_snapshot(
    entries: Sequence[LedgerEntry],
    *,
    clock: Callable[[], datetime] = utcnow,
) -> LedgerSnapshot:
    """Render entries into a snapshot in seq order (pure, no I/O).

    Builds the ``superseded_by`` reverse index from ``supersedes``
    edges and marks each entry current exactly when no other entry
    supersedes it. Superseded and contradicted entries are always
    retained. ``generated_at`` comes from ``clock``.
    """
    ordered = sorted(entries, key=lambda item: item.ledger_seq)
    superseded_by: dict[str, str] = {}
    for item in ordered:
        if item.supersedes is not None:
            superseded_by[item.supersedes] = (
                item.observation.observation_id
            )
    rendered = [
        SnapshotEntry(
            schema_version=item.schema_version,
            observation=item.observation,
            ledger_seq=item.ledger_seq,
            appended_at=item.appended_at,
            supersedes=item.supersedes,
            contradiction_of=item.contradiction_of,
            reason=item.reason,
            superseded_by=superseded_by.get(
                item.observation.observation_id
            ),
            is_current=item.observation.observation_id
            not in superseded_by,
        )
        for item in ordered
    ]
    return LedgerSnapshot(
        schema_version=CANONICAL_SCHEMA_VERSION,
        generated_at=clock(),
        entries=rendered,
    )


def write_snapshot(
    snapshot_path: Path, snapshot: LedgerSnapshot
) -> None:
    """Write a snapshot as deterministic JSON for ``observations.json``.

    Creates parent directories. Output uses sorted keys, 2-space
    indent, and a trailing line feed.
    """
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(
            snapshot.model_dump(mode="json"), sort_keys=True, indent=2
        )
        + "\n"
    )
    snapshot_path.write_text(text, encoding="utf-8")
