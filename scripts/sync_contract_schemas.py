"""Regenerate canonical JSON Schemas from the M1-01 Pydantic models.

The Pydantic models in ``src/bakery_intelligence/models/`` are the
implementation source of truth; the committed artifacts under
``src/bakery_intelligence/models/schemas/`` must byte-match this script's
output. Drift is enforced by
``tests/test_contracts.py::test_json_schemas_match_model_exports``.

Generation is ``model_json_schema()`` plus a small set of explicit,
narrowly scoped JSON Schema Draft 2020-12 extensions for approved domain
rules that Pydantic cannot export (validators are runtime-only). Each
extension anchors on a node the generator must produce and fails loudly if
the anchor is missing, so model changes cannot silently drop a constraint.
Rules that JSON Schema cannot represent reliably (cross-property numeric
comparison, whitespace stripping, strict datetime parsing) stay
Pydantic-only and are documented in ``docs/schemas/migration_notes.md``
instead of being approximated.

Offline, deterministic, standard library only. Serialization is canonical
(indent 2, sorted keys, LF newlines) so output is stable across runs and
platforms for a fixed pydantic version, and generation is idempotent.

Usage (from the repository root):
    python scripts/sync_contract_schemas.py          # rewrite schema files
    python scripts/sync_contract_schemas.py --check  # report drift only
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMAS_DIR = REPO_ROOT / "src" / "bakery_intelligence" / "models" / "schemas"

sys.path.insert(0, str(REPO_ROOT / "src"))

from bakery_intelligence.models import (  # noqa: E402
    ActionEvent,
    FrameEvidence,
    Observation,
    RecipeDraft,
    ReviewDecision,
    RunManifest,
    VerificationReport,
    VideoSource,
    WeighingEvent,
)

MODELS = {
    "video_source.json": VideoSource,
    "frame_evidence.json": FrameEvidence,
    "observation.json": Observation,
    "action_event.json": ActionEvent,
    "weighing_event.json": WeighingEvent,
    "recipe_draft.json": RecipeDraft,
    "verification.json": VerificationReport,
    "run_manifest.json": RunManifest,
    "review_decision.json": ReviewDecision,
}

SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_ID_BASE = "https://bakery-intelligence.local/schemas/1.0.0"

# Mirrors SchemaVersionedModel: semver shape with canonical major version 1.
SCHEMA_VERSION_PATTERN = r"^1\.\d+\.\d+$"

# Exact mirror of the VideoSource.source_ref validator, which rejects a value
# when posixpath.isabs or ntpath.isabs accepts it: a leading "/" or "\", or a
# drive letter followed by "/" or "\". Drive-relative "C:foo" stays allowed,
# exactly as at runtime.
ABSOLUTE_PATH_PATTERN = r"^([/\\]|[A-Za-z]:[/\\])"

# Derivation values that permit an asserted quantity (Derivation minus
# UNKNOWN). Used to mirror the model default: an omitted derivation behaves
# as "unknown" and therefore forbids value/unit.
SUPPORTED_DERIVATIONS = [
    "direct_after_tare",
    "difference_between_readings",
    "independently_verified",
]


def _require(node: dict[str, Any], path: list[str]) -> dict[str, Any]:
    """Navigate to a mapping node, failing loudly when the anchor moved."""
    current: Any = node
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise RuntimeError(
                "schema extension anchor missing: " + ".".join(path)
            )
        current = current[key]
    if not isinstance(current, dict):
        raise RuntimeError(
            "schema extension anchor is not an object: " + ".".join(path)
        )
    return current


def _ext_schema_version(schema: dict[str, Any]) -> None:
    """Restrict schema_version to canonical major version 1 (all files)."""
    node = _require(schema, ["properties", "schema_version"])
    node["pattern"] = SCHEMA_VERSION_PATTERN


def _ext_source_ref(schema: dict[str, Any]) -> None:
    """Forbid absolute source_ref paths (video_source.json only)."""
    node = _require(schema, ["properties", "source_ref"])
    node["not"] = {"pattern": ABSOLUTE_PATH_PATTERN}


def _ext_observation_time(schema: dict[str, Any]) -> None:
    """Require exactly one populated temporal form (observation.json only).

    A branch matches only when its key is present with a non-null value, so
    omitted and null fields both count as unset, exactly as in the model
    validator. Both populated matches twice and neither matches zero times;
    oneOf rejects both cases.
    """
    schema.setdefault("allOf", []).append(
        {
            "oneOf": [
                {
                    "properties": {"timestamp_seconds": {"type": "number"}},
                    "required": ["timestamp_seconds"],
                },
                {
                    "properties": {"interval": {"type": "object"}},
                    "required": ["interval"],
                },
            ]
        }
    )


def _ext_unknown_derivation(def_name: str) -> Callable[[dict[str, Any]], None]:
    """Forbid asserted value/unit under an effective unknown derivation.

    Applies to the NetIngredientQuantity ($defs of weighing_event.json) and
    DraftQuantity ($defs of recipe_draft.json) definitions. The condition
    mirrors the model default: derivation omitted behaves as "unknown", so
    the rule fires unless a supported derivation is explicitly present. An
    omitted value or unit passes, matching the model (missing means null).
    """

    def apply(schema: dict[str, Any]) -> None:
        node = _require(schema, ["$defs", def_name])
        node.setdefault("allOf", []).append(
            {
                "if": {
                    "not": {
                        "required": ["derivation"],
                        "properties": {
                            "derivation": {"enum": SUPPORTED_DERIVATIONS}
                        },
                    }
                },
                "then": {
                    "properties": {
                        "value": {"type": "null"},
                        "unit": {"type": "null"},
                    }
                },
            }
        )

    return apply


# filename -> extensions beyond the universal schema_version pattern.
FILE_EXTENSIONS: dict[str, list[Callable[[dict[str, Any]], None]]] = {
    "video_source.json": [_ext_source_ref],
    "observation.json": [_ext_observation_time],
    "weighing_event.json": [_ext_unknown_derivation("NetIngredientQuantity")],
    "recipe_draft.json": [_ext_unknown_derivation("DraftQuantity")],
}


def _apply_extensions(filename: str, schema: dict[str, Any]) -> None:
    _ext_schema_version(schema)
    for extension in FILE_EXTENSIONS.get(filename, []):
        extension(schema)


def render_model(filename: str) -> str:
    """Render the canonical schema text for one committed artifact."""
    schema = MODELS[filename].model_json_schema()
    _apply_extensions(filename, schema)
    schema["$schema"] = SCHEMA_DIALECT
    schema["$id"] = f"{SCHEMA_ID_BASE}/{filename}"
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def render_all() -> dict[str, str]:
    """Render all nine canonical schemas keyed by artifact filename."""
    return {name: render_model(name) for name in MODELS}


def _read_committed(filename: str) -> str:
    path = SCHEMAS_DIR / filename
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report drift without writing files",
    )
    args = parser.parse_args(argv)

    rendered = render_all()
    drifted = [
        name
        for name, expected in sorted(rendered.items())
        if _read_committed(name) != expected
    ]
    if args.check:
        for name in drifted:
            diff = difflib.unified_diff(
                _read_committed(name).splitlines(keepends=True),
                rendered[name].splitlines(keepends=True),
                fromfile=f"committed/{name}",
                tofile=f"generated/{name}",
            )
            print("".join(diff), end="")
        if drifted:
            print(f"{len(drifted)} schema(s) differ: {', '.join(drifted)}")
            return 1
        print("All 9 schemas match the model exports.")
        return 0

    for name, expected in sorted(rendered.items()):
        (SCHEMAS_DIR / name).write_text(
            expected, encoding="utf-8", newline="\n"
        )
    if drifted:
        print(f"Updated {len(drifted)} schema(s): {', '.join(drifted)}")
    else:
        print("All 9 schemas already match the model exports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
