"""Build the proposed v3 snapshot from v2 (chef-readable prose).

Offline and deterministic (no provider calls)::

    uv run python -m bakery_intelligence.recipe_export.build_v3
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from bakery_intelligence.api.recipe_editor import (
    create_record,
    read_record,
)
from bakery_intelligence.recipe_export.build import (
    sha256_file,
    write_json,
)
from bakery_intelligence.recipe_export.v3snapshot import (
    V3_SNAPSHOT_ID,
    V3_VERSION,
    build_v3,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the proposed v3 snapshot.")
    parser.add_argument("--root", default="runs_api")
    args = parser.parse_args(argv)

    root = Path(args.root)
    v2_path = (root / "recipes" / "bakery_demo_v1_proposed_v2"
               / "snapshot.json")
    if not v2_path.is_file():
        raise FileNotFoundError(f"Missing v2 snapshot: {v2_path}")
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))

    v3_dir = root / "recipes" / V3_SNAPSHOT_ID
    v3_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    snapshot = build_v3(v2, generated_at)
    snapshot_path = v3_dir / "snapshot.json"
    write_json(snapshot_path, snapshot)
    if read_record(root, V3_SNAPSHOT_ID) is None:
        create_record(root, snapshot.get("title")
                      or "Cinnamon Rolls (Roles de Canela)",
                      "en", source_run_id="bakery_demo_v1",
                      parent_version="bakery_demo_v1_proposed_v2",
                      recipe_id=V3_SNAPSHOT_ID)
    # The demo v2 recipe also gets a record so Home lists it.
    if read_record(root, "bakery_demo_v1_proposed_v2") is None:
        create_record(root, v2.get("title")
                      or "Cinnamon Rolls (Roles de Canela)",
                      "en", source_run_id="bakery_demo_v1",
                      parent_version="bakery_demo_v1_ai_reviewed_v1",
                      recipe_id="bakery_demo_v1_proposed_v2")
    manifest = {
        "snapshot_id": V3_SNAPSHOT_ID,
        "snapshot_version": V3_VERSION,
        "run_name": "bakery_demo_v1",
        "generated_at": generated_at,
        "sources": {
            "v2_snapshot": str(v2_path),
            "v2_snapshot_sha256": sha256_file(v2_path),
        },
        "artifacts": {
            "snapshot.json": sha256_file(snapshot_path),
        },
    }
    write_json(v3_dir / "manifest.json", manifest)
    print(f"snapshot: {snapshot_path}")
    print(f"ingredients: {len(snapshot['ingredients'])}; method: "
          f"{len(snapshot['method'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
