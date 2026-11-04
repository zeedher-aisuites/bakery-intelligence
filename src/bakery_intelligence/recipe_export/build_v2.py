"""Build the proposed v2 snapshot from v1 + recorded completion.

Offline and deterministic (no provider calls)::

    uv run python -m bakery_intelligence.recipe_export.build_v2
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from bakery_intelligence.recipe_export.build import (
    sha256_file,
    write_json,
)
from bakery_intelligence.recipe_export.v2snapshot import (
    V2_SNAPSHOT_ID,
    V2_VERSION,
    build_v2,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the proposed v2 snapshot.")
    parser.add_argument("--root", default="runs_api")
    args = parser.parse_args(argv)

    root = Path(args.root)
    v1_path = (root / "recipes" / "bakery_demo_v1_ai_reviewed_v1"
               / "snapshot.json")
    v2_dir = root / "recipes" / "bakery_demo_v1_proposed_v2"
    raw_path = v2_dir / "completion_raw.json"
    for label, path in (("v1 snapshot", v1_path),
                        ("completion record", raw_path)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_text(encoding="utf-8"))

    generated_at = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    snapshot = build_v2(v1, raw.get("parsed") or {}, raw,
                        generated_at)
    snapshot_path = v2_dir / "snapshot.json"
    write_json(snapshot_path, snapshot)
    manifest = {
        "snapshot_id": V2_SNAPSHOT_ID,
        "snapshot_version": V2_VERSION,
        "run_name": "bakery_demo_v1",
        "generated_at": generated_at,
        "sources": {
            "v1_snapshot": str(v1_path),
            "v1_snapshot_sha256": sha256_file(v1_path),
            "completion_raw": str(raw_path),
            "completion_raw_sha256": sha256_file(raw_path),
            "completion_request_id": raw.get("request_id"),
            "completion_actual_usd": raw.get("actual_usd"),
        },
        "artifacts": {
            "snapshot.json": sha256_file(snapshot_path),
        },
    }
    write_json(v2_dir / "manifest.json", manifest)
    estimated = sum(1 for r in snapshot["ingredients"]
                    if r.get("quantity_status") == "ai_estimated")
    guided = sum(1 for m in snapshot["method"]
                 if m.get("ai_guidance"))
    print(f"snapshot: {snapshot_path}")
    print(f"ingredients: {len(snapshot['ingredients'])} "
          f"({estimated} AI-estimated); method: "
          f"{len(snapshot['method'])} ({guided} with guidance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
