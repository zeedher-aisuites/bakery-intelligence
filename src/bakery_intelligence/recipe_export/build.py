"""Build the frozen recipe snapshot + PDF for a reviewed run.

Reads the persisted fusion draft and review decisions (never
modifies them), writes ``snapshot.json``,
``bakery_demo_v1_recipe_ai_reviewed_v1.pdf`` and ``manifest.json``
with SHA-256 checksums into the output directory::

    uv run python -m bakery_intelligence.recipe_export.build \\
        --out runs_api/recipes/bakery_demo_v1_ai_reviewed_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bakery_intelligence.recipe_export.pdf import build_pdf
from bakery_intelligence.recipe_export.snapshot import (
    SNAPSHOT_ID,
    SNAPSHOT_VERSION,
    build_snapshot,
)

PDF_FILENAME = "bakery_demo_v1_recipe_ai_reviewed_v1.pdf"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True)
                    + "\n", encoding="utf-8")


def load_decisions(path: Path) -> dict[tuple[str, str], dict]:
    stored = json.loads(path.read_text(encoding="utf-8"))
    decisions = stored.get("decisions") or {}
    items = decisions.values() if isinstance(decisions, dict) \
        else decisions
    return {(d["target_type"], d["target_id"]): d for d in items}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the frozen recipe snapshot + PDF.")
    parser.add_argument("--run", default="bakery_demo_v1")
    parser.add_argument("--root", default="runs_api")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    root = Path(args.root)
    draft_path = root / "fusion" / args.run / "fusion_draft.json"
    reviews_path = root / "fusion_reviews" / f"{args.run}.json"
    agent_f_path = (root / "phase2c" / args.run
                    / "session.agent-f.json")
    for label, path in (("draft", draft_path),
                        ("reviews", reviews_path)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    decisions = load_decisions(reviews_path)
    agent_f = json.loads(agent_f_path.read_text(encoding="utf-8")) \
        if agent_f_path.is_file() else {}

    out = Path(args.out) if args.out else (
        root / "recipes" / f"{args.run}_ai_reviewed_v1")
    out.mkdir(parents=True, exist_ok=False)

    generated_at = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    snapshot = build_snapshot(draft, decisions, agent_f, generated_at)
    snapshot_path = out / "snapshot.json"
    write_json(snapshot_path, snapshot)
    pdf_path = out / PDF_FILENAME
    pdf_path.write_bytes(build_pdf(snapshot))
    manifest = {
        "snapshot_id": SNAPSHOT_ID,
        "snapshot_version": SNAPSHOT_VERSION,
        "run_name": args.run,
        "generated_at": generated_at,
        "sources": {
            "fusion_draft": str(draft_path),
            "fusion_draft_sha256": sha256_file(draft_path),
            "reviews": str(reviews_path),
            "reviews_sha256": sha256_file(reviews_path),
        },
        "artifacts": {
            "snapshot.json": sha256_file(snapshot_path),
            PDF_FILENAME: sha256_file(pdf_path),
        },
    }
    write_json(out / "manifest.json", manifest)
    print(f"snapshot: {snapshot_path}")
    print(f"pdf: {pdf_path}")
    print(f"ingredients: {len(snapshot['ingredients'])}; "
          f"method steps: {len(snapshot['method'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
