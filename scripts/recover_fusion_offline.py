"""Deterministic offline repair of a failed fusion draft.

Recovers runs whose Baseten fusion response parsed as JSON but
failed proposal validation (for example, over-long ingredient
names). The repair is local and lossless: names are split into
primary + basis note via
``proposal.normalize_proposal_draft`` and the draft is
re-validated against the run's own merged evidence. The failed
artifacts are snapshotted before anything is written, and the
repaired ``fusion_recipe.json`` keeps the original provider
request id and usage so the worker recovery reuses it without
another billable call.

Usage (from the repository root)::

    uv run python scripts/recover_fusion_offline.py <run_id>
    uv run python scripts/recover_fusion_offline.py <run_id> --apply

Without ``--apply`` the script only reports what it would do.
After ``--apply`` succeeds, complete the run with::

    uv run python -m bakery_intelligence.api.worker \\
        --recover <run_id> --recover-note "<why>"

Never calls Meta, Baseten, or any other provider.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from bakery_intelligence.recipe_export import proposal as proposal_mod


def _checks_bundle(merged: dict[str, Any], language: str,
                   recipe_name: str) -> dict[str, Any]:
    return {
        "readings": [{
            "value": item.get("value")}
            for item in merged.get("items") or []
            if item.get("kind") == "scale_reading"
            and isinstance(item.get("value"), (int, float))],
        "language": language,
        "recipe_name": recipe_name,
    }


def _snapshot_failed_state(work: Path, run_id: str,
                           storage: Path) -> Path:
    stamp = datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = storage / "recovery_backups" / f"{run_id}_{stamp}"
    dest.mkdir(parents=True, exist_ok=False)
    for name in ("fusion_recipe_failed.json",
                 "fusion_recipe_failed.txt",
                 "fusion_input.json",
                 "cost_report.json"):
        src = work / "multimodal" / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    db_path = storage / "jobs.sqlite3"
    if db_path.is_file():
        con = sqlite3.connect(db_path)
        try:
            row = con.execute(
                "SELECT run_id, workflow, status, attempts, "
                "error, updated_at FROM jobs WHERE run_id = ?",
                (run_id,)).fetchone()
        finally:
            con.close()
        if row is not None:
            (dest / "job_row.json").write_text(
                json.dumps({
                    "run_id": row[0], "workflow": row[1],
                    "status": row[2], "attempts": row[3],
                    "error": row[4], "updated_at": row[5],
                }, indent=2),
                encoding="utf-8")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Repair a failed fusion draft offline.")
    parser.add_argument("run_id")
    parser.add_argument("--storage-dir", type=Path,
                        default=REPO_ROOT / "runs_api")
    parser.add_argument("--apply", action="store_true",
                        help="Write the repaired record.")
    args = parser.parse_args(argv)

    storage: Path = args.storage_dir
    work = storage / "runs" / args.run_id / "work"
    mm_dir = work / "multimodal"
    failed_path = mm_dir / "fusion_recipe_failed.json"
    merged_path = mm_dir / "merged.json"
    target_path = mm_dir / "fusion_recipe.json"
    if not failed_path.is_file():
        print(f"error: no failed fusion draft for "
              f"{args.run_id}", file=sys.stderr)
        return 2
    if not merged_path.is_file():
        print(f"error: merged evidence missing for "
              f"{args.run_id}", file=sys.stderr)
        return 2
    if target_path.is_file():
        print(f"error: {target_path} already exists; "
              f"refusing to overwrite", file=sys.stderr)
        return 2
    failed = json.loads(failed_path.read_text(encoding="utf-8"))
    merged = json.loads(merged_path.read_text(encoding="utf-8"))
    draft = failed.get("draft")
    if not isinstance(draft, dict):
        print("error: failed file has no parsed draft; "
              "offline repair is impossible", file=sys.stderr)
        return 2
    fusion_input_path = mm_dir / "fusion_input.json"
    language = "en"
    recipe_name = args.run_id
    if fusion_input_path.is_file():
        fusion_input = json.loads(
            fusion_input_path.read_text(encoding="utf-8"))
        language = fusion_input.get("language") or "en"
        recipe_name = fusion_input.get(
            "recipe_name") or args.run_id
    repaired = copy.deepcopy(draft)
    try:
        normalizations = proposal_mod.normalize_proposal_draft(
            repaired)
        validated = proposal_mod.validate_proposal(
            _checks_bundle(merged, language, recipe_name),
            repaired)
    except proposal_mod.ProposalError as exc:
        print(f"error: draft is not recoverable offline: "
              f"{exc}", file=sys.stderr)
        return 1
    print(f"run: {args.run_id}")
    print(f"failed request: {failed.get('request_id')}")
    print(f"normalizations: {len(normalizations)}")
    for item in normalizations:
        if item.get("kind") == "suggestion_overflow":
            print(f"  suggestion {item['id']!r} moved to notes")
        else:
            print(f"  row {item['row']}: "
                  f"{item['original_chars']} chars -> "
                  f"{item['name']!r}")
    print(f"ingredients: {len(validated['ingredients'])}; "
          f"steps: {len(validated['method'])}")
    if not args.apply:
        print("dry run: pass --apply to write "
              f"{target_path.name}")
        return 0
    backup = _snapshot_failed_state(work, args.run_id, storage)
    print(f"failed state snapshotted to {backup}")
    record = {
        "model": failed.get("model"),
        "provider": failed.get("provider", "baseten"),
        "prompt_version": failed.get("prompt_version", "v1.1"),
        "request_id": failed.get("request_id"),
        "finish_reason": failed.get("finish_reason"),
        "usage": failed.get("usage") or {},
        "normalizations": normalizations,
        "recovery": {
            "offline": True,
            "source": "fusion_recipe_failed.json",
            "repaired_at": datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
        },
        "result": validated,
    }
    target_path.write_text(json.dumps(record, indent=2),
                           encoding="utf-8")
    print(f"wrote {target_path}")
    print("next: run the worker with --recover to reuse all "
          "stages without new billable calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
