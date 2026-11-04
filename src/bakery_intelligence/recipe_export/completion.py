"""One bounded AI culinary-completion call for the v2 recipe.

Reads the frozen v1 snapshot, sends a compact reviewed-state
bundle to the culinary-completion role, validates the structured
proposal, and persists the raw record. Exactly one provider call;
the v2 snapshot builder consumes the recorded file offline::

    uv run python -m bakery_intelligence.recipe_export.completion \\
        --execute
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from bakery_intelligence.adversarial.live import _load_local_env
from bakery_intelligence.phase2c import agents, media, prompts
from bakery_intelligence.phase2c.clients import (
    BudgetLedger,
    MetaResponsesClient,
)

V1_DIRNAME = "bakery_demo_v1_ai_reviewed_v1"
V2_DIRNAME = "bakery_demo_v1_proposed_v2"
RAW_FILENAME = "completion_raw.json"
ESTIMATE_USD = 0.30


def completion_bundle(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Compact reviewed state: rows, unknowns, steps, flags."""
    ingredients = []
    for row in snapshot.get("ingredients") or []:
        ingredients.append({
            "name": row.get("name"),
            "quantity": row.get("quantity"),
            "unit": row.get("unit"),
            "quantity_status": row.get("quantity_status"),
            "note": row.get("note"),
        })
    return {
        "title": snapshot.get("title"),
        "ingredients": ingredients,
        "method": [{
            "number": item.get("number"),
            "instruction": item.get("instruction"),
            "caveats": item.get("caveats"),
        } for item in snapshot.get("method") or []],
        "review_flags": snapshot.get("review_flags") or [],
        "limitations": snapshot.get("limitations") or [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one AI culinary-completion call.")
    parser.add_argument("--root", default="runs_api")
    parser.add_argument("--execute", action="store_true",
                        help="Authorize the single billable call.")
    args = parser.parse_args(argv)
    _load_local_env()

    root = Path(args.root)
    v1_path = (root / "recipes" / V1_DIRNAME / "snapshot.json")
    if not v1_path.is_file():
        raise FileNotFoundError(f"Missing v1 snapshot: {v1_path}")
    snapshot = json.loads(v1_path.read_text(encoding="utf-8"))
    bundle = completion_bundle(snapshot)
    out_dir = root / "recipes" / V2_DIRNAME
    raw_path = out_dir / RAW_FILENAME
    if raw_path.is_file():
        print(f"completion already recorded: {raw_path}")
        print("Refusing a repeat billable call; delete the file to "
              "re-run explicitly.")
        return 0
    if not args.execute:
        print("dry run only; pass --execute for one billable call.")
        print(f"bundle ingredients: {len(bundle['ingredients'])}, "
              f"steps: {len(bundle['method'])}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = BudgetLedger.load(root / "phase2c" / "ledger.json")
    client = MetaResponsesClient()
    result = agents.run_text_agent(
        client, ledger, prompts.CULINARY_COMPLETION, "completion",
        "session", bundle, out_dir, RAW_FILENAME, ESTIMATE_USD)
    if result.status != "ok":
        print(f"completion failed: {result.error}", file=sys.stderr)
        return 1
    stored = media.read_json(raw_path) or {}
    stored["prompt_version"] = prompts.CULINARY_COMPLETION.version
    stored["estimate_usd"] = ESTIMATE_USD
    stored["actual_usd"] = result.actual_usd
    media.write_json(raw_path, stored)
    ledger.record({
        "label": "culinary-completion/session",
        "estimate_usd": ESTIMATE_USD,
        "actual_usd": result.actual_usd,
        "request_id": result.request_id,
    })
    print(f"request: {result.request_id}")
    print(f"actual_usd: {result.actual_usd}")
    print(f"recorded: {raw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
