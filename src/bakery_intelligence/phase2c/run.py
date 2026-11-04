"""Phase 2C execution entry point (billable when run for real).

Usage:
    uv run python -m bakery_intelligence.phase2c.run --execute

Without --execute the command only validates media, prints the
session manifest summary and the budget plan, and exits. With
--execute it runs the full orchestrator against live providers
within the ledger gates in runs_api/phase2c/ledger.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bakery_intelligence.adversarial.live import _load_local_env
from bakery_intelligence.phase2c import media
from bakery_intelligence.phase2c.manifest import (
    CLIP_MIXING,
    CLIP_PREPARATION,
    SESSION_ID,
    build_session,
    write_manifest,
)
from bakery_intelligence.phase2c.orchestrator import Orchestrator

CLIP_PATHS = {
    CLIP_PREPARATION: Path(
        "data/raw/bakery_demo/"
        "bakery_demo_01_preparation_with_audio.mp4"),
    CLIP_MIXING: Path(
        "data/raw/bakery_demo/"
        "bakery_demo_02_mixing_with_audio.mp4"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Phase 2C bakery demo pipeline.")
    parser.add_argument("--execute", action="store_true",
                        help="Authorize live billable provider calls.")
    parser.add_argument("--root", default="runs_api",
                        help="Storage root (default: runs_api).")
    args = parser.parse_args(argv)
    _load_local_env()
    root = Path(args.root)
    manifest = build_session(
        [(CLIP_PREPARATION, 1, CLIP_PATHS[CLIP_PREPARATION]),
         (CLIP_MIXING, 2, CLIP_PATHS[CLIP_MIXING])])
    manifest_path = root / "phase2c" / SESSION_ID / "manifest.json"
    write_manifest(manifest, manifest_path)
    total = sum(clip.duration_seconds for clip in manifest.clips)
    print(f"session: {manifest.session_id}")
    for clip in manifest.clips:
        print(f"  {clip.order}. {clip.clip_id}: "
              f"{clip.duration_seconds:.1f}s, "
              f"{clip.width}x{clip.height} {clip.video_codec}, "
              f"audio={clip.audio_codec or 'none'}, "
              f"vfr={clip.variable_frame_rate}, "
              f"sha256={clip.sha256[:16]}...")
    print(f"total media: {total:.1f}s; gap: {manifest.inter_clip_gap}")
    print(f"manifest: {manifest_path}")
    orchestrator = Orchestrator(root)
    print(f"budget: ${orchestrator.ledger.budget_usd:.2f}, "
          f"spent: ${orchestrator.ledger.spent_usd:.4f}, "
          f"remaining: ${orchestrator.ledger.remaining():.2f}")
    if not args.execute:
        print("dry run only; pass --execute for live providers.")
        return 0
    clips = {}
    for clip in manifest.clips:
        print(f"--- clip: {clip.clip_id} ---", flush=True)
        clips[clip.clip_id] = orchestrator.run_clip(
            manifest, clip.clip_id)
    out = orchestrator.run_session(manifest, clips)
    print(f"session draft: {out}")
    print(f"spent total: ${orchestrator.ledger.spent_usd:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
