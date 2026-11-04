"""EPIC-KITCHENS-100 weighing-scale video discovery (Phase 1).

Reconstructs the original video selection as a reproducible pipeline:
broad keyword search, narrow instrument search, and exclusion of
semantic false positives (fish scales), ranked into a candidate
catalog for recipe reconstruction experiments.

Inputs are the official EPIC-KITCHENS-100 annotation CSVs
(train + validation). Outputs (CSV/JSON) are written outside the
public repository by default (``data/`` is git-ignored).

Attribution: EPIC-KITCHENS-100, Dima Damen et al., University of
Bristol. Annotations and audiovisual material are subject to the
dataset licence (CC BY-NC 4.0 for the annotation release; video
rights remain with the participants). This script only reads the
local annotation files and never downloads, copies, or
redistributes video or audio material.

Usage (from the repository root):

    .venv\\Scripts\\python.exe scripts/epic_dataset_discovery.py \\
        --train ..\\epic-kitchens-100-annotations\\EPIC_100_train.csv \\
        --validation ..\\epic-kitchens-100-annotations\\EPIC_100_validation.csv \\
        --out-dir data\\dataset_discovery
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SEARCH_FIELDS = ("narration", "verb", "noun", "all_nouns")
NOUN_FIELDS = ("noun", "all_nouns")
BROAD_TERMS = ("scale", "weigh", "measur")
NARROW_NOUN_TERMS = ("scale", "balance")
NARROW_NARRATION_TERM = "scale"
NARROW_NARRATION_VERB_TERM = "weigh"
FISH_NOUNS = ("scale", "scales")
KNOWN_SEEDS = ("P04_104", "P04_103", "P03_120", "P27_103", "P04_24")


class DiscoveryError(ValueError):
    """Invalid input or unreadable annotations."""


@dataclass(frozen=True)
class AnnotationFile:
    path: Path
    split: str
    sha256: str
    rows: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _blob(row: dict[str, str], fields: tuple[str, ...]) -> str:
    return " | ".join((row.get(field) or "") for field in fields).lower()


def _to_seconds(value: str) -> float:
    """Parse seconds or HH:MM:SS.cs timestamps to seconds."""
    text = (value or "").strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        pass
    parts = text.split(":")
    try:
        total = 0.0
        for part in parts:
            total = total * 60.0 + float(part)
    except ValueError:
        return 0.0
    return total


def _noun_list(row: dict[str, str]) -> list[str]:
    try:
        values = ast.literal_eval(row.get("all_nouns") or "[]")
    except (SyntaxError, ValueError):
        return []
    return [str(v).strip().lower() for v in values
            if isinstance(v, str)]


def is_broad_match(row: dict[str, str]) -> bool:
    """Broad stage: scale/weigh/measur in the relevant fields."""
    blob = _blob(row, SEARCH_FIELDS)
    return any(term in blob for term in BROAD_TERMS)


def is_narrow_match(row: dict[str, str]) -> bool:
    """Narrow stage: instrument evidence in nouns or narration."""
    noun_blob = _blob(row, NOUN_FIELDS)
    narration = (row.get("narration") or "").lower()
    verb = (row.get("verb") or "").lower()
    return (
        any(term in noun_blob for term in NARROW_NOUN_TERMS)
        or NARROW_NARRATION_TERM in narration
        or (NARROW_NARRATION_VERB_TERM in narration
            or NARROW_NARRATION_VERB_TERM in verb))


def is_fish_scale(row: dict[str, str]) -> bool:
    """False positive: the noun is a fish scale, not an instrument."""
    nouns = _noun_list(row)
    return (row.get("noun") or "").strip().lower() in FISH_NOUNS \
        and "fish" in nouns


def load_annotations(path: Path, split: str) -> tuple[AnnotationFile,
                                                      list[dict[str, str]]]:
    if not path.is_file():
        raise DiscoveryError(f"Annotation file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"video_id", "narration", "verb", "noun",
                    "all_nouns", "start_timestamp", "stop_timestamp"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise DiscoveryError(
                f"{path} lacks required columns: {sorted(missing)}")
        rows = list(reader)
    return AnnotationFile(path=path, split=split,
                          sha256=_sha256(path), rows=len(rows)), rows


def discover(train_rows: list[dict[str, str]],
             validation_rows: list[dict[str, str]]
             ) -> dict[str, list[dict[str, str]]]:
    """Apply broad, narrow, and fish-exclusion stages."""
    rows = ([{**r, "_split": "train"} for r in train_rows]
            + [{**r, "_split": "validation"} for r in validation_rows])
    broad = [r for r in rows if is_broad_match(r)]
    narrow = [r for r in broad if is_narrow_match(r)]
    curated = [r for r in narrow if not is_fish_scale(r)]
    return {"broad": broad, "narrow": narrow, "curated": curated}


def rank_candidates(curated: list[dict[str, str]]
                    ) -> list[dict[str, object]]:
    """Rank videos by curated segment count, then broad evidence."""
    by_video: dict[str, list[dict[str, str]]] = {}
    for row in curated:
        by_video.setdefault(row["video_id"], []).append(row)
    ranked = []
    for video_id, segments in by_video.items():
        segments.sort(key=lambda r: _to_seconds(r["start_timestamp"]))
        ranked.append({
            "video_id": video_id,
            "segments": len(segments),
            "known_seed": video_id in KNOWN_SEEDS,
            "splits": sorted({r["_split"] for r in segments}),
            "first_timestamp": segments[0]["start_timestamp"],
            "sample_narrations": [
                s["narration"] for s in segments[:3]],
        })
    ranked.sort(key=lambda c: (-int(c["segments"]),
                               str(c["video_id"])))
    return ranked


def write_outputs(out_dir: Path, stages: dict[str, list[dict[str, str]]],
                  files: list[AnnotationFile]) -> dict[str, object]:
    """Write candidates CSV/JSON plus a run summary; return summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    curated = stages["curated"]
    fieldnames = ["video_id", "split", "narration", "verb", "noun",
                  "all_nouns", "start_timestamp", "stop_timestamp",
                  "start_frame", "stop_frame", "stage_fish_excluded"]
    fish_rows = {id(r) for r in stages["narrow"]} - {
        id(r) for r in curated}
    csv_path = out_dir / "candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in stages["narrow"]:
            writer.writerow({
                "video_id": row["video_id"],
                "split": row["_split"],
                "narration": row["narration"],
                "verb": row["verb"],
                "noun": row["noun"],
                "all_nouns": row["all_nouns"],
                "start_timestamp": row["start_timestamp"],
                "stop_timestamp": row["stop_timestamp"],
                "start_frame": row.get("start_frame", ""),
                "stop_frame": row.get("stop_frame", ""),
                "stage_fish_excluded": id(row) in fish_rows,
            })
    ranked = rank_candidates(curated)
    json_path = out_dir / "candidates.json"
    json_path.write_text(json.dumps({
        "candidates": ranked,
        "inputs": [{"path": str(f.path), "split": f.split,
                    "sha256": f.sha256, "rows": f.rows}
                   for f in files],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary: dict[str, object] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "EPIC-KITCHENS-100",
        "licence": ("Annotation release under CC BY-NC 4.0; video "
                    "rights remain with the participants."),
        "stages": {
            name: {
                "segments": len(rows),
                "videos": len({r["video_id"] for r in rows}),
            } for name, rows in stages.items()},
        "fish_excluded_segments": len(fish_rows),
        "ranked_videos": len(ranked),
        "known_seeds_found": sorted(
            c["video_id"] for c in ranked if c["known_seed"]),
        "outputs": {"candidates_csv": str(csv_path),
                    "candidates_json": str(json_path)},
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="EPIC-KITCHENS-100 weighing-scale discovery.")
    parser.add_argument("--train", type=Path, required=True,
                        help="Path to EPIC_100_train.csv")
    parser.add_argument("--validation", type=Path, required=True,
                        help="Path to EPIC_100_validation.csv")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory (kept out of Git)")
    args = parser.parse_args(argv)
    try:
        train_file, train_rows = load_annotations(args.train, "train")
        val_file, val_rows = load_annotations(args.validation,
                                              "validation")
        stages = discover(train_rows, val_rows)
        summary = write_outputs(args.out_dir, stages,
                                [train_file, val_file])
    except DiscoveryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for name, counts in summary["stages"].items():  # type: ignore[union-attr]
        print(f"{name}: {counts['segments']} segments across "
              f"{counts['videos']} videos")
    print(f"fish-scale segments excluded: "
          f"{summary['fish_excluded_segments']}")
    print(f"known seeds found: "
          f"{', '.join(summary['known_seeds_found'])}")  # type: ignore[union-attr]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
