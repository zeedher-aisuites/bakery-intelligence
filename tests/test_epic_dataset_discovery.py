"""Isolated tests for the EPIC-KITCHENS discovery script.

Synthetic fixtures only; the real annotation files are never read.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "epic_dataset_discovery.py"


def _load_module():
    import sys

    spec = importlib.util.spec_from_file_location(
        "epic_dataset_discovery", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["epic_dataset_discovery"] = module
    spec.loader.exec_module(module)
    return module


discovery = _load_module()

FIELDS = ["narration_id", "participant_id", "video_id",
          "narration_timestamp", "start_timestamp", "stop_timestamp",
          "start_frame", "stop_frame", "narration", "verb",
          "verb_class", "noun", "noun_class", "all_nouns",
          "all_noun_classes"]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def _row(video: str, narration: str, verb: str = "take",
         noun: str = "bowl", all_nouns: str = "['bowl']",
         start: str = "1.0") -> dict[str, str]:
    return {"narration_id": f"{video}-{start}", "participant_id": "P1",
            "video_id": video, "narration_timestamp": start,
            "start_timestamp": start, "stop_timestamp": "2.0",
            "start_frame": "10", "stop_frame": "20",
            "narration": narration, "verb": verb, "verb_class": "1",
            "noun": noun, "noun_class": "2", "all_nouns": all_nouns,
            "all_noun_classes": "[2]"}


def test_broad_narrow_and_fish_exclusion(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    _write_csv(train, [
        _row("V1", "take spoon", noun="spoon:measuring",
             all_nouns="['spoon:measuring']"),
        _row("V1", "weigh the flour", verb="weigh", noun="bowl",
             all_nouns="['bowl']", start="3.0"),
        _row("V2", "remove the scale off the fish", noun="scale",
             all_nouns="['scale', 'fish']"),
        _row("V2", "take cup", noun="cup"),
    ])
    val = tmp_path / "val.csv"
    _write_csv(val, [_row("V3", "put scale on table", noun="scale",
                          all_nouns="['scale']")])
    train_file, train_rows = discovery.load_annotations(train, "train")
    val_file, val_rows = discovery.load_annotations(val, "validation")
    assert train_file.rows == 4 and val_file.rows == 1
    stages = discovery.discover(train_rows, val_rows)
    assert len(stages["broad"]) == 4
    assert len(stages["narrow"]) == 3
    assert len(stages["curated"]) == 2
    ranked = discovery.rank_candidates(stages["curated"])
    assert [c["video_id"] for c in ranked] == ["V1", "V3"]
    assert ranked[0]["segments"] == 1


def test_missing_columns_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    with bad.open("w", encoding="utf-8") as handle:
        handle.write("video_id,narration\nV1,hello\n")
    try:
        discovery.load_annotations(bad, "train")
    except discovery.DiscoveryError:
        return
    raise AssertionError("expected DiscoveryError")


def test_to_seconds_parses_both_timestamp_formats() -> None:
    assert discovery._to_seconds("12.5") == 12.5
    assert discovery._to_seconds("00:01:17.00") == 77.0
    assert discovery._to_seconds("") == 0.0
    assert discovery._to_seconds("n/a") == 0.0


def test_outputs_are_reproducible_and_ranked(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    _write_csv(train, [
        _row("V9", "weigh sugar", verb="weigh", start="1.0"),
        _row("V9", "take scale", noun="scale",
             all_nouns="['scale']", start="5.0"),
        _row("V2", "take bowl"),
    ])
    val = tmp_path / "val.csv"
    _write_csv(val, [])
    train_file, train_rows = discovery.load_annotations(train, "train")
    val_file, val_rows = discovery.load_annotations(val, "validation")
    stages = discovery.discover(train_rows, val_rows)
    out = tmp_path / "out"
    summary = discovery.write_outputs(out, stages,
                                      [train_file, val_file])
    assert summary["stages"]["curated"]["segments"] == 2
    assert summary["ranked_videos"] == 1
    payload = json.loads((out / "candidates.json").read_text(
        encoding="utf-8"))
    assert payload["candidates"][0]["video_id"] == "V9"
    assert payload["inputs"][0]["rows"] == 3
    with (out / "candidates.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["split"] == "train"
