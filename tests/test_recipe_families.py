"""Offline tests for recipe families, versions, and renames.

Synthetic records in tmp storage. No network, no inference,
no reads from experimental evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from bakery_intelligence.api.app import build_app
from bakery_intelligence.api.recipe_editor import (
    create_record,
    effective_display_name,
    group_families,
    list_recipes,
    merged_recipe,
    rename_recipe,
    version_info_for,
)
from bakery_intelligence.recipe_export import pdf as pdf_mod


def _card(recipe_id: str,
          parent: str | None = None,
          created: str = "2026-01-01T00:00:00Z",
          **extra: Any) -> dict[str, Any]:
    card: dict[str, Any] = {
        "recipe_id": recipe_id,
        "parent_version": parent,
        "created_at": created,
        "updated_at": created,
    }
    card.update(extra)
    return card


def test_singleton_family_numbers_version_one():
    families = group_families([_card("a")])
    assert len(families) == 1
    assert families[0]["family_id"] == "a"
    assert families[0]["family_size"] == 1
    member = families[0]["members"][0]
    assert member["version_number"] == 1
    assert member["parent_version_number"] is None


def test_chain_numbers_follow_creation_order():
    cards = [_card("v1", None, "2026-01-01T00:00:00Z"),
             _card("v2", "v1", "2026-01-02T00:00:00Z"),
             _card("v3", "v2", "2026-01-03T00:00:00Z")]
    families = group_families(cards)
    assert len(families) == 1
    numbers = [m["version_number"]
               for m in families[0]["members"]]
    assert numbers == [1, 2, 3]
    assert families[0]["members"][2][
        "parent_version_number"] == 2


def test_branch_from_earlier_version_keeps_true_parent():
    cards = [_card("v1", None, "2026-01-01T00:00:00Z"),
             _card("v2", "v1", "2026-01-02T00:00:00Z"),
             _card("v3", "v2", "2026-01-03T00:00:00Z"),
             _card("v4", "v3", "2026-01-04T00:00:00Z"),
             _card("v5", "v4", "2026-01-05T00:00:00Z"),
             _card("v6", "v4", "2026-01-06T00:00:00Z")]
    families = group_families(cards)
    assert len(families) == 1
    assert families[0]["family_size"] == 6
    by_id = {m["recipe_id"]: m
             for m in families[0]["members"]}
    assert by_id["v6"]["version_number"] == 6
    assert by_id["v6"]["parent_version"] == "v4"
    assert by_id["v6"]["parent_version_number"] == 4
    assert by_id["v5"]["version_number"] == 5


def test_identical_titles_do_not_merge_families():
    cards = [_card("a", None, "2026-01-01T00:00:00Z",
                   name="Cookies"),
             _card("b", None, "2026-01-02T00:00:00Z",
                   name="Cookies")]
    families = group_families(cards)
    assert len(families) == 2
    assert {f["family_id"] for f in families} == {"a", "b"}


def test_missing_parent_starts_own_family():
    cards = [_card("a"), _card("b", "ghost")]
    families = group_families(cards)
    assert len(families) == 2
    lone = [f for f in families if f["family_id"] == "b"][0]
    assert lone["members"][0]["parent_version_number"] \
        is None


def test_lineage_cycle_degrades_to_singletons():
    cards = [_card("a", "b"), _card("b", "a")]
    families = group_families(cards)
    assert len(families) == 2
    for family in families:
        assert family["family_size"] == 1


def _seed(root: Path, recipe_id: str,
          parent: str | None = None,
          name: str = "Roles") -> None:
    record = create_record(root, name, "es",
                           parent_version=parent,
                           recipe_id=recipe_id)
    assert record["recipe_id"] == recipe_id
    snapshot = {
        "snapshot_id": recipe_id,
        "title": f"AI title {recipe_id}",
        "ingredients": [{"name": "flour"}],
        "method": [{"number": 1, "instruction": "Mix."}],
        "limitations": ["n"],
    }
    (root / "recipes" / recipe_id / "snapshot.json"
     ).write_text(json.dumps(snapshot), encoding="utf-8")


def test_list_recipes_carries_version_fields(tmp_path: Path):
    root = tmp_path / "runs_api"
    _seed(root, "r1")
    _seed(root, "r1_r1", parent="r1")
    cards = {c["recipe_id"]: c for c in list_recipes(root)}
    assert cards["r1"]["version_number"] == 1
    assert cards["r1"]["family_size"] == 2
    assert cards["r1"]["is_latest"] is False
    assert cards["r1_r1"]["version_number"] == 2
    assert cards["r1_r1"]["is_latest"] is True
    assert cards["r1_r1"]["parent_version_number"] == 1
    assert cards["r1"]["family_id"] == \
        cards["r1_r1"]["family_id"] == "r1"


def test_effective_display_name_prefers_override():
    name, custom = effective_display_name(
        {"recipe_id": "r", "name": "Req",
         "display_name": "  Custom  "}, "AI title")
    assert (name, custom) == ("Custom", True)
    name, custom = effective_display_name(
        {"recipe_id": "r", "name": "Req"}, "AI title")
    assert (name, custom) == ("Req", False)
    name, custom = effective_display_name(
        {"recipe_id": "r"}, "AI title")
    assert (name, custom) == ("AI title", False)
    name, custom = effective_display_name(
        {"recipe_id": "r"}, None)
    assert (name, custom) == ("r", False)


def test_rename_sets_trims_and_clears(tmp_path: Path):
    root = tmp_path / "runs_api"
    _seed(root, "r1")
    before = (root / "recipes" / "r1" / "snapshot.json"
              ).read_bytes()
    record = rename_recipe(root, "r1", "  Galletas  ")
    assert record["display_name"] == "Galletas"
    assert (root / "recipes" / "r1" / "snapshot.json"
            ).read_bytes() == before
    assert sorted(p.name for p in
                  (root / "recipes").iterdir()) == ["r1"]
    record = rename_recipe(root, "r1", None)
    assert "display_name" not in record
    with pytest.raises(ValueError, match="non-empty"):
        rename_recipe(root, "r1", "   ")
    with pytest.raises(ValueError, match="exceeds"):
        rename_recipe(root, "r1", "x" * 121)
    with pytest.raises(ValueError, match="Unknown recipe"):
        rename_recipe(root, "missing", "Name")


def test_rename_route_persists_and_validates(
        tmp_path: Path):
    root = tmp_path / "runs_api"
    _seed(root, "r1")
    client = TestClient(build_app(storage_dir=root))
    response = client.post("/api/recipes/r1/rename",
                           json={"name": "Galletas"})
    assert response.status_code == 200
    assert response.json()["display_name"] == "Galletas"
    assert response.json()["has_custom_name"] is True
    listing = client.get("/api/recipes").json()["recipes"]
    assert listing[0]["display_name"] == "Galletas"
    detail = client.get("/api/recipes/r1").json()
    assert detail["display_name"] == "Galletas"
    assert detail["title"] == "AI title r1"
    bad = client.post("/api/recipes/r1/rename",
                      json={"name": "  "})
    assert bad.status_code == 422
    missing = client.post("/api/recipes/nope/rename",
                          json={"name": "x"})
    assert missing.status_code == 404


def test_merged_version_info_lists_navigable_history(
        tmp_path: Path):
    root = tmp_path / "runs_api"
    _seed(root, "r1")
    _seed(root, "r1_r1", parent="r1")
    rename_recipe(root, "r1_r1", "Segunda")
    info = version_info_for(root, "r1_r1")
    assert info["version_number"] == 2
    assert info["family_size"] == 2
    assert info["is_latest"] is True
    assert info["parent_version"] == "r1"
    assert info["parent_version_number"] == 1
    assert [m["recipe_id"] for m in info["members"]] == \
        ["r1", "r1_r1"]
    assert info["members"][1]["display_name"] == "Segunda"
    merged = merged_recipe(root, "r1_r1")
    assert merged["version_info"]["version_number"] == 2
    assert merged["display_name"] == "Segunda"


def test_branch_allocation_scopes_children_per_parent(
        tmp_path: Path):
    from bakery_intelligence.api.app import _next_version_id

    root = tmp_path / "runs_api"
    _seed(root, "v4")
    _seed(root, "v5", parent="v4")
    first = _next_version_id(root, "v4")
    assert first == "v4_r1"
    assert first != "v5"
    _seed(root, first, parent="v4")
    assert _next_version_id(root, "v4") == "v4_r2"
    ghost = root / "recipes" / "v4_ghost"
    ghost.mkdir(parents=True)
    (ghost / "revision_raw.json").write_text("{}",
                                             encoding="utf-8")
    assert _next_version_id(root, "v4_ghost") == \
        "v4_ghost_r1"


def _pdf_snapshot(**over: Any) -> dict[str, Any]:
    shot: dict[str, Any] = {
        "snapshot_id": "r1",
        "title": "AI title",
        "recipe_status": "draft",
        "ingredients": [{
            "name": "Polvo blanco fino para pasta cocida "
                    "(apariencia harina)",
            "quantity": 40.0, "unit": "g",
            "quantity_status": "seen_video",
            "quantity_basis": "Displayed 40 on scale.",
            "chef_edited": False, "original": None}],
        "method": [{"number": 1, "instruction": "Mix well."}],
        "limitations": ["Note one."],
    }
    shot.update(over)
    return shot


def _pdf_text(data: bytes, tmp_path: Path) -> str:
    import pymupdf

    target = tmp_path / "v.pdf"
    target.write_bytes(data)
    with pymupdf.open(target) as doc:
        return "\n".join(page.get_text() for page in doc)


def test_pdf_shows_version_and_effective_name(
        tmp_path: Path):
    original = _pdf_snapshot(version_info={
        "version_number": 1, "family_size": 2})
    text = _pdf_text(pdf_mod.build_pdf(original), tmp_path)
    assert "Version 1 - Original" in text
    revised = _pdf_snapshot(
        display_name="Mi versión",
        parent_version="r1",
        version_info={"version_number": 2, "family_size": 2,
                      "parent_version_number": 1})
    text = _pdf_text(pdf_mod.build_pdf(revised), tmp_path)
    assert "Mi versi" in text
    assert "Version 2 - Chef Revision" in text
    assert "Based on Version 1" in text


def test_pdf_version_line_defaults_without_info():
    assert pdf_mod.version_line(
        {"title": "t"}) == "Version 1 - Original"
    assert pdf_mod.version_line(
        {"title": "t",
         "parent_version": "p"}) == "Chef Revision"
