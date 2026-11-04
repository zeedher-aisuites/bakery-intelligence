// Chef-feedback regression tests (plain Node, no DOM).
//
// Verifies the recipe-only Chef Review, New Recipe metadata, and
// AI revision wiring that are checkable without rendering.

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

const ROOT = process.cwd();
const EDITOR_SRC = path.join(ROOT, "src/RecipeEditor.tsx");
const WORKSPACE_SRC = path.join(ROOT, "src/RecipeWorkspace.tsx");
const UPLOAD_SRC = path.join(ROOT, "src/UploadWorkflow.tsx");
const API_SRC = path.join(ROOT, "src/api.ts");
const APP_SRC = path.join(ROOT, "src/App.tsx");

test("editor is recipe-only with the simplified action bar", () => {
  const src = fs.readFileSync(EDITOR_SRC, "utf-8");
  assert.ok(!src.includes("Request changes"),
    "Request changes action is still present");
  assert.ok(!src.includes("needs_changes"),
    "needs-changes status is still present");
  assert.ok(!src.includes("saved.interval"),
    "timestamp intervals are still displayed in method cards");
  assert.ok(!src.includes("saved.caveats"),
    "technical caveats are still displayed in method cards");
  assert.ok(!src.includes("review_flags"),
    "technical review flags are still displayed");
  assert.ok(!src.includes("completion_open_items"),
    "passive open-items list is still displayed");
  assert.ok(src.includes("Generate New Version")
    || src.includes("generate"),
    "Generate New Version action is missing");
  assert.ok(src.includes("improvement_suggestions"),
    "selectable improvement suggestions are missing");
  assert.ok(src.includes("allow_ingredient_changes"),
    "ingredient protection flag is missing");
});

test("editor uses human-readable provenance labels", () => {
  const src = fs.readFileSync(EDITOR_SRC, "utf-8");
  assert.ok(src.includes("Seen in video"),
    "Seen in video label is missing");
  assert.ok(src.includes("Mentioned in audio"),
    "Mentioned in audio label is missing");
  assert.ok(src.includes("Suggested by AI"),
    "Suggested by AI label is missing");
  assert.ok(src.includes("Edited by chef"),
    "Edited by chef label is missing");
  assert.ok(src.includes("Visto en el video"),
    "Spanish provenance labels are missing");
  assert.ok(!src.includes("AI estimate"),
    "technical AI estimate label is still present");
  assert.ok(!src.includes("Displayed "),
    "confusing Displayed label is still present");
});

test("workspace has no evidence tab for chefs", () => {
  const src = fs.readFileSync(WORKSPACE_SRC, "utf-8");
  assert.ok(!src.includes("FusionReview"),
    "technical evidence view is still wired into Chef Review");
  assert.ok(!src.includes("rc-tabs"),
    "Recipe/Evidence tab selector is still present");
  assert.ok(src.includes("has_snapshot"),
    "processing state for recipes without drafts is missing");
});

test("new recipe captures name and language", () => {
  const src = fs.readFileSync(UPLOAD_SRC, "utf-8");
  assert.ok(src.includes("Recipe Name"),
    "recipe name field is missing");
  assert.ok(src.includes("Recipe Language"),
    "language selector is missing");
  assert.ok(src.includes("createRecipe"),
    "recipe record creation is missing");
  assert.ok(!src.includes("maximum budget"),
    "budget consent copy is still present");
  assert.ok(!src.includes("MAX_AUTHORIZED_COST_USD"),
    "budget constant is still referenced in the UI");
});

test("api client supports recipes and revisions", () => {
  const src = fs.readFileSync(API_SRC, "utf-8");
  assert.ok(src.includes("listRecipes"),
    "listRecipes client is missing");
  assert.ok(src.includes("createRecipe"),
    "createRecipe client is missing");
  assert.ok(src.includes("reviseRecipe"),
    "reviseRecipe client is missing");
  assert.ok(src.includes("/revise"),
    "revision endpoint is missing");
  assert.ok(src.includes("buildRecipe"),
    "buildRecipe client is missing");
  assert.ok(src.includes("/build"),
    "build endpoint is missing");
});

test("workspace auto-builds from analysis to ready", () => {
  const src = fs.readFileSync(WORKSPACE_SRC, "utf-8");
  assert.ok(src.includes("buildRecipe"),
    "workspace does not trigger the recipe build");
  assert.ok(src.includes("fetchAnalysis"),
    "workspace does not track analysis progress");
  assert.ok(src.includes("already in progress"),
    "workspace does not handle concurrent builds");
  assert.ok(src.includes("Try again"),
    "failed builds have no retry action");
  assert.ok(!src.includes("bakery_demo_v1_proposed_v2"),
    "workspace falls back to the demo recipe");
});

test("recipe context survives reload via hash routing", () => {
  const src = fs.readFileSync(APP_SRC, "utf-8");
  assert.ok(src.includes("#recipe/"),
    "recipe hash routing is missing");
  assert.ok(src.includes("initialRecipeId"),
    "initial recipe restore is missing");
});
