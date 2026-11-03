// Home page regression tests (plain Node, no DOM).
//
// Verifies the chef-first landing requirements that are checkable
// without rendering: the static hero asset exists and is wired in,
// developer copy is gone, and navigation exposes only the three
// chef-facing destinations.

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

const ROOT = process.cwd();
const HERO_PATH = path.join(ROOT, "public/images/bakery-hero.jpg");
const LANDING_SRC = path.join(ROOT, "src/ReviewWorkspace.tsx");
const APP_SRC = path.join(ROOT, "src/App.tsx");

test("hero image exists and is a non-empty JPEG", () => {
  assert.ok(fs.existsSync(HERO_PATH), "missing public/images/bakery-hero.jpg");
  const bytes = fs.readFileSync(HERO_PATH);
  assert.ok(bytes.length > 0, "hero image is empty");
  assert.equal(bytes[0], 0xff, "hero image is not a JPEG");
  assert.equal(bytes[1], 0xd8, "hero image is not a JPEG");
});

test("landing renders the static hero, not the evidence panel", () => {
  const src = fs.readFileSync(LANDING_SRC, "utf-8");
  assert.ok(src.includes('src="/images/bakery-hero.jpg"'),
    "landing does not reference the static hero image");
  assert.ok(src.includes('alt="Freshly baked goods in a bakery"'),
    "hero image has no meaningful alt text");
  assert.ok(!src.includes("Loading real analysis evidence"),
    "black loading panel copy is still present");
  assert.ok(!src.includes("Explore real analysis"),
    "old explore action is still present");
  assert.ok(!src.includes("how-it-works"),
    "How it works section is still present");
  assert.ok(!src.includes("Every reading links back to the frame"),
    "implementation-detail sentence is still present");
});

test("landing guides chefs to the New Analysis workflow", () => {
  const src = fs.readFileSync(LANDING_SRC, "utf-8");
  assert.ok(src.includes("Create my recipe"),
    "primary hero action is missing");
  assert.ok(src.includes("Your first recipe starts here"),
    "onboarding empty state is missing");
  assert.ok(src.includes("Create my first recipe"),
    "empty-state action is missing");
  assert.ok(src.includes("Run an offline analysis first") === false,
    "developer empty-state instructions are still present");
});

test("navigation keeps only Home, New analysis, and queue", () => {
  const src = fs.readFileSync(APP_SRC, "utf-8");
  assert.ok(src.includes("Home"), "Home nav item is missing");
  assert.ok(src.includes("+ New analysis"), "New analysis nav item is missing");
  assert.ok(src.includes("My recipes"), "queue nav item is missing");
  assert.ok(!src.includes("Analyses"),
    "redundant Analyses nav item is still present");
  assert.ok(src.includes('backend.kind !== "online"'),
    "backend badge is not hidden behind the online check");
});

test("featured demo renders only when a demo exists", () => {
  const src = fs.readFileSync(LANDING_SRC, "utf-8");
  assert.ok(src.includes("(featured ? ("),
    "featured section lost its existence guard");
  assert.ok(!src.match(/\)\s*:\s*\(\s*<section id="analyses"/),
    "empty analyses section must not render without a demo");
});
