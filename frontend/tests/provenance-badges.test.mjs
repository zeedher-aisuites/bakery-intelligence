// Provenance badge mapping tests.
//
// The five evidence states must map to visually and semantically
// appropriate treatments. In particular, a human correction is a
// human assertion, never independent verification: only a
// confirmation ("human-verified") earns the verified treatment.

import assert from "node:assert/strict";
import fs from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const esbuild = require("esbuild");

for (const [ext, loader] of [[".tsx", "tsx"], [".ts", "ts"]]) {
  require.extensions[ext] = function (module, filename) {
    const source = fs.readFileSync(filename, "utf8");
    const { code } = esbuild.transformSync(source, {
      loader,
      jsx: "automatic",
      format: "cjs",
    });
    module._compile(code, filename);
  };
}
require.extensions[".css"] = function (module) {
  module.exports = {};
};

const { readingBadge } = require("../src/review.ts");
const { provenanceBadge } = require("../src/RecipeReview.tsx");

test("readingBadge maps the five evidence states", () => {
  assert.equal(
    readingBadge("human-verified"), "badge badge-verified");
  assert.equal(
    readingBadge("ai-extracted"), "badge badge-ai");
  assert.equal(
    readingBadge("human-corrected"), "badge badge-corrected");
  assert.equal(readingBadge("unknown"), "badge badge-unknown");
});

test("readingBadge never equates correction with verification", () => {
  assert.notEqual(
    readingBadge("human-corrected"), readingBadge("human-verified"));
});

test("readingBadge leaves rejected and uncertain on pills", () => {
  assert.equal(readingBadge("rejected"), null);
  assert.equal(readingBadge("uncertain"), null);
});

test("provenanceBadge maps recipe provenances", () => {
  assert.equal(
    provenanceBadge("ai-extracted"), "badge badge-ai");
  assert.equal(
    provenanceBadge("chef-supplied"), "badge badge-chef");
  assert.equal(
    provenanceBadge("human-corrected"), "badge badge-corrected");
  assert.equal(provenanceBadge(""), "badge badge-unknown");
  assert.equal(provenanceBadge("mystery"), "badge badge-unknown");
});

test("provenanceBadge never labels model output verified", () => {
  for (const provenance of ["ai-extracted", "chef-supplied",
    "human-corrected", "", "mystery"]) {
    assert.ok(!provenanceBadge(provenance).includes("badge-verified"),
      `${provenance} must not earn the verified treatment`);
  }
});
