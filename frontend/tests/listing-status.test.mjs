// Home listing status test: real render in jsdom.
//
// Cards in failed / awaiting_build / processing states must show
// truthful labels, and the latest row must not repeat the family
// head name.

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

const { JSDOM } = require("jsdom");

function card(over) {
  return {
    recipe_id: "r1",
    name: "Stale Dish",
    display_name: "Stale Dish",
    has_custom_name: false,
    language: "en",
    source_run_id: "run_x",
    parent_version: null,
    parent_version_number: null,
    family_id: "r1",
    family_size: 1,
    version_number: 1,
    is_latest: true,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ingredients: 0,
    steps: 0,
    has_snapshot: false,
    status: "processing",
    ...over,
  };
}

const CARDS = [
  card({}),
  card({
    recipe_id: "r2",
    name: "Broken Dish",
    display_name: "Broken Dish",
    family_id: "r2",
    status: "failed",
    build_error: "Stage audio failed.",
  }),
  card({
    recipe_id: "r3",
    name: "Built Dish",
    display_name: "Built Dish",
    family_id: "r3",
    status: "awaiting_build",
  }),
];

async function renderApp() {
  globalThis.fetch = async (url) => {
    if (String(url).endsWith("/api/recipes")) {
      return {
        ok: true, status: 200,
        json: async () => ({ recipes: CARDS }),
      };
    }
    return { ok: true, status: 200, json: async () => ({}) };
  };
  const dom = new JSDOM("<!DOCTYPE html><div id=\"root\"></div>", {
    url: "http://localhost/",
  });
  for (const key of ["window", "document", "HTMLElement",
    "Element", "Node", "MouseEvent", "Event", "KeyboardEvent",
    "getComputedStyle"]) {
    globalThis[key] = dom.window[key] ?? globalThis[key];
  }
  if (!globalThis.KeyboardEvent) {
    globalThis.KeyboardEvent = dom.window.Event;
  }
  Object.defineProperty(globalThis, "navigator", {
    value: dom.window.navigator,
    configurable: true,
  });
  const React = require("react");
  const act = React.act
    ?? require("react-dom/test-utils").act;
  const { createRoot } = require("react-dom/client");
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const App = require("../src/App.tsx").default;
  const container = document.getElementById("root");
  const root = createRoot(container);
  await act(async () => {
    root.render(React.createElement(App));
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 150));
  });
  return { container, cleanup: () => root.unmount() };
}

test("listing shows truthful processing states", async () => {
  const { container, cleanup } = await renderApp();
  try {
    const cards = Array.from(
      container.querySelectorAll("li.family-card"));
    assert.equal(cards.length, 3);
    const byName = (name) => cards.find((card) =>
      card.textContent.includes(name)).textContent;
    assert.ok(byName("Stale Dish").includes("Processing"));
    assert.ok(byName("Broken Dish").includes("Failed"));
    assert.ok(
      byName("Built Dish").includes("Ready to build"));
  } finally {
    cleanup();
  }
});

test("view action sits in a footer below the title", async () => {
  const { container, cleanup } = await renderApp();
  try {
    const card = Array.from(
      container.querySelectorAll("li.family-card"))
      .find((el) => el.textContent.includes("Stale Dish"));
    const head = card.querySelector(".family-head");
    assert.ok(
      !head.querySelector("button"),
      "family head must not contain the action button");
    const foot = card.querySelector(".family-foot");
    assert.ok(foot, "family footer is missing");
    const action = foot.querySelector("button");
    assert.ok(action, "footer action is missing");
    assert.equal(action.textContent.trim(), "View Recipe");
    const badge = head.querySelector(".version-badge");
    assert.ok(badge, "version badge is missing");
    assert.equal(badge.textContent.trim(), "v1 \u00b7 Latest");
  } finally {
    cleanup();
  }
});

test("latest row does not repeat the family name", async () => {
  const { container, cleanup } = await renderApp();
  try {
    const card = Array.from(
      container.querySelectorAll("li.family-card"))
      .find((el) => el.textContent.includes("Stale Dish"));
    const head = card.querySelector(".family-head");
    assert.ok(head.textContent.includes("Stale Dish"));
    const row = card.querySelector(
      ".family-versions .version-row");
    assert.ok(
      !row.textContent.includes("Stale Dish"),
      "latest row repeats the family head name");
    assert.ok(row.textContent.includes("v1 · Latest"));
  } finally {
    cleanup();
  }
});
