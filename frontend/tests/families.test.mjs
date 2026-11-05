// Recipe family + inline rename tests: real clicks in jsdom.
//
// Renders the actual <App/> landing with a stubbed backend:
// one 2-version family plus an unrelated same-title recipe.
// Exercises grouping, history disclosure, and the rename flow.

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
    name: "Roles de Canela XL",
    display_name: "Roles de Canela XL",
    has_custom_name: false,
    language: "es",
    source_run_id: "run_x",
    parent_version: null,
    parent_version_number: null,
    family_id: "r1",
    family_size: 2,
    version_number: 1,
    is_latest: false,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ingredients: 9,
    steps: 16,
    has_snapshot: true,
    status: "draft",
    ...over,
  };
}

const FAMILY = [
  card({}),
  card({
    recipe_id: "r1_r1",
    parent_version: "r1",
    parent_version_number: 1,
    version_number: 2,
    is_latest: true,
    created_at: "2026-01-02T00:00:00Z",
    updated_at: "2026-01-02T00:00:00Z",
    ingredients: 11,
  }),
  card({
    recipe_id: "other",
    family_id: "other",
    family_size: 1,
    version_number: 1,
    is_latest: true,
    created_at: "2026-01-03T00:00:00Z",
    updated_at: "2026-01-03T00:00:00Z",
  }),
];

let posted = [];

function stubFetch() {
  posted = [];
  globalThis.fetch = async (url, options) => {
    const text = String(url);
    if (text.endsWith("/api/recipes")) {
      return {
        ok: true, status: 200,
        json: async () => ({ recipes: FAMILY }),
      };
    }
    if (text.includes("/rename")) {
      const body = JSON.parse(options.body);
      posted.push(body);
      return {
        ok: true, status: 200,
        json: async () => ({
          recipe_id: "r1_r1",
          display_name: body.name,
          has_custom_name: true,
          name: "Roles de Canela XL",
          updated_at: "2026-01-04T00:00:00Z",
        }),
      };
    }
    return { ok: true, status: 200, json: async () => ({}) };
  };
}

async function renderApp() {
  stubFetch();
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
  return { act, container, cleanup: () => root.unmount() };
}

function families(container) {
  return Array.from(
    container.querySelectorAll("li.family-card"));
}

function familyWithVersions(container, label) {
  const found = families(container).find((card) =>
    card.textContent.includes(label));
  assert.ok(found, `no family card shows ${label}`);
  return found;
}

test("related versions group into one family card", async () => {
  const { container, cleanup } = await renderApp();
  try {
    const cards = families(container);
    assert.equal(cards.length, 2);
    const first = familyWithVersions(
      container, "2 versions").textContent;
    assert.ok(first.includes("2 versions"),
      "family version count is missing");
    assert.ok(first.includes("v2 · Latest"),
      "latest version is not prominent");
    assert.ok(first.includes("Chef Revision"),
      "revision designation is missing");
  } finally {
    cleanup();
  }
});

test("identical titles stay in separate families", async () => {
  const { container, cleanup } = await renderApp();
  try {
    const cards = families(container);
    assert.equal(cards.length, 2);
    assert.ok(cards[0].textContent.includes("v1 · Latest"),
      "unrelated singleton family is missing");
  } finally {
    cleanup();
  }
});

test("older versions hide inside version history", async () => {
  const { container, cleanup } = await renderApp();
  try {
    const card = familyWithVersions(container, "2 versions");
    const history = card.querySelector("details.family-history");
    assert.ok(history, "version history disclosure missing");
    assert.ok(history.textContent.includes("Version history"),
      "history summary missing");
  } finally {
    cleanup();
  }
});

test("rename persists and updates the displayed name", async () => {
  const { act, container, cleanup } = await renderApp();
  try {
    const card = familyWithVersions(container, "2 versions");
    const rename = Array.from(card.querySelectorAll("button"))
      .find((b) => b.textContent === "Rename");
    assert.ok(rename, "rename button is missing");
    await act(async () => {
      rename.dispatchEvent(
        new MouseEvent("click", { bubbles: true }));
    });
    const input = card.querySelector(
      '.rename-editor input[aria-label="Recipe name"]');
    assert.ok(input, "rename input did not open");
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype, "value").set;
    await act(async () => {
      setter.call(input, "  Galletas  ");
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    const save = Array.from(
      card.querySelectorAll(".rename-editor button"))
      .find((b) => b.textContent === "Save");
    await act(async () => {
      save.dispatchEvent(
        new MouseEvent("click", { bubbles: true }));
      await new Promise((r) => setTimeout(r, 100));
    });
    assert.equal(posted.length, 1);
    assert.equal(posted[0].name, "Galletas");
    assert.ok(card.textContent.includes("Galletas"),
      "renamed title is not displayed immediately");
  } finally {
    cleanup();
  }
});

test("rename rejects empty names and Escape cancels", async () => {
  const { act, container, cleanup } = await renderApp();
  try {
    const card = familyWithVersions(container, "2 versions");
    const rename = Array.from(card.querySelectorAll("button"))
      .find((b) => b.textContent === "Rename");
    await act(async () => {
      rename.dispatchEvent(
        new MouseEvent("click", { bubbles: true }));
    });
    const input = card.querySelector(".rename-editor input");
    const setter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype, "value").set;
    await act(async () => {
      setter.call(input, "   ");
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    const save = Array.from(
      card.querySelectorAll(".rename-editor button"))
      .find((b) => b.textContent === "Save");
    await act(async () => {
      save.dispatchEvent(
        new MouseEvent("click", { bubbles: true }));
    });
    assert.equal(posted.length, 0);
    assert.ok(card.textContent.includes("Name cannot be empty"),
      "empty-name validation message is missing");
    await act(async () => {
      input.dispatchEvent(
        new KeyboardEvent("keydown",
          { key: "Escape", bubbles: true }));
    });
    assert.equal(
      card.querySelector(".rename-editor input"), null,
      "Escape did not close the rename editor");
  } finally {
    cleanup();
  }
});
