// Home CTA navigation tests: real button clicks in jsdom.
//
// Renders the actual <App/> with a stubbed backend (zero analyses),
// clicks "Create my recipe" and "Create my first recipe", and asserts
// the New Analysis view appears, exactly like the "+ New analysis"
// header control. Static copy assertions live in hero.test.mjs.

import assert from "node:assert/strict";
import fs from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
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

function stubFetch() {
  globalThis.fetch = async (url) => {
    const text = String(url);
    if (text.includes("/api/analyses")) {
      return { ok: true, status: 200, json: async () => [] };
    }
    return {
      ok: true,
      status: 200,
      json: async () => ({
        status: "ok",
        service: "test",
        version: "0",
        storage_backend: "LocalStorageBackend",
        storage_ready: true,
      }),
    };
  };
}

async function renderApp() {
  stubFetch();
  const dom = new JSDOM("<!DOCTYPE html><div id=\"root\"></div>", {
    url: "http://localhost/",
  });
  for (const key of ["window", "document", "HTMLElement",
    "Element", "Node", "MouseEvent", "Event", "getComputedStyle"]) {
    globalThis[key] = dom.window[key];
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
    await new Promise((resolve) => setTimeout(resolve, 100));
  });
  return { act, container, cleanup: () => root.unmount() };
}

async function clickButton(act, container, label) {
  const buttons = Array.from(container.querySelectorAll("button"));
  const target = buttons.find(
    (button) => button.textContent.trim() === label);
  assert.ok(target, `button "${label}" is rendered`);
  await act(async () => {
    target.dispatchEvent(
      new globalThis.MouseEvent("click", { bubbles: true }));
    await new Promise((resolve) => setTimeout(resolve, 50));
  });
}

function showsNewAnalysis(container) {
  const headings = Array.from(container.querySelectorAll("h2")).map(
    (heading) => heading.textContent);
  return headings.includes("New recipe")
    && headings.includes("Upload video");
}

test("hero CTA navigates to New Analysis on click", async () => {
  const { act, container, cleanup } = await renderApp();
  try {
    await clickButton(act, container, "Create my recipe");
    assert.ok(showsNewAnalysis(container),
      "New Analysis view is not displayed after hero CTA click");
  } finally {
    cleanup();
  }
});

test("empty-state CTA navigates to New Analysis on click", async () => {
  const { act, container, cleanup } = await renderApp();
  try {
    await clickButton(act, container, "Create my first recipe");
    assert.ok(showsNewAnalysis(container),
      "New Analysis view is not displayed after empty-state CTA click");
  } finally {
    cleanup();
  }
});

test("header control navigates to New Analysis on click", async () => {
  const { act, container, cleanup } = await renderApp();
  try {
    await clickButton(act, container, "+ New analysis");
    assert.ok(showsNewAnalysis(container),
      "New Analysis view is not displayed after header navigation click");
  } finally {
    cleanup();
  }
});
