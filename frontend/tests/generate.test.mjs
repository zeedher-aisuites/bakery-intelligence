// Generate stage tests: one primary action that always runs the real
// AI workflow with explicit confirmation, never simulation.

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

const RUN_ID = "run_generate_test";
const posted = [];
let wentBack = false;

function stubFetch({ readiness, queue }) {
  posted.length = 0;
  wentBack = false;
  globalThis.fetch = async (url, options) => {
    const text = String(url);
    const method = (options?.method ?? "GET").toUpperCase();
    if (method === "POST") {
      posted.push({
        url: text,
        body: options?.body ? JSON.parse(String(options.body)) : null,
      });
      return { ok: true, status: 200, json: async () => ({}) };
    }
    if (text.endsWith("/recipe/readiness")) {
      return { ok: true, status: 200, json: async () => readiness };
    }
    if (text.endsWith(`/api/analyses/${RUN_ID}/observation-review`)) {
      return { ok: true, status: 200, json: async () => queue };
    }
    return { ok: true, status: 200, json: async () => ({}) };
  };
}

async function renderPanel({ readiness, queue }) {
  stubFetch({ readiness, queue });
  const dom = new JSDOM("<!DOCTYPE html><div id=\"root\"></div>", {
    url: "http://localhost/",
  });
  for (const key of ["window", "document", "HTMLElement", "Element",
    "Node", "MouseEvent", "Event", "InputEvent", "HTMLInputElement",
    "getComputedStyle", "localStorage"]) {
    if (dom.window[key] !== undefined) globalThis[key] = dom.window[key];
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
  const GeneratePanel = require("../src/GeneratePanel.tsx").default;
  const container = document.getElementById("root");
  const root = createRoot(container);
  await act(async () => {
    root.render(React.createElement(GeneratePanel, {
      runId: RUN_ID,
      onBackToFindings: () => { wentBack = true; },
    }));
    await new Promise((resolve) => setTimeout(resolve, 300));
  });
  return { act, dom, container, cleanup: () => root.unmount() };
}

const OPEN_QUEUE = { tasks: [] };

const READY = {
  ready: true,
  open_tasks: 0,
  live_generation: true,
  blocking_errors: [],
  deferred: 0,
  live_available: true,
  live_blockers: ["billed per request"],
  cost_note: "note from backend",
};

const BLOCKED = {
  ready: false,
  open_tasks: 1,
  live_generation: false,
  blocking_errors: ["task-9"],
  deferred: 0,
  live_available: true,
  live_blockers: [],
  cost_note: null,
};

const BLOCKED_QUEUE = {
  tasks: [{
    task_id: "task-9",
    field: "scale_value",
    state: "contradictory",
    status: "open",
    frame_refs: [4],
    start_seconds: 40,
    end_seconds: 42,
    correction: null,
  }],
};

async function clickButton(act, button) {
  await act(async () => {
    button.dispatchEvent(
      new globalThis.MouseEvent("click", { bubbles: true }));
    await new Promise((resolve) => setTimeout(resolve, 150));
  });
}

test("generation offers one action that cannot invoke simulation", async () => {
  const { act, container, cleanup } = await renderPanel({
    readiness: READY, queue: OPEN_QUEUE,
  });
  try {
    assert.ok(!/simulation|practice version/i.test(container.textContent),
      "simulation is not part of the chef-facing workflow");
    const buttons = Array.from(container.querySelectorAll("button"));
    assert.deepEqual(
      buttons.map((button) => button.textContent),
      ["Generate Recipe"],
      "a single primary action is presented");
    const action = buttons[0];
    assert.ok(action.disabled,
      "billable inference waits for explicit confirmation");
    const confirm = container.querySelector('input[type="checkbox"]');
    await clickButton(act, confirm);
    assert.ok(!action.disabled, "confirmation unlocks the action");
    await clickButton(act, action);
    const call = posted.find((entry) => entry.url.endsWith(
      "/recipe/generate"));
    assert.ok(call, "the action calls the real generation endpoint");
    assert.deepEqual(call.body,
      { mode: "live", confirmed: true, max_rounds: 1 },
      "the chef-facing action can only request live generation");
  } finally {
    cleanup();
  }
});

test("blocked generation explains, navigates back, and hides cost text", async () => {
  const { act, container, cleanup } = await renderPanel({
    readiness: BLOCKED, queue: BLOCKED_QUEUE,
  });
  try {
    assert.ok(container.textContent.includes(
      "1 conflicting finding still blocks generation."),
      "blocking is explained in plain language");
    const action = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent === "Generate Recipe");
    assert.ok(action && action.disabled,
      "blocked generation cannot fire a billable request");
    const back = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent === "Review findings");
    assert.ok(back, "blocked stage offers a direct way back");
    await clickButton(act, back);
    assert.ok(wentBack, "back navigation fires");
    const details = container.querySelector("details");
    assert.ok(details && !details.open,
      "technical details stay collapsed by default");
    assert.ok(details.textContent.includes("task-9"),
      "raw identifier stays available for diagnosis");
    assert.ok(!container.textContent.includes("cannot be estimated"),
      "cost explanation is removed everywhere, disclosure included");
    assert.ok(!container.textContent.includes("note from backend"),
      "backend cost notes are not rendered in the chef workflow");
  } finally {
    cleanup();
  }
});
