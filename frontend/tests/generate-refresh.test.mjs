// Generate-stage freshness regression tests.
//
// Eight findings are resolved in Review Findings, but the Generate
// panel used to keep the readiness snapshot from its first mount:
// both stages must agree on the same persisted queue state.

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

const RUN_ID = "run_generate_refresh_test";

function bundleFrame(index) {
  return {
    frame_index: index,
    timestamp_seconds: index * 10 + 2.5,
    image_available: true,
    image_name: `frame_${String(index).padStart(6, "0")}.jpg`,
    image_sha256: "ab".repeat(32),
    image_matches_record: true,
    observation: {
      model_fields: {
        scale_display_verbatim: "120 g",
        container_description: "bowl",
        scale_readable: true,
        scale_state: "stable",
        confidence: "high",
        ambiguity_notes: null,
      },
      terminal_state: "valid_prediction",
      request_id: `req-${index}`,
      latency_seconds: 1.2,
      input_tokens: 100,
      output_tokens: 50,
      reasoning_output_tokens: null,
      response_status: "completed",
      parse_error: null,
    },
    measurement: { value: 120, value_status: "ok", unit_observed: "g" },
    review: null,
  };
}

const BUNDLE = {
  run_id: RUN_ID,
  workflow: "test-pipeline",
  status: "complete",
  source_object_id: "upload_abcd1234abcd5678_P35_101_clip.mp4",
  source_sha256: null,
  frames: Array.from({ length: 8 }, (_, index) => bundleFrame(index)),
  reviews: {},
  missing: [],
};

// All eight findings carry a saved human decision.
const QUEUE = {
  tasks: Array.from({ length: 8 }, (_, index) => ({
    task_id: `task-${index + 1}`,
    field: "scale_value",
    state: index === 2 ? "contradictory" : "missing_relevant",
    status: "resolved",
    frame_refs: [index],
    start_seconds: index * 10 + 2.5,
    end_seconds: index * 10 + 2.5,
    correction: {
      action: "confirm",
      human_value: null,
      reviewer: "chef-test",
      revision: 1,
    },
  })),
  evaluated: [],
  counts: { frames: 8, usable_readings: 8, abstentions: 0, accepted: 1 },
};

// While the review is still in progress the backend reports the stale
// mid-review snapshot; once the chef finishes all eight findings the
// persisted queue reports everything resolved. The test flips this
// flag after mount to simulate corrections saved after the Generate
// panel's first fetch.
const STALE_READINESS = {
  ready: false,
  open_tasks: 3,
  live_generation: false,
  blocking_errors: ["task-3"],
  deferred: 1,
  live_available: true,
  live_blockers: [],
  cost_note: null,
};

const FRESH_READINESS = {
  ready: true,
  open_tasks: 0,
  live_generation: false,
  blocking_errors: [],
  deferred: 0,
  live_available: true,
  live_blockers: [],
  cost_note: null,
};

const posted = [];
let readinessCalls = 0;
let serveFreshReadiness = false;

function stubFetch() {
  posted.length = 0;
  readinessCalls = 0;
  serveFreshReadiness = false;
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
    if (text.endsWith(`/api/analyses/${RUN_ID}/review`)) {
      return { ok: true, status: 200, json: async () => BUNDLE };
    }
    if (text.endsWith(`/api/analyses/${RUN_ID}/observation-review`)) {
      return { ok: true, status: 200, json: async () => QUEUE };
    }
    if (text.endsWith("/recipe/evidence")) {
      return { ok: true, status: 200, json: async () => ({ frames: [] }) };
    }
    if (text.endsWith("/recipe/readiness")) {
      readinessCalls += 1;
      const payload = serveFreshReadiness ? FRESH_READINESS : STALE_READINESS;
      return {
        ok: true,
        status: 200,
        json: async () => payload,
      };
    }
    if (text.endsWith(`/api/analyses/${RUN_ID}/recipe`)) {
      return {
        ok: false,
        status: 404,
        json: async () => ({ detail: "no recipe yet" }),
      };
    }
    if (text.endsWith(`/api/analyses/${RUN_ID}`)) {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          run_id: RUN_ID,
          workflow: "test-pipeline",
          status: "complete",
          source_object_id: BUNDLE.source_object_id,
          source_sha256: null,
          artifact_ids: [],
          error: null,
          attempts: 1,
        }),
      };
    }
    return { ok: true, status: 200, json: async () => ({}) };
  };
}

async function renderWorkspace() {
  stubFetch();
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
  const ReviewApp = require("../src/ReviewWorkspace.tsx").default;
  const container = document.getElementById("root");
  const root = createRoot(container);
  await act(async () => {
    root.render(React.createElement(ReviewApp, {
      initialRunId: RUN_ID,
      onNewAnalysis: () => {},
    }));
    await new Promise((resolve) => setTimeout(resolve, 400));
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 200));
  });
  return { act, dom, container, cleanup: () => root.unmount() };
}

function stageButton(container, stage) {
  const buttons = Array.from(container.querySelectorAll('[role="tab"]'));
  const target = buttons.find((button) => button.textContent
    .replace(/\s+/g, " ").includes(stage));
  assert.ok(target, `stage tab "${stage}" is rendered`);
  return target;
}

async function clickStage(act, container, stage) {
  await act(async () => {
    stageButton(container, stage).dispatchEvent(
      new globalThis.MouseEvent("click", { bubbles: true }));
    await new Promise((resolve) => setTimeout(resolve, 250));
  });
}

async function clickButton(act, button) {
  await act(async () => {
    button.dispatchEvent(
      new globalThis.MouseEvent("click", { bubbles: true }));
    await new Promise((resolve) => setTimeout(resolve, 250));
  });
}

test("generate stage refetches readiness instead of showing a stale block", async () => {
  const { act, container, cleanup } = await renderWorkspace();
  try {
    assert.ok(container.textContent.includes(
      "0 of 8 findings still need your decision"),
      "findings stage reports all eight resolved");
    // The chef finished the review after the Generate panel mounted:
    // persisted corrections are now visible to fresh readiness reads.
    serveFreshReadiness = true;
    const callsBefore = readinessCalls;
    await clickStage(act, container, "2 · Generate recipe");
    const generate = container.querySelector(
      'section[aria-label="Generate recipe"]');
    assert.ok(generate && !generate.hidden, "generate stage is visible");
    assert.ok(readinessCalls > callsBefore,
      `readiness is refetched on entering the stage (calls: ${readinessCalls} vs ${callsBefore})`);
    assert.ok(generate.textContent.includes(
      "0 findings still need your attention"),
      "generate stage agrees that nothing is pending");
    assert.ok(generate.textContent.includes("Nothing blocks generation"),
      "a resolved conflict no longer blocks generation");
    const confirm = generate.querySelector('input[type="checkbox"]');
    assert.ok(confirm, "explicit confirmation control is rendered");
    await act(async () => {
      confirm.click();
      await new Promise((resolve) => setTimeout(resolve, 100));
    });
    const action = Array.from(generate.querySelectorAll("button")).find(
      (button) => button.textContent === "Generate Recipe");
    assert.ok(action, "a single Generate Recipe action is presented");
    assert.ok(!action.disabled,
      "completed review enables generation once confirmed");
  } finally {
    cleanup();
  }
});

test("stage navigation keeps both stages consistent", async () => {
  const { act, container, cleanup } = await renderWorkspace();
  try {
    serveFreshReadiness = true;
    await clickStage(act, container, "2 · Generate recipe");
    await clickStage(act, container, "1 · Review findings");
    assert.ok(container.textContent.includes(
      "0 of 8 findings still need your decision"),
      "returning to findings keeps the resolved state");
    await clickStage(act, container, "2 · Generate recipe");
    const generate = container.querySelector(
      'section[aria-label="Generate recipe"]');
    assert.ok(generate.textContent.includes("Nothing blocks generation"),
      "generate stage stays consistent across navigation");
    const back = Array.from(generate.querySelectorAll("button")).find(
      (button) => button.textContent.includes("Review findings"));
    assert.equal(back, undefined,
      "no back navigation is shown when nothing blocks");
  } finally {
    cleanup();
  }
});
