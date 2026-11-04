// Chef workflow stage tests: three-stage navigation in ReviewWorkspace.
//
// Renders the real <ReviewApp/> with a stubbed backend (no generation yet)
// and exercises actual clicks: stage tabs, finding navigation, correction
// persistence (captured POST), readiness from the real contract, the honest
// empty-recipe notice, and the technical-details expander. Static label
// unit tests live in observation.test.mjs / recipe.test.mjs.

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

const RUN_ID = "run_stage_test";

function frame(index, observation, measurement) {
  return {
    frame_index: index,
    timestamp_seconds: index * 10 + 2.5,
    image_available: true,
    image_name: `frame_${String(index).padStart(6, "0")}.jpg`,
    image_sha256: "ab".repeat(32),
    image_matches_record: true,
    observation,
    measurement,
    review: null,
  };
}

const BUNDLE = {
  run_id: RUN_ID,
  workflow: "test-pipeline",
  status: "complete",
  source_object_id: "upload_abcd1234abcd5678_P35_101_clip.mp4",
  source_sha256: null,
  frames: [
    frame(0, {
      model_fields: {
        scale_display_verbatim: "120 g",
        container_description: "bowl",
        scale_readable: true,
        scale_state: "stable",
        confidence: "high",
        ambiguity_notes: null,
      },
      terminal_state: "valid_prediction",
      request_id: "req-0",
      latency_seconds: 1.2,
      input_tokens: 100,
      output_tokens: 50,
      reasoning_output_tokens: null,
      response_status: "completed",
      parse_error: null,
    }, { value: 120, value_status: "ok", unit_observed: "g" }),
    frame(1, null, null),
  ],
  reviews: {},
  missing: [],
};

const QUEUE = {
  tasks: [
    {
      task_id: "task-1",
      field: "scale_value",
      state: "missing_relevant",
      status: "open",
      frame_refs: [0],
      start_seconds: 2.5,
      end_seconds: 2.5,
      correction: null,
    },
    {
      task_id: "task-2",
      field: "scale_value",
      state: "contradictory",
      status: "deferred",
      frame_refs: [1],
      start_seconds: 12.5,
      end_seconds: 12.5,
      correction: null,
    },
  ],
  evaluated: [],
  counts: { frames: 2, usable_readings: 1, abstentions: 0, accepted: 1 },
};

const READINESS = {
  ready: false,
  open_tasks: 2,
  live_generation: false,
  blocking_errors: ["task-2"],
  deferred: 1,
  live_available: false,
  live_blockers: ["Live generation is disabled in this fixture."],
  cost_note: null,
};

const posted = [];

function stubFetch() {
  posted.length = 0;
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
      return { ok: true, status: 200, json: async () => READINESS };
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
    await new Promise((resolve) => setTimeout(resolve, 150));
  });
}

function setInputValue(dom, input, value) {
  const setter = Object.getOwnPropertyDescriptor(
    dom.window.HTMLInputElement.prototype, "value").set;
  setter.call(input, value);
  input.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
}

test("completed analysis opens at findings with the video filename", async () => {
  const { container, cleanup } = await renderWorkspace();
  try {
    const headings = Array.from(container.querySelectorAll("h2")).map(
      (heading) => heading.textContent);
    assert.ok(headings.includes("P35_101_clip.mp4"),
      `header shows the video filename, saw: ${headings.join(" | ")}`);
    assert.ok(!headings.includes(RUN_ID),
      "run id is not a primary heading");
    assert.ok(stageButton(container, "1 · Review findings")
      .getAttribute("aria-selected") === "true",
      "findings stage is selected by default");
    assert.ok(container.textContent.includes("Finding 1 of 2"),
      "one active finding is shown at a time");
  } finally {
    cleanup();
  }
});

test("finding navigation moves one task at a time", async () => {
  const { act, container, cleanup } = await renderWorkspace();
  try {
    const buttons = Array.from(container.querySelectorAll("button"));
    const next = buttons.find((button) => button.textContent
      .includes("Next finding"));
    assert.ok(next, "next-finding control is rendered");
    await act(async () => {
      next.dispatchEvent(
        new globalThis.MouseEvent("click", { bubbles: true }));
      await new Promise((resolve) => setTimeout(resolve, 100));
    });
    assert.ok(container.textContent.includes("Finding 2 of 2"),
      "navigation advances to the second finding");
  } finally {
    cleanup();
  }
});

test("stage switch keeps the findings mounted instead of discarding them", async () => {
  const { act, container, cleanup } = await renderWorkspace();
  try {
    await clickStage(act, container, "2 · Generate recipe");
    const findings = container.querySelector(
      'section[aria-label="Review findings"]');
    assert.ok(findings, "findings section exists");
    assert.equal(findings.hidden, true, "findings section is hidden");
    assert.ok(findings.textContent.includes("Finding"),
      "findings content stays mounted while hidden");
  } finally {
    cleanup();
  }
});

test("generate stage reads the real readiness contract", async () => {
  const { act, container, cleanup } = await renderWorkspace();
  try {
    await clickStage(act, container, "2 · Generate recipe");
    const generate = container.querySelector(
      'section[aria-label="Generate recipe"]');
    assert.ok(generate && !generate.hidden, "generate stage is visible");
    assert.ok(generate.textContent.includes(
      "2 findings still need your attention"),
      "readiness summary uses open_tasks from the backend");
    assert.ok(generate.textContent.includes(
      "1 conflicting finding still blocks generation."),
      "blocked message names the count, not raw identifiers");
    assert.ok(!/simulation|practice version/i.test(generate.textContent),
      "simulation is not part of the chef-facing workflow");
    const action = Array.from(generate.querySelectorAll("button")).find(
      (button) => button.textContent === "Generate Recipe");
    assert.ok(action, "a single Generate Recipe action is presented");
    assert.ok(action.disabled,
      "blocked generation cannot fire a billable request");
    const blockedNote = Array.from(
      generate.querySelectorAll("p")).find(
      (paragraph) => paragraph.textContent.includes("conflicting finding"));
    assert.ok(blockedNote
      && !blockedNote.textContent.includes("task-2"),
      "raw task identifiers are not in the main workflow message");
    const details = generate.querySelector("details");
    assert.ok(details, "technical disclosure is rendered");
    assert.ok(details.textContent.includes("task-2"),
      "raw task identifier stays available in the disclosure");
    assert.ok(details.textContent.includes("Conflicting measurements"),
      "disclosure pairs the identifier with a human description");
    assert.ok(!generate.textContent.includes("cannot be estimated"),
      "cost explanation is removed from the workflow");
    const back = Array.from(generate.querySelectorAll("button")).find(
      (button) => button.textContent.includes("Review findings"));
    assert.ok(back, "blocked stage offers navigation back");
    await act(async () => {
      back.dispatchEvent(
        new globalThis.MouseEvent("click", { bubbles: true }));
      await new Promise((resolve) => setTimeout(resolve, 100));
    });
    const findings = container.querySelector(
      'section[aria-label="Review findings"]');
    assert.ok(findings && !findings.hidden,
      "back navigation returns to Review Findings");
  } finally {
    cleanup();
  }
});

test("recipe stage is honest when nothing was generated", async () => {
  const { act, container, cleanup } = await renderWorkspace();
  try {
    await clickStage(act, container, "3 · Review recipe");
    const recipe = container.querySelector(
      'section[aria-label="Review recipe"]');
    assert.ok(recipe && !recipe.hidden, "recipe stage is visible");
    assert.ok(recipe.textContent.includes("Your recipe isn't ready yet"),
      "empty state welcomes instead of showing an editor");
    assert.ok(recipe.textContent.includes("Go to Generate recipe"),
      "empty state navigates to generation");
    assert.ok(!recipe.textContent.includes("Chef decision"),
      "empty state hides decisions that require a recipe");
    assert.ok(stageButton(container, "3 · Review recipe")
      .textContent.includes("(none yet)"),
      "stage tab marks the missing generation");
  } finally {
    cleanup();
  }
});

test("correction persists through the observation-review API", async () => {
  const { act, dom, container, cleanup } = await renderWorkspace();
  try {
    const reviewer = container.querySelector(
      'input[placeholder="e.g. chef-maria"]');
    assert.ok(reviewer, "reviewer identity input is rendered");
    await act(async () => {
      setInputValue(dom, reviewer, "chef-test");
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    const value = container.querySelector(
      'input[aria-label="Correct reading"]');
    assert.ok(value, "correct-reading input is rendered with its label");
    await act(async () => {
      setInputValue(dom, value, "120");
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    const save = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent.includes("Provide correction"));
    assert.ok(save, "correction action is rendered");
    await act(async () => {
      save.dispatchEvent(
        new globalThis.MouseEvent("click", { bubbles: true }));
      await new Promise((resolve) => setTimeout(resolve, 200));
    });
    const call = posted.find((entry) => entry.url.endsWith(
      "/observation-review/corrections"));
    assert.ok(call, "correction POST was sent");
    assert.deepEqual(call.body, {
      task_id: "task-1",
      action: "correct",
      human_value: 120,
      reviewer: "chef-test",
    });
  } finally {
    cleanup();
  }
});

test("secondary sections stay collapsed and hide technical UI", async () => {
  const { container, cleanup } = await renderWorkspace();
  try {
    const sections = Array.from(
      container.querySelectorAll("details.card.evidence"));
    const labels = sections.map((details) => {
      const summary = details.querySelector("summary");
      return summary ? summary.textContent.replace(/\s+/g, " ").trim() : "";
    });
    assert.ok(labels.some((label) => label.startsWith("Analysis summary")),
      "analysis summary section is rendered");
    assert.ok(labels.some((label) => label.startsWith("Evidence explorer")),
      "evidence explorer section is rendered");
    for (const details of sections) {
      assert.equal(details.open, false,
        "secondary sections are closed on initial entry");
    }
    for (const label of ["Provenance", "Response ID", "Tokens",
      "Download CSV", "Raw model observation", "Technical identifiers",
      "Evidence & technical details", RUN_ID]) {
      assert.ok(!container.textContent.includes(label),
        `technical UI "${label}" is not chef-facing anymore`);
    }
  } finally {
    cleanup();
  }
});
