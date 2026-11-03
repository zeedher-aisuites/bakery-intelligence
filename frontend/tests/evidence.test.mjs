// Evidence explorer + finding consistency tests.
//
// Renders the real <ReviewApp/> with a stateful stubbed backend: the
// stub applies corrections to the queue so save, edit, undo, and
// reload consistency are exercised through the real components.
// Existing finding and recipe flows are covered by stages.test.mjs
// and recipe-review.test.mjs.

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

const RUN_ID = "run_evidence_test";

function validFrame(index, value, confidence) {
  return {
    frame_index: index,
    timestamp_seconds: index * 10 + 5,
    image_available: true,
    image_name: `frame_${String(index).padStart(8, "0")}.jpg`,
    image_sha256: "cd".repeat(32),
    image_matches_record: true,
    observation: {
      model_fields: {
        visible_objects: null,
        scale_display_verbatim: `${value} g`,
        scale_readable: true,
        scale_state: "stable",
        tare_action_visible: null,
        addition_or_removal_visible: null,
        container_description: index === 0 ? "bowl" : null,
        confidence,
        ambiguity_notes: null,
      },
      terminal_state: "valid_prediction",
      request_id: `req-${index}`,
      latency_seconds: 1.1,
      input_tokens: 100,
      output_tokens: 50,
      reasoning_output_tokens: null,
      response_status: "completed",
      parse_error: null,
      frame_index: index,
      timestamp_seconds: index * 10 + 5,
    },
    measurement: {
      measurement_id: null,
      value,
      value_source: "pipeline",
      value_status: "ok",
      unit_observed: "g",
    },
    review: null,
  };
}

function emptyFrame(index, terminal) {
  return {
    frame_index: index,
    timestamp_seconds: index * 10 + 5,
    image_available: index === 3,
    image_name: index === 3 ? `frame_${String(index).padStart(8, "0")}.jpg` : null,
    image_sha256: null,
    image_matches_record: null,
    observation: terminal === null ? null : {
      model_fields: {
        visible_objects: null,
        scale_display_verbatim: null,
        scale_readable: null,
        scale_state: null,
        tare_action_visible: null,
        addition_or_removal_visible: null,
        container_description: null,
        confidence: "low",
        ambiguity_notes: null,
      },
      terminal_state: terminal,
      request_id: `req-${index}`,
      latency_seconds: 0.9,
      input_tokens: 90,
      output_tokens: 20,
      reasoning_output_tokens: null,
      response_status: "completed",
      parse_error: null,
      frame_index: index,
      timestamp_seconds: index * 10 + 5,
    },
    measurement: null,
    review: null,
  };
}

const BUNDLE = {
  run_id: RUN_ID,
  workflow: "test-pipeline",
  status: "complete",
  source_object_id: "upload_0011223344556677_soup.mp4",
  source_sha256: null,
  frames: [
    validFrame(0, 120, "high"),
    validFrame(1, 95, "medium"),
    emptyFrame(2, null),
    emptyFrame(3, "uncertain"),
    emptyFrame(4, null),
    emptyFrame(5, null),
    emptyFrame(6, null),
  ],
  reviews: {},
  missing: [],
};

const BASE_TASKS = [
  {
    task_id: "task-1",
    field: "scale_value",
    state: "missing_relevant",
    status: "open",
    frame_refs: [0],
    start_seconds: 5,
    end_seconds: 5,
    correction: null,
  },
  {
    task_id: "task-2",
    field: "scale_value",
    state: "uncertain",
    status: "deferred",
    frame_refs: [3],
    start_seconds: 35,
    end_seconds: 35,
    correction: null,
  },
];

const READINESS = {
  ready: false,
  open_tasks: 2,
  live_generation: false,
  blocking_errors: [],
  deferred: 1,
  live_available: false,
  live_blockers: [],
  cost_note: null,
};

const EVIDENCE = {
  frames: [
    {
      frame_index: 0, timestamp_seconds: 5, image_available: true,
      image_name: "frame_00000000.jpg",
      measurement: { value: 120, value_status: "ok" },
    },
    {
      frame_index: 1, timestamp_seconds: 15, image_available: true,
      image_name: "frame_00000001.jpg",
      measurement: { value: 95, value_status: "ok" },
    },
    {
      frame_index: 2, timestamp_seconds: 25, image_available: false,
      image_name: null, measurement: null,
    },
    {
      frame_index: 3, timestamp_seconds: 35, image_available: true,
      image_name: "frame_00000003.jpg", measurement: null,
    },
    {
      frame_index: 4, timestamp_seconds: 45, image_available: false,
      image_name: null, measurement: null,
    },
    {
      frame_index: 5, timestamp_seconds: 55, image_available: false,
      image_name: null, measurement: null,
    },
    {
      frame_index: 6, timestamp_seconds: 65, image_available: false,
      image_name: null, measurement: null,
    },
  ],
};

const posted = [];
let corrections = {};

function taskWithCorrection(task) {
  const stored = corrections[task.task_id];
  if (!stored) return { ...task };
  return {
    ...task,
    status: stored.action === "review_later" ? "deferred" : "resolved",
    correction: stored,
  };
}

function stubFetch() {
  posted.length = 0;
  corrections = {};
  globalThis.fetch = async (url, options) => {
    const text = String(url);
    const method = (options?.method ?? "GET").toUpperCase();
    if (method === "POST") {
      const body = options?.body ? JSON.parse(String(options.body)) : null;
      posted.push({ url: text, body });
      if (text.endsWith("/observation-review/corrections")) {
        if ((body.action === "correct" || body.action === "supply")
          && (body.human_value === null
            || !Number.isFinite(Number(body.human_value)))) {
          return {
            ok: false, status: 422,
            json: async () => ({ detail: "finite number required" }),
          };
        }
        const previous = corrections[body.task_id];
        const stored = {
          action: body.action,
          human_value: body.human_value,
          reviewer: body.reviewer,
          revision: (previous?.revision ?? 0) + 1,
        };
        corrections[body.task_id] = stored;
        return { ok: true, status: 200, json: async () => stored };
      }
      if (text.endsWith("/observation-review/corrections/undo")) {
        const previous = corrections[body.task_id];
        if (!previous) {
          return {
            ok: false, status: 422,
            json: async () => ({ detail: "nothing to undo" }),
          };
        }
        delete corrections[body.task_id];
        return {
          ok: true, status: 200,
          json: async () => ({
            undone: previous, correction: null, status: "open",
          }),
        };
      }
      return { ok: true, status: 200, json: async () => ({}) };
    }
    if (text.endsWith(`/api/analyses/${RUN_ID}/review`)) {
      return { ok: true, status: 200, json: async () => BUNDLE };
    }
    if (text.endsWith(`/api/analyses/${RUN_ID}/observation-review`)) {
      const tasks = BASE_TASKS.map(taskWithCorrection);
      return {
        ok: true, status: 200,
        json: async () => ({
          tasks,
          evaluated: [],
          counts: { frames: 7, usable_readings: 2, abstentions: 0, accepted: 1 },
        }),
      };
    }
    if (text.endsWith("/recipe/evidence")) {
      return { ok: true, status: 200, json: async () => EVIDENCE };
    }
    if (text.endsWith("/recipe/readiness")) {
      return { ok: true, status: 200, json: async () => READINESS };
    }
    if (text.endsWith(`/api/analyses/${RUN_ID}/recipe`)) {
      return {
        ok: false, status: 404, json: async () => ({ detail: "none" }),
      };
    }
    if (text.endsWith(`/api/analyses/${RUN_ID}`)) {
      return {
        ok: true, status: 200,
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
    "HTMLSelectElement", "KeyboardEvent", "getComputedStyle",
    "localStorage"]) {
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
  globalThis.IS_REACT_ENVIRONMENT = true;
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

function setInputValue(dom, input, value) {
  const setter = Object.getOwnPropertyDescriptor(
    dom.window.HTMLInputElement.prototype, "value").set;
  setter.call(input, value);
  input.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
}

async function clickButton(act, button) {
  await act(async () => {
    button.dispatchEvent(
      new globalThis.MouseEvent("click", { bubbles: true }));
    await new Promise((resolve) => setTimeout(resolve, 200));
  });
}

function reviewAppButtons(container) {
  return Array.from(container.querySelectorAll("button"));
}

function findingsSection(container) {
  return container.querySelector('section[aria-label="Review findings"]');
}

async function setReviewer(act, dom, container, name) {
  const input = container.querySelector(
    'input[aria-label="Reviewer identity"]');
  assert.ok(input, "reviewer identity control is rendered");
  await act(async () => {
    setInputValue(dom, input, name);
    await new Promise((resolve) => setTimeout(resolve, 50));
  });
}

async function setCorrectionValue(act, dom, container, value) {
  const input = container.querySelector(
    'input[aria-label="Correct reading"]');
  assert.ok(input, "correct-reading input is rendered");
  await act(async () => {
    setInputValue(dom, input, value);
    await new Promise((resolve) => setTimeout(resolve, 50));
  });
  return input;
}

function provideButton(container) {
  return reviewAppButtons(container).find(
    (button) => button.textContent === "Provide correction");
}

test("both secondary sections are collapsed on initial entry", async () => {
  const { act, container, cleanup } = await renderWorkspace();
  try {
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 100));
    });
    for (const title of ["Analysis summary", "Evidence explorer"]) {
      const target = Array.from(
        container.querySelectorAll("details.card.evidence")).find(
        (details) => details.querySelector("summary").textContent
          .replace(/\s+/g, " ").startsWith(title));
      assert.ok(target, `"${title}" section is rendered`);
      assert.equal(target.open, false,
        `"${title}" is closed on initial entry`);
    }
  } finally {
    cleanup();
  }
});

test("finding navigation updates identity, image, and observations", async () => {
  const { act, container, cleanup } = await renderWorkspace();
  try {
    const explorer = Array.from(
      container.querySelectorAll("details.card.evidence")).find(
      (details) => details.querySelector("summary").textContent
        .includes("Evidence explorer"));
    await act(async () => {
      explorer.open = true;
      await new Promise((resolve) => setTimeout(resolve, 100));
    });
    const next = explorer.querySelector(
      'button[aria-label="Next frame"]');
    assert.ok(next, "next-frame control is rendered");
    assert.ok(explorer.textContent.includes("00:05"),
      "first frame timestamp is shown as mm:ss");
    await clickButton(act, next);
    const caption = explorer.querySelector("figcaption").textContent
      .replace(/\s+/g, " ");
    assert.ok(caption.includes("00:15"),
      `timestamp follows the frame, saw: ${caption}`);
    assert.ok(caption.includes("Sample #1"),
      `sample identity follows the frame, saw: ${caption}`);
    const image = explorer.querySelector("figure img");
    assert.ok(image && image.getAttribute("src").includes(
      "frame_00000001.jpg"), "image follows the frame");
    assert.ok(explorer.textContent.includes("95"),
      "observation strip follows the frame");
  } finally {
    cleanup();
  }
});

test("filters apply only on Apply and survive Cancel", async () => {
  const { act, dom, container, cleanup } = await renderWorkspace();
  try {
    const explorer = Array.from(
      container.querySelectorAll("details.card.evidence")).find(
      (details) => details.querySelector("summary").textContent
        .includes("Evidence explorer"));
    await act(async () => {
      explorer.open = true;
      await new Promise((resolve) => setTimeout(resolve, 100));
    });
    const filtersButton = Array.from(
      explorer.querySelectorAll("button")).find((button) => button
      .textContent.startsWith("Filters"));
    assert.ok(filtersButton, "Filters button is rendered");
    await clickButton(act, filtersButton);
    const dialog = explorer.querySelector('[role="dialog"]');
    assert.ok(dialog, "filter dialog opens");
    const modelSelect = dialog.querySelectorAll("select")[0];
    await act(async () => {
      modelSelect.value = "reading";
      modelSelect.dispatchEvent(
        new dom.window.Event("change", { bubbles: true }));
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    assert.ok(explorer.textContent.includes("7 of 7 frames"),
      "changing the draft does not filter yet");
    const apply = Array.from(dialog.querySelectorAll("button")).find(
      (button) => button.textContent === "Apply");
    await clickButton(act, apply);
    assert.ok(!explorer.querySelector('[role="dialog"]'),
      "dialog closes on Apply");
    assert.ok(explorer.textContent.includes("2 of 7 frames"),
      "applied filter narrows to frames with readings");
    assert.ok(filtersButton.textContent.includes("1 active"),
      "active filter count stays visible");
  } finally {
    cleanup();
  }
});

test("empty correction cannot be saved", async () => {
  const { act, dom, container, cleanup } = await renderWorkspace();
  try {
    const section = findingsSection(container);
    await setReviewer(act, dom, container, "chef-test");
    const save = provideButton(container);
    assert.ok(save, "Provide correction action is rendered");
    assert.ok(save.disabled, "empty correction cannot be submitted");
    await setCorrectionValue(act, dom, container, "abc");
    assert.ok(provideButton(container).disabled,
      "non-finite input cannot be submitted");
    await setCorrectionValue(act, dom, container, "121");
    assert.ok(!provideButton(container).disabled,
      "finite input enables submission");
    assert.equal(posted.filter((entry) => entry.url.endsWith(
      "/observation-review/corrections")).length, 0,
      "nothing is sent before an explicit valid action");
    assert.ok(section.textContent.includes("Needs decision"),
      "unresolved finding keeps its pending status");
  } finally {
    cleanup();
  }
});

test("correction displays exact original and new value", async () => {
  const { act, dom, container, cleanup } = await renderWorkspace();
  try {
    await setReviewer(act, dom, container, "chef-test");
    await setCorrectionValue(act, dom, container, "121");
    await clickButton(act, provideButton(container));
    const section = findingsSection(container);
    assert.ok(section.textContent.includes(
      "Corrected from 120 to 121 (unit unknown)."),
      "result names the stored original and the new value");
    assert.ok(section.textContent.includes("Corrected to 121"),
      "status stays distinguishable from confirmation");
    assert.ok(!section.textContent.includes("Reviewed as correct"),
      "misleading generic status is gone");
    const call = posted.find((entry) => entry.url.endsWith(
      "/observation-review/corrections"));
    assert.ok(call, "correction POST was sent");
    assert.deepEqual(call.body, {
      task_id: "task-1", action: "correct", human_value: 121,
      reviewer: "chef-test",
    });
  } finally {
    cleanup();
  }
});

test("confirmation stays distinct from correction", async () => {
  const { act, dom, container, cleanup } = await renderWorkspace();
  try {
    await setReviewer(act, dom, container, "chef-test");
    const confirm = reviewAppButtons(container).find(
      (button) => button.textContent === "Confirm reading");
    assert.ok(confirm, "Confirm reading is offered for a single value");
    await clickButton(act, confirm);
    const section = findingsSection(container);
    assert.ok(section.textContent.includes(
      "AI reading confirmed: 120."),
      "confirmation names the confirmed model value");
    const call = posted.find((entry) => entry.url.endsWith(
      "/observation-review/corrections"));
    assert.deepEqual(call.body, {
      task_id: "task-1", action: "confirm", human_value: null,
      reviewer: "chef-test",
    });
  } finally {
    cleanup();
  }
});

test("previously reviewed findings can be edited", async () => {
  const { act, dom, container, cleanup } = await renderWorkspace();
  try {
    await setReviewer(act, dom, container, "chef-test");
    await setCorrectionValue(act, dom, container, "121");
    await clickButton(act, provideButton(container));
    const value = container.querySelector(
      'input[aria-label="Correct reading"]');
    assert.equal(value.value, "121",
      "selecting a reviewed finding pre-fills its value");
    await setCorrectionValue(act, dom, container, "122");
    await clickButton(act, provideButton(container));
    const section = findingsSection(container);
    assert.ok(section.textContent.includes(
      "Corrected from 120 to 122 (unit unknown)."),
      "edit replaces the effective value");
  } finally {
    cleanup();
  }
});

test("undo restores the previous decision without losing audit", async () => {
  const { act, dom, container, cleanup } = await renderWorkspace();
  try {
    await setReviewer(act, dom, container, "chef-test");
    await setCorrectionValue(act, dom, container, "121");
    await clickButton(act, provideButton(container));
    const undo = reviewAppButtons(container).find(
      (button) => button.textContent === "Undo last change");
    assert.ok(undo, "undo is offered for a reviewed finding");
    await clickButton(act, undo);
    const section = findingsSection(container);
    assert.ok(section.textContent.includes("Needs decision"),
      "undo reopens the finding");
    const undoCall = posted.find((entry) => entry.url.endsWith(
      "/observation-review/corrections/undo"));
    assert.ok(undoCall, "undo is persisted through its own endpoint");
    assert.deepEqual(undoCall.body, {
      task_id: "task-1", reviewer: "chef-test",
    });
    assert.ok(!posted.some((entry) => entry.url.endsWith(
      "/observation-review/classifications")),
      "undo never touches the classification family");
  } finally {
    cleanup();
  }
});

test("explorer shows the effective finding correction", async () => {
  const { act, dom, container, cleanup } = await renderWorkspace();
  try {
    await setReviewer(act, dom, container, "chef-test");
    await setCorrectionValue(act, dom, container, "121");
    await clickButton(act, provideButton(container));
    const explorer = Array.from(
      container.querySelectorAll("details.card.evidence")).find(
      (details) => details.querySelector("summary").textContent
        .includes("Evidence explorer"));
    await act(async () => {
      explorer.open = true;
      await new Promise((resolve) => setTimeout(resolve, 150));
    });
    const strip = explorer.querySelector(
      'section[aria-label="Current frame observation"]');
    assert.ok(strip.textContent.includes(
      "Corrected from 120 to 121 (unit unknown)."),
      "explorer shows the same effective decision");
    assert.ok(strip.textContent.includes("finding correction"),
      "explorer names the decision scope");
    assert.ok(!strip.textContent.includes("Awaiting human review"),
      "no stale awaiting message when a decision exists");
  } finally {
    cleanup();
  }
});

test("summary and header stay consistent after persistence", async () => {
  const { act, dom, container, cleanup } = await renderWorkspace();
  try {
    await setReviewer(act, dom, container, "chef-test");
    await setCorrectionValue(act, dom, container, "121");
    await clickButton(act, provideButton(container));
    assert.ok(container.textContent.includes(
      "0 of 2 findings need your attention"),
      "header reflects the persisted resolution");
    const summary = Array.from(
      container.querySelectorAll("details.card.evidence")).find(
      (details) => details.querySelector("summary").textContent
        .includes("Analysis summary"));
    assert.ok(summary.textContent.includes(
      "1 of 2 findings resolved"),
      "summary counts the same persisted decision");
    assert.ok(summary.textContent.includes("1 set aside for later"),
      "deferred findings stay visible separately");
  } finally {
    cleanup();
  }
});

test("stage navigation still works with the new sections", async () => {
  const { act, container, cleanup } = await renderWorkspace();
  try {
    const tabs = Array.from(container.querySelectorAll('[role="tab"]'));
    const generate = tabs.find((tab) => tab.textContent.includes(
      "Generate recipe"));
    await clickButton(act, generate);
    const visible = container.querySelector(
      'section[aria-label="Generate recipe"]');
    assert.ok(visible && !visible.hidden, "generate stage still opens");
  } finally {
    cleanup();
  }
});

test("backend record contracts remain available", async () => {
  const api = require("../src/api.ts");
  const review = require("../src/review.ts");
  for (const name of ["saveReview", "saveObservationCorrection",
    "saveFrameClassification", "fetchReviewBundle", "fetchRecipe",
    "fetchObservationQueue", "undoObservationCorrection"]) {
    assert.equal(typeof api[name], "function",
      `api keeps ${name}`);
  }
  for (const name of ["exportReviewCsv", "exportReviewJson",
    "buildRecipeRows", "effectiveReading", "orderFrames"]) {
    assert.equal(typeof review[name], "function",
      `review keeps ${name}`);
  }
});
