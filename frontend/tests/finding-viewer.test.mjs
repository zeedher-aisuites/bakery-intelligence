// Finding frame viewer regression tests.
//
// The main Review Findings workflow must show the selected finding's
// frame image directly, update it when the finding changes, support
// multi-frame findings without changing the finding, keep exactly one
// Previous/Next finding toolbar, and leave Evidence Explorer intact.

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

const RUN_ID = "run_finding_viewer_test";

function bundleFrame(index, value) {
  return {
    frame_index: index,
    timestamp_seconds: 5 + index * 10,
    image_available: true,
    image_name: `frame_${String(index).padStart(8, "0")}.jpg`,
    image_sha256: "ab".repeat(32),
    image_matches_record: true,
    observation: {
      model_fields: {
        visible_objects: null,
        scale_display_verbatim: `${value} g`,
        scale_readable: true,
        scale_state: "stable",
        tare_action_visible: null,
        addition_or_removal_visible: null,
        container_description: null,
        confidence: "high",
        ambiguity_notes: null,
      },
      terminal_state: "valid_prediction",
      request_id: `req-${index}`,
      latency_seconds: 1,
      input_tokens: 10,
      output_tokens: 5,
      reasoning_output_tokens: null,
      response_status: "completed",
      parse_error: null,
      frame_index: index,
      timestamp_seconds: 5 + index * 10,
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

const BUNDLE = {
  run_id: RUN_ID,
  workflow: "test-pipeline",
  status: "complete",
  source_object_id: "upload_abcd1234abcd5678_soup.mp4",
  source_sha256: null,
  frames: [bundleFrame(0, 120), bundleFrame(1, 95), bundleFrame(2, 310)],
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
      frame_refs: [0, 1],
      start_seconds: 5,
      end_seconds: 15,
      correction: null,
    },
    {
      task_id: "task-2",
      field: "scale_value",
      state: "uncertain",
      status: "open",
      frame_refs: [2],
      start_seconds: 25,
      end_seconds: 25,
      correction: null,
    },
  ],
  evaluated: [],
  counts: { frames: 3, usable_readings: 3, abstentions: 0, accepted: 1 },
};

const READINESS = {
  ready: false,
  open_tasks: 2,
  live_generation: false,
  blocking_errors: [],
  deferred: 0,
  live_available: false,
  live_blockers: [],
  cost_note: null,
};

const EVIDENCE = {
  frames: [0, 1, 2].map((index) => ({
    frame_index: index,
    timestamp_seconds: 5 + index * 10,
    image_available: true,
    image_name: `frame_${String(index).padStart(8, "0")}.jpg`,
    measurement: {
      value: [120, 95, 310][index],
      value_status: "ok",
    },
  })),
};

function stubFetch() {
  globalThis.fetch = async (url) => {
    const text = String(url);
    if (text.endsWith(`/api/analyses/${RUN_ID}/review`)) {
      return { ok: true, status: 200, json: async () => BUNDLE };
    }
    if (text.endsWith(`/api/analyses/${RUN_ID}/observation-review`)) {
      return { ok: true, status: 200, json: async () => QUEUE };
    }
    if (text.endsWith("/recipe/evidence")) {
      return { ok: true, status: 200, json: async () => EVIDENCE };
    }
    if (text.endsWith("/recipe/readiness")) {
      return { ok: true, status: 200, json: async () => READINESS };
    }
    if (text.endsWith(`/api/analyses/${RUN_ID}/recipe`)) {
      return {
        ok: false,
        status: 404,
        json: async () => ({ detail: "none" }),
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

function findingsSection(container) {
  return container.querySelector('section[aria-label="Review findings"]');
}

function findingViewer(section) {
  return section.querySelector('section[aria-label="Finding frame viewer"]');
}

function navGroup(section) {
  return Array.from(section.querySelectorAll(
    '[aria-label="Finding navigation"]'));
}

async function clickButton(act, button) {
  await act(async () => {
    button.dispatchEvent(
      new globalThis.MouseEvent("click", { bubbles: true }));
    await new Promise((resolve) => setTimeout(resolve, 200));
  });
}

test("selected finding shows its frame image and timestamp", async () => {
  const { container, cleanup } = await renderWorkspace();
  try {
    const section = findingsSection(container);
    assert.ok(section, "review findings section is rendered");
    const viewer = findingViewer(section);
    assert.ok(viewer, "finding frame viewer is rendered in Review Findings");
    const image = viewer.querySelector("figure img");
    assert.ok(image, "finding viewer shows an image");
    assert.ok(image.getAttribute("src").includes("frame_00000000.jpg"),
      "image follows the selected finding's first frame");
    assert.ok(viewer.textContent.includes("00:05"),
      "timestamp follows the selected finding");
    assert.ok(viewer.textContent.includes("Sample #0"),
      "sample identity follows the selected finding");
  } finally {
    cleanup();
  }
});

test("changing findings updates the displayed frame", async () => {
  const { act, container, cleanup } = await renderWorkspace();
  try {
    const section = findingsSection(container);
    const groups = navGroup(section);
    assert.equal(groups.length, 1,
      "exactly one finding navigation group is rendered");
    const next = Array.from(groups[0].querySelectorAll("button")).find(
      (button) => button.textContent.includes("Next finding"));
    assert.ok(next, "next-finding control is rendered");
    await clickButton(act, next);
    const viewer = findingViewer(findingsSection(container));
    const image = viewer.querySelector("figure img");
    assert.ok(image.getAttribute("src").includes("frame_00000002.jpg"),
      "image follows the newly selected finding");
    assert.ok(viewer.textContent.includes("00:25"),
      "timestamp follows the newly selected finding");
    assert.ok(container.textContent.includes("Finding 2 of 2"),
      "finding position follows navigation");
  } finally {
    cleanup();
  }
});

test("multi-frame findings expose every referenced frame", async () => {
  const { act, container, cleanup } = await renderWorkspace();
  try {
    const section = findingsSection(container);
    const viewer = findingViewer(section);
    assert.ok(viewer.textContent.includes("Frame 1 of 2"),
      "multi-frame position is shown");
    const nextFrame = viewer.querySelector(
      'button[aria-label="Next frame"]');
    assert.ok(nextFrame && !nextFrame.disabled,
      "frame step control is available for multi-frame findings");
    await clickButton(act, nextFrame);
    const updated = findingViewer(findingsSection(container));
    const image = updated.querySelector("figure img");
    assert.ok(image.getAttribute("src").includes("frame_00000001.jpg"),
      "frame navigation shows the second referenced frame");
    assert.ok(updated.textContent.includes("Sample #1"),
      "sample identity follows intra-finding navigation");
    assert.ok(findingsSection(container).textContent.includes(
      "Finding 1 of 2"),
      "intra-finding navigation does not change the finding");
  } finally {
    cleanup();
  }
});

test("frame navigation region stays rendered for single-frame findings", async () => {
  const { act, container, cleanup } = await renderWorkspace();
  try {
    const groups = navGroup(findingsSection(container));
    const next = Array.from(groups[0].querySelectorAll("button")).find(
      (button) => button.textContent.includes("Next finding"));
    await clickButton(act, next);
    const viewer = findingViewer(findingsSection(container));
    assert.ok(viewer, "viewer stays rendered on a single-frame finding");
    assert.ok(viewer.textContent.includes("Frame 1 of 1"),
      "frame counter stays rendered for a single frame");
    const slider = viewer.querySelector(
      'input[aria-label="Frame timeline position"]');
    assert.ok(slider, "timeline stays rendered for a single frame");
    assert.equal(slider.disabled, true,
      "timeline is disabled rather than removed for a single frame");
    const prevFrame = viewer.querySelector(
      'button[aria-label="Previous frame"]');
    assert.ok(prevFrame && prevFrame.disabled,
      "frame stepping is disabled rather than removed");
    assert.equal(findingsSection(container).querySelector(
      '[aria-label="Finding frames"]'), null,
      "no extra selector row shifts the layout");
  } finally {
    cleanup();
  }
});

test("finder cards keep three slots on the first and last findings", async () => {
  const { act, container, cleanup } = await renderWorkspace();
  try {
    const firstNav = findingsSection(container).querySelector(
      '.finder-nav[aria-label="Findings"]');
    assert.ok(firstNav, "findings navigator is rendered");
    assert.equal(firstNav.children.length, 3,
      "three card slots are reserved on the first finding");
    const groups = navGroup(findingsSection(container));
    const next = Array.from(groups[0].querySelectorAll("button")).find(
      (button) => button.textContent.includes("Next finding"));
    await clickButton(act, next);
    const lastNav = findingsSection(container).querySelector(
      '.finder-nav[aria-label="Findings"]');
    assert.equal(lastNav.children.length, 3,
      "three card slots are reserved on the last finding");
    assert.equal(navGroup(findingsSection(container)).length, 1,
      "exactly one finding toolbar survives finding switches");
  } finally {
    cleanup();
  }
});

test("finding viewer zoom updates and resets without losing the frame", async () => {
  const { act, container, cleanup } = await renderWorkspace();
  try {
    const viewer = findingViewer(findingsSection(container));
    const zoomIn = viewer.querySelector('button[aria-label="Zoom in"]');
    assert.ok(zoomIn, "zoom control is rendered");
    await clickButton(act, zoomIn);
    const zoomed = findingViewer(findingsSection(container));
    assert.ok(zoomed.textContent.includes("1.5×"), "zoom level updates");
    assert.ok(zoomed.querySelector("figure img").getAttribute("src")
      .includes("frame_00000000.jpg"), "frame is kept while zooming");
    const reset = Array.from(zoomed.querySelectorAll("button")).find(
      (button) => button.textContent === "Reset");
    await clickButton(act, reset);
    assert.ok(findingViewer(findingsSection(container)).textContent
      .includes("1.0×"), "reset restores the default zoom");
  } finally {
    cleanup();
  }
});

test("evidence explorer keeps its own frame viewer", async () => {
  const { act, container, cleanup } = await renderWorkspace();
  try {
    const explorer = Array.from(container.querySelectorAll(
      "details.card.evidence")).find(
      (details) => details.querySelector("summary").textContent
        .includes("Evidence explorer"));
    assert.ok(explorer, "evidence explorer is rendered");
    await act(async () => {
      explorer.open = true;
      await new Promise((resolve) => setTimeout(resolve, 150));
    });
    const viewer = explorer.querySelector(
      'section[aria-label="Frame viewer"]');
    assert.ok(viewer, "evidence explorer keeps its frame viewer");
    assert.ok(viewer.querySelector("figure img"),
      "explorer viewer shows an image");
  } finally {
    cleanup();
  }
});
