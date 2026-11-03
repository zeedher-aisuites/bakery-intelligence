// My recipes (processing queue) component tests (plain Node + jsdom).
//
// Renders the real <ProcessingQueue/> with a stubbed backend and
// exercises clicks: friendly labels, hidden internals, completed-job
// navigation, waiting/failed states, retry safety, empty state.

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
      loader, jsx: "automatic", format: "cjs",
    });
    module._compile(code, filename);
  };
}
require.extensions[".css"] = function (module) {
  module.exports = {};
};

const { JSDOM } = require("jsdom");

function setupDom() {
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
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
}

const queuedJob = {
  run_id: "run_aaaabbbbcccc", workflow: "video_pipeline_offline",
  status: "queued", source_object_id: "upload_ab12cd34ef56ab12_soup.mp4",
  source_sha256: null, artifact_ids: [], artifact_sha256: {},
  error: null, error_category: null, claimed_by: null, attempts: 0,
  created_at: "2026-11-02T10:18:22.163Z", updated_at: null,
};

async function renderQueue({
  jobs, recipe = "absent", onOpenReview, onNewAnalysis,
}) {
  const calls = [];
  globalThis.fetch = async (url, options = {}) => {
    const text = String(url);
    if (options.method === "POST" && text.includes("/retry")) {
      calls.push(text);
      const id = text.split("/api/jobs/")[1].split("/retry")[0];
      return {
        ok: true, status: 200,
        json: async () => ({ ...queuedJob, run_id: id, status: "queued" }),
      };
    }
    if (/\/recipe$/.test(text)) {
      if (recipe === "ready") {
        return {
          ok: true, status: 200,
          json: async () => ({ generation: { mode: "simulation" } }),
        };
      }
      return { ok: false, status: 404, json: async () => ({}) };
    }
    return { ok: true, status: 200, json: async () => jobs };
  };
  setupDom();
  const React = require("react");
  const act = React.act ?? require("react-dom/test-utils").act;
  const { createRoot } = require("react-dom/client");
  const ProcessingQueue = require("../src/ProcessingQueue.tsx").default;
  const container = document.getElementById("root");
  const root = createRoot(container);
  const opened = [];
  const created = [];
  await act(async () => {
    root.render(React.createElement(ProcessingQueue, {
      highlightRunId: null,
      onOpenReview: (runId) => {
        opened.push(runId);
        if (onOpenReview) onOpenReview(runId);
      },
      onNewAnalysis: () => {
        created.push(true);
        if (onNewAnalysis) onNewAnalysis();
      },
    }));
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 100));
  });
  return {
    act, container, calls, opened, created,
    cleanup: () => root.unmount(),
  };
}

function buttonByLabel(container, label) {
  const buttons = Array.from(container.querySelectorAll("button"));
  return buttons.find(
    (button) => button.textContent.trim() === label) ?? null;
}

test("cards show filenames and friendly status, never internals", async () => {
  const job = {
    ...queuedJob, status: "running", claimed_by: "worker-7", attempts: 4,
  };
  const { container, cleanup } = await renderQueue({ jobs: [job] });
  try {
    const text = container.textContent;
    assert.match(text, /soup\.mp4/);
    assert.match(text, /Processing/);
    assert.doesNotMatch(text, /run_aaaabbbbcccc/);
    assert.doesNotMatch(text, /video_pipeline_offline/);
    assert.doesNotMatch(text, /worker-7/);
    assert.doesNotMatch(text, /Attempts/);
  } finally {
    cleanup();
  }
});

test("waiting jobs explain the state without mentioning workers", async () => {
  const { container, cleanup } = await renderQueue({ jobs: [queuedJob] });
  try {
    assert.match(container.textContent, /Waiting to begin\./);
    assert.doesNotMatch(container.textContent, /worker/i);
  } finally {
    cleanup();
  }
});

test("completed analysis without a recipe promises only findings", async () => {
  const job = { ...queuedJob, status: "completed" };
  const { act, container, opened, cleanup } = await renderQueue({
    jobs: [job],
  });
  try {
    assert.match(container.textContent, /Analysis ready/);
    assert.match(container.textContent, /review findings/i);
    assert.doesNotMatch(container.textContent, /ready to review/i);
    const target = buttonByLabel(container, "Review findings");
    assert.ok(target, "Review findings action is rendered");
    await act(async () => {
      target.dispatchEvent(
        new globalThis.MouseEvent("click", { bubbles: true }));
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    assert.deepEqual(opened, [job.run_id]);
  } finally {
    cleanup();
  }
});

test("completed jobs with a recipe offer the recipe draft", async () => {
  const job = { ...queuedJob, status: "completed" };
  const { act, container, opened, cleanup } = await renderQueue({
    jobs: [job], recipe: "ready",
  });
  try {
    assert.match(container.textContent, /Recipe ready/);
    assert.match(container.textContent, /review draft/i);
    const target = buttonByLabel(container, "Review recipe");
    assert.ok(target, "Review recipe action is rendered");
    await act(async () => {
      target.dispatchEvent(
        new globalThis.MouseEvent("click", { bubbles: true }));
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    assert.deepEqual(opened, [job.run_id]);
  } finally {
    cleanup();
  }
});

test("failed free jobs can retry safely without repeating billing", async () => {
  const job = {
    ...queuedJob, status: "failed", error: "boom",
    error_category: "processing", attempts: 1,
  };
  const { act, container, calls, cleanup } = await renderQueue({
    jobs: [job],
  });
  try {
    assert.match(container.textContent, /Failed/);
    const target = buttonByLabel(container, "Retry");
    assert.ok(target, "Retry action is rendered for safe failures");
    await act(async () => {
      target.dispatchEvent(
        new globalThis.MouseEvent("click", { bubbles: true }));
      await new Promise((resolve) => setTimeout(resolve, 100));
    });
    assert.equal(calls.length, 1);
    assert.match(calls[0], new RegExp(`${job.run_id}/retry`));
  } finally {
    cleanup();
  }
});

test("paid and invalid failures offer no Retry button", async () => {
  const metaFailed = {
    ...queuedJob, status: "failed", workflow: "video_pipeline_meta",
    error: "boom", error_category: "processing", attempts: 1,
  };
  const invalidFailed = {
    ...queuedJob, status: "failed", error: "too short",
    error_category: "validation", attempts: 1,
  };
  const first = await renderQueue({ jobs: [metaFailed] });
  try {
    assert.equal(buttonByLabel(first.container, "Retry"), null);
    assert.match(first.container.textContent, /never retried/i);
  } finally {
    first.cleanup();
  }
  const second = await renderQueue({ jobs: [invalidFailed] });
  try {
    assert.equal(buttonByLabel(second.container, "Retry"), null);
    assert.match(second.container.textContent, /different MP4/i);
  } finally {
    second.cleanup();
  }
});

test("empty queue navigates to New Analysis on click", async () => {
  const { act, container, created, cleanup } = await renderQueue({
    jobs: [],
  });
  try {
    assert.match(container.textContent, /No recipes yet/);
    const target = buttonByLabel(container, "Start a new analysis");
    assert.ok(target, "empty-state action is rendered");
    await act(async () => {
      target.dispatchEvent(
        new globalThis.MouseEvent("click", { bubbles: true }));
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    assert.equal(created.length, 1);
  } finally {
    cleanup();
  }
});
