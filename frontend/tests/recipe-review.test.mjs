// Recipe review stage tests: empty versus populated states, editor,
// decisions, regeneration authorization, and version history.
//
// Renders the real <RecipeReview/> with a stubbed backend. Finding and
// generation flows are covered by stages.test.mjs and evidence.test.mjs.

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

const RUN_ID = "run_recipe_test";

const POPULATED = {
  run_id: RUN_ID,
  draft_id: "draft-1",
  version: 2,
  status: "draft",
  simulation: false,
  simulation_note: null,
  ingredients: [{
    ingredient_id: "flour",
    quantity: { value: 500, unit: "g", derivation: "difference_between_readings" },
    provenance: "ai-extracted",
  }],
  steps: [{
    step_id: "step_1",
    description: "Mix the flour",
    start_seconds: 5,
    end_seconds: 15,
    frame_refs: [{ frame_index: 0 }],
    ingredient_refs: ["flour"],
    provenance: "ai-extracted",
    review_status: "pending",
    uncertainty: "",
    not_observed: false,
  }],
  claims: [],
  verification: { verdict: "pass", issues: [] },
  decisions: [],
  approved_snapshots: [],
  history: [
    {
      version: 1, action: "generated", actor: "system",
      at: "2026-03-04T10:30:00Z",
    },
    {
      version: 2, action: "corrected", actor: "chef-ana",
      at: "2026-03-04T11:00:00Z", summary: "Fixed quantities",
    },
  ],
  updated_at: "2026-03-04T11:00:00Z",
  generation: {
    mode: "live",
    simulated: false,
    rounds_used: 1,
    providers: [{ role: "proposer", provider: "meta", model: "muse-spark" }],
    costs: [{
      provider: "meta", model: "muse-spark", operation: "propose",
      estimated_cost_usd: 0.42,
    }],
  },
};

const EVIDENCE = {
  frames: [{
    frame_index: 0,
    timestamp_seconds: 5,
    image_available: false,
    image_name: null,
    measurement: { value: 500, value_status: "ok" },
  }],
};

const READINESS = {
  ready: true,
  open_tasks: 0,
  live_generation: true,
  blocking_errors: [],
  deferred: 0,
  live_available: true,
  live_blockers: [],
  cost_note: null,
};

const posted = [];
let navigated = false;

function stubFetch(recipe) {
  posted.length = 0;
  navigated = false;
  globalThis.fetch = async (url, options) => {
    const text = String(url);
    const method = (options?.method ?? "GET").toUpperCase();
    if (method === "POST") {
      const body = options?.body ? JSON.parse(String(options.body)) : null;
      posted.push({ url: text, body });
      if (text.endsWith("/recipe/corrections")) {
        return {
          ok: true, status: 200,
          json: async () => ({ ...POPULATED, version: 3 }),
        };
      }
      if (text.endsWith("/recipe/decisions")) {
        return {
          ok: true, status: 200,
          json: async () => ({ ...POPULATED, status: "approved" }),
        };
      }
      if (text.endsWith("/recipe/generate")) {
        return { ok: true, status: 200, json: async () => POPULATED };
      }
      return { ok: true, status: 200, json: async () => ({}) };
    }
    if (text.endsWith("/recipe/evidence")) {
      return { ok: true, status: 200, json: async () => EVIDENCE };
    }
    if (text.endsWith("/recipe/readiness")) {
      return { ok: true, status: 200, json: async () => READINESS };
    }
    if (text.endsWith(`/api/analyses/${RUN_ID}/recipe`)) {
      if (recipe === null) {
        return {
          ok: false, status: 404, json: async () => ({ detail: "none" }),
        };
      }
      return { ok: true, status: 200, json: async () => recipe };
    }
    return { ok: true, status: 200, json: async () => ({}) };
  };
}

async function renderRecipe(recipe) {
  stubFetch(recipe);
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
  const RecipeReview = require("../src/RecipeReview.tsx").default;
  const container = document.getElementById("root");
  const root = createRoot(container);
  await act(async () => {
    root.render(React.createElement(RecipeReview, {
      runId: RUN_ID,
      sourceName: "soup.mp4",
      onNavigateToGenerate: () => { navigated = true; },
    }));
    await new Promise((resolve) => setTimeout(resolve, 400));
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 150));
  });
  return { act, dom, container, cleanup: () => root.unmount() };
}

function setInputValue(dom, input, value) {
  const setter = Object.getOwnPropertyDescriptor(
    dom.window.HTMLInputElement.prototype, "value").set;
  setter.call(input, value);
  input.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
}

function setSelectValue(dom, select, value) {
  const setter = Object.getOwnPropertyDescriptor(
    dom.window.HTMLSelectElement.prototype, "value").set;
  setter.call(select, value);
  select.dispatchEvent(new dom.window.Event("change", { bubbles: true }));
}

const WITH_USAGES = {
  ...POPULATED,
  ingredients: [
    {
      ingredient_id: "flour",
      provenance: "ai-extracted",
    },
    { ingredient_id: "sugar", provenance: "chef-supplied" },
  ],
  steps: [{
    ...POPULATED.steps[0],
    usages: [{
      ingredient_id: "flour",
      quantity: { value: 200, unit: "g", derivation: "chef_supplied" },
      provenance: "chef-supplied",
    }],
  }],
};

async function click(act, button) {
  await act(async () => {
    button.dispatchEvent(
      new globalThis.MouseEvent("click", { bubbles: true }));
    await new Promise((resolve) => setTimeout(resolve, 200));
  });
}

test("populated recipe renders as a chef-friendly editor", async () => {
  const { container, cleanup } = await renderRecipe(POPULATED);
  try {
    assert.ok(container.textContent.includes("Version 2 · In review"),
      "header shows readable version and status");
    for (const raw of ["Draft v2", "verification", "technical frame",
      "step_1", "(frame 0)"]) {
      assert.ok(!container.textContent.includes(raw),
        `technical residue "${raw}" is not in the main layout`);
    }
    const ingredient = container.querySelector(
      'input[aria-label="Ingredient"]');
    assert.ok(ingredient && ingredient.value === "flour",
      "ingredient row is editable with its value");
    const addButtons = Array.from(container.querySelectorAll("button"))
      .map((button) => button.textContent);
    assert.ok(addButtons.includes("+ Add ingredient"),
      "add-ingredient action is offered");
    assert.ok(addButtons.includes("+ Add step"),
      "add-step action is offered");
    assert.ok(container.textContent.includes("Version 1 — generated"),
      "history names the event");
    assert.ok(container.textContent.includes("2026"),
      "history shows a human date");
    assert.ok(container.textContent.includes("Fixed quantities"),
      "history shows the change summary when present");
  } finally {
    cleanup();
  }
});

test("approval requires a reviewer and a non-empty valid recipe", async () => {
  const { act, dom, container, cleanup } = await renderRecipe(POPULATED);
  try {
    const approve = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent === "Approve recipe");
    assert.ok(approve, "Approve recipe is offered");
    assert.ok(approve.disabled, "approval is blocked without a reviewer");
    assert.ok(container.textContent.includes(
      "Enter your name above to approve or request changes."),
      "missing reviewer is explained");
    const reviewer = container.querySelector(
      'input[placeholder="e.g. chef-maria"]');
    await act(async () => {
      setInputValue(dom, reviewer, "chef-ana");
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    assert.ok(!approve.disabled, "reviewer unlocks approval");
    await click(act, approve);
    const call = posted.find((entry) => entry.url.endsWith(
      "/recipe/decisions"));
    assert.ok(call, "approval decision is persisted");
    assert.deepEqual(call.body, {
      decision: "approved", reviewer: "chef-ana", notes: null,
    });
    assert.ok(container.textContent.includes("Version 2 · Approved"),
      "approved status is reflected");
  } finally {
    cleanup();
  }
});

test("saving edits persists ingredients without running inference", async () => {
  const { act, dom, container, cleanup } = await renderRecipe(POPULATED);
  try {
    const ingredient = container.querySelector(
      'input[aria-label="Ingredient"]');
    await act(async () => {
      setInputValue(dom, ingredient, "rye flour");
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    const save = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent === "Save changes");
    assert.ok(save, "Save changes is a primary action");
    await click(act, save);
    const call = posted.find((entry) => entry.url.endsWith(
      "/recipe/corrections"));
    assert.ok(call, "chef edits are persisted");
    assert.equal(call.body.reviewer, "");
    assert.equal(call.body.ingredients[0].ingredient_id, "rye flour");
    assert.ok(!posted.some((entry) => entry.url.endsWith(
      "/recipe/generate")), "saving never triggers inference");
  } finally {
    cleanup();
  }
});

test("regeneration is live-only with explicit confirmation", async () => {
  const { act, container, cleanup } = await renderRecipe(POPULATED);
  try {
    assert.ok(!container.textContent.includes("Practice version"),
      "simulation is not a normal regeneration control");
    assert.ok(!container.textContent.includes("Use live AI"),
      "no simulation/live toggle competes here");
    const regen = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent === "Regenerate with live AI");
    assert.ok(regen, "live regeneration is offered");
    assert.ok(regen.disabled, "regeneration waits for confirmation");
    const regenDetails = regen.closest("details");
    const confirm = regenDetails.querySelector('input[type="checkbox"]');
    await act(async () => {
      confirm.dispatchEvent(
        new globalThis.MouseEvent("click", { bubbles: true }));
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    assert.ok(!regen.disabled, "confirmation unlocks regeneration");
    await click(act, regen);
    const call = posted.find((entry) => entry.url.endsWith(
      "/recipe/generate"));
    assert.ok(call, "regeneration runs inference");
    assert.deepEqual(call.body, {
      mode: "live", confirmed: true, max_rounds: 1,
    });
  } finally {
    cleanup();
  }
});

test("empty state welcomes and navigates without editor chrome", async () => {
  const { act, container, cleanup } = await renderRecipe(null);
  try {
    assert.ok(container.textContent.includes(
      "Your recipe isn't ready yet"),
      "empty state is welcoming");
    const headings = Array.from(container.querySelectorAll("h3")).map(
      (heading) => heading.textContent);
    for (const label of ["Ingredients", "Preparation steps",
      "Chef decision", "Version history"]) {
      assert.ok(!headings.includes(label),
        `empty state hides the "${label}" section`);
    }
    assert.ok(!container.textContent.includes("Regenerate"),
      "empty state hides regeneration");
    assert.ok(!container.textContent.includes("Draft v"),
      "empty state shows no contradictory draft header");
    const go = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent === "Go to Generate recipe");
    assert.ok(go, "empty state offers navigation");
    await click(act, go);
    assert.ok(navigated, "navigation action fires");
  } finally {
    cleanup();
  }
});

test("record without generation is marked preliminary", async () => {
  const { container, cleanup } = await renderRecipe({
    ...POPULATED, generation: null,
  });
  try {
    assert.ok(container.textContent.includes("Preliminary record"),
      "internal record is distinguished from a generated recipe");
    assert.ok(container.textContent.includes("Ingredients"),
      "existing content stays editable");
  } finally {
    cleanup();
  }
});

test("live generation is never labeled with the draft simulation note", async () => {
  const { container, cleanup } = await renderRecipe({
    ...POPULATED,
    simulation: true,
    simulation_note: "Simulated adversarial exchange: deterministic "
      + "evidence proposer and fake challenger only. "
      + "No live model was called.",
  });
  try {
    assert.ok(container.textContent.includes("Live AI version."),
      "attached live generation is labeled live");
    assert.ok(!container.textContent.includes("No live model was called"),
      "draft-creation simulation note is hidden next to live output");
  } finally {
    cleanup();
  }
});

test("simulated generation keeps its simulation note", async () => {
  const { container, cleanup } = await renderRecipe({
    ...POPULATED,
    simulation: true,
    simulation_note: "Simulated adversarial exchange: deterministic "
      + "evidence proposer and fake challenger only. "
      + "No live model was called.",
    generation: { ...POPULATED.generation, simulated: true },
  });
  try {
    assert.ok(container.textContent.includes(
      "Practice version (simulation)."),
      "simulated generation is labeled simulation");
    assert.ok(container.textContent.includes("No live model was called"),
      "simulation note is shown for simulated output");
  } finally {
    cleanup();
  }
});

test("failed checks block approval with reasons", async () => {
  const { act, dom, container, cleanup } = await renderRecipe({
    ...POPULATED,
    verification: {
      verdict: "fail",
      issues: [{
        code: "step_without_evidence",
        entity_id: "step_1",
        message: "Step 1 has no linked evidence.",
      }],
    },
  });
  try {
    const reviewer = container.querySelector(
      'input[placeholder="e.g. chef-maria"]');
    await act(async () => {
      setInputValue(dom, reviewer, "chef-ana");
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    const approve = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent === "Approve recipe");
    assert.ok(approve.disabled, "failed checks block approval");
    assert.ok(container.textContent.includes(
      "can't be approved yet — 1 automatic check failed."),
      "block message names the count");
    assert.ok(container.textContent.includes(
      "Step 1 has no linked evidence."),
      "issue reason is shown");
  } finally {
    cleanup();
  }
});

test("step usages reload persisted chef-supplied quantities", async () => {
  const { container, cleanup } = await renderRecipe(WITH_USAGES);
  try {
    const qty = container.querySelector(
      'input[aria-label="Quantity of flour"]');
    assert.ok(qty, "usage quantity input is rendered per step");
    assert.equal(qty.value, "200", "persisted quantity reloads");
    const unit = container.querySelector(
      'input[aria-label="Unit of flour"]');
    assert.ok(unit, "usage unit input is rendered per step");
    assert.equal(unit.value, "g", "persisted unit reloads");
    assert.ok(container.textContent.includes("chef-supplied"),
      "chef-entered usage is labeled chef-supplied");
    assert.ok(!container.querySelector('input[aria-label="Quantity"]'),
      "summary no longer edits quantities");
  } finally {
    cleanup();
  }
});

test("legacy steps without usages initialize from ingredient refs", async () => {
  const { container, cleanup } = await renderRecipe(POPULATED);
  try {
    const qty = container.querySelector(
      'input[aria-label="Quantity of flour"]');
    assert.ok(qty, "legacy ingredient ref becomes a usage row");
    assert.equal(qty.value, "", "legacy usage starts unknown");
    assert.ok(container.textContent.includes("Used in Step 1"),
      "summary shows step linkage instead of quantities");
  } finally {
    cleanup();
  }
});

test("saving a usage persists quantity without summary quantity", async () => {
  const { act, dom, container, cleanup } = await renderRecipe(POPULATED);
  try {
    const qty = container.querySelector(
      'input[aria-label="Quantity of flour"]');
    const unit = container.querySelector(
      'input[aria-label="Unit of flour"]');
    await act(async () => {
      setInputValue(dom, qty, "200");
      setInputValue(dom, unit, "g");
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    const save = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent === "Save changes");
    await click(act, save);
    const call = posted.find((entry) => entry.url.endsWith(
      "/recipe/corrections"));
    assert.ok(call, "chef edits are persisted");
    assert.deepEqual(call.body.steps[0].usages, [{
      ingredient_id: "flour",
      quantity: { value: 200, unit: "g", derivation: "chef_supplied" },
      provenance: "chef-supplied",
    }]);
    assert.ok(!("quantity" in call.body.ingredients[0]),
      "summary carries no net quantity to double-count");
  } finally {
    cleanup();
  }
});

test("blank usage quantity persists as unknown, never zero", async () => {
  const { act, container, cleanup } = await renderRecipe(POPULATED);
  try {
    const save = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent === "Save changes");
    await click(act, save);
    const call = posted.find((entry) => entry.url.endsWith(
      "/recipe/corrections"));
    assert.ok(call, "chef edits are persisted");
    assert.deepEqual(call.body.steps[0].usages, [{
      ingredient_id: "flour",
      quantity: { value: null, unit: null, derivation: "chef_supplied" },
      provenance: "chef-supplied",
    }]);
  } finally {
    cleanup();
  }
});

test("invalid usage quantity blocks saving with an explanation", async () => {
  const { act, dom, container, cleanup } = await renderRecipe(POPULATED);
  try {
    const qty = container.querySelector(
      'input[aria-label="Quantity of flour"]');
    await act(async () => {
      setInputValue(dom, qty, "lots");
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    const save = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent === "Save changes");
    await click(act, save);
    assert.ok(!posted.some((entry) => entry.url.endsWith(
      "/recipe/corrections")), "invalid quantity is never persisted");
    assert.ok(container.textContent.includes("Cannot save:"),
      "save failure is explained");
  } finally {
    cleanup();
  }
});

test("linking and removing step usages updates the editor", async () => {
  const { act, dom, container, cleanup } = await renderRecipe(WITH_USAGES);
  try {
    assert.ok(!container.querySelector(
      'input[aria-label="Quantity of sugar"]'),
      "sugar starts unlinked");
    const select = container.querySelector(
      'select[aria-label="Ingredient to link"]');
    assert.ok(select, "link control is offered");
    await act(async () => {
      setSelectValue(dom, select, "sugar");
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    const link = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent === "Link ingredient");
    await click(act, link);
    assert.ok(container.querySelector(
      'input[aria-label="Quantity of sugar"]'),
      "linked ingredient gains a usage row");
    const removeButtons = Array.from(
      container.querySelectorAll("button")).filter(
      (button) => button.textContent === "Remove");
    assert.equal(removeButtons.length, 2, "each usage can be removed");
    await click(act, removeButtons[0]);
    assert.ok(!container.querySelector(
      'input[aria-label="Quantity of flour"]'),
      "removed usage leaves the editor");
  } finally {
    cleanup();
  }
});

test("undo discards unsaved edits without persisting", async () => {
  const { act, dom, container, cleanup } = await renderRecipe(POPULATED);
  try {
    const ingredient = container.querySelector(
      'input[aria-label="Ingredient"]');
    await act(async () => {
      setInputValue(dom, ingredient, "rye flour");
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    assert.ok(container.textContent.includes("Unsaved edits"),
      "edits are marked unsaved");
    const undo = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent === "Undo unsaved changes");
    assert.ok(undo, "undo action is offered");
    await click(act, undo);
    assert.equal(ingredient.value, "flour", "edit is reverted");
    assert.ok(!container.textContent.includes("Unsaved edits"),
      "dirty state clears");
    assert.ok(!posted.some((entry) => entry.url.endsWith(
      "/recipe/corrections")), "undo persists nothing");
  } finally {
    cleanup();
  }
});

test("empty recipe cannot be approved", async () => {
  const { act, dom, container, cleanup } = await renderRecipe({
    ...POPULATED, ingredients: [], steps: [],
  });
  try {
    const reviewer = container.querySelector(
      'input[placeholder="e.g. chef-maria"]');
    await act(async () => {
      setInputValue(dom, reviewer, "chef-ana");
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    const approve = Array.from(container.querySelectorAll("button")).find(
      (button) => button.textContent === "Approve recipe");
    assert.ok(approve.disabled, "empty recipe cannot be approved");
    assert.ok(container.textContent.includes(
      "Add at least one ingredient or step before approving."),
      "empty approval is explained");
  } finally {
    cleanup();
  }
});
