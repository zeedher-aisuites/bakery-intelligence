// Chef Review table + regeneration-form tests.
//
// Renders the real <RecipeEditor/> with a stubbed backend:
// multiline ingredient names and the optional-instructions
// revision validation matrix.

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

const LONG_NAME =
  "Líquido blanco opaco en bol metálico grande con apariencia de leche";

const RECIPE = {
  recipe_id: "r1",
  recipe_name: "Roles XL",
  title: "Roles XL",
  language: "es",
  recipe_status: "draft",
  ingredients: [{
    name: LONG_NAME,
    quantity: 238,
    unit: "g",
    quantity_status: "seen_video",
    quantity_basis: "Stable display.",
    evidence: [],
  }],
  method: [{ number: 1, instruction: "Mezcle." }],
  limitations: ["Confirme la unidad."],
  improvement_suggestions: [{
    id: "flour-amount",
    label: "Cantidad de harina",
    hint: "Gramos de harina.",
    placeholder: "700",
  }],
  last_saved_at: null,
  parent_name: null,
};

const posted = [];
let reviseImpl = null;

function stubFetch() {
  posted.length = 0;
  reviseImpl = null;
  globalThis.fetch = async (url, options) => {
    const text = String(url);
    const method = (options?.method ?? "GET").toUpperCase();
    if (method === "POST" && text.endsWith("/revise")) {
      const body = JSON.parse(String(options.body));
      posted.push(body);
      if (reviseImpl) return reviseImpl(body);
      return {
        ok: true,
        status: 201,
        json: async () => ({
          recipe_id: "r1_r1",
          parent_version: "r1",
          provider: "baseten",
          model: "m",
          request_id: "chat_x",
          actual_usd: null,
          recipe: { ...RECIPE, recipe_status: "draft" },
        }),
      };
    }
    if (text.endsWith("/api/recipes/r1")) {
      return { ok: true, status: 200, json: async () => RECIPE };
    }
    return { ok: true, status: 200, json: async () => ({}) };
  };
}

async function renderEditor() {
  stubFetch();
  const dom = new JSDOM("<!DOCTYPE html><div id=\"root\"></div>", {
    url: "http://localhost/",
  });
  for (const key of ["window", "document", "HTMLElement", "Element",
    "Node", "MouseEvent", "Event", "InputEvent", "HTMLInputElement",
    "HTMLTextAreaElement", "getComputedStyle", "localStorage"]) {
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
  const RecipeEditor =
    require("../src/RecipeEditor.tsx").default;
  let navigatedTo = null;
  const container = document.getElementById("root");
  const root = createRoot(container);
  await act(async () => {
    root.render(React.createElement(RecipeEditor, {
      recipeId: "r1",
      onNewVersion: (id) => { navigatedTo = id; },
    }));
    await new Promise((resolve) => setTimeout(resolve, 300));
  });
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 100));
  });
  return {
    act,
    dom,
    container,
    navigated: () => navigatedTo,
    cleanup: () => root.unmount(),
  };
}

function setField(dom, field, value) {
  const proto = field.tagName === "TEXTAREA"
    ? dom.window.HTMLTextAreaElement.prototype
    : dom.window.HTMLInputElement.prototype;
  const setter = Object.getOwnPropertyDescriptor(proto, "value").set;
  setter.call(field, value);
  field.dispatchEvent(new dom.window.Event("input", { bubbles: true }));
}

function generateButton(container) {
  return Array.from(container.querySelectorAll("button")).find(
    (button) => (button.textContent ?? "").includes("Generar nueva"),
  );
}

async function click(act, element) {
  await act(async () => {
    element.dispatchEvent(
      new MouseEvent("click", { bubbles: true }),
    );
    await new Promise((resolve) => setTimeout(resolve, 150));
  });
}

test("ingredient name is a multiline field with the full value", async () => {
  const { container, cleanup } = await renderEditor();
  try {
    const name = container.querySelector(
      'textarea[aria-label="Ingrediente 1 name"]',
    );
    assert.ok(name, "name field is not a textarea");
    assert.equal(name.tagName, "TEXTAREA");
    assert.equal(name.value, LONG_NAME);
    assert.ok(
      container.querySelector('input[aria-label="Ingrediente 1 cantidad"]'),
      "quantity field changed shape",
    );
    assert.ok(
      container.querySelector('input[aria-label="Ingrediente 1 unidad"]'),
      "unit field changed shape",
    );
  } finally {
    cleanup();
  }
});

test("ingredient table CSS wraps names without clipping", () => {
  const css = fs.readFileSync(
    path.join(process.cwd(), "src/recipe.css"), "utf-8");
  assert.ok(css.includes(".rc-table textarea"),
    "table textarea rules are missing");
  assert.ok(css.includes("field-sizing: content"),
    "auto-growing textarea rule is missing");
});

test("empty form blocks revision with a Spanish message", async () => {
  const { act, container, cleanup } = await renderEditor();
  try {
    await click(act, generateButton(container));
    const error = container.querySelector(".rc-error");
    assert.ok(error, "no validation error shown");
    assert.match(error.textContent ?? "",
      /Selecciona al menos una mejora/);
    assert.equal(posted.length, 0);
  } finally {
    cleanup();
  }
});

test("selected suggestion without value names the gap", async () => {
  const { act, container, cleanup } = await renderEditor();
  try {
    const checkbox = container.querySelector(
      '.rc-suggest-item input[type="checkbox"]',
    );
    assert.ok(checkbox, "suggestion checkbox missing");
    await click(act, checkbox);
    await click(act, generateButton(container));
    const error = container.querySelector(".rc-error");
    assert.ok(error, "no validation error shown");
    assert.match(error.textContent ?? "",
      /Escribe el valor de cada mejora/);
    assert.equal(posted.length, 0);
  } finally {
    cleanup();
  }
});

test("valued correction submits without free text", async () => {
  const { act, dom, container, cleanup, navigated } =
    await renderEditor();
  try {
    const checkbox = container.querySelector(
      '.rc-suggest-item input[type="checkbox"]',
    );
    await click(act, checkbox);
    const value = container.querySelector(".rc-suggest-input");
    assert.ok(value, "correction value field missing");
    await act(async () => {
      setField(dom, value, "750");
      await new Promise((resolve) => setTimeout(resolve, 100));
    });
    await click(act, generateButton(container));
    assert.equal(posted.length, 1);
    assert.equal(posted[0].instructions, "");
    assert.deepEqual(posted[0].suggestions, [{
      id: "flour-amount", selected: true, value: "750",
    }]);
    assert.equal(navigated(), "r1_r1");
  } finally {
    cleanup();
  }
});

test("free text alone submits without selections", async () => {
  const { act, dom, container, cleanup, navigated } =
    await renderEditor();
  try {
    const box = container.querySelector(
      'textarea[aria-label="Indica qué quieres mejorar"]',
    );
    assert.ok(box, "instruction box missing");
    await act(async () => {
      setField(dom, box, "Aclara el amasado.");
      await new Promise((resolve) => setTimeout(resolve, 100));
    });
    await click(act, generateButton(container));
    assert.equal(posted.length, 1);
    assert.equal(posted[0].instructions, "Aclara el amasado.");
    assert.equal(navigated(), "r1_r1");
  } finally {
    cleanup();
  }
});

test("failed revision keeps inputs and shows Spanish error", async () => {
  const { act, dom, container, cleanup, navigated } =
    await renderEditor();
  reviseImpl = async () => ({
    ok: false,
    status: 502,
    json: async () => ({
      detail: "Revision model call failed: provider exploded",
    }),
  });
  const silenced = [];
  const realError = console.error;
  console.error = (...args) => { silenced.push(args); };
  try {
    const checkbox = container.querySelector(
      '.rc-suggest-item input[type="checkbox"]',
    );
    await click(act, checkbox);
    const value = container.querySelector(".rc-suggest-input");
    await act(async () => {
      setField(dom, value, "750");
      await new Promise((resolve) => setTimeout(resolve, 100));
    });
    await click(act, generateButton(container));
    const error = container.querySelector(".rc-error");
    assert.ok(error, "no failure message shown");
    assert.match(error.textContent ?? "",
      /No se pudo generar la nueva versión/);
    assert.doesNotMatch(error.textContent ?? "", /exploded/,
      "raw provider error leaked to the chef");
    assert.ok(silenced.length > 0,
      "technical detail was not logged for support");
    assert.equal(navigated(), null);
    assert.equal(
      container.querySelector(".rc-suggest-input").value, "750",
      "correction value was cleared by the failure",
    );
    assert.ok(
      container.querySelector(
        '.rc-suggest-item input[type="checkbox"]').checked,
      "selection was cleared by the failure",
    );
  } finally {
    console.error = realError;
    cleanup();
  }
});

test("double submit sends one revision request", async () => {
  const { act, container, cleanup } = await renderEditor();
  let release = null;
  reviseImpl = () => new Promise((resolve) => {
    release = () => resolve({
      ok: true,
      status: 201,
      json: async () => ({
        recipe_id: "r1_r1",
        parent_version: "r1",
        provider: "baseten",
        model: "m",
        request_id: "chat_x",
        actual_usd: null,
        recipe: { ...RECIPE, recipe_status: "draft" },
      }),
    });
  });
  try {
    const box = container.querySelector(
      'textarea[aria-label="Indica qué quieres mejorar"]',
    );
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(
        HTMLTextAreaElement.prototype, "value").set;
      setter.call(box, "Aclara el amasado.");
      box.dispatchEvent(new Event("input", { bubbles: true }));
      await new Promise((resolve) => setTimeout(resolve, 100));
    });
    const button = generateButton(container);
    await act(async () => {
      button.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    assert.equal(
      button.disabled, true,
      "generate button stays enabled while revising",
    );
    assert.match(button.textContent ?? "", /Generando/,
      "no loading state while revising");
    await act(async () => {
      button.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
      await new Promise((resolve) => setTimeout(resolve, 50));
    });
    await act(async () => {
      release();
      await new Promise((resolve) => setTimeout(resolve, 150));
    });
    assert.equal(posted.length, 1);
  } finally {
    cleanup();
  }
});
