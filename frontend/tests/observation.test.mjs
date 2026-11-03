// Observation Review queue helper tests (plain Node).

import assert from "node:assert/strict";
import test from "node:test";

import {
  applyFrameFilters,
  correctionEffectiveValue,
  correctionResult,
  correctionStatus,
  DEFAULT_FILTERS,
  humanLabel,
  nextTaskId,
  modelLabel,
  openTasks,
  reviewProgress,
  taskLabel,
  taskModelValues,
  validationLabel,
} from "../src/observation.ts";

const tasks = [
  { task_id: "t1", state: "missing_relevant", status: "open", frame_refs: [100, 150] },
  { task_id: "t2", state: "uncertain", status: "resolved", frame_refs: [200] },
];

test("queue progress counts only actionable open tasks", () => {
  assert.equal(openTasks(tasks).length, 1);
  assert.deepEqual(reviewProgress(tasks), {
    total: 2, resolved: 1, open: 1, deferred: 0,
  });
});

test("friendly labels never expose raw enum names", () => {
  assert.equal(taskLabel("missing_relevant"), "Missing measurement");
  assert.equal(taskLabel("contradictory"), "Conflicting measurements");
  assert.equal(validationLabel("accepted"), "Passed automatic checks");
  assert.equal(modelLabel(null), "No reading predicted");
  assert.equal(humanLabel("deferred"), "Review later");
});

test("culinary exception states use chef-friendly labels", () => {
  assert.equal(taskLabel("ingredient_uncertain"), "Uncertain ingredient");
  assert.equal(taskLabel("action_conflicting"), "Conflicting actions");
  assert.equal(taskLabel("transition_uncertain"), "Uncertain transition");
  assert.equal(taskLabel("unit_missing"), "Missing unit");
  assert.equal(taskLabel("tare_unresolved"), "Unresolved tare");
  assert.equal(validationLabel("ingredient_uncertain"), "Needs review");
  assert.equal(validationLabel("unit_missing"), "Needs review");
});

test("combined frame filters select canonical subsets", () => {
  const frames = [
    { frame_index: 0,
      measurement: { value: 10, value_status: "ok" },
      observation: { model_fields: { confidence: "high" } } },
    { frame_index: 50,
      measurement: { value: null, value_status: "firstpass_unusable" },
      observation: { model_fields: { confidence: null } } },
  ];
  const pass = applyFrameFilters(frames, () => "passed", () => "Pending",
    new Set([50]), { ...DEFAULT_FILTERS, inTasks: true });
  assert.deepEqual(pass.map((frame) => frame.frame_index), [50]);
  const empty = applyFrameFilters(frames, () => "passed", () => "Pending",
    new Set(), { ...DEFAULT_FILTERS, model: "reading",
      validation: "conflicting" });
  assert.equal(empty.length, 0);
});

test("stale selection advances or clears", () => {
  const tasks = [
    { task_id: "t1", state: "missing_relevant", status: "resolved", frame_refs: [100] },
    { task_id: "t2", state: "uncertain", status: "open", frame_refs: [200] },
  ];
  assert.equal(nextTaskId(tasks, "t1", false), "t2");
  assert.equal(nextTaskId(tasks, "t9", false), "t2");
  assert.equal(nextTaskId(
    tasks.map((task) => ({ ...task, status: "resolved" })), "t2", false), null);
});

test("correction helpers keep decision types distinct", () => {
  const base = { task_id: "t", frame_refs: [0], status: "resolved" };
  assert.equal(correctionStatus({ ...base }), "Needs decision");
  assert.equal(
    correctionStatus({ ...base, status: "deferred" }), "Set aside");
  assert.equal(correctionStatus({
    ...base, correction: { action: "correct", human_value: 297 },
  }), "Corrected to 297");
  assert.equal(correctionStatus({
    ...base, correction: { action: "correct", human_value: null },
  }), "Corrected");
  assert.equal(correctionStatus({
    ...base, correction: { action: "confirm" },
  }), "Confirmed");
  assert.equal(correctionStatus({
    ...base, correction: { action: "review_later" },
  }), "Set aside");
  assert.equal(correctionResult({
    ...base, correction: { action: "correct", human_value: 297 },
  }, [291]), "Corrected from 291 to 297 (unit unknown).");
  assert.equal(correctionResult({
    ...base, status: "open",
    correction: { action: "correct", human_value: 297 },
  }, [291]), null);
  assert.equal(correctionResult({
    ...base, correction: { action: "confirm" },
  }, [291]), "AI reading confirmed: 291.");
  assert.equal(correctionResult({
    ...base, correction: { action: "not_identifiable" },
  }, []), "Marked as cannot determine — no usable reading established.");
  assert.deepEqual(
    taskModelValues(base, (index) => (index === 0 ? 291 : null)), [291]);
  assert.equal(correctionEffectiveValue({
    ...base, correction: { action: "correct", human_value: 297 },
  }, [291]), 297);
  assert.equal(correctionEffectiveValue({
    ...base, correction: { action: "confirm" },
  }, [291]), 291);
  assert.equal(correctionEffectiveValue({
    ...base, correction: { action: "confirm" },
  }, [291, 292]), null);
  assert.equal(correctionEffectiveValue({
    ...base, correction: { action: "review_later" },
  }, [291]), null);
});
