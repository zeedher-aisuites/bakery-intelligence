// Chef recipe-review helper tests (plain Node).

import assert from "node:assert/strict";
import test from "node:test";

import {
  approvalBlocked,
  canDecide,
  evidenceScopeLabel,
  formatVideoTime,
  intervalLabel,
  linkFrameToStep,
  parseVideoTime,
  provenanceLabel,
  stepsReferencingFrame,
} from "../src/recipe.ts";

test("provenance labels distinguish chef-supplied content", () => {
  assert.match(provenanceLabel("chef-supplied"), /not observed/i);
  assert.equal(provenanceLabel("human-corrected"), "Human corrected");
  assert.equal(provenanceLabel("ai-extracted"), "AI extracted");
});

test("decisions require reviewer identity and pass verification", () => {
  assert.equal(canDecide(""), false);
  assert.equal(canDecide("chef-maria"), true);
  assert.equal(approvalBlocked("fail"), true);
  assert.equal(approvalBlocked("review_needed"), false);
});

test("frame identifiers map to human video time", () => {
  // Source frame 1150 at 50 fps is second 23 of the video.
  assert.equal(formatVideoTime(1150 / 50), "00:23");
  assert.equal(formatVideoTime(2500 / 50), "00:50");
  assert.equal(formatVideoTime(null), "unknown time");
  assert.equal(parseVideoTime("00:23"), 23);
  assert.equal(parseVideoTime(""), null);
  assert.equal(intervalLabel(23, 29), "00:23–00:29");
  assert.equal(intervalLabel(null, null), "time not recorded");
});

test("representative evidence is not labeled complete", () => {
  assert.equal(evidenceScopeLabel({
    evidence_scope: "representative", evidence_total: 24,
    evidence_refs: [{ frame_index: 1150 }, { frame_index: 2500 }],
  }), "Representative evidence (2 of 24 frames)");
});

test("observation-to-step lookup and explicit linking only", () => {
  const step = {
    step_id: "step_1", frame_refs: [{ frame_index: 1150 }],
    ingredient_refs: [],
  };
  assert.equal(stepsReferencingFrame([step], 1150).length, 1);
  assert.equal(stepsReferencingFrame([step], 2500).length, 0);
  const before = JSON.stringify(step);
  const linked = linkFrameToStep(step, 2500);
  assert.equal(linked.frame_refs.length, 2);
  // No automatic association: the original step is untouched
  // until the reviewer explicitly saves the linked copy.
  assert.equal(JSON.stringify(step), before);
});
