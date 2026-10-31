// Regression tests for workflow-aware summary selection (M2-06C.4).
// Guards the pipeline 404 fix: each workflow must resolve its
// own summary artifact, and pipeline summaries must be
// recognizable as synthetic. Runs on plain Node.js; no extra
// packages. Requires Node.js 22.18 or later.

import assert from "node:assert/strict";
import test from "node:test";

import {
  WORKFLOW_OFFLINE,
  WORKFLOW_PIPELINE,
  WORKFLOW_SMOKE,
} from "../src/api.ts";
import {
  isPipelineSummary,
  summaryArtifactId,
} from "../src/results.ts";

test("summaryArtifactId resolves per workflow", () => {
  assert.equal(summaryArtifactId(WORKFLOW_OFFLINE),
    "offline_summary.json");
  assert.equal(summaryArtifactId(WORKFLOW_PIPELINE),
    "pipeline_summary.json");
  assert.equal(summaryArtifactId(WORKFLOW_SMOKE), null);
  assert.equal(summaryArtifactId("unknown_workflow"), null);
});

test("isPipelineSummary detects synthetic inference", () => {
  assert.equal(
    isPipelineSummary({ synthetic_inference: true }), true);
  assert.equal(
    isPipelineSummary({ synthetic_inference: false }), false);
  assert.equal(isPipelineSummary({}), false);
});
