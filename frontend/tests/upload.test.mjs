// Phase 3 upload-workflow pure-logic tests (plain Node).

import assert from "node:assert/strict";
import test from "node:test";

import {
  buildMetaGrant,
  confirmationSentence,
  estimateFrames,
  extraMetaRequests,
  formatDuration,
} from "../src/upload.ts";

test("estimateFrames rounds up at 1 FPS", () => {
  assert.equal(estimateFrames(0), 0);
  assert.equal(estimateFrames(29.2), 30);
  assert.equal(estimateFrames(720), 720);
  assert.equal(estimateFrames(Number.NaN), 0);
});

test("formatDuration renders minutes and seconds", () => {
  assert.equal(formatDuration(45), "45s");
  assert.equal(formatDuration(754), "12m 34s");
});

test("buildMetaGrant matches the backend confirmation contract", () => {
  const grant = buildMetaGrant(720, "web-user",
    new Date("2026-11-01T00:00:00.000Z"));
  assert.equal(grant.workflow, "video_pipeline_multimodal");
  assert.equal(grant.model, "muse-spark-1.3");
  assert.equal(grant.max_requests, 726);
  assert.equal(grant.max_frames, 720);
  assert.equal(grant.token_budget, 4096);
  assert.equal(grant.strict_cost, true);
  assert.equal(grant.cost_ceiling_usd, 20);
  assert.equal(
    grant.confirmation,
    "authorize video_pipeline_multimodal model muse-spark-1.3 "
      + "up to 726 requests, 4096 output tokens, 720 frames, "
      + "operator web-user",
  );
  assert.equal(
    confirmationSentence(726, 720, "web-user"), grant.confirmation);
});

test("buildMetaGrant always applies the USD 20 authorized cap", () => {
  const grant = buildMetaGrant(30);
  assert.equal(grant.cost_ceiling_usd, 20);
  assert.equal(grant.max_requests, 34);
});

test("buildMetaGrant reserves one call per ASR chunk", () => {
  assert.equal(buildMetaGrant(300).max_requests, 304);
  assert.equal(buildMetaGrant(645).max_requests, 650);
  assert.equal(buildMetaGrant(700).max_requests, 706);
  assert.equal(extraMetaRequests(330), 4);
  assert.equal(extraMetaRequests(331), 5);
});
