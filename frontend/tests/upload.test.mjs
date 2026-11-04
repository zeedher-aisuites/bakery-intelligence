// Phase 3 upload-workflow pure-logic tests (plain Node).

import assert from "node:assert/strict";
import test from "node:test";

import {
  FIRSTPASS_ENGINE_BATCH,
  FIRSTPASS_ENGINE_FOCUSED,
  FIRSTPASS_ENGINE_SCALE,
  FIRSTPASS_ENGINE_SINGLE,
  buildMetaGrant,
  confirmationSentence,
  estimateFrames,
  extraMetaRequests,
  firstpassRequests,
  formatDuration,
  scaleWorstCaseUsd,
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
test("firstpassRequests scales by engine batch size", () => {
  assert.equal(firstpassRequests(645, FIRSTPASS_ENGINE_SINGLE), 645);
  assert.equal(firstpassRequests(645, FIRSTPASS_ENGINE_SCALE), 65);
  assert.equal(firstpassRequests(645, FIRSTPASS_ENGINE_BATCH), 65);
  assert.equal(firstpassRequests(645, FIRSTPASS_ENGINE_FOCUSED), 13);
  assert.equal(firstpassRequests(12, FIRSTPASS_ENGINE_SCALE), 2);
  assert.equal(firstpassRequests(0, FIRSTPASS_ENGINE_SCALE), 0);
  assert.equal(firstpassRequests(645, "unknown-engine"), 645);
});

test("scaleWorstCaseUsd matches the backend reservation", () => {
  const bound = scaleWorstCaseUsd(65);
  assert.ok(Math.abs(bound - 2.99429) < 0.001);
  assert.ok(scaleWorstCaseUsd(65) < 20);
});

test("buildMetaGrant carries the scale engine and batch math", () => {
  const grant = buildMetaGrant(645, "web-user",
    new Date("2026-11-01T00:00:00.000Z"), FIRSTPASS_ENGINE_SCALE);
  assert.equal(grant.firstpass_engine, "parallel_scale");
  assert.equal(grant.max_requests, 65 + extraMetaRequests(645));
  assert.equal(grant.token_budget, 4096);
  assert.equal(
    grant.confirmation,
    "authorize video_pipeline_multimodal model muse-spark-1.3 "
      + `up to ${grant.max_requests} requests, 4096 output tokens, `
      + "645 frames, operator web-user",
  );
});

test("buildMetaGrant defaults to single-frame math", () => {
  const grant = buildMetaGrant(645);
  assert.equal(grant.firstpass_engine, "single_frame");
  assert.equal(grant.max_requests, 645 + extraMetaRequests(645));
});
