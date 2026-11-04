// Processing Queue pure-logic tests (plain Node).

import assert from "node:assert/strict";
import test from "node:test";

import {
  canRetry,
  countByStatus,
  displaySourceName,
  errorSummary,
  filterJobs,
  hasActiveJobs,
  retryHint,
  statusLabel,
} from "../src/queue.ts";

const queued = {
  run_id: "run_0123456789ab", workflow: "video_pipeline_offline",
  status: "queued", source_object_id: "upload_ab12cd34ef56ab12_clip.mp4",
  error: null, claimed_by: null, attempts: 0,
};
const metaFailed = {
  ...queued, status: "failed", workflow: "video_pipeline_multimodal",
  error: "Pipeline video is 9 bytes; limit is 1.",
  error_category: "validation", attempts: 1,
};
const offlineFailed = {
  ...metaFailed, workflow: "video_pipeline_offline",
  error: "Artifact read-back mismatch.",
  error_category: "processing", attempts: 1,
};
const offlineInvalid = { ...offlineFailed, error_category: "validation" };

test("displaySourceName strips the stored upload prefix", () => {
  assert.equal(displaySourceName(queued), "clip.mp4");
  assert.equal(
    displaySourceName({ ...queued, source_object_id: "clip.avi" }),
    "clip.avi");
  assert.equal(
    displaySourceName({ ...queued, source_object_id: null }),
    "Unknown source");
});

test("statusLabel uses chef-friendly labels without internals", () => {
  assert.equal(statusLabel(queued), "Waiting");
  assert.equal(
    statusLabel({ ...queued, status: "running", claimed_by: "worker-1" }),
    "Processing");
  assert.equal(
    statusLabel({ ...queued, status: "completed" }), "Ready");
  assert.equal(statusLabel(metaFailed), "Failed");
});

test("retry is offered only for safe non-billable recovery", () => {
  assert.equal(canRetry(offlineFailed), true);
  assert.equal(canRetry(metaFailed), false);
  assert.equal(canRetry(offlineInvalid), false);
  assert.equal(canRetry(queued), false);
  assert.match(retryHint(metaFailed) ?? "", /never retried/i);
  assert.equal(retryHint(offlineFailed), null);
});

test("errorSummary explains failures without internals", () => {
  assert.match(errorSummary(metaFailed) ?? "", /never retried/i);
  assert.match(errorSummary(offlineInvalid) ?? "", /different MP4/i);
  assert.match(errorSummary(offlineInvalid) ?? "", /kept/i);
  assert.match(errorSummary(offlineFailed) ?? "", /kept/i);
  assert.equal(errorSummary(queued), null);
  for (const summary of [
    errorSummary(metaFailed),
    errorSummary(offlineInvalid),
    errorSummary(offlineFailed),
  ]) {
    assert.doesNotMatch(summary ?? "", /worker/i);
  }
});

test("filter, counts, and active detection", () => {
  const jobs = [queued, metaFailed,
    { ...queued, run_id: "run_abcdef012345", status: "completed" }];
  assert.equal(filterJobs(jobs, "all").length, 3);
  assert.equal(filterJobs(jobs, "failed").length, 1);
  assert.equal(countByStatus(jobs).queued, 1);
  assert.equal(hasActiveJobs(jobs), true);
  assert.equal(hasActiveJobs([metaFailed]), false);
});
