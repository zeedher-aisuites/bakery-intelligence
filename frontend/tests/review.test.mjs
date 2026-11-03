// Regression tests for review-workspace pure logic (demo-first).
// Covers timeline ordering, frame states, effective readings,
// recipe provenance, and review export builders. Runs on plain
// Node.js; no extra packages. Requires Node.js 22.18 or later.

import assert from "node:assert/strict";
import test from "node:test";

import {
  buildRecipeRows,
  effectiveReading,
  exportReviewCsv,
  exportReviewJson,
  frameState,
  orderFrames,
  summarizeReview,
} from "../src/review.ts";

function frame(overrides = {}) {
  return {
    frame_index: 0,
    timestamp_seconds: 0.0,
    image_available: true,
    image_name: "frame_00000000.jpg",
    image_sha256: "abc",
    image_matches_record: true,
    observation: {
      frame_index: 0,
      timestamp_seconds: 0.0,
      terminal_state: "valid_prediction",
      parse_error: null,
      response_status: "completed",
      model_fields: {
        visible_objects: ["scale"],
        scale_display_verbatim: "29",
        scale_readable: true,
        scale_state: "stable",
        tare_action_visible: false,
        addition_or_removal_visible: "none",
        container_description: "bowl",
        container_changed_hint: false,
        confidence: "high",
        ambiguity_notes: null,
      },
      request_id: "resp_1",
      latency_seconds: 1.0,
      input_tokens: 100,
      output_tokens: 50,
      reasoning_output_tokens: 10,
    },
    measurement: {
      measurement_id: "m_00000000",
      value: 29.0,
      value_source: "firstpass",
      value_status: "ok",
      unit_observed: null,
    },
    review: null,
    ...overrides,
  };
}

function bundle(frames) {
  return {
    run_id: "run_0123456789ab",
    workflow: "video_pipeline_meta",
    status: "completed",
    source_object_id: "clip.mp4",
    source_sha256: "def",
    frames,
    reviews: {},
    missing: [],
  };
}

test("orderFrames sorts chronologically without mutating", () => {
  const b = frame({ frame_index: 10 });
  const a = frame({ frame_index: 0 });
  const input = [b, a];
  const ordered = orderFrames(input);
  assert.deepEqual(ordered.map((item) => item.frame_index), [0, 10]);
  assert.equal(input[0].frame_index, 10);
});

test("frameState distinguishes numeric, abstained, missing", () => {
  assert.equal(frameState(frame()).kind, "numeric");
  const abstained = frame({
    observation: { ...frame().observation, terminal_state: "abstained" },
    measurement: {
      measurement_id: "m",
      value: null,
      value_source: "firstpass_unusable",
      value_status: "invalid",
      unit_observed: null,
    },
  });
  assert.equal(frameState(abstained).kind, "abstained");
  assert.equal(
    frameState(frame({ image_available: false })).kind, "missing");
  assert.equal(
    frameState(frame({ observation: null })).kind, "missing");
});

test("frameState marks reviewed only with an annotation", () => {
  assert.equal(frameState(frame()).reviewed, false);
  const reviewed = frame({
    review: {
      frame_index: 0,
      decision: "confirmed",
      value: null,
      notes: null,
      reviewer: null,
      updated_at: "2026-11-01T00:00:00+00:00",
      schema_version: "review-annotations/1",
    },
  });
  const state = frameState(reviewed);
  assert.equal(state.reviewed, true);
  assert.equal(state.kind, "numeric");
  assert.match(state.label, /reviewed/);
});

test("effectiveReading resolves provenance honestly", () => {
  assert.deepEqual(effectiveReading(frame()), {
    value: 29.0,
    provenance: "ai-extracted",
    pill: "pill-pending",
    note: "Awaiting human review.",
  });
  const corrected = frame({
    review: {
      frame_index: 0,
      decision: "corrected",
      value: 30.0,
      notes: null,
      reviewer: "op",
      updated_at: "t",
      schema_version: "review-annotations/1",
    },
  });
  assert.equal(effectiveReading(corrected).value, 30.0);
  assert.equal(
    effectiveReading(corrected).provenance, "human-corrected");
  assert.equal(effectiveReading(corrected).pill, "pill-info");
  const confirmed = frame({
    review: { ...corrected.review, decision: "confirmed", value: null },
  });
  assert.equal(
    effectiveReading(confirmed).provenance, "human-verified");
  const rejected = frame({
    review: { ...corrected.review, decision: "rejected", value: null },
  });
  assert.equal(effectiveReading(rejected).value, null);
  assert.equal(effectiveReading(rejected).provenance, "rejected");
  const uncertain = frame({
    review: { ...corrected.review, decision: "uncertain", value: null },
  });
  assert.equal(effectiveReading(uncertain).provenance, "uncertain");
  const invalid = frame({
    measurement: {
      measurement_id: "m",
      value: null,
      value_source: "firstpass_unusable",
      value_status: "invalid",
      unit_observed: null,
    },
  });
  assert.equal(effectiveReading(invalid).provenance, "unknown");
});

test("buildRecipeRows keeps gaps as gaps", () => {
  const rows = buildRecipeRows(bundle([
    frame({ frame_index: 10 }),
    frame({
      frame_index: 0,
      measurement: {
        measurement_id: "m",
        value: null,
        value_source: "firstpass_unusable",
        value_status: "invalid",
        unit_observed: null,
      },
    }),
  ]));
  assert.deepEqual(rows.map((row) => row.frame_index), [0, 10]);
  assert.equal(rows[0].provenance, "unknown");
  assert.equal(rows[0].value, null);
  assert.equal(rows[1].provenance, "ai-extracted");
});

test("summarizeReview counts states", () => {
  const counts = summarizeReview(bundle([
    frame({ frame_index: 0 }),
    frame({
      frame_index: 10,
      image_available: false,
      review: {
        frame_index: 10,
        decision: "uncertain",
        value: null,
        notes: null,
        reviewer: null,
        updated_at: "t",
        schema_version: "review-annotations/1",
      },
    }),
  ]));
  assert.deepEqual(counts, {
    frames: 2,
    reviewed: 1,
    pending: 1,
    numeric: 1,
    abstained: 0,
    missing: 1,
  });
});

test("export builders preserve model values and reviews", () => {
  const data = bundle([frame({
    review: {
      frame_index: 0,
      decision: "corrected",
      value: 30.0,
      notes: "digit check",
      reviewer: "op",
      updated_at: "t",
      schema_version: "review-annotations/1",
    },
  })]);
  const csv = exportReviewCsv(data);
  assert.match(csv, /frame_index,timestamp_seconds/);
  assert.match(csv, /0,0,valid_prediction,29,29,ok,corrected,30/);
  const parsed = JSON.parse(exportReviewJson(data));
  assert.equal(parsed.run_id, "run_0123456789ab");
  assert.equal(parsed.exported_decisions, 1);
  assert.equal(parsed.rows[0].model_value, 29.0);
  assert.equal(parsed.rows[0].effective_value, 30.0);
  assert.equal(
    parsed.rows[0].effective_provenance, "human-corrected");
});

test("effectiveReading degrades unknown stored decisions to gaps", () => {
  const odd = frame({
    review: {
      frame_index: 0,
      decision: "archived",
      value: 30.0,
      notes: null,
      reviewer: null,
      updated_at: "2026-01-01T00:00:00Z",
      schema_version: "review-annotations/1",
    },
  });
  const reading = effectiveReading(odd);
  assert.equal(reading.value, null);
  assert.equal(reading.provenance, "unknown");
  const rows = buildRecipeRows(bundle([odd]));
  assert.equal(rows[0].value, null);
  assert.equal(rows[0].provenance, "unknown");
});
