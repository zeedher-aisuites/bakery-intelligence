// Unit tests for results.ts presentation helpers (M2-06B.2).
// Runs on plain Node.js with type stripping; no extra packages.
// Requires Node.js 22.18 or later: node --test tests/

import assert from "node:assert/strict";
import test from "node:test";

import {
  numeric,
  seconds,
  sourceChip,
  statusPill,
  text,
  unitLabel,
} from "../src/results.ts";

test("sourceChip maps backend value_source vocabulary", () => {
  assert.deepEqual(sourceChip("verification"), {
    label: "Verified reading",
    pill: "pill-ok",
  });
  assert.deepEqual(sourceChip("firstpass"), {
    label: "First-pass model reading",
    pill: "pill-pending",
  });
  assert.equal(sourceChip("verification_unusable").pill, "pill-bad");
  assert.equal(sourceChip("firstpass_unusable").pill, "pill-bad");
  assert.deepEqual(sourceChip("future_source"), {
    label: "future_source",
    pill: "pill-pending",
  });
});

test("statusPill maps value_status vocabulary", () => {
  assert.equal(statusPill("ok"), "pill-ok");
  assert.equal(statusPill("ambiguous"), "pill-pending");
  assert.equal(statusPill("invalid"), "pill-bad");
  assert.equal(statusPill("future_status"), "pill-pending");
});

test("unitLabel prefers observed over assigned over undetermined", () => {
  assert.deepEqual(unitLabel("kg", undefined), {
    text: "kg (observed)",
    pill: "pill-ok",
  });
  assert.deepEqual(
    unitLabel(null, {
      measurement_id: "m_1",
      unit: "g",
      provenance: "human_assigned",
      assignment_id: "ua_0001",
    }),
    { text: "g (human-assigned ua_0001)", pill: "pill-pending" },
  );
  assert.deepEqual(unitLabel(null, undefined), {
    text: "undetermined",
    pill: "pill-pending",
  });
  assert.deepEqual(
    unitLabel(null, {
      measurement_id: "m_1",
      unit: null,
      provenance: "undetermined",
      assignment_id: null,
    }),
    { text: "undetermined", pill: "pill-pending" },
  );
});

test("text, numeric, and seconds never emit nulls", () => {
  assert.equal(text(null), "—");
  assert.equal(text(undefined), "—");
  assert.equal(text("g"), "g");
  assert.equal(numeric(null), "—");
  assert.equal(numeric(undefined), "—");
  assert.equal(numeric(515.0), "515");
  assert.equal(seconds(148.0), "148.0 s");
});
