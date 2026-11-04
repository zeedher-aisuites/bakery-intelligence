// Chef fusion-review helper tests (plain Node).

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import test from "node:test";

import {
  bucketIngredients,
  clipLabel,
  clipsOf,
  effectiveProvenance,
  formatTimestamp,
  intervalLabel,
  orderSteps,
  provenanceBadge,
  provenanceLabel,
  stepsForClip,
  summarizeFusion,
  targetKey,
} from "../src/fusion.ts";

test("timestamps format as m:ss and never invent zero", () => {
  assert.equal(formatTimestamp(0), "0:00");
  assert.equal(formatTimestamp(78), "1:18");
  assert.equal(formatTimestamp(684.7), "11:24");
  assert.equal(formatTimestamp(null), null);
  assert.equal(formatTimestamp(Number.NaN), null);
  assert.equal(formatTimestamp(-5), null);
});

test("intervals keep open ends explicit", () => {
  assert.equal(intervalLabel(0, 10), "0:00–0:10");
  assert.equal(intervalLabel(78, null), "from 1:18");
  assert.equal(intervalLabel(null, null), "time unknown");
});

test("correction never implies verification", () => {
  assert.equal(effectiveProvenance(null), "ai-extracted");
  assert.equal(
    effectiveProvenance({ action: "confirm" }),
    "human-verified",
  );
  assert.equal(
    effectiveProvenance({ action: "correct" }),
    "human-corrected",
  );
  assert.equal(
    effectiveProvenance({ action: "supply" }),
    "chef-supplied",
  );
  assert.equal(
    effectiveProvenance({ action: "cannot_determine" }),
    "unknown",
  );
  assert.equal(provenanceBadge("human-verified"), "badge badge-verified");
  assert.equal(provenanceBadge("chef-supplied"), "badge badge-chef");
  assert.equal(provenanceLabel("ai-extracted"), "AI extracted");
});

test("targets key stably across node types", () => {
  assert.equal(targetKey("step", "eg_000"), "step:eg_000");
  assert.equal(targetKey("question", "q3"), "question:q3");
});

test("progress counts every reviewable node", () => {
  const draft = {
    ingredients: [{ group_id: "ig_a" }],
    steps: [{ group_id: "eg_0", step_index: 1 }],
    conflicts: [{ conflict_id: "cf_1" }],
    quantity_assessment: {
      stable_clusters: [{ cluster_id: "qc_1" }],
    },
    unresolved_questions: ["why?"],
  };
  const empty = summarizeFusion(draft, {});
  assert.equal(empty.total, 5);
  assert.equal(empty.open, 5);
  assert.equal(empty.decided, 0);
  const decided = summarizeFusion(draft, {
    "step:eg_0": { action: "confirm" },
    "ingredient:ig_a": { action: "correct" },
    "quantity_cluster:qc_1": { action: "supply" },
  });
  assert.equal(decided.decided, 3);
  assert.equal(decided.verified, 1);
  assert.equal(decided.corrected, 1);
  assert.equal(decided.supplied, 1);
  assert.equal(decided.open, 2);
});

test("steps order by index and ingredients bucket by status", () => {
  const steps = orderSteps([
    { group_id: "b", step_index: 2 },
    { group_id: "a", step_index: 1 },
  ]);
  assert.deepEqual(
    steps.map((step) => step.group_id),
    ["a", "b"],
  );
  const buckets = bucketIngredients([
    { group_id: "u", identity_status: "unresolved", display_label: "z" },
    { group_id: "n", identity_status: "model_proposed", display_label: "a" },
    { group_id: "d", identity_status: "described", display_label: "m" },
  ]);
  assert.deepEqual(Object.keys(buckets), [
    "model_proposed",
    "described",
    "unresolved",
  ]);
  assert.equal(buckets.model_proposed[0].group_id, "n");
});

test("session clips order and filter steps", () => {
  const draft = {
    session: {
      session_id: "bakery_demo_v1",
      inter_clip_gap: "unknown",
      clips: [
        { clip_id: "clip_02_mixing", order: 2 },
        { clip_id: "clip_01_preparation", order: 1 },
      ],
    },
  };
  const clips = clipsOf(draft);
  assert.deepEqual(
    clips.map((clip) => clip.clip_id),
    ["clip_01_preparation", "clip_02_mixing"],
  );
  assert.equal(clipLabel(clips[0]), "Clip 1 · preparation");
  assert.deepEqual(clipsOf({}), []);
  const steps = [
    { group_id: "a", clip_id: "clip_01_preparation" },
    { group_id: "b", clip_id: "clip_02_mixing" },
    { group_id: "c", clip_id: null },
  ];
  assert.deepEqual(
    stepsForClip(steps, "clip_01_preparation").map(
      (step) => step.group_id,
    ),
    ["a"],
  );
  assert.deepEqual(
    stepsForClip(steps, null).map((step) => step.group_id),
    ["c"],
  );
});

test("draft node lists use collision-proof React keys", () => {
  // P04 v6 carries one ingredient group id repeated 44 times and
  // must stay byte-identical, so rendering keys (not review
  // targets) carry a list-position suffix. Review decisions still
  // key by bare group id.
  const src = fs.readFileSync(
    path.join(process.cwd(), "src/FusionReview.tsx"),
    "utf-8",
  );
  for (const key of [
    "step.group_id}#${stepPosition}",
    "cluster.cluster_id}#${clusterPosition}",
    "entry.group_id}#${rowPosition}",
    "conflict.conflict_id}#${conflictPosition}",
    'member.source_ref ?? "member"}#${memberPosition}',
  ]) {
    assert.ok(
      src.includes(key),
      `missing position-suffixed key: ${key}`,
    );
  }
  assert.ok(
    !src.includes("key={step.group_id}") &&
      !src.includes("key={entry.group_id}") &&
      !src.includes("key={cluster.cluster_id}"),
    "bare draft-id React key still present",
  );
});

test("session switch holds loading until the new bundle lands", () => {
  // Without the guard the stale bundle renders once with the new
  // run name, and the video element requests a URL that 404s.
  const src = fs.readFileSync(
    path.join(process.cwd(), "src/FusionReview.tsx"),
    "utf-8",
  );
  assert.ok(
    src.includes("bundle.run_name !== runName"),
    "stale-bundle loading guard is missing",
  );
  assert.ok(
    !src.includes("Loading the P04 fusion draft"),
    "loading text still hardcodes the P04 run",
  );
});
