// Analysis Library estimated-cost tests (plain Node).

import assert from "node:assert/strict";
import test from "node:test";

import { estimatedCostUsd, formatEstimatedCost } from "../src/cost.ts";

const priced = {
  billable: true,
  cost_status: "estimated_from_configured_rates",
  cost_usd_estimated: 0.54,
  input_tokens: 148563,
  output_tokens: 83362,
  budget: { cost_unknown_requests: 0, spent_cost_usd_estimate: 0.54 },
};

test("available pricing renders the recorded estimate", () => {
  assert.equal(estimatedCostUsd(priced), 0.54);
  assert.equal(
    formatEstimatedCost(priced),
    "Estimated inference cost: US$0.54");
});

test("unavailable pricing never falls back to zero", () => {
  assert.equal(formatEstimatedCost(null),
    "Estimated inference cost: unavailable");
  assert.equal(formatEstimatedCost({
    ...priced, cost_status: "COST_UNKNOWN", cost_usd_estimated: null,
  }), "Estimated inference cost: unavailable");
  assert.equal(formatEstimatedCost({
    ...priced, cost_usd_estimated: 0,
  }), "Estimated inference cost: unavailable");
});

test("incomplete historical usage stays unavailable", () => {
  assert.equal(formatEstimatedCost({
    ...priced, budget: { cost_unknown_requests: 2 },
  }), "Estimated inference cost: unavailable");
  assert.equal(formatEstimatedCost({
    ...priced, input_tokens: null,
  }), "Estimated inference cost: unavailable");
  assert.equal(formatEstimatedCost({
    billable: false, cost_status: "estimated_from_configured_rates",
    cost_usd_estimated: 1.5, input_tokens: 10, output_tokens: 10,
  }), "Estimated inference cost: unavailable");
});
