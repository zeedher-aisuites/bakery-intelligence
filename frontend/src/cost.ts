// Estimated inference cost display for completed analyses.
//
// The amount comes from the recorded first-pass report, which the
// backend computed with its existing token accounting and the
// rates configured for that run. No pricing is recalculated here.

export interface FirstpassCostReport {
  billable?: boolean;
  cost_status?: string | null;
  cost_usd_estimated?: number | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  budget?: {
    cost_unknown_requests?: number | null;
    spent_cost_usd_estimate?: number | null;
  } | null;
  cost?: {
    spent_cost_usd_estimate?: number | null;
  } | null;
}

export function estimatedCostUsd(
  report: FirstpassCostReport | null | undefined,
): number | null {
  if (!report) return null;
  if (report.billable !== true) return null;
  if (report.cost_status !== "estimated_from_configured_rates") {
    return null;
  }
  const unknown = report.budget?.cost_unknown_requests;
  if (typeof unknown === "number" && unknown > 0) return null;
  if (typeof report.input_tokens !== "number"
    || typeof report.output_tokens !== "number") {
    return null;
  }
  const value = report.cost_usd_estimated
    ?? report.budget?.spent_cost_usd_estimate
    ?? report.cost?.spent_cost_usd_estimate;
  if (typeof value !== "number" || !Number.isFinite(value)
    || value <= 0) {
    return null;
  }
  return value;
}

export function formatEstimatedCost(
  report: FirstpassCostReport | null | undefined,
  settled = true,
): string {
  if (!settled) return "Estimated inference cost: —";
  const value = estimatedCostUsd(report);
  if (value === null) return "Estimated inference cost: unavailable";
  return `Estimated inference cost: US$${value.toFixed(2)}`;
}
