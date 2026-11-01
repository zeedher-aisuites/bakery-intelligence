// Pure helpers for the real-video upload workflow (Phase 3).

export const META_MODEL = "muse-spark-1.3";
export const META_WORKFLOW = "video_pipeline_meta";
export const OFFLINE_WORKFLOW = "video_pipeline_offline";
export const TOKEN_BUDGET = 4096;
export const MAX_AUTHORIZED_COST_USD = 20;
export const RATES_USD_PER_M: [number, number] = [1.25, 4.25];

export function estimateFrames(durationSeconds: number): number {
  if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) return 0;
  return Math.ceil(durationSeconds);
}

export function formatDuration(totalSeconds: number): string {
  if (!Number.isFinite(totalSeconds) || totalSeconds < 0) return "—";
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.round(totalSeconds % 60);
  return minutes > 0 ? `${minutes}m ${seconds}s` : `${seconds}s`;
}

export interface MetaGrant {
  grant_format: string;
  workflow: string;
  provider: string;
  model: string;
  max_requests: number;
  token_budget: number;
  max_frames: number;
  operator: string;
  created_at: string;
  confirmation: string;
  timeout_seconds: number;
  store: boolean;
  cost_ceiling_usd: number | null;
  rates_usd_per_m: [number, number];
  strict_cost: boolean;
}

export function confirmationSentence(
  frames: number,
  operator: string,
): string {
  return (
    `authorize ${META_WORKFLOW} model ${META_MODEL} `
    + `up to ${frames} requests, ${TOKEN_BUDGET} output tokens, `
    + `${frames} frames, operator ${operator}`
  );
}

export function buildMetaGrant(
  frames: number,
  operator = "web-user",
  now: Date = new Date(),
): MetaGrant {
  const safeFrames = Math.max(1, Math.floor(frames));
  return {
    grant_format: "bakery-billable-grant/1",
    workflow: META_WORKFLOW,
    provider: "meta",
    model: META_MODEL,
    max_requests: safeFrames,
    token_budget: TOKEN_BUDGET,
    max_frames: safeFrames,
    operator,
    created_at: now.toISOString(),
    confirmation: confirmationSentence(safeFrames, operator),
    timeout_seconds: 120,
    store: true,
    cost_ceiling_usd: MAX_AUTHORIZED_COST_USD,
    rates_usd_per_m: RATES_USD_PER_M,
    strict_cost: true,
  };
}
