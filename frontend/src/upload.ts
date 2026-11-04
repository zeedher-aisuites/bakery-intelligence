// Pure helpers for the real-video upload workflow (Phase 3).

export const META_MODEL = "muse-spark-1.3";
export const META_WORKFLOW = "video_pipeline_multimodal";
export const OFFLINE_WORKFLOW = "video_pipeline_offline";
export const TOKEN_BUDGET = 4096;
// Extra Meta calls beyond one-per-frame: global, speech
// claims, validation (3) plus one transcribe call per
// 330 s audio chunk. Mirrors non_frame_reserve in
// multimodal.py. Frames approximate seconds at 1 FPS.
export const ASR_CHUNK_SECONDS = 330;
export const NON_FRAME_BASE_CALLS = 3;

export function firstpassRequests(
  frames: number, engine: string,
): number {
  if (!Number.isFinite(frames) || frames <= 0) return 0;
  if (engine === FIRSTPASS_ENGINE_SCALE
    || engine === FIRSTPASS_ENGINE_BATCH) {
    return Math.ceil(frames / 10);
  }
  if (engine === FIRSTPASS_ENGINE_FOCUSED) {
    return Math.ceil(frames / 50);
  }
  return frames;
}

export function scaleWorstCaseUsd(batches: number): number {
  return batches * (
    (SCALE_INPUT_TOKEN_BOUND / 1e6) * RATES_USD_PER_M[0]
    + (SCALE_OUTPUT_TOKEN_BUDGET / 1e6) * RATES_USD_PER_M[1]);
}

export function extraMetaRequests(frames: number): number {
  const chunks = Math.max(
    1, Math.ceil(Math.max(1, Math.floor(frames)) / ASR_CHUNK_SECONDS));
  return NON_FRAME_BASE_CALLS + chunks;
}
export const MAX_AUTHORIZED_COST_USD = 20;
export const RATES_USD_PER_M: [number, number] = [1.25, 4.25];
export const FIRSTPASS_ENGINE_SINGLE = "single_frame";
export const FIRSTPASS_ENGINE_SCALE = "parallel_scale";
export const FIRSTPASS_ENGINE_BATCH = "contextual_batch";
export const FIRSTPASS_ENGINE_FOCUSED = "focused_batch";
// Worst-case input bound per scale batch (10 anchored
// images + prompt), matching the backend reservation.
export const SCALE_INPUT_TOKEN_BOUND = 9000;
export const SCALE_OUTPUT_TOKEN_BUDGET = 8192;

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
  firstpass_engine: string;
}

export function confirmationSentence(
  requests: number,
  frames: number,
  operator: string,
): string {
  return (
    `authorize ${META_WORKFLOW} model ${META_MODEL} `
    + `up to ${requests} requests, ${TOKEN_BUDGET} output tokens, `
    + `${frames} frames, operator ${operator}`
  );
}

export function buildMetaGrant(
  frames: number,
  operator = "web-user",
  now: Date = new Date(),
  engine: string = FIRSTPASS_ENGINE_SINGLE,
): MetaGrant {
  const safeFrames = Math.max(1, Math.floor(frames));
  const maxRequests = firstpassRequests(safeFrames, engine)
    + extraMetaRequests(safeFrames);
  return {
    grant_format: "bakery-billable-grant/1",
    workflow: META_WORKFLOW,
    provider: "meta",
    model: META_MODEL,
    max_requests: maxRequests,
    token_budget: TOKEN_BUDGET,
    max_frames: safeFrames,
    operator,
    created_at: now.toISOString(),
    confirmation: confirmationSentence(
      maxRequests, safeFrames, operator),
    timeout_seconds: 120,
    store: true,
    cost_ceiling_usd: MAX_AUTHORIZED_COST_USD,
    rates_usd_per_m: RATES_USD_PER_M,
    strict_cost: true,
    firstpass_engine: engine,
  };
}
