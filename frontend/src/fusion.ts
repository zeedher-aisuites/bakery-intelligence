// Pure fusion-review logic for the chef review workspace.
// Shapes mirror the backend fusion draft exactly; helpers never
// invent values for absent fields. Importable from plain Node
// tests (no React, no DOM).

export type FusionTargetType =
  | "ingredient"
  | "quantity_cluster"
  | "step"
  | "conflict"
  | "question";

export type FusionAction =
  | "confirm"
  | "correct"
  | "supply"
  | "cannot_determine"
  | "reject";

export interface FusionDecision {
  target_type: FusionTargetType;
  target_id: string;
  action: FusionAction;
  value: Record<string, unknown> | null;
  reviewer: string;
  notes: string | null;
  updated_at: string;
}

export type FusionProvenance =
  | "ai-extracted"
  | "chef-supplied"
  | "human-corrected"
  | "human-verified"
  | "unknown";

export interface FusionStepMember {
  member_id: string | null;
  source: string | null;
  source_ref: string | null;
  start_seconds: number | null;
  end_seconds: number | null;
  action: string | null;
}

export interface FusionStep {
  step_index: number;
  group_id: string;
  clip_id?: string | null;
  start_seconds: number | null;
  end_seconds: number | null;
  actions: string[];
  ingredients: string[];
  sources: string[];
  source_count: number;
  member_ids: string[];
  members: FusionStepMember[];
  preliminary_links: string[];
  speech_links?: string[];
  uncertainties: string[];
  status: string;
}

export interface FusionSpeechTurn {
  speech_id: string;
  clip_id: string | null;
  turn_index: number;
  speaker: string | null;
  start_seconds: number | null;
  end_seconds: number | null;
  transcript: string;
  claims: Array<Record<string, unknown>>;
}

export interface FusionSessionClip {
  clip_id: string;
  order: number;
  duration_seconds: number | null;
  sha256?: string;
}

export interface FusionIngredientEntry {
  group_id: string;
  clip_id?: string | null;
  identity_status: string;
  display_label: string | null;
  member_count: number;
  names: string[];
  visible_labels: string[];
  used_or_visible: string[];
  quantity: number | null;
  unit: string | null;
  supporting_timestamps: number[];
  sources: string[];
  source_count: number;
  member_ids: string[];
  possibly_related: string[];
  ambiguities: string[];
  status: string;
}

export interface FusionCluster {
  cluster_id: string;
  clip_id?: string | null;
  displayed_value: string;
  start_seconds: number | null;
  end_seconds: number | null;
  reading_ids: string[];
  sources: string[];
  units: string[];
  derivation_status: string;
  missing_for_quantity: string[];
}

export interface FusionConflict {
  conflict_id: string;
  kind: string;
  detail: string;
  event_ids: string[];
  sources: string[];
}

export interface FusionDraft {
  schema: string;
  status: string;
  working_title: string | null;
  working_title_status: string;
  ingredients: FusionIngredientEntry[];
  ingredient_review: {
    counts: Record<string, number>;
    named_candidates: Array<Record<string, unknown>>;
    described_unidentified: Array<Record<string, unknown>>;
    unresolved: Array<Record<string, unknown>>;
  } | null;
  steps: FusionStep[];
  quantity_assessment: {
    status: string;
    stable_clusters: FusionCluster[];
    duplicates_folded: number;
    tare_claims: Array<Record<string, unknown>>;
  } | null;
  conflicts: FusionConflict[];
  corroborated_reading_pairs: Array<Record<string, unknown>>;
  unresolved_questions: string[];
  speech?: FusionSpeechTurn[];
  session?: {
    session_id: string;
    clips: FusionSessionClip[];
    inter_clip_gap: string;
  } | null;
  provenance: Record<string, unknown>;
}

/** Session clips in declared order (empty for single-video). */
export function clipsOf(draft: FusionDraft): FusionSessionClip[] {
  const clips = draft.session?.clips ?? [];
  return [...clips].sort((a, b) => a.order - b.order);
}

/** Steps for one clip; null clip keeps unscoped (P04) steps. */
export function stepsForClip(
  steps: FusionStep[],
  clipId: string | null,
): FusionStep[] {
  if (clipId === null) {
    return steps.filter((step) => !step.clip_id);
  }
  return steps.filter((step) => step.clip_id === clipId);
}

/** Friendly clip label: order + short id suffix. */
export function clipLabel(clip: FusionSessionClip): string {
  const suffix = clip.clip_id.replace(/^clip_\d+_/, "");
  return `Clip ${clip.order} · ${suffix || clip.clip_id}`;
}

/** Stable map key for one decision target. */
export function targetKey(
  targetType: FusionTargetType,
  targetId: string,
): string {
  return `${targetType}:${targetId}`;
}

/**
 * Resolve the effective provenance of one draft node. Human
 * decisions override AI extraction; a correction is never a
 * verification, and chef-supplied values keep their own badge.
 */
export function effectiveProvenance(
  decision: FusionDecision | null | undefined,
): FusionProvenance {
  if (!decision) return "ai-extracted";
  switch (decision.action) {
    case "confirm":
      return "human-verified";
    case "correct":
      return "human-corrected";
    case "supply":
      return "chef-supplied";
    case "cannot_determine":
    case "reject":
      return "unknown";
    default:
      return "unknown";
  }
}

/** Badge class for a fusion provenance (null = no badge). */
export function provenanceBadge(
  provenance: FusionProvenance,
): string | null {
  switch (provenance) {
    case "human-verified":
      return "badge badge-verified";
    case "human-corrected":
      return "badge badge-corrected";
    case "chef-supplied":
      return "badge badge-chef";
    case "ai-extracted":
      return "badge badge-ai";
    case "unknown":
      return "badge badge-unknown";
    default:
      return null;
  }
}

export function provenanceLabel(provenance: FusionProvenance): string {
  switch (provenance) {
    case "human-verified":
      return "Human verified";
    case "human-corrected":
      return "Human corrected";
    case "chef-supplied":
      return "Chef supplied";
    case "ai-extracted":
      return "AI extracted";
    case "unknown":
      return "Unknown";
    default:
      return "Unknown";
  }
}

/** Format seconds as m:ss; null stays null (never 0:00). */
export function formatTimestamp(value: number | null): string | null {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  if (value < 0) return null;
  const total = Math.floor(value);
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

/** Human interval label; open ends stay explicit. */
export function intervalLabel(
  start: number | null,
  end: number | null,
): string {
  const left = formatTimestamp(start);
  const right = formatTimestamp(end);
  if (left && right) return `${left}–${right}`;
  if (left) return `from ${left}`;
  if (right) return `until ${right}`;
  return "time unknown";
}

export function identityLabel(status: string): string {
  switch (status) {
    case "model_proposed":
      return "Model proposed";
    case "described":
      return "Described, unidentified";
    case "unresolved":
      return "Unresolved";
    default:
      return "Unknown";
  }
}

export function sourceLabel(source: string): string {
  if (source.includes("muse-spark-1.3-full-video")) return "Spark 1.3 · full video";
  if (source.includes("muse-spark-1.3-global-video")) return "Spark 1.3 · global";
  if (source.includes("muse-spark-1.3-local-video")) return "Spark 1.3 · local";
  if (source.includes("muse-spark-1.3-measurement")) return "Spark 1.3 · measure";
  if (source.includes("muse-spark-open")) return "Spark Open · segments";
  if (source.includes("preliminary")) return "Human notes";
  if (source.includes("voice-transcribe")) return "Voice transcript";
  return source;
}

export interface FusionProgress {
  total: number;
  decided: number;
  verified: number;
  corrected: number;
  supplied: number;
  open: number;
}

/** Count review progress across every reviewable draft node. */
export function summarizeFusion(
  draft: FusionDraft,
  decisions: Record<string, FusionDecision>,
): FusionProgress {
  const keys: string[] = [];
  for (const entry of draft.ingredients ?? []) {
    keys.push(targetKey("ingredient", entry.group_id));
  }
  for (const step of draft.steps ?? []) {
    keys.push(targetKey("step", step.group_id));
  }
  for (const conflict of draft.conflicts ?? []) {
    keys.push(targetKey("conflict", conflict.conflict_id));
  }
  for (const cluster of draft.quantity_assessment?.stable_clusters ?? []) {
    keys.push(targetKey("quantity_cluster", cluster.cluster_id));
  }
  (draft.unresolved_questions ?? []).forEach((_, index) => {
    keys.push(targetKey("question", `q${index}`));
  });
  const progress: FusionProgress = {
    total: keys.length,
    decided: 0,
    verified: 0,
    corrected: 0,
    supplied: 0,
    open: 0,
  };
  for (const key of keys) {
    const decision = decisions[key];
    if (!decision) {
      progress.open += 1;
      continue;
    }
    progress.decided += 1;
    const provenance = effectiveProvenance(decision);
    if (provenance === "human-verified") progress.verified += 1;
    else if (provenance === "human-corrected") progress.corrected += 1;
    else if (provenance === "chef-supplied") progress.supplied += 1;
  }
  return progress;
}

/** Steps ordered by index for timeline and list rendering. */
export function orderSteps(steps: FusionStep[]): FusionStep[] {
  return [...steps].sort((a, b) => a.step_index - b.step_index);
}

/** Entries grouped by identity status for the review list. */
export function bucketIngredients(
  entries: FusionIngredientEntry[],
): Record<string, FusionIngredientEntry[]> {
  const buckets: Record<string, FusionIngredientEntry[]> = {
    model_proposed: [],
    described: [],
    unresolved: [],
  };
  for (const entry of entries) {
    const bucket = buckets[entry.identity_status] ?? buckets.unresolved;
    bucket.push(entry);
  }
  for (const rows of Object.values(buckets)) {
    rows.sort((a, b) => (a.display_label ?? "").localeCompare(
      b.display_label ?? ""));
  }
  return buckets;
}
