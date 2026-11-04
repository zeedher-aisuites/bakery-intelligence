// Pure review-workspace logic for the evidence review UI.
// Shapes mirror the backend review bundle exactly; helpers never
// invent values for absent fields. Importable from plain Node
// tests (no React, no DOM).

export type ReviewDecision =
  | "confirmed"
  | "corrected"
  | "rejected"
  | "uncertain";

export interface ReviewAnnotation {
  frame_index: number;
  decision: ReviewDecision;
  value: number | null;
  notes: string | null;
  reviewer: string | null;
  updated_at: string;
  schema_version: string;
}

export interface ReviewModelFields {
  visible_objects: unknown;
  scale_display_verbatim: string | null;
  scale_readable: boolean | null;
  scale_state: string | null;
  tare_action_visible: boolean | null;
  addition_or_removal_visible: string | null;
  container_description: string | null;
  container_changed_hint: boolean | null;
  confidence: string | null;
  ambiguity_notes: string | null;
}

export interface ReviewObservation {
  frame_index: number;
  timestamp_seconds: number | null;
  terminal_state: string | null;
  parse_error: string | null;
  response_status: string | null;
  model_fields: ReviewModelFields;
  request_id: string | null;
  latency_seconds: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  reasoning_output_tokens: number | null;
}

export interface ReviewMeasurement {
  measurement_id: string | null;
  value: number | null;
  value_source: string | null;
  value_status: string | null;
  unit_observed: string | null;
}

export interface ReviewFrame {
  frame_index: number;
  timestamp_seconds: number | null;
  image_available: boolean;
  image_name: string | null;
  image_sha256: string | null;
  image_matches_record: boolean | null;
  observation: ReviewObservation | null;
  measurement: ReviewMeasurement | null;
  review: ReviewAnnotation | null;
}

export interface ReviewBundle {
  run_id: string;
  workflow: string;
  status: string;
  source_object_id: string | null;
  source_sha256: string | null;
  frames: ReviewFrame[];
  reviews: Record<string, ReviewAnnotation>;
  missing: string[];
}

export type FrameKind = "numeric" | "abstained" | "missing";

export interface FrameState {
  kind: FrameKind;
  reviewed: boolean;
  label: string;
  pill: string;
}

/** Order frames chronologically (defensive copy, index order). */
export function orderFrames(frames: ReviewFrame[]): ReviewFrame[] {
  return [...frames].sort((a, b) => a.frame_index - b.frame_index);
}

/**
 * Classify one timeline frame. A frame counts as reviewed only
 * when a stored annotation exists; model output alone never does.
 */
export function frameState(frame: ReviewFrame): FrameState {
  const reviewed = frame.review !== null;
  const suffix = reviewed ? " · reviewed" : " · pending review";
  if (!frame.image_available || frame.observation === null) {
    return {
      kind: "missing",
      reviewed,
      label: `Evidence missing${suffix}`,
      pill: "pill-bad",
    };
  }
  const terminal = frame.observation.terminal_state;
  const value = frame.measurement?.value ?? null;
  const status = frame.measurement?.value_status ?? null;
  if (terminal === "valid_prediction" && status === "ok"
    && typeof value === "number") {
    return {
      kind: "numeric",
      reviewed,
      label: `Reading ${value}${suffix}`,
      pill: reviewed ? "pill-ok" : "pill-pending",
    };
  }
  return {
    kind: "abstained",
    reviewed,
    label: `Model abstained${suffix}`,
    pill: reviewed ? "pill-ok" : "pill-pending",
  };
}

export type ReadingProvenance =
  | "ai-extracted"
  | "human-verified"
  | "human-corrected"
  | "rejected"
  | "uncertain"
  | "unknown";

export interface EffectiveReading {
  value: number | null;
  provenance: ReadingProvenance;
  pill: string;
  note: string;
}

/**
 * Resolve the review-effective reading for one frame. Human
 * decisions override model output; rejection and uncertainty
 * clear the value instead of guessing.
 */
export function effectiveReading(frame: ReviewFrame): EffectiveReading {
  const review = frame.review;
  const modelValue = frame.measurement?.value ?? null;
  const modelOk = frame.measurement?.value_status === "ok"
    && typeof modelValue === "number";
  if (review === null) {
    if (modelOk) {
      return {
        value: modelValue,
        provenance: "ai-extracted",
        pill: "pill-pending",
        note: "Awaiting human review.",
      };
    }
    return {
      value: null,
      provenance: "unknown",
      pill: "pill-pending",
      note: "No usable model reading.",
    };
  }
  switch (review.decision) {
    case "corrected":
      return {
        value: review.value,
        provenance: "human-corrected",
        pill: "pill-info",
        note: "Human-corrected value.",
      };
    case "confirmed":
      return {
        value: modelOk ? modelValue : null,
        provenance: "human-verified",
        pill: "pill-ok",
        note: modelOk
          ? "Human-verified model reading."
          : "Human agreed the frame is unreadable.",
      };
    case "rejected":
      return {
        value: null,
        provenance: "rejected",
        pill: "pill-bad",
        note: "Human rejected the model output.",
      };
    case "uncertain":
      return {
        value: null,
        provenance: "uncertain",
        pill: "pill-pending",
        note: "Human marked this frame uncertain.",
      };
    default:
      return {
        value: null,
        provenance: "unknown",
        pill: "pill-pending",
        note: "Stored review has an unrecognized decision.",
      };
  }
}

/**
 * Evidence-state badge for a reading provenance. Mirrors the
 * Confirm/Correct distinction in effectiveReading: only a
 * confirmation is verification. Rejected and uncertain frames
 * keep their existing pill treatment (null = no badge).
 */
export function readingBadge(
  provenance: ReadingProvenance,
): string | null {
  switch (provenance) {
    case "human-verified": return "badge badge-verified";
    case "human-corrected": return "badge badge-corrected";
    case "ai-extracted": return "badge badge-ai";
    case "unknown": return "badge badge-unknown";
    default: return null;
  }
}

export interface RecipeRow {
  frame_index: number;
  timestamp_seconds: number | null;
  value: number | null;
  provenance: ReadingProvenance;
  pill: string;
  note: string;
  verbatim: string | null;
  unit: string | null;
}

/** Build honest recipe rows: reviewed values win, gaps stay gaps. */
export function buildRecipeRows(bundle: ReviewBundle): RecipeRow[] {
  return orderFrames(bundle.frames).map((frame) => {
    const reading = effectiveReading(frame);
    return {
      frame_index: frame.frame_index,
      timestamp_seconds: frame.timestamp_seconds,
      value: reading.value,
      provenance: reading.provenance,
      pill: reading.pill,
      note: reading.note,
      verbatim: frame.observation?.model_fields
        .scale_display_verbatim ?? null,
      unit: frame.measurement?.unit_observed ?? null,
    };
  });
}

export interface ReviewCounts {
  frames: number;
  reviewed: number;
  pending: number;
  numeric: number;
  abstained: number;
  missing: number;
}

/** Count timeline and review states across the bundle. */
export function summarizeReview(bundle: ReviewBundle): ReviewCounts {
  const counts: ReviewCounts = {
    frames: bundle.frames.length,
    reviewed: 0,
    pending: 0,
    numeric: 0,
    abstained: 0,
    missing: 0,
  };
  for (const frame of bundle.frames) {
    const state = frameState(frame);
    counts[state.kind] += 1;
    if (state.reviewed) counts.reviewed += 1;
    else counts.pending += 1;
  }
  return counts;
}

function csvCell(value: unknown): string {
  if (value === null || value === undefined) return "";
  const text = String(value);
  if (/[",\n]/.test(text)) return `"${text.replace(/"/g, '""')}"`;
  return text;
}

/** Export model values alongside human corrections as CSV. */
export function exportReviewCsv(bundle: ReviewBundle): string {
  const header = [
    "frame_index", "timestamp_seconds", "terminal_state",
    "model_verbatim", "model_value", "model_status",
    "decision", "review_value", "reviewer", "review_notes",
    "effective_value", "effective_provenance",
  ];
  const lines = [header.join(",")];
  for (const frame of orderFrames(bundle.frames)) {
    const reading = effectiveReading(frame);
    lines.push([
      frame.frame_index, frame.timestamp_seconds ?? "",
      frame.observation?.terminal_state ?? "",
      frame.observation?.model_fields.scale_display_verbatim ?? "",
      frame.measurement?.value ?? "",
      frame.measurement?.value_status ?? "",
      frame.review?.decision ?? "", frame.review?.value ?? "",
      frame.review?.reviewer ?? "", frame.review?.notes ?? "",
      reading.value ?? "", reading.provenance,
    ].map(csvCell).join(","));
  }
  return lines.join("\n") + "\n";
}

/** Export the full review record as JSON. */
export function exportReviewJson(bundle: ReviewBundle): string {
  const rows = orderFrames(bundle.frames).map((frame) => ({
    frame_index: frame.frame_index,
    timestamp_seconds: frame.timestamp_seconds,
    terminal_state: frame.observation?.terminal_state ?? null,
    model_verbatim: frame.observation?.model_fields
      .scale_display_verbatim ?? null,
    model_value: frame.measurement?.value ?? null,
    model_status: frame.measurement?.value_status ?? null,
    review: frame.review,
    effective_value: effectiveReading(frame).value,
    effective_provenance: effectiveReading(frame).provenance,
  }));
  return JSON.stringify({
    run_id: bundle.run_id,
    workflow: bundle.workflow,
    exported_decisions: rows.filter(
      (row) => row.review !== null).length,
    rows,
  }, null, 2) + "\n";
}
