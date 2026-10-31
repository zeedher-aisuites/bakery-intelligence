// Artifact contracts and pure presentation helpers for the
// offline reconstruction review UI (M2-06B.2). Shapes mirror the
// backend artifacts exactly; helpers never invent values for
// absent fields.

export interface OfflineSummary {
  workflow: string;
  run_id: string;
  video_id: string;
  event_id: string;
  bundle_format: string;
  bundle_sha256: string;
  source_object_id: string;
  inference_performed: boolean;
  network_accessed: boolean;
  paid_services_used: boolean;
  draft_id: string;
  measurements: number;
  actions: number;
  unresolved: string[];
  issues: number;
  artifacts: string[];
  note: string;
}

export interface DraftStep {
  index: number;
  description: string;
  evidence_summary: string;
  observation_ids: string[];
  action_ids: string[];
}

export interface UnresolvedQuestion {
  code: string;
  message: string;
  related_entity_ids: string[];
}

export interface RecipeDraft {
  draft_id: string;
  version: number;
  video_id: string;
  run_id: string;
  status: string;
  steps: DraftStep[];
  ingredients: unknown[];
  unresolved_questions: UnresolvedQuestion[];
  evidence_refs: {
    frame_evidence_ids: string[];
    observation_ids: string[];
  };
  verification_ref: string;
  schema_version: string;
}

export interface MeasurementOriginal {
  verbatim: string | null;
  timestamp_seconds: number;
  readable: boolean;
  scale_state: string;
  confidence: string;
  tare_visible: boolean;
  action_visible: string | null;
  batch_id: string;
}

export interface VerifiedReading {
  display_text: string;
  numeric_value: number;
  readable: boolean;
  confidence: string;
  uncertainty_notes: string | null;
}

export interface Measurement {
  measurement_id: string;
  frame_index: number;
  timestamp_seconds: number;
  value: number | null;
  value_source: string;
  value_status: string;
  delta_from_previous: number | null;
  unit_observed: string | null;
  original: MeasurementOriginal;
  verified: VerifiedReading | null;
}

export interface UnitResolution {
  measurement_id: string;
  unit: string | null;
  provenance: string;
  assignment_id: string | null;
}

export interface TemporalAction {
  event_id: string;
  candidate_type: string;
  start_timestamp_seconds: number;
  end_timestamp_seconds: number;
  verification_status: string;
  frame_indices: number[];
}

export interface MeasurementBundle {
  event_id: string;
  window: { start_seconds: number; end_seconds: number };
  measurements: Measurement[];
  unit_resolutions: UnitResolution[];
  action_evidence: {
    addition_frames: number[];
    removal_frames: number[];
    tare_frames: number[];
    temporal_actions: TemporalAction[];
  };
}

export interface ValidationIssue {
  code: string;
  entity_id: string;
  gate: string;
  message: string;
  severity: string;
}

export interface ValidationResult {
  issue_count: number;
  issues: ValidationIssue[];
}

export interface WeighingEvent {
  event_id: string;
  start_seconds: number;
  end_seconds: number;
  observation_ids: string[];
  action_ids: string[];
  final_display_reading: {
    observation_id: string | null;
    value: number | null;
    unit: string | null;
  };
  net_ingredient_quantity: {
    value: number | null;
    unit: string | null;
    derivation: string;
  };
  ingredient_id: string | null;
  measurement_mode: string;
  tare_observed: boolean;
  review_notes: string;
  verification_status: string;
  schema_version: string;
}

export interface SourceChip {
  label: string;
  pill: string;
}

/** Map a measurement value_source to an honest display chip. */
export function sourceChip(source: string): SourceChip {
  switch (source) {
    case "verification":
      return { label: "Verified reading", pill: "pill-ok" };
    case "firstpass":
      return { label: "First-pass model reading", pill: "pill-pending" };
    case "verification_unusable":
      return { label: "Verified but unusable", pill: "pill-bad" };
    case "firstpass_unusable":
      return { label: "First-pass unusable", pill: "pill-bad" };
    default:
      return { label: source, pill: "pill-pending" };
  }
}

/** Map a value_status to a pill class; unknown statuses stay neutral. */
export function statusPill(status: string): string {
  switch (status) {
    case "ok":
      return "pill-ok";
    case "ambiguous":
      return "pill-pending";
    case "invalid":
      return "pill-bad";
    default:
      return "pill-pending";
  }
}

export interface UnitLabel {
  text: string;
  pill: string;
}

/**
 * Describe the unit applicable to one measurement. Observed units
 * win over human assignments; absence is shown as undetermined,
 * never guessed.
 */
export function unitLabel(
  observed: string | null,
  resolution: UnitResolution | undefined,
): UnitLabel {
  if (observed) return { text: `${observed} (observed)`, pill: "pill-ok" };
  if (resolution?.unit) {
    const scope =
      resolution.assignment_id !== null
        ? `human-assigned ${resolution.assignment_id}`
        : "human-assigned";
    return { text: `${resolution.unit} (${scope})`, pill: "pill-pending" };
  }
  return { text: "undetermined", pill: "pill-pending" };
}

/** Render free text or a placeholder when absent. */
export function text(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  return String(value);
}

/** Render a numeric value or a placeholder when absent. */
export function numeric(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return String(value);
}

/** Render seconds with one decimal place. */
export function seconds(value: number): string {
  return `${value.toFixed(1)} s`;
}
