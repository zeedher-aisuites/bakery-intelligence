// Pure helpers for exception-driven Observation Review.

export interface TaskLike {
  task_id: string;
  state: string;
  status: string;
  frame_refs: number[];
}

export function openTasks<T extends TaskLike>(tasks: T[]): T[] {
  return tasks.filter((task) => task.status === "open");
}

export function reviewProgress(tasks: TaskLike[]): {
  total: number; resolved: number; open: number; deferred: number;
} {
  return {
    total: tasks.length,
    resolved: tasks.filter((task) => task.status === "resolved").length,
    open: openTasks(tasks).length,
    deferred: tasks.filter((task) => task.status === "deferred").length,
  };
}

export function nextTaskId<T extends TaskLike>(
  tasks: T[], current: string | null, showAll: boolean,
): string | null {
  const visible = showAll ? tasks : tasks.filter(
    (task) => task.status === "open" || task.status === "deferred");
  if (current && visible.some((task) => task.task_id === current)) {
    return current;
  }
  return visible.length > 0 ? visible[0].task_id : null;
}

export function taskLabel(state: string): string {
  switch (state) {
    case "missing_relevant": return "Missing measurement";
    case "contradictory": return "Conflicting measurements";
    case "uncertain": return "Uncertain measurement";
    case "flagged_for_review": return "Flagged from viewer";
    case "ingredient_uncertain": return "Uncertain ingredient";
    case "action_conflicting": return "Conflicting actions";
    case "transition_uncertain": return "Uncertain transition";
    case "unit_missing": return "Missing unit";
    case "tare_unresolved": return "Unresolved tare";
    default: return state;
  }
}

export function validationLabel(state: string): string {
  switch (state) {
    case "accepted": return "Passed automatic checks";
    case "missing_relevant": return "Needs review";
    case "contradictory": return "Conflicting measurements";
    case "uncertain": return "Needs review";
    case "flagged_for_review": return "Needs review";
    case "ingredient_uncertain": return "Needs review";
    case "action_conflicting": return "Conflicting measurements";
    case "transition_uncertain": return "Needs review";
    case "unit_missing": return "Needs review";
    case "tare_unresolved": return "Needs review";
    case "not_observable": return "Not relevant";
    case "not_evaluated": return "Not relevant";
    default: return "Needs review";
  }
}

export function modelLabel(value: number | null | undefined): string {
  return value === null || value === undefined
    ? "No reading predicted" : "Reading predicted";
}

export function humanLabel(status: string | undefined): string {
  switch (status) {
    case "resolved": return "Reviewed";
    case "deferred": return "Review later";
    default: return "Not reviewed";
  }
}

export interface FrameFilterState {
  model: string;
  validation: string;
  human: string;
  confidence: string;
  inTasks: boolean;
}

export const DEFAULT_FILTERS: FrameFilterState = {
  model: "all", validation: "all", human: "all",
  confidence: "all", inTasks: false,
};

export interface FilterableFrame {
  frame_index: number;
  measurement?: { value: number | null | undefined } | null;
  observation?: {
    model_fields?: {
      confidence?: string | null;
    } | null;
  } | null;
  review?: unknown;
}

export function confidenceOf(frame: FilterableFrame): string {
  const confidence = frame.observation?.model_fields?.confidence;
  return typeof confidence === "string" ? confidence : "not reported";
}

export function applyFrameFilters<
  T extends FilterableFrame & {
    measurement?: { value_status?: string | null } | null;
  },
>(
  frames: T[],
  validationOf: (frame: T) => string,
  humanOf: (frame: T) => string,
  taskFrames: Set<number>,
  filters: FrameFilterState,
): T[] {
  return frames.filter((frame) => {
    const value = frame.measurement?.value ?? null;
    if (filters.model === "reading" && value === null) return false;
    if (filters.model === "abstained" && value !== null) return false;
    if (filters.model === "not_evaluated"
      && frame.measurement?.value_status !== undefined
      && frame.measurement?.value_status !== "not_evaluated") {
      return false;
    }
    const validation = validationOf(frame);
    if (filters.validation !== "all" && validation !== filters.validation) {
      return false;
    }
    if (filters.human !== "all" && humanOf(frame) !== filters.human) {
      return false;
    }
    if (filters.confidence !== "all"
      && confidenceOf(frame) !== filters.confidence) {
      return false;
    }
    if (filters.inTasks && !taskFrames.has(frame.frame_index)) {
      return false;
    }
    return true;
  });
}

export const FILTER_SUMMARIES: Record<string, string> = {
  passed: "Passed structural checks only: parseable value on a readable display, consistent with neighbors. Not a claim of visual correctness.",
  limited: "Accepted with medium or low model confidence. Flagged for optional inspection, not required review.",
};

export interface CorrectionRecord {
  action?: string;
  human_value?: number | null;
  reviewer?: string | null;
  revision?: number;
}

export interface CorrectionTask {
  task_id: string;
  frame_refs: number[];
  status: string;
  correction?: CorrectionRecord | null;
}

function finiteOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** Distinct model values behind a finding, for exact-scope display. */
export function taskModelValues(
  task: CorrectionTask,
  lookup: (index: number) => number | null,
): number[] {
  const found = task.frame_refs.map(lookup).filter(
    (value): value is number => typeof value === "number");
  return [...new Set(found)];
}

function modelList(values: number[]): string {
  if (values.length === 0) return "no model reading";
  return values.map((value) => String(value)).join(" · ");
}

/** Chef-facing status that never conflates distinct decision types. */
export function correctionStatus(task: CorrectionTask): string {
  const action = task.correction?.action;
  if (!action) {
    return task.status === "deferred" ? "Set aside" : "Needs decision";
  }
  switch (action) {
    case "correct":
    case "supply":
      return finiteOrNull(task.correction?.human_value) === null
        ? "Corrected" : `Corrected to ${task.correction?.human_value}`;
    case "confirm":
      return "Confirmed";
    case "not_identifiable":
    case "skip_unknown":
      return "Cannot determine";
    case "not_applicable":
      return "Not applicable";
    case "review_later":
      return "Set aside";
    default:
      return "Reviewed";
  }
}

/** Exact persisted result: original prediction plus effective decision. */
export function correctionResult(
  task: CorrectionTask, values: number[],
): string | null {
  const action = task.correction?.action;
  if (!action || task.status !== "resolved") return null;
  const original = modelList(values);
  const human = finiteOrNull(task.correction?.human_value);
  if (action === "correct" || action === "supply") {
    if (human === null) {
      return `Corrected, but the earlier value was not recorded. `
        + `Model showed: ${original}.`;
    }
    return `Corrected from ${original} to ${human} (unit unknown).`;
  }
  if (action === "confirm") {
    return `AI reading confirmed: ${original}.`;
  }
  if (action === "not_identifiable" || action === "skip_unknown") {
    return "Marked as cannot determine — no usable reading established.";
  }
  if (action === "not_applicable") {
    return "Marked as not applicable to this recipe.";
  }
  return `Reviewed (${action}).`;
}

/** Effective numeric value of a finding-level correction, if any. */
export function correctionEffectiveValue(
  task: CorrectionTask, values: number[],
): number | null {
  const action = task.correction?.action;
  if (action === "correct" || action === "supply") {
    return finiteOrNull(task.correction?.human_value);
  }
  if (action === "confirm") {
    return values.length === 1 ? values[0] : null;
  }
  return null;
}
