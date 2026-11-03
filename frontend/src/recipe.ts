// Pure helpers for the chef recipe-review workflow.

export const PROVENANCES = [
  "ai-extracted", "human-corrected", "chef-supplied",
] as const;

export interface RecipeIngredient {
  ingredient_id: string;
  name?: string;
  quantity?: { value: number | null; unit: string | null; derivation?: string };
  provenance: string;
}

export interface RecipeStep {
  index: number;
  description: string;
  provenance: string;
}

export function provenanceLabel(provenance: string): string {
  if (provenance === "chef-supplied") return "Chef supplied (not observed)";
  if (provenance === "human-corrected") return "Human corrected";
  return "AI extracted";
}

export function canDecide(reviewer: string): boolean {
  return reviewer.trim().length > 0;
}

export function approvalBlocked(verdict: string | undefined): boolean {
  return verdict === "fail";
}

export function loadChefIdentity(): string {
  try {
    return localStorage.getItem("bakery.chefName") ?? "";
  } catch {
    return "";
  }
}

export function saveChefIdentity(name: string): void {
  try {
    localStorage.setItem("bakery.chefName", name);
  } catch {
    // Storage unavailable; the field still works for this session.
  }
}

export function nextVersionLabel(version: number): string {
  return `Draft v${version}`;
}

export function formatVideoTime(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined
    || !Number.isFinite(seconds) || seconds < 0) return "unknown time";
  const total = Math.round(seconds);
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  return `${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
}

export function parseVideoTime(text: string): number | null {
  const trimmed = text.trim();
  if (!trimmed) return null;
  const match = /^(\d+):([0-5]\d)$/.exec(trimmed);
  if (match) return Number(match[1]) * 60 + Number(match[2]);
  const plain = Number(trimmed);
  return Number.isFinite(plain) && plain >= 0 ? plain : null;
}

export function intervalLabel(
  start: number | null | undefined,
  end: number | null | undefined,
): string {
  if (start === null || start === undefined) return "time not recorded";
  if (end === null || end === undefined || end === start) {
    return formatVideoTime(start);
  }
  return `${formatVideoTime(start)}–${formatVideoTime(end)}`;
}

export interface TemporalStep {
  step_id: string;
  frame_refs: Array<{ frame_index: number }>;
  ingredient_refs: string[];
}

export function stepsReferencingFrame(
  steps: TemporalStep[],
  frameIndex: number,
): TemporalStep[] {
  return steps.filter((step) => step.frame_refs.some(
    (ref) => ref.frame_index === frameIndex));
}

export function linkFrameToStep<T extends TemporalStep>(
  step: T,
  frameIndex: number,
): T {
  if (step.frame_refs.some((ref) => ref.frame_index === frameIndex)) {
    return step;
  }
  return {
    ...step,
    frame_refs: [...step.frame_refs, { frame_index: frameIndex }],
  };
}

export function evidenceScopeLabel(claim: {
  evidence_scope?: string;
  evidence_total?: number;
  evidence_refs: unknown[];
}): string {
  if (claim.evidence_scope === "representative") {
    return `Representative evidence (${claim.evidence_refs.length} of `
      + `${claim.evidence_total ?? "?"} frames)`;
  }
  return `Supporting evidence (${claim.evidence_refs.length} frames)`;
}
