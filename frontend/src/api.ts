import type { ReviewAnnotation, ReviewBundle } from "./review";
import type {
  FusionAction,
  FusionDecision,
  FusionDraft,
  FusionTargetType,
} from "./fusion";

export const API_BASE =
  import.meta.env?.VITE_API_BASE ?? "http://localhost:8000";

export const WORKFLOW_SMOKE = "storage_smoke";
export const WORKFLOW_META = "video_pipeline_multimodal";
export const WORKFLOW_OFFLINE = "offline_reconstruct";
export const WORKFLOW_PIPELINE = "video_pipeline_offline";

export interface Health {
  status: string;
  service: string;
  version: string;
  storage_backend: string;
  storage_ready: boolean;
}

export interface AnalysisCreated {
  run_id: string;
  workflow: string;
  status: string;
  source_object_id: string;
}

export interface AnalysisStatus {
  run_id: string;
  workflow: string;
  status: string;
  source_object_id: string | null;
  source_sha256: string | null;
  artifact_ids: string[];
  error: string | null;
  created_at?: string;
  updated_at?: string | null;
  attempts?: number;
}

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function parseError(response: Response): Promise<string> {
  try {
    const payload = await response.json();
    if (typeof payload?.detail === "string") return payload.detail;
  } catch {
    /* fall through to status text */
  }
  return response.statusText || `Request failed (${response.status})`;
}

export async function fetchHealth(): Promise<Health> {
  const response = await fetch(`${API_BASE}/api/health`);
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as Health;
}

export async function createAnalysis(
  file: File,
  workflow: string,
): Promise<AnalysisCreated> {
  const form = new FormData();
  form.append("file", file, file.name);
  form.append("workflow", workflow);
  const response = await fetch(`${API_BASE}/api/analyses`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as AnalysisCreated;
}

export interface JobRecord extends AnalysisStatus {
  error_category?: string | null;
  claimed_by?: string | null;
}

export async function listJobs(
  status?: string,
  limit = 50,
): Promise<JobRecord[]> {
  const query = new URLSearchParams({ limit: String(limit) });
  if (status) query.set("status", status);
  const response = await fetch(`${API_BASE}/api/jobs?${query}`);
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as JobRecord[];
}

export async function retryJob(runId: string): Promise<JobRecord> {
  const response = await fetch(`${API_BASE}/api/jobs/${runId}/retry`, {
    method: "POST",
  });
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as JobRecord;
}

export interface RecipeClaim {
  claim_id: string;
  kind: string;
  title: string;
  ingredient_id: string | null;
  quantity: { value: number | null; unit: string | null; derivation: string };
  evidence_refs: Array<{ frame_index: number; measurement_id?: string | null }>;
  provenance: string;
  status: string;
  uncertainty: string | null;
  objections: string[];
  evidence_scope?: string;
  evidence_total?: number;
}

export interface RecipeReviewState {
  run_id: string;
  draft_id: string;
  version: number;
  status: string;
  simulation: boolean;
  simulation_note: string;
  ingredients: Array<Record<string, unknown>>;
  steps: Array<Record<string, unknown>>;
  claims: RecipeClaim[];
  verification: { verdict: string; issues: Array<{ code: string; entity_id: string; message: string }> };
  decisions: Array<Record<string, unknown>>;
  approved_snapshots: Array<Record<string, unknown>>;
  history: Array<Record<string, unknown>>;
  updated_at: string;
  generation?: Record<string, unknown> | null;
}

export interface EvidenceFrame {
  frame_index: number;
  timestamp_seconds: number | null;
  image_name: string | null;
  image_available: boolean;
  measurement?: { value: number | null; value_status?: string | null } | null;
}

export async function fetchRecipeEvidence(
  runId: string,
): Promise<{ frames: EvidenceFrame[] }> {
  const response = await fetch(`${API_BASE}/api/analyses/${runId}/recipe/evidence`);
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as { frames: EvidenceFrame[] };
}

export interface ReviewTask {
  task_id: string;
  field: string;
  state: string;
  frame_refs: number[];
  correction?: {
    action?: string;
    human_value?: number | null;
    reviewer?: string | null;
    revision?: number;
  } | null;
  from_viewer?: boolean;
  start_seconds: number | null;
  end_seconds: number | null;
  status: string;
}

export interface EvaluatedObservation {
  frame_index: number;
  timestamp_seconds: number | null;
  field: string;
  state: string;
  value: number | null;
  limited_confidence?: boolean;
}

export interface ObservationQueue {
  evaluated: EvaluatedObservation[];
  tasks: ReviewTask[];
  counts: {
    accepted: number; actionable: number; tasks: number;
    resolved: number; not_evaluated: number;
  };
}

export async function fetchObservationQueue(
  runId: string,
): Promise<ObservationQueue> {
  const response = await fetch(`${API_BASE}/api/analyses/${runId}/observation-review`);
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as ObservationQueue;
}

export async function saveObservationCorrection(
  runId: string, taskId: string, action: string,
  humanValue: number | null, reviewer: string,
): Promise<Record<string, unknown>> {
  const response = await fetch(`${API_BASE}/api/analyses/${runId}/observation-review/corrections`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      task_id: taskId, action, human_value: humanValue, reviewer,
    }),
  });
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as Record<string, unknown>;
}

export interface ObservationUndoResult {
  undone: Record<string, unknown>;
  correction: Record<string, unknown> | null;
  status: string;
}

export async function undoObservationCorrection(
  runId: string, taskId: string, reviewer: string,
): Promise<ObservationUndoResult> {
  const response = await fetch(`${API_BASE}/api/analyses/${runId}/observation-review/corrections/undo`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ task_id: taskId, reviewer }),
  });
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as ObservationUndoResult;
}

export async function saveFrameClassification(
  runId: string, frameIndex: number, action: string,
  humanValue: number | null, reviewer: string,
): Promise<Record<string, unknown>> {
  const response = await fetch(`${API_BASE}/api/analyses/${runId}/observation-review/classifications`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      frame_index: frameIndex, action,
      human_value: humanValue, reviewer,
    }),
  });
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as Record<string, unknown>;
}

export async function fetchReadiness(
  runId: string,
): Promise<{ ready: boolean; open_tasks: number; live_generation: boolean;
  blocking_errors?: string[] | null; deferred?: number;
  live_available?: boolean; live_blockers?: string[];
  cost_note?: string }> {
  const response = await fetch(`${API_BASE}/api/analyses/${runId}/recipe/readiness`);
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as {
    ready: boolean; open_tasks: number; live_generation: boolean;
  blocking_errors?: string[] | null;
  deferred?: number;
  live_available?: boolean;
  live_blockers?: string[];
  cost_note?: string;
  };
}

export async function generateRecipe(
  runId: string,
  options?: { mode?: string; confirmed?: boolean; max_rounds?: number },
): Promise<RecipeReviewState> {
  const response = await fetch(`${API_BASE}/api/analyses/${runId}/recipe/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      mode: options?.mode ?? "simulation",
      confirmed: options?.confirmed ?? false,
      max_rounds: options?.max_rounds ?? 1,
    }),
  });
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as RecipeReviewState;
}

export async function fetchRecipe(runId: string): Promise<RecipeReviewState> {
  const response = await fetch(`${API_BASE}/api/analyses/${runId}/recipe`);
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as RecipeReviewState;
}

export async function saveRecipeCorrections(
  runId: string,
  ingredients: Array<Record<string, unknown>>,
  steps: Array<Record<string, unknown>>,
  reviewer: string,
): Promise<RecipeReviewState> {
  const response = await fetch(`${API_BASE}/api/analyses/${runId}/recipe/corrections`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ingredients, steps, reviewer }),
  });
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as RecipeReviewState;
}

export async function saveRecipeDecision(
  runId: string,
  decision: string,
  reviewer: string,
  notes?: string,
): Promise<RecipeReviewState> {
  const response = await fetch(`${API_BASE}/api/analyses/${runId}/recipe/decisions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decision, reviewer, notes: notes ?? null }),
  });
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as RecipeReviewState;
}

export async function fetchAnalysis(runId: string): Promise<AnalysisStatus> {
  const response = await fetch(`${API_BASE}/api/analyses/${runId}`);
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as AnalysisStatus;
}

export function artifactUrl(runId: string, artifactId: string): string {
  return `${API_BASE}/api/analyses/${runId}/artifacts/${artifactId}`;
}

export async function fetchArtifactJson<T>(
  runId: string,
  artifactId: string,
): Promise<T> {
  const response = await fetch(artifactUrl(runId, artifactId));
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as T;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export async function listAnalyses(limit = 12): Promise<AnalysisStatus[]> {
  const response = await fetch(`${API_BASE}/api/analyses?limit=${limit}`);
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as AnalysisStatus[];
}

export function frameUrl(runId: string, frameName: string): string {
  return `${API_BASE}/api/analyses/${runId}/frames/${frameName}`;
}

export async function fetchReviewBundle(runId: string): Promise<ReviewBundle> {
  const response = await fetch(`${API_BASE}/api/analyses/${runId}/review`);
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as ReviewBundle;
}

export interface ReviewInput {
  frame_index: number;
  decision: string;
  value?: number | null;
  notes?: string | null;
  reviewer?: string | null;
}

export interface CostEstimate {
  estimated_cost_usd: number;
  frames: number;
  nonbinding: boolean;
  basis: string;
}

export interface UploadInfo {
  upload_id: string;
  file_name: string;
  bytes: number;
  sha256: string;
  duration_seconds: number;
  fps: number;
  frame_count: number;
  width: number;
  height: number;
  sample_fps: number;
  estimated_frames: number;
  exceeds_frame_cap: boolean;
  deduplicated: boolean;
  status: string;
  estimate: CostEstimate | null;
}

export interface ProgressInfo {
  run_id: string;
  workflow: string;
  status: string;
  stage: string;
  frames_total: number | null;
  frames_processed: number | null;
  percent: number | null;
  error: string | null;
  updated_at: string;
}

export function uploadVideo(
  file: File,
  onProgress: (percent: number) => void,
): Promise<UploadInfo> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/api/uploads`);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    };
    xhr.onload = () => {
      try {
        const payload = JSON.parse(xhr.responseText) as UploadInfo & {
          detail?: string;
        };
        if (xhr.status >= 200 && xhr.status < 300) resolve(payload);
        else {
          reject(new ApiError(
            xhr.status, payload.detail ?? `Upload failed (${xhr.status})`));
        }
      } catch {
        reject(new ApiError(xhr.status, `Upload failed (${xhr.status})`));
      }
    };
    xhr.onerror = () => reject(new ApiError(0, "Upload failed: network error."));
    const form = new FormData();
    form.append("file", file, file.name);
    xhr.send(form);
  });
}

export async function createUploadAnalysis(
  uploadId: string,
  workflow: string,
  grant: Record<string, unknown> | null,
): Promise<AnalysisCreated> {
  const response = await fetch(
    `${API_BASE}/api/uploads/${uploadId}/analyses`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ workflow, grant }),
    });
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as AnalysisCreated;
}

export async function fetchProgress(runId: string): Promise<ProgressInfo> {
  const response = await fetch(`${API_BASE}/api/analyses/${runId}/progress`);
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as ProgressInfo;
}

export async function saveReview(
  runId: string,
  input: ReviewInput,
): Promise<ReviewAnnotation> {
  const response = await fetch(`${API_BASE}/api/analyses/${runId}/reviews`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as ReviewAnnotation;
}

export interface FusionBundle {
  run_name: string;
  draft: FusionDraft;
  evaluation: Record<string, unknown>;
}

export async function fetchFusionDraft(
  runName: string,
): Promise<FusionBundle> {
  const response = await fetch(`${API_BASE}/api/fusion/drafts/${runName}`);
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as FusionBundle;
}

export function fusionVideoUrl(
  runName: string,
  clipId?: string | null,
): string {
  const base = `${API_BASE}/api/fusion/drafts/${runName}/video`;
  return clipId ? `${base}/${clipId}` : base;
}

export async function fetchFusionDecisions(
  runName: string,
): Promise<FusionDecision[]> {
  const response = await fetch(`${API_BASE}/api/fusion/drafts/${runName}/reviews`);
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  const payload = (await response.json()) as {
    decisions: FusionDecision[];
  };
  return payload.decisions;
}

export async function saveFusionDecision(
  runName: string,
  input: {
    target_type: FusionTargetType;
    target_id: string;
    action: FusionAction;
    value?: Record<string, unknown> | null;
    reviewer: string;
    notes?: string | null;
  },
): Promise<FusionDecision> {
  const response = await fetch(`${API_BASE}/api/fusion/drafts/${runName}/reviews`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as FusionDecision;
}

export interface ProposedIngredient {
  name: string;
  quantity: number | null;
  unit: string | null;
  quantity_status: string;
  quantity_basis: string | null;
  review_status: string[];
  group_ids: string[];
  note: string | null;
  evidence?: Array<{ kind?: string | null }>;
  draft_adjustment: {
    displayed_value: number | null;
    draft_value: number | null;
    reason: string;
  } | null;
  chef_edited: boolean;
  original: Record<string, unknown> | null;
}

export interface ProposedMethodStep {
  number: number;
  clip_id: string | null;
  interval: string;
  instruction: string;
  step_group_ids: string[];
  review_status: string[];
  caveats: string[];
  ai_guidance: { text: string; provenance: string } | null;
  chef_edited: boolean;
  original: Record<string, unknown> | null;
}

export interface ImprovementSuggestion {
  id: string;
  label_en?: string;
  label_es?: string;
  hint_en?: string;
  hint_es?: string;
  placeholder_en?: string;
  placeholder_es?: string;
  label?: string;
  hint?: string;
  placeholder?: string;
}

export interface ProposedRecipe {
  snapshot_version: string;
  snapshot_id: string;
  run_name: string;
  recipe_id?: string;
  recipe_name?: string | null;
  language?: string;
  parent_version?: string | null;
  parent_name?: string | null;
  generated_at: string;
  title: string;
  title_basis: string;
  recipe_status: string;
  status_note: string | null;
  status_updated_at: string | null;
  last_saved_at: string | null;
  review_summary: Record<string, unknown>;
  ingredients: ProposedIngredient[];
  method: ProposedMethodStep[];
  review_flags: Array<Record<string, unknown>>;
  completion_open_items: string[];
  limitations: string[];
  improvement_suggestions?: ImprovementSuggestion[];
}

export interface ProposedIngredientEdit {
  index: number;
  name?: string;
  quantity?: number | null;
  unit?: string | null;
}

export interface ProposedMethodEdit {
  number: number;
  instruction?: string;
}

export async function fetchProposedRecipe(recipeId: string): Promise<ProposedRecipe> {
  const response = await fetch(`${API_BASE}/api/recipes/${recipeId}`);
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as ProposedRecipe;
}

export async function saveProposedRecipe(
  recipeId: string,
  input: {
    ingredients: ProposedIngredientEdit[];
    method: ProposedMethodEdit[];
  },
): Promise<ProposedRecipe> {
  const response = await fetch(`${API_BASE}/api/recipes/${recipeId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as ProposedRecipe;
}

export async function setProposedRecipeStatus(
  recipeId: string,
  input: { status: string; note?: string | null },
): Promise<ProposedRecipe> {
  const response = await fetch(`${API_BASE}/api/recipes/${recipeId}/status`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as ProposedRecipe;
}

export function recipeExportUrl(
  recipeId: string,
  format: "pdf" | "docx",
): string {
  return `${API_BASE}/api/recipes/${recipeId}/export.${format}`;
}

export interface RecipeCard {
  recipe_id: string;
  name: string;
  language: string;
  source_run_id: string | null;
  parent_version: string | null;
  created_at: string | null;
  updated_at: string | null;
  ingredients: number;
  steps: number;
  has_snapshot: boolean;
  status: string;
  build_error?: string | null;
}

export async function listRecipes(): Promise<RecipeCard[]> {
  const response = await fetch(`${API_BASE}/api/recipes`);
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  const payload = (await response.json()) as { recipes: RecipeCard[] };
  return payload.recipes;
}

export async function createRecipe(input: {
  name: string;
  language: string;
  source_run_id?: string | null;
}): Promise<{ recipe_id: string }> {
  const response = await fetch(`${API_BASE}/api/recipes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as { recipe_id: string };
}

export interface RevisionSuggestion {
  id: string;
  selected: boolean;
  value?: string | null;
}

export interface RevisionResult {
  recipe_id: string;
  parent_version: string;
  provider: string | null;
  model: string | null;
  request_id: string | null;
  actual_usd: number | null;
  recipe: ProposedRecipe;
}

export interface BuildResult {
  status: string;
  recipe: ProposedRecipe;
}

export async function buildRecipe(
  recipeId: string,
): Promise<BuildResult> {
  const response = await fetch(`${API_BASE}/api/recipes/${recipeId}/build`, {
    method: "POST",
  });
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as BuildResult;
}

export async function reviseRecipe(
  recipeId: string,
  input: {
    instructions: string;
    suggestions: RevisionSuggestion[];
    allow_ingredient_changes: boolean;
    confirmed: boolean;
  },
): Promise<RevisionResult> {
  const response = await fetch(`${API_BASE}/api/recipes/${recipeId}/revise`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  if (!response.ok) throw new ApiError(response.status, await parseError(response));
  return (await response.json()) as RevisionResult;
}
