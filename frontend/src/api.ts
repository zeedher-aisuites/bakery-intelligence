import type { ReviewAnnotation, ReviewBundle } from "./review";

export const API_BASE =
  import.meta.env?.VITE_API_BASE ?? "http://localhost:8000";

export const WORKFLOW_SMOKE = "storage_smoke";
export const WORKFLOW_META = "video_pipeline_meta";
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
