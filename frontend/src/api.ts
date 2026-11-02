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
