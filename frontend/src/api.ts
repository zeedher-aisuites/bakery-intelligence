export const API_BASE =
  import.meta.env?.VITE_API_BASE ?? "http://localhost:8000";

export const WORKFLOW_SMOKE = "storage_smoke";
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
