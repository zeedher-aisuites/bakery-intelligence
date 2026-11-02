// Pure helpers for the persistent Processing Queue.

export const META_WORKFLOW = "video_pipeline_meta";
export const QUEUE_STATUSES = [
  "queued", "running", "completed", "failed",
] as const;
export type QueueStatus = (typeof QUEUE_STATUSES)[number];

export interface QueueJob {
  run_id: string;
  workflow: string;
  status: string;
  source_object_id: string | null;
  error: string | null;
  error_category?: string | null;
  claimed_by?: string | null;
  attempts?: number;
  created_at?: string;
  updated_at?: string | null;
}

export function isQueueStatus(value: string): value is QueueStatus {
  return (QUEUE_STATUSES as readonly string[]).includes(value);
}

export function displaySourceName(job: QueueJob): string {
  const raw = job.source_object_id ?? "Unknown source";
  // Stored upload ids look like upload_<sha16>_<original name>.
  const match = /^upload_[0-9a-f]{16}_(.+)$/.exec(raw);
  return match ? match[1] : raw;
}

export function statusLabel(job: QueueJob): string {
  switch (job.status) {
    case "queued":
      return "Queued — waiting for a worker";
    case "running":
      return job.claimed_by
        ? `Running — claimed by ${job.claimed_by}`
        : "Running";
    case "completed":
      return "Completed";
    case "failed":
      return job.error_category === "validation"
        ? "Failed — validation"
        : "Failed — processing";
    default:
      return job.status;
  }
}

export function canRetry(job: QueueJob): boolean {
  return job.status === "failed" && job.workflow !== META_WORKFLOW;
}

export function retryHint(job: QueueJob): string | null {
  if (job.status !== "failed") return null;
  if (job.workflow === META_WORKFLOW) {
    return (
      "Paid Meta jobs are never retried automatically. "
      + "Submit a new authorized analysis to run this video again."
    );
  }
  return null;
}

export function hasActiveJobs(jobs: QueueJob[]): boolean {
  return jobs.some(
    (job) => job.status === "queued" || job.status === "running",
  );
}

export function filterJobs(
  jobs: QueueJob[],
  filter: string,
): QueueJob[] {
  if (filter === "all") return jobs;
  return jobs.filter((job) => job.status === filter);
}

export function countByStatus(
  jobs: QueueJob[],
): Record<string, number> {
  const counts: Record<string, number> = { all: jobs.length };
  for (const status of QUEUE_STATUSES) counts[status] = 0;
  for (const job of jobs) {
    if (isQueueStatus(job.status)) counts[job.status] += 1;
  }
  return counts;
}
