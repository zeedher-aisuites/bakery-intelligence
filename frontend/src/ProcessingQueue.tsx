import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, listJobs, retryJob, type JobRecord } from "./api";
import {
  canRetry,
  countByStatus,
  displaySourceName,
  filterJobs,
  hasActiveJobs,
  retryHint,
  statusLabel,
} from "./queue";

const POLL_MS = 3000;
const FILTERS = [
  { id: "all", label: "All" },
  { id: "queued", label: "Queued" },
  { id: "running", label: "Running" },
  { id: "completed", label: "Completed" },
  { id: "failed", label: "Failed" },
] as const;

function fmtDate(value?: string): string {
  if (!value) return "—";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

function pillClass(status: string): string {
  if (status === "completed") return "pill pill-completed";
  if (status === "failed") return "pill pill-failed";
  if (status === "running") return "pill pill-running";
  return "pill pill-queued";
}

export default function ProcessingQueue(props: {
  highlightRunId?: string | null;
  onOpenReview: (runId: string) => void;
}): JSX.Element {
  const [jobs, setJobs] = useState<JobRecord[] | null>(null);
  const [filter, setFilter] = useState<string>("all");
  const [connError, setConnError] = useState<string | null>(null);
  const [retrying, setRetrying] = useState<string | null>(null);
  const [retryError, setRetryError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  const load = useCallback(async () => {
    try {
      const rows = await listJobs(undefined, 100);
      setJobs(rows);
      setConnError(null);
    } catch (err) {
      setConnError(
        err instanceof ApiError
          ? `Backend unavailable (HTTP ${err.status}): ${err.message}`
          : "Backend unavailable. Showing the last known queue state.",
      );
    }
  }, []);

  useEffect(() => {
    void load();
    timer.current = window.setInterval(() => void load(), POLL_MS);
    return () => {
      if (timer.current !== null) window.clearInterval(timer.current);
      timer.current = null;
    };
  }, [load]);

  async function onRetry(runId: string): Promise<void> {
    setRetrying(runId);
    setRetryError(null);
    try {
      await retryJob(runId);
      await load();
    } catch (err) {
      setRetryError(
        err instanceof Error ? err.message : "Retry was refused.",
      );
    } finally {
      setRetrying(null);
    }
  }

  const counts = countByStatus(jobs ?? []);
  const visible = filterJobs(jobs ?? [], filter);
  const active = hasActiveJobs(jobs ?? []);

  return (
    <div className="queue-page">
      <section className="card queue-card">
        <div className="queue-head">
          <div>
            <h2>Processing queue</h2>
            <p className="muted">
              Persisted jobs from the backend. This view survives a
              browser refresh: queued work stays visible even when
              no worker is currently running.
            </p>
          </div>
          <button
            type="button"
            className="btn-ghost"
            onClick={() => void load()}
          >
            Refresh now
          </button>
        </div>

        {connError && (
          <p className="error" role="alert">
            {connError} Retrying automatically every{" "}
            {POLL_MS / 1000} seconds.
          </p>
        )}
        {retryError && (
          <p className="error" role="alert">{retryError}</p>
        )}
        {jobs !== null && !active && (
          <p className="muted small" role="status">
            No job is currently queued or running. New uploads will
            appear here as queued until a worker claims them.
          </p>
        )}

        <div className="queue-filters" role="tablist" aria-label="Filter by status">
          {FILTERS.map((item) => (
            <button
              key={item.id}
              type="button"
              role="tab"
              aria-selected={filter === item.id}
              className={filter === item.id ? "active" : ""}
              onClick={() => setFilter(item.id)}
            >
              {item.label} ({counts[item.id] ?? 0})
            </button>
          ))}
        </div>

        {jobs === null && !connError && (
          <p className="muted">Loading processing queue…</p>
        )}
        {jobs !== null && visible.length === 0 && (
          <div className="empty">
            <h3>No jobs in this view</h3>
            <p>
              {jobs.length === 0
                ? "No analysis has been submitted yet. Upload a video to create the first queued job."
                : "No persisted job matches this status filter."}
            </p>
          </div>
        )}

        <ul className="queue-list">
          {visible.map((job) => {
            const hint = retryHint(job);
            const highlighted = props.highlightRunId === job.run_id;
            return (
              <li
                key={job.run_id}
                className={highlighted ? "queue-item highlighted" : "queue-item"}
              >
                <div className="queue-item-main">
                  <strong>{displaySourceName(job)}</strong>
                  <span className={pillClass(job.status)}>
                    {statusLabel(job)}
                  </span>
                  {highlighted && (
                    <span className="pill pill-info">Just submitted</span>
                  )}
                </div>
                <dl className="queue-facts">
                  <div><dt>Run ID</dt><dd><code>{job.run_id}</code></dd></div>
                  <div><dt>Workflow</dt><dd><code>{job.workflow}</code></dd></div>
                  <div><dt>Created</dt><dd>{fmtDate(job.created_at)}</dd></div>
                  <div><dt>Updated</dt><dd>{fmtDate(job.updated_at ?? undefined)}</dd></div>
                  <div><dt>Attempts</dt><dd>{job.attempts ?? 0}</dd></div>
                </dl>
                {job.status === "queued" && (
                  <p className="muted small">
                    Accepted and persisted. Waiting for a worker to
                    claim this job; it is not processing yet.
                  </p>
                )}
                {job.status === "failed" && (
                  <div className="queue-error">
                    <p className="error" role="alert">
                      {job.error ?? "Analysis failed without a recorded message."}
                    </p>
                    {hint && <p className="muted small">{hint}</p>}
                    {job.workflow === "video_pipeline_meta" && (
                      <p className="muted small">
                        Operator action: inspect the failure, then
                        submit a new authorized analysis. The
                        original failed record is preserved.
                      </p>
                    )}
                  </div>
                )}
                <div className="queue-actions">
                  {job.status === "completed" && (
                    <button
                      type="button"
                      className="btn-primary"
                      onClick={() => props.onOpenReview(job.run_id)}
                    >
                      Open completed analysis
                    </button>
                  )}
                  {canRetry(job) && (
                    <button
                      type="button"
                      className="btn-ghost"
                      disabled={retrying === job.run_id}
                      onClick={() => void onRetry(job.run_id)}
                    >
                      {retrying === job.run_id ? "Requeueing…" : "Retry job"}
                    </button>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
        <p className="muted small">
          Status refreshes automatically while this page is open.
          Completed results open in the evidence review workspace.
        </p>
      </section>
    </div>
  );
}
