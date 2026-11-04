import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError, fetchRecipe, listJobs, retryJob, type JobRecord,
} from "./api";
import {
  canRetry,
  countByStatus,
  displaySourceName,
  errorSummary,
  filterJobs,
  hasActiveJobs,
  statusLabel,
} from "./queue";

const POLL_MS = 3000;
const FILTERS = [
  { id: "all", label: "All" },
  { id: "queued", label: "Waiting" },
  { id: "running", label: "Processing" },
  { id: "completed", label: "Ready" },
  { id: "failed", label: "Failed" },
] as const;

function fmtDate(value?: string | null): string {
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
  onNewAnalysis: () => void;
}): JSX.Element {
  const [jobs, setJobs] = useState<JobRecord[] | null>(null);
  const [filter, setFilter] = useState<string>("all");
  const [connError, setConnError] = useState<string | null>(null);
  const [retrying, setRetrying] = useState<string | null>(null);
  const [retryError, setRetryError] = useState<string | null>(null);
  const [recipes, setRecipes] = useState<Record<string, "ready" | "absent">>(
    {});
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
          : "Backend unavailable. Showing the last known recipes.",
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

  // A completed analysis only means evidence is ready. Probe the real
  // persisted recipe state so "ready" never promises a missing recipe.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const pending = (jobs ?? []).filter((job) => job.status === "completed"
        && recipes[job.run_id] === undefined);
      for (const job of pending) {
        try {
          const state = await fetchRecipe(job.run_id);
          if (cancelled) return;
          setRecipes((prev) => prev[job.run_id] === undefined
            ? {
              ...prev,
              [job.run_id]: state.generation != null ? "ready" : "absent",
            }
            : prev);
        } catch {
          if (cancelled) return;
          setRecipes((prev) => prev[job.run_id] === undefined
            ? { ...prev, [job.run_id]: "absent" } : prev);
        }
      }
    })();
    return () => { cancelled = true; };
  }, [jobs, recipes]);

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
            <h2>My recipes</h2>
            <p className="muted">
              Track your recipes as they are prepared, and review
              the results when they&apos;re ready.
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
          <p className="error" role="alert">{connError}</p>
        )}
        {retryError && (
          <p className="error" role="alert">{retryError}</p>
        )}
        {jobs !== null && !active && jobs.length > 0 && (
          <p className="muted small" role="status">
            Nothing is being prepared right now.
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
          <p className="muted">Loading your recipes…</p>
        )}
        {jobs !== null && visible.length === 0 && (
          <div className="empty">
            <h3>
              {jobs.length === 0 ? "No recipes yet" : "No recipes here"}
            </h3>
            <p>
              {jobs.length === 0
                ? "Upload a cooking video to prepare your first recipe."
                : "No recipes match this filter."}
            </p>
            {jobs.length === 0 && (
              <button
                type="button"
                className="btn-primary"
                onClick={props.onNewAnalysis}
              >
                Start a new analysis
              </button>
            )}
          </div>
        )}

        <ul className="queue-list">
          {visible.map((job) => {
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
                  <div><dt>Created</dt><dd>{fmtDate(job.created_at)}</dd></div>
                </dl>
                {job.status === "queued" && (
                  <p className="muted small">Waiting to begin.</p>
                )}
                {job.status === "running" && (
                  <p className="muted small">
                    Being prepared — progress updates automatically.
                  </p>
                )}
                {job.status === "completed" && (
                  <p className="muted small">
                    {recipes[job.run_id] === "ready"
                      ? "Recipe ready — review draft."
                      : "Analysis ready — review findings."}
                  </p>
                )}
                {job.status === "failed" && (
                  <div className="queue-error">
                    <p className="error" role="alert">
                      {errorSummary(job)
                        ?? "This recipe could not be prepared."}
                    </p>
                  </div>
                )}
                <div className="queue-actions">
                  {job.status === "completed" && (
                    <button
                      type="button"
                      className="btn-primary"
                      onClick={() => props.onOpenReview(job.run_id)}
                    >
                      {recipes[job.run_id] === "ready"
                        ? "Review recipe" : "Review findings"}
                    </button>
                  )}
                  {canRetry(job) && (
                    <button
                      type="button"
                      className="btn-ghost"
                      disabled={retrying === job.run_id}
                      onClick={() => void onRetry(job.run_id)}
                    >
                      {retrying === job.run_id ? "Retrying…" : "Retry"}
                    </button>
                  )}
                </div>
              </li>
            );
          })}
        </ul>
        <p className="muted small">
          Your recipe progress updates automatically.
        </p>
      </section>
    </div>
  );
}
