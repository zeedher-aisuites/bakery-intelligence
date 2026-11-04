import { useEffect, useState } from "react";
import {
  fetchObservationQueue,
  fetchReadiness,
  generateRecipe,
  type ReviewTask,
} from "./api";
import { taskLabel } from "./observation";
import { formatVideoTime } from "./recipe";

/**
 * Stage 2 generation panel. One primary action — Generate Recipe —
 * which always runs the real AI workflow with explicit confirmation.
 * Simulation stays available in the backend and API for development
 * and automated tests, but it is not part of this workflow.
 */
export default function GeneratePanel(props: {
  runId: string;
  /**
   * Whether the stage is currently visible. The workspace keeps all
   * stages mounted and only toggles `hidden`, so the panel must
   * re-read the authoritative readiness every time it is shown:
   * corrections saved in Review Findings after the first fetch would
   * otherwise leave a stale block (and a disabled action) on screen.
   */
  active?: boolean;
  /**
   * Incremented whenever findings change. Re-reads readiness while
   * the stage is visible so both stages always agree.
   */
  refreshSignal?: number;
  onGenerated?: () => void;
  onBackToFindings?: () => void;
}): JSX.Element {
  const [readiness, setReadiness] = useState<{
    open: number; deferred: number; blocking: string[];
  } | null>(null);
  const [tasks, setTasks] = useState<ReviewTask[]>([]);
  const [confirmed, setConfirmed] = useState(false);
  const [liveInfo, setLiveInfo] = useState<{
    live_available: boolean; live_blockers: string[];
  } | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  const active = props.active ?? true;
  const refreshSignal = props.refreshSignal ?? 0;

  useEffect(() => {
    let cancelled = false;
    setReadiness(null);
    setTasks([]);
    setDone(false);
    setError(null);
    fetchReadiness(props.runId).then((value) => {
      if (cancelled) return;
      setReadiness({
        open: value.open_tasks,
        deferred: value.deferred ?? 0,
        blocking: value.blocking_errors ?? [],
      });
      setLiveInfo({
        live_available: value.live_available ?? true,
        live_blockers: value.live_blockers ?? [],
      });
    }).catch((err: unknown) => {
      if (!cancelled) {
        setError(err instanceof Error ? err.message : "Readiness failed.");
      }
    });
    fetchObservationQueue(props.runId).then(
      (queue) => { if (!cancelled) setTasks(queue.tasks); },
      () => { if (!cancelled) setTasks([]); });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.runId]);

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    fetchReadiness(props.runId).then((value) => {
      if (cancelled) return;
      setReadiness({
        open: value.open_tasks,
        deferred: value.deferred ?? 0,
        blocking: value.blocking_errors ?? [],
      });
      setLiveInfo({
        live_available: value.live_available ?? true,
        live_blockers: value.live_blockers ?? [],
      });
    }).catch((err: unknown) => {
      if (!cancelled) {
        setError(err instanceof Error ? err.message : "Readiness failed.");
      }
    });
    fetchObservationQueue(props.runId).then(
      (queue) => { if (!cancelled) setTasks(queue.tasks); },
      () => { if (!cancelled) setTasks([]); });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, refreshSignal]);

  async function generate(): Promise<void> {
    setBusy(true);
    setError(null);
    try {
      await generateRecipe(props.runId, {
        mode: "live",
        confirmed,
        max_rounds: 1,
      });
      setConfirmed(false);
      setDone(true);
      props.onGenerated?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed.");
    } finally {
      setBusy(false);
    }
  }

  function describeTask(task: ReviewTask): string {
    const start = formatVideoTime(task.start_seconds);
    const end = task.end_seconds !== null
      && task.end_seconds !== task.start_seconds
      ? `–${formatVideoTime(task.end_seconds)}` : "";
    return `${taskLabel(task.state)} · ${start}${end}`;
  }

  const blocking = readiness?.blocking ?? [];
  const blockedDetails = blocking.map((entry) => {
    const task = tasks.find((item) => item.task_id === entry);
    return {
      entry,
      label: task ? describeTask(task) : null,
    };
  });
  const liveBlocked = liveInfo !== null && !liveInfo.live_available;
  const actionDisabled = busy || blocking.length > 0
    || !liveInfo?.live_available || !confirmed;

  return (
    <section className="card" aria-label="Generate recipe">
      <h2>Generate recipe</h2>
      {readiness === null && !error && (
        <p className="muted">Checking what is ready…</p>
      )}
      {readiness !== null && (
        <p className="muted">
          {readiness.open} finding{readiness.open === 1 ? "" : "s"} still
          need{readiness.open === 1 ? "s" : ""} your attention
          {readiness.deferred > 0
            ? ` (${readiness.deferred} set aside for later)` : ""}.
          {blocking.length === 0
            ? " Nothing blocks generation."
            : blocking.length === 1
              ? " 1 conflicting finding still blocks generation."
                + " Resolve it first."
              : ` ${blocking.length} conflicting findings still block`
                + " generation. Resolve them first."}
        </p>
      )}
      {blocking.length > 0 && props.onBackToFindings && (
        <div className="flow-actions">
          <button
            type="button"
            className="btn-ghost"
            onClick={props.onBackToFindings}
          >
            Review findings
          </button>
        </div>
      )}
      <div className="flow-actions">
        <button
          type="button"
          className="btn-primary"
          disabled={actionDisabled}
          onClick={() => void generate()}
        >
          Generate Recipe
        </button>
      </div>
      <label className="confirm-row">
        <input
          type="checkbox"
          checked={confirmed}
          onChange={(event) => setConfirmed(event.target.checked)}
        />
        <span>I confirm billable AI inference for this run</span>
      </label>
      {(liveInfo?.live_blockers ?? []).length > 0 && (
        <p className="muted small">
          {(liveInfo?.live_blockers ?? []).join(" ")}
        </p>
      )}
      {liveBlocked && (
        <p className="muted small">
          Live AI is currently unavailable, so generation is disabled.
        </p>
      )}
      {done && (
        <p className="muted small" role="status">
          Recipe generated. Continue to review and approve it.
        </p>
      )}
      {error && <p className="error" role="alert">{error}</p>}
      {blocking.length > 0 && (
        <details>
          <summary>
            View technical details
            <span className="muted small">
              {" "}Finding references that block generation.
            </span>
          </summary>
          <ul>
            {blockedDetails.map((item) => (
              <li key={item.entry} className="muted small">
                <code>{item.entry}</code>
                {item.label ? ` — ${item.label}` : ""}
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}
