import { useCallback, useEffect, useRef, useState } from "react";
import {
  ApiError,
  WORKFLOW_OFFLINE,
  WORKFLOW_SMOKE,
  artifactUrl,
  createAnalysis,
  fetchAnalysis,
  fetchArtifactJson,
  fetchHealth,
  formatBytes,
  type AnalysisStatus,
  type Health,
} from "./api";
import {
  numeric,
  seconds,
  sourceChip,
  statusPill,
  text,
  unitLabel,
  type MeasurementBundle,
  type OfflineSummary,
  type RecipeDraft,
  type UnitResolution,
  type ValidationResult,
  type WeighingEvent,
} from "./results";

type BackendState =
  | { kind: "checking" }
  | { kind: "online"; health: Health }
  | { kind: "offline"; message: string };

type ResultsState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | {
      kind: "ready";
      summary: OfflineSummary;
      draft: RecipeDraft;
      measurements: MeasurementBundle;
      validation: ValidationResult;
      weighing: WeighingEvent;
    };

const TERMINAL = new Set(["completed", "failed"]);
const SMOKE_ACCEPT = "video/*,.mp4,.mov,.mkv,.bin";
const OFFLINE_ACCEPT = ".json,application/json";

export default function App() {
  const [backend, setBackend] = useState<BackendState>({ kind: "checking" });
  const [workflow, setWorkflow] = useState<string>(WORKFLOW_SMOKE);
  const [file, setFile] = useState<File | null>(null);
  const [run, setRun] = useState<AnalysisStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  const checkBackend = useCallback(async () => {
    try {
      const health = await fetchHealth();
      setBackend({ kind: "online", health });
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : "Backend unreachable.";
      setBackend({ kind: "offline", message });
    }
  }, []);

  useEffect(() => {
    void checkBackend();
  }, [checkBackend]);

  useEffect(() => {
    return () => {
      if (timer.current !== null) window.clearInterval(timer.current);
    };
  }, []);

  function selectWorkflow(next: string) {
    if (next === workflow) return;
    if (timer.current !== null) {
      window.clearInterval(timer.current);
      timer.current = null;
    }
    setWorkflow(next);
    setFile(null);
    setRun(null);
    setError(null);
  }

  async function startRun() {
    if (!file || busy) return;
    if (timer.current !== null) window.clearInterval(timer.current);
    setBusy(true);
    setError(null);
    setRun(null);
    const submitted = workflow;
    try {
      const created = await createAnalysis(file, submitted);
      const first = await fetchAnalysis(created.run_id);
      setRun(first);
      if (!TERMINAL.has(first.status)) {
        timer.current = window.setInterval(async () => {
          try {
            const next = await fetchAnalysis(created.run_id);
            setRun(next);
            if (TERMINAL.has(next.status) && timer.current !== null) {
              window.clearInterval(timer.current);
              timer.current = null;
            }
          } catch (err) {
            if (timer.current !== null) {
              window.clearInterval(timer.current);
              timer.current = null;
            }
            setError(err instanceof Error ? err.message : "Polling failed.");
          }
        }, 1500);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Submission failed.");
    } finally {
      setBusy(false);
    }
  }

  const isOffline = workflow === WORKFLOW_OFFLINE;
  const showResults =
    run !== null &&
    run.status === "completed" &&
    run.workflow === WORKFLOW_OFFLINE;

  return (
    <div className="page">
      <header className="masthead">
        <div>
          <h1>Bakery Intelligence</h1>
          <p className="subtitle">Analysis review console (M2-06B)</p>
        </div>
        <BackendBadge backend={backend} onRetry={() => void checkBackend()} />
      </header>

      <main className="grid">
        <section className="card">
          <h2>Submit a run</h2>
          <fieldset className="radio-group">
            <legend>Workflow</legend>
            <label>
              <input
                type="radio"
                name="workflow"
                value={WORKFLOW_SMOKE}
                checked={workflow === WORKFLOW_SMOKE}
                onChange={() => selectWorkflow(WORKFLOW_SMOKE)}
              />
              <span>
                <strong>Storage smoke test</strong>
                <span className="hint">
                  {" "}
                  Stores a file and verifies its SHA-256. No analysis.
                </span>
              </span>
            </label>
            <label>
              <input
                type="radio"
                name="workflow"
                value={WORKFLOW_OFFLINE}
                checked={workflow === WORKFLOW_OFFLINE}
                onChange={() => selectWorkflow(WORKFLOW_OFFLINE)}
              />
              <span>
                <strong>Offline reconstruction</strong>
                <span className="hint">
                  {" "}
                  Reconstructs a recipe draft from an uploaded evidence
                  bundle. No inference, no network, no paid services.
                </span>
              </span>
            </label>
          </fieldset>
          <p className="hint">
            {isOffline
              ? "Upload a JSON offline bundle (bakery-offline-bundle/1). The worker reconstructs measurements and a draft from its evidence."
              : "Upload a small test file. The backend stores it, verifies its SHA-256, and records a result artifact. No video is analyzed."}
          </p>
          <label className="field">
            <span>
              {isOffline ? "Evidence bundle (JSON, max 16 MB)" : "Test file (max 16 MB)"}
            </span>
            <input
              key={workflow}
              type="file"
              accept={isOffline ? OFFLINE_ACCEPT : SMOKE_ACCEPT}
              onChange={(event) =>
                setFile(event.target.files?.[0] ?? null)
              }
            />
          </label>
          {file && (
            <p className="selection">
              Selected: <strong>{file.name}</strong> ({formatBytes(file.size)})
            </p>
          )}
          <button
            className="primary"
            disabled={!file || busy || backend.kind !== "online"}
            onClick={() => void startRun()}
          >
            {busy
              ? "Submitting…"
              : isOffline
                ? "Run offline reconstruction"
                : "Run storage smoke test"}
          </button>
          {error && <p className="error" role="alert">{error}</p>}
        </section>

        <section className="card">
          <h2>Run status</h2>
          {!run && <p className="hint">No run submitted yet.</p>}
          {run && (
            <dl className="facts">
              <dt>Run id</dt>
              <dd><code>{run.run_id}</code></dd>
              <dt>Workflow</dt>
              <dd><code>{run.workflow}</code></dd>
              <dt>Status</dt>
              <dd>
                <span className={`pill pill-${run.status}`}>{run.status}</span>
              </dd>
              <dt>Source</dt>
              <dd><code>{run.source_object_id ?? "—"}</code></dd>
              <dt>SHA-256</dt>
              <dd><code className="hash">{run.source_sha256 ?? "—"}</code></dd>
              {run.artifact_ids.map((id) => (
                <div className="artifact-row" key={id}>
                  <dt>Artifact</dt>
                  <dd>
                    <a
                      href={artifactUrl(run.run_id, id)}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {id}
                    </a>
                  </dd>
                </div>
              ))}
              {run.error && (
                <>
                  <dt>Error</dt>
                  <dd className="error">{run.error}</dd>
                </>
              )}
            </dl>
          )}
        </section>

        {showResults && run && (
          <ReconstructionResults key={run.run_id} runId={run.run_id} />
        )}
      </main>

      <footer className="footnote">
        Smoke runs prove connectivity plus storage integrity only. Offline
        reconstruction replays uploaded evidence deterministically; drafts
        require human review before operational use. Evidence images are not
        served by this console.
      </footer>
    </div>
  );
}

function ReconstructionResults({ runId }: { runId: string }) {
  const [state, setState] = useState<ResultsState>({ kind: "loading" });

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [summary, draft, measurements, validation, weighing] =
          await Promise.all([
            fetchArtifactJson<OfflineSummary>(runId, "offline_summary.json"),
            fetchArtifactJson<RecipeDraft>(runId, "recipe_draft.json"),
            fetchArtifactJson<MeasurementBundle>(runId, "measurements.json"),
            fetchArtifactJson<ValidationResult>(runId, "validation.json"),
            fetchArtifactJson<WeighingEvent>(runId, "weighing_event.json"),
          ]);
        if (!cancelled) {
          setState({ kind: "ready", summary, draft, measurements, validation, weighing });
        }
      } catch (err) {
        if (!cancelled) {
          setState({
            kind: "error",
            message: err instanceof Error ? err.message : "Failed to load results.",
          });
        }
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [runId]);

  if (state.kind === "loading") {
    return (
      <section className="card wide">
        <h2>Reconstruction results</h2>
        <p className="hint">Loading result artifacts…</p>
      </section>
    );
  }
  if (state.kind === "error") {
    return (
      <section className="card wide">
        <h2>Reconstruction results</h2>
        <p className="error" role="alert">{state.message}</p>
        <p className="hint">
          The run completed but its artifacts could not be rendered. Use the
          artifact links above to inspect the original JSON.
        </p>
      </section>
    );
  }

  const { summary, draft, measurements, validation, weighing } = state;
  const resolutions = new Map<string, UnitResolution>(
    measurements.measurements.map((item) => [
      item.measurement_id,
      measurements.unit_resolutions.find(
        (entry) => entry.measurement_id === item.measurement_id,
      ),
    ]).filter((entry): entry is [string, UnitResolution] => entry[1] !== undefined),
  );

  return (
    <section className="card wide">
      <h2>Reconstruction results</h2>
      <p className="hint">{summary.note}</p>
      <div className="flag-row">
        <Flag off={!summary.inference_performed} label="model inference" />
        <Flag off={!summary.network_accessed} label="network access" />
        <Flag off={!summary.paid_services_used} label="paid services" />
      </div>

      <h3>Run summary</h3>
      <dl className="facts">
        <dt>Draft</dt>
        <dd><code>{summary.draft_id}</code></dd>
        <dt>Video</dt>
        <dd><code>{summary.video_id}</code></dd>
        <dt>Event</dt>
        <dd><code>{summary.event_id}</code></dd>
        <dt>Window</dt>
        <dd>
          {seconds(measurements.window.start_seconds)} –{" "}
          {seconds(measurements.window.end_seconds)}
        </dd>
        <dt>Measurements</dt>
        <dd>{summary.measurements}</dd>
        <dt>Actions</dt>
        <dd>{summary.actions}</dd>
        <dt>Issues</dt>
        <dd>{summary.issues}</dd>
        <dt>Bundle</dt>
        <dd><code className="hash">{summary.bundle_sha256}</code></dd>
      </dl>

      <h3>Recipe draft</h3>
      <dl className="facts">
        <dt>Status</dt>
        <dd><span className="pill pill-pending">{draft.status}</span></dd>
        <dt>Version</dt>
        <dd>{draft.version}</dd>
        <dt>Ingredients</dt>
        <dd>
          {draft.ingredients.length === 0
            ? "None identified — the draft names no ingredients."
            : draft.ingredients.map((item, index) => (
                <code key={index}>{JSON.stringify(item)} </code>
              ))}
        </dd>
      </dl>
      {draft.steps.map((step) => (
        <div key={step.index} className="step">
          <p>{step.description}</p>
          <p className="hint">
            Observations:{" "}
            {step.observation_ids.map((id) => (
              <code key={id}>{id} </code>
            ))}
            {step.action_ids.length > 0 && (
              <>
                Actions:{" "}
                {step.action_ids.map((id) => (
                  <code key={id}>{id} </code>
                ))}
              </>
            )}
          </p>
        </div>
      ))}

      <h3>Measurements</h3>
      <p className="hint">
        Scale display readings only — never net ingredient quantities.
        Verbatim text is the first-pass model reading; the value column
        shows the reading selected for the draft.
      </p>
      <div className="table-wrap">
        <table className="grid-table">
          <thead>
            <tr>
              <th>Frame</th>
              <th>Time</th>
              <th>Value</th>
              <th>Unit</th>
              <th>Source</th>
              <th>Status</th>
              <th>Verbatim</th>
              <th>Δ prev</th>
            </tr>
          </thead>
          <tbody>
            {measurements.measurements.map((item) => {
              const chip = sourceChip(item.value_source);
              const unit = unitLabel(
                item.unit_observed,
                resolutions.get(item.measurement_id),
              );
              return (
                <tr key={item.measurement_id}>
                  <td><code>{item.frame_index}</code></td>
                  <td>{seconds(item.timestamp_seconds)}</td>
                  <td>{numeric(item.value)}</td>
                  <td><span className={`pill ${unit.pill}`}>{unit.text}</span></td>
                  <td><span className={`pill ${chip.pill}`}>{chip.label}</span></td>
                  <td>
                    <span className={`pill ${statusPill(item.value_status)}`}>
                      {item.value_status}
                    </span>
                  </td>
                  <td><code>{text(item.original.verbatim)}</code></td>
                  <td>{numeric(item.delta_from_previous)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <h3>Weighing event</h3>
      <dl className="facts">
        <dt>Event</dt>
        <dd><code>{weighing.event_id}</code></dd>
        <dt>Interval</dt>
        <dd>{seconds(weighing.start_seconds)} – {seconds(weighing.end_seconds)}</dd>
        <dt>Readings</dt>
        <dd>{weighing.observation_ids.length} linked observations</dd>
        <dt>Final display</dt>
        <dd>
          {numeric(weighing.final_display_reading.value)}
          {weighing.final_display_reading.unit
            ? ` ${weighing.final_display_reading.unit}`
            : " (unit not observed)"}
        </dd>
        <dt>Net quantity</dt>
        <dd>
          {weighing.net_ingredient_quantity.value === null
            ? `unknown (derivation: ${text(weighing.net_ingredient_quantity.derivation)})`
            : numeric(weighing.net_ingredient_quantity.value)}
        </dd>
        <dt>Ingredient</dt>
        <dd>
          {weighing.ingredient_id === null
            ? "not identified"
            : <code>{weighing.ingredient_id}</code>}
        </dd>
        <dt>Tare</dt>
        <dd>{weighing.tare_observed ? "observed in window" : "No tare observed in the window."}</dd>
        <dt>Mode</dt>
        <dd><code>{weighing.measurement_mode}</code></dd>
        <dt>Verification</dt>
        <dd><code>{weighing.verification_status}</code></dd>
      </dl>
      <p className="hint">{weighing.review_notes}</p>

      <h3>Temporal evidence</h3>
      {measurements.action_evidence.temporal_actions.length === 0 ? (
        <p className="hint">No temporal action candidates linked to this event.</p>
      ) : (
        <ul className="plain-list">
          {measurements.action_evidence.temporal_actions.map((action) => (
            <li key={action.event_id}>
              <code>{action.candidate_type}</code> · {seconds(action.start_timestamp_seconds)} –{" "}
              {seconds(action.end_timestamp_seconds)} · status{" "}
              <code>{action.verification_status}</code> ·{" "}
              <code>{action.event_id}</code>
            </li>
          ))}
        </ul>
      )}
      <p className="hint">
        Addition frames: {measurements.action_evidence.addition_frames.length};
        removal frames: {measurements.action_evidence.removal_frames.length};
        tare frames: {measurements.action_evidence.tare_frames.length}.
      </p>

      <h3>Unresolved uncertainties</h3>
      {draft.unresolved_questions.length === 0 ? (
        <p className="hint">No unresolved questions recorded.</p>
      ) : (
        <ul className="plain-list">
          {draft.unresolved_questions.map((question) => (
            <li key={question.code}>
              <code>{question.code}</code> — {question.message}
            </li>
          ))}
        </ul>
      )}

      <h3>Validation</h3>
      {validation.issue_count === 0 ? (
        <p>
          <span className="pill pill-ok">0 issues</span>{" "}
          <span className="hint">
            Structural checks only — a clean result does not certify correct
            interpretation.
          </span>
        </p>
      ) : (
        <ul className="plain-list">
          {validation.issues.map((issue, index) => (
            <li key={`${issue.code}-${index}`}>
              <code>{issue.code}</code> · gate <code>{issue.gate}</code> ·
              severity <code>{issue.severity}</code> · entity{" "}
              <code>{issue.entity_id}</code> — {issue.message}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function Flag({ off, label }: { off: boolean; label: string }) {
  if (off) {
    return <span className="pill pill-ok">No {label}</span>;
  }
  return <span className="pill pill-bad">{label} used</span>;
}

function BackendBadge({
  backend,
  onRetry,
}: {
  backend: BackendState;
  onRetry: () => void;
}) {
  if (backend.kind === "checking") {
    return <span className="pill pill-pending">checking backend…</span>;
  }
  if (backend.kind === "online") {
    return (
      <span className="pill pill-ok" title={backend.health.version}>
        backend online · {backend.health.storage_backend}
      </span>
    );
  }
  return (
    <span className="badge-row">
      <span className="pill pill-bad">backend offline</span>
      <button className="ghost" onClick={onRetry}>Retry</button>
    </span>
  );
}
