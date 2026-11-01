import { useEffect, useMemo, useState } from "react";
import {
  AnalysisStatus,
  ApiError,
  WORKFLOW_META,
  WORKFLOW_OFFLINE,
  WORKFLOW_PIPELINE,
  WORKFLOW_SMOKE,
  fetchAnalysis,
  fetchArtifactJson,
  fetchReviewBundle,
  frameUrl,
  listAnalyses,
  saveReview,
} from "./api";
import type { RecipeDraft } from "./results";
import {
  ReviewBundle,
  ReviewDecision,
  ReviewFrame,
  buildRecipeRows,
  effectiveReading,
  exportReviewCsv,
  exportReviewJson,
  frameState,
  orderFrames,
  summarizeReview,
} from "./review";

const DEMO_RUN_ID = "run_387662c3eaab";
const ZOOM_STEPS = [1, 1.5, 2, 3];

interface FirstpassReport {
  billable?: boolean;
  model?: string;
  provider?: string;
  pass_id?: string;
  requests_sent?: number;
  requests_submitted?: number;
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
  reasoning_output_tokens?: number | null;
  cost?: {
    spent_cost_usd_estimate?: number | null;
    cost_status?: string | null;
    unknown_cost_requests?: number | null;
  } | null;
  terminal_states?: Record<string, number>;
  frames_total?: number;
  frames_observed?: number;
  frames_failed?: number;
  prompt_sha256?: string;
}

interface PipelineSummaryOil {
  workflow?: string;
  note?: string;
  video?: {
    name?: string;
    sha256?: string;
    duration_seconds?: number | null;
  } | null;
  [key: string]: unknown;
}

function errorText(err: unknown): string {
  if (err instanceof ApiError) {
    return `Request failed (HTTP ${err.status}): ${err.message}`;
  }
  if (err instanceof Error) return err.message;
  return "Unexpected error.";
}

function download(filename: string, text: string, mime: string): void {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function fmtTime(value: number | null): string {
  if (value === null || value === undefined) return "—";
  return `${value.toFixed(1)} s`;
}

function fmtDate(value: string | undefined): string {
  if (!value) return "date unavailable";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/* ---------------- Landing ---------------- */

function Landing(props: { onOpen: (runId: string) => void }): JSX.Element {
  const [items, setItems] = useState<AnalysisStatus[] | null>(null);
  const [stats, setStats] = useState<Record<string, {
    frames: number; readings: number; abstentions: number;
  }>>({});
  const [error, setError] = useState<string | null>(null);
  const [heroImage, setHeroImage] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listAnalyses().then((rows) => {
      if (cancelled) return;
      setItems(rows);
      void Promise.allSettled(rows.map(async (row) => {
        const bundle = await fetchReviewBundle(row.run_id);
        return { run_id: row.run_id, bundle };
      })).then((results) => {
        if (cancelled) return;
        const next: Record<string, {
          frames: number; readings: number; abstentions: number;
        }> = {};
        for (const result of results) {
          if (result.status !== "fulfilled") continue;
          const counts = summarizeReview(result.value.bundle);
          next[result.value.run_id] = {
            frames: counts.frames,
            readings: counts.numeric,
            abstentions: counts.abstained,
          };
          if (result.value.run_id === DEMO_RUN_ID) {
            const frame = orderFrames(result.value.bundle.frames)
              .find((item) => item.image_available && item.image_name
                && typeof item.measurement?.value === "number");
            if (frame?.image_name) setHeroImage(frame.image_name);
          }
        }
        setStats(next);
      });
    }).catch((err: unknown) => {
      if (!cancelled) setError(errorText(err));
    });
    return () => { cancelled = true; };
  }, []);

  if (error) return <p className="error">{error}</p>;
  if (items === null) {
    return <p className="muted">Loading analyses…</p>;
  }
  const featured = items.find((item) => item.run_id === DEMO_RUN_ID)
    ?? null;
  const rest = featured
    ? items.filter((item) => item.run_id !== DEMO_RUN_ID) : items;
  const isTechnical = (item: AnalysisStatus): boolean =>
    item.workflow === WORKFLOW_SMOKE
    || (stats[item.run_id] !== undefined
      && stats[item.run_id].frames === 0);
  const library = rest.filter((item) => !isTechnical(item));
  const technical = rest.filter((item) => isTechnical(item));
  const featuredStats = featured ? stats[featured.run_id] : undefined;
  const scrollToId = (id: string): void => {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  };
  return (
    <div className="landing">
      <section className="hero">
        <div className="hero-copy">
          <p className="hero-eyebrow">Evidence-backed recipe intelligence</p>
          <h2>Turn cooking videos into traceable recipes.</h2>
          <p className="hero-lede">
            Transform real preparation footage into structured,
            evidence-backed records with AI analysis and human
            validation. Every reading links back to the frame it
            came from.
          </p>
          <div className="hero-ctas">
            <button
              type="button"
              className="btn-primary btn-lg"
              onClick={() => (featured
                ? props.onOpen(featured.run_id)
                : scrollToId("analyses"))}
            >
              Explore real analysis
            </button>
            <button
              type="button"
              className="btn-ghost btn-lg"
              onClick={() => scrollToId("how-it-works")}
            >
              How it works
            </button>
          </div>
          <ul className="capabilities">
            <li>
              <strong>Analyze footage</strong>
              <span>Real preparation videos, sampled frame by frame</span>
            </li>
            <li>
              <strong>Extract evidence</strong>
              <span>Timestamped readings with source provenance</span>
            </li>
            <li>
              <strong>Validate &amp; refine</strong>
              <span>Human review corrects and confirms AI output</span>
            </li>
          </ul>
        </div>
        <div className="hero-card">
          {featured && heroImage ? (
            <img
              className="hero-img"
              src={frameUrl(featured.run_id, heroImage)}
              alt="Retained frame from the featured real analysis"
            />
          ) : (
            <div className="hero-frame" aria-hidden="true" />
          )}
          <div className="hero-chip">
            {featuredStats
              ? `${featuredStats.frames} frames · `
                + `${featuredStats.readings} readings · `
                + `${featuredStats.abstentions} abstentions`
              : "Loading real analysis evidence…"}
          </div>
          {featured && (
            <span className="hero-badge">Real analysis · Muse Spark 1.3</span>
          )}
        </div>
      </section>
      <section id="how-it-works" className="how" aria-label="How it works">
        <h2>How it works</h2>
        <ol className="how-steps">
          <li>
            <span className="how-num">1</span>
            <div>
              <strong>Open a completed analysis</strong>
              <p>Pick a real video analysis with retained frame evidence.</p>
            </div>
          </li>
          <li>
            <span className="how-num">2</span>
            <div>
              <strong>Inspect the evidence</strong>
              <p>Walk the filmstrip and compare each frame with the AI reading.</p>
            </div>
          </li>
          <li>
            <span className="how-num">3</span>
            <div>
              <strong>Confirm or correct</strong>
              <p>Save a human decision; the preparation record updates instantly.</p>
            </div>
          </li>
        </ol>
      </section>
      <section id="analyses" aria-label="Your analyses">
        <div className="section-head">
          <h2>Featured demonstration</h2>
        </div>
        {items.length === 0 && (
          <div className="empty">
            <h3>No completed analyses yet</h3>
            <p>
              The review workspace shows finished video analyses
              with retained frame evidence. Run an offline
              analysis first, then return here to review it.
            </p>
          </div>
        )}
        {featured && (
          <button
            type="button"
            className="demo-card"
            onClick={() => props.onOpen(featured.run_id)}
          >
            <span className="pill pill-demo">Demonstration case</span>
            <strong className="run-title">{featured.run_id}</strong>
            <span className="muted small">
              {featured.workflow} · {featured.status} ·{" "}
              {fmtDate(featured.created_at)}
            </span>
            <span className="stat-row">
              <span>{featuredStats?.frames ?? "—"} frames</span>
              <span>{featuredStats?.readings ?? "—"} readings</span>
              <span>
                {featuredStats?.abstentions ?? "—"} abstentions
              </span>
            </span>
            {featured && heroImage && (
              <img
                className="demo-thumb"
                src={frameUrl(featured.run_id, heroImage)}
                alt="Frame from the featured analysis"
              />
            )}
            <span className="btn-primary">Review evidence →</span>
          </button>
        )}
      </section>
      {library.length > 0 && (
        <section aria-label="Analysis library">
          <h2>Analysis library</h2>
          <ul className="run-list">
            {library.map((item) => {
              const row = stats[item.run_id];
              return (
                <li key={item.run_id}>
                  <button
                    type="button"
                    onClick={() => props.onOpen(item.run_id)}
                  >
                    <code>{item.run_id}</code>
                    <span className="muted small">
                      {item.workflow} · {fmtDate(item.created_at)}
                    </span>
                    <span className="stat-row small">
                      <span>{row?.frames ?? "—"} frames</span>
                      <span>{row?.readings ?? "—"} readings</span>
                      <span>{row?.abstentions ?? "—"} abstentions</span>
                    </span>
                    <span className="btn-ghost">Review</span>
                  </button>
                </li>
              );
            })}
          </ul>
        </section>
      )}
      {technical.length > 0 && (
        <details className="tech-runs">
          <summary>
            Technical runs ({technical.length}) — storage and
            connectivity checks with no frame evidence
          </summary>
          <ul className="run-list compact">
            {technical.map((item) => (
              <li key={item.run_id}>
                <button
                  type="button"
                  onClick={() => props.onOpen(item.run_id)}
                >
                  <code>{item.run_id}</code>
                  <span className="muted small">
                    {item.workflow} · {fmtDate(item.created_at)}
                  </span>
                  <span className="btn-ghost">Open</span>
                </button>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

/* ---------------- Timeline ---------------- */

function Filmstrip(props: {
  runId: string;
  frames: ReviewFrame[];
  selected: number | null;
  onSelect: (index: number) => void;
}): JSX.Element {
  return (
    <div className="timeline" role="listbox" aria-label="Frame timeline">
      {props.frames.map((frame) => {
        const state = frameState(frame);
        const classes = ["tl-item", state.kind];
        if (frame.frame_index === props.selected) {
          classes.push("selected");
        }
        if (state.reviewed) classes.push("reviewed");
        const caption = state.kind === "numeric"
          ? String(frame.measurement?.value ?? "")
          : state.kind === "abstained" ? "Abstained" : "Missing";
        return (
          <button
            key={frame.frame_index}
            type="button"
            role="option"
            aria-selected={frame.frame_index === props.selected}
            className={classes.join(" ")}
            onClick={() => props.onSelect(frame.frame_index)}
            title={state.label}
          >
            {frame.image_available && frame.image_name ? (
              <img
                src={frameUrl(props.runId, frame.image_name)}
                alt={`Frame ${frame.frame_index}`}
                loading="lazy"
              />
            ) : (
              <span className="tl-missing">No image</span>
            )}
            <span className="tl-meta">
              #{frame.frame_index} · {fmtTime(frame.timestamp_seconds)}
            </span>
            <span className={`tl-state ${state.pill}`}>
              {state.reviewed ? "✓ " : ""}{caption}
            </span>
          </button>
        );
      })}
    </div>
  );
}

/* ---------------- Viewer + side panel ---------------- */

function Viewer(props: {
  runId: string;
  frame: ReviewFrame;
}): JSX.Element {
  const [zoom, setZoom] = useState(0);
  const [broken, setBroken] = useState(false);
  useEffect(() => {
    setZoom(0);
    setBroken(false);
  }, [props.frame.frame_index]);
  const level = ZOOM_STEPS[zoom] ?? 1;
  return (
    <div>
      <div className="viewer-bar">
        <span className="viewer-title">
          Frame #{props.frame.frame_index} ·{" "}
          {fmtTime(props.frame.timestamp_seconds)}
        </span>
        <span className="viewer-zoom" role="group" aria-label="Zoom">
          <button
            type="button"
            disabled={zoom <= 0}
            onClick={() => setZoom(zoom - 1)}
            aria-label="Zoom out"
          >
            −
          </button>
          <span>{level}×</span>
          <button
            type="button"
            disabled={zoom >= ZOOM_STEPS.length - 1}
            onClick={() => setZoom(zoom + 1)}
            aria-label="Zoom in"
          >
            +
          </button>
          <button type="button" onClick={() => setZoom(0)}>
            Reset
          </button>
        </span>
      </div>
      {props.frame.image_available && props.frame.image_name && !broken ? (
        <div className="viewer-scroll">
          <img
            className="viewer"
            style={{ transform: `scale(${level})` }}
            src={frameUrl(props.runId, props.frame.image_name)}
            alt={`Frame ${props.frame.frame_index}`}
            onError={() => setBroken(true)}
          />
        </div>
      ) : (
        <p className="muted">
          Retained frame image unavailable for this observation.
        </p>
      )}
      <p className="muted small">
        {props.frame.image_matches_record === true
          ? "Hash check passed: file matches the hash recorded with the response."
          : props.frame.image_matches_record === false
            ? "Hash mismatch: pixels differ from the response record."
            : "No hash record available for this frame."}
      </p>
    </div>
  );
}

function ObservationTab(props: { frame: ReviewFrame }): JSX.Element {
  const obs = props.frame.observation;
  const measurement = props.frame.measurement;
  if (obs === null) {
    return <p className="muted">No model observation retained.</p>;
  }
  const fields = obs.model_fields;
  const valid = obs.terminal_state === "valid_prediction";
  return (
    <div>
      <p>
        <span className={`pill ${valid ? "pill-ok" : "pill-pending"}`}>
          {valid ? "Valid prediction" : (obs.terminal_state ?? "Unknown")}
        </span>
      </p>
      <p className="big-reading-label">Predicted reading</p>
      <p className="big-reading">
        {fields.scale_display_verbatim ?? "—"}
      </p>
      <dl className="kv">
        <div>
          <dt>Container</dt>
          <dd>{fields.container_description ?? "—"}</dd>
        </div>
        <div>
          <dt>Readable</dt>
          <dd>
            {fields.scale_readable === null ? "unavailable"
              : String(fields.scale_readable)}
          </dd>
        </div>
        <div>
          <dt>Scale state</dt>
          <dd>{fields.scale_state ?? "unavailable"}</dd>
        </div>
        <div>
          <dt>Confidence</dt>
          <dd>{fields.confidence ?? "unavailable"}</dd>
        </div>
        <div>
          <dt>Notes</dt>
          <dd>{fields.ambiguity_notes ?? "—"}</dd>
        </div>
        <div>
          <dt>Pipeline value</dt>
          <dd>
            {measurement?.value === null || measurement?.value === undefined
              ? `none (${measurement?.value_status ?? "unavailable"})`
              : `${measurement.value} (${measurement.value_source ?? "unavailable"})`}
          </dd>
        </div>
      </dl>
      <details>
        <summary>Raw model observation</summary>
        <pre className="dump">{JSON.stringify(obs, null, 2)}</pre>
      </details>
    </div>
  );
}

function ReviewTab(props: {
  frame: ReviewFrame;
  saving: boolean;
  feedback: string | null;
  failed: boolean;
  onSave: (
    decision: ReviewDecision,
    value: number | null,
    notes: string | null,
  ) => void;
}): JSX.Element {
  const existing = props.frame.review;
  const modelValue = props.frame.measurement?.value ?? null;
  const [decision, setDecision] = useState<ReviewDecision>(
    existing?.decision ?? "confirmed");
  const [valueText, setValueText] = useState(
    existing?.value === null || existing?.value === undefined
      ? "" : String(existing.value));
  const [notes, setNotes] = useState(existing?.notes ?? "");
  const [localError, setLocalError] = useState<string | null>(null);

  useEffect(() => {
    setDecision(existing?.decision ?? "confirmed");
    setValueText(existing?.value === null || existing?.value === undefined
      ? "" : String(existing.value));
    setNotes(existing?.notes ?? "");
    setLocalError(null);
  }, [props.frame.frame_index]); // eslint-disable-line react-hooks/exhaustive-deps

  function submit(): void {
    let value: number | null = null;
    if (decision === "corrected") {
      const parsed = Number(valueText);
      if (valueText.trim() === "" || !Number.isFinite(parsed)) {
        setLocalError("A corrected review needs a finite number.");
        return;
      }
      value = parsed;
    }
    setLocalError(null);
    props.onSave(decision, value, notes.trim() === "" ? null : notes);
  }

  return (
    <div>
      <dl className="kv">
        <div>
          <dt>AI prediction (original)</dt>
          <dd>
            <strong>
              {props.frame.observation?.model_fields
                .scale_display_verbatim ?? "—"}
            </strong>
          </dd>
        </div>
      </dl>
      {existing && (
        <p className="saved-note">
          Saved: <strong>{existing.decision}</strong>
          {existing.value !== null && existing.value !== undefined
            ? ` (${existing.value})` : ""}
          {" · "}{existing.updated_at}
          {existing.notes ? ` — ${existing.notes}` : ""}
        </p>
      )}
      <fieldset className="decision-group">
        <legend>Review decision</legend>
        <label>
          <input
            type="radio"
            name={`decision-${props.frame.frame_index}`}
            checked={decision === "confirmed"}
            onChange={() => setDecision("confirmed")}
          />
          Confirm{modelValue !== null ? ` (use ${modelValue})` : ""}
        </label>
        <label>
          <input
            type="radio"
            name={`decision-${props.frame.frame_index}`}
            checked={decision === "corrected"}
            onChange={() => setDecision("corrected")}
          />
          Correct (enter new reading)
        </label>
        {decision === "corrected" && (
          <input
            className="correction-input"
            inputMode="decimal"
            aria-label="Corrected reading"
            placeholder={modelValue === null
              ? "e.g. 29" : `Model: ${modelValue}`}
            value={valueText}
            onChange={(event) => setValueText(event.target.value)}
          />
        )}
        <label>
          <input
            type="radio"
            name={`decision-${props.frame.frame_index}`}
            checked={decision === "rejected"}
            onChange={() => setDecision("rejected")}
          />
          Reject (not a valid reading)
        </label>
        <label>
          <input
            type="radio"
            name={`decision-${props.frame.frame_index}`}
            checked={decision === "uncertain"}
            onChange={() => setDecision("uncertain")}
          />
          Uncertain (not readable)
        </label>
      </fieldset>
      <label className="field">
        Reviewer notes (optional)
        <textarea
          value={notes}
          maxLength={1000}
          rows={3}
          placeholder="What did you see?"
          onChange={(event) => setNotes(event.target.value)}
        />
      </label>
      {localError && <p className="error">{localError}</p>}
      {props.feedback && (
        <p className={props.failed ? "error" : "success"}>
          {props.feedback}
        </p>
      )}
      <button
        type="button"
        className="btn-primary"
        disabled={props.saving}
        onClick={submit}
      >
        {props.saving ? "Saving…" : "✓ Save review"}
      </button>
    </div>
  );
}

function DetailsTab(props: { frame: ReviewFrame }): JSX.Element {
  const obs = props.frame.observation;
  if (obs === null) {
    return <p className="muted">No technical details retained.</p>;
  }
  return (
    <dl className="kv">
      <div><dt>Response ID</dt><dd className="mono">
        {obs.request_id ?? "unavailable"}
      </dd></div>
      <div><dt>Latency</dt><dd>
        {obs.latency_seconds === null || obs.latency_seconds === undefined
          ? "unavailable" : `${obs.latency_seconds.toFixed(2)} s`}
      </dd></div>
      <div><dt>Tokens</dt><dd>
        {obs.input_tokens ?? "—"} in / {obs.output_tokens ?? "—"} out
        {obs.reasoning_output_tokens === null ||
          obs.reasoning_output_tokens === undefined
          ? "" : ` (${obs.reasoning_output_tokens} reasoning)`}
      </dd></div>
      <div><dt>Response status</dt><dd>
        {obs.response_status ?? "unavailable"}
      </dd></div>
      <div><dt>Parse</dt><dd>
        {obs.parse_error === null ? "valid JSON" : obs.parse_error}
      </dd></div>
      <div><dt>Frame hash</dt><dd className="mono">
        {props.frame.image_sha256?.slice(0, 16) ?? "unavailable"}…
      </dd></div>
    </dl>
  );
}

/* ---------------- Bottom tabs ---------------- */

function RecordTab(props: {
  bundle: ReviewBundle;
  draft: RecipeDraft | null;
  onSelectFrame: (index: number) => void;
}): JSX.Element {
  const rows = useMemo(
    () => buildRecipeRows(props.bundle), [props.bundle]);
  const usable = rows.filter((row) => row.value !== null);
  const corrected = rows.filter(
    (row) => row.provenance === "human-corrected").length;
  const extracted = rows.filter(
    (row) => row.provenance === "ai-extracted").length;
  const abstained = props.bundle.frames.filter(
    (frame) => frameState(frame).kind === "abstained").length;
  return (
    <div className="record-grid">
      <div>
        <h3>Preparation record (draft)</h3>
        <p className="muted">
          Evidence-based record from video analysis. Values update
          as you save reviews.
        </p>
        <div className="table-wrap">
        <table className="recipe">
          <thead>
            <tr>
              <th>Time</th>
              <th>Reading</th>
              <th>Source</th>
              <th>Evidence</th>
            </tr>
          </thead>
          <tbody>
            {rows.filter((row) => row.value !== null
              || row.provenance !== "unknown").map((row) => (
              <tr key={row.frame_index}>
                <td>{fmtTime(row.timestamp_seconds)}</td>
                <td>{row.value === null ? "—" : row.value}</td>
                <td>
                  <span className={`pill ${row.pill}`}>
                    {row.provenance}
                  </span>
                </td>
                <td>
                  <button
                    type="button"
                    className="linklike"
                    onClick={() => props.onSelectFrame(row.frame_index)}
                  >
                    Frame #{row.frame_index}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        </div>
        {props.draft && props.draft.steps.length > 0 && (
          <div className="draft">
            <h4>Draft steps (AI text, pending review)</h4>
            <ol>
              {props.draft.steps.map((step) => (
                <li key={step.index}>{step.description}</li>
              ))}
            </ol>
          </div>
        )}
      </div>
      <aside>
        <div className="summary-card">
          <h4>Analysis summary</h4>
          <ul>
            <li><strong>{usable.length}</strong> usable readings</li>
            <li><strong>{corrected}</strong> human corrected</li>
            <li><strong>{extracted}</strong> AI extracted</li>
            <li><strong>{abstained}</strong> abstentions</li>
          </ul>
        </div>
        <div className="summary-card warn">
          <h4>Still unknown</h4>
          {props.draft && props.draft.unresolved_questions.length > 0 ? (
            <ul>
              {props.draft.unresolved_questions.map((item) => (
                <li key={item.code}>{item.message}</li>
              ))}
            </ul>
          ) : (
            <p className="muted">No unresolved list retained.</p>
          )}
        </div>
      </aside>
    </div>
  );
}

function ObservationsTab(props: {
  bundle: ReviewBundle;
  onSelectFrame: (index: number) => void;
}): JSX.Element {
  return (
    <div className="table-wrap">
    <table className="recipe">
      <thead>
        <tr>
          <th>Frame</th>
          <th>Time</th>
          <th>Outcome</th>
          <th>Verbatim</th>
          <th>Confidence</th>
          <th>Review</th>
        </tr>
      </thead>
      <tbody>
        {orderFrames(props.bundle.frames).map((frame) => (
          <tr key={frame.frame_index}>
            <td>
              <button
                type="button"
                className="linklike"
                onClick={() => props.onSelectFrame(frame.frame_index)}
              >
                #{frame.frame_index}
              </button>
            </td>
            <td>{fmtTime(frame.timestamp_seconds)}</td>
            <td>{frame.observation?.terminal_state ?? "—"}</td>
            <td>
              {frame.observation?.model_fields.scale_display_verbatim
                ?? "—"}
            </td>
            <td>
              {frame.observation?.model_fields.confidence ?? "—"}
            </td>
            <td>{frame.review?.decision ?? "pending"}</td>
          </tr>
        ))}
      </tbody>
    </table>
    </div>
  );
}

function ProvenanceTab(props: {
  bundle: ReviewBundle;
  summary: PipelineSummaryOil | null;
  report: FirstpassReport | null;
}): JSX.Element {
  const counts = useMemo(
    () => summarizeReview(props.bundle), [props.bundle]);
  const report = props.report;
  return (
    <div>
      <dl className="kv">
        <div><dt>Model</dt><dd>{report?.model ?? "unavailable"}</dd></div>
        <div><dt>Status</dt><dd>{props.bundle.status}</dd></div>
        <div>
          <dt>Frames</dt>
          <dd>
            {counts.frames} ({counts.numeric} readings,{" "}
            {counts.abstained} abstentions, {counts.missing} missing)
          </dd>
        </div>
        <div>
          <dt>Review progress</dt>
          <dd>{counts.reviewed} reviewed · {counts.pending} pending</dd>
        </div>
        <div><dt>Requests</dt><dd>
          {report?.requests_submitted ?? report?.requests_sent
            ?? "unavailable"}
        </dd></div>
        <div><dt>Tokens</dt><dd>
          {report ? (<span>
            {report.input_tokens ?? "—"} in /{" "}
            {report.output_tokens ?? "—"} out
            {report.reasoning_output_tokens === null ||
              report.reasoning_output_tokens === undefined
              ? "" : ` (${report.reasoning_output_tokens} reasoning)`}
            {" · "}{report.total_tokens ?? "—"} total
          </span>) : "unavailable"}
        </dd></div>
        <div><dt>Estimated cost</dt><dd>
          {report?.cost?.spent_cost_usd_estimate === null ||
            report?.cost?.spent_cost_usd_estimate === undefined
            ? "unavailable"
            : `$${report.cost.spent_cost_usd_estimate.toFixed(4)}`}
        </dd></div>
        <div><dt>Source</dt><dd className="mono">
          {props.bundle.source_sha256?.slice(0, 16) ?? "unavailable"}…
        </dd></div>
      </dl>
      {props.summary?.note && (
        <p className="muted">{props.summary.note}</p>
      )}
    </div>
  );
}

function ExportTab(props: { bundle: ReviewBundle }): JSX.Element {
  return (
    <div>
      <p className="muted">
        Download model values alongside human corrections.
      </p>
      <div className="btn-row">
        <button
          type="button"
          onClick={() => download(
            `${props.bundle.run_id}-reviews.csv`,
            exportReviewCsv(props.bundle), "text/csv")}
        >
          Download CSV
        </button>
        <button
          type="button"
          onClick={() => download(
            `${props.bundle.run_id}-reviews.json`,
            exportReviewJson(props.bundle), "application/json")}
        >
          Download JSON
        </button>
      </div>
    </div>
  );
}

/* ---------------- Workspace ---------------- */

type SideTab = "observation" | "review" | "details";
type BottomTab = "record" | "observations" | "provenance" | "export";

function Workspace(props: {
  runId: string;
  onBack: () => void;
}): JSX.Element {
  const [bundle, setBundle] = useState<ReviewBundle | null>(null);
  const [summary, setSummary] = useState<PipelineSummaryOil | null>(null);
  const [report, setReport] = useState<FirstpassReport | null>(null);
  const [draft, setDraft] = useState<RecipeDraft | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [sideTab, setSideTab] = useState<SideTab>("observation");
  const [bottomTab, setBottomTab] = useState<BottomTab>("record");
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setLoadError(null);
    (async () => {
      try {
        const data = await fetchReviewBundle(props.runId);
        if (cancelled) return;
        setBundle(data);
        const ordered = orderFrames(data.frames);
        const firstNumeric = ordered.find(
          (frame) => frame.measurement?.value_status === "ok"
            && typeof frame.measurement?.value === "number");
        setSelected(
          (firstNumeric ?? ordered[0] ?? null)?.frame_index ?? null);
        const status = await fetchAnalysis(props.runId);
        if (cancelled) return;
        const workflow = status.workflow;
        const wantPipeline = workflow === WORKFLOW_PIPELINE
          || workflow === WORKFLOW_META;
        const wantOffline = workflow === WORKFLOW_OFFLINE;
        const [summaryRes, reportRes, draftRes] = await Promise.allSettled([
          wantPipeline
            ? fetchArtifactJson<PipelineSummaryOil>(
              props.runId, "pipeline_summary.json")
            : wantOffline
              ? fetchArtifactJson<PipelineSummaryOil>(
                props.runId, "offline_summary.json")
              : Promise.resolve(null),
          wantPipeline
            ? fetchArtifactJson<FirstpassReport>(
              props.runId, "firstpass_report.json")
            : Promise.resolve(null),
          wantPipeline || wantOffline
            ? fetchArtifactJson<RecipeDraft>(
              props.runId, "recipe_draft.json")
            : Promise.resolve(null),
        ]);
        if (cancelled) return;
        if (summaryRes.status === "fulfilled") setSummary(summaryRes.value);
        if (reportRes.status === "fulfilled") setReport(reportRes.value);
        if (draftRes.status === "fulfilled") setDraft(draftRes.value);
      } catch (err) {
        if (!cancelled) setLoadError(errorText(err));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [props.runId]);

  const frames = useMemo(
    () => (bundle ? orderFrames(bundle.frames) : []), [bundle]);
  const current: ReviewFrame | null = useMemo(() => (
    frames.find((frame) => frame.frame_index === selected) ?? null
  ), [frames, selected]);
  const counts = useMemo(
    () => (bundle ? summarizeReview(bundle)
      : { frames: 0, reviewed: 0, pending: 0, numeric: 0,
        abstained: 0, missing: 0 }),
    [bundle]);

  function select(index: number): void {
    setSelected(index);
    setFeedback(null);
  }

  function step(delta: number): void {
    if (frames.length === 0) return;
    const position = frames.findIndex(
      (frame) => frame.frame_index === selected);
    const next = frames[
      (position < 0 ? 0 : position + delta + frames.length)
      % frames.length];
    if (next) select(next.frame_index);
  }

  async function handleSave(
    decision: ReviewDecision,
    value: number | null,
    notes: string | null,
  ): Promise<void> {
    if (selected === null) return;
    setSaving(true);
    setFailed(false);
    setFeedback(null);
    try {
      const saved = await saveReview(props.runId, {
        frame_index: selected,
        decision,
        value,
        notes,
      });
      setBundle((prev) => {
        if (prev === null) return prev;
        return {
          ...prev,
          frames: prev.frames.map((frame) => (
            frame.frame_index === selected
              ? { ...frame, review: saved } : frame
          )),
          reviews: {
            ...prev.reviews,
            [String(selected)]: saved,
          },
        };
      });
      setFeedback(
        `Review saved: frame #${selected} marked ${decision}.`);
    } catch (err) {
      setFailed(true);
      setFeedback(errorText(err));
    } finally {
      setSaving(false);
    }
  }

  if (loading) return <p className="muted">Loading evidence…</p>;
  if (loadError || bundle === null) {
    return (
      <div>
        <p className="error">{loadError ?? "Evidence unavailable."}</p>
        <button type="button" onClick={props.onBack}>
          Back to analyses
        </button>
      </div>
    );
  }
  const sliderIndex = Math.max(0, frames.findIndex(
    (frame) => frame.frame_index === selected));
  return (
    <div className="workspace">
      <button type="button" className="linklike" onClick={props.onBack}>
        ← All analyses
      </button>
      <div className="ws-head">
        <div>
          <h2>Evidence review</h2>
          <p className="muted small">
            Real video analysis · <code>{bundle.run_id}</code> ·{" "}
            {bundle.status}
          </p>
        </div>
        <dl className="metrics">
          <div><dt>Frames</dt><dd>{counts.frames}</dd></div>
          <div><dt>Readings</dt><dd>{counts.numeric}</dd></div>
          <div><dt>Abstentions</dt><dd>{counts.abstained}</dd></div>
          <div><dt>Reviewed</dt><dd>{counts.reviewed}</dd></div>
        </dl>
      </div>
      {bundle.missing.length > 0 && (
        <div className="banner-warn">
          <strong>Incomplete evidence:</strong>
          <ul>{bundle.missing.map((note) => <li key={note}>{note}</li>)}</ul>
        </div>
      )}
      <div className="ws-main">
        <section className="card viewer-card" aria-label="Evidence viewer">
          {current && (
            <Viewer runId={bundle.run_id} frame={current} />
          )}
          <div className="slider-row">
            <button
              type="button"
              aria-label="Previous frame"
              onClick={() => step(-1)}
            >
              ‹
            </button>
            <input
              type="range"
              min={0}
              max={Math.max(0, frames.length - 1)}
              value={sliderIndex}
              aria-label="Frame timeline position"
              onChange={(event) => {
                const next = frames[Number(event.target.value)];
                if (next) select(next.frame_index);
              }}
            />
            <button
              type="button"
              aria-label="Next frame"
              onClick={() => step(1)}
            >
              ›
            </button>
          </div>
          <div className="filmstrip-head">
            <h3>Frames</h3>
            <span>{frames.length} sampled · scroll to browse</span>
          </div>
          <Filmstrip
            runId={bundle.run_id}
            frames={frames}
            selected={selected}
            onSelect={select}
          />
        </section>
        <section className="card side-card" aria-label="Frame detail">
          <div className="tabs small" role="tablist" aria-label="Detail tabs">
            {(["observation", "review", "details"] as const).map((tab) => (
              <button
                key={tab}
                type="button"
                role="tab"
                aria-selected={sideTab === tab}
                className={sideTab === tab ? "active" : ""}
                onClick={() => setSideTab(tab)}
              >
                {tab === "observation" ? "Observation"
                  : tab === "review" ? "Your review" : "Details"}
              </button>
            ))}
          </div>
          {current === null && (
            <p className="muted">Select a frame to inspect it.</p>
          )}
          {current !== null && sideTab === "observation" && (
            <ObservationTab frame={current} />
          )}
          {current !== null && sideTab === "review" && (
            <ReviewTab
              frame={current}
              saving={saving}
              feedback={feedback}
              failed={failed}
              onSave={handleSave}
            />
          )}
          {current !== null && sideTab === "details" && (
            <DetailsTab frame={current} />
          )}
        </section>
      </div>
      <section className="card" aria-label="Results">
        <div className="tabs small" role="tablist" aria-label="Result tabs">
          {(["record", "observations", "provenance", "export"] as const)
            .map((tab) => (
              <button
                key={tab}
                type="button"
                role="tab"
                aria-selected={bottomTab === tab}
                className={bottomTab === tab ? "active" : ""}
                onClick={() => setBottomTab(tab)}
              >
                {tab === "record" ? "Preparation record"
                  : tab === "observations"
                    ? `Observations (${frames.length})`
                    : tab === "provenance" ? "Provenance" : "Export"}
              </button>
            ))}
        </div>
        {bottomTab === "record" && (
          <RecordTab
            bundle={bundle}
            draft={draft}
            onSelectFrame={(index) => {
              select(index);
              setSideTab("observation");
            }}
          />
        )}
        {bottomTab === "observations" && (
          <ObservationsTab bundle={bundle} onSelectFrame={select} />
        )}
        {bottomTab === "provenance" && (
          <ProvenanceTab
            bundle={bundle}
            summary={summary}
            report={report}
          />
        )}
        {bottomTab === "export" && <ExportTab bundle={bundle} />}
      </section>
    </div>
  );
}

export default function ReviewApp(props: {
  initialRunId?: string | null;
}): JSX.Element {
  const [selected, setSelected] = useState<string | null>(
    props.initialRunId ?? null);
  if (selected === null) {
    return <Landing onOpen={setSelected} />;
  }
  return <Workspace runId={selected} onBack={() => setSelected(null)} />;
}
