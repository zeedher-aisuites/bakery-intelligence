import { useEffect, useRef, useState } from "react";
import {
  ProgressInfo,
  UploadInfo,
  createUploadAnalysis,
  fetchProgress,
  formatBytes,
  uploadVideo,
} from "./api";
import {
  MAX_AUTHORIZED_COST_USD,
  META_MODEL,
  OFFLINE_WORKFLOW,
  buildMetaGrant,
  formatDuration,
} from "./upload";

const STAGE_LABELS: Record<string, string> = {
  queued: "Queued — waiting for the local worker",
  running: "Processing",
  extracted: "Frames extracted — analysis running",
  firstpass_running: "Analyzing frames",
  firstpass_done: "Analysis finished — building record",
  completed: "Completed",
  failed: "Failed",
};

function stageLabel(progress: ProgressInfo | null): string {
  if (!progress) return "Queued";
  if (progress.status === "completed") return "Completed";
  if (progress.status === "failed") return "Failed";
  return STAGE_LABELS[progress.stage] ?? progress.stage;
}

export default function UploadWorkflow(props: {
  onOpenReview: (runId: string) => void;
  onViewQueue?: (runId: string) => void;
}): JSX.Element {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState<number | null>(null);
  const [upload, setUpload] = useState<UploadInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirmPaid, setConfirmPaid] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [job, setJob] = useState<{
    run_id: string; workflow: string;
  } | null>(null);
  const [progress, setProgress] = useState<ProgressInfo | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!job) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const next = await fetchProgress(job.run_id);
        if (!cancelled) setProgress(next);
      } catch {
        /* keep last known progress */
      }
    };
    void tick();
    const timer = window.setInterval(() => void tick(), 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [job]);

  const terminal = progress?.status === "completed"
    || progress?.status === "failed";

  async function startUpload(selected: File): Promise<void> {
    setFile(selected);
    setError(null);
    setUpload(null);
    setJob(null);
    setProgress(null);
    setConfirmPaid(false);
    setUploading(0);
    try {
      const info = await uploadVideo(selected, setUploading);
      setUpload(info);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setUploading(null);
    }
  }

  async function startAnalysis(workflow: string): Promise<void> {
    if (!upload || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const grant = workflow === OFFLINE_WORKFLOW
        ? null
        : buildMetaGrant(upload.estimated_frames);
      const created = await createUploadAnalysis(
        upload.upload_id, workflow,
        grant as unknown as Record<string, unknown> | null);
      setJob({ run_id: created.run_id, workflow: created.workflow });
      setProgress(null);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not start analysis.");
    } finally {
      setSubmitting(false);
    }
  }

  function reset(): void {
    setFile(null);
    setUpload(null);
    setError(null);
    setJob(null);
    setProgress(null);
    setConfirmPaid(false);
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <div className="upload-flow">
      <section className="card flow-card">
        <div className="flow-head">
          <span className="step-num">1</span>
          <div>
            <h2>Upload a preparation video</h2>
            <p className="muted">
              MP4 recordings up to about 12 minutes. Uploading only
              validates and stages the file — it never starts paid
              inference.
            </p>
          </div>
        </div>
        <div
          className={`dropzone${dragging ? " dragging" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            const dropped = event.dataTransfer.files?.[0];
            if (dropped) void startUpload(dropped);
          }}
        >
          <input
            ref={inputRef}
            type="file"
            accept="video/mp4,.mp4"
            aria-label="Choose an MP4 video"
            onChange={(event) => {
              const selected = event.target.files?.[0];
              if (selected) void startUpload(selected);
            }}
          />
          <strong>Drop an MP4 here, or click to choose</strong>
          <span className="muted small">
            The original file is stored unchanged with a SHA-256
            integrity record.
          </span>
        </div>
        {file && (
          <p className="selection">
            Selected: <strong>{file.name}</strong>{" "}
            ({formatBytes(file.size)})
          </p>
        )}
        {uploading !== null && (
          <div className="progress-block">
            <div className="progress-track">
              <div style={{ width: `${uploading}%` }} />
            </div>
            <span className="muted small">
              Uploading and validating… {uploading}%
            </span>
          </div>
        )}
        {error && !job && (
          <p className="error" role="alert">{error}</p>
        )}
        {upload && (
          <dl className="summary-grid">
            <div><dt>Duration</dt>
              <dd>{formatDuration(upload.duration_seconds)}</dd></div>
            <div><dt>Resolution</dt>
              <dd>{upload.width} × {upload.height}</dd></div>
            <div><dt>Source frames</dt>
              <dd>{upload.frame_count} at {upload.fps.toFixed(1)} FPS</dd></div>
            <div><dt>SHA-256</dt>
              <dd className="mono">{upload.sha256.slice(0, 16)}…</dd></div>
            <div><dt>Status</dt>
              <dd>{upload.deduplicated
                ? "Ready (identical upload reused)" : "Ready"}</dd></div>
          </dl>
        )}
      </section>

      {upload && !job && (
        <section className="card flow-card">
          <div className="flow-head">
            <span className="step-num">2</span>
            <div>
              <h2>Configure analysis</h2>
              <p className="muted">
                Local preprocessing (frame extraction at 1 FPS) is
                free. Model inference is billable and only starts
                after your explicit authorization below.
              </p>
            </div>
          </div>
          <dl className="summary-grid">
            <div><dt>Provider / model</dt>
              <dd>Meta · {META_MODEL}</dd></div>
            <div><dt>Sampling</dt><dd>1 FPS</dd></div>
            <div><dt>Projected frames</dt>
              <dd>{upload.estimated_frames}</dd></div>
            <div><dt>Estimated cost</dt>
              <dd>{upload.estimate
                ? `≈ $${upload.estimate.estimated_cost_usd.toFixed(2)} (nonbinding)`
                : "Unavailable — cost is uncertain"}</dd></div>
            <div><dt>Authorized maximum</dt>
              <dd>USD {MAX_AUTHORIZED_COST_USD.toFixed(2)} hard cap</dd></div>
          </dl>
          {upload.estimate && (
            <p className="muted small">{upload.estimate.basis}</p>
          )}
          {upload.exceeds_frame_cap && (
            <p className="error" role="alert">
              This recording projects more than 1,000 sampled
              frames, the current pipeline limit. Shorten the
              video before starting analysis.
            </p>
          )}
          <div className="flow-actions">
            <button
              type="button"
              className="btn-ghost"
              disabled={submitting || upload.exceeds_frame_cap}
              onClick={() => void startAnalysis(OFFLINE_WORKFLOW)}
            >
              Run free offline preview
            </button>
            <span className="muted small">
              Simulated observations, clearly labeled. No Meta
              request is sent.
            </span>
          </div>
          <div className="auth-box">
            <h3>Paid Meta analysis</h3>
            <p className="muted">
              Model {META_MODEL} · {upload.estimated_frames} frames
              {" "}·{" "}
              {upload.estimate
                ? `estimated ≈ $${upload.estimate.estimated_cost_usd.toFixed(2)} (nonbinding)`
                : "cost estimate unavailable"}
            </p>
            <p className="muted">
              Authorized maximum:{" "}
              <strong>
                USD {MAX_AUTHORIZED_COST_USD.toFixed(2)}
              </strong>{" "}
              — strict cap. Sending stops when known spend
              reaches this limit; the estimate above does not
              change the cap.
            </p>
            <label className="confirm-row">
              <input
                type="checkbox"
                checked={confirmPaid}
                onChange={(event) => setConfirmPaid(event.target.checked)}
              />
              <span>
                I authorize paid Meta inference for this video
                with the configuration above, up to a maximum
                of USD {MAX_AUTHORIZED_COST_USD.toFixed(2)}.
              </span>
            </label>
            <button
              type="button"
              className="btn-primary"
              disabled={!confirmPaid || submitting
                || upload.exceeds_frame_cap}
              onClick={() => void startAnalysis("video_pipeline_meta")}
            >
              {submitting ? "Starting…" : "Authorize and start analysis"}
            </button>
          </div>
          {error && <p className="error" role="alert">{error}</p>}
        </section>
      )}

      {job && (
        <section className="card flow-card">
          <div className="flow-head">
            <span className="step-num">3</span>
            <div>
              <h2>Analysis progress</h2>
              <p className="muted">
                {job.workflow === OFFLINE_WORKFLOW
                  ? "Offline preview — simulated observations."
                  : "Meta Muse Spark 1.3 — real inference."}
                {" "}Run <code>{job.run_id}</code>
              </p>
            </div>
          </div>
          <p className="stage-label">{stageLabel(progress)}</p>
          <div className="progress-track large">
            <div style={{
              width: `${progress?.percent ?? (terminal ? 100 : 0)}%`,
            }} />
          </div>
          <dl className="summary-grid">
            <div><dt>Frames processed</dt>
              <dd>{progress?.frames_processed ?? "—"}
                {progress?.frames_total
                  ? ` of ${progress.frames_total}` : ""}</dd></div>
            <div><dt>Progress</dt>
              <dd>{progress?.percent !== null
                && progress?.percent !== undefined
                ? `${progress.percent}%`
                : "Not available yet"}</dd></div>
            <div><dt>Last update</dt>
              <dd>{progress?.updated_at ?? "—"}</dd></div>
          </dl>
          {progress?.status === "queued" && (
            <p className="muted small">
              Waiting for the local worker. Free jobs are processed
              by the standard worker; paid Meta jobs only by the
              secure billable launcher after authorization.
            </p>
          )}
          {progress?.status === "failed" && (
            <p className="error" role="alert">
              {progress.error ?? "Analysis failed."} Paid jobs are
              never retried automatically.
            </p>
          )}
          <div className="flow-actions">
            {progress?.status === "completed" && (
              <button
                type="button"
                className="btn-primary"
                onClick={() => props.onOpenReview(job.run_id)}
              >
                Open evidence review
              </button>
            )}
            {props.onViewQueue && (
              <button
                type="button"
                className="btn-ghost"
                onClick={() => props.onViewQueue?.(job.run_id)}
              >
                View in processing queue
              </button>
            )}
            <button type="button" className="btn-ghost" onClick={reset}>
              Analyze another video
            </button>
          </div>
          <p className="muted small">
            Refreshing this page does not lose the job: state is
            stored durably and polling resumes after reload from
            the Analyses library.
          </p>
        </section>
      )}
    </div>
  );
}
