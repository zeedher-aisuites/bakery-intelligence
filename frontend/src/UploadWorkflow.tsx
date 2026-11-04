import { useEffect, useRef, useState } from "react";
import {
  ProgressInfo,
  UploadInfo,
  createRecipe,
  createUploadAnalysis,
  fetchProgress,
  formatBytes,
  uploadVideo,
} from "./api";
import {
  META_WORKFLOW,
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
  onOpenRecipe?: (recipeId: string) => void;
}): JSX.Element {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState<number | null>(null);
  const [upload, setUpload] = useState<UploadInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [recipeName, setRecipeName] = useState("");
  const [language, setLanguage] = useState("en");
  const [confirmPaid, setConfirmPaid] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [job, setJob] = useState<{
    run_id: string; workflow: string; recipe_id: string | null;
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

  async function startRecipeAnalysis(): Promise<void> {
    if (!upload || submitting) return;
    const name = recipeName.trim();
    if (!name) {
      setError("Enter a recipe name before generating the recipe.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const grant = buildMetaGrant(upload.estimated_frames);
      const created = await createUploadAnalysis(
        upload.upload_id, META_WORKFLOW,
        grant as unknown as Record<string, unknown> | null);
      let recipeId: string | null = null;
      try {
        const record = await createRecipe({
          name,
          language,
          source_run_id: created.run_id,
        });
        recipeId = record.recipe_id;
      } catch (err) {
        setError(
          err instanceof Error
            ? `Analysis started, but the recipe record could not be created (${err.message}).`
            : "Analysis started, but the recipe record could not be created.",
        );
      }
      setJob({
        run_id: created.run_id,
        workflow: created.workflow,
        recipe_id: recipeId,
      });
      setProgress(null);
      if (recipeId && props.onOpenRecipe) {
        props.onOpenRecipe(recipeId);
      }
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

  function changeVideo(): void {
    setFile(null);
    setUpload(null);
    setError(null);
    setJob(null);
    setProgress(null);
    setConfirmPaid(false);
    if (inputRef.current) inputRef.current.value = "";
    inputRef.current?.click();
  }

  const canSubmit = upload !== null
    && !upload.exceeds_frame_cap
    && recipeName.trim() !== ""
    && confirmPaid
    && !submitting;

  return (
    <div className="upload-flow">
      <div>
        <h2>New recipe</h2>
        <p className="muted">Turn your cooking video into a recipe.</p>
      </div>
      <section className="card flow-card">
        <div className="flow-head">
          <span className="step-num">1</span>
          <div>
            <h2>Upload video</h2>
            <p className="muted">MP4 recordings up to about 12 minutes.</p>
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
        </div>
        {file && (
          <p className="selection">
            Selected: <strong>{file.name}</strong>{" "}
            ({formatBytes(file.size)})
          </p>
        )}
        <label className="field">
          <span>Recipe Name</span>
          <input
            type="text"
            value={recipeName}
            maxLength={120}
            placeholder="e.g. XL Cinnamon Rolls"
            aria-label="Recipe Name"
            onChange={(event) => setRecipeName(event.target.value)}
          />
        </label>
        <label className="field">
          <span>Recipe Language</span>
          <select
            value={language}
            aria-label="Recipe Language"
            onChange={(event) => setLanguage(event.target.value)}
          >
            <option value="en">English</option>
            <option value="es">Español</option>
          </select>
        </label>
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
          <>
            <dl className="summary-grid">
              <div><dt>Duration</dt>
                <dd>{formatDuration(upload.duration_seconds)}</dd></div>
              <div><dt>Resolution</dt>
                <dd>{upload.width} × {upload.height}</dd></div>
              <div><dt>File size</dt>
                <dd>{formatBytes(upload.bytes)}</dd></div>
              <div><dt>Status</dt>
                <dd>{upload.exceeds_frame_cap
                  ? "Too long for analysis" : "Ready"}</dd></div>
            </dl>
            {upload.exceeds_frame_cap && (
              <p className="error" role="alert">
                This video is too long for analysis. Please use a
                shorter recording.
              </p>
            )}
            <div className="flow-actions">
              <button
                type="button"
                className="btn-ghost"
                onClick={changeVideo}
              >
                Change video
              </button>
            </div>
          </>
        )}
      </section>

      {upload && !job && !upload.exceeds_frame_cap && (
        <section className="card flow-card">
          <div className="flow-head">
            <span className="step-num">2</span>
            <div>
              <h2>AI processing</h2>
            </div>
          </div>
          <p className="muted">
            AI will analyze your video and prepare a recipe with
            ingredients and preparation steps for your review.
          </p>
          <label className="confirm-row">
            <input
              type="checkbox"
              checked={confirmPaid}
              onChange={(event) => setConfirmPaid(event.target.checked)}
            />
            <span>
              I authorize AI processing of this video and understand
              that it may incur processing costs.
            </span>
          </label>
          <div className="flow-actions">
            <button
              type="button"
              className="btn-primary"
              disabled={!canSubmit}
              onClick={() => void startRecipeAnalysis()}
            >
              {submitting ? "Starting…" : "Generate my recipe"}
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
                Run <code>{job.run_id}</code>
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
              Waiting for the local worker to pick up the job.
            </p>
          )}
          {progress?.status === "failed" && (
            <p className="error" role="alert">
              {progress.error ?? "Analysis failed."} Paid jobs are
              never retried automatically.
            </p>
          )}
          <div className="flow-actions">
            {job.recipe_id && props.onOpenRecipe && (
              <button
                type="button"
                className="btn-primary"
                onClick={() => props.onOpenRecipe?.(job.recipe_id as string)}
              >
                Open recipe
              </button>
            )}
            {progress?.status === "completed" && (
              <button
                type="button"
                className="btn-ghost"
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
        </section>
      )}
    </div>
  );
}
