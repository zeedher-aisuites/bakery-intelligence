import { useEffect, useMemo, useState } from "react";
import {
  fetchObservationQueue,
  fetchRecipeEvidence,
  fetchReadiness,
  saveObservationCorrection,
  undoObservationCorrection,
  type EvidenceFrame,
  type ObservationQueue,
  type ReviewTask,
} from "./api";
import FrameViewer from "./FrameViewer";
import {
  correctionResult,
  correctionStatus,
  nextTaskId,
  reviewProgress,
  taskLabel,
  taskModelValues,
} from "./observation";
import { intervalLabel } from "./recipe";

function reviewStartLabel(task: ReviewTask): string {
  if (task.start_seconds === null) return "unknown time";
  return intervalLabel(task.start_seconds, task.end_seconds);
}

function CountsHelp(props: { onClose: () => void }): JSX.Element {
  return (
    <div
      className="modal-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) props.onClose();
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="What the counters mean"
        className="modal"
        onKeyDown={(event) => {
          if (event.key === "Escape") props.onClose();
        }}
        tabIndex={-1}
        ref={(node) => node?.focus()}
      >
        <h4>What the counters mean</h4>
        <p className="muted small">
          <strong>Findings</strong> are measurements the model could not
          settle alone: missing, uncertain, or conflicting readings.
          Each finding needs one human decision.
        </p>
        <p className="muted small">
          <strong>Resolved</strong> findings have a saved decision:
          a correction, a confirmation, or a cannot-determine note.
        </p>
        <p className="muted small">
          <strong>Set aside</strong> findings were deferred with Review
          later. They still need a decision eventually.
        </p>
        <p className="muted small">
          <strong>Frames, readings, abstentions, reviewed</strong> (top)
          describe the video analysis itself: sampled frames, usable
          numeric readings, frames where the model declined to read,
          and frames with a saved frame-level review.
        </p>
        <p className="muted small">
          A scale reading is an observation, not a verified ingredient
          quantity.
        </p>
        <div className="flow-actions">
          <button type="button" className="btn-primary" onClick={props.onClose}>
            Got it
          </button>
        </div>
      </div>
    </div>
  );
}

export default function ObservationReview(props: {
  runId: string;
  onSelectFrame?: (frameIndex: number) => void;
  onGenerated?: () => void;
  showTasks?: boolean;
  showGeneration?: boolean;
  onProgressChange?: (progress: {
    open: number; deferred: number; total: number;
  }) => void;
  onChanged?: () => void;
  reviewer: string;
}): JSX.Element {
  const [queue, setQueue] = useState<ObservationQueue | null>(null);
  const [frames, setFrames] = useState<EvidenceFrame[]>([]);
  const [values, setValues] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showHelp, setShowHelp] = useState(false);
  const [ready, setReady] = useState<{
    open: number; deferred: number; blocking: string[]; live: boolean;
  } | null>(null);
  const [mode, setMode] = useState("simulation");
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  const frameByIndex = useMemo(
    () => new Map(frames.map((frame) => [frame.frame_index, frame])),
    [frames]);

  async function load(): Promise<void> {
    const [next, evidence, readiness] = await Promise.all([
      fetchObservationQueue(props.runId),
      fetchRecipeEvidence(props.runId).catch(() => ({ frames: [] })),
      fetchReadiness(props.runId).catch(() => null),
    ]);
    setQueue(next);
    setFrames(evidence.frames);
    setSelectedId((current) => nextTaskId(next.tasks, current, showAll));
    setReady(readiness === null ? null : {
      open: readiness.open_tasks,
      deferred: readiness.deferred ?? 0,
      blocking: readiness.blocking_errors ?? [],
      live: (readiness.live_available ?? true)
        && (readiness.live_generation ?? true),
    });
    setError(null);
  }

  useEffect(() => {
    setQueue(null);
    setValues({});
    setError(null);
    setShowAll(false);
    setSelectedId(null);
    setReady(null);
    setMode("simulation");
    setConfirmed(false);
    setNotice(null);
    load().catch((err: unknown) => {
      setError(err instanceof Error ? err.message : "Findings unavailable.");
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.runId]);

  const tasks = queue?.tasks ?? [];
  const progress = useMemo(
    () => reviewProgress(tasks), [tasks]);
  const { open, deferred, total } = progress;

  useEffect(() => {
    props.onProgressChange?.({ open, deferred, total });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, deferred, total]);

  const visible = showAll
    ? tasks
    : tasks.filter((task) => task.status === "open"
      || task.status === "deferred");
  const order = useMemo(
    () => new Map(visible.map((task, index) => [task.task_id, index])),
    [visible]);
  const selectedTask = visible.find((task) => task.task_id === selectedId)
    ?? visible[0] ?? null;
  const selectedIndex = selectedTask === null
    ? -1 : order.get(selectedTask.task_id) ?? -1;

  function stepTask(delta: -1 | 1): void {
    if (visible.length === 0) return;
    const next = visible[
      (selectedIndex < 0 ? 0 : selectedIndex + delta + visible.length)
      % visible.length];
    if (next) selectTask(next.task_id);
  }

  function selectTask(taskId: string): void {
    const task = tasks.find((item) => item.task_id === taskId);
    const human = task?.correction?.human_value;
    if (typeof human === "number" && Number.isFinite(human)) {
      setValues((prev) => ({ ...prev, [taskId]: String(human) }));
    }
    setSelectedId(taskId);
    setNotice(null);
  }

  function parsedValue(taskId: string): number | null {
    const raw = (values[taskId] ?? "").trim();
    if (raw === "") return null;
    const parsed = Number(raw);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function isValueValid(taskId: string): boolean {
    const raw = (values[taskId] ?? "").trim();
    return raw !== "" && Number.isFinite(Number(raw));
  }

  const canAct = props.reviewer.trim() !== "";

  async function act(taskId: string, action: string): Promise<void> {
    if (action === "correct" && !isValueValid(taskId)) {
      setError("Enter a finite number to save a correction.");
      return;
    }
    const value = action === "correct" ? parsedValue(taskId) : null;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      await saveObservationCorrection(
        props.runId, taskId, action, value, props.reviewer);
      await load();
      // Keep the just-saved finding in view so the chef sees the
      // exact persisted result instead of losing it from the list.
      setShowAll(true);
      setSelectedId(taskId);
      props.onChanged?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed.");
    } finally {
      setBusy(false);
    }
  }

  async function undo(taskId: string): Promise<void> {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const result = await undoObservationCorrection(
        props.runId, taskId, props.reviewer);
      await load();
      props.onChanged?.();
      if (result.correction === null) {
        setNotice("Undid the last change — this finding needs "
          + "a decision again.");
      } else {
        setNotice("Undid the last change — the previous decision "
          + "is effective again.");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Undo failed.");
    } finally {
      setBusy(false);
    }
  }

  async function generate(): Promise<void> {
    const { generateRecipe } = await import("./api");
    setBusy(true);
    setError(null);
    try {
      await generateRecipe(props.runId, {
        mode,
        confirmed: mode === "live" ? confirmed : false,
        max_rounds: 1,
      });
      setConfirmed(false);
      props.onGenerated?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed.");
    } finally {
      setBusy(false);
    }
  }

  if (queue === null) {
    if (error) return <p className="error">{error}</p>;
    return <p className="muted">Loading findings…</p>;
  }

  return (
    <div>
      <div className="queue-head">
        <div>
          <h3>
            Review findings{" "}
            <button
              type="button"
              className="help-dot"
              aria-label="What the counters mean"
              onClick={() => setShowHelp(true)}
            >
              ?
            </button>
          </h3>
          <p className="muted">
            {`${open} of ${total} findings still need your decision. `}
            {showAll
              ? "Reviewed findings included."
              : "Reviewed findings hidden until requested. "}
            {`${deferred} set aside for later.`}
          </p>
        </div>
        <label className="confirm-row">
          <input
            type="checkbox"
            checked={showAll}
            onChange={(event) => {
              const nextShow = event.target.checked;
              setShowAll(nextShow);
              setSelectedId((current) => nextTaskId(
                tasks, current, nextShow));
            }}
          />
          <span>Reviewed findings</span>
        </label>
      </div>
      {showHelp && <CountsHelp onClose={() => setShowHelp(false)} />}
      {error && <p className="error" role="alert">{error}</p>}
      {notice && <p className="muted" role="status">{notice}</p>}

      {visible.length === 0 ? (
        <p className="muted">
          No findings need attention.{" "}
          {!showAll && total > 0 && "Reviewed findings are hidden. "}
          {!showAll && total === 0 && "Nothing was flagged in this video."}
        </p>
      ) : (
        <div className="finder-nav" role="list" aria-label="Findings">
          <div className="finder-slot">
            {selectedIndex > 0 ? (
              <FindingCard
                task={visible[selectedIndex - 1]}
                relation="Previous finding"
                onOpen={() => stepTask(-1)}
              />
            ) : (
              <span
                className="finder-card placeholder"
                aria-hidden="true"
              />
            )}
          </div>
          <div className="finder-slot">
            {selectedTask && (
              <FindingCard
                task={selectedTask}
                relation={`Finding ${selectedIndex + 1} of ${visible.length}`}
                current
              />
            )}
          </div>
          <div className="finder-slot">
            {selectedIndex >= 0 && selectedIndex < visible.length - 1 ? (
              <FindingCard
                task={visible[selectedIndex + 1]}
                relation="Next finding"
                onOpen={() => stepTask(1)}
              />
            ) : (
              <span
                className="finder-card placeholder"
                aria-hidden="true"
              />
            )}
          </div>
        </div>
      )}

      {selectedTask && props.showTasks !== false && (
        <TaskDetail
          runId={props.runId}
          task={selectedTask}
          position={selectedIndex}
          total={visible.length}
          frameByIndex={frameByIndex}
          value={values[selectedTask.task_id] ?? ""}
          onValueChange={(next) => setValues((prev) => ({
            ...prev, [selectedTask.task_id]: next,
          }))}
          valueValid={isValueValid(selectedTask.task_id)}
          canAct={canAct}
          busy={busy}
          onPrev={() => stepTask(-1)}
          onNext={() => stepTask(1)}
          onAct={(action) => void act(selectedTask.task_id, action)}
          onUndo={() => void undo(selectedTask.task_id)}
          onSelectFrame={props.onSelectFrame}
        />
      )}

      {props.showGeneration !== false && (
        <div className="flow-actions">
          <button
            type="button" className="btn-primary"
            disabled={busy || (mode === "live" && !ready?.live)}
            onClick={() => void generate()}
          >
            {mode === "live" ? "Generate recipe" : "Generate recipe (Simulation)"}
          </button>
          <label className="confirm-row">
            <input
              type="checkbox" checked={mode === "live"}
              onChange={(event) => setMode(event.target.checked ? "live" : "simulation")}
            />
            <span>Use live AI</span>
          </label>
        </div>
      )}
    </div>
  );
}

function FindingCard(props: {
  task: ReviewTask;
  relation: string;
  current?: boolean;
  onOpen?: () => void;
}): JSX.Element {
  const inner = (
    <>
      <span className="muted small">{props.relation}</span>
      <strong>{taskLabel(props.task.state)}</strong>
      <span className="muted small">
        {reviewStartLabel(props.task)} ·{" "}
        {props.task.frame_refs.length} frame
        {props.task.frame_refs.length === 1 ? "" : "s"}
      </span>
      <span className="pill pill-queued">{correctionStatus(props.task)}</span>
    </>
  );
  if (props.current || !props.onOpen) {
    return (
      <div
        role="listitem"
        aria-current={props.current ? "true" : undefined}
        className={props.current
          ? "finder-card current" : "finder-card"}
      >
        {inner}
      </div>
    );
  }
  return (
    <button
      type="button"
      role="listitem"
      className="finder-card"
      onClick={props.onOpen}
    >
      {inner}
    </button>
  );
}

function TaskDetail(props: {
  runId: string;
  task: ReviewTask;
  position: number;
  total: number;
  frameByIndex: Map<number, EvidenceFrame>;
  value: string;
  onValueChange: (value: string) => void;
  valueValid: boolean;
  canAct: boolean;
  busy: boolean;
  onPrev: () => void;
  onNext: () => void;
  onAct: (action: string) => void;
  onUndo: () => void;
  onSelectFrame?: (frameIndex: number) => void;
}): JSX.Element {
  const task = props.task;
  const values = taskModelValues(
    task, (index) => props.frameByIndex.get(index)?.measurement?.value
      ?? null);
  const single = values.length === 1 ? values[0] : null;
  const resolved = task.status === "resolved";
  // Culinary findings are categorical (ingredients, actions):
  // a numerical "correct reading" cannot meaningfully apply.
  // The chef confirms, marks cannot-determine, defers, or edits
  // quantities later at the recipe step instead.
  const numericTask = task.field !== "culinary_event";
  const line = correctionResult(task, values);
  const findingFrames = useMemo(
    () => task.frame_refs.map((index) => props.frameByIndex.get(index)
      ?? {
        frame_index: index,
        timestamp_seconds: null,
        image_name: null,
        image_available: false,
        measurement: null,
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [task.task_id, props.frameByIndex]);
  const [framePos, setFramePos] = useState(0);
  useEffect(() => {
    setFramePos(0);
  }, [task.task_id]);
  const safePos = Math.min(
    framePos, Math.max(0, findingFrames.length - 1));
  const activeFrame = findingFrames[safePos] ?? null;

  function selectFindingFrame(position: number): void {
    setFramePos(position);
    const frame = findingFrames[position];
    if (frame) props.onSelectFrame?.(frame.frame_index);
  }

  return (
    <div className="task-detail">
      <FrameViewer
        key={task.task_id}
        runId={props.runId}
        frames={findingFrames}
        selected={activeFrame?.frame_index ?? null}
        onSelect={(index) => {
          const position = findingFrames.findIndex(
            (frame) => frame.frame_index === index);
          if (position >= 0) selectFindingFrame(position);
        }}
        label="Finding frame viewer"
      />
      <div className="flow-actions" aria-label="Finding navigation">
        <button type="button" className="btn-ghost" onClick={props.onPrev}>
          ← Previous finding
        </button>
        <span className="muted small" role="status">
          Finding {props.position + 1} of {props.total}
        </span>
        <button type="button" className="btn-ghost" onClick={props.onNext}>
          Next finding →
        </button>
      </div>
      <h4>{resolved ? "Edit decision" : "Your decision"}</h4>
      <p className="muted">
        {taskLabel(task.state)} · {reviewStartLabel(task)} ·{" "}
        {task.frame_refs.length} frame
        {task.frame_refs.length === 1 ? "" : "s"}
      </p>
      {line && <p className="saved-note" role="status">{line}</p>}
      {!props.canAct && (
        <p className="muted small">
          Add your name at the top to save decisions.
        </p>
      )}
      {!numericTask && (
        <p className="muted small">
          Numerical corrections do not apply to this finding. Confirm
          what the evidence supports, mark it as cannot determine, or
          adjust quantities later at the recipe step.
        </p>
      )}
      <div className="flow-actions">
        {numericTask && (
          <button
            type="button" className="btn-ghost"
            disabled={props.busy || !props.canAct || !props.valueValid}
            onClick={() => props.onAct("correct")}
          >
            Provide correction
          </button>
        )}
        {single !== null && (
          <button
            type="button" className="btn-ghost"
            disabled={props.busy || !props.canAct}
            onClick={() => props.onAct("confirm")}
          >
            Confirm reading
          </button>
        )}
        <button
          type="button" className="btn-ghost"
          disabled={props.busy || !props.canAct}
          onClick={() => props.onAct("not_identifiable")}
        >
          Cannot determine
        </button>
        <button
          type="button" className="btn-ghost"
          disabled={props.busy || !props.canAct}
          onClick={() => props.onAct("review_later")}
        >
          Review later
        </button>
        {task.correction && (
          <button
            type="button" className="btn-ghost"
            disabled={props.busy || !props.canAct}
            onClick={props.onUndo}
          >
            Undo last change
          </button>
        )}
      </div>
      {numericTask && (
        <label className="field">
          <span>Correct reading</span>
          <input
            value={props.value}
            inputMode="decimal"
            placeholder="Enter value"
            aria-label="Correct reading"
            onChange={(event) => props.onValueChange(event.target.value)}
          />
        </label>
      )}
      {numericTask && (
        <p className="muted small">
          Unit unknown. Numbers are observations until a recipe assigns
          them a role.
        </p>
      )}
    </div>
  );
}
