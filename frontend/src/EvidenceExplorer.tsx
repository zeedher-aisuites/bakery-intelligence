import { useEffect, useMemo, useRef, useState } from "react";
import {
  type ObservationQueue,
} from "./api";
import FrameViewer from "./FrameViewer";
import {
  applyFrameFilters,
  correctionEffectiveValue,
  correctionResult,
  correctionStatus,
  DEFAULT_FILTERS,
  taskModelValues,
  type FrameFilterState,
} from "./observation";
import {
  effectiveReading,
  orderFrames,
  readingBadge,
  type ReviewBundle,
  type ReviewFrame,
} from "./review";

export interface EvidenceExplorerProps {
  runId: string;
  bundle: ReviewBundle;
  queue: ObservationQueue | null;
  selected: number | null;
  onSelect: (index: number) => void;
}

const MODEL_OPTIONS: Array<[string, string]> = [
  ["all", "All"],
  ["reading", "Reading predicted"],
  ["abstained", "Abstained"],
  ["not_evaluated", "Not evaluated"],
];

const VALIDATION_OPTIONS: Array<[string, string]> = [
  ["all", "All"],
  ["passed", "Passed automatic checks"],
  ["needs", "Needs review"],
  ["conflicting", "Conflicting evidence"],
  ["not relevant", "Not relevant"],
];

const HUMAN_OPTIONS: Array<[string, string]> = [
  ["all", "All"],
  ["Pending", "Pending"],
  ["Corrected", "Corrected"],
  ["Confirmed", "Confirmed"],
  ["Cannot determine", "Cannot determine"],
  ["Deferred", "Deferred"],
];

const CONFIDENCE_OPTIONS: Array<[string, string]> = [
  ["all", "All"],
  ["high", "High"],
  ["medium", "Medium"],
  ["low", "Low"],
  ["not reported", "Not reported"],
];

function countActiveFilters(filters: FrameFilterState): number {
  let active = 0;
  if (filters.model !== "all") active += 1;
  if (filters.validation !== "all") active += 1;
  if (filters.human !== "all") active += 1;
  if (filters.confidence !== "all") active += 1;
  if (filters.inTasks) active += 1;
  return active;
}

function humanizeProvenance(provenance: string): string {
  switch (provenance) {
    case "ai-extracted": return "AI prediction";
    case "human-verified": return "Verified by you";
    case "human-corrected": return "Your correction";
    case "rejected": return "Rejected by you";
    case "uncertain": return "Marked uncertain";
    default: return "No usable reading";
  }
}

/* ---------------- Filter dialog ---------------- */

function FilterDialog(props: {
  open: boolean;
  draft: FrameFilterState;
  onDraft: (next: FrameFilterState) => void;
  onApply: () => void;
  onClear: () => void;
  onCancel: () => void;
}): JSX.Element | null {
  const dialogRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (props.open) dialogRef.current?.focus();
  }, [props.open]);

  if (!props.open) return null;
  function select(
    label: string,
    value: string,
    options: Array<[string, string]>,
    onChange: (value: string) => void,
  ): JSX.Element {
    return (
      <label className="field">
        <span>{label}</span>
        <select
          value={value}
          onChange={(event) => onChange(event.target.value)}
        >
          {options.map(([optionValue, optionLabel]) => (
            <option key={optionValue} value={optionValue}>
              {optionLabel}
            </option>
          ))}
        </select>
      </label>
    );
  }
  return (
    <div
      className="ev-dialog-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) props.onCancel();
      }}
    >
      <div
        ref={dialogRef}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-label="Frame filters"
        className="ev-dialog"
        onKeyDown={(event) => {
          if (event.key === "Escape") props.onCancel();
        }}
      >
        <h4>Frame filters</h4>
        <p className="muted small">
          Filters apply together when you choose Apply. Cancel leaves
          the current filters unchanged.
        </p>
        {select("Model output", props.draft.model, MODEL_OPTIONS,
          (model) => props.onDraft({ ...props.draft, model }))}
        {select("Automatic validation", props.draft.validation,
          VALIDATION_OPTIONS,
          (validation) => props.onDraft({ ...props.draft, validation }))}
        {select("Human review status", props.draft.human, HUMAN_OPTIONS,
          (human) => props.onDraft({ ...props.draft, human }))}
        {select("Confidence", props.draft.confidence, CONFIDENCE_OPTIONS,
          (confidence) => props.onDraft({ ...props.draft, confidence }))}
        <label className="confirm-row">
          <input
            type="checkbox"
            checked={props.draft.inTasks}
            onChange={(event) => props.onDraft({
              ...props.draft, inTasks: event.target.checked,
            })}
          />
          <span>Only frames in review tasks</span>
        </label>
        <div className="flow-actions">
          <button
            type="button"
            className="btn-primary"
            onClick={props.onApply}
          >
            Apply
          </button>
          <button
            type="button"
            className="btn-ghost"
            onClick={props.onClear}
          >
            Clear filters
          </button>
          <button
            type="button"
            className="btn-ghost"
            onClick={props.onCancel}
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

/* ---------------- Observation strip ---------------- */

interface FrameDecision {
  source: "frame" | "finding";
  summary: string;
  value: number | null;
  shortLabel: string;
  reviewer: string | null;
  notes: string | null;
}

function frameReviewDecision(frame: ReviewFrame): FrameDecision {
  const review = frame.review;
  const value = typeof review?.value === "number"
    && Number.isFinite(review.value) ? review.value : null;
  const reviewer = review?.reviewer ?? null;
  const notes = review?.notes ?? null;
  switch (review?.decision) {
    case "corrected":
      return {
        source: "frame",
        summary: `Corrected to ${value ?? "an unrecorded value"}.`,
        value,
        shortLabel: "Your correction",
        reviewer,
        notes,
      };
    case "confirmed":
      return {
        source: "frame",
        summary: "AI reading confirmed.",
        value,
        shortLabel: "Verified by you",
        reviewer,
        notes,
      };
    case "rejected":
      return {
        source: "frame",
        summary: "Rejected: not a valid reading.",
        value: null,
        shortLabel: "Rejected by you",
        reviewer,
        notes,
      };
    default:
      return {
        source: "frame",
        summary: "Marked as cannot determine.",
        value: null,
        shortLabel: "Marked uncertain",
        reviewer,
        notes,
      };
  }
}

function ObservationStrip(props: {
  frame: ReviewFrame | null;
  decision: FrameDecision | null;
}): JSX.Element {
  if (props.frame === null) {
    return <p className="muted">Select a frame to inspect it.</p>;
  }
  const frame = props.frame;
  const fields = frame.observation?.model_fields ?? null;
  const reading = effectiveReading(frame);
  const decision = props.decision;
  const value = decision?.value ?? reading.value;
  const readable = fields?.scale_readable;
  return (
    <section aria-label="Current frame observation">
      <dl className="ev-strip">
        <div>
          <dt>Scale reading</dt>
          <dd>
            {value === null ? "Not reported" : String(value)}
            {" "}
            <span className={readingBadge(reading.provenance)
              ?? `pill ${reading.pill}`}>
              {decision ? decision.shortLabel
                : humanizeProvenance(reading.provenance)}
            </span>
          </dd>
        </div>
        <div>
          <dt>AI prediction (original)</dt>
          <dd>{fields?.scale_display_verbatim ?? "Not reported"}</dd>
        </div>
      </dl>
      {decision ? (
        <>
          <p className="saved-note" role="status">{decision.summary}</p>
          <p className="muted small">
            Decided by{" "}
            {decision.source === "frame"
              ? "frame review" : "finding correction"}
            {decision.reviewer ? ` · ${decision.reviewer}` : ""}.
          </p>
          {decision.notes && (
            <p className="muted small">
              Reviewer note: {decision.notes}
            </p>
          )}
        </>
      ) : (
        reading.note && (
          <p className="muted small">{reading.note}</p>
        )
      )}
      <details>
        <summary>Observation details</summary>
        <dl className="ev-strip">
          <div>
            <dt>Scale state</dt>
            <dd>{fields?.scale_state ?? "Not reported"}</dd>
          </div>
          <div>
            <dt>Confidence</dt>
            <dd>{fields?.confidence ?? "Not reported"}</dd>
          </div>
          <div>
            <dt>Container</dt>
            <dd>{fields?.container_description ?? "Not reported"}</dd>
          </div>
          <div>
            <dt>Readable</dt>
            <dd>
              {readable === null || readable === undefined
                ? "Not reported"
                : readable ? "Readable" : "Not readable"}
            </dd>
          </div>
        </dl>
      </details>
      <p className="muted small">
        Scale readings are observations, not verified ingredient
        quantities.
      </p>
    </section>
  );
}




/* ---------------- Evidence explorer ---------------- */

export default function EvidenceExplorer(
  props: EvidenceExplorerProps,
): JSX.Element {
  const frames = useMemo(
    () => orderFrames(props.bundle.frames), [props.bundle]);
  const evaluatedByFrame = useMemo(() => new Map(
    (props.queue?.evaluated ?? []).map(
      (item) => [item.frame_index, item])), [props.queue]);
  const taskFrames = useMemo(() => new Set(
    (props.queue?.tasks ?? []).flatMap((task) => task.frame_refs)),
    [props.queue]);

  function validationOf(frame: ReviewFrame): string {
    const state = evaluatedByFrame.get(frame.frame_index)?.state;
    if (state === "accepted") return "passed";
    if (state === "missing_relevant" || state === "uncertain"
      || state === "flagged_for_review") return "needs";
    if (state === "contradictory") return "conflicting";
    return "not relevant";
  }

  function humanOf(frame: ReviewFrame): string {
    const task = props.queue?.tasks.find((item) => item.frame_refs.includes(
      frame.frame_index));
    const action = task?.correction?.action;
    if (action === "correct" || action === "supply") return "Corrected";
    if (action === "confirm") return "Confirmed";
    if (action === "not_identifiable" || action === "cannot_determine"
      || action === "skip_unknown") return "Cannot determine";
    if (action === "review_later" || task?.status === "deferred") {
      return "Deferred";
    }
    return "Pending";
  }

  const [applied, setApplied] = useState<FrameFilterState>(DEFAULT_FILTERS);
  const [draft, setDraft] = useState<FrameFilterState>(DEFAULT_FILTERS);
  const [dialogOpen, setDialogOpen] = useState(false);

  const visible = useMemo(() => applyFrameFilters(
    frames, validationOf, humanOf, taskFrames, applied),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [frames, props.queue, applied]);
  const visibleKey = visible.map(
    (frame) => frame.frame_index).join(",");
  const current = useMemo(
    () => visible.find((frame) => frame.frame_index === props.selected)
      ?? visible[0] ?? null,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [visibleKey, props.selected]);

  useEffect(() => {
    if (visible.length > 0 && (props.selected === null
      || !visible.some(
        (frame) => frame.frame_index === props.selected))) {
      props.onSelect(visible[0].frame_index);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visibleKey]);

  function openDialog(): void {
    setDraft(applied);
    setDialogOpen(true);
  }

  const bundleValue = (index: number): number | null => {
    const value = frames.find(
      (frame) => frame.frame_index === index)?.measurement?.value;
    return typeof value === "number" ? value : null;
  };

  const decision = useMemo((): FrameDecision | null => {
    if (current === null) return null;
    if (current.review) {
      return frameReviewDecision(current);
    }
    const task = props.queue?.tasks.find(
      (item) => item.correction && item.frame_refs.includes(
        current.frame_index));
    if (!task?.correction) return null;
    const values = taskModelValues(task, bundleValue);
    const line = correctionResult(task, values)
      ?? correctionStatus(task);
    return {
      source: "finding",
      summary: line,
      value: correctionEffectiveValue(task, values),
      shortLabel: correctionStatus(task),
      reviewer: task.correction.reviewer ?? null,
      notes: null,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [current, props.queue, visibleKey]);

  const activeFilters = countActiveFilters(applied);

  return (
    <details className="card evidence">
      <summary>
        Evidence explorer
        <span className="muted small">
          {" "}Frame viewer and observations. Closed until you open it.
        </span>
      </summary>
      <div className="ev-toolbar">
        <button
          type="button"
          className="btn-ghost"
          onClick={openDialog}
        >
          Filters{activeFilters > 0 ? ` · ${activeFilters} active` : ""}
        </button>
        <span className="muted small" role="status">
          {visible.length} of {frames.length} frames
        </span>
      </div>
      <FilterDialog
        open={dialogOpen}
        draft={draft}
        onDraft={setDraft}
        onApply={() => {
          setApplied(draft);
          setDialogOpen(false);
        }}
        onClear={() => setDraft(DEFAULT_FILTERS)}
        onCancel={() => setDialogOpen(false)}
      />
      {visible.length === 0 ? (
        <p className="muted">
          No frames match the active filters.{" "}
          <button
            type="button"
            className="linklike"
            onClick={() => setApplied(DEFAULT_FILTERS)}
          >
            Reset filters
          </button>
        </p>
      ) : (
        <>
          <FrameViewer
            runId={props.runId}
            frames={visible}
            selected={props.selected}
            onSelect={props.onSelect}
          />
          <ObservationStrip frame={current} decision={decision} />
        </>
      )}
    </details>
  );
}
