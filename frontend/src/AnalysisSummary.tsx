import { useMemo } from "react";
import type { ObservationQueue } from "./api";
import type { ReviewBundle } from "./review";
import type { RecipeDraft } from "./results";

/**
 * Compact actionable summary for the Review Findings stage.
 *
 * Counts only mandatory finding decisions (resolved / still to review /
 * set aside / generation blockers), never optional AI observation
 * tallies. Scientific limits of the current pipeline are stated
 * separately so they are not mistaken for chef tasks.
 */
export default function AnalysisSummary(props: {
  bundle: ReviewBundle;
  queue: ObservationQueue | null;
  draft: RecipeDraft | null;
}): JSX.Element {
  const tasks = useMemo(
    () => props.queue?.tasks ?? [], [props.queue]);
  const resolved = tasks.filter(
    (task) => task.status === "resolved").length;
  const open = tasks.filter(
    (task) => task.status === "open").length;
  const deferred = tasks.filter(
    (task) => task.status === "deferred").length;
  const blockers = tasks.filter(
    (task) => task.state === "contradictory"
      && task.status !== "resolved").length;
  const unknown = props.draft?.unresolved_questions ?? [];
  const unavailable = useMemo(
    () => props.bundle.frames.filter(
      (frame) => !frame.image_available
        || frame.observation === null).length,
    [props.bundle]);

  return (
    <details className="card evidence">
      <summary>
        Analysis summary
        <span className="muted small">
          {" "}{resolved} of {tasks.length} findings resolved
          {open > 0 ? ` · ${open} still need your decision` : ""}
          {blockers > 0 ? ` · ${blockers} block generation` : ""}.
        </span>
      </summary>
      <div className="ev-groups">
        <section aria-label="Findings progress">
          <h4>Your progress</h4>
          <p className="muted">
            {resolved} of {tasks.length} findings resolved.
            {open > 0 && ` ${open} still need${open === 1 ? "s" : ""} `
              + "your decision."}
            {deferred > 0 && ` ${deferred} set aside for later.`}
          </p>
          {blockers > 0 && (
            <p className="muted">
              {blockers} conflicting finding{blockers === 1 ? "" : "s"}{" "}
              must be resolved before live generation. Simulation
              stays available.
            </p>
          )}
        </section>
        <section aria-label="Open questions">
          <h4>Still unknown</h4>
          {unknown.length > 0 ? (
            <ul>
              {unknown.map((item) => (
                <li key={item.code}>{item.message}</li>
              ))}
            </ul>
          ) : (
            <p className="muted">No open questions retained.</p>
          )}
        </section>
        <section aria-label="Pipeline limits">
          <h4>What this analysis cannot establish</h4>
          <ul>
            <li>Ingredient identities are never observed.</li>
            <li>Corrected readings carry no observed unit.</li>
            <li>Scale readings are not net quantities.</li>
          </ul>
          {unavailable > 0 && (
            <p className="muted">
              {unavailable} frame{unavailable === 1 ? "" : "s"} without
              retained evidence.
            </p>
          )}
        </section>
      </div>
    </details>
  );
}
