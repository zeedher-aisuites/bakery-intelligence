import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  fetchFusionDecisions,
  fetchFusionDraft,
  fusionVideoUrl,
  saveFusionDecision,
  type FusionBundle,
} from "./api";
import {
  bucketIngredients,
  clipLabel,
  clipsOf,
  effectiveProvenance,
  formatTimestamp,
  identityLabel,
  intervalLabel,
  orderSteps,
  provenanceBadge,
  provenanceLabel,
  sourceLabel,
  stepsForClip,
  summarizeFusion,
  targetKey,
  type FusionAction,
  type FusionCluster,
  type FusionConflict,
  type FusionDecision,
  type FusionDraft,
  type FusionIngredientEntry,
  type FusionStep,
  type FusionTargetType,
} from "./fusion";

const KNOWN_RUNS = ["p04_104_v6", "bakery_demo_v1"];

const REVIEWER_KEY = "bakery-fusion-reviewer";

interface DecisionInput {
  action: FusionAction;
  value?: Record<string, unknown> | null;
}

function Badge({ provenance }: { provenance: string }) {
  const cls = provenanceBadge(
    provenance as Parameters<typeof provenanceBadge>[0],
  );
  if (!cls) return null;
  return (
    <span className={cls}>
      {provenanceLabel(
        provenance as Parameters<typeof provenanceLabel>[0],
      )}
    </span>
  );
}

function useReviewer(): [string, (value: string) => void] {
  const [reviewer, setReviewer] = useState("");
  useEffect(() => {
    try {
      setReviewer(window.localStorage.getItem(REVIEWER_KEY) ?? "");
    } catch {
      /* private mode; keep the field empty */
    }
  }, []);
  const update = useCallback((value: string) => {
    setReviewer(value);
    try {
      window.localStorage.setItem(REVIEWER_KEY, value);
    } catch {
      /* ignore */
    }
  }, []);
  return [reviewer, update];
}

export default function FusionReview({
  initialRun = "p04_104_v6",
}: {
  initialRun?: string;
}) {
  const [runName, setRunName] = useState(initialRun);
  const [bundle, setBundle] = useState<FusionBundle | null>(null);
  const [activeClip, setActiveClip] = useState<string | null>(null);
  const [decisions, setDecisions] = useState<
    Record<string, FusionDecision>
  >({});
  const [reviewer, setReviewer] = useReviewer();
  const [selectedStep, setSelectedStep] = useState<string | null>(null);
  const [expandedSteps, setExpandedSteps] = useState<
    Record<string, boolean>
  >({});
  const [saving, setSaving] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [videoMissing, setVideoMissing] = useState(false);
  const [videoTime, setVideoTime] = useState<number | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const pendingSeek = useRef<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setBundle(null);
      setError(null);
      setVideoMissing(false);
      try {
        const [loaded, saved] = await Promise.all([
          fetchFusionDraft(runName),
          fetchFusionDecisions(runName),
        ]);
        if (cancelled) return;
        setBundle(loaded);
        const map: Record<string, FusionDecision> = {};
        for (const decision of saved) {
          map[targetKey(decision.target_type, decision.target_id)] =
            decision;
        }
        setDecisions(map);
        const clips = clipsOf(loaded.draft);
        const firstClip = clips[0]?.clip_id ?? null;
        setActiveClip(firstClip);
        const first = orderSteps(
          stepsForClip(loaded.draft.steps ?? [], firstClip),
        )[0];
        setSelectedStep(first ? first.group_id : null);
      } catch (err) {
        if (cancelled) return;
        setError(
          err instanceof ApiError
            ? err.message
            : "Could not load the fusion draft.",
        );
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [runName]);

  const seek = useCallback(
    (seconds: number | null, clipId?: string | null) => {
      if (typeof seconds !== "number" || !Number.isFinite(seconds)
        || seconds < 0) {
        return;
      }
      if (clipId !== undefined && clipId !== activeClip) {
        pendingSeek.current = seconds;
        setActiveClip(clipId);
        return;
      }
      const element = videoRef.current;
      if (element) {
        element.currentTime = seconds;
        element.focus({ preventScroll: true });
      }
    },
    [activeClip],
  );

  const onVideoReady = useCallback(() => {
    const element = videoRef.current;
    if (element && pendingSeek.current !== null) {
      element.currentTime = pendingSeek.current;
      pendingSeek.current = null;
    }
  }, []);

  const save = useCallback(
    async (
      targetType: FusionTargetType,
      targetId: string,
      input: DecisionInput,
    ) => {
      const key = targetKey(targetType, targetId);
      const name = reviewer.trim();
      if (!name) {
        setError("Enter a reviewer name before saving a decision.");
        return;
      }
      setSaving((state) => ({ ...state, [key]: true }));
      setError(null);
      setNotice(null);
      try {
        const decision = await saveFusionDecision(runName, {
          target_type: targetType,
          target_id: targetId,
          action: input.action,
          value: input.value ?? null,
          reviewer: name,
        });
        setDecisions((state) => ({ ...state, [key]: decision }));
        setNotice(`Decision saved for ${targetType} ${targetId}.`);
      } catch (err) {
        setError(
          err instanceof ApiError
            ? err.message
            : "Could not save the decision.",
        );
      } finally {
        setSaving((state) => ({ ...state, [key]: false }));
      }
    },
    [reviewer, runName],
  );

  const progress = useMemo(
    () =>
      bundle
        ? summarizeFusion(bundle.draft, decisions)
        : null,
    [bundle, decisions],
  );

  if (error && !bundle) {
    return (
      <section className="card fx-wrap" aria-label="Chef review">
        <h2>Chef review</h2>
        <p className="fx-error" role="alert">{error}</p>
        <p className="muted small">
          The fusion draft loads from the local backend. Start it
          and make sure run {runName} exists under runs_api/fusion.
        </p>
      </section>
    );
  }
  if (!bundle || bundle.run_name !== runName) {
    // During a session switch the previous bundle is stale: its
    // video URL would pair the new run with the old clip state
    // and 404. Hold the loading state until the new bundle lands.
    return (
      <section className="card fx-wrap" aria-label="Chef review">
        <h2>Chef review</h2>
        <p className="muted">Loading the {runName} fusion draft…</p>
      </section>
    );
  }
  const draft = bundle.draft;
  const sessionClips = clipsOf(draft);
  const steps = orderSteps(
    stepsForClip(draft.steps ?? [], activeClip),
  );
  const buckets = bucketIngredients(draft.ingredients ?? []);
  const clusters =
    draft.quantity_assessment?.stable_clusters ?? [];
  const speechTurns = (draft.speech ?? []).filter(
    (turn) => activeClip === null || turn.clip_id === activeClip,
  );
  const activeDuration =
    sessionClips.find((clip) => clip.clip_id === activeClip)
      ?.duration_seconds ?? null;
  const duration = Math.max(
    1,
    activeDuration ?? 0,
    ...steps.map((step) => step.end_seconds ?? 0),
  );

  return (
    <div className="fx-wrap">
      <section className="card" aria-label="Review header">
        <p className="fx-kicker">
          {draft.session ? draft.session.session_id : "P04"} · chef
          review · {runName}
        </p>
        <h2 className="fx-title">
          {draft.working_title ?? "Untitled reconstruction"}
        </h2>
        <p className="muted">
          Inspect AI-extracted claims against the video, correct
          what is wrong, and verify what you checked. Nothing here
          is approved automatically.
        </p>
        {draft.session && (
          <p className="muted small">
            Multi-clip session: clips play in declared order with
            an unknown gap between recordings. Timestamps are
            clip-local.
          </p>
        )}
        <div className="fx-header-row">
          <label className="fx-reviewer">
            Session
            <select
              value={runName}
              onChange={(event) => setRunName(event.target.value)}
              aria-label="Review session"
            >
              {KNOWN_RUNS.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>
          <label className="fx-reviewer">
            Reviewer
            <input
              type="text"
              value={reviewer}
              onChange={(event) => setReviewer(event.target.value)}
              placeholder="chef name"
              maxLength={64}
              autoComplete="off"
            />
          </label>
          {progress && (
            <p className="fx-progress" aria-live="polite">
              {progress.decided} of {progress.total} reviewed ·{" "}
              {progress.verified} verified · {progress.corrected}{" "}
              corrected · {progress.supplied} chef-supplied
            </p>
          )}
        </div>
        {error && (
          <p className="fx-error" role="alert">{error}</p>
        )}
        {notice && (
          <p className="fx-notice" role="status">{notice}</p>
        )}
      </section>

      <section className="card" aria-label="Video and timeline">
        <h3>Video evidence</h3>
        {sessionClips.length > 0 && (
          <div
            className="fx-actions"
            role="group"
            aria-label="Clip selection"
          >
            {sessionClips.map((clip) => (
              <button
                key={clip.clip_id}
                type="button"
                className={
                  activeClip === clip.clip_id
                    ? "btn-primary"
                    : "btn-ghost"
                }
                onClick={() => {
                  setVideoMissing(false);
                  setVideoTime(null);
                  setActiveClip(clip.clip_id);
                  const first = orderSteps(
                    stepsForClip(
                      draft.steps ?? [],
                      clip.clip_id,
                    ),
                  )[0];
                  setSelectedStep(first ? first.group_id : null);
                }}
              >
                {clipLabel(clip)}
              </button>
            ))}
          </div>
        )}
        {videoMissing ? (
          <p className="muted">
            The review video is not available on this host. Steps
            and timestamps below still work for navigation.
          </p>
        ) : (
          <div className="fx-video-frame">
            <video
              ref={videoRef}
              key={activeClip ?? runName}
              className="fx-video"
              controls
              preload="metadata"
              src={fusionVideoUrl(runName, activeClip)}
              onTimeUpdate={(event) =>
                setVideoTime(event.currentTarget.currentTime)
              }
              onLoadedData={onVideoReady}
              onError={() => setVideoMissing(true)}
            />
          </div>
        )}
        <p className="muted small fx-video-time">
          {videoTime === null
            ? "Press play, or pick a step below to jump."
            : `Playing at ${formatTimestamp(videoTime) ?? "–"}`}
        </p>
        <ol className="fx-timeline" aria-label="Step timeline">
          {steps.map((step, stepPosition) => (
            <li key={`${step.group_id}#${stepPosition}`}>
              <button
                type="button"
                className={
                  `fx-tick${selectedStep === step.group_id ? " active" : ""}` +
                  (decisions[targetKey("step", step.group_id)]
                    ? " decided"
                    : "")
                }
                style={{
                  left: `${((step.start_seconds ?? 0) / duration) * 100}%`,
                  width: `${Math.max(
                    1.2,
                    (((step.end_seconds ?? 0) - (step.start_seconds ?? 0)) /
                      duration) * 100,
                  )}%`,
                }}
                title={`Step ${step.step_index}: ${intervalLabel(step.start_seconds, step.end_seconds)}`}
                aria-label={`Step ${step.step_index}, ${intervalLabel(step.start_seconds, step.end_seconds)}`}
                onClick={() => {
                  setSelectedStep(step.group_id);
                  seek(step.start_seconds);
                }}
              />
            </li>
          ))}
        </ol>
      </section>

      <section className="card" aria-label="Preparation steps">
        <h3>
          Preparation steps ({steps.length}
          {sessionClips.length > 0 &&
            ` of ${draft.steps.length} in session`}
          )
        </h3>
        <ol className="fx-steps">
          {steps.map((step, stepPosition) => (
            <StepCard
              key={`${step.group_id}#${stepPosition}`}
              step={step}
              selected={selectedStep === step.group_id}
              decision={decisions[targetKey("step", step.group_id)]}
              expanded={expandedSteps[step.group_id] ?? false}
              saving={saving[targetKey("step", step.group_id)] ?? false}
              onSelect={() => {
                setSelectedStep(step.group_id);
                seek(step.start_seconds);
              }}
              onToggle={() =>
                setExpandedSteps((state) => ({
                  ...state,
                  [step.group_id]: !(state[step.group_id] ?? false),
                }))
              }
              onSeek={seek}
              onSave={(input) =>
                void save("step", step.group_id, input)
              }
            />
          ))}
        </ol>
      </section>

      <section className="card" aria-label="Ingredient review">
        <h3>Ingredients under review</h3>
        <p className="muted small">
          {draft.ingredients.length} evidence groups, not{" "}
          {draft.ingredients.length} confirmed ingredients. Confirm
          only what you can identify from the evidence.
        </p>
        <IngredientBucket
          title="Model proposed"
          rows={buckets.model_proposed}
          decisions={decisions}
          saving={saving}
          onSeek={seek}
          onSave={(id, input) =>
            void save("ingredient", id, input)
          }
        />
        <IngredientBucket
          title="Described, unidentified"
          rows={buckets.described}
          decisions={decisions}
          saving={saving}
          onSeek={seek}
          onSave={(id, input) =>
            void save("ingredient", id, input)
          }
        />
        <IngredientBucket
          title="Unresolved"
          rows={buckets.unresolved}
          decisions={decisions}
          saving={saving}
          onSeek={seek}
          onSave={(id, input) =>
            void save("ingredient", id, input)
          }
        />
      </section>

      <section className="card" aria-label="Measurement review">
        <h3>Scale evidence ({clusters.length} clusters)</h3>
        <p className="muted small">
          Displayed digits are observations with unknown units —
          never net quantities. Supply a unit only when you know
          it.
        </p>
        <div className="fx-clusters">
          {clusters.map((cluster, clusterPosition) => (
            <ClusterCard
              key={`${cluster.cluster_id}#${clusterPosition}`}
              cluster={cluster}
              decision={
                decisions[
                  targetKey("quantity_cluster", cluster.cluster_id)
                ]
              }
              saving={
                saving[
                  targetKey("quantity_cluster", cluster.cluster_id)
                ] ?? false
              }
              onSeek={seek}
              onSave={(input) =>
                void save("quantity_cluster", cluster.cluster_id, input)
              }
            />
          ))}
        </div>
      </section>

      {speechTurns.length > 0 && (
        <section className="card" aria-label="Transcript evidence">
          <h3>What was said ({speechTurns.length} turns)</h3>
          <p className="muted small">
            Speech is evidence of utterance, not proof of action.
            Timestamps are turn-level and clip-local.
          </p>
          <ul className="fx-queue">
            {speechTurns.map((turn) => (
              <li key={turn.speech_id} className="fx-queue-row">
                <p>
                  {turn.speaker && (
                    <strong>{turn.speaker}: </strong>
                  )}
                  {turn.transcript}
                </p>
                <p className="muted small">
                  <button
                    type="button"
                    className="fx-link"
                    onClick={() =>
                      seek(turn.start_seconds, turn.clip_id)
                    }
                  >
                    {intervalLabel(
                      turn.start_seconds,
                      turn.end_seconds,
                    )}
                  </button>
                  {turn.claims.length > 0 &&
                    ` · ${turn.claims.length} culinary claim(s)`}
                </p>
                {turn.claims.length > 0 && (
                  <ul className="fx-claims">
                    {turn.claims.map((claim, claimIndex) => (
                      <li
                        key={
                          typeof claim.claim_id === "string"
                            ? claim.claim_id
                            : `claim-${claimIndex}`
                        }
                        className="muted small"
                      >
                        {typeof claim.claim_kind === "string" && (
                          <span className="mono">
                            {claim.claim_kind}
                            {": "}
                          </span>
                        )}
                        {typeof claim.interpretation === "string"
                          ? claim.interpretation
                          : "Claim recorded without interpretation."}
                      </li>
                    ))}
                  </ul>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}

      <section className="card" aria-label="Review queue">
        <h3>Review queue</h3>
        <h4 className="fx-subhead">
          Conflicts ({draft.conflicts.length})
        </h4>
        <ul className="fx-queue">
          {draft.conflicts.map((conflict, conflictPosition) => (
            <ConflictRow
              key={`${conflict.conflict_id}#${conflictPosition}`}
              conflict={conflict}
              decision={
                decisions[
                  targetKey("conflict", conflict.conflict_id)
                ]
              }
              saving={
                saving[
                  targetKey("conflict", conflict.conflict_id)
                ] ?? false
              }
              onSave={(input) =>
                void save("conflict", conflict.conflict_id, input)
              }
            />
          ))}
        </ul>
        <h4 className="fx-subhead">
          Open questions ({draft.unresolved_questions.length})
        </h4>
        <ul className="fx-queue">
          {draft.unresolved_questions.map((question, index) => (
            <QuestionRow
              key={`q${index}`}
              targetId={`q${index}`}
              question={question}
              decision={
                decisions[targetKey("question", `q${index}`)]
              }
              saving={
                saving[targetKey("question", `q${index}`)] ?? false
              }
              onSave={(input) =>
                void save("question", `q${index}`, input)
              }
            />
          ))}
        </ul>
      </section>
    </div>
  );
}

function DecisionButtons({
  saving,
  onSave,
}: {
  saving: boolean;
  onSave: (input: DecisionInput) => void;
}) {
  return (
    <div className="fx-actions" role="group" aria-label="Decision">
      <button
        type="button"
        className="btn-ghost"
        disabled={saving}
        onClick={() => onSave({ action: "confirm" })}
      >
        Verify
      </button>
      <button
        type="button"
        className="btn-ghost"
        disabled={saving}
        onClick={() => onSave({ action: "cannot_determine" })}
      >
        Cannot determine
      </button>
    </div>
  );
}

function StepCard({
  step,
  selected,
  decision,
  expanded,
  saving,
  onSelect,
  onToggle,
  onSeek,
  onSave,
}: {
  step: FusionStep;
  selected: boolean;
  decision: FusionDecision | undefined;
  expanded: boolean;
  saving: boolean;
  onSelect: () => void;
  onToggle: () => void;
  onSeek: (seconds: number | null, clipId?: string | null) => void;
  onSave: (input: DecisionInput) => void;
}) {
  const provenance = effectiveProvenance(decision);
  return (
    <li className={`fx-step${selected ? " selected" : ""}`}>
      <div className="fx-step-head">
        <button
          type="button"
          className="fx-step-title"
          onClick={onSelect}
          aria-label={`Jump to step ${step.step_index}`}
        >
          <span className="fx-step-index">{step.step_index}</span>
          <span>
            {step.actions.length > 0
              ? step.actions.join(" · ")
              : "No labeled action"}
          </span>
        </button>
        <Badge provenance={provenance} />
      </div>
      <p className="muted small">
        <button
          type="button"
          className="fx-link"
          onClick={() => onSeek(step.start_seconds)}
        >
          {intervalLabel(step.start_seconds, step.end_seconds)}
        </button>
        {" · "}
        {step.member_ids.length} evidence members ·{" "}
        {step.sources.map(sourceLabel).join(" + ") || "no source"}
      </p>
      {step.uncertainties.length > 0 && (
        <p className="fx-note">
          {step.uncertainties.length} uncertaint
          {step.uncertainties.length === 1 ? "y" : "ies"} noted
        </p>
      )}
      <div className="fx-actions">
        <button
          type="button"
          className="btn-ghost"
          onClick={onToggle}
          aria-expanded={expanded}
        >
          {expanded ? "Hide evidence" : "Show evidence"}
        </button>
        <DecisionButtons saving={saving} onSave={onSave} />
      </div>
      {expanded && (
        <ul className="fx-members">
          {step.members.map((member, memberPosition) => (
            <li key={`${member.member_id ?? member.source_ref ?? "member"}#${memberPosition}`}>
              <button
                type="button"
                className="fx-link mono"
                onClick={() => onSeek(member.start_seconds)}
              >
                {intervalLabel(
                  member.start_seconds,
                  member.end_seconds,
                )}
              </button>{" "}
              <span>
                {member.action ?? "unlabeled"} ·{" "}
                {sourceLabel(member.source ?? "")}
              </span>
            </li>
          ))}
          {step.preliminary_links.length > 0 && (
            <li className="muted small">
              {step.preliminary_links.length} linked human note
              {step.preliminary_links.length === 1 ? "" : "s"}
            </li>
          )}
          {step.uncertainties.slice(0, 3).map((text, index) => (
            <li
              key={`${step.group_id}-u${index}`}
              className="muted small"
            >
              {text}
            </li>
          ))}
        </ul>
      )}
      {decision && (
        <p className="muted small">
          {decision.action} by {decision.reviewer} ·{" "}
          {decision.updated_at}
        </p>
      )}
    </li>
  );
}

function IngredientBucket({
  title,
  rows,
  decisions,
  saving,
  onSeek,
  onSave,
}: {
  title: string;
  rows: FusionIngredientEntry[];
  decisions: Record<string, FusionDecision>;
  saving: Record<string, boolean>;
  onSeek: (seconds: number | null, clipId?: string | null) => void;
  onSave: (id: string, input: DecisionInput) => void;
}) {
  const [open, setOpen] = useState(title === "Model proposed");
  return (
    <details
      className="fx-bucket"
      open={open}
      onToggle={(event) =>
        setOpen((event.target as HTMLDetailsElement).open)
      }
    >
      <summary>
        {title} ({rows.length})
      </summary>
      {rows.length === 0 ? (
        <p className="muted small">None.</p>
      ) : (
        <ul className="fx-ingredients">
          {rows.map((entry, rowPosition) => (
            <IngredientRow
              key={`${entry.group_id}#${rowPosition}`}
              entry={entry}
              decision={
                decisions[targetKey("ingredient", entry.group_id)]
              }
              saving={
                saving[targetKey("ingredient", entry.group_id)] ??
                false
              }
              onSeek={onSeek}
              onSave={(input) => onSave(entry.group_id, input)}
            />
          ))}
        </ul>
      )}
    </details>
  );
}

function IngredientRow({
  entry,
  decision,
  saving,
  onSeek,
  onSave,
}: {
  entry: FusionIngredientEntry;
  decision: FusionDecision | undefined;
  saving: boolean;
  onSeek: (seconds: number | null, clipId?: string | null) => void;
  onSave: (input: DecisionInput) => void;
}) {
  const [name, setName] = useState("");
  const provenance = effectiveProvenance(decision);
  const chefName =
    decision?.action === "correct"
      ? (decision.value?.name as string | undefined)
      : undefined;
  const firstStamp = entry.supporting_timestamps[0] ?? null;
  return (
    <li className="fx-ingredient">
      <div className="fx-step-head">
        <span className="fx-ingredient-name">
          {entry.display_label ?? "Unidentified observation"}
        </span>
        <Badge provenance={provenance} />
      </div>
      <p className="muted small">
        {identityLabel(entry.identity_status)} ·{" "}
        {entry.member_count} record
        {entry.member_count === 1 ? "" : "s"} ·{" "}
        {entry.sources.map(sourceLabel).join(" + ") || "no source"}
      </p>
      {chefName && (
        <p className="fx-note">
          Chef correction: <strong>{chefName}</strong> (original
          evidence preserved)
        </p>
      )}
      {entry.visible_labels.length > 0 && (
        <p className="muted small">
          Seen as: {entry.visible_labels.join(" | ")}
        </p>
      )}
      {entry.ambiguities.length > 0 && (
        <p className="muted small">
          Ambiguity: {entry.ambiguities[0]}
        </p>
      )}
      <div className="fx-actions">
        {firstStamp !== null && (
          <button
            type="button"
            className="fx-link"
            onClick={() =>
              onSeek(firstStamp, entry.clip_id ?? null)
            }
          >
            View at {formatTimestamp(firstStamp)}
          </button>
        )}
        <input
          type="text"
          className="fx-input"
          value={name}
          maxLength={500}
          placeholder="Correct identity…"
          aria-label={`Correct identity for ${entry.display_label ?? entry.group_id}`}
          onChange={(event) => setName(event.target.value)}
        />
        <button
          type="button"
          className="btn-ghost"
          disabled={saving || !name.trim()}
          onClick={() => {
            onSave({ action: "correct", value: { name: name.trim() } });
            setName("");
          }}
        >
          Correct
        </button>
        <DecisionButtons saving={saving} onSave={onSave} />
      </div>
      {decision && (
        <p className="muted small">
          {decision.action} by {decision.reviewer} ·{" "}
          {decision.updated_at}
        </p>
      )}
    </li>
  );
}

function ClusterCard({
  cluster,
  decision,
  saving,
  onSeek,
  onSave,
}: {
  cluster: FusionCluster;
  decision: FusionDecision | undefined;
  saving: boolean;
  onSeek: (seconds: number | null, clipId?: string | null) => void;
  onSave: (input: DecisionInput) => void;
}) {
  const [unit, setUnit] = useState("");
  const [amount, setAmount] = useState("");
  const provenance = effectiveProvenance(decision);
  const chefValue = decision?.action === "supply" ? decision.value : null;
  return (
    <article className="fx-cluster">
      <div className="fx-step-head">
        <span className="fx-cluster-value">
          {cluster.displayed_value}
          <span className="muted"> unit unknown</span>
        </span>
        <Badge provenance={provenance} />
      </div>
      <p className="muted small">
        <button
          type="button"
          className="fx-link"
          onClick={() =>
            onSeek(cluster.start_seconds, cluster.clip_id ?? null)
          }
        >
          {intervalLabel(cluster.start_seconds, cluster.end_seconds)}
        </button>
        {" · "}
        {cluster.reading_ids.length} readings ·{" "}
        {cluster.sources.map(sourceLabel).join(" + ")}
      </p>
      {chefValue && (
        <p className="fx-note">
          Chef supplied:{" "}
          <strong>
            {[
              chefValue.amount !== undefined
                ? String(chefValue.amount)
                : null,
              (chefValue.unit as string | undefined) ?? null,
            ]
              .filter(Boolean)
              .join(" ") || "–"}
          </strong>
        </p>
      )}
      <ul className="fx-missing">
        {cluster.missing_for_quantity.map((item) => (
          <li key={item.slice(0, 48)}>{item}</li>
        ))}
      </ul>
      <div className="fx-actions">
        <input
          type="text"
          className="fx-input fx-unit"
          value={unit}
          maxLength={60}
          placeholder="Unit (g, ml…)"
          aria-label={`Unit for display ${cluster.displayed_value}`}
          onChange={(event) => setUnit(event.target.value)}
        />
        <input
          type="text"
          className="fx-input fx-amount"
          value={amount}
          inputMode="decimal"
          placeholder="Amount"
          aria-label={`Amount for display ${cluster.displayed_value}`}
          onChange={(event) => setAmount(event.target.value)}
        />
        <button
          type="button"
          className="btn-ghost"
          disabled={saving || (!unit.trim() && !amount.trim())}
          onClick={() => {
            const value: Record<string, unknown> = {};
            if (unit.trim()) value.unit = unit.trim();
            if (amount.trim()) {
              const parsed = Number(amount.trim());
              if (!Number.isFinite(parsed)) return;
              value.amount = parsed;
            }
            onSave({ action: "supply", value });
            setUnit("");
            setAmount("");
          }}
        >
          Supply
        </button>
        <DecisionButtons saving={saving} onSave={onSave} />
      </div>
      {decision && (
        <p className="muted small">
          {decision.action} by {decision.reviewer} ·{" "}
          {decision.updated_at}
        </p>
      )}
    </article>
  );
}

function ConflictRow({
  conflict,
  decision,
  saving,
  onSave,
}: {
  conflict: FusionConflict;
  decision: FusionDecision | undefined;
  saving: boolean;
  onSave: (input: DecisionInput) => void;
}) {
  return (
    <li className="fx-queue-row">
      <div className="fx-step-head">
        <span>{conflict.detail}</span>
        <Badge provenance={effectiveProvenance(decision)} />
      </div>
      <p className="muted small">
        {conflict.kind} · {conflict.sources.map(sourceLabel).join(" vs ")}
      </p>
      <DecisionButtons saving={saving} onSave={onSave} />
    </li>
  );
}

function QuestionRow({
  targetId,
  question,
  decision,
  saving,
  onSave,
}: {
  targetId: string;
  question: string;
  decision: FusionDecision | undefined;
  saving: boolean;
  onSave: (input: DecisionInput) => void;
}) {
  return (
    <li className="fx-queue-row">
      <div className="fx-step-head">
        <span>
          <span className="mono muted small">{targetId} </span>
          {question}
        </span>
        <Badge provenance={effectiveProvenance(decision)} />
      </div>
      <DecisionButtons saving={saving} onSave={onSave} />
    </li>
  );
}
