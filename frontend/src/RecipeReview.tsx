import { useEffect, useMemo, useState } from "react";
import {
  ApiError,
  fetchReadiness,
  fetchRecipe,
  fetchRecipeEvidence,
  generateRecipe,
  frameUrl,
  saveRecipeCorrections,
  saveRecipeDecision,
  type EvidenceFrame,
  type RecipeReviewState,
} from "./api";
import {
  approvalBlocked,
  canDecide,
  evidenceScopeLabel,
  formatVideoTime,
  intervalLabel,
  linkFrameToStep,
  loadChefIdentity,
  saveChefIdentity,
  parseVideoTime,
  provenanceLabel,
  stepsReferencingFrame,
} from "./recipe";

interface IngredientDraft {
  ingredient_id: string;
  provenance: string;
}

interface GenerationInfo {
  mode?: string;
  simulated?: boolean;
  rounds_used?: number;
  providers?: Array<{ role: string; provider: string; model: string }>;
  costs?: Array<{
    provider: string; model: string; operation: string;
    input_tokens?: number | null; output_tokens?: number | null;
    estimated_cost_usd?: number | null;
  }>;
  estimated_total_cost_usd?: number | null;
}

function GenerationRecord(props: {
  generation: GenerationInfo;
  simulationNote: string | null;
}): JSX.Element {
  const costs = props.generation.costs ?? [];
  const known = costs
    .filter((item) => typeof item.estimated_cost_usd === "number");
  const estimated = known
    .reduce((sum, item) => sum + (item.estimated_cost_usd ?? 0), 0);
  const costText = known.length === 0 ? "Cost unavailable"
    : `US$${estimated.toFixed(4)} estimated`;
  const providers = (props.generation.providers ?? [])
    .map((item) => `${item.provider} ${item.model}`.trim())
    .filter((item) => item.length > 0);
  return (
    <div>
      <p className="muted small">
        {props.generation.simulated === false
          ? "Live AI version." : "Practice version (simulation)."}
        {providers.length > 0 ? ` Providers: ${providers.join(", ")}.` : ""}
      </p>
      <p className="muted small">
        Additional AI cost this generation: {costText}.{" "}
        Historical analysis cost is already incurred and never counted here.
      </p>
      {props.simulationNote && (
        <p className="muted small">{props.simulationNote}</p>
      )}
    </div>
  );
}

interface StepUsageDraft {
  ingredient_id: string;
  quantity_value: string;
  quantity_unit: string;
}

interface StepDraft {
  step_id: string;
  description: string;
  start_text: string;
  end_text: string;
  frame_refs: Array<{ frame_index: number }>;
  ingredient_refs: string[];
  usages: StepUsageDraft[];
  provenance: string;
  review_status: string;
  uncertainty: string;
  not_observed: boolean;
}

export function provenanceBadge(provenance: string): string {
  if (provenance === "human-corrected") return "badge badge-corrected";
  if (provenance === "chef-supplied") return "badge badge-chef";
  if (provenance === "ai-extracted") return "badge badge-ai";
  return "badge badge-unknown";
}

function toIngredientDrafts(state: RecipeReviewState): IngredientDraft[] {
  // Quantities live only on per-step usages; the summary carries
  // identity and provenance, never a net quantity.
  return state.ingredients.map((item) => ({
    ingredient_id: String(item.ingredient_id ?? ""),
    provenance: String(item.provenance ?? "chef-supplied"),
  }));
}

function toStepDrafts(state: RecipeReviewState): StepDraft[] {
  return state.steps.map((item, index) => ({
    step_id: String(item.step_id ?? `step_${index + 1}`),
    description: String(item.description ?? ""),
    start_text: typeof item.start_seconds === "number"
      ? formatVideoTime(item.start_seconds) : "",
    end_text: typeof item.end_seconds === "number"
      ? formatVideoTime(item.end_seconds) : "",
    frame_refs: Array.isArray(item.frame_refs)
      ? (item.frame_refs as Array<{ frame_index: number }>) : [],
    ingredient_refs: Array.isArray(item.ingredient_refs)
      ? (item.ingredient_refs as string[]) : [],
    usages: Array.isArray(item.usages)
      ? item.usages.map((usage) => ({
          ingredient_id: String(usage.ingredient_id ?? ""),
          quantity_value:
            usage.quantity && usage.quantity.value != null
              ? String(usage.quantity.value)
              : "",
          quantity_unit:
            usage.quantity && usage.quantity.unit != null
              ? String(usage.quantity.unit)
              : "",
        }))
      : (Array.isArray(item.ingredient_refs)
          ? (item.ingredient_refs as string[])
          : []
        ).map((ref) => ({
          ingredient_id: String(ref ?? ""),
          quantity_value: "",
          quantity_unit: "",
        })),
    provenance: String(item.provenance ?? "chef-supplied"),
    review_status: String(item.review_status ?? "pending"),
    uncertainty: String(item.uncertainty ?? ""),
    not_observed: item.not_observed === true,
  }));
}

function statusLabel(status: string): string {
  switch (status) {
    case "approved": return "Approved";
    case "needs_revision": return "Needs revision";
    case "rejected": return "Rejected";
    default: return "In review";
  }
}

function formatDateTime(value: unknown): string {
  if (typeof value !== "string" || value.trim() === "") {
    return "date unavailable";
  }
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

export default function RecipeReview(props: {
  runId: string;
  currentFrameIndex?: number | null;
  onSelectFrame?: (frameIndex: number) => void;
  refreshSignal?: number;
  sourceName?: string;
  onNavigateToGenerate?: () => void;
}): JSX.Element {
  const [state, setState] = useState<RecipeReviewState | null>(null);
  const [frames, setFrames] = useState<EvidenceFrame[]>([]);
  const [ingredients, setIngredients] = useState<IngredientDraft[]>([]);
  const [steps, setSteps] = useState<StepDraft[]>([]);
  const [selectedStep, setSelectedStep] = useState(0);
  const [linkChoice, setLinkChoice] = useState("");
  const [reviewer, setReviewer] = useState(() => loadChefIdentity());
  const [error, setError] = useState<string | null>(null);
  const [missing, setMissing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [showAllClaimFrames, setShowAllClaimFrames] = useState(false);
  const [genConfirmed, setGenConfirmed] = useState(false);
  const [genLive, setGenLive] = useState<{
    live_available?: boolean; live_blockers?: string[];
  } | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      fetchRecipe(props.runId),
      fetchRecipeEvidence(props.runId).catch(() => ({ frames: [] })),
      fetchReadiness(props.runId).catch(() => null),
    ]).then(([next, evidence, readiness]) => {
      if (cancelled) return;
      setState(next);
      setFrames(evidence.frames);
      setIngredients(toIngredientDrafts(next));
      setSteps(toStepDrafts(next));
      setGenLive(readiness);
      setDirty(false);
    }).catch((err: unknown) => {
      if (!cancelled) {
        if (err instanceof ApiError && err.status === 404) {
          setMissing(true);
        } else {
          setError(err instanceof Error ? err.message : "Recipe unavailable.");
        }
      }
    });
    return () => { cancelled = true; };
  }, [props.runId, props.refreshSignal]);

  const frameByIndex = useMemo(
    () => new Map(frames.map((frame) => [frame.frame_index, frame])),
    [frames]);
  const usableFrames = useMemo(
    () => frames.filter((frame) => frame.measurement?.value !== null
      && frame.measurement?.value !== undefined),
    [frames]);
  const step = steps[selectedStep] ?? null;
  const currentLinked = props.currentFrameIndex === null
    || props.currentFrameIndex === undefined
    ? []
    : stepsReferencingFrame(steps, props.currentFrameIndex);

  function stepNumber(stepId: string): number | null {
    const index = steps.findIndex((item) => item.step_id === stepId);
    return index < 0 ? null : index + 1;
  }

  function apply(next: RecipeReviewState): void {
    setState(next);
    setIngredients(toIngredientDrafts(next));
    setSteps(toStepDrafts(next));
    setDirty(false);
  }

  function patchStep(index: number, patch: Partial<StepDraft>): void {
    setSteps(steps.map((row, i) => (i === index ? { ...row, ...patch } : row)));
    setDirty(true);
  }

  function patchUsage(
    stepIndex: number, usageIndex: number, patch: Partial<StepUsageDraft>,
  ): void {
    setSteps(steps.map((row, i) => (i === stepIndex ? {
      ...row,
      usages: row.usages.map((usage, j) => (j === usageIndex
        ? { ...usage, ...patch } : usage)),
    } : row)));
    setDirty(true);
  }

  function usageQuantityError(value: string): string | null {
    if (!value.trim()) return null;
    const parsed = Number(value);
    if (!Number.isFinite(parsed) || parsed < 0) {
      return "Quantity must be a non-negative number, or left unknown.";
    }
    return null;
  }

  function firstInvalidUsage(): string | null {
    for (const step of steps) {
      for (const usage of step.usages) {
        const problem = usageQuantityError(usage.quantity_value);
        if (problem) {
          return `Step ${stepNumber(step.step_id) ?? step.step_id}, `
            + `${usage.ingredient_id || "unnamed ingredient"}: ${problem}`;
        }
      }
    }
    return null;
  }

  async function saveCorrections(): Promise<void> {
    if (!state || busy) return;
    const invalid = firstInvalidUsage();
    if (invalid) {
      setError(`Cannot save: ${invalid}`);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const knownIds = new Set(ingredients
        .filter((item) => item.ingredient_id.trim())
        .map((item) => item.ingredient_id.trim()));
      const next = await saveRecipeCorrections(
        props.runId,
        ingredients.filter((item) => item.ingredient_id.trim()).map(
          (item) => ({
            ingredient_id: item.ingredient_id.trim(),
            provenance: item.provenance,
          })),
        steps.filter((item) => item.description.trim()).map(
          (item, index) => {
            const validUsages = item.usages.filter(
              (usage) => usage.ingredient_id.trim()
                && knownIds.has(usage.ingredient_id.trim()));
            const usageIds = validUsages.map(
              (usage) => usage.ingredient_id.trim());
            return {
              step_id: item.step_id,
              index,
              description: item.description.trim(),
              start_seconds: parseVideoTime(item.start_text),
              end_seconds: parseVideoTime(item.end_text),
              frame_refs: item.frame_refs,
              ingredient_refs: Array.from(new Set([
                ...item.ingredient_refs, ...usageIds])),
              usages: validUsages.map((usage) => ({
                ingredient_id: usage.ingredient_id.trim(),
                quantity: {
                  value: usage.quantity_value.trim()
                    ? Number(usage.quantity_value) : null,
                  unit: usage.quantity_unit.trim()
                    ? usage.quantity_unit.trim() : null,
                  derivation: "chef_supplied",
                },
                provenance: "chef-supplied",
              })),
              provenance: item.provenance,
              review_status: item.not_observed
                ? "not_observed" : item.review_status,
              uncertainty: item.uncertainty.trim()
                ? item.uncertainty.trim() : null,
              not_observed: item.not_observed,
            };
          }),
        reviewer,
      );
      apply(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed.");
    } finally {
      setBusy(false);
    }
  }

  async function regenerate(): Promise<void> {
    if (!state || busy) return;
    setBusy(true);
    setError(null);
    try {
      const next = await generateRecipe(props.runId, {
        mode: "live",
        confirmed: genConfirmed,
        max_rounds: 1,
      });
      apply(next);
      setGenConfirmed(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Generation failed.");
    } finally {
      setBusy(false);
    }
  }

  async function decide(decision: string): Promise<void> {
    if (!state || busy) return;
    setBusy(true);
    setError(null);
    try {
      apply(await saveRecipeDecision(props.runId, decision, reviewer));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Decision failed.");
    } finally {
      setBusy(false);
    }
  }

  if (error && !state) {
    return <p className="error" role="alert">{error}</p>;
  }
  if ((missing && !state) || (state && !state.generation
    && ingredients.length === 0 && steps.length === 0)) {
    return (
      <div className="recipe-review">
        <h3>Your recipe isn&apos;t ready yet</h3>
        <p className="muted">
          Generate a recipe from your reviewed findings first.
          Ingredients, preparation steps, and approval will appear here.
        </p>
        {props.onNavigateToGenerate && (
          <div className="flow-actions">
            <button
              type="button"
              className="btn-primary"
              onClick={props.onNavigateToGenerate}
            >
              Go to Generate recipe
            </button>
          </div>
        )}
      </div>
    );
  }
  if (!state) return <p className="muted">Loading recipe review…</p>;

  const verdict = state.verification?.verdict ?? "review_needed";
  const generation = state.generation as GenerationInfo | undefined;
  // The draft-level simulation flag describes how version 1 was
  // created (deterministic evidence), not the attached generation
  // record. Showing that note under "How it was generated" next to
  // a live generation falsely claims no live model was called.
  const generationNote = generation?.simulated === false
    ? null
    : state.simulation ? state.simulation_note : null;
  const blocked = approvalBlocked(verdict);
  const issueCount = state.verification?.issues.length ?? 0;
  const isEmpty = ingredients.length === 0 && steps.length === 0;
  const canApprove = !busy && canDecide(reviewer) && !blocked && !isEmpty;

  return (
    <div className="recipe-review">
      <div className="queue-head">
        <div>
          <h3>{props.sourceName ?? "Recipe draft"}</h3>
          <p className="muted small">
            Version {state.version} · {statusLabel(state.status)}
          </p>
        </div>
      </div>
      {!state.generation && (
        <p className="muted small" role="status">
          Preliminary record — no AI generation exists yet. The content
          below comes from earlier review work, not from a generated recipe.
        </p>
      )}
      {error && <p className="error" role="alert">{error}</p>}
      {dirty && (
        <p className="muted small" role="status">
          Unsaved edits. Saving creates a new version; it never runs
          new AI inference.
        </p>
      )}
      {props.currentFrameIndex !== null
        && props.currentFrameIndex !== undefined && (
        <p className="muted small">
          Currently viewing{" "}
          {formatVideoTime(
            frameByIndex.get(props.currentFrameIndex)?.timestamp_seconds)}
          . Linked steps:{" "}
          {currentLinked.length
            ? currentLinked.map((item) => {
              const number = stepNumber(item.step_id);
              return number === null ? item.step_id : `Step ${number}`;
            }).join(", ")
            : "none"}. Viewing a frame never changes a step automatically.
        </p>
      )}

      <h3>Ingredients</h3>
      {ingredients.length === 0 && (
        <p className="muted small">
          No ingredient is established by the evidence yet. Add one
          only as chef-supplied or human-corrected information.
        </p>
      )}
      <p className="muted small">
        Quantities are set on each preparation step, never here, so
        an ingredient used in several steps is never double-counted.
      </p>
      {ingredients.map((item, index) => {
        const usedIn = steps
          .map((step, stepIndex) => ({ step, stepIndex }))
          .filter(({ step }) => step.usages.some(
            (usage) => usage.ingredient_id.trim()
              && usage.ingredient_id.trim() === item.ingredient_id.trim())
            || step.ingredient_refs.includes(item.ingredient_id));
        return (
        <div key={index} className="recipe-row">
          <input
            aria-label="Ingredient"
            value={item.ingredient_id}
            placeholder="Ingredient name"
            onChange={(event) => {
              setIngredients(ingredients.map((row, i) => i === index
                ? { ...row, ingredient_id: event.target.value } : row));
              setDirty(true);
            }}
          />
          <span className="muted small">
            {usedIn.length === 0
              ? "Not used in any step yet"
              : `Used in ${usedIn.map(
                ({ stepIndex }) => `Step ${stepIndex + 1}`).join(", ")}`}
          </span>
          <select
            aria-label="Provenance"
            value={item.provenance}
            onChange={(event) => {
              setIngredients(ingredients.map((row, i) => i === index
                ? { ...row, provenance: event.target.value } : row));
              setDirty(true);
            }}
          >
            <option value="ai-extracted">AI extracted</option>
            <option value="human-corrected">Human corrected</option>
            <option value="chef-supplied">Chef supplied (not observed)</option>
          </select>
        </div>
        );
      })}
      <button
        type="button" className="btn-ghost"
        onClick={() => {
          setIngredients([...ingredients, {
            ingredient_id: "",
            provenance: "chef-supplied",
          }]);
          setDirty(true);
        }}
      >
        + Add ingredient
      </button>

      <h3>Preparation steps</h3>
      {steps.length === 0 && (
        <p className="muted small">
          No culinary steps are supported by this measurement-only
          analysis. Steps you add are chef-supplied until linked to
          explicit evidence.
        </p>
      )}
      <div className="queue-list">
        {steps.map((item, index) => (
          <article
            key={item.step_id}
            className={index === selectedStep
              ? "queue-item highlighted" : "queue-item"}
          >
            <button
              type="button" className="btn-ghost"
              onClick={() => setSelectedStep(index)}
            >
              Step {index + 1}: {intervalLabel(
                parseVideoTime(item.start_text),
                parseVideoTime(item.end_text))}
            </button>
            <input
              aria-label={`Step ${index + 1} description`}
              value={item.description}
              onChange={(event) => patchStep(index, {
                description: event.target.value,
              })}
            />
            <p className="muted small">
              <span className={provenanceBadge(item.provenance)}>
                {provenanceLabel(item.provenance)}
              </span> ·{" "}
              {item.frame_refs.length} linked frame(s)
            </p>
          </article>
        ))}
      </div>
      <button
        type="button" className="btn-ghost"
        onClick={() => {
          setSteps([...steps, {
            step_id: `step_${steps.length + 1}`, description: "",
            start_text: "", end_text: "", frame_refs: [],
            ingredient_refs: [], usages: [], provenance: "chef-supplied",
            review_status: "pending", uncertainty: "",
            not_observed: false,
          }]);
          setSelectedStep(steps.length);
          setDirty(true);
        }}
      >
        + Add step
      </button>

      {step && (
        <section className="card" aria-label="Evidence for selected step">
          <h3>Evidence for step {selectedStep + 1}</h3>
          <div className="recipe-row">
            <label>
              Start (mm:ss){" "}
              <input
                aria-label="Start time"
                value={step.start_text}
                placeholder="unknown"
                onChange={(event) => patchStep(selectedStep, {
                  start_text: event.target.value,
                })}
              />
            </label>
            <label>
              End (mm:ss){" "}
              <input
                aria-label="End time"
                value={step.end_text}
                placeholder="unknown"
                onChange={(event) => patchStep(selectedStep, {
                  end_text: event.target.value,
                })}
              />
            </label>
            <label>
              <input
                type="checkbox"
                checked={step.not_observed}
                onChange={(event) => patchStep(selectedStep, {
                  not_observed: event.target.checked,
                })}
              />
              {" "}Not observed in the video
            </label>
          </div>
          <div className="recipe-frames">
            {step.frame_refs.map((ref) => {
              const frame = frameByIndex.get(ref.frame_index);
              return (
                <button
                  key={ref.frame_index}
                  type="button" className="btn-ghost"
                  onClick={() => props.onSelectFrame?.(ref.frame_index)}
                >
                  {frame?.image_available && frame.image_name ? (
                    <img
                      src={frameUrl(props.runId, frame.image_name)}
                      alt={`Frame at ${formatVideoTime(frame.timestamp_seconds)}`}
                      loading="lazy" className="recipe-thumb"
                    />
                  ) : null}
                  {formatVideoTime(frame?.timestamp_seconds)}
                </button>
              );
            })}
            {step.frame_refs.length === 0 && (
              <p className="muted small">No frames linked yet.</p>
            )}
          </div>
          {props.currentFrameIndex !== null
            && props.currentFrameIndex !== undefined && (
            <button
              type="button" className="btn-ghost"
              onClick={() => patchStep(selectedStep, linkFrameToStep(
                step, props.currentFrameIndex as number))}
            >
              Link currently viewed observation (
              {formatVideoTime(frameByIndex.get(
                props.currentFrameIndex)?.timestamp_seconds)})
            </button>
          )}
          <h4>Ingredients used in this step</h4>
          <p className="muted small">
            Quantities you enter are recorded as chef-supplied.
            Blank means unknown — it is never treated as zero.
          </p>
          {step.usages.length === 0 && (
            <p className="muted small">No ingredients linked yet.</p>
          )}
          {step.usages.map((usage, usageIndex) => {
            const problem = usageQuantityError(usage.quantity_value);
            return (
              <div key={`${usage.ingredient_id}-${usageIndex}`}
                className="recipe-row">
                <span>{usage.ingredient_id || "Unnamed ingredient"}</span>
                <input
                  aria-label={`Quantity of ${usage.ingredient_id || "ingredient"}`}
                  value={usage.quantity_value}
                  placeholder="Qty (blank = unknown)"
                  inputMode="decimal"
                  onChange={(event) => patchUsage(selectedStep, usageIndex, {
                    quantity_value: event.target.value,
                  })}
                />
                <input
                  aria-label={`Unit of ${usage.ingredient_id || "ingredient"}`}
                  value={usage.quantity_unit}
                  placeholder="Unit unknown"
                  onChange={(event) => patchUsage(selectedStep, usageIndex, {
                    quantity_unit: event.target.value,
                  })}
                />
                <span className="badge badge-chef">chef-supplied</span>
                <button
                  type="button" className="btn-ghost"
                  onClick={() => patchStep(selectedStep, {
                    usages: step.usages.filter((_, j) => j !== usageIndex),
                  })}
                >
                  Remove
                </button>
                {problem && <span className="error">{problem}</span>}
              </div>
            );
          })}
          {ingredients.filter((item) => item.ingredient_id.trim()
            && !step.usages.some((usage) => usage.ingredient_id.trim()
              === item.ingredient_id.trim())).length > 0 && (
            <div className="recipe-row">
              <select
                aria-label="Ingredient to link"
                value={linkChoice}
                onChange={(event) => setLinkChoice(event.target.value)}
              >
                <option value="">Link an ingredient…</option>
                {ingredients.filter((item) => item.ingredient_id.trim()
                  && !step.usages.some((usage) => usage.ingredient_id.trim()
                    === item.ingredient_id.trim())).map((item) => (
                  <option key={item.ingredient_id} value={item.ingredient_id}>
                    {item.ingredient_id} ({provenanceLabel(item.provenance)})
                  </option>
                ))}
              </select>
              <button
                type="button" className="btn-ghost"
                disabled={!linkChoice}
                onClick={() => {
                  if (!linkChoice) return;
                  patchStep(selectedStep, {
                    usages: [...step.usages, {
                      ingredient_id: linkChoice,
                      quantity_value: "",
                      quantity_unit: "",
                    }],
                  });
                  setLinkChoice("");
                }}
              >
                Link ingredient
              </button>
            </div>
          )}
          <label className="field">
            <span>Uncertainty</span>
            <input
              value={step.uncertainty}
              onChange={(event) => patchStep(selectedStep, {
                uncertainty: event.target.value,
              })}
              placeholder="What remains uncertain about this step?"
            />
          </label>
        </section>
      )}

      <details className="tech-runs">
        <summary>Evidence and model audit</summary>
        <ul className="queue-list">
          {state.claims.map((claim) => (
            <li key={claim.claim_id} className="queue-item">
              <div className="queue-item-main">
                <strong>{claim.title}</strong>
                <span className="pill pill-queued">{claim.status}</span>
              </div>
              <p className="muted small">
                {evidenceScopeLabel(claim)} ·{" "}
                <span className={provenanceBadge(claim.provenance)}>
                  {provenanceLabel(claim.provenance)}
                </span>
              </p>
              <div className="recipe-frames">
                {(showAllClaimFrames && claim.kind === "measurement"
                  ? usableFrames.map((frame) => ({
                    frame_index: frame.frame_index,
                  }))
                  : claim.evidence_refs).map((ref) => (
                  <button
                    key={`${claim.claim_id}-${ref.frame_index}`}
                    type="button" className="btn-ghost"
                    onClick={() => props.onSelectFrame?.(ref.frame_index)}
                  >
                    {formatVideoTime(frameByIndex.get(
                      ref.frame_index)?.timestamp_seconds)}
                  </button>
                ))}
              </div>
              {claim.kind === "measurement" && !showAllClaimFrames && (
                <button
                  type="button" className="btn-ghost"
                  onClick={() => setShowAllClaimFrames(true)}
                >
                  Show all supporting frames
                </button>
              )}
              {claim.objections.length > 0 && (
                <p className="error">
                  Disagreement: {claim.objections.join("; ")}
                </p>
              )}
            </li>
          ))}
        </ul>
      </details>

      <details>
        <summary>
          About this recipe
          <span className="muted small">
            {" "}How it was generated.
          </span>
        </summary>
        {generation ? (
          <GenerationRecord
            generation={generation}
            simulationNote={generationNote}
          />
        ) : (
          <p className="muted small">
            No AI generation record exists for this version.
          </p>
        )}
      </details>

      <details>
        <summary>
          Regenerate recipe
          <span className="muted small">
            {" "}Runs new live AI inference. Only needed after
            corrections change the evidence.
          </span>
        </summary>
        <p className="muted small">
          Saving your edits above never runs inference. Regenerating
          replaces the AI proposal with a fresh live run.
        </p>
        <div className="flow-actions">
          <button
            type="button" className="btn-primary"
            disabled={busy || !genLive?.live_available || !genConfirmed}
            onClick={() => void regenerate()}
          >
            Regenerate with live AI
          </button>
        </div>
        <label className="confirm-row">
          <input
            type="checkbox"
            checked={genConfirmed}
            onChange={(event) => setGenConfirmed(event.target.checked)}
          />
          <span>I confirm billable live inference for this run</span>
        </label>
        {(genLive?.live_blockers ?? []).length > 0 && (
          <p className="muted small">
            {(genLive?.live_blockers ?? []).join(" ")}
          </p>
        )}
        {!genLive?.live_available && (
          <p className="muted small">
            Live AI is currently unavailable, so regeneration is disabled.
          </p>
        )}
      </details>

      <h3>Chef decision</h3>
      <label className="field">
        <span>Reviewer identity (required)</span>
        <input
          value={reviewer}
          onChange={(event) => {
            setReviewer(event.target.value);
            saveChefIdentity(event.target.value);
          }}
          placeholder="e.g. chef-maria"
        />
      </label>
      {!canDecide(reviewer) && (
        <p className="muted small">
          Enter your name above to approve or request changes.
        </p>
      )}
      <div className="flow-actions">
        <button
          type="button" className="btn-primary" disabled={busy}
          onClick={() => void saveCorrections()}
        >
          Save changes
        </button>
        <button
          type="button" className="btn-ghost" disabled={busy || !dirty}
          onClick={() => {
            if (!state) return;
            setIngredients(toIngredientDrafts(state));
            setSteps(toStepDrafts(state));
            setError(null);
            setDirty(false);
          }}
        >
          Undo unsaved changes
        </button>
        <button
          type="button"
          className="btn-primary"
          disabled={!canApprove}
          onClick={() => void decide("approved")}
        >
          Approve recipe
        </button>
        <button
          type="button" className="btn-ghost"
          disabled={busy || !canDecide(reviewer)}
          onClick={() => void decide("needs_revision")}
        >
          Request revision
        </button>
        <button
          type="button" className="btn-ghost"
          disabled={busy || !canDecide(reviewer)}
          onClick={() => void decide("rejected")}
        >
          Reject
        </button>
      </div>
      {isEmpty && (
        <p className="muted small" role="status">
          Add at least one ingredient or step before approving.
        </p>
      )}
      {blocked && (
        <p className="error" role="alert">
          This recipe can&apos;t be approved yet — {issueCount}{" "}
          automatic check{issueCount === 1 ? "" : "s"} failed.
        </p>
      )}
      {blocked && state.verification && state.verification.issues.length > 0 && (
        <details>
          <summary>Why approval is blocked</summary>
          <ul>
            {state.verification.issues.map((issue, index) => (
              <li key={index} className="muted small">
                {issue.message}
              </li>
            ))}
          </ul>
        </details>
      )}
      <h3>Version history</h3>
      {state.history.length === 0 ? (
        <p className="muted small">No versions recorded yet.</p>
      ) : (
        <ul>
          {state.history.map((item, index) => {
            const record = item as {
              version?: unknown; action?: unknown; actor?: unknown;
              at?: unknown; summary?: unknown;
            };
            const summary = typeof record.summary === "string"
              && record.summary.trim() !== "" ? record.summary : null;
            return (
              <li key={index} className="muted small">
                Version {String(record.version ?? "?")} —{" "}
                {String(record.action ?? "recorded").replace(/_/g, " ")}
                {typeof record.actor === "string" && record.actor !== ""
                  ? ` by ${record.actor}` : " (reviewer not recorded)"}
                {" · "}{formatDateTime(record.at)}
                {summary ? ` — ${summary}` : ""}
              </li>
            );
          })}
        </ul>
      )}
      {state.approved_snapshots.length > 0 && (
        <>
          <h3>Approved versions (immutable)</h3>
          <ul>
            {state.approved_snapshots.map((snap, index) => {
              const record = snap as {
                version?: unknown; reviewer_id?: unknown; decided_at?: unknown;
              };
              const reviewerName = typeof record.reviewer_id === "string"
                && record.reviewer_id !== "" ? record.reviewer_id
                : "reviewer not recorded";
              return (
                <li key={index} className="muted small">
                  Version {String(record.version ?? "?")} approved by{" "}
                  {reviewerName} · {formatDateTime(record.decided_at)}
                </li>
              );
            })}
          </ul>
        </>
      )}
    </div>
  );
}

