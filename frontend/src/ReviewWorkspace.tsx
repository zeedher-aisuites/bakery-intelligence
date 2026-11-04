import { useEffect, useMemo, useState } from "react";
import AnalysisSummary from "./AnalysisSummary";
import EvidenceExplorer from "./EvidenceExplorer";
import ObservationReview from "./ObservationReview";
import { loadChefIdentity, saveChefIdentity } from "./recipe";
import GeneratePanel from "./GeneratePanel";
import RecipeReview from "./RecipeReview";
import { displaySourceName } from "./queue";
import {
  ApiError,
  WORKFLOW_META,
  WORKFLOW_OFFLINE,
  WORKFLOW_PIPELINE,
  fetchAnalysis,
  fetchArtifactJson,
  fetchObservationQueue,
  type ObservationQueue,
  type RecipeCard,
  fetchReviewBundle,
  fetchRecipe,
  listRecipes,
} from "./api";
import type { RecipeDraft } from "./results";
import {
  ReviewBundle,
  orderFrames,
  summarizeReview,
} from "./review";


function errorText(err: unknown): string {
  if (err instanceof ApiError) {
    return `Request failed (HTTP ${err.status}): ${err.message}`;
  }
  if (err instanceof Error) return err.message;
  return "Unexpected error.";
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

function recipeStatusLabel(status: string): string {
  if (status === "approved") return "Approved";
  if (status === "needs_changes") return "Needs Review";
  if (status === "processing") return "Processing";
  return "Draft";
}

function recipeLanguageLabel(language: string): string {
  return language === "es" ? "Español" : "English";
}

function Landing(props: {
  onNewAnalysis: () => void;
  onOpenRecipe: (recipeId: string) => void;
}): JSX.Element {
  const [items, setItems] = useState<RecipeCard[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [heroMissing, setHeroMissing] = useState(false);

  useEffect(() => {
    let cancelled = false;
    listRecipes().then((rows) => {
      if (cancelled) return;
      setItems(rows);
    }).catch((err: unknown) => {
      if (!cancelled) setError(errorText(err));
    });
    return () => { cancelled = true; };
  }, []);

  if (error) return <p className="error">{error}</p>;
  if (items === null) {
    return <p className="muted">Loading recipes…</p>;
  }
  return (
    <div className="landing">
      <section className="hero">
        <div className="hero-copy">
          <p className="hero-eyebrow">Evidence-backed recipe intelligence</p>
          <h2>Turn cooking videos into traceable recipes.</h2>
          <p className="hero-lede">
            Transform cooking videos into structured recipes with
            AI, including ingredients, measurements, and
            preparation steps, ready for your review.
          </p>
          <div className="hero-ctas">
            <button
              type="button"
              className="btn-primary btn-lg"
              onClick={() => props.onNewAnalysis()}
            >
              Create my recipe
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
          {heroMissing ? (
            <div className="hero-frame" aria-hidden="true" />
          ) : (
            <img
              className="hero-img"
              src="/images/bakery-hero.jpg"
              alt="Freshly baked goods in a bakery"
              onError={() => setHeroMissing(true)}
            />
          )}
        </div>
      </section>
      {items.length === 0 ? (
        <section className="empty" aria-label="Get started">
          <h2>Your first recipe starts here</h2>
          <p className="muted">
            Upload a cooking video and let AI analyze the preparation.
          </p>
          <button
            type="button"
            className="btn-primary btn-lg"
            onClick={() => props.onNewAnalysis()}
          >
            Create my first recipe
          </button>
        </section>
      ) : (
        <section aria-label="Your recipes">
          <h2>Your Recipes</h2>
          <ul className="run-list">
            {items.map((item) => (
              <li key={item.recipe_id}>
                <button
                  type="button"
                  onClick={() => props.onOpenRecipe(item.recipe_id)}
                >
                  <strong>{item.name}</strong>
                  <span className="muted small">
                    {recipeLanguageLabel(item.language)}
                    {" · "}
                    {fmtDate(item.updated_at ?? undefined)}
                  </span>
                  <span className="stat-row small">
                    <span>{item.ingredients} ingredients</span>
                    <span>{item.steps} steps</span>
                    <span>{recipeStatusLabel(item.status)}</span>
                  </span>
                  <span className="btn-ghost">View Recipe</span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}





/* ---------------- Workspace ---------------- */

type Stage = "findings" | "generate" | "recipe";

function Workspace(props: {
  runId: string;
  onBack: () => void;
}): JSX.Element {
  const [bundle, setBundle] = useState<ReviewBundle | null>(null);
  const [draft, setDraft] = useState<RecipeDraft | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [stage, setStage] = useState<Stage>("findings");
  const [taskProgress, setTaskProgress] = useState<{
    open: number; deferred: number; total: number;
  } | null>(null);
  const [recipeRefresh, setRecipeRefresh] = useState(0);
  const [findingsVersion, setFindingsVersion] = useState(0);
  const [hasGeneration, setHasGeneration] = useState(false);
  const [sourceName, setSourceName] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [queue, setQueue] = useState<ObservationQueue | null>(null);
  const [reviewer, setReviewer] = useState(() => loadChefIdentity());

  function rememberChef(value: string): void {
    setReviewer(value);
    saveChefIdentity(value);
  }

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setLoadError(null);
    (async () => {
      try {
        const data = await fetchReviewBundle(props.runId);
        if (cancelled) return;
        fetchObservationQueue(props.runId).then(setQueue).catch(() => null);
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
        const [draftRes] = await Promise.allSettled([
          wantPipeline || wantOffline
            ? fetchArtifactJson<RecipeDraft>(
              props.runId, "recipe_draft.json")
            : Promise.resolve(null),
        ]);
        if (cancelled) return;
        if (draftRes.status === "fulfilled") setDraft(draftRes.value);
        setSourceName(displaySourceName(status));
        const recipe = await fetchRecipe(props.runId).catch(() => null);
        if (cancelled) return;
        setHasGeneration(recipe?.generation != null);
        setStage(recipe?.generation != null ? "recipe" : "findings");
      } catch (err) {
        if (!cancelled) setLoadError(errorText(err));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [props.runId]);

  const counts = useMemo(
    () => (bundle ? summarizeReview(bundle)
      : { frames: 0, reviewed: 0, pending: 0, numeric: 0,
        abstained: 0, missing: 0 }),
    [bundle]);

  function select(index: number): void {
    setSelected(index);
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
  return (
    <div className="workspace">
      <button type="button" className="linklike" onClick={props.onBack}>
        ← All analyses
      </button>
      <div className="ws-head">
        <div>
          <h2>{sourceName ?? "Video analysis"}</h2>
          <p className="muted">
            Check uncertain measurements and fill in missing
            information so AI can reconstruct your recipe.
          </p>
          <p className="muted small">
            Review the findings, generate the recipe, then review
            and approve it.{" "}
            {taskProgress !== null
              ? `${taskProgress.open} of ${taskProgress.total} findings `
                + `need your attention`
                + (taskProgress.deferred > 0
                  ? ` (${taskProgress.deferred} set aside for later)` : "")
                + "."
              : "Loading findings…"}
          </p>
        </div>
        <dl className="metrics">
          <div><dt>Frames</dt><dd>{counts.frames}</dd></div>
          <div><dt>Readings</dt><dd>{counts.numeric}</dd></div>
          <div><dt>Abstentions</dt><dd>{counts.abstained}</dd></div>
          <div><dt>Reviewed</dt><dd>{counts.reviewed}</dd></div>
        </dl>
      </div>
      <div className="reviewer-control">
        <span className="avatar" aria-hidden="true">
          {reviewer.trim() ? reviewer.trim()[0].toUpperCase() : "?"}
        </span>
        <label className="field compact">
          <span>Reviewer identity</span>
          <input
            value={reviewer}
            onChange={(event) => rememberChef(event.target.value)}
            placeholder="e.g. chef-maria"
            aria-label="Reviewer identity"
          />
        </label>
      </div>
      {bundle.missing.length > 0 && (
        <div className="banner-warn">
          <strong>Incomplete evidence:</strong>
          <ul>{bundle.missing.map((note) => <li key={note}>{note}</li>)}</ul>
        </div>
      )}
      <div className="tabs stage-nav" role="tablist" aria-label="Recipe workflow">
        <button
          type="button"
          role="tab"
          aria-selected={stage === "findings"}
          className={stage === "findings" ? "active" : ""}
          onClick={() => setStage("findings")}
        >
          1 · Review findings
          {taskProgress !== null && taskProgress.open > 0
            ? ` (${taskProgress.open} open)` : ""}
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={stage === "generate"}
          className={stage === "generate" ? "active" : ""}
          onClick={() => setStage("generate")}
        >
          2 · Generate recipe
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={stage === "recipe"}
          className={stage === "recipe" ? "active" : ""}
          onClick={() => setStage("recipe")}
        >
          3 · Review recipe{hasGeneration ? "" : " (none yet)"}
        </button>
      </div>
      <section hidden={stage !== "findings"} aria-label="Review findings">
        <ObservationReview
          runId={bundle.run_id}
          showTasks
          showGeneration={false}
          onProgressChange={setTaskProgress}
          onSelectFrame={select}
          onGenerated={() => setStage("generate")}
          reviewer={reviewer}
          onChanged={() => {
            fetchObservationQueue(props.runId)
              .then(setQueue).catch(() => null);
            setFindingsVersion((value) => value + 1);
          }}
        />
        <AnalysisSummary
          bundle={bundle}
          queue={queue}
          draft={draft}
        />
        <EvidenceExplorer
          runId={bundle.run_id}
          bundle={bundle}
          queue={queue}
          selected={selected}
          onSelect={select}
        />
      </section>
      <section hidden={stage !== "generate"} aria-label="Generate recipe">
        <GeneratePanel
          runId={bundle.run_id}
          active={stage === "generate"}
          refreshSignal={findingsVersion}
          onGenerated={() => {
            setRecipeRefresh((value) => value + 1);
            setHasGeneration(true);
            setStage("recipe");
          }}
          onBackToFindings={() => setStage("findings")}
        />
      </section>
      <section hidden={stage !== "recipe"} aria-label="Review recipe">
        <RecipeReview
          runId={bundle.run_id}
          currentFrameIndex={selected}
          onSelectFrame={select}
          refreshSignal={recipeRefresh}
          sourceName={sourceName ?? undefined}
          onNavigateToGenerate={() => setStage("generate")}
        />
      </section>
    </div>
  );
}

export default function ReviewApp(props: {
  initialRunId?: string | null;
  onNewAnalysis: () => void;
  onOpenRecipe: (recipeId: string) => void;
}): JSX.Element {
  const [selected, setSelected] = useState<string | null>(
    props.initialRunId ?? null);
  if (selected === null) {
    return (
      <Landing
        onNewAnalysis={props.onNewAnalysis}
        onOpenRecipe={props.onOpenRecipe}
      />
    );
  }
  return <Workspace runId={selected} onBack={() => setSelected(null)} />;
}
