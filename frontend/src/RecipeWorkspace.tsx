import { useEffect, useRef, useState } from "react";
import RecipeEditor from "./RecipeEditor";
import {
  ApiError,
  buildRecipe,
  fetchAnalysis,
  listRecipes,
  type RecipeCard,
} from "./api";

const DEFAULT_RECIPE_ID = "bakery_demo_v1_proposed_v3";
const POLL_MS = 2500;

type Phase = "loading" | "analysis" | "building" | "ready" | "failed";

function VersionBanner(props: {
  card: RecipeCard;
  rows: RecipeCard[];
  onSelectRecipe: (recipeId: string) => void;
}): JSX.Element {
  const { card } = props;
  const members = props.rows
    .filter((row) => row.family_id === card.family_id)
    .sort((a, b) => a.version_number - b.version_number);
  const parent = card.parent_version === null ? null : {
    id: card.parent_version,
    number: card.parent_version_number,
  };
  return (
    <div className="rc-version-banner">
      <span className="rc-version-label">
        {`Version ${card.version_number} of ${card.family_size}`}
        {card.parent_version === null
          ? " — Original"
          : ` — Based on Version ${card.parent_version_number}`}
        {!card.is_latest ? " — Previous Version" : ""}
      </span>
      {parent !== null && (
        <button
          type="button"
          className="rc-btn rc-btn-link"
          onClick={() => props.onSelectRecipe(parent.id)}
        >
          {parent.number === null
            ? "Open parent version"
            : `Open Version ${parent.number}`}
        </button>
      )}
      {members.length > 1 && (
        <details className="rc-version-history">
          <summary>Version History</summary>
          <ul>
            {members.map((member) => (
              <li key={member.recipe_id}>
                <button
                  type="button"
                  className="rc-btn rc-btn-link"
                  disabled={member.recipe_id === card.recipe_id}
                  onClick={() =>
                    props.onSelectRecipe(member.recipe_id)}
                >
                  {`Version ${member.version_number} — `
                    + `${member.display_name}`}
                  {member.recipe_id === card.recipe_id
                    ? " (current)"
                    : ""}
                </button>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

export default function RecipeWorkspace(props: {
  recipeId: string | null;
  onSelectRecipe: (recipeId: string) => void;
  onNewRecipe: () => void;
}) {
  const [card, setCard] = useState<RecipeCard | null | undefined>(
    undefined,
  );
  const [rows, setRows] = useState<RecipeCard[]>([]);
  const [phase, setPhase] = useState<Phase>("loading");
  const [detail, setDetail] = useState<string | null>(null);
  const [runStatus, setRunStatus] = useState<string | null>(null);
  const [runFailed, setRunFailed] = useState(false);
  const [nonce, setNonce] = useState(0);
  const buildStarted = useRef(false);
  const target = props.recipeId ?? DEFAULT_RECIPE_ID;

  useEffect(() => {
    buildStarted.current = false;
    setRunFailed(false);
    setDetail(null);
  }, [target]);

  useEffect(() => {
    let cancelled = false;
    setCard(undefined);
    setPhase("loading");
    setDetail(null);
    listRecipes()
      .then((rows) => {
        if (cancelled) return;
        const found = rows.find((row) => row.recipe_id === target) ?? null;
        setRows(rows);
        setCard(found);
        if (found === null) {
          setPhase("failed");
          setDetail("This recipe is no longer available.");
        } else if (found.has_snapshot) {
          setPhase("ready");
        } else if (found.status === "failed") {
          setPhase("failed");
          setDetail(
            found.build_error ?? "Recipe generation failed.",
          );
        } else if (!found.source_run_id) {
          setPhase("failed");
          setDetail("This recipe has no linked video analysis.");
        } else {
          setPhase("analysis");
        }
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setPhase("failed");
        setDetail(
          err instanceof ApiError
            ? `Could not load recipes (${err.message}).`
            : "Could not load recipes.",
        );
      });
    return () => {
      cancelled = true;
    };
  }, [target, nonce]);

  useEffect(() => {
    if (phase !== "analysis" && phase !== "building") return;
    const runId = card?.source_run_id;
    if (!runId) return;
    let cancelled = false;

    const refreshCard = async (): Promise<boolean> => {
      const rows = await listRecipes();
      if (cancelled) return false;
      const found = rows.find((row) => row.recipe_id === target) ?? null;
      setCard(found);
      if (found?.has_snapshot) {
        setPhase("ready");
        return true;
      }
      if (found?.status === "failed") {
        setPhase("failed");
        setDetail(found.build_error ?? "Recipe generation failed.");
        return true;
      }
      return false;
    };

    const tick = async (): Promise<void> => {
      try {
        const job = await fetchAnalysis(runId);
        if (cancelled) return;
        setRunStatus(job.status);
        if (job.status === "failed") {
          setRunFailed(true);
          setPhase("failed");
          setDetail(
            job.error
              ? `Video analysis failed: ${job.error}`
              : "Video analysis failed.",
          );
          return;
        }
        if (job.status !== "completed") return;
        if (buildStarted.current) {
          await refreshCard().catch(() => null);
          return;
        }
        buildStarted.current = true;
        setPhase("building");
        try {
          await buildRecipe(target);
          if (cancelled) return;
          await refreshCard().catch(() => null);
        } catch (err: unknown) {
          if (cancelled) return;
          const message =
            err instanceof ApiError ? err.message : "Build failed.";
          if (
            err instanceof ApiError &&
            err.status === 409 &&
            message.includes("already in progress")
          ) {
            buildStarted.current = false;
            return;
          }
          buildStarted.current = false;
          setPhase("failed");
          setDetail(message);
        }
      } catch {
        /* transient fetch error: keep polling */
      }
    };

    void tick();
    const timer = window.setInterval(() => void tick(), POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [phase, card, target]);

  function retry(): void {
    buildStarted.current = false;
    setRunFailed(false);
    setDetail(null);
    setPhase("analysis");
    setNonce((value) => value + 1);
  }

  return (
    <section className="card rc-workspace" aria-label="Chef review">
      {phase === "loading" && (
        <p className="muted">Loading the recipe…</p>
      )}
      {phase === "failed" && card === null && (
        <div className="rc-doc">
          <h2 className="rc-title">Recipe not found</h2>
          <p className="muted">
            This recipe is no longer available. Choose another
            recipe from Home.
          </p>
        </div>
      )}
      {phase === "failed" && card !== null && card !== undefined && (
        <div className="rc-doc">
          <p className="rc-kicker">Something went wrong</p>
          <h2 className="rc-title">
            {card.display_name ?? card.name}
          </h2>
          <p className="rc-error" role="alert">
            {detail ?? "Recipe generation failed."}
          </p>
          <div className="rc-actions">
            {!runFailed && (
              <button
                type="button"
                className="rc-btn rc-btn-primary"
                onClick={retry}
              >
                Try again
              </button>
            )}
            <button
              type="button"
              className="rc-btn"
              onClick={props.onNewRecipe}
            >
              Create a new recipe
            </button>
          </div>
        </div>
      )}
      {(phase === "analysis" || phase === "building") &&
        card !== null &&
        card !== undefined && (
          <div className="rc-doc">
            <p className="rc-kicker">AI processing</p>
            <h2 className="rc-title">
              {card.display_name ?? card.name}
            </h2>
            <p className="muted">
              {phase === "analysis"
                ? `Analyzing your video${
                    runStatus ? ` (${runStatus})` : ""
                  }…`
                : "Writing your recipe draft…"}
            </p>
            <p className="muted small">
              This view updates automatically when the draft is
              ready; nothing is shown until the AI has produced
              real content.
            </p>
          </div>
        )}
      {phase === "ready" && card?.has_snapshot && (
        <>
          <VersionBanner
            card={card}
            rows={rows}
            onSelectRecipe={props.onSelectRecipe}
          />
          <RecipeEditor
            key={target}
            recipeId={target}
            onNewVersion={(nextId) => props.onSelectRecipe(nextId)}
          />
        </>
      )}
    </section>
  );
}
