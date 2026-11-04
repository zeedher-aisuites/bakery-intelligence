import { useCallback, useEffect, useState } from "react";
import {
  ApiError,
  fetchHealth,
  type Health,
} from "./api";
import RecipeWorkspace from "./RecipeWorkspace";
import ProcessingQueue from "./ProcessingQueue";
import ReviewApp from "./ReviewWorkspace";
import UploadWorkflow from "./UploadWorkflow";

type BackendState =
  | { kind: "checking" }
  | { kind: "online"; health: Health }
  | { kind: "offline"; message: string };

function initialRecipeId(): string | null {
  try {
    const match = window.location.hash.match(/^#recipe\/([\w.\-]+)$/);
    if (match) return match[1];
  } catch {
    /* non-browser render; fall through */
  }
  return null;
}

function initialView(): "run" | "review" | "queue" | "fusion" {
  try {
    if (window.location.hash === "#chef-review") return "fusion";
    if (initialRecipeId() !== null) return "fusion";
  } catch {
    /* non-browser render; fall through */
  }
  return "review";
}

export default function App() {
  const [backend, setBackend] = useState<BackendState>({ kind: "checking" });
  const [view, setView] = useState<
    "run" | "review" | "queue" | "fusion"
  >(initialView);
  const [queueRunId, setQueueRunId] = useState<string | null>(null);
  const [queueNonce, setQueueNonce] = useState(0);
  const [homeNonce, setHomeNonce] = useState(0);
  const [reviewRunId, setReviewRunId] = useState<string | null>(null);
  const [recipeId, setRecipeId] = useState<string | null>(
    initialRecipeId,
  );

  const goHome = useCallback(() => {
    try {
      window.location.hash = "";
    } catch {
      /* non-browser render */
    }
    setHomeNonce((value) => value + 1);
    setReviewRunId(null);
    setView("review");
  }, []);

  const openRecipe = useCallback((nextId: string) => {
    try {
      window.location.hash = `#recipe/${nextId}`;
    } catch {
      /* non-browser render */
    }
    setRecipeId(nextId);
    setView("fusion");
  }, []);

  const selectRecipe = useCallback((nextId: string) => {
    try {
      window.location.hash = `#recipe/${nextId}`;
    } catch {
      /* non-browser render */
    }
    setRecipeId(nextId);
  }, []);

  const checkBackend = useCallback(async () => {
    try {
      const health = await fetchHealth();
      setBackend({ kind: "online", health });
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : "Backend unreachable.";
      setBackend({ kind: "offline", message });
    }
  }, []);

  const goNewAnalysis = useCallback(() => setView("run"), []);

  useEffect(() => {
    void checkBackend();
  }, [checkBackend]);

  return (
    <div className="page">
      <header className="masthead">
        <div>
          <button
            type="button"
            className="brand-home"
            onClick={goHome}
            aria-label="Bakery Intelligence home"
          >
            <span className="brand-title">Bakery Intelligence</span>
          </button>
          <p className="subtitle">
            AI video analysis with evidence-backed measurements
            and human validation.
          </p>
          <nav className="topnav" aria-label="Primary">
            <button
              type="button"
              className={view === "review" ? "active" : ""}
              onClick={goHome}
            >
              Home
            </button>
            <button
              type="button"
              className={view === "run" ? "active" : ""}
              onClick={goNewAnalysis}
            >
              New Recipe
            </button>
            <button
              type="button"
              className={view === "fusion" ? "active" : ""}
              onClick={() => setView("fusion")}
            >
              Chef Review
            </button>
          </nav>
        </div>
        {backend.kind !== "online" && (
          <BackendBadge backend={backend} onRetry={() => void checkBackend()} />
        )}
      </header>

      {view === "run" ? (
        <UploadWorkflow
          onOpenRecipe={openRecipe}
          onOpenReview={(runId) => {
            setReviewRunId(runId);
            setView("review");
          }}
          onViewQueue={(runId) => {
            setQueueRunId(runId);
            setQueueNonce((value) => value + 1);
            setView("queue");
          }}
        />
      ) : view === "queue" ? (
        <ProcessingQueue
          key={queueNonce}
          highlightRunId={queueRunId}
          onOpenReview={(runId) => {
            setReviewRunId(runId);
            setView("review");
          }}
          onNewAnalysis={goNewAnalysis}
        />
      ) : view === "fusion" ? (
        <RecipeWorkspace
          key={recipeId ?? "default"}
          recipeId={recipeId}
          onSelectRecipe={selectRecipe}
          onNewRecipe={goNewAnalysis}
        />
      ) : (
        <ReviewApp
          key={`${homeNonce}-${reviewRunId ?? "home"}`}
          initialRunId={reviewRunId}
          onNewAnalysis={goNewAnalysis}
          onOpenRecipe={openRecipe}
        />
      )}
    </div>
  );
}

function BackendBadge({
  backend,
  onRetry,
}: {
  backend: BackendState;
  onRetry: () => void;
}) {
  if (backend.kind === "checking") {
    return <span className="pill pill-pending">checking backend…</span>;
  }
  if (backend.kind === "online") {
    return (
      <span className="pill pill-ok" title={backend.health.version}>
        backend online · {backend.health.storage_backend}
      </span>
    );
  }
  return (
    <span className="badge-row">
      <span className="pill pill-bad">backend offline</span>
      <button className="ghost" onClick={onRetry}>Retry</button>
    </span>
  );
}
