import { useCallback, useEffect, useState } from "react";
import {
  ApiError,
  fetchHealth,
  type Health,
} from "./api";
import ProcessingQueue from "./ProcessingQueue";
import ReviewApp from "./ReviewWorkspace";
import UploadWorkflow from "./UploadWorkflow";

type BackendState =
  | { kind: "checking" }
  | { kind: "online"; health: Health }
  | { kind: "offline"; message: string };

export default function App() {
  const [backend, setBackend] = useState<BackendState>({ kind: "checking" });
  const [view, setView] = useState<"run" | "review" | "queue">("review");
  const [queueRunId, setQueueRunId] = useState<string | null>(null);
  const [queueNonce, setQueueNonce] = useState(0);
  const [homeNonce, setHomeNonce] = useState(0);
  const [reviewRunId, setReviewRunId] = useState<string | null>(null);

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
          <h1>Bakery Intelligence</h1>
          <p className="subtitle">
            AI video analysis with evidence-backed measurements
            and human validation.
          </p>
          <nav className="topnav" aria-label="Primary">
            <button
              type="button"
              onClick={() => {
                setHomeNonce((value) => value + 1);
                setReviewRunId(null);
                setView("review");
              }}
            >
              Home
            </button>
            <button
              type="button"
              className={view === "run" ? "active" : ""}
              onClick={goNewAnalysis}
            >
              + New analysis
            </button>
            <button
              type="button"
              className={view === "queue" ? "active" : ""}
              onClick={() => setView("queue")}
            >
              My recipes
            </button>
          </nav>
        </div>
        {backend.kind !== "online" && (
          <BackendBadge backend={backend} onRetry={() => void checkBackend()} />
        )}
      </header>

      {view === "run" ? (
        <UploadWorkflow
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
      ) : (
        <ReviewApp
          key={`${homeNonce}-${reviewRunId ?? "home"}`}
          initialRunId={reviewRunId}
          onNewAnalysis={goNewAnalysis}
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
