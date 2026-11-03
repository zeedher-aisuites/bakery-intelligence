import { useState } from "react";
import { saveFrameClassification } from "./api";

// Human classification of the selected observation, persisted
// separately from the original model output.
const ACTIONS = [
  { id: "confirm", label: "Confirm prediction" },
  { id: "flag", label: "Needs human review" },
  { id: "cannot_determine", label: "Cannot determine" },
  { id: "not_relevant", label: "Not relevant" },
] as const;

export default function FrameClassification(props: {
  runId: string;
  frameIndex: number | null;
  onChanged?: () => void;
}): JSX.Element {
  const [reviewer, setReviewer] = useState("");
  const [value, setValue] = useState("");
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function act(action: string): Promise<void> {
    if (props.frameIndex === null) return;
    setBusy(true);
    setMessage(null);
    try {
      await saveFrameClassification(
        props.runId, props.frameIndex, action,
        value.trim() ? Number(value) : null, reviewer);
      setMessage("Classification saved. The original model output is unchanged.");
      props.onChanged?.();
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Classification failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="frame-classify">
      <h4>Human classification</h4>
      <p className="muted small">
        Classifies only this observation. Never edits the model output.
      </p>
      <label className="field">
        <span>Reviewer identity (required)</span>
        <input
          value={reviewer}
          onChange={(event) => setReviewer(event.target.value)}
          placeholder="e.g. chef-maria"
        />
      </label>
      <label className="field">
        <span>Corrected value (only when providing one)</span>
        <input
          value={value}
          onChange={(event) => setValue(event.target.value)}
          placeholder="blank unless correcting"
        />
      </label>
      <div className="flow-actions">
        {ACTIONS.map((item) => (
          <button
            key={item.id} type="button" className="btn-ghost"
            disabled={busy || props.frameIndex === null}
            onClick={() => void act(item.id)}
          >
            {item.label}
          </button>
        ))}
      </div>
      {message && <p className="muted small" role="status">{message}</p>}
    </div>
  );
}
