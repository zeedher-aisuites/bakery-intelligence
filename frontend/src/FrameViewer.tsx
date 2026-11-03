import { useState } from "react";
import { frameUrl } from "./api";
import { formatVideoTime } from "./recipe";

export interface ViewerFrame {
  frame_index: number;
  timestamp_seconds: number | null;
  image_name: string | null;
  image_available: boolean;
}

function neighbor(
  visible: ViewerFrame[],
  position: number,
  delta: -1 | 1,
): ViewerFrame | null {
  return visible[position + delta] ?? null;
}

/**
 * Shared proportional frame viewer. The image keeps its aspect ratio
 * with object-fit: contain, never crops, and reports timestamp plus
 * sample identity. One instance is used per navigation scope so the
 * finding workflow and Evidence Explorer never share selection state.
 */
export default function FrameViewer(props: {
  runId: string;
  frames: ViewerFrame[];
  selected: number | null;
  onSelect: (frameIndex: number) => void;
  label?: string;
  thumbs?: boolean;
}): JSX.Element {
  const [zoom, setZoom] = useState(1);
  const visible = props.frames;
  const fallback = visible[0] ?? null;
  const current = visible.find(
    (frame) => frame.frame_index === props.selected,
  ) ?? fallback;
  const position = current === null ? 0 : visible.indexOf(current);

  function step(delta: -1 | 1): void {
    if (visible.length === 0) return;
    const next = visible[
      (position + delta + visible.length) % visible.length
    ];
    if (next) props.onSelect(next.frame_index);
  }

  function thumbnail(
    frame: ViewerFrame | null,
    label: string,
  ): JSX.Element {
    if (props.thumbs === false) return <></>;
    if (frame === null) return <span className="ev-thumb empty" />;
    return (
      <button
        type="button"
        className="ev-thumb"
        aria-label={`${label}: sample ${frame.frame_index}`}
        onClick={() => props.onSelect(frame.frame_index)}
      >
        {frame.image_available && frame.image_name ? (
          <img
            src={frameUrl(props.runId, frame.image_name)}
            alt=""
            loading="lazy"
          />
        ) : (
          <span className="ev-thumb empty">No image</span>
        )}
      </button>
    );
  }

  const singleFrame = visible.length < 2;

  return (
    <section
      aria-label={props.label ?? "Frame viewer"}
      className="ev-viewer"
    >
      <div className="ev-stage">
        <button
          type="button"
          className="ev-step"
          aria-label="Previous frame"
          disabled={singleFrame}
          onClick={() => step(-1)}
        >
          ‹
        </button>
        {thumbnail(neighbor(visible, position, -1), "Previous frame")}
        <figure className="ev-figure">
          <div className="ev-viewport">
            {current?.image_available && current.image_name ? (
              <img
                className="ev-image"
                style={{ transform: `scale(${zoom})` }}
                src={frameUrl(props.runId, current.image_name)}
                alt={`Sample ${current.frame_index} at `
                  + `${formatVideoTime(current.timestamp_seconds)}`}
              />
            ) : (
              <div className="ev-noimage">
                No retained image for this frame.
              </div>
            )}
          </div>
          <figcaption>
            <strong>
              {formatVideoTime(current?.timestamp_seconds ?? null)}
            </strong>
            <span className="muted small">
              {" "}Sample #{current?.frame_index ?? "—"}
              {" · "}
              {current?.image_name ?? "no file retained"}
            </span>
          </figcaption>
        </figure>
        {thumbnail(neighbor(visible, position, 1), "Next frame")}
        <button
          type="button"
          className="ev-step"
          aria-label="Next frame"
          disabled={singleFrame}
          onClick={() => step(1)}
        >
          ›
        </button>
      </div>
      <div className="ev-scrub">
        <input
          type="range"
          min={0}
          max={Math.max(0, visible.length - 1)}
          value={position}
          disabled={singleFrame}
          aria-label="Frame timeline position"
          onChange={(event) => {
            const next = visible[Number(event.target.value)];
            if (next) props.onSelect(next.frame_index);
          }}
        />
        <span className="muted small">
          Frame {visible.length === 0 ? 0 : position + 1} of{" "}
          {visible.length}
        </span>
      </div>
      <div className="ev-zoom" role="group" aria-label="Zoom">
        <button
          type="button"
          aria-label="Zoom out"
          onClick={() => setZoom(Math.max(1, zoom - 0.5))}
        >
          −
        </button>
        <span className="muted small">{zoom.toFixed(1)}×</span>
        <button
          type="button"
          aria-label="Zoom in"
          onClick={() => setZoom(Math.min(3, zoom + 0.5))}
        >
          +
        </button>
        <button type="button" onClick={() => setZoom(1)}>
          Reset
        </button>
      </div>
    </section>
  );
}
