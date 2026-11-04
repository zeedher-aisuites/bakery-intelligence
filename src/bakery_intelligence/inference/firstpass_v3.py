"""Focused large-batch FirstPass (V3 experiment).

Alternative to the single-frame ``FirstPassRunner``
(``firstpass.py``) and the contextual ``FirstPassV2Runner``
(``firstpass_v2.py``): fixed groups of up to 50 consecutive
frames share one provider request. Each batch receives its
own images plus a compact deterministic context built only
from the validated observations of the immediately preceding
batch. There is no model-generated rolling context and no
recursive accumulation: the full observation history stays
on disk for downstream reconstruction instead of being
reinjected into every request.

Scope and safety properties mirror V2: strictly sequential
dispatch in input order, single-submit billable invariants
(one transport attempt per submit, no uncertain replay),
explicit gap notes for failed groups, per-group
checkpoints, and a request/cost ledger with strict
unknown-cost halting. The module performs no network access
itself: inference flows only through an injected client
factory, so automated tests run fully offline.

Downstream compatibility: per-frame records reuse the
``build_observation`` envelope key-for-key (timestamps,
manifest anchor, call provenance, terminal state) with a
focused ``model_fields`` contract. Shared keys keep V1
semantics; downstream readers use tolerant ``.get``
access. V3-only provenance travels inside the ``call``
mapping plus dedicated group-level artifacts.
"""

from __future__ import annotations

import datetime
import hashlib
import io
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, ImageDraw, ImageFont

from bakery_intelligence.inference.firstpass import (
    MARKER_PREFIX,
    MARKER_SUFFIX,
    RETRYABLE_CODES,
    STOP_COST_CEILING,
    STOP_COST_RESERVE,
    STOP_REQUEST_LIMIT,
    STOP_STRICT_UNKNOWN_COST,
    TERMINAL_EMPTY,
    TERMINAL_EXHAUSTED,
    TERMINAL_FAILED,
    TERMINAL_INVALID_JSON,
    TERMINAL_UNSUPPORTED,
    TERMINAL_VALID,
    TIMESTAMP_BASIS,
    UNCERTAIN_REFUSE,
    UNCERTAIN_RERUN,
    FirstPassError,
    FrameRef,
    InferenceError,
    InferenceResponse,
    InferenceUsage,
    SourceRef,
    estimate_cost,
    is_token_limit_reason,
    parse_model_json,
    prompt_hash,
)
from bakery_intelligence.inference.firstpass_v2 import (
    IMAGE_LIMIT_CODES,
    STOP_CONSECUTIVE_EXHAUSTED,
    estimate_context_tokens,
    group_frames,
    group_id_for,
)
from bakery_intelligence.inference.meta_spark import (
    MAX_IMAGES_PER_REQUEST,
    MODEL_SPARK,
)

# Experiment identity. Recorded as ``pass_id`` so V1, V2,
# and V3 observations never mix silently downstream.
PASS_ID_V3 = "firstpass_v3"

# Default group shape: 50 consecutive frames per request
# (the verified per-request image maximum), with the final
# group allowed to be partial.
DEFAULT_FOCUSED_GROUP_SIZE = 50

# Default output budget for one batch request. Fifty
# per-frame observations plus optional events need far more
# headroom than a single frame; 65536 sits at the lowest
# externally reported muse-spark-1.3 output maximum and is
# verified by live acceptance, never assumed.
DEFAULT_FOCUSED_TOKEN_BUDGET = 65536

# Focused observation contract: one entry per frame. Shared
# keys keep V1 semantics and vocabularies; V3-only keys
# (unit, utensil, material, action) extend the evidence.
V3_MODEL_OUTPUT_FIELDS = (
    "scale_readable",
    "scale_display_verbatim",
    "scale_unit_visible",
    "scale_state",
    "tare_action_visible",
    "container_description",
    "container_changed_hint",
    "primary_utensil",
    "primary_visible_material",
    "action",
    "addition_or_removal_visible",
    "confidence",
    "ambiguity_notes",
)

# Deterministic previous-batch context bound (~375 tokens
# at ~4 chars per token). The builder truncates ambiguity
# notes first, then the whole block with a marker.
PREV_CONTEXT_MAX_CHARS = 1500
PREV_CONTEXT_TRUNCATION_MARKER = "[context truncated to bound]"

# Batch response contract name for structured decoding.
FOCUSED_SCHEMA_NAME = "FocusedFrameBatch"


V3_ANCHORED_OUTPUT_FIELDS = V3_MODEL_OUTPUT_FIELDS + (
    "visual_frame_id",
)


def build_focused_schema(*, visual_anchor: bool = False
                         ) -> dict[str, Any]:
    """Build the V3 batch JSON schema (flat, no $ref)."""
    required = (V3_ANCHORED_OUTPUT_FIELDS if visual_anchor
                else V3_MODEL_OUTPUT_FIELDS)
    observation: dict[str, Any] = {
        "type": "object",
        "properties": {
            "scale_readable": {"type": "boolean"},
            "scale_display_verbatim": {
                "type": ["string", "null"]},
            "scale_unit_visible": {
                "type": ["string", "null"]},
            "scale_state": {
                "type": "string",
                "enum": ["stable", "changing", "zero", "off",
                         "occluded", "unknown"],
            },
            "tare_action_visible": {"type": "boolean"},
            "container_description": {
                "type": ["string", "null"]},
            "container_changed_hint": {"type": "boolean"},
            "primary_utensil": {"type": ["string", "null"]},
            "primary_visible_material": {
                "type": ["string", "null"]},
            "action": {"type": ["string", "null"]},
            "addition_or_removal_visible": {
                "type": "string",
                "enum": ["addition", "removal", "none",
                         "unknown"],
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
            "ambiguity_notes": {"type": ["string", "null"]},
            "visual_frame_id": {"type": "string"},
        },
        "required": list(required),
        "additionalProperties": False,
    }
    if not visual_anchor:
        del observation["properties"]["visual_frame_id"]
    return {
        "type": "object",
        "properties": {
            "frames": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "frame_index": {"type": "integer"},
                        "timestamp_seconds": {
                            "type": "number"},
                        "observation": observation,
                    },
                    "required": ["frame_index",
                                 "timestamp_seconds",
                                 "observation"],
                    "additionalProperties": False,
                },
            },
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "frame_indices": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                    },
                    "required": ["description",
                                 "frame_indices"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["frames"],
        "additionalProperties": False,
    }


FOCUSED_JSON_SCHEMA = build_focused_schema()

FOCUSED_ANCHORED_SCHEMA_NAME = "FocusedFrameBatchAnchored"
FOCUSED_ANCHORED_JSON_SCHEMA = build_focused_schema(
    visual_anchor=True)

# Frozen batch prompt. The per-request frame table and the
# deterministic previous-batch context are appended by
# ``build_focused_prompt``. The four evidence categories
# and compact-output rules are fixed here.
V3_BASE_PROMPT = """Study consecutive bakery-video frames as production evidence.
The attached images are chronological: IMAGE 1 is earliest, the last image is latest.
Reply with exactly one JSON object and no other text:
{
  "frames": [
    {"frame_index": 0, "timestamp_seconds": 0.0, "observation": {
      "scale_readable": true/false,
      "scale_display_verbatim": "exact display digits or null",
      "scale_unit_visible": "visible unit or null",
      "scale_state": "stable/changing/zero/off/occluded/unknown",
      "tare_action_visible": true/false,
      "container_description": "short phrase or null",
      "container_changed_hint": true/false,
      "primary_utensil": "short phrase or null",
      "primary_visible_material": "short phrase or null",
      "action": "short verb phrase or null",
      "addition_or_removal_visible": "addition/removal/none/unknown",
      "confidence": "high/medium/low",
      "ambiguity_notes": "short note or null"}}
  ],
  "events": [{"description": "short phrase", "frame_indices": [0]}]
}
Focus, in order: (1) scale readings exactly as displayed, with units only when visible; readability, tare actions, additions, removals, changes between readings; (2) the primary container and utensils, changes, interactions; (3) the primary visible ingredient or material only when supported, else descriptive labels such as white powder, liquid, or unknown; (4) production actions: additions, removals, tare, transfers, mixing, container changes.
Rules: one frames entry per image, same order, exact frame_index and timestamp_seconds from the table; keep every string to a few words; no narration or explanations; null whenever evidence is insufficient; never interpret a display number as net quantity; never invent units, identities, or continuity; list "events" only for multi-frame occurrences worth naming, otherwise an empty list."""


# --- Visual frame anchoring. ---
#
# Each submitted image carries a printed identifier band above
# the original pixels so the model can echo which physical
# image an observation describes. The band is an attribution
# aid, not proof of semantic correctness: a correct identifier
# echo does not prove the described pixels belong to that
# image, so experiments evaluate observation correctness
# separately against verified references.
ANCHOR_BAND_HEIGHT = 128
ANCHOR_BAND_COLOR = (0, 0, 0)
ANCHOR_TEXT_COLOR = (255, 255, 255)
ANCHOR_RULE_COLOR = (255, 214, 0)
ANCHOR_RULE_HEIGHT = 6
ANCHOR_FONT_SIZE = 40
ANCHOR_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
ANCHOR_CODE_LENGTH = 6
ANCHOR_DEFAULT_SALT = "bakery-v3a"


def visual_frame_id_for(frame_index: int) -> str:
    """Return the deterministic printed identifier."""
    if (isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0):
        raise FirstPassError(
            "INVALID_CONFIG",
            f"frame_index must be a non-negative int, "
            f"got {frame_index!r}.")
    return f"F{frame_index:06d}"


def visual_check_code(frame_index: int, *,
                      salt: str = ANCHOR_DEFAULT_SALT) -> str:
    """Return a short deterministic redundant code."""
    visual_frame_id_for(frame_index)
    digest = hashlib.sha256(
        f"{salt}|{frame_index}".encode("utf-8")).digest()
    alphabet = ANCHOR_CODE_ALPHABET
    return "".join(
        alphabet[byte % len(alphabet)]
        for byte in digest[:ANCHOR_CODE_LENGTH])


@dataclass
class AnchoredImage:
    """Provenance for one labeled submission image."""

    frame_index: int
    timestamp_seconds: float
    visual_frame_id: str
    check_code: str
    source_path: Path
    source_sha256: str
    source_size: tuple[int, int]
    prepared_path: Path
    prepared_sha256: str
    prepared_size: tuple[int, int]


def prepare_anchored_image(
        *, source: Path, frame_index: int,
        timestamp_seconds: float, dest_dir: Path,
        salt: str = ANCHOR_DEFAULT_SALT) -> AnchoredImage:
    """Extend the canvas with an identifier band (lossless).

    The source pixels are pasted 1:1 below the band with no
    resampling, cropping, or overwrite; the prepared image is
    saved as PNG so the original region stays byte-exact.
    """
    payload = source.read_bytes()
    with Image.open(io.BytesIO(payload)) as handle:
        photo = handle.convert("RGB")
    width, height = photo.size
    visual_id = visual_frame_id_for(frame_index)
    code = visual_check_code(frame_index, salt=salt)
    canvas = Image.new(
        "RGB", (width, height + ANCHOR_BAND_HEIGHT),
        ANCHOR_BAND_COLOR)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=ANCHOR_FONT_SIZE)
    lines = [f"FRAME-ID: {visual_id}", f"CODE: {code}"]
    cursor = 12
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font)
        text_width = box[2] - box[0]
        draw.text(((width - text_width) / 2, cursor), line,
                  font=font, fill=ANCHOR_TEXT_COLOR)
        cursor += (box[3] - box[1]) + 8
    draw.rectangle(
        [(0, ANCHOR_BAND_HEIGHT - ANCHOR_RULE_HEIGHT),
         (width, ANCHOR_BAND_HEIGHT)],
        fill=ANCHOR_RULE_COLOR)
    canvas.paste(photo, (0, ANCHOR_BAND_HEIGHT))
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"anchored_{frame_index:08d}.png"
    canvas.save(dest, format="PNG")
    prepared = dest.read_bytes()
    return AnchoredImage(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        visual_frame_id=visual_id, check_code=code,
        source_path=source,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        source_size=(width, height), prepared_path=dest,
        prepared_sha256=hashlib.sha256(prepared).hexdigest(),
        prepared_size=canvas.size)


def verify_anchored_image(record: AnchoredImage) -> bool:
    """Confirm the photo region is byte-identical to source."""
    with Image.open(record.source_path) as handle:
        want = handle.convert("RGB").tobytes()
    with Image.open(record.prepared_path) as handle:
        photo = handle.convert("RGB")
    width, height = record.source_size
    if photo.size != (width, height + ANCHOR_BAND_HEIGHT):
        return False
    region = photo.crop(
        (0, ANCHOR_BAND_HEIGHT, width,
         height + ANCHOR_BAND_HEIGHT))
    return region.tobytes() == want


def _truncate_notes(notes: list[str], room: int) -> list[str]:
    """Fit ambiguity notes into room, shortening deterministically."""
    kept: list[str] = []
    used = 0
    for note in notes:
        short = " ".join(note.split())
        if len(short) > 120:
            short = short[:117].rstrip() + "..."
        need = len(short) + (2 if kept else 0)
        if used + need > room:
            break
        kept.append(short)
        used += need
    return kept


def build_previous_batch_context(
        *, prev_group_id: str,
        entries: Sequence[dict[str, Any]]) -> str:
    """Build the deterministic context for the next batch.

    ``entries`` carries the validated ``(frame_index,
    observation)`` pairs of the immediately preceding batch
    only; nothing older is consulted and no model-generated
    narrative is forwarded. An empty batch yields a fixed
    fresh-start note. Output is capped deterministically.
    """
    header = f"PREVIOUS BATCH {prev_group_id}"
    rows = [entry for entry in entries
            if isinstance(entry, dict)
            and isinstance(entry.get("frame_index"), int)
            and isinstance(entry.get("observation"), dict)]
    if not rows:
        return (f"{header}: no usable observations; treat "
                f"this batch as a fresh start.")
    first = rows[0]["frame_index"]
    last = rows[-1]["frame_index"]
    obs = [row["observation"] for row in rows]

    def latest(field: str) -> tuple[Any, int | None]:
        for row in reversed(rows):
            value = row["observation"].get(field)
            if value is not None and value is not False:
                return value, row["frame_index"]
        return None, None

    scale: tuple[Any, int | None] = (None, None)
    for row in reversed(rows):
        candidate = row["observation"]
        if (candidate.get("scale_readable") is True
                and candidate.get("scale_display_verbatim")):
            scale = (candidate.get("scale_display_verbatim"),
                     row["frame_index"])
            break
    unit, _ = latest("scale_unit_visible")
    container, _ = latest("container_description")
    changed = any(row["observation"].get(
        "container_changed_hint") is True for row in rows)
    utensil, _ = latest("primary_utensil")
    material, _ = latest("primary_visible_material")
    action, action_frame = latest("action")
    ongoing: list[str] = []
    if obs[-1].get("scale_state") == "changing":
        ongoing.append("scale changing at batch end")
    if changed:
        ongoing.append("container change flagged")
    if obs[-1].get("tare_action_visible") is True:
        ongoing.append("tare visible at batch end")
    seen: set[str] = set()
    notes: list[str] = []
    for row in rows:
        note = row["observation"].get("ambiguity_notes")
        if (isinstance(note, str) and note.strip()
                and note not in seen):
            seen.add(note)
            notes.append(note)
    lines = [
        f"{header} ({len(rows)} frames, "
        f"indices {first}-{last}):",
        f"- last readable scale: "
        f"{scale[0]!r} at frame {scale[1]}"
        if scale[0] is not None
        else "- last readable scale: none",
        f"- visible unit: {unit!r}"
        if unit is not None
        else "- visible unit: unknown",
        f"- container: {container!r}"
        f"{' (change flagged)' if changed else ''}"
        if container is not None
        else "- container: unknown",
        f"- utensil: {utensil!r}"
        if utensil is not None
        else "- utensil: unknown",
        f"- visible material: {material!r}"
        if material is not None
        else "- visible material: unknown",
        f"- last action: {action!r} at frame {action_frame}"
        if action is not None
        else "- last action: none",
        ("- ongoing: " + "; ".join(ongoing)
         if ongoing else "- ongoing: none indicated"),
    ]
    base = "\n".join(lines)
    room = (PREV_CONTEXT_MAX_CHARS
            - len(base) - len("\n- open ambiguities: ")
            - len(PREV_CONTEXT_TRUNCATION_MARKER) - 1)
    kept = _truncate_notes(notes, max(0, room))
    tail = ("- open ambiguities: " + "; ".join(kept)
            if kept else "- open ambiguities: none recorded")
    text = base + "\n" + tail
    if len(text) > PREV_CONTEXT_MAX_CHARS:
        room = (PREV_CONTEXT_MAX_CHARS
                - len(PREV_CONTEXT_TRUNCATION_MARKER) - 1)
        text = (text[:room].rstrip() + "\n"
                + PREV_CONTEXT_TRUNCATION_MARKER)
    return text


class FocusedClient(Protocol):
    """Client seam for one multi-image batch request.

    ``MetaSparkClient.submit`` satisfies this protocol: the
    runner passes the batch prompt, image paths, key, token
    budget, and the V3 schema, and receives a normalized
    ``InferenceResponse`` or a coded ``InferenceError``.
    """

    def submit(
        self,
        prompt_text: str,
        image_paths: Sequence[Path],
        api_key: str,
        max_output_tokens: int,
        response_schema: dict[str, Any] | None = ...,
        response_schema_name: str | None = ...,
    ) -> InferenceResponse:
        """Submit one batch request."""
        ...  # pragma: no cover - protocol shape


V3_ANCHOR_PROMPT = """Each image carries a printed black identifier band above the photo showing FRAME-ID and CODE. Read the FRAME_ID from the band of the image you are describing and return it as visual_frame_id. Describe only the photo below that band: never borrow a scale reading, container, material, or action from a neighboring image, and never assign one image's reading to the next image. Do not infer the identifier from image position; if the band is unreadable, return the expected identifier from the table and note the uncertainty in ambiguity_notes. The band digits are identifiers, never scale readings."""


def build_focused_prompt(
        *,
        frames: Sequence[FrameRef],
        group_sequence: int,
        total_groups: int,
        incoming_context: str | None,
        visual_ids: Sequence[str] | None = None) -> str:
    """Assemble the full prompt for one batch request."""
    if visual_ids is not None and len(visual_ids) != len(frames):
        raise FirstPassError(
            "INVALID_CONFIG",
            f"visual_ids has {len(visual_ids)} entries for "
            f"{len(frames)} frames.")
    lines = [V3_BASE_PROMPT, "",
             f"BATCH {group_sequence + 1} OF {total_groups}: "
             f"{len(frames)} image(s), in this exact order:"]
    for position, frame in enumerate(frames, start=1):
        row = (f"IMAGE {position}: "
               f"frame_index={frame.frame_index}, "
               f"timestamp_seconds={frame.timestamp_seconds}")
        if visual_ids is not None:
            row += (f", visual_id={visual_ids[position - 1]} "
                    f"(printed on the image)")
        lines.append(row)
    if visual_ids is not None:
        lines += ["", V3_ANCHOR_PROMPT]
    if incoming_context:
        lines += ["", "PREVIOUS-BATCH CONTEXT (deterministic "
                  "summary of the immediately preceding batch "
                  "only):",
                  incoming_context]
    else:
        lines += ["", "PREVIOUS-BATCH CONTEXT: none (first "
                  "batch, no earlier observations)."]
    return "\n".join(lines)


@dataclass
class FocusedValidation:
    """Outcome of structural validation for one batch reply."""

    ok: bool
    entries: tuple[dict[str, Any], ...] = ()
    events: tuple[dict[str, Any], ...] = ()
    error: str | None = None
    deduped: int = 0


def _valid_events(value: Any) -> tuple[dict[str, Any], ...] | None:
    """Validate the optional compact event list, if present."""
    if value is None:
        return ()
    if not isinstance(value, list):
        return None
    cleaned: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        description = item.get("description")
        indices = item.get("frame_indices")
        if (not isinstance(description, str)
                or not isinstance(indices, list)
                or not indices
                or any(not isinstance(index, int)
                       or isinstance(index, bool)
                       for index in indices)):
            return None
        cleaned.append({"description": description,
                        "frame_indices": list(indices)})
    return tuple(cleaned)


def validate_focused_response(
        raw_text: str,
        expected: Sequence[tuple[int, float]],
        *, visual_ids: Sequence[str] | None = None
        ) -> FocusedValidation:
    """Validate structure, coverage, ordering, and types.

    Acceptance requires: a JSON object, a ``frames`` array
    with exactly one entry per requested frame,
    ``frame_index`` values matching the request in order,
    numeric ``timestamp_seconds`` echoes, every observation
    carrying all required model-output keys, and an
    absent-or-well-formed ``events`` list. When
    ``visual_ids`` is given, each observation must also echo
    the printed identifier of its own image; the echo is an
    attribution aid, not proof the pixels were described
    correctly. Anything else is rejected with a reason;
    partial payloads never become observations.
    """
    if visual_ids is not None and len(visual_ids) != len(
            expected):
        raise FirstPassError(
            "INVALID_CONFIG",
            f"visual_ids has {len(visual_ids)} entries for "
            f"{len(expected)} expected frames.")
    required_fields = (V3_ANCHORED_OUTPUT_FIELDS
                       if visual_ids is not None
                       else V3_MODEL_OUTPUT_FIELDS)
    parsed, parse_error = parse_model_json(raw_text)
    if parsed is None:
        return FocusedValidation(
            ok=False, error=f"unparseable_json:{parse_error}")
    entries = parsed.get("frames")
    if not isinstance(entries, list):
        return FocusedValidation(
            ok=False, error="frames_not_a_list")
    deduped = 0
    if len(entries) > len(expected):
        # Live Test C returned all 50 requested frames plus
        # one byte-identical repeat of the last frame.
        # Collapse exact-duplicate repeats; any conflicting
        # repeat or non-duplicate surplus still fails below.
        seen: dict[int, dict[str, Any]] = {}
        collapsed: list[Any] = []
        for entry in entries:
            index = (entry.get("frame_index")
                     if isinstance(entry, dict) else None)
            if (isinstance(index, int)
                    and not isinstance(index, bool)
                    and index in seen):
                if entry != seen[index]:
                    return FocusedValidation(
                        ok=False,
                        error=("duplicate_frame_conflict:"
                               f"frame_{index}"))
                deduped += 1
                continue
            if (isinstance(index, int)
                    and not isinstance(index, bool)):
                seen[index] = entry
            collapsed.append(entry)
        entries = collapsed
    if len(entries) != len(expected):
        return FocusedValidation(
            ok=False,
            error=(f"frame_count_mismatch: got {len(entries)}, "
                   f"expected {len(expected)}"))
    for position, (entry, (want_index, _)) in enumerate(
            zip(entries, expected)):
        if not isinstance(entry, dict):
            return FocusedValidation(
                ok=False,
                error=f"frame_entry_not_an_object:position_{position}")
        if entry.get("frame_index") != want_index:
            return FocusedValidation(
                ok=False,
                error=(f"frame_order_mismatch:position_{position}_"
                       f"got_{entry.get('frame_index')!r}_"
                       f"expected_{want_index}"))
        stamp = entry.get("timestamp_seconds")
        if (not isinstance(stamp, (int, float))
                or isinstance(stamp, bool)):
            return FocusedValidation(
                ok=False,
                error=(f"timestamp_malformed:frame_{want_index}"))
        observation = entry.get("observation")
        if not isinstance(observation, dict):
            return FocusedValidation(
                ok=False,
                error=f"observation_not_an_object:frame_{want_index}")
        missing = [name for name in required_fields
                   if name not in observation]
        if missing:
            return FocusedValidation(
                ok=False,
                error=(f"observation_missing_keys:frame_{want_index}:"
                       f"{','.join(missing)}"))
        if visual_ids is not None:
            want_visual = visual_ids[position]
            got_visual = observation.get("visual_frame_id")
            if got_visual != want_visual:
                return FocusedValidation(
                    ok=False,
                    error=(f"visual_id_mismatch:position_{position}_"
                           f"got_{got_visual!r}_"
                           f"expected_{want_visual}"))
    events = _valid_events(parsed.get("events"))
    if events is None:
        return FocusedValidation(
            ok=False, error="events_malformed")
    return FocusedValidation(
        ok=True, entries=tuple(entries), events=events,
        deduped=deduped)


@dataclass
class FirstPassV3Config:
    """Complete focused-batch configuration (explicit)."""

    frames: list[FrameRef]
    source: SourceRef
    out_dir: Path
    model: str = MODEL_SPARK
    provider: str = "meta"
    pass_id: str = PASS_ID_V3
    group_size: int = DEFAULT_FOCUSED_GROUP_SIZE
    token_budget: int = DEFAULT_FOCUSED_TOKEN_BUDGET
    visual_anchor: bool = False
    anchor_salt: str = ANCHOR_DEFAULT_SALT
    max_groups: int | None = None
    rates_usd_per_m: tuple[float, float] | None = None
    cost_ceiling_usd: float | None = None
    strict_cost: bool = False
    billable: bool = False
    uncertain_policy: str = UNCERTAIN_RERUN
    max_attempts: int = 1
    # Total provider submits per batch when each response is
    # received but unusable. Transport without a response is
    # never replayed, regardless of this value.
    max_validation_attempts: int = 1
    # Stop the run after this many consecutive exhausted
    # batches (None disables the stop; a validated batch,
    # including a valid abstention, resets the counter).
    max_consecutive_exhausted: int | None = None
    timeout_seconds: float = 120.0
    api_key: str = ""
    client_factory: Callable[[], Any] | None = None
    backoff_base_seconds: float = 0.5
    backoff_cap_seconds: float = 30.0
    sleeper: Callable[[float], None] = field(
        default_factory=lambda: time.sleep)


def _positive_int(name: str, value: Any) -> int:
    """Reject non-integer, boolean, or non-positive values."""
    if (isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1):
        raise FirstPassError(
            "INVALID_CONFIG",
            f"{name} must be a positive int, got {value!r}.")
    return value


def _validate_v3_config(config: FirstPassV3Config) -> None:
    """Fail fast on inconsistent focused-batch configuration."""
    _positive_int("group_size", config.group_size)
    if config.group_size > MAX_IMAGES_PER_REQUEST:
        raise FirstPassError(
            "INVALID_CONFIG",
            f"group_size {config.group_size} exceeds the "
            f"per-request image maximum "
            f"{MAX_IMAGES_PER_REQUEST}.")
    _positive_int("token_budget", config.token_budget)
    _positive_int("max_attempts", config.max_attempts)
    _positive_int("max_validation_attempts",
                  config.max_validation_attempts)
    consecutive = config.max_consecutive_exhausted
    if consecutive is not None:
        _positive_int("max_consecutive_exhausted", consecutive)
    if config.max_groups is not None:
        if (isinstance(config.max_groups, bool)
                or not isinstance(config.max_groups, int)
                or config.max_groups < 0):
            raise FirstPassError(
                "INVALID_CONFIG",
                "max_groups must be a non-negative int or "
                f"None, got {config.max_groups!r}.")
    if config.uncertain_policy not in (
            UNCERTAIN_RERUN, UNCERTAIN_REFUSE):
        raise FirstPassError(
            "INVALID_CONFIG",
            "uncertain_policy must be "
            f"{UNCERTAIN_RERUN!r} or {UNCERTAIN_REFUSE!r}, "
            f"got {config.uncertain_policy!r}.")
    if config.billable:
        if config.max_attempts != 1:
            raise FirstPassError(
                "INVALID_CONFIG",
                "billable execution requires max_attempts=1 "
                "(single submit, no transport retry), "
                f"got {config.max_attempts!r}.")
        # Validation replays stay permitted: they only rerun
        # a batch whose response was fully received (billing
        # is certain) but unusable, and every replay is
        # ledger-counted against the request and cost guards.
        if config.uncertain_policy != UNCERTAIN_REFUSE:
            raise FirstPassError(
                "INVALID_CONFIG",
                "billable execution requires "
                f"uncertain_policy={UNCERTAIN_REFUSE!r}, got "
                f"{config.uncertain_policy!r}.")
        # An empty key is legal for test doubles; the
        # authorized executor always supplies a real key.
    if config.visual_anchor and not config.anchor_salt:
        raise FirstPassError(
            "INVALID_CONFIG",
            "visual_anchor requires a non-empty anchor_salt.")


class DeterministicFocusedFakeClient:
    """Deterministic offline double for focused batches.

    Returns schema-valid payloads derived from image file
    names (``frame_<index>.*``), so offline tests exercise
    the real prompt assembly, validation, checkpoint, and
    report paths without network access. ``script`` overrides
    single batches by first frame index: a string becomes the
    raw reply text and an exception raises from ``submit``.
    """

    MODEL_ID = "deterministic-focused-fake-v1"

    def __init__(
            self,
            script: dict[int, str | Exception] | None = None,
            *, visual_anchor: bool = False,
    ) -> None:
        self._script = dict(script or {})
        self._visual_anchor = visual_anchor
        self.calls: list[dict[str, Any]] = []

    @property
    def model_id(self) -> str:
        """Stable fake model identity for provenance."""
        return self.MODEL_ID

    def submit(self, prompt_text: str,
               image_paths: Sequence[Path],
               api_key: str, max_output_tokens: int,
               response_schema: dict[str, Any] | None = None,
               response_schema_name: str | None = None
               ) -> InferenceResponse:
        """Return one deterministic valid batch reply."""
        _ = (prompt_text, api_key, max_output_tokens,
             response_schema)
        paths = list(image_paths)
        self.calls.append({
            "images": len(paths),
            "schema": response_schema_name,
        })
        indices = [int(path.stem.split("_")[1])
                   for path in paths]
        override = self._script.get(indices[0])
        if isinstance(override, Exception):
            raise override
        if isinstance(override, str):
            text = override
        else:
            frames_out: list[dict[str, Any]] = []
            for index in indices:
                observation: dict[str, Any] = {
                    "scale_readable": True,
                    "scale_display_verbatim": (
                        f"{100 + index}"),
                    "scale_unit_visible": "g",
                    "scale_state": "stable",
                    "tare_action_visible": False,
                    "container_description": "steel bowl",
                    "container_changed_hint": False,
                    "primary_utensil": "spatula",
                    "primary_visible_material":
                        "white powder",
                    "action": "pouring",
                    "addition_or_removal_visible":
                        "addition",
                    "confidence": "high",
                    "ambiguity_notes": None,
                }
                if self._visual_anchor:
                    observation["visual_frame_id"] = (
                        visual_frame_id_for(index))
                frames_out.append({
                    "frame_index": index,
                    "timestamp_seconds": float(index),
                    "observation": observation})
            text = json.dumps({"frames": frames_out,
                               "events": []})
        return InferenceResponse(
            model=self.MODEL_ID, status="completed",
            output_text=text,
            usage=InferenceUsage(
                input_tokens=100 * len(paths),
                output_tokens=50 * len(paths),
                total_tokens=150 * len(paths)),
            latency_seconds=0.5, request_id="fake-focused-1")


class _V3Ledger:
    """Request/cost guard adapted from the V2 ledger.

    Same semantics: exact submit counting, rate-based cost
    estimates only when usage is fully reported, a
    pre-dispatch reserve heuristic under a strict ceiling,
    and immediate halting when strict spend cannot be
    proven. Unknown usage is counted, never priced.
    """

    def __init__(self, *, max_groups: int | None,
                 rates: tuple[float, float] | None,
                 ceiling: float | None,
                 strict: bool) -> None:
        self._max_groups = max_groups
        self._rates = rates
        self._ceiling = ceiling
        self._strict = strict
        self.requests = 0
        self.spent = 0.0
        self.unknown = 0
        self.max_single = 0.0
        self.stopped_reason: str | None = None
        self.stopped_detail: dict[str, Any] | None = None

    def seed(self, *, requests: int, spent: float,
             unknown: int, max_single: float) -> None:
        """Restore counters from adopted checkpoints."""
        self.requests = requests
        self.spent = spent
        self.unknown = unknown
        self.max_single = max_single

    def halt(self, reason: str,
             detail: dict[str, Any] | None = None) -> bool:
        """Record the first halt; later halts keep it."""
        if self.stopped_reason is None:
            self.stopped_reason = reason
            self.stopped_detail = dict(detail or {})
        return True

    def pre_dispatch(self) -> str | None:
        """Veto a submit that would breach a guard."""
        if (self._max_groups is not None
                and self.requests >= self._max_groups):
            self.halt(STOP_REQUEST_LIMIT, {
                "requests": self.requests,
                "max_requests": self._max_groups,
            })
            return "BUDGET_HALT"
        if self._ceiling is None:
            return None
        if self.spent > self._ceiling:
            self.halt(STOP_COST_CEILING, {
                "spent_usd": self.spent,
                "ceiling_usd": self._ceiling,
            })
            return "BUDGET_HALT"
        if self._strict and self._rates is not None:
            if self.spent + self.max_single > self._ceiling:
                self.halt(STOP_COST_RESERVE, {
                    "spent_usd": self.spent,
                    "reserve_usd": self.max_single,
                    "ceiling_usd": self._ceiling,
                })
                return "BUDGET_HALT"
        return None

    def note_usage(self, usage: InferenceUsage) -> bool:
        """Account one response; True requests a halt."""
        estimate = estimate_cost(
            usage.input_tokens, usage.output_tokens, self._rates)
        if estimate is None:
            self.unknown += 1
            if self._strict and self._ceiling is not None:
                self.halt(STOP_STRICT_UNKNOWN_COST, {
                    "unknown_requests": self.unknown,
                })
                return True
            return False
        self.spent += estimate
        self.max_single = max(self.max_single, estimate)
        if (self._ceiling is not None
                and self.spent > self._ceiling):
            self.halt(STOP_COST_CEILING, {
                "spent_usd": self.spent,
                "ceiling_usd": self._ceiling,
            })
            return True
        return False

    def note_transport_failure(self) -> bool:
        """Account a submit with no response; True halts."""
        self.unknown += 1
        if self._strict and self._ceiling is not None:
            self.halt(STOP_STRICT_UNKNOWN_COST, {
                "unknown_requests": self.unknown,
                "transport_without_response": True,
            })
            return True
        return False


def _usage_block(usage: InferenceUsage) -> dict[str, int | None]:
    """Persist reported token usage; never invent counts."""
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "reasoning_output_tokens": usage.reasoning_output_tokens,
    }


@dataclass
class _PreparedBatch:
    sequence: int
    group_id: str
    frames: list[FrameRef]
    image_paths: list[Path]
    image_sha256: list[str]
    image_encoding: list[str]
    visual_ids: list[str] = field(default_factory=list)
    prepared_sha256: list[str] = field(default_factory=list)


class FirstPassV3Runner:
    """Sequential focused-batch FirstPass execution."""

    def __init__(self, config: FirstPassV3Config) -> None:
        _validate_v3_config(config)
        self._config = config

    def _paths(self) -> dict[str, Path]:
        root = self._config.out_dir
        return {
            "root": root,
            "raw": root / "raw_responses",
            "obs": root / "observations",
            "checkpoints": root / "checkpoints" / "checkpoints.jsonl",
            "state": root / "checkpoints" / "run_state.json",
            "reports": root / "reports",
        }

    def _prepare_batches(self) -> list[_PreparedBatch]:
        cfg = self._config
        frames = sorted(cfg.frames,
                        key=lambda frame: frame.frame_index)
        indices = [frame.frame_index for frame in frames]
        if len(set(indices)) != len(indices):
            raise FirstPassError(
                "INVALID_CONFIG",
                "frame_index values must be unique.")
        anchor_dir = cfg.out_dir / "anchored"
        manifest: list[dict[str, Any]] = []
        prepared: list[_PreparedBatch] = []
        for sequence, chunk in enumerate(group_frames(
                frames, cfg.group_size)):
            image_paths: list[Path] = []
            image_sha256: list[str] = []
            image_encoding: list[str] = []
            visual_ids: list[str] = []
            prepared_sha256: list[str] = []
            for frame in chunk:
                payload = frame.image_path.read_bytes()
                if payload[:3] == b"\xff\xd8\xff":
                    encoding = "jpeg"
                elif payload[:4] == b"\x89PNG":
                    encoding = "png"
                else:
                    raise FirstPassError(
                        "INVALID_IMAGE",
                        f"{frame.image_path.name} is not a "
                        f"JPEG/PNG payload.")
                source_sha = hashlib.sha256(payload).hexdigest()
                if cfg.visual_anchor:
                    record = prepare_anchored_image(
                        source=frame.image_path,
                        frame_index=frame.frame_index,
                        timestamp_seconds=frame.timestamp_seconds,
                        dest_dir=anchor_dir,
                        salt=cfg.anchor_salt)
                    if not verify_anchored_image(record):
                        raise FirstPassError(
                            "ANCHOR_INTEGRITY",
                            f"prepared pixels differ from source "
                            f"for frame {frame.frame_index}.")
                    manifest.append({
                        "frame_index": record.frame_index,
                        "timestamp_seconds":
                            record.timestamp_seconds,
                        "visual_frame_id": record.visual_frame_id,
                        "check_code": record.check_code,
                        "source_name": record.source_path.name,
                        "source_sha256": record.source_sha256,
                        "source_size": list(record.source_size),
                        "prepared_name":
                            record.prepared_path.name,
                        "prepared_sha256": record.prepared_sha256,
                        "prepared_size": list(
                            record.prepared_size),
                        "photo_region_verified": True,
                    })
                    image_paths.append(record.prepared_path)
                    image_sha256.append(source_sha)
                    image_encoding.append("png")
                    visual_ids.append(record.visual_frame_id)
                    prepared_sha256.append(
                        record.prepared_sha256)
                else:
                    image_paths.append(frame.image_path)
                    image_sha256.append(source_sha)
                    image_encoding.append(encoding)
            prepared.append(_PreparedBatch(
                sequence=sequence,
                group_id=group_id_for(sequence),
                frames=chunk, image_paths=image_paths,
                image_sha256=image_sha256,
                image_encoding=image_encoding,
                visual_ids=visual_ids,
                prepared_sha256=prepared_sha256))
        if cfg.visual_anchor:
            (anchor_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8")
        return prepared

    def _check_prior_state(
            self, prepared: list[_PreparedBatch]) -> None:
        """Refuse reuse of a directory completed for other frames.

        Mirrors the V2 runner: only the frame set is
        compared here. Execution policy (model, budgets,
        guards, retry rules) is guarded by the pipeline
        fingerprint one layer up.
        """
        state_path = self._paths()["state"]
        if not state_path.is_file():
            return
        try:
            prior = json.loads(state_path.read_text(
                encoding="utf-8"))
            prior_frames = prior["frame_indices"]
        except (OSError, ValueError, KeyError) as exc:
            raise FirstPassError(
                "CHECKPOINT_MISMATCH",
                "Prior run state is unreadable; refusing "
                "resume.",
            ) from exc
        current = [frame.frame_index for group in prepared
                   for frame in group.frames]
        if prior_frames != current:
            raise FirstPassError(
                "CHECKPOINT_MISMATCH",
                "Prior run state covers a different frame "
                "set; refusing resume into this directory.",
            )

    def _load_checkpoint_index(
            self, prepared: list[_PreparedBatch]
    ) -> dict[str, dict[str, Any]]:
        """Adopt only checkpoints matching current batches."""
        cfg = self._config
        index: dict[str, dict[str, Any]] = {}
        path = self._paths()["checkpoints"]
        if not path.exists():
            return index
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return index
        wanted = {group.group_id: [frame.frame_index
                                   for frame in group.frames]
                  for group in prepared}
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            group_id = record.get("group_id")
            if group_id not in wanted:
                continue
            if record.get("frame_indices") != wanted[group_id]:
                raise FirstPassError(
                    "CHECKPOINT_MISMATCH",
                    f"refusing resume: {group_id} checkpoint "
                    f"covers different frames.")
            if record.get("status") not in (
                    "completed", "needs_human_review",
                    "budget_stopped"):
                raise FirstPassError(
                    "CHECKPOINT_CORRUPT",
                    f"refusing resume: {group_id} checkpoint "
                    f"has invalid status "
                    f"{record.get('status')!r}.")
            index[group_id] = record
        return index

    def _load_batch_observations(
            self, group_id: str) -> list[dict[str, Any]]:
        """Load validated observations for context rebuild."""
        path = (self._paths()["obs"] / f"{group_id}.json")
        try:
            payload = json.loads(path.read_text(
                encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise FirstPassError(
                "CHECKPOINT_CORRUPT",
                f"refusing resume: {group_id} observations "
                f"are unreadable ({exc}).")
        if not isinstance(payload, list):
            raise FirstPassError(
                "CHECKPOINT_CORRUPT",
                f"refusing resume: {group_id} observations "
                f"are malformed.")
        return payload

    def _seed_ledger(self, completed: dict[str, dict[str, Any]],
                     ledger: _V3Ledger) -> None:
        """Restore request/cost counters from checkpoints."""
        requests = 0
        spent = 0.0
        unknown = 0
        max_single = 0.0
        for record in completed.values():
            block = record.get("budget") or {}
            requests += int(block.get("requests_submitted") or 0)
            estimate = block.get("cost_usd_estimated")
            if isinstance(estimate, (int, float)):
                spent += float(estimate)
                max_single = max(max_single, float(estimate))
            else:
                unknown += 1
        ledger.seed(requests=requests, spent=spent,
                    unknown=unknown, max_single=max_single)

    def _scan_uncertain(
            self, completed: dict[str, dict[str, Any]]) -> None:
        """Refuse resume when a submit may have billed."""
        cfg = self._config
        if cfg.uncertain_policy != UNCERTAIN_REFUSE:
            return
        markers = sorted(
            self._paths()["checkpoints"].parent.glob(
                f"{MARKER_PREFIX}*{MARKER_SUFFIX}"))
        for marker in markers:
            try:
                payload = json.loads(marker.read_text(
                    encoding="utf-8"))
            except (OSError, ValueError):
                continue
            group_id = payload.get("group_id")
            if (group_id not in completed
                    and isinstance(group_id, str)):
                raise FirstPassError(
                    "UNCERTAIN_REQUESTS",
                    "refusing resume: "
                    f"{marker.name} has no checkpoint; the "
                    "provider may have billed that batch.")

    def _dispatch(self, client: Any, prompt_text: str,
                  group: _PreparedBatch, ledger: _V3Ledger
                  ) -> tuple[InferenceResponse | None,
                             str | None,
                             list[dict[str, Any]]]:
        """Submit with ledger guards and transport retries."""
        cfg = self._config
        attempts: list[dict[str, Any]] = []
        for step in range(cfg.max_attempts):
            if step:
                wait = min(cfg.backoff_base_seconds
                           * (2.0 ** (step - 1)),
                           cfg.backoff_cap_seconds)
                cfg.sleeper(wait)
                attempts[-1]["slept_seconds"] = wait
            veto = ledger.pre_dispatch()
            if veto is not None:
                return None, veto, attempts
            ledger.requests += 1
            try:
                result = client.submit(
                    prompt_text, group.image_paths,
                    cfg.api_key, cfg.token_budget,
                    response_schema=(
                        FOCUSED_ANCHORED_JSON_SCHEMA
                        if cfg.visual_anchor
                        else FOCUSED_JSON_SCHEMA),
                    response_schema_name=(
                        FOCUSED_ANCHORED_SCHEMA_NAME
                        if cfg.visual_anchor
                        else FOCUSED_SCHEMA_NAME))
            except InferenceError as exc:
                if exc.code in RETRYABLE_CODES:
                    attempts.append({
                        "attempt": len(attempts) + 1,
                        "code": exc.code,
                        "message": str(exc),
                    })
                    if step + 1 < cfg.max_attempts:
                        continue
                if exc.code in IMAGE_LIMIT_CODES:
                    raise FirstPassError(
                        "PROVIDER_IMAGE_LIMIT",
                        f"{exc.code}: provider or client "
                        f"rejected the {len(group.image_paths)}-"
                        f"image request ({exc}).")
                ledger.note_transport_failure()
                return None, exc.code, attempts
            attempts.append({"attempt": len(attempts) + 1,
                             "code": "ok"})
            return result, None, attempts
        raise AssertionError("unreachable dispatch state")

    def _run_batch(
            self, client: Any, group: _PreparedBatch,
            total: int, incoming_context: str | None,
            context_kind: str, ledger: _V3Ledger,
            paths: dict[str, Path]
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Execute one batch with validation replays."""
        cfg = self._config
        indices = [frame.frame_index for frame in group.frames]
        expected = [(frame.frame_index,
                     frame.timestamp_seconds)
                    for frame in group.frames]
        visual_ids = (list(group.visual_ids)
                      if cfg.visual_anchor else None)
        prompt_text = build_focused_prompt(
            frames=group.frames,
            group_sequence=group.sequence,
            total_groups=total,
            incoming_context=incoming_context,
            visual_ids=visual_ids)
        marker_path = (paths["checkpoints"].parent
                       / f"{MARKER_PREFIX}{group.group_id}"
                       f"{MARKER_SUFFIX}")
        marker: dict[str, Any] = {
            "group_id": group.group_id,
            "sequence": group.sequence,
            "frame_indices": indices,
            "image_sha256": group.image_sha256,
            "attempt_started_at": datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
        }
        if cfg.visual_anchor:
            marker["visual_frame_ids"] = list(group.visual_ids)
            marker["prepared_sha256"] = list(
                group.prepared_sha256)
        marker_path.write_text(json.dumps(marker),
                               encoding="utf-8")
        spent_before = ledger.spent
        unknown_before = ledger.unknown
        max_before = ledger.max_single
        halted = False
        status = "needs_human_review"
        terminal: str | None = None
        failure_reason: str | None = None
        validation: FocusedValidation | None = None
        raw_text = ""
        result: InferenceResponse | None = None
        error: str | None = None
        attempts: list[dict[str, Any]] = []
        for validation_try in range(cfg.max_validation_attempts):
            if validation_try:
                wait = min(
                    cfg.backoff_base_seconds * (
                        2.0 ** (validation_try - 1)),
                    cfg.backoff_cap_seconds)
                cfg.sleeper(wait)
                attempts[-1]["slept_seconds"] = wait
            validation = None
            raw_text = ""
            terminal = None
            attempt_result, attempt_error, dispatch_notes = (
                self._dispatch(client, prompt_text, group, ledger))
            for note in dispatch_notes:
                note["attempt"] = len(attempts) + 1
                attempts.append(note)
            result, error = attempt_result, attempt_error
            if error == "BUDGET_HALT":
                halted = True
                status = "budget_stopped"
                failure_reason = (
                    ledger.stopped_reason or "budget_halt")
                break
            if error is not None or result is None:
                status = "needs_human_review"
                terminal = TERMINAL_FAILED
                failure_reason = f"transport:{error}"
                if ledger.stopped_reason is not None:
                    halted = True
                break
            attempts[-1]["latency_seconds"] = (
                result.latency_seconds)
            attempts[-1]["request_id"] = result.request_id
            attempts[-1]["input_tokens"] = (
                result.usage.input_tokens)
            attempts[-1]["output_tokens"] = (
                result.usage.output_tokens)
            attempts[-1]["reasoning_output_tokens"] = (
                result.usage.reasoning_output_tokens)
            raw_text = result.output_text
            if (result.status == "completed"
                    and result.output_text.strip()):
                validation = validate_focused_response(
                    raw_text, expected, visual_ids=visual_ids)
                if validation.ok:
                    status = "completed"
                    terminal = TERMINAL_VALID
                    attempts[-1]["validation"] = "ok"
                    attempts[-1]["deduped_entries"] = (
                        validation.deduped)
                else:
                    status = "needs_human_review"
                    terminal = TERMINAL_INVALID_JSON
                    failure_reason = (
                        f"validation:{validation.error}")
                    attempts[-1]["validation"] = failure_reason
            else:
                status = "needs_human_review"
                if is_token_limit_reason(
                        result.incomplete_reason):
                    terminal = TERMINAL_EXHAUSTED
                    failure_reason = (
                        "token_budget_exhausted:"
                        "no_escalation_in_v3")
                elif not raw_text.strip():
                    terminal = TERMINAL_EMPTY
                    failure_reason = "empty_output"
                else:
                    terminal = TERMINAL_UNSUPPORTED
                    failure_reason = (
                        f"status:{result.status}:"
                        f"{result.incomplete_reason}")
                if result.status != "completed" and raw_text.strip():
                    failure_reason = (
                        f"status:{result.status}:"
                        f"{result.incomplete_reason}")
                attempts[-1]["validation"] = failure_reason
            halted = ledger.note_usage(result.usage) or halted
            if halted:
                if status == "completed":
                    status = "budget_stopped"
                    failure_reason = (
                        ledger.stopped_reason or "budget_halt")
                break
            if validation is not None and validation.ok:
                break
        by_index = {frame.frame_index: frame
                    for frame in group.frames}
        observations: list[dict[str, Any]] = []
        if validation is not None and validation.ok:
            manifest_ref = {
                "manifest_sha256": cfg.source.sha256,
                "source_name": cfg.source.name,
            }
            for position, entry in enumerate(
                    validation.entries):
                frame = by_index[entry["frame_index"]]
                observations.append({
                    "frame_index": frame.frame_index,
                    "image_path": frame.image_path.name,
                    "timestamp_seconds":
                        frame.timestamp_seconds,
                    "source_timestamp_seconds":
                        frame.source_timestamp_seconds,
                    "requested_timestamp_seconds":
                        frame.requested_timestamp_seconds,
                    "timestamp_basis": TIMESTAMP_BASIS,
                    "manifest_ref": manifest_ref,
                    "batch_id": group.group_id,
                    "pass_id": cfg.pass_id,
                    "call": {
                        "provider": cfg.provider,
                        "model": cfg.model,
                        "prompt_sha256": prompt_hash(
                            prompt_text),
                        "group_id": group.group_id,
                        "group_sequence": group.sequence,
                        "image_position": position,
                        "group_size": len(group.frames),
                        "incoming_context_kind": context_kind,
                        "max_output_tokens": cfg.token_budget,
                        "store": None,
                        "timeout_seconds": cfg.timeout_seconds,
                        "input_tokens":
                            result.usage.input_tokens
                            if result is not None else None,
                        "output_tokens":
                            result.usage.output_tokens
                            if result is not None else None,
                        "latency_seconds": (
                            result.latency_seconds
                            if result is not None else None),
                        "request_id": (
                            result.request_id
                            if result is not None else None),
                        "attempts": len(attempts),
                        "response_status": (
                            result.status
                            if result is not None else None),
                        "incomplete_reason": (
                            result.incomplete_reason
                            if result is not None else None),
                        "terminal_state": terminal,
                    },
                    "model_fields": dict(
                        entry["observation"]),
                    "model_timestamp_seconds": entry.get(
                        "timestamp_seconds"),
                    "parse_error": None,
                    "raw_text_excerpt": json.dumps(entry)[:500],
                    "response_status": (
                        result.status
                        if result is not None else None),
                    "incomplete_reason": (
                        result.incomplete_reason
                        if result is not None else None),
                    "terminal_state": terminal,
                    "image_ref": {
                        "sha256": group.image_sha256[position],
                        "encoding":
                            group.image_encoding[position],
                        **({"prepared_sha256":
                            group.prepared_sha256[position]}
                           if cfg.visual_anchor else {}),
                    },
                })
        raw_record: dict[str, Any] = {
            "group_id": group.group_id,
            "sequence": group.sequence,
            "frame_indices": indices,
            "visual_anchor": cfg.visual_anchor,
            "image_order": [path.name
                            for path in group.image_paths],
            "prompt_sha256": prompt_hash(prompt_text),
            "incoming_context_kind": context_kind,
            "incoming_context_chars": (
                len(incoming_context)
                if incoming_context else 0),
            "incoming_context_token_estimate":
                estimate_context_tokens(incoming_context or ""),
            "status": status,
            "terminal_state": terminal,
            "failure_reason": failure_reason,
            "attempts": attempts,
            "output_text": raw_text,
            "usage": (_usage_block(result.usage)
                      if result is not None else {
                          "input_tokens": None,
                          "output_tokens": None,
                          "total_tokens": None,
                          "cached_input_tokens": None,
                          "reasoning_output_tokens": None,
                      }),
            "latency_seconds": (
                result.latency_seconds
                if result is not None else None),
            "request_id": (
                result.request_id
                if result is not None else None),
            "model": cfg.model,
            "response_status": (
                result.status if result is not None else None),
            "incomplete_reason": (
                result.incomplete_reason
                if result is not None else None),
        }
        if validation is not None and validation.ok:
            raw_record["events"] = [
                dict(event) for event in validation.events]
        raw_path = paths["raw"] / f"{group.group_id}.json"
        raw_path.write_text(json.dumps(raw_record, indent=2),
                            encoding="utf-8")
        obs_path = paths["obs"] / f"{group.group_id}.json"
        obs_path.write_text(json.dumps(observations, indent=2),
                            encoding="utf-8")
        estimate = estimate_cost(
            raw_record["usage"]["input_tokens"],
            raw_record["usage"]["output_tokens"], cfg.rates_usd_per_m)
        record = {
            "group_id": group.group_id,
            "sequence": group.sequence,
            "frame_indices": indices,
            "status": status,
            "terminal_state": terminal,
            "failure_reason": failure_reason,
            "raw_path": str(raw_path),
            "obs_path": str(obs_path),
            "input_tokens": raw_record["usage"]["input_tokens"],
            "output_tokens":
                raw_record["usage"]["output_tokens"],
            "attempts": attempts,
            "frames_observed": len(observations),
            "frames_expected": len(indices),
            "budget": {
                "requests_submitted": len(attempts),
                "cost_usd_estimated": estimate,
                "cost_unknown": estimate is None,
                "spent_before_usd": spent_before,
                "spent_after_usd": ledger.spent,
                "unknown_before": unknown_before,
                "unknown_after": ledger.unknown,
                "max_single_before_usd": max_before,
                "max_single_after_usd": ledger.max_single,
                "stopped_reason": ledger.stopped_reason,
                "stopped_detail": ledger.stopped_detail,
            },
        }
        with paths["checkpoints"].open(
                "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        summary = {
            "group_id": group.group_id,
            "status": status,
            "terminal_state": terminal,
            "submits": len(attempts),
            "frames_observed": len(observations),
            "frames_expected": len(indices),
            "input_tokens":
                raw_record["usage"]["input_tokens"] or 0,
            "output_tokens":
                raw_record["usage"]["output_tokens"] or 0,
            "reasoning_output_tokens": sum(
                note.get("reasoning_output_tokens") or 0
                for note in attempts),
            "latency_seconds": raw_record["latency_seconds"],
            "request_id": raw_record["request_id"],
            "halted": halted,
            "exhausted": (status != "completed" and not halted),
        }
        return summary, observations

    def _previous_entries(
            self, prev_group_id: str | None,
            prev_observations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Shape previous observations for the builder."""
        if prev_group_id is None:
            return []
        entries = []
        for item in prev_observations:
            if not isinstance(item, dict):
                continue
            observation = item.get("model_fields")
            if (isinstance(item.get("frame_index"), int)
                    and isinstance(observation, dict)):
                entries.append({
                    "frame_index": item["frame_index"],
                    "observation": observation,
                })
        return entries

    def run(self) -> dict[str, Any]:
        """Execute all batches sequentially with resume."""
        cfg = self._config
        _validate_v3_config(cfg)
        if cfg.client_factory is None:
            raise FirstPassError(
                "INVALID_CONFIG",
                "client_factory is required.")
        prepared = self._prepare_batches()
        if not prepared:
            raise FirstPassError(
                "INVALID_CONFIG", "no frames to analyze.")
        total = len(prepared)
        paths = self._paths()
        for directory in (paths["raw"], paths["obs"],
                          paths["reports"],
                          paths["checkpoints"].parent):
            directory.mkdir(parents=True, exist_ok=True)
        self._check_prior_state(prepared)
        completed = self._load_checkpoint_index(prepared)
        self._scan_uncertain(completed)
        ledger = _V3Ledger(
            max_groups=cfg.max_groups, rates=cfg.rates_usd_per_m,
            ceiling=cfg.cost_ceiling_usd, strict=cfg.strict_cost)
        self._seed_ledger(completed, ledger)
        client = cfg.client_factory()
        summaries: list[dict[str, Any]] = []
        stats: dict[str, Any] = {
            "requests_sent": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
            "frames_total": sum(len(group.frames)
                                for group in prepared),
            "frames_observed": 0,
            "frames_failed": 0,
            "groups_total": total,
            "groups_valid": 0,
            "groups_review": 0,
            "groups_exhausted": 0,
            "groups_resumed": 0,
            "frames_resumed": 0,
            "terminal_states": {},
        }
        start_sequence = 0
        prev_group_id: str | None = None
        prev_observations: list[dict[str, Any]] = []
        for group in prepared:
            record = completed.get(group.group_id)
            if record is None:
                break
            observed = record.get("frames_observed") or 0
            current = [frame.frame_index
                       for frame in group.frames]
            summaries.append({
                "group_id": group.group_id,
                "status": record.get("status"),
                "terminal_state": record.get("terminal_state"),
                "submits": 0,
                "frames_observed": observed,
                "frames_expected": len(current),
                "input_tokens":
                    record.get("input_tokens") or 0,
                "output_tokens":
                    record.get("output_tokens") or 0,
                "reasoning_output_tokens": 0,
                "latency_seconds": None,
                "request_id": None,
                "halted": record.get("status") == "budget_stopped",
                "exhausted": False,
                "resumed": True,
            })
            stats["groups_resumed"] += 1
            stats["frames_resumed"] += observed
            stats["frames_observed"] += observed
            stats["frames_failed"] += (
                len(current) - observed)
            prev_group_id = group.group_id
            prev_observations = self._load_batch_observations(
                group.group_id)
            start_sequence = group.sequence + 1
            if record.get("status") == "budget_stopped":
                break
        consecutive_exhausted = 0
        if cfg.max_consecutive_exhausted is not None:
            for prior in prepared[:start_sequence]:
                prior_record = completed.get(prior.group_id)
                if prior_record is None:
                    break
                if prior_record.get("status") == "completed":
                    consecutive_exhausted = 0
                elif (prior_record.get("status")
                        == "needs_human_review"):
                    consecutive_exhausted += 1
                else:
                    break
        pending = prepared[start_sequence:]
        if (cfg.max_consecutive_exhausted is not None
                and consecutive_exhausted
                >= cfg.max_consecutive_exhausted):
            ledger.halt(STOP_CONSECUTIVE_EXHAUSTED, {
                "consecutive_exhausted": consecutive_exhausted,
                "resumed_stop": True,
            })
            pending = []
        for group in pending:
            if prev_group_id is None:
                incoming_context = None
                context_kind = "none"
            else:
                incoming_context = (
                    build_previous_batch_context(
                        prev_group_id=prev_group_id,
                        entries=self._previous_entries(
                            prev_group_id, prev_observations)))
                context_kind = "previous_batch"
            summary, observations = self._run_batch(
                client, group, total, incoming_context,
                context_kind, ledger, paths)
            summaries.append(summary)
            stats["requests_sent"] += summary["submits"]
            stats["input_tokens"] += summary["input_tokens"]
            stats["output_tokens"] += summary["output_tokens"]
            stats["reasoning_output_tokens"] += summary[
                "reasoning_output_tokens"]
            stats["frames_observed"] += summary["frames_observed"]
            stats["frames_failed"] += (
                summary["frames_expected"]
                - summary["frames_observed"])
            if summary["status"] == "completed":
                stats["groups_valid"] += 1
                consecutive_exhausted = 0
            else:
                stats["groups_review"] += 1
                if summary.get("exhausted"):
                    stats["groups_exhausted"] += 1
                    consecutive_exhausted += 1
            terminal = summary.get("terminal_state")
            if terminal:
                stats["terminal_states"][terminal] = (
                    stats["terminal_states"].get(terminal, 0) + 1)
            prev_group_id = group.group_id
            prev_observations = observations
            if (cfg.max_consecutive_exhausted is not None
                    and consecutive_exhausted
                    >= cfg.max_consecutive_exhausted):
                ledger.halt(STOP_CONSECUTIVE_EXHAUSTED, {
                    "consecutive_exhausted":
                        consecutive_exhausted,
                    "group_id": summary["group_id"],
                    "groups_attempted": len(summaries),
                })
                break
            if summary["halted"]:
                break
        attempted_ids = {summary["group_id"]
                         for summary in summaries
                         if summary["submits"]
                         or summary.get("resumed")}
        remaining: list[int] = []
        for group in prepared:
            if group.group_id not in attempted_ids:
                remaining.extend(frame.frame_index
                                 for frame in group.frames)
        return self._write_report(
            summaries, stats, ledger, paths, remaining,
            prepared)

    def _write_report(
            self, summaries: list[dict[str, Any]],
            stats: dict[str, Any], ledger: _V3Ledger,
            paths: dict[str, Path],
            remaining: list[int],
            prepared: list[_PreparedBatch]) -> dict[str, Any]:
        """Persist and return the run report."""
        cfg = self._config
        rates = cfg.rates_usd_per_m
        report = {
            "experiment": PASS_ID_V3,
            "pass_id": cfg.pass_id,
            "model": cfg.model,
            "provider": cfg.provider,
            "source_name": cfg.source.name,
            "source_sha256": cfg.source.sha256,
            "frame_indices": [frame.frame_index
                              for group in prepared
                              for frame in group.frames],
            "group_size": cfg.group_size,
            "token_budget_per_group": cfg.token_budget,
            "groups_total": stats["groups_total"],
            "groups_valid": stats["groups_valid"],
            "groups_review": stats["groups_review"],
            "groups_exhausted": stats["groups_exhausted"],
            "groups_resumed": stats["groups_resumed"],
            "frames_total": stats["frames_total"],
            "frames_observed": stats["frames_observed"],
            "frames_failed": stats["frames_failed"],
            "frames_resumed": stats["frames_resumed"],
            "requests_sent": stats["requests_sent"],
            "requests_total": ledger.requests,
            "input_tokens": stats["input_tokens"],
            "output_tokens": stats["output_tokens"],
            "reasoning_output_tokens":
                stats["reasoning_output_tokens"],
            "total_tokens": (stats["input_tokens"]
                             + stats["output_tokens"]),
            "stopped_reason": ledger.stopped_reason,
            "stopped_detail": ledger.stopped_detail,
            "terminal_states": stats["terminal_states"],
            "groups": summaries,
            "budget": {
                "max_groups": cfg.max_groups,
                "cost_ceiling_usd": cfg.cost_ceiling_usd,
                "strict_cost": cfg.strict_cost,
                "cost_usd_spent": ledger.spent,
                "cost_unknown_requests": ledger.unknown,
                "cost_usd_estimated": (
                    estimate_cost(stats["input_tokens"],
                                  stats["output_tokens"], rates)
                    if rates is not None else None),
                "stopped_reason": ledger.stopped_reason,
                "remaining_frame_indices": remaining,
            },
            "cost_usd_estimated": (
                estimate_cost(stats["input_tokens"],
                              stats["output_tokens"], rates)
                if rates is not None else None),
        }
        payload = json.dumps(report, indent=2)
        paths["state"].write_text(payload, encoding="utf-8")
        (paths["reports"] / "run_report.json").write_text(
            payload, encoding="utf-8")
        return report
