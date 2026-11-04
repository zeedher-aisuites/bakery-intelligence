"""Phase 2C end-to-end orchestrator (staged, resumable).

Runs the full lifecycle per clip, then fuses the session:
ingest, transcribe, global analysis, targeted local windows,
measurement frames, speech claims, reconciliation, recipe
drafting, deterministic + LLM criticism, persistence. Every
stage is idempotent: completed artifacts are reused, failures
are recorded with explicit status, and reruns resume from the
first incomplete stage. One retry per billable call; budget
violations stop immediately without retry.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from bakery_intelligence.phase2c import agents, media, prompts
from bakery_intelligence.phase2c.clients import (
    BudgetLedger,
    MetaResponsesClient,
    ProviderError,
    VoiceTranscribeClient,
)
from bakery_intelligence.phase2c.manifest import (
    SessionManifest,
    build_session,
)
from bakery_intelligence.phase2c.session import (
    fuse_session,
    persist_session,
)

MAX_WINDOWS_PER_CLIP = 3
MAX_WINDOW_SECONDS = 60.0
MAX_FRAMES_PER_CLIP = 10
MIN_FRAME_GAP = 5.0

TARGET_KEYWORDS = ("scale", "weigh", "tare", "label", "package",
                   "packet", "bag", "add", "pour", "unknown",
                   "unclear", "illegible", "unreadable")


def select_windows(operations: list[dict[str, Any]],
                   duration: float) -> list[tuple[float, float]]:
    """Pick targeted re-analysis windows from global operations.

    Deterministic heuristic: operations mentioning measurement,
    packaging, additions, or uncertainty score higher; windows
    are capped in count and length.
    """
    scored: list[tuple[int, float, float]] = []
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        start = _num(operation.get("start_seconds"))
        end = _num(operation.get("end_seconds"))
        if start is None or end is None or end <= start:
            continue
        text = " ".join(str(operation.get(key) or "")
                        for key in ("action_type",
                                    "action_description",
                                    "evidence", "uncertainty"))
        score = sum(1 for word in TARGET_KEYWORDS
                    if word in text.lower())
        scored.append((score, start, min(end, duration)))
    scored.sort(key=lambda item: (-item[0], item[1]))
    windows: list[tuple[float, float]] = []
    for _, start, end in scored:
        if len(windows) >= MAX_WINDOWS_PER_CLIP:
            break
        if end - start > MAX_WINDOW_SECONDS:
            mid = (start + end) / 2
            start = max(0.0, mid - MAX_WINDOW_SECONDS / 2)
            end = min(mid + MAX_WINDOW_SECONDS / 2, duration)
        if any(not (end <= wstart or start >= wend)
               for wstart, wend in windows):
            continue
        windows.append((round(start, 3), round(end, 3)))
    return windows


def select_frames(operations: list[dict[str, Any]],
                  measurements: list[dict[str, Any]],
                  duration: float) -> list[float]:
    """Pick frame timestamps biased to scale observations."""
    stamps: list[float] = []
    for measurement in measurements:
        stamp = _num((measurement or {}).get("timestamp_seconds"))
        if stamp is not None and 0 <= stamp <= duration:
            stamps.append(stamp)
    for operation in operations:
        text = " ".join(
            str((operation or {}).get(key) or "")
            for key in ("action_type", "action_description",
                        "evidence")).lower()
        if "scale" not in text and "weigh" not in text:
            continue
        start = _num(operation.get("start_seconds"))
        end = _num(operation.get("end_seconds"))
        if start is not None and end is not None:
            stamps.append(round((start + end) / 2, 3))
    chosen: list[float] = []
    for stamp in sorted(set(stamps)):
        if len(chosen) >= MAX_FRAMES_PER_CLIP:
            break
        if all(abs(stamp - kept) >= MIN_FRAME_GAP
               for kept in chosen):
            chosen.append(stamp)
    return chosen


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def deterministic_critic(
        recipe: dict[str, Any],
        draft: dict[str, Any]) -> dict[str, Any]:
    """Check recipe claims against cited evidence (no LLM)."""
    known_steps = {s.get("group_id")
                   for s in draft.get("steps") or []}
    known_ings = {i.get("group_id")
                  for i in draft.get("ingredients") or []}
    known_speech = {t.get("speech_id")
                    for t in draft.get("speech") or []}
    session = draft.get("session") or {}
    known_clips = {c.get("clip_id")
                   for c in session.get("clips") or []}
    assessment = draft.get("quantity_assessment") or {}
    known_clusters = {c.get("cluster_id")
                      for c in assessment.get("stable_clusters")
                      or []}
    known = (known_steps | known_ings | known_speech
             | known_clips | known_clusters)
    findings: list[dict[str, Any]] = []
    for section in ("ingredients", "steps"):
        for item in recipe.get(section) or []:
            refs = item.get("evidence_refs") or []
            if not refs:
                findings.append({
                    "claim_ref": item.get("name",
                                          item.get("text",
                                                   "?"))[:80],
                    "check": "provenance_present",
                    "result": "fail",
                    "explanation": "No evidence_refs cited.",
                })
                continue
            for ref in refs:
                findings.append({
                    "claim_ref": str(ref),
                    "check": "provenance_present",
                    "result": "pass" if ref in known else "fail",
                    "explanation": "Cited node present."
                    if ref in known else
                    "Cited node absent from draft.",
                })
                if ref not in known:
                    break
    fails = sum(1 for item in findings
                if item["result"] == "fail")
    return {
        "verdict": "unsupported" if fails else "needs_review",
        "findings": findings,
        "deterministic": True,
    }


class Orchestrator:
    """Staged multi-clip execution with resume support."""

    def __init__(self, root: Path,
                 ledger: BudgetLedger | None = None,
                 meta: MetaResponsesClient | None = None,
                 asr: VoiceTranscribeClient | None = None):
        self.root = root
        self.run_dir = root / "phase2c" / "bakery_demo_v1"
        self.ledger = ledger or BudgetLedger.load(
            root / "phase2c" / "ledger.json")
        self.meta = meta or MetaResponsesClient()
        self.asr = asr or VoiceTranscribeClient()
        self.status_path = self.run_dir / "status.json"

    def status(self) -> dict[str, Any]:
        if self.status_path.is_file():
            try:
                payload = media.read_json(self.status_path)
                if isinstance(payload, dict):
                    return payload
            except ValueError:
                pass
        return {"stages": {}}

    def _mark(self, stage: str, state: str,
              detail: str | None = None) -> None:
        payload = self.status()
        payload["stages"][stage] = {
            "state": state, "detail": detail,
            "updated_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        media.write_json(self.status_path, payload)

    def _done(self, stage: str) -> bool:
        return self.status().get("stages", {}).get(
            stage, {}).get("state") == "ok"

    def run_clip(self, manifest: SessionManifest,
                 clip_id: str) -> dict[str, Any]:
        """Execute all per-clip stages with resume."""
        clip = next(c for c in manifest.clips
                    if c.clip_id == clip_id)
        work = self.run_dir / "clips" / clip_id
        source = Path(clip.path)

        def attempt(stage: str, func: Callable[[], Any],
                    agent: str) -> agents.AgentResult:
            if self._done(f"{clip_id}/{stage}"):
                stored = media.read_json(
                    work / f"{clip_id}.{stage}.json")
                return agents.AgentResult(
                    agent=agent, clip_id=clip_id, status="ok",
                    payload=_resumed_payload(stage, stored))
            last_error: str | None = None
            for trial in range(2):
                try:
                    result = func()
                except Exception as exc:  # noqa: BLE001 - staged
                    last_error = str(exc)[:300]
                    result = None
                if result is not None and result.status == "ok":
                    self._mark(f"{clip_id}/{stage}", "ok")
                    return result
                if result is not None:
                    last_error = (result.error or "unknown")[:300]
                if trial == 0 and _transient(last_error or ""):
                    time.sleep(5)
                    continue
                break
            self._mark(f"{clip_id}/{stage}", "failed", last_error)
            raise ProviderError(
                f"{clip_id}/{stage} failed: {last_error}")

        wav = work / f"{clip_id}.wav"
        if not self._done(f"{clip_id}/wav"):
            media.extract_wav(source, wav)
            media.write_json(work / f"{clip_id}.wav.json",
                             {"path": str(wav)})
            self._mark(f"{clip_id}/wav", "ok")

        result_c = attempt(
            "agent-c",
            lambda: agents.run_agent_c_transcribe(
                self.asr, self.ledger, clip_id, wav,
                clip.duration_seconds, work),
            "C")
        result_a = attempt(
            "agent-a",
            lambda: agents.run_agent_a_global(
                self.meta, self.ledger, clip_id, source, work),
            "A")
        ops_a = (result_a.payload or {}).get("operations") or []
        if not self._done(f"{clip_id}/windows"):
            windows = select_windows(ops_a, clip.duration_seconds)
            media.write_json(work / f"{clip_id}.windows.json",
                             {"windows": windows})
            self._mark(f"{clip_id}/windows", "ok")
        else:
            windows = media.read_json(
                work / f"{clip_id}.windows.json")["windows"]
        attempt(
            "agent-b",
            lambda: agents.run_agent_b_local(
                self.meta, self.ledger, clip_id,
                [(w[0], w[1]) for w in windows],
                lambda s, d, p: media.cut_subclip(source, s, d, p),
                work),
            "B")
        meas_a = (result_a.payload or {}).get("measurements") or []
        if not self._done(f"{clip_id}/frames"):
            stamps = select_frames(ops_a, meas_a,
                                   clip.duration_seconds)
            media.write_json(work / f"{clip_id}.frames.json",
                             {"stamps": stamps})
            self._mark(f"{clip_id}/frames", "ok")
        else:
            stamps = media.read_json(
                work / f"{clip_id}.frames.json")["stamps"]
        frames = [(work / f"{clip_id}.f{i}.jpg", stamp)
                  for i, stamp in enumerate(stamps)]
        for path, stamp in frames:
            if not path.is_file():
                media.grab_frame(source, stamp, path)
        agent_c_turns = (result_c.payload or {}).get("turns") or []
        if frames:
            attempt(
                "agent-d",
                lambda: agents.run_agent_d_measure(
                    self.meta, self.ledger, clip_id, frames,
                    work),
                "D")
        else:
            media.write_json(work / f"{clip_id}.agent-d.json",
                             {"parsed": {"readings": []},
                              "skipped": "no frames selected"})
            self._mark(f"{clip_id}/agent-d", "ok")
        if agent_c_turns:
            attempt(
                "agent-c-claims",
                lambda: agents.run_text_agent(
                    self.meta, self.ledger,
                    prompts.SPEECH_ANALYST, "C", clip_id,
                    {"clip_id": clip_id, "turns": agent_c_turns},
                    work, f"{clip_id}.agent-c-claims.json",
                    0.10),
                "C")
        else:
            media.write_json(
                work / f"{clip_id}.agent-c-claims.json",
                {"parsed": {"claims": []},
                 "skipped": "no transcript turns"})
            self._mark(f"{clip_id}/agent-c-claims", "ok")
        return {"agent_a": _saved(work, clip_id, "agent-a"),
                "agent_b": (_saved(work, clip_id, "agent-b") or {}
                            ).get("segments") or [],
                "agent_c_turns": (result_c.payload or {}).get(
                    "turns") or [],
                "agent_c_claims": _saved(
                    work, clip_id, "agent-c-claims"),
                "agent_d": _saved(work, clip_id, "agent-d")}

    def run_session(self, manifest: SessionManifest,
                    clips: dict[str, dict[str, Any]]) -> Path:
        """Fuse clips, draft the recipe, criticize, persist."""
        fused = fuse_session(manifest, clips)
        draft = fused["draft"]
        recipe: dict[str, Any] | None = None
        if not self._done("session/agent-f"):
            result_f = agents.run_text_agent(
                self.meta, self.ledger, prompts.RECIPE_SPECIALIST,
                "F", "session",
                {"session": draft.get("session"),
                 "steps": draft.get("steps"),
                 "ingredients": draft.get("ingredients"),
                 "quantity_assessment": draft.get(
                     "quantity_assessment"),
                 "conflicts": draft.get("conflicts"),
                 "speech": (draft.get("speech") or [])[:40]},
                self.run_dir, "session.agent-f.json", 0.30)
            if result_f.status != "ok":
                self._mark("session/agent-f", "failed",
                           result_f.error)
                raise ProviderError(
                    f"Agent F failed: {result_f.error}")
            recipe = result_f.payload
            self._mark("session/agent-f", "ok")
        else:
            recipe = (media.read_json(
                self.run_dir / "session.agent-f.json") or {}
                ).get("parsed")
        checks = deterministic_critic(recipe or {}, draft)
        critic_llm: dict[str, Any] | None = None
        if not self._done("session/agent-g"):
            result_g = agents.run_text_agent(
                self.meta, self.ledger, prompts.EVIDENCE_CRITIC,
                "G", "session",
                {"recipe": recipe,
                 "deterministic_findings": checks,
                 "steps": draft.get("steps"),
                 "ingredients": draft.get("ingredients")},
                self.run_dir, "session.agent-g.json", 0.20)
            if result_g.status == "ok":
                critic_llm = result_g.payload
                self._mark("session/agent-g", "ok")
            else:
                self._mark("session/agent-g", "failed",
                           result_g.error)
        else:
            critic_llm = (media.read_json(
                self.run_dir / "session.agent-g.json") or {}
                ).get("parsed")
        fused = fuse_session(
            manifest, clips,
            recipe_proposal={
                "parsed": recipe,
                "prompt_version":
                    prompts.RECIPE_SPECIALIST.version,
                "status": "ai_suggested",
            },
            critic_verdict={
                "deterministic": checks,
                "llm": critic_llm,
                "prompt_version":
                    prompts.EVIDENCE_CRITIC.version
                if critic_llm else None,
                "status": "ai_suggested",
            })
        paths = persist_session(
            fused, self.root / "fusion", "bakery_demo_v1")
        media.write_json(self.run_dir / "execution.json", {
            "prompt_versions": prompts.prompt_versions(),
            "ledger_calls": self.ledger.calls,
            "spent_usd": round(self.ledger.spent_usd, 4),
            "artifacts": paths,
        })
        self._mark("session/persist", "ok")
        return Path(paths["output_dir"])


def _transient(error: str) -> bool:
    """Retry only probable server-side flakiness (once).

    Client errors (400s, schema violations) and budget stops
    never retry: repeating them cannot succeed.
    """
    lowered = error.lower()
    if "budget" in lowered or "schema violation" in lowered:
        return False
    return any(token in lowered for token in
               ("504", "503", "502", "529", "timeout",
                "timed out", "server_error", "rate limit",
                "429", "connection", "temporarily"))


def _resumed_payload(stage: str,
                     stored: dict[str, Any]) -> dict[str, Any]:
    """Rebuild an agent payload from its persisted artifact."""
    if stage == "agent-c":
        return {"turns": stored.get("turns") or []}
    if stage == "agent-b":
        return {"segments": stored.get("segments") or []}
    parsed = stored.get("parsed")
    return parsed if isinstance(parsed, dict) else {}


def _saved(work: Path, clip_id: str,
           stage: str) -> dict[str, Any]:
    path = work / f"{clip_id}.{stage}.json"
    if not path.is_file():
        return {}
    try:
        payload = media.read_json(path)
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}
