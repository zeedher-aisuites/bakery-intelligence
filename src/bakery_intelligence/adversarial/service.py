"""Recipe-review service: evidence -> debate -> durable draft.

The bootstrap debate uses the deterministic evidence proposer
and a deterministic fake challenger. It is simulated and
labeled as such; no live model provider is contacted.
"""

from __future__ import annotations

from typing import Any

from bakery_intelligence.adversarial.evidence import (
    build_evidence_package,
)
from bakery_intelligence.adversarial.providers import (
    DeterministicFakeChallenger,
    EvidenceProposer,
)
from bakery_intelligence.adversarial.resolution import debate_claims
from bakery_intelligence.adversarial.store import RecipeReviewStore
from bakery_intelligence.api.jobs import JobRecord
from bakery_intelligence.api.storage import LocalStorageBackend

SIMULATION_NOTE = (
    "Simulated adversarial exchange: deterministic evidence "
    "proposer and fake challenger only. No live model was called."
)


class _FixedObjectionsChallenger:
    """Replay one recorded objection round without a provider call."""

    provider_id = "live-replay"
    simulated = False

    def __init__(self, objections: list[dict[str, Any]]) -> None:
        self._objections = objections
        self.calls = 0

    def critique(self, claims: list[dict[str, Any]],
                 package: dict[str, Any],
                 ) -> list[dict[str, Any]]:
        self.calls += 1
        wanted = {str(claim.get("claim_id")) for claim in claims}
        return [dict(item) for item in self._objections
                if str(item.get("claim_id")) in wanted]


def _challenger_failure_category(exc: BaseException) -> str:
    """Extract the challenger failure category from the error text."""
    text = str(exc)
    marker = "failure_category="
    index = text.find(marker)
    if index != -1:
        category = text[index + len(marker):].split(")")[0]
        category = category.split()[0].strip(",;'\"") if category else ""
        if category:
            return category
    name = type(exc).__name__
    if name in ("RuntimeError", "ConnectionError", "TimeoutError"):
        return "transport"
    if "Timeout" in name or "Connection" in name or "API" in name:
        return "transport"
    return "unknown"


class RecipeReviewService:
    """Owns recipe-review creation and reads for one storage root."""

    def __init__(self, storage: LocalStorageBackend) -> None:
        self._storage = storage
        self._store = RecipeReviewStore(storage.base_dir)

    @property
    def store(self) -> RecipeReviewStore:
        return self._store

    def get_or_create(self, record: JobRecord) -> dict[str, Any]:
        """Return the durable recipe review, creating it once."""
        existing = self._store.load(record.run_id)
        if existing is not None:
            return self._normalized(existing)
        package = build_evidence_package(self._storage, record)
        proposer = EvidenceProposer()
        claims = proposer.propose(package)
        result = debate_claims(
            package, claims, DeterministicFakeChallenger())
        draft_id = (
            (package.get("provenance") or {}).get("draft_id")
            or f"draft_{record.run_id}")
        return self._store.create(
            run_id=record.run_id,
            draft_id=str(draft_id),
            claims=result.claims,
            verification=result.verification.model_dump(mode="json"),
            simulation_note=SIMULATION_NOTE,
        )

    @staticmethod
    def _normalized(state: dict[str, Any]) -> dict[str, Any]:
        """Read-time defaults for legacy states (never rewrites disk)."""
        import copy

        from bakery_intelligence.adversarial.store import (
            normalize_ingredients,
            normalize_steps,
        )
        view = copy.deepcopy(state)
        view["ingredients"] = normalize_ingredients(
            view.get("ingredients", []))
        view["steps"] = normalize_steps(view.get("steps", []))
        return view

    def save_corrections(self, record: JobRecord,
                         ingredients: list[dict[str, Any]],
                         steps: list[dict[str, Any]],
                         reviewer: str) -> dict[str, Any]:
        """Validate temporal links against the package, then save."""
        from bakery_intelligence.adversarial.store import (
            RecipeReviewError,
            normalize_steps,
        )
        package = self.evidence(record)
        known_frames = {frame["frame_index"]
                        for frame in package.get("frames", [])}
        duration = package.get("video", {}).get("duration_seconds")
        for step in normalize_steps(steps):
            for ref in step.get("frame_refs", []):
                if ref.get("frame_index") not in known_frames:
                    raise RecipeReviewError(
                        f"Step references missing frame "
                        f"{ref.get('frame_index')!r}.")
            for value in (step.get("start_seconds"),
                          step.get("end_seconds")):
                if (isinstance(value, (int, float))
                        and isinstance(duration, (int, float))
                        and value > duration):
                    raise RecipeReviewError(
                        "Step timestamp exceeds the video duration.")
        return self._store.save_corrections(
            record.run_id, ingredients, steps, reviewer)

    def culinary_payload(self, record: JobRecord) -> dict[str, Any]:
        """Culinary observations plus deterministic events, if any.

        Empty for analyses never enriched: the review queue and the
        consolidated package keep their historical shape.
        """
        from bakery_intelligence.adversarial.culinary import (
            CulinaryStore,
            build_culinary_events,
        )
        stored = CulinaryStore(
            self._storage.base_dir).load(record.run_id)
        observations = [
            stored["observations"][key]
            for key in sorted(stored.get("observations", {}),
                              key=int)
        ]
        scale_by_frame = {
            frame["frame_index"]:
            (frame.get("measurement") or {}).get("value")
            for frame in self.evidence(record).get("frames", [])
        }
        events = build_culinary_events(
            record.run_id, observations, scale_by_frame)
        return {
            "observations": observations,
            "events": events,
            "failures": stored.get("failures", {}),
            "usage": stored.get("usage", []),
        }

    def observation_queue(self, record: JobRecord) -> dict[str, Any]:
        """Exception queue for this run (accepted frames hidden)."""
        from bakery_intelligence.adversarial.observations import (
            ObservationReviewStore,
            build_review_tasks,
        )
        package = self.evidence(record)
        stored = ObservationReviewStore(
            self._storage.base_dir).load(record.run_id)
        return build_review_tasks(
            package, stored["corrections"],
            stored.get("classifications"),
            self.culinary_payload(record))

    def save_observation_correction(self, record: JobRecord,
                                    task_id: str, action: str,
                                    human_value: Any, reviewer: str,
                                    ) -> dict[str, Any]:
        from bakery_intelligence.adversarial.observations import (
            ObservationReviewStore,
        )
        package = self.evidence(record)
        return ObservationReviewStore(
            self._storage.base_dir).save_correction(
                record.run_id, task_id, package, action,
                human_value, reviewer,
                culinary=self.culinary_payload(record))

    def undo_observation_correction(self, record: JobRecord,
                                    task_id: str, reviewer: str,
                                    ) -> dict[str, Any]:
        from bakery_intelligence.adversarial.observations import (
            ObservationReviewStore,
        )
        package = self.evidence(record)
        return ObservationReviewStore(
            self._storage.base_dir).undo_correction(
                record.run_id, task_id, package, reviewer,
                culinary=self.culinary_payload(record))

    def save_frame_classification(self, record: JobRecord,
                                  frame_index: int, action: str,
                                  human_value: Any, reviewer: str,
                                  ) -> dict[str, Any]:
        from bakery_intelligence.adversarial.observations import (
            ObservationReviewStore as _ObservationReviewStore,
        )
        package = self.evidence(record)
        return _ObservationReviewStore(
            self._storage.base_dir).save_classification(
                record.run_id, package, frame_index, action,
                human_value, reviewer)

    def consolidated(self, record: JobRecord) -> dict[str, Any]:
        from bakery_intelligence.adversarial.consolidation import (
            consolidate,
        )
        from bakery_intelligence.adversarial.observations import (
            ObservationReviewStore,
        )
        package = self.evidence(record)
        stored = ObservationReviewStore(
            self._storage.base_dir).load(record.run_id)
        merged = consolidate(package, stored["corrections"])
        culinary = self.culinary_payload(record)
        merged["culinary_observations"] = culinary["observations"]
        merged["culinary_events"] = culinary["events"]
        merged["culinary_failures"] = culinary["failures"]
        merged["culinary_unknowns"] = [
            {"code": item.get("code"), "message": item.get("message"),
             "event_id": event.get("event_id"),
             "frame_refs": event.get("frame_refs")}
            for event in culinary["events"]
            for item in event.get("unresolved_questions", [])
            if isinstance(item, dict)
        ]
        import hashlib
        import json as _json
        merged["culinary_hash"] = hashlib.sha256(_json.dumps(
            {"observations": culinary["observations"],
             "events": culinary["events"]},
            sort_keys=True, default=str).encode()).hexdigest()
        return merged

    def temporal_events(self, record: JobRecord) -> dict[str, Any]:
        from bakery_intelligence.adversarial.temporal import (
            reconstruct_events,
        )
        return reconstruct_events(self.consolidated(record))

    def readiness(self, record: JobRecord) -> dict[str, Any]:
        queue = self.observation_queue(record)
        open_tasks = [task for task in queue["tasks"]
                      if task["status"] == "open"]
        consolidated = self.consolidated(record)
        blocking = list(queue["counts"]["blocking_errors"])
        from bakery_intelligence.adversarial.live import live_available
        live_ok, live_blockers = live_available()
        return {
            "ready": not blocking,
            "blocking_errors": blocking,
            "open_tasks": len(open_tasks),
            "deferred": queue["counts"]["deferred"],
            "explicit_unknowns": len(consolidated["unknowns"]),
            "consolidated_hash": consolidated["content_hash"],
            "live_generation": False,
            "simulation_available": True,
            "live_available": live_ok,
            "live_blockers": live_blockers,
            "cost_note": (
                "Live cost cannot be estimated before execution; "
                "it depends on the tokens each provider returns. "
                "Historical first-pass costs are already incurred "
                "and are never counted again."),
        }

    def culinary_status(self, record: JobRecord) -> dict[str, Any]:
        """Enrichment progress for this run (no provider calls)."""
        payload = self.culinary_payload(record)
        package = self.evidence(record)
        enrichable = [
            frame["frame_index"]
            for frame in package.get("frames", [])
            if self._frame_image_path(
                record, frame["frame_index"]) is not None
        ]
        done = {item["frame_index"]
                for item in payload["observations"]}
        done |= {int(key) for key in payload.get("failures", {})}
        return {
            "run_id": record.run_id,
            "model": "muse-spark-1.3",
            "enriched": len(payload["observations"]),
            "failed": len(payload.get("failures", {})),
            "completed": len(done),
            "total": len(enrichable),
            "done": len(done) >= len(enrichable),
            "events": payload["events"],
            "observations": payload["observations"],
        }

    def enrich_culinary(self, record: JobRecord,
                        max_frames: int = 5,
                        confirmed: bool = False,
                        max_output_tokens: int = 16384,
                        frame_indices: list[int] | None = None,
                        submit: Any | None = None) -> dict[str, Any]:
        """Enrich a batch of retained frames (explicit only).

        Requires explicit confirmation plus the live flag and a Meta
        key. Each frame is submitted at most once; failures are
        terminal and never retried. Reuses retained frame images;
        the video is never re-extracted. An optional explicit frame
        list restricts the batch (completed frames are still
        skipped); without it, the earliest unprocessed frames run.
        """
        from bakery_intelligence.adversarial.culinary import (
            CulinaryStore,
            enrichment_api_key,
            enrichment_available,
            enrich_next_batch,
        )
        from bakery_intelligence.adversarial.store import (
            RecipeReviewError,
        )
        if not confirmed:
            raise RecipeReviewError(
                "Culinary enrichment requires explicit confirmation.")
        ok, blockers = enrichment_available()
        if not ok:
            raise RecipeReviewError("; ".join(blockers))
        wanted = (set(frame_indices)
                  if frame_indices is not None else None)
        package = self.evidence(record)
        frames: list[dict[str, Any]] = []
        image_paths: list[Any] = []
        for frame in package.get("frames", []):
            if (wanted is not None
                    and frame["frame_index"] not in wanted):
                continue
            path = self._frame_image_path(
                record, frame["frame_index"])
            if path is None:
                continue
            frames.append({
                "frame_index": frame["frame_index"],
                "timestamp_seconds": frame.get("timestamp_seconds"),
            })
            image_paths.append(path)
        if submit is None:
            from bakery_intelligence.inference.meta_spark import (
                MetaSparkClient,
            )
            client = MetaSparkClient()
            submit = client.submit
        return enrich_next_batch(
            run_id=record.run_id,
            frames=frames,
            image_paths=image_paths,
            submit=submit,
            store=CulinaryStore(self._storage.base_dir),
            api_key=enrichment_api_key(),
            max_output_tokens=max_output_tokens,
            max_frames=max_frames)

    def requeue_culinary_frames(
            self, record: JobRecord,
            frame_indices: list[int],
            reason: str, reviewer: str) -> dict[str, Any]:
        """Authorize reprocessing of terminal-failed frames.

        Records the authorization and preserves the original
        failures in history; submits nothing and retries nothing
        automatically.
        """
        from bakery_intelligence.adversarial.culinary import (
            CulinaryStore,
        )
        return CulinaryStore(
            self._storage.base_dir).request_reprocessing(
                record.run_id, frame_indices, reason, reviewer)

    def generate_recipe(
            self, record: JobRecord,
            request: Any | None = None,
            proposer: Any | None = None,
            challenger: Any | None = None) -> dict[str, Any]:
        """Attach a generation to the recipe review.

        Simulation is the default and performs no provider call.
        Live mode uses the Meta proposer and the Baseten
        challenger only with an explicit request, confirmation,
        and configured credentials; any provider failure is
        reported without falling back to simulated output.
        Idempotent per (consolidated hash, mode, rounds,
        provider identities, output budget, culinary hash):
        repeated identical requests reuse the stored generation
        without new provider calls.
        """
        from bakery_intelligence.adversarial.claims import (
            evidence_ref,
            make_claim,
        )
        from bakery_intelligence.adversarial.live import (
            GenerateRecipeRequest,
            check_live_or_raise,
        )
        request = request or GenerateRecipeRequest()
        check_live_or_raise(request)
        consolidated = self.consolidated(record)
        state = self.get_or_create(record)
        temporal = self.temporal_events(record)
        if request.mode == "live":
            return self._generate_live(
                record, state, consolidated, temporal, request,
                proposer=proposer, challenger=challenger)
        existing = state.get("generation") or {}
        if existing.get("consolidated_hash") == (
                consolidated["content_hash"]) and existing.get(
                "mode", "simulation") == "simulation":
            return state
        claims = []
        for event in temporal["events"]:
            if event["event_type"] != "ingredient_addition_candidate":
                continue
            refs = [
                evidence_ref(
                    record.run_id, frame_index, None,
                    next((frame.get("timestamp_seconds")
                          for frame in consolidated["frames"]
                          if frame["frame_index"] == frame_index),
                         None))
                for frame_index in event["supporting_evidence"]
            ]
            net = event["net_quantity"]
            claims.append(make_claim(
                kind="quantity",
                title=(
                    f"Candidate addition {event['start_seconds']}s"
                    f"–{event['end_seconds']}s: readings "
                    f"{event['initial_reading']} to "
                    f"{event['final_reading']}."),
                quantity_value=net["value"],
                quantity_unit=net["unit"],
                derivation=net["derivation"],
                evidence_refs=refs,
                uncertainty=event["uncertainty"],
            ))
        result = debate_claims(
            consolidated, claims, DeterministicFakeChallenger())
        stored = self._store.load(record.run_id) or state
        stored["generation"] = {
            "simulated": True,
            "mode": "simulation",
            "label": "Generate Recipe (Simulation)",
            "consolidated_hash": consolidated["content_hash"],
            "events": temporal["events"],
            "claims": result.claims,
            "rounds_used": result.rounds_used,
            "termination_reason": result.termination_reason,
            "requested_rounds": request.max_rounds,
        }
        stored["claims"] = result.claims or stored.get("claims", [])
        self._store.save(stored)
        return self.get_or_create(record)

    def _generate_live(self, record: JobRecord,
                       state: dict[str, Any],
                       consolidated: dict[str, Any],
                       temporal: dict[str, Any], request: Any,
                       proposer: Any | None,
                       challenger: Any | None) -> dict[str, Any]:
        """One bounded live adversarial execution, fully recorded."""
        from bakery_intelligence.adversarial.claims import make_claim
        from bakery_intelligence.adversarial.live import (
            BASETEN_AUTHORIZED_MODEL,
            REQUESTED_REASONING,
            SPARK_MODEL,
            SPARK_RATES_USD_PER_M,
            GenerateRecipeRequest,
            MetaSparkLiveProposer,
            BasetenLiveChallenger,
            cost_entry,
            live_credentials,
        )
        from bakery_intelligence.adversarial.resolution import MAX_ROUNDS
        request = request or GenerateRecipeRequest()
        rounds = min(max(1, int(request.max_rounds)), MAX_ROUNDS)
        existing = state.get("generation") or {}
        key = (consolidated["content_hash"], "live", rounds,
               SPARK_MODEL, BASETEN_AUTHORIZED_MODEL,
               int(request.max_output_tokens),
               consolidated.get("culinary_hash"))
        if existing.get("key") == list(key):
            return state
        credentials = live_credentials()
        requested_tokens = int(request.max_output_tokens)
        proposer = proposer or MetaSparkLiveProposer(
            max_output_tokens=requested_tokens)
        challenger = challenger or BasetenLiveChallenger()
        from bakery_intelligence.adversarial.live import (
            choose_evidence_frames,
        )
        action = consolidated.get("action_evidence") or {}
        action_frames = set(action.get("addition_frames") or []) | set(
            action.get("tare_frames") or [])
        representative = choose_evidence_frames(
            [frame for frame in consolidated["frames"]
             if frame.get("timestamp_seconds") is not None],
            action_frames, 8)
        def _describe(frame: dict[str, Any]) -> str:
            effective = frame.get("effective") or {}
            value = effective.get("value", (frame.get("measurement")
                                            or {}).get("value"))
            corrected = ("human-corrected "
                         if effective.get("value") is not None else "")
            reading = (f"{corrected}scale {value}"
                       if value is not None else "no usable reading")
            return (f"frame {frame['frame_index']} @ "
                    f"{frame.get('timestamp_seconds')}s ({reading})")

        def _describe_culinary(event: dict[str, Any]) -> str:
            names = ", ".join(
                str(item.get("name") or "unnamed")
                for item in event.get("ingredient_candidates", [])
                if isinstance(item, dict)) or "no ingredient established"
            frames = event.get("frame_refs", [])
            span = (f"frames {frames[0]}-{frames[-1]}"
                    if frames else "no frames")
            return (
                f"culinary {event.get('operation')} ({span}; "
                f"candidates: {names}; "
                f"uncertainty: {event.get('uncertainty')})")

        from bakery_intelligence.adversarial.culinary import (
            describe_for_prompt,
        )
        culinary_events = consolidated.get("culinary_events") or []
        culinary_observations = (
            consolidated.get("culinary_observations") or [])
        evidence_text = (
            f"run {record.run_id}; video "
            f"{consolidated.get('video', {}).get('duration_seconds')}s; "
            f"visual moments: "
            + ", ".join(_describe(frame) for frame in representative)
            + f"; unknowns: {consolidated.get('unknowns')}"
            + f"; culinary unknowns: "
            f"{consolidated.get('culinary_unknowns')}"
            + ("; culinary events: "
               + " | ".join(_describe_culinary(event)
                            for event in culinary_events)
               if culinary_events else "; culinary events: none")
            + ("; enriched frames: "
               + " | ".join(describe_for_prompt(item)
                            for item in culinary_observations)
               if culinary_observations
               else "; enriched frames: none"))
        image_paths = [
            self._frame_image_path(record, frame["frame_index"])
            for frame in representative
        ]
        image_paths = [path for path in image_paths if path is not None]
        resumed = self._load_resumable_proposal(
            record, state, list(key))
        if resumed is not None:
            proposal, proposer_usage, costs = resumed
        else:
            try:
                proposal, proposer_usage = proposer.propose(
                    evidence_text, image_paths, credentials["meta"])
            except Exception as exc:
                from bakery_intelligence.api.jobs import utc_now_iso
                failure: dict[str, Any] = {
                    "at": utc_now_iso(),
                    "operation": "recipe-proposal",
                    "provider": getattr(
                        proposer, "provider_id", "meta"),
                    "model": getattr(
                        proposer, "last_model", SPARK_MODEL),
                    "request_id": getattr(
                        proposer, "last_request_id", None),
                    "usage": getattr(
                        proposer, "last_usage", None),
                    "status": getattr(
                        proposer, "last_status", None),
                    "incomplete_reason": getattr(
                        proposer, "last_incomplete_reason", None),
                    "output_length": getattr(
                        proposer, "last_output_length", None),
                    "max_output_tokens": requested_tokens,
                    "images": len(image_paths),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                try:
                    failed = self._store.load(record.run_id) or state
                    attempts = list(failed.get("live_attempts") or [])
                    attempts.append(failure)
                    failed["live_attempts"] = attempts
                    self._store.save(failed)
                except Exception:
                    pass
                raise
            costs = [cost_entry(
                provider="meta", model=SPARK_MODEL,
                operation="recipe-proposal",
                usage=proposer_usage, rates=SPARK_RATES_USD_PER_M,
                rates_source="operator-configured-spark-rates",
                request_id=getattr(proposer, "last_request_id", None),
                status="ok")]
            self._save_proposal_checkpoint(
                record, state, list(key), proposal, proposer_usage,
                costs, requested_tokens, len(image_paths))
        claims: list[dict[str, Any]] = []
        for index, step in enumerate(proposal.steps):
            refs = [
                {"run_id": record.run_id, "frame_index": frame_index,
                 "measurement_id": None, "timestamp_seconds": None}
                for frame_index in (step.get("frame_refs") or [])
            ]
            claims.append(make_claim(
                kind="step", title=str(step.get("description", "")),
                step_index=index, evidence_refs=refs,
                uncertainty="Live proposal; challenger review pending."))
        evidence_summary = {
            "run_id": record.run_id,
            "events": temporal["events"],
            "unknowns": consolidated.get("unknowns"),
            "proposal_ingredients": proposal.ingredients,
        }
        objections: list[dict[str, Any]] = []
        failed_round = 1
        try:
            for round_number in range(1, rounds + 1):
                failed_round = round_number
                round_objections, usage = challenger.critique(
                    [dict(claim, status="uncertain") for claim in claims
                     if claim["status"] != "supported"],
                    evidence_summary, credentials["baseten"])
                costs.append(cost_entry(
                    provider="baseten",
                    model=BASETEN_AUTHORIZED_MODEL,
                    operation=f"challenger-round-{round_number}",
                    usage=usage, rates=None,
                    rates_source="unavailable-no-baseten-pricing",
                    request_id=getattr(
                        challenger, "last_request_id", None),
                    status="ok"))
                objections = round_objections
                if not round_objections:
                    break
        except Exception as exc:
            from bakery_intelligence.api.jobs import utc_now_iso
            failure: dict[str, Any] = {
                "at": utc_now_iso(),
                "operation": f"challenger-round-{failed_round}",
                "provider": getattr(
                    challenger, "provider_id", "baseten"),
                "model": BASETEN_AUTHORIZED_MODEL,
                "request_id": getattr(
                    challenger, "last_request_id", None),
                "usage": getattr(challenger, "last_usage", None),
                "finish_reason": getattr(
                    challenger, "last_finish_reason", None),
                "content_length": getattr(
                    challenger, "last_content_length", None),
                "failure_category": _challenger_failure_category(exc),
                "round": failed_round,
                "claims": len(claims),
                "error": f"{type(exc).__name__}: {exc}",
            }
            try:
                failed = self._store.load(record.run_id) or state
                attempts = list(failed.get("live_attempts") or [])
                attempts.append(failure)
                failed["live_attempts"] = attempts
                self._store.save(failed)
            except Exception:
                pass
            raise
        result = debate_claims(
            consolidated, claims,
            _FixedObjectionsChallenger(objections))
        stored = self._store.load(record.run_id) or state
        stored.pop("live_proposal", None)
        stored["generation"] = {
            "simulated": False,
            "mode": "live",
            "key": list(key),
            "label": "Live adversarial generation",
            "consolidated_hash": consolidated["content_hash"],
            "events": temporal["events"],
            "proposal": proposal.model_dump(mode="json"),
            "claims": result.claims,
            "rounds_used": rounds,
            "requested_rounds": rounds,
            "termination_reason": result.termination_reason,
            "requested_reasoning": REQUESTED_REASONING,
            "providers": [
                {"role": "proposer", "provider": "meta",
                 "model": SPARK_MODEL},
                {"role": "challenger", "provider": "baseten",
                 "model": BASETEN_AUTHORIZED_MODEL,
                 "input_modality": "text-evidence-only"},
            ],
            "costs": costs,
            "estimated_total_cost_usd": round(sum(
                item["estimated_cost_usd"] or 0.0 for item in costs), 4),
        }
        self._store.save(stored)
        return self.get_or_create(record)

    def _load_resumable_proposal(self, record: JobRecord,
                                   state: dict[str, Any],
                                   key: list[Any],
                                   ) -> tuple[Any, dict[str, Any],
                                              list[dict[str, Any]]] | None:
        """Reuse a saved proposal without repeating billable inference."""
        saved = self._store.load(record.run_id) or state
        checkpoint = saved.get("live_proposal")
        if not isinstance(checkpoint, dict):
            return None
        if checkpoint.get("key") != key:
            return None
        proposal_dict = checkpoint.get("proposal")
        if not isinstance(proposal_dict, dict):
            return None
        from bakery_intelligence.adversarial.live import RecipeProposal
        try:
            proposal = RecipeProposal(**proposal_dict)
        except (ValueError, TypeError):
            return None
        proposer_usage = checkpoint.get("proposer_usage")
        if not isinstance(proposer_usage, dict):
            return None
        costs = checkpoint.get("costs")
        if not isinstance(costs, list) or not costs:
            return None
        return proposal, proposer_usage, list(costs)

    def _save_proposal_checkpoint(
            self, record: JobRecord, state: dict[str, Any],
            key: list[Any], proposal: Any,
            proposer_usage: dict[str, Any],
            costs: list[dict[str, Any]], max_output_tokens: int,
            images: int) -> None:
        """Persist a validated proposal before the challenger runs."""
        if hasattr(proposal, "model_dump"):
            proposal_dict = proposal.model_dump(mode="json")
        else:
            proposal_dict = {
                "recipe_name": getattr(proposal, "recipe_name", None),
                "ingredients": list(
                    getattr(proposal, "ingredients", [])),
                "steps": list(getattr(proposal, "steps", [])),
                "unresolved_questions": list(getattr(
                    proposal, "unresolved_questions", [])),
            }
        from bakery_intelligence.api.jobs import utc_now_iso
        checkpoint: dict[str, Any] = {
            "key": key,
            "at": utc_now_iso(),
            "proposal": proposal_dict,
            "proposer_usage": dict(proposer_usage),
            "costs": list(costs),
            "max_output_tokens": max_output_tokens,
            "images": images,
        }
        saved = self._store.load(record.run_id) or state
        saved["live_proposal"] = checkpoint
        self._store.save(saved)

    def _frame_image_path(self, record: JobRecord,
                          frame_index: int) -> Any | None:
        path = (self._storage.base_dir / "runs" / record.run_id
                / "work" / "frames" / f"frame_{frame_index:08d}.jpg")
        return path if path.is_file() else None

    def evidence(self, record: JobRecord) -> dict[str, Any]:
        """Return the normalized evidence package (read-only)."""
        return build_evidence_package(self._storage, record)
