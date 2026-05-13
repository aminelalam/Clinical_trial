"""LangGraph node functions for the agent.

Each node receives the current ``AgentStateDict`` and returns a small dict
with ONLY the slots it modifies. LangGraph merges those updates into the
shared state using the per-channel reducers declared in
``models/agent_state.py``. Two consequences:

- Nodes never mutate the input state in place. This is what makes parallel
  branches (e.g. ``retrieve_lexical`` and ``retrieve_dense``) safe.
- Counter increments must use additive reducers (already the case for
  ``llm_calls``, ``retrieval_attempts``, etc.). A node that wants to add 1
  to ``llm_calls`` returns ``{"llm_calls": 1}`` — NOT the previous value
  plus one — and the reducer accumulates.
"""

from __future__ import annotations

import time
from typing import Any

from ..config import get_settings
from ..eligibility.aggregator import aggregate_to_trial_eval
from ..eligibility.irrelevance import (
    apply_multisignal_irrelevance_policy,
    retrieval_prior_for_candidate,
)
from ..logging import logger
from ..models.agent_state import AgentStateDict, NodeTiming
from ..models.criterion import CriterionType
from ..models.eligibility import EligibilityLabel
from ..models.ranking import JudgedTrial, RankedTrial
from ..nlp.patient_extractor import PatientExtractor
from ..nlp.question_generator import QuestionGenerator
from ..ranking.critique import SelfCritic
from ..ranking.criterion_evidence import apply_criterion_evidence_adjustment
from ..ranking.fill import fill_hard_excluded_to_top_k
from ..ranking.llm_judge import LLMJudge
from ..ranking.scorer import score_trial
from ..retrieval.entity_negation import apply_entity_negation_rerank
from ..dossier.builder import DossierBuilder
from .candidate_selection import select_viable_candidates
from .planner import SearchPlanner
from .tools import AgentTools


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _timed(name: str):
    """Decorator that wraps a node coroutine with timing/log instrumentation.

    The wrapped node is responsible for returning its full update dict; we
    splice in a NodeTiming entry under ``node_timings`` so the reducer
    accumulates one record per call.
    """

    def deco(fn):
        async def inner(state: AgentStateDict) -> dict[str, Any]:
            t0 = time.time()
            logger.info(f"node_start {name}")
            try:
                update = await fn(state)
            finally:
                seconds = time.time() - t0
                logger.info(f"node_end {name} seconds={seconds:.3f}")
            update = dict(update or {})
            update.setdefault("node_timings", [])
            update["node_timings"] = list(update["node_timings"]) + [
                NodeTiming(node=name, seconds=seconds)
            ]
            return update

        inner.__name__ = fn.__name__
        return inner

    return deco


def _err(state: AgentStateDict, msg: str) -> dict[str, Any]:
    logger.warning(msg)
    return {"errors": [msg]}


# ----------------------------------------------------------------------
# Stage 1 — patient understanding
# ----------------------------------------------------------------------


def make_parse_patient_node(extractor: PatientExtractor):
    @_timed("parse_patient")
    async def parse_patient_node(state: AgentStateDict) -> dict[str, Any]:
        profile = await extractor.extract(state["run_id"], state["patient_raw"])
        return {"patient_profile": profile, "llm_calls": 1}

    return parse_patient_node


def make_normalize_mesh_node(mesh_normalizer: Any | None):
    @_timed("normalize_mesh")
    async def normalize_mesh_node(state: AgentStateDict) -> dict[str, Any]:
        profile = state.get("patient_profile")
        if mesh_normalizer is None or profile is None:
            return {}
        terms = []
        if profile.primary_diagnosis:
            terms.append(profile.primary_diagnosis)
        terms.extend(profile.secondary_diagnoses)
        for c in profile.comorbidities:
            terms.append(c.name)

        concepts: list[dict] = []
        for term in terms:
            for c in mesh_normalizer.normalize(term):
                concepts.append(
                    {
                        "concept_id": c.concept_id,
                        "name": c.name,
                        "synonyms": c.synonyms[:8],
                    }
                )
        # PatientProfile is a Pydantic model; mutate via model_copy to keep
        # the rest of the pipeline reading the same object identity.
        new_profile = profile.model_copy(update={"mesh_concepts": concepts})
        return {"patient_profile": new_profile}

    return normalize_mesh_node


# ----------------------------------------------------------------------
# Stage 2 — planning
# ----------------------------------------------------------------------


def make_plan_search_node(planner: SearchPlanner):
    @_timed("plan_search")
    async def plan_search_node(state: AgentStateDict) -> dict[str, Any]:
        profile = state.get("patient_profile")
        if profile is None:
            return _err(state, "plan_search called without patient_profile")
        s = get_settings()
        mesh_summary = ", ".join(
            f"{c['name']} ({c['concept_id']})"
            for c in (profile.mesh_concepts or [])[:6]
        )
        plan = await planner.plan(
            profile,
            mesh_concepts_summary=mesh_summary,
            relax=bool(state.get("needs_re_retrieval", False)),
            mode=s.runner.mode,
        )
        return {
            "search_plan": plan,
            "retrieval_attempts": 1,  # additive reducer accumulates
            "needs_re_retrieval": False,
            "relaxed_plan_used": bool(plan.relax_optional_filters),
            "llm_calls": 1,
        }

    return plan_search_node


# ----------------------------------------------------------------------
# Stage 3 — retrieval
# ----------------------------------------------------------------------


def make_retrieve_lexical_node(tools: AgentTools):
    @_timed("retrieve_lexical")
    async def retrieve_lexical_node(state: AgentStateDict) -> dict[str, Any]:
        plan = state.get("search_plan")
        if plan is None:
            return {}
        profile = state.get("patient_profile")
        s = get_settings()
        candidates = tools.search_trials_lexical(
            plan.primary_disease_query,
            mesh_terms=plan.expansion_terms,
            top_k=s.retrieval.bm25_top_k,
            plan=plan,
            patient=profile,
        )
        return {"bm25_candidates": candidates}

    return retrieve_lexical_node


def make_retrieve_dense_node(tools: AgentTools):
    @_timed("retrieve_dense")
    async def retrieve_dense_node(state: AgentStateDict) -> dict[str, Any]:
        profile = state.get("patient_profile")
        if profile is None:
            return {}
        s = get_settings()
        if not s.runner.use_dense_retrieval:
            return {"dense_candidates": []}
        candidates = await tools.search_trials_dense(
            profile.raw_text,
            top_k=s.retrieval.dense_top_k,
            use_hyde=s.runner.use_hyde,
        )
        return {"dense_candidates": candidates}

    return retrieve_dense_node


def make_fuse_node():
    @_timed("fuse_rrf")
    async def fuse_node(state: AgentStateDict) -> dict[str, Any]:
        s = get_settings()
        weighted_runs = [
            (state.get("bm25_candidates") or [], s.retrieval.bm25_rrf_weight),
            (state.get("dense_candidates") or [], s.retrieval.dense_rrf_weight),
        ]
        weighted_runs = [(r, w) for r, w in weighted_runs if r]
        if not weighted_runs:
            return {}
        if s.retrieval.bm25_mode == "fielded" and len(weighted_runs) == 1:
            keep = max(s.retrieval.fused_top_k, s.retrieval.rerank_top_k)
            fused = [
                c.model_copy(update={"source": "fused", "rank": i + 1})
                for i, c in enumerate(weighted_runs[0][0][:keep])
            ]
            return {"fused_candidates": fused}
        from ..retrieval.hybrid import reciprocal_rank_fusion

        runs = [r for r, _w in weighted_runs]
        weights = [w for _r, w in weighted_runs]
        fused = reciprocal_rank_fusion(
            runs,
            weights=weights,
            top_k=max(s.retrieval.fused_top_k, s.retrieval.rerank_top_k),
        )
        return {"fused_candidates": fused}

    return fuse_node


def make_rerank_pointwise_node(tools: AgentTools):
    @_timed("rerank_pointwise")
    async def rerank_pointwise_node(state: AgentStateDict) -> dict[str, Any]:
        profile = state.get("patient_profile")
        fused = state.get("fused_candidates") or []
        if profile is None or not fused:
            return {}
        s = get_settings()
        ranked = tools.rerank(profile.raw_text, fused, top_k=s.retrieval.rerank_top_k)
        return {"reranked_candidates": ranked}

    return rerank_pointwise_node


def make_rerank_listwise_node(tools: AgentTools):
    @_timed("rerank_listwise")
    async def rerank_listwise_node(state: AgentStateDict) -> dict[str, Any]:
        s = get_settings()
        reranked = state.get("reranked_candidates") or []
        if not s.runner.use_listwise or not reranked:
            return {"listwise_candidates": list(reranked[: s.retrieval.listwise_top_k])}
        profile = state.get("patient_profile")
        if profile is None:
            return {}
        listwise = tools.listwise_rerank(
            profile.raw_text, reranked, top_k=s.retrieval.listwise_top_k
        )
        return {"listwise_candidates": listwise}

    return rerank_listwise_node


def make_apply_filters_node(tools: AgentTools):
    @_timed("apply_hard_filters")
    async def apply_filters_node(state: AgentStateDict) -> dict[str, Any]:
        profile = state.get("patient_profile")
        listwise = state.get("listwise_candidates") or []
        if profile is None or not listwise:
            return {}
        s = get_settings()
        plan = state.get("search_plan")
        allowed_statuses: list[str] | None = None
        if plan is not None and plan.mandatory_filters:
            statuses = plan.mandatory_filters.get("status")
            if isinstance(statuses, list):
                allowed_statuses = statuses
        final = tools.apply_filters(
            list(listwise),
            profile,
            allowed_statuses=allowed_statuses,
            filter_status=s.runner.mode == "clinical_active",
        )
        entity = apply_entity_negation_rerank(
            final,
            get_trial=tools.get_trial,
            patient=profile,
            plan=plan,
            mode=s.runner.mode,
            policy=s.runner.benchmark_entity_rerank_policy,
            weight=s.runner.benchmark_entity_rerank_weight,
            protect_top=s.runner.benchmark_entity_protect_top,
        )
        return {
            "final_candidates": entity.candidates,
            "entity_negation_diagnostics": entity.diagnostics,
        }

    return apply_filters_node


# ----------------------------------------------------------------------
# Stage 4 — eligibility
# ----------------------------------------------------------------------


def make_evaluate_eligibility_node(tools: AgentTools):
    @_timed("evaluate_eligibility")
    async def evaluate_eligibility_node(state: AgentStateDict) -> dict[str, Any]:
        profile = state.get("patient_profile")
        final = state.get("final_candidates") or []
        if profile is None or not final:
            return {}
        s = get_settings()
        viable = [c for c in final if not c.hard_excluded]
        selected = select_viable_candidates(
            viable,
            get_trial=tools.get_trial,
            patient=profile,
            plan=state.get("search_plan"),
            policy=s.runner.benchmark_candidate_selection_policy,
            mode=s.runner.mode,
            cap=s.runner.max_trials_per_topic,
            keep_top=s.runner.benchmark_diverse_keep_top,
            select_total=s.runner.benchmark_diverse_select_total,
        )
        viability_diagnostics = selected.diagnostics
        if (
            s.runner.max_trials_per_topic > 0
            and len(viable) > len(selected.selected)
        ):
            dropped = len(viable) - len(selected.selected)
            kept_min = selected.selected[-1].score if selected.selected else None
            selected_ids = {c.nct_id for c in selected.selected}
            dropped_scores = [c.score for c in viable if c.nct_id not in selected_ids]
            dropped_max = max(dropped_scores) if dropped_scores else None
            logger.info(
                f"evaluate_eligibility: selecting viable {len(viable)}->{len(selected.selected)}; "
                f"dropped={dropped}, kept_min_score={kept_min}, "
                f"dropped_max_score={dropped_max}"
            )
        viable = selected.selected
        max_rank = max(
            (c.rank for c in (final or []) if not c.hard_excluded and c.rank > 0),
            default=max(len(final), 1),
        )

        extracted: dict[str, list] = {}
        selection_diagnostics: dict[str, dict[str, Any]] = {}
        irrelevance_diagnostics: dict[str, dict[str, Any]] = {}
        evals_by_id: dict[str, list] = {}
        trial_evals: dict[str, Any] = {}
        llm_calls_added = 0
        for fallback_rank, cand in enumerate(viable, start=1):
            trial = tools.get_trial(cand.nct_id)
            if trial is None:
                continue
            if hasattr(tools, "extract_criteria_with_diagnostics"):
                result = await tools.extract_criteria_with_diagnostics(
                    trial,
                    max_criteria=s.runner.max_criteria_per_trial,
                    patient=profile,
                )
                criteria = list(result.criteria)
                selection_diagnostics[cand.nct_id] = dict(result.diagnostics)
            else:
                criteria = await tools.extract_criteria(
                    trial, max_criteria=s.runner.max_criteria_per_trial
                )
            if s.runner.use_section_header_policy:
                before_header_filter = len(criteria)
                criteria = [c for c in criteria if c.type != CriterionType.SECTION_HEADER]
                if before_header_filter != len(criteria):
                    selection_diagnostics.setdefault(cand.nct_id, {})[
                        "section_headers_excluded_before_eval"
                    ] = before_header_filter - len(criteria)
            evals = await tools.evaluate_eligibility(criteria, profile)
            llm_calls_added += sum(int(getattr(ev, "llm_calls", 0) or 0) for ev in evals)
            extracted[cand.nct_id] = list(criteria)
            evals_by_id[cand.nct_id] = list(evals)
            trial_eval = aggregate_to_trial_eval(
                cand.nct_id,
                criteria,
                evals,
                use_irrelevance_heuristic=(
                    s.runner.use_irrelevance_heuristic
                    and not s.runner.use_multisignal_irrelevance_heuristic
                ),
            )
            irrel = apply_multisignal_irrelevance_policy(
                trial=trial,
                trial_eval=trial_eval,
                candidate=cand,
                patient=profile,
                plan=state.get("search_plan"),
                enabled=s.runner.use_multisignal_irrelevance_heuristic,
                mode=s.runner.mode,
                retrieval_prior=retrieval_prior_for_candidate(
                    cand,
                    max_rank=max_rank,
                    fallback_rank=fallback_rank,
                ),
                min_nei_fraction=s.runner.irrelevant_min_nei_fraction,
                max_inclusion_met=s.runner.irrelevant_max_inclusion_met,
                max_retrieval_prior=s.runner.irrelevant_max_retrieval_prior,
                min_signal_count=s.runner.irrelevant_min_signal_count,
            )
            trial_evals[cand.nct_id] = irrel.trial_eval
            irrelevance_diagnostics[cand.nct_id] = irrel.diagnostics
        return {
            "extracted_criteria": extracted,
            "criterion_selection_diagnostics": selection_diagnostics,
            "candidate_selection_diagnostics": viability_diagnostics,
            "irrelevance_diagnostics": irrelevance_diagnostics,
            "criterion_evals": evals_by_id,
            "trial_evals": trial_evals,
            "llm_calls": llm_calls_added,
        }

    return evaluate_eligibility_node


# ----------------------------------------------------------------------
# Stage 5 — ranking
# ----------------------------------------------------------------------


def make_rank_node(tools: AgentTools):
    @_timed("rank")
    async def rank_node(state: AgentStateDict) -> dict[str, Any]:
        profile = state.get("patient_profile")
        trial_evals = state.get("trial_evals") or {}
        if profile is None:
            return {}
        s = get_settings()
        output_k = max(1, int(s.runner.output_top_k or 10))
        viable_candidates = [
            c for c in (state.get("final_candidates") or []) if not c.hard_excluded
        ]
        max_rank = max((c.rank for c in viable_candidates if c.rank > 0), default=0)
        if max_rank <= 1:
            max_rank = max(len(viable_candidates), 1)
        candidate_by_id = {c.nct_id: c for c in viable_candidates}
        selection_sources = (
            (state.get("candidate_selection_diagnostics") or {}).get(
                "selection_source_by_id"
            )
            or {}
        )

        def retrieval_prior(nct_id: str) -> float:
            cand = candidate_by_id.get(nct_id)
            if cand is None:
                return 0.0
            rank = cand.rank if cand.rank > 0 else max_rank
            if max_rank <= 1:
                return 1.0
            return max(0.0, min(1.0, 1.0 - ((rank - 1) / (max_rank - 1))))

        ranked: list[RankedTrial] = []
        criterion_evidence_diagnostics: dict[str, Any] = {}
        for nct_id, evald in trial_evals.items():
            trial = tools.get_trial(nct_id)
            if trial is None:
                continue
            scored = score_trial(
                trial,
                evald,
                patient_location=profile.location,
                mode=s.runner.mode,
                benchmark_soft_veto=s.runner.benchmark_soft_veto,
                retrieval_prior=retrieval_prior(nct_id),
            )
            evidence = apply_criterion_evidence_adjustment(
                scored,
                mode=s.runner.mode,
                policy=s.runner.benchmark_criterion_evidence_policy,
                weight=s.runner.benchmark_criterion_evidence_weight,
            )
            ranked.append(evidence.ranked)
            criterion_evidence_diagnostics[nct_id] = evidence.diagnostics
        outside_selection_sources = {
            "diverse_outside_top_score",
            "entity_outside_top_score",
        }
        protected_scores = [
            r.score
            for r in ranked
            if selection_sources.get(r.nct_id) not in outside_selection_sources
        ]
        if protected_scores:
            demotion_ceiling = min(protected_scores) - 0.001
            for i, r in enumerate(ranked):
                source = selection_sources.get(r.nct_id)
                if source not in outside_selection_sources:
                    continue
                if r.score <= demotion_ceiling:
                    continue
                components = dict(r.components or {})
                components["outside_selection_demoted"] = 1.0
                components["outside_selection_original_score"] = float(r.score)
                if source == "diverse_outside_top_score":
                    components["outside_selection_source_code"] = 1.0
                    components["diverse_selection_demoted"] = 1.0
                    components["diverse_original_score"] = float(r.score)
                if source == "entity_outside_top_score":
                    components["outside_selection_source_code"] = 2.0
                    components["entity_selection_demoted"] = 1.0
                    components["entity_original_score"] = float(r.score)
                ranked[i] = r.model_copy(
                    update={
                        "score": demotion_ceiling,
                        "components": components,
                    }
                )
        ranked.sort(key=lambda r: r.score, reverse=True)

        # ── P7: fill ranking with hard-excluded when viable < output_top_k ──
        fill_count = 0
        retrieval_tail_fill_count = 0
        fill_reasons: dict[str, int] = {}
        skipped_corpus_miss = 0
        if (
            s.runner.mode == "benchmark"
            and s.runner.use_hard_excluded_fill
            and len(ranked) < output_k
        ):
            fill = fill_hard_excluded_to_top_k(
                ranked=ranked,
                final_candidates=state.get("final_candidates") or [],
                evaluated_ids=set(trial_evals),
                get_trial=tools.get_trial,
                output_k=output_k,
                include_retrieval_tail=s.runner.use_retrieval_tail_fill,
            )
            ranked = fill.ranked
            fill_count = fill.count
            retrieval_tail_fill_count = fill.retrieval_tail_count
            fill_reasons = fill.reasons
            skipped_corpus_miss = fill.skipped_corpus_miss
            if fill_count > 0:
                logger.info(
                    f"P7 excluded fill: added {fill_count} hard-excluded trials "
                    f"to ranking (viable={len(ranked) - fill_count}, "
                    f"output_top_k={output_k})"
                )

        for i, r in enumerate(ranked, start=1):
            r.rank = i
        return {
            "ranked_trials": ranked,
            "hard_excluded_fill_count": fill_count,
            "retrieval_tail_fill_count": retrieval_tail_fill_count,
            "hard_excluded_fill_reasons": fill_reasons,
            "hard_excluded_fill_skipped_corpus_miss": skipped_corpus_miss,
            "criterion_evidence_diagnostics": criterion_evidence_diagnostics,
        }

    return rank_node


def make_judge_node(judge: LLMJudge, tools: AgentTools):
    @_timed("llm_judge_top10")
    async def judge_node(state: AgentStateDict) -> dict[str, Any]:
        profile = state.get("patient_profile")
        ranked = state.get("ranked_trials") or []
        if profile is None or not ranked:
            return {}
        s = get_settings()
        output_k = max(1, int(s.runner.output_top_k or 10))
        if not s.runner.use_llm_judge:
            judged = [
                JudgedTrial(
                    nct_id=r.nct_id,
                    rank=i,
                    score=r.score,
                    eval=r.eval,
                    rationale="LLM judge disabled; preserving deterministic score order.",
                    components=r.components,
                    hard_excluded_fill=r.hard_excluded_fill,
                    retrieval_tail_fill=r.retrieval_tail_fill,
                    excluded_reason=r.excluded_reason,
                )
                for i, r in enumerate(ranked[:output_k], start=1)
            ]
            return {"judged_top10": judged}
        if all(
            r.hard_excluded_fill or r.retrieval_tail_fill
            for r in ranked[:output_k]
        ):
            judged = [
                JudgedTrial(
                    nct_id=r.nct_id,
                    rank=i,
                    score=r.score,
                    eval=r.eval,
                    rationale="Only benchmark hard-excluded fills available; preserving deterministic order.",
                    components=r.components,
                    hard_excluded_fill=r.hard_excluded_fill,
                    retrieval_tail_fill=r.retrieval_tail_fill,
                    excluded_reason=r.excluded_reason,
                )
                for i, r in enumerate(ranked[:output_k], start=1)
            ]
            return {"judged_top10": judged}
        meta: dict[str, dict[str, str]] = {}
        for r in ranked[:10]:
            t = tools.get_trial(r.nct_id)
            if t is None:
                meta[r.nct_id] = {"title": "", "phase": "NA", "status": "UNKNOWN"}
                continue
            meta[r.nct_id] = {
                "title": t.title,
                "phase": getattr(t.phase, "value", str(t.phase)),
                "status": getattr(t.status, "value", str(t.status)),
            }
        judged = await judge.judge(profile, ranked, meta, top_n=10)
        return {"judged_top10": judged, "llm_calls": 1}

    return judge_node


def make_critique_node(critic: SelfCritic):
    @_timed("self_critique")
    async def critique_node(state: AgentStateDict) -> dict[str, Any]:
        profile = state.get("patient_profile")
        judged = state.get("judged_top10") or []
        if profile is None or not judged:
            return {}
        new_top10, critique = await critic.critique(profile, judged)
        update: dict[str, Any] = {
            "judged_top10": new_top10,
            "critique": critique,
        }
        if critique is not None:
            notes = [c.issue for c in critique.issues_found]
            if notes:
                update["critique_notes"] = notes
            update["llm_calls"] = 1
        return update

    return critique_node


def make_apply_critique_order_node():
    @_timed("apply_critique_order")
    async def apply_critique_order_node(state: AgentStateDict) -> dict[str, Any]:
        judged = list(state.get("judged_top10") or [])
        if not judged:
            return {}

        rerank_needed = bool(
            state.get("critique") is not None and state["critique"].rerank_needed
        )
        normalized_judged: list[JudgedTrial] = []
        for i, j in enumerate(judged, start=1):
            normalized_judged.append(j.model_copy(update={"rank": i}))

        synced_ranked = [
            RankedTrial(
                nct_id=j.nct_id,
                rank=j.rank,
                score=j.score,
                eval=j.eval,
                components=j.components,
                hard_excluded_fill=j.hard_excluded_fill,
                retrieval_tail_fill=j.retrieval_tail_fill,
                excluded_reason=j.excluded_reason,
            )
            for j in normalized_judged
        ]
        update: dict[str, Any] = {
            "judged_top10": normalized_judged,
            "ranked_trials": synced_ranked,
        }
        if rerank_needed:
            update["critique_iterations"] = 1
        return update

    return apply_critique_order_node


# ----------------------------------------------------------------------
# Stage 6 — output
# ----------------------------------------------------------------------


def make_question_node(qgen: QuestionGenerator, tools: AgentTools):
    @_timed("generate_questions")
    async def question_node(state: AgentStateDict) -> dict[str, Any]:
        s = get_settings()
        if not s.runner.use_questions:
            return {"questions": []}
        profile = state.get("patient_profile")
        if profile is None:
            return {}
        questions: list = []
        llm_calls_added = 0
        criterion_evals = state.get("criterion_evals") or {}
        for j in state.get("judged_top10") or []:
            trial = tools.get_trial(j.nct_id)
            if trial is None:
                continue
            for ev in criterion_evals.get(j.nct_id, []):
                if ev.label != EligibilityLabel.NEI:
                    continue
                if ev.criterion is None:
                    continue
                q = await qgen.generate(ev.criterion, profile, trial, ev)
                questions.append(q)
                llm_calls_added += 1
        return {"questions": questions, "llm_calls": llm_calls_added}

    return question_node


def make_dossier_node(builder: DossierBuilder, tools: AgentTools):
    @_timed("build_dossiers")
    async def dossier_node(state: AgentStateDict) -> dict[str, Any]:
        s = get_settings()
        if not s.runner.use_dossiers:
            return {"dossiers": []}
        profile = state.get("patient_profile")
        if profile is None:
            return {}
        dossiers: list = []
        questions = state.get("questions") or []
        critique_notes = list(state.get("critique_notes") or [])
        for j in state.get("judged_top10") or []:
            trial = tools.get_trial(j.nct_id)
            if trial is None:
                continue
            qs = [q for q in questions if q.trial_id == j.nct_id]
            d = await builder.build(
                trial=trial,
                judged=j,
                patient=profile,
                questions=qs,
                critique_notes=critique_notes,
            )
            dossiers.append(d)
        return {"dossiers": dossiers}

    return dossier_node
