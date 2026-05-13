"""Minimal FastAPI service exposing /match for interactive testing."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import get_settings
from .logging import setup_logging
from .models.agent_state import AgentState, initial_state
from .runtime import AgentRuntime, build_agent_runtime

setup_logging()

app = FastAPI(title="Trial Matcher", version="0.1.0")

_RUNTIME: AgentRuntime | None = None


def _get_runtime() -> AgentRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        s = get_settings()
        _RUNTIME = build_agent_runtime(use_few_shot=s.runner.use_few_shot)
    return _RUNTIME


class MatchRequest(BaseModel):
    patient_id: str
    raw_text: str
    top_k: int = 10


class MatchResponse(BaseModel):
    patient_id: str
    ranked_trials: list[dict]
    questions: list[dict]
    dossiers: list[dict]
    stats: dict


@app.get("/health")
def health() -> dict:
    s = get_settings()
    runtime = _get_runtime()
    return {
        "status": "ok",
        "version": "0.1.0",
        "default_provider": s.llm.default_provider,
        "mini_deployment": s.llm.azure_deployment_mini,
        "large_deployment": s.llm.azure_deployment_large,
        "api_version": s.llm.azure_api_version,
        "indices_dir": str(s.paths.indices_dir),
        "runner_mode": s.runner.mode,
        "use_self_consistency": s.runner.use_self_consistency,
        "use_llm_judge": s.runner.use_llm_judge,
        "use_self_critique": s.runner.use_self_critique,
        "use_questions": s.runner.use_questions,
        "use_dossiers": s.runner.use_dossiers,
        "use_criterion_triage": s.runner.use_criterion_triage,
        "use_section_header_policy": s.runner.use_section_header_policy,
        "benchmark_candidate_selection_policy": s.runner.benchmark_candidate_selection_policy,
        "benchmark_entity_rerank_policy": s.runner.benchmark_entity_rerank_policy,
        "benchmark_entity_rerank_weight": s.runner.benchmark_entity_rerank_weight,
        "benchmark_entity_protect_top": s.runner.benchmark_entity_protect_top,
        "benchmark_criterion_evidence_policy": (
            s.runner.benchmark_criterion_evidence_policy
        ),
        "benchmark_criterion_evidence_weight": (
            s.runner.benchmark_criterion_evidence_weight
        ),
        "use_hard_excluded_fill": s.runner.use_hard_excluded_fill,
        "use_retrieval_tail_fill": s.runner.use_retrieval_tail_fill,
        "use_irrelevance_heuristic": s.runner.use_irrelevance_heuristic,
        "use_multisignal_irrelevance_heuristic": (
            s.runner.use_multisignal_irrelevance_heuristic
        ),
        "corpus_loaded": runtime.corpus_loaded,
        "corpus_size": runtime.corpus_size,
        "mesh_loaded": runtime.mesh_loaded,
        "mesh_concepts": runtime.mesh_concepts,
        "qdrant_collection": runtime.qdrant_collection,
        "qdrant_mode": runtime.qdrant_mode,
        "qdrant_url": runtime.qdrant_url,
        "qdrant_collection_exists": runtime.qdrant_collection_exists,
        "qdrant_points_count": runtime.qdrant_points_count,
    }


@app.post("/match", response_model=MatchResponse)
async def match(req: MatchRequest) -> MatchResponse:
    runtime = _get_runtime()
    s = get_settings()
    s.runner.output_top_k = max(1, req.top_k)
    state = initial_state(
        patient_raw=req.raw_text,
        run_id=req.patient_id,
        max_retrieval_attempts=s.runner.max_retrieval_attempts,
        max_critique_iterations=s.runner.max_critique_iterations,
    )
    try:
        final = await runtime.agent.ainvoke(state)
        final_state = AgentState.model_validate(final) if isinstance(final, dict) else final
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return MatchResponse(
        patient_id=req.patient_id,
        ranked_trials=[
            {
                "nct_id": j.nct_id,
                "rank": j.rank,
                "score": round(float(j.score), 4),
                "label": j.eval.label.value,
                "rationale": j.rationale,
                "hard_excluded_fill": bool(j.hard_excluded_fill),
                "retrieval_tail_fill": bool(j.retrieval_tail_fill),
                "excluded_reason": j.excluded_reason,
            }
            for j in final_state.judged_top10[: req.top_k]
        ],
        questions=[q.model_dump(mode="json") for q in final_state.questions],
        dossiers=[d.model_dump(mode="json") for d in final_state.dossiers[: req.top_k]],
        stats={
            "llm_calls": final_state.llm_calls,
            "cache_hits": final_state.cache_hits,
            "corpus_loaded": runtime.corpus_loaded,
            "corpus_size": runtime.corpus_size,
            "qdrant_mode": runtime.qdrant_mode,
            "qdrant_points_count": runtime.qdrant_points_count,
            "retrieval_attempts": final_state.retrieval_attempts,
            "re_retrieval_triggered": final_state.re_retrieval_triggered,
            "relaxed_plan_used": final_state.relaxed_plan_used,
            "critique_iterations": final_state.critique_iterations,
        },
    )
