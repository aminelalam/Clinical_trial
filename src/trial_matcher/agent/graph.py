"""LangGraph state machine assembling the full agent.

State shape: ``AgentStateDict`` (TypedDict). Nodes return partial updates;
LangGraph merges them via reducers declared in ``models/agent_state.py``.

Agentic controls:
- ``retrieve_lexical`` and ``retrieve_dense`` fan out after planning and join
  at ``fuse_rrf``.
- ``decide_re_retrieval`` loops back through ``mark_retrieval_retry`` so the
  next planner call receives ``relax=True``.
- ``self_critique`` is a final ordering pass. It can demote judged trials, but
  it no longer loops back into deterministic ``rank`` or another LLM judge.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from ..dossier.builder import DossierBuilder
from ..llm.client import UnifiedLLM
from ..models.agent_state import AgentStateDict
from ..nlp.patient_extractor import PatientExtractor
from ..nlp.question_generator import QuestionGenerator
from ..ranking.critique import SelfCritic
from ..ranking.llm_judge import LLMJudge
from .edges import (
    decide_re_retrieval,
    decide_self_critique_enabled,
    mark_retrieval_retry,
)
from .nodes import (
    make_apply_filters_node,
    make_apply_critique_order_node,
    make_critique_node,
    make_dossier_node,
    make_evaluate_eligibility_node,
    make_fuse_node,
    make_judge_node,
    make_normalize_mesh_node,
    make_parse_patient_node,
    make_plan_search_node,
    make_question_node,
    make_rank_node,
    make_rerank_listwise_node,
    make_rerank_pointwise_node,
    make_retrieve_dense_node,
    make_retrieve_lexical_node,
)
from .planner import SearchPlanner
from .tools import AgentTools


def build_agent(
    tools: AgentTools | None = None,
    mesh_normalizer: Any | None = None,
    llm: UnifiedLLM | None = None,
) -> Any:
    """Compile and return the LangGraph agent over ``AgentStateDict``."""
    tools = tools or AgentTools()
    llm = llm or UnifiedLLM()

    extractor = PatientExtractor(llm)
    planner = SearchPlanner(llm)
    judge = LLMJudge(llm)
    critic = SelfCritic(llm)
    qgen = QuestionGenerator(llm)
    builder = DossierBuilder(llm)

    g: StateGraph = StateGraph(AgentStateDict)

    g.add_node("parse_patient", make_parse_patient_node(extractor))
    g.add_node("normalize_mesh", make_normalize_mesh_node(mesh_normalizer))
    g.add_node("plan_search", make_plan_search_node(planner))
    g.add_node("retrieve_lexical", make_retrieve_lexical_node(tools))
    g.add_node("retrieve_dense", make_retrieve_dense_node(tools))
    g.add_node("fuse_rrf", make_fuse_node())
    g.add_node("rerank_pointwise", make_rerank_pointwise_node(tools))
    g.add_node("rerank_listwise", make_rerank_listwise_node(tools))
    g.add_node("apply_hard_filters", make_apply_filters_node(tools))
    g.add_node("evaluate_eligibility", make_evaluate_eligibility_node(tools))
    g.add_node("rank", make_rank_node(tools))
    g.add_node("llm_judge_top10", make_judge_node(judge, tools))
    g.add_node("self_critique", make_critique_node(critic))
    g.add_node("mark_retrieval_retry", mark_retrieval_retry)
    g.add_node("apply_critique_order", make_apply_critique_order_node())
    g.add_node("generate_questions", make_question_node(qgen, tools))
    g.add_node("build_dossiers", make_dossier_node(builder, tools))

    g.set_entry_point("parse_patient")
    g.add_edge("parse_patient", "normalize_mesh")
    g.add_edge("normalize_mesh", "plan_search")

    # Parallel retrieval — TypedDict + per-channel reducers make this safe.
    g.add_edge("plan_search", "retrieve_lexical")
    g.add_edge("plan_search", "retrieve_dense")
    g.add_edge("retrieve_lexical", "fuse_rrf")
    g.add_edge("retrieve_dense", "fuse_rrf")
    g.add_edge("fuse_rrf", "rerank_pointwise")
    g.add_edge("rerank_pointwise", "rerank_listwise")
    g.add_edge("rerank_listwise", "apply_hard_filters")

    # Conditional edge: re-retrieve when fewer than 10 viable candidates remain.
    g.add_conditional_edges(
        "apply_hard_filters",
        decide_re_retrieval,
        {"re_retrieve": "mark_retrieval_retry", "continue": "evaluate_eligibility"},
    )
    g.add_edge("mark_retrieval_retry", "plan_search")

    g.add_edge("evaluate_eligibility", "rank")
    g.add_edge("rank", "llm_judge_top10")
    g.add_conditional_edges(
        "llm_judge_top10",
        decide_self_critique_enabled,
        {"critique": "self_critique", "skip": "apply_critique_order"},
    )

    # Optional final critique; no loop back into rank/judge.
    g.add_edge("self_critique", "apply_critique_order")

    g.add_edge("apply_critique_order", "generate_questions")
    g.add_edge("generate_questions", "build_dossiers")
    g.add_edge("build_dossiers", END)

    return g.compile()
