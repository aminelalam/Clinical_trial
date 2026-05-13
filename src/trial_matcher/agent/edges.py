"""Conditional edges for the LangGraph state machine.

Both edges are written against the ``AgentStateDict`` TypedDict. They never
mutate the state; they only inspect it and return the next branch label.
The actual loop-control mutations are returned by the nodes (e.g.
``plan_search_node`` adds 1 to ``retrieval_attempts`` via the additive
reducer). When a branch needs to change state before the next real node, we
use tiny helper nodes that return explicit updates.
"""

from __future__ import annotations

from ..config import get_settings
from ..models.agent_state import AgentStateDict
from ..retrieval.filters import viable_count


def decide_re_retrieval(state: AgentStateDict) -> str:
    """Branch label for the post-filter conditional edge.

    Returns ``"re_retrieve"`` if more than ``max_retrieval_attempts``
    iterations remain available AND fewer than 10 viable candidates
    survived the hard filter AND the planner did not already relax
    optional filters in the previous attempt. Otherwise ``"continue"``.
    """
    attempts = int(state.get("retrieval_attempts") or 0)
    cap = int(state.get("max_retrieval_attempts") or 2)
    if attempts >= cap:
        return "continue"
    final = state.get("final_candidates") or []
    if not final:
        return "continue"
    plan = state.get("search_plan")
    if plan is not None and plan.relax_optional_filters:
        return "continue"
    if viable_count(list(final)) < 10:
        return "re_retrieve"
    return "continue"


def mark_retrieval_retry(_state: AgentStateDict) -> dict:
    """Mark the next planner call as a real relaxed re-retrieval pass."""
    return {
        "needs_re_retrieval": True,
        "re_retrieval_triggered": True,
    }


def decide_self_critique_enabled(state: AgentStateDict) -> str:
    """Skip the critique node when disabled or already exhausted.

    Self-critique is now a final ordering pass, not an expensive loop back
    into the deterministic scorer. The cap still acts as a hard guard for
    users who set ``max_critique_iterations=0``.
    """
    s = get_settings()
    if not s.runner.use_self_critique:
        return "skip"
    if not (state.get("judged_top10") or []):
        return "skip"
    iters = int(state.get("critique_iterations") or 0)
    cap = int(state.get("max_critique_iterations") or 1)
    if iters >= cap:
        return "skip"
    return "critique"

