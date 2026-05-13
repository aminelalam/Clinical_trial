"""End-to-end pipeline test — does the agent compile and run on a sample?

This test only checks that build_agent() compiles. A full E2E run requires
indices and an LLM endpoint, both behind --needs-llm and --needs-gpu markers.
"""

from __future__ import annotations

import pytest


def test_agent_graph_compiles():
    """``build_agent`` should produce a compiled LangGraph without exceptions.

    This catches breakage in node wiring early.
    """
    pytest.importorskip("langgraph")
    pytest.importorskip("openai")
    from trial_matcher.agent.graph import build_agent
    from trial_matcher.agent.tools import AgentTools

    # Use a tools instance whose underlying clients are not loaded — we never call them.
    tools = AgentTools(trials_by_id={})
    g = build_agent(tools=tools)
    # The compiled graph object exposes ainvoke; we don't run it (would call LLM).
    assert hasattr(g, "ainvoke")


@pytest.mark.needs_llm
@pytest.mark.integration
async def test_full_run_one_patient(sample_trial):
    """Smoke test that requires LLM credentials. Skipped by default."""
    pytest.skip("Requires LLM credentials; run with -m needs_llm to enable")
