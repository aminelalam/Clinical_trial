"""Output-stage feature flags.

The agent graph keeps T4/T5 nodes in place, but evaluation runs can disable
them to measure retrieval, eligibility, and ranking without spending time or
LLM calls on question/dossier generation.
"""

from __future__ import annotations

import asyncio

from trial_matcher.agent.nodes import make_dossier_node, make_question_node
from trial_matcher.config import get_settings
from trial_matcher.models.agent_state import initial_state


class _QuestionGeneratorShouldNotRun:
    async def generate(self, *args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError("question generator should not run when disabled")


class _DossierBuilderShouldNotRun:
    async def build(self, *args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError("dossier builder should not run when disabled")


class _ToolsShouldNotRun:
    def get_trial(self, *args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError("trial lookup should not run when output stage is disabled")


def test_questions_disabled_returns_empty_without_llm_call():
    get_settings.cache_clear()
    settings = get_settings()
    old = settings.runner.use_questions
    settings.runner.use_questions = False
    try:
        state = initial_state(patient_raw="patient", run_id="t1")
        node = make_question_node(_QuestionGeneratorShouldNotRun(), _ToolsShouldNotRun())

        out = asyncio.run(node(state))

        assert out["questions"] == []
        assert "llm_calls" not in out
    finally:
        settings.runner.use_questions = old
        get_settings.cache_clear()


def test_dossiers_disabled_returns_empty_without_trial_lookup():
    get_settings.cache_clear()
    settings = get_settings()
    old = settings.runner.use_dossiers
    settings.runner.use_dossiers = False
    try:
        state = initial_state(patient_raw="patient", run_id="t1")
        node = make_dossier_node(_DossierBuilderShouldNotRun(), _ToolsShouldNotRun())

        out = asyncio.run(node(state))

        assert out["dossiers"] == []
    finally:
        settings.runner.use_dossiers = old
        get_settings.cache_clear()
