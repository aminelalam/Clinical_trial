"""Tests for the LangGraph conditional edges and loop-control logic.

Covers:
- decide_re_retrieval: caps at max_retrieval_attempts, continues when enough viables.
- Parallel retrieval: no INVALID_CONCURRENT_GRAPH_UPDATE when both branches write
  their own slots.
"""

from __future__ import annotations

from trial_matcher.agent.edges import (
    decide_re_retrieval,
    decide_self_critique_enabled,
    mark_retrieval_retry,
)
from trial_matcher.agent.nodes import (
    make_apply_critique_order_node,
    make_judge_node,
    make_plan_search_node,
)
from trial_matcher.models.agent_state import TrialCandidate, initial_state
from trial_matcher.models.critique import Critique
from trial_matcher.models.eligibility import TrialEval, TrialLabel
from trial_matcher.models.ranking import JudgedTrial, RankedTrial
from trial_matcher.models.search_plan import SearchPlan


class TestDecideReRetrieval:
    def test_returns_continue_when_attempts_exhausted(self):
        state = initial_state(patient_raw="x", run_id="t1")
        state["retrieval_attempts"] = 2
        state["max_retrieval_attempts"] = 2
        state["final_candidates"] = [
            TrialCandidate(nct_id="NCT001", score=0.5, hard_excluded=True),
        ]
        assert decide_re_retrieval(state) == "continue"

    def test_returns_continue_when_enough_viables(self):
        state = initial_state(patient_raw="x", run_id="t1")
        state["retrieval_attempts"] = 1
        state["max_retrieval_attempts"] = 2
        # 10 viable candidates → no need to re-retrieve
        state["final_candidates"] = [
            TrialCandidate(nct_id=f"NCT{i:04d}", score=0.1 * i, hard_excluded=False)
            for i in range(10, 20)
        ]
        assert decide_re_retrieval(state) == "continue"

    def test_returns_re_retrieve_when_few_viables_and_attempts_remain(self):
        state = initial_state(patient_raw="x", run_id="t1")
        state["retrieval_attempts"] = 0
        state["max_retrieval_attempts"] = 2
        # Only 3 viable candidates (< 10 threshold)
        state["final_candidates"] = [
            TrialCandidate(nct_id="NCT001", score=0.5, hard_excluded=False),
            TrialCandidate(nct_id="NCT002", score=0.3, hard_excluded=False),
            TrialCandidate(nct_id="NCT003", score=0.1, hard_excluded=False),
        ]
        state["search_plan"] = SearchPlan(primary_disease_query="cancer")
        assert decide_re_retrieval(state) == "re_retrieve"

    def test_returns_continue_when_plan_already_relaxed(self):
        state = initial_state(patient_raw="x", run_id="t1")
        state["retrieval_attempts"] = 1
        state["max_retrieval_attempts"] = 3
        state["final_candidates"] = [
            TrialCandidate(nct_id="NCT001", score=0.5, hard_excluded=False),
        ]
        state["search_plan"] = SearchPlan(
            primary_disease_query="cancer", relax_optional_filters=True
        )
        # Already relaxed → don't loop again
        assert decide_re_retrieval(state) == "continue"


class TestReRetrievalSemantics:
    def test_mark_retrieval_retry_sets_relax_flag(self):
        update = mark_retrieval_retry(initial_state(patient_raw="x", run_id="t1"))
        assert update == {
            "needs_re_retrieval": True,
            "re_retrieval_triggered": True,
        }

    def test_second_planner_call_receives_relax_true(self, sample_patient):
        import asyncio

        class Planner:
            def __init__(self):
                self.relax_calls = []

            async def plan(self, patient, mesh_concepts_summary="", relax=False, mode=None):
                self.relax_calls.append(relax)
                return SearchPlan(
                    primary_disease_query="cancer",
                    relax_optional_filters=relax,
                    source="llm",
                )

        planner = Planner()
        node = make_plan_search_node(planner)
        state = initial_state(patient_raw="x", run_id="t1")
        state["patient_profile"] = sample_patient
        state["needs_re_retrieval"] = True

        out = asyncio.run(node(state))

        assert planner.relax_calls == [True]
        assert out["search_plan"].relax_optional_filters is True
        assert out["relaxed_plan_used"] is True
        assert out["needs_re_retrieval"] is False


class TestSelfCritiqueFinalOrdering:
    def test_apply_critique_order_preserves_demoted_judged_order(self):
        import asyncio

        evald = TrialEval(nct_id="x", label=TrialLabel.EXCLUDES)
        judged = [
            JudgedTrial(nct_id="NCT_C", rank=3, score=0.3, eval=evald),
            JudgedTrial(nct_id="NCT_A", rank=1, score=0.9, eval=evald),
            JudgedTrial(nct_id="NCT_B", rank=2, score=0.5, eval=evald),
        ]
        state = initial_state(patient_raw="x", run_id="t1")
        state["judged_top10"] = judged
        state["critique"] = Critique(rerank_needed=True)

        out = asyncio.run(make_apply_critique_order_node()(state))

        assert [j.nct_id for j in out["judged_top10"]] == ["NCT_C", "NCT_A", "NCT_B"]
        assert [j.rank for j in out["judged_top10"]] == [1, 2, 3]
        assert [r.nct_id for r in out["ranked_trials"]] == ["NCT_C", "NCT_A", "NCT_B"]
        assert out["critique_iterations"] == 1

    def test_self_critique_can_be_skipped_from_settings(self):
        from trial_matcher.config import get_settings

        get_settings.cache_clear()
        settings = get_settings()
        old = settings.runner.use_self_critique
        settings.runner.use_self_critique = False
        try:
            state = initial_state(patient_raw="x", run_id="t1")
            state["judged_top10"] = [
                JudgedTrial(
                    nct_id="NCT_A",
                    rank=1,
                    score=1.0,
                    eval=TrialEval(nct_id="NCT_A", label=TrialLabel.ELIGIBLE),
                )
            ]
            assert decide_self_critique_enabled(state) == "skip"
        finally:
            settings.runner.use_self_critique = old
            get_settings.cache_clear()

    def test_llm_judge_disabled_preserves_ranked_order_without_llm_call(self, sample_patient):
        import asyncio
        from trial_matcher.config import get_settings

        class JudgeShouldNotRun:
            async def judge(self, *args, **kwargs):  # pragma: no cover - failure path
                raise AssertionError("judge should not run when use_llm_judge=False")

        get_settings.cache_clear()
        settings = get_settings()
        old = settings.runner.use_llm_judge
        old_top_k = settings.runner.output_top_k
        settings.runner.use_llm_judge = False
        settings.runner.output_top_k = 20
        try:
            state = initial_state(patient_raw="x", run_id="t1")
            state["patient_profile"] = sample_patient
            state["ranked_trials"] = [
                RankedTrial(
                    nct_id=f"NCT_{i:02d}",
                    rank=i,
                    score=1.0 / i,
                    eval=TrialEval(
                        nct_id=f"NCT_{i:02d}",
                        label=TrialLabel.ELIGIBLE if i == 1 else TrialLabel.EXCLUDES,
                    ),
                )
                for i in range(1, 13)
            ]

            out = asyncio.run(make_judge_node(JudgeShouldNotRun(), tools=object())(state))

            assert len(out["judged_top10"]) == 12
            assert [j.nct_id for j in out["judged_top10"][:3]] == [
                "NCT_01",
                "NCT_02",
                "NCT_03",
            ]
            assert "llm_calls" not in out
        finally:
            settings.runner.use_llm_judge = old
            settings.runner.output_top_k = old_top_k
            get_settings.cache_clear()
