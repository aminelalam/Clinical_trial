"""evaluate_eligibility_node sorts by score before slicing (B4)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace


def test_viable_sliced_by_score(monkeypatch, sample_patient):
    """If max_trials_per_topic=2, the two HIGHEST-score viable candidates
    should be evaluated, not the first two in the input order."""
    from trial_matcher.agent.nodes import make_evaluate_eligibility_node
    from trial_matcher.models.agent_state import TrialCandidate, initial_state
    from trial_matcher.models.criterion import Criterion, CriterionType, Polarity
    from trial_matcher.models.eligibility import (
        CriterionEval,
        EligibilityLabel,
    )
    from trial_matcher.models.trial import (
        AgeRange,
        Eligibility,
        Phase,
        RecruitmentStatus,
        Sex,
        Trial,
    )

    # Three candidates in a non-monotonic order so a naive prefix slice would
    # pick the wrong two.
    cands = [
        TrialCandidate(nct_id="NCT_LOW", score=0.10, source="bm25", rank=1),
        TrialCandidate(nct_id="NCT_HIGH", score=0.90, source="bm25", rank=2),
        TrialCandidate(nct_id="NCT_MID", score=0.50, source="bm25", rank=3),
    ]

    def _fake_trial(nct_id: str) -> Trial:
        return Trial(
            nct_id=nct_id,
            title=nct_id,
            brief_summary="",
            conditions=[],
            phase=Phase.PHASE_2,
            status=RecruitmentStatus.RECRUITING,
            interventional=True,
            eligibility=Eligibility(
                raw_text="Inclusion Criteria:\n- Age >= 18",
                inclusion_text="Age >= 18",
                exclusion_text="",
                age_range=AgeRange(),
                sex=Sex.ALL,
            ),
        )

    class _Tools:
        def get_trial(self, nct_id: str):
            return _fake_trial(nct_id)

        async def extract_criteria(self, trial: Trial, max_criteria: int = 0):
            return [
                Criterion(
                    id="i_1",
                    polarity=Polarity.INCLUSION,
                    raw_text="age >= 18",
                    type=CriterionType.AGE,
                )
            ]

        async def evaluate_eligibility(self, criteria, patient):
            return [
                CriterionEval(
                    criterion_id=c.id,
                    label=EligibilityLabel.MET,
                    confidence=0.9,
                    evidence="age 47",
                    reasoning="ok",
                    evaluator="deterministic",
                    llm_calls=2,
                    criterion=c,
                )
                for c in criteria
            ]

    # Force max_trials_per_topic=2.
    monkeypatch.setenv("TRIAL_MATCHER__RUNNER__MAX_TRIALS_PER_TOPIC", "2")
    from trial_matcher.config import get_settings

    get_settings.cache_clear()

    # Build a proper AgentStateDict (plain dict) so node .get() calls work.
    state = initial_state(patient_raw="hi", run_id="test")
    state["patient_profile"] = sample_patient
    state["final_candidates"] = cands

    node = make_evaluate_eligibility_node(_Tools())
    out = asyncio.run(node(state))

    evaluated = set(out["trial_evals"].keys())
    assert evaluated == {"NCT_HIGH", "NCT_MID"}, (
        f"Expected the two highest-score viable trials, got {evaluated}"
    )
    assert out["llm_calls"] == 4
    get_settings.cache_clear()


def test_evaluate_node_returns_criterion_triage_diagnostics(monkeypatch, sample_patient):
    from trial_matcher.agent.nodes import make_evaluate_eligibility_node
    from trial_matcher.models.agent_state import TrialCandidate, initial_state
    from trial_matcher.models.criterion import Criterion, CriterionType, Polarity
    from trial_matcher.models.eligibility import CriterionEval, EligibilityLabel
    from trial_matcher.models.trial import Eligibility, Trial

    trial = Trial(
        nct_id="NCT_TRIAGE",
        title="triage",
        eligibility=Eligibility(raw_text="Eligibility", inclusion_text="Age >= 18"),
    )
    criterion = Criterion(
        id="i_1",
        polarity=Polarity.INCLUSION,
        raw_text="Age >= 18",
        type=CriterionType.AGE,
    )

    class _Tools:
        def get_trial(self, nct_id: str):
            return trial if nct_id == "NCT_TRIAGE" else None

        async def extract_criteria_with_diagnostics(self, trial, max_criteria=0, patient=None):
            return SimpleNamespace(
                criteria=[criterion],
                diagnostics={
                    "triage_enabled": True,
                    "total_seen": 4,
                    "selected": 1,
                    "dropped": 3,
                    "selected_by_type": {"age": 1},
                    "dropped_by_type": {"consent": 2, "other": 1},
                },
            )

        async def evaluate_eligibility(self, criteria, patient):
            return [
                CriterionEval(
                    criterion_id=c.id,
                    label=EligibilityLabel.MET,
                    confidence=0.9,
                    evaluator="deterministic",
                    criterion=c,
                )
                for c in criteria
            ]

    monkeypatch.setenv("TRIAL_MATCHER__RUNNER__MAX_TRIALS_PER_TOPIC", "1")
    from trial_matcher.config import get_settings

    get_settings.cache_clear()

    state = initial_state(patient_raw="hi", run_id="test")
    state["patient_profile"] = sample_patient
    state["final_candidates"] = [
        TrialCandidate(nct_id="NCT_TRIAGE", score=1.0, source="bm25", rank=1)
    ]

    node = make_evaluate_eligibility_node(_Tools())
    out = asyncio.run(node(state))

    assert out["criterion_selection_diagnostics"]["NCT_TRIAGE"]["dropped"] == 3
    assert out["extracted_criteria"]["NCT_TRIAGE"][0].type == CriterionType.AGE
    get_settings.cache_clear()
