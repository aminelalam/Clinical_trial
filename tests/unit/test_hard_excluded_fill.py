"""P7 hard-excluded fill behavior in the rank node."""

from __future__ import annotations

import asyncio


def _configure_runner(monkeypatch, *, enabled: bool, mode: str = "benchmark", top_k: int = 3):
    monkeypatch.setenv("TRIAL_MATCHER__RUNNER__USE_HARD_EXCLUDED_FILL", str(enabled).lower())
    monkeypatch.setenv("TRIAL_MATCHER__RUNNER__MODE", mode)
    monkeypatch.setenv("TRIAL_MATCHER__RUNNER__OUTPUT_TOP_K", str(top_k))
    from trial_matcher.config import get_settings

    get_settings.cache_clear()
    return get_settings()


def _enable_retrieval_tail(monkeypatch):
    monkeypatch.setenv("TRIAL_MATCHER__RUNNER__USE_RETRIEVAL_TAIL_FILL", "true")
    from trial_matcher.config import get_settings

    get_settings.cache_clear()


def _trial(nct_id: str):
    from trial_matcher.models.trial import Eligibility, Phase, RecruitmentStatus, Trial

    return Trial(
        nct_id=nct_id,
        title=nct_id,
        brief_summary="Study summary",
        conditions=["breast cancer"],
        phase=Phase.PHASE_2,
        status=RecruitmentStatus.RECRUITING,
        eligibility=Eligibility(raw_text="Inclusion Criteria:\n- Age >= 18"),
    )


def _trial_eval(nct_id: str):
    from trial_matcher.models.eligibility import TrialEval, TrialLabel

    return TrialEval(
        nct_id=nct_id,
        label=TrialLabel.ELIGIBLE,
        n_inclusion=1,
        n_inclusion_met=1,
        n_exclusion=0,
        fraction_nei=0.0,
    )


class _Tools:
    def __init__(self, trial_ids: list[str]):
        self.trials = {nct_id: _trial(nct_id) for nct_id in trial_ids}

    def get_trial(self, nct_id: str):
        return self.trials.get(nct_id)


def _state(sample_patient, final_candidates, trial_evals):
    from trial_matcher.models.agent_state import initial_state

    state = initial_state(patient_raw="patient", run_id="p7")
    state["patient_profile"] = sample_patient
    state["final_candidates"] = final_candidates
    state["trial_evals"] = trial_evals
    return state


def test_hard_excluded_fill_default_off_does_not_change_ranking(monkeypatch, sample_patient):
    from trial_matcher.agent.nodes import make_rank_node
    from trial_matcher.models.agent_state import TrialCandidate

    _configure_runner(monkeypatch, enabled=False, top_k=3)
    state = _state(
        sample_patient,
        [
            TrialCandidate(nct_id="NCT_VIABLE", score=0.9, rank=1),
            TrialCandidate(
                nct_id="NCT_EXCLUDED",
                score=0.8,
                rank=2,
                hard_excluded=True,
                excluded_reason="age 15 below minimum",
            ),
        ],
        {"NCT_VIABLE": _trial_eval("NCT_VIABLE")},
    )

    out = asyncio.run(make_rank_node(_Tools(["NCT_VIABLE", "NCT_EXCLUDED"]))(state))

    assert [r.nct_id for r in out["ranked_trials"]] == ["NCT_VIABLE"]
    assert out["hard_excluded_fill_count"] == 0
    assert out.get("llm_calls", 0) == 0


def test_hard_excluded_fill_is_default_on_for_benchmark(monkeypatch, sample_patient):
    from trial_matcher.agent.nodes import make_rank_node
    from trial_matcher.config import get_settings
    from trial_matcher.models.agent_state import TrialCandidate

    monkeypatch.delenv("TRIAL_MATCHER__RUNNER__USE_HARD_EXCLUDED_FILL", raising=False)
    monkeypatch.setenv("TRIAL_MATCHER__RUNNER__MODE", "benchmark")
    monkeypatch.setenv("TRIAL_MATCHER__RUNNER__OUTPUT_TOP_K", "2")
    get_settings.cache_clear()
    state = _state(
        sample_patient,
        [
            TrialCandidate(nct_id="NCT_VIABLE", score=0.9, rank=1),
            TrialCandidate(
                nct_id="NCT_EXCLUDED",
                score=0.8,
                rank=2,
                hard_excluded=True,
                excluded_reason="age 15 below minimum",
            ),
        ],
        {"NCT_VIABLE": _trial_eval("NCT_VIABLE")},
    )

    out = asyncio.run(make_rank_node(_Tools(["NCT_VIABLE", "NCT_EXCLUDED"]))(state))

    assert [r.nct_id for r in out["ranked_trials"]] == ["NCT_VIABLE", "NCT_EXCLUDED"]
    assert out["hard_excluded_fill_count"] == 1
    get_settings.cache_clear()


def test_hard_excluded_fill_on_appends_excludes_after_viables(monkeypatch, sample_patient):
    from trial_matcher.agent.nodes import make_rank_node
    from trial_matcher.models.agent_state import TrialCandidate
    from trial_matcher.models.eligibility import TrialLabel

    _configure_runner(monkeypatch, enabled=True, top_k=3)
    state = _state(
        sample_patient,
        [
            TrialCandidate(nct_id="NCT_VIABLE", score=0.9, rank=1),
            TrialCandidate(
                nct_id="NCT_EXCLUDED_HIGH",
                score=0.8,
                rank=2,
                hard_excluded=True,
                excluded_reason="age 15 below minimum",
            ),
            TrialCandidate(
                nct_id="NCT_EXCLUDED_LOW",
                score=0.4,
                rank=3,
                hard_excluded=True,
                excluded_reason="sex FEMALE vs trial MALE",
            ),
        ],
        {"NCT_VIABLE": _trial_eval("NCT_VIABLE")},
    )

    out = asyncio.run(
        make_rank_node(_Tools(["NCT_VIABLE", "NCT_EXCLUDED_HIGH", "NCT_EXCLUDED_LOW"]))(
            state
        )
    )
    ranked = out["ranked_trials"]

    assert [r.nct_id for r in ranked] == [
        "NCT_VIABLE",
        "NCT_EXCLUDED_HIGH",
        "NCT_EXCLUDED_LOW",
    ]
    assert [r.rank for r in ranked] == [1, 2, 3]
    assert ranked[1].score < ranked[0].score
    assert ranked[2].score < ranked[0].score
    assert ranked[1].hard_excluded_fill
    assert ranked[1].excluded_reason == "age 15 below minimum"
    assert ranked[1].eval.label == TrialLabel.EXCLUDES
    assert ranked[1].eval.trec_qrel == 1
    assert out["hard_excluded_fill_count"] == 2
    assert out["hard_excluded_fill_reasons"] == {
        "age 15 below minimum": 1,
        "sex FEMALE vs trial MALE": 1,
    }
    assert out.get("llm_calls", 0) == 0


def test_retrieval_tail_fill_precedes_hard_excluded_fill(monkeypatch, sample_patient):
    from trial_matcher.agent.nodes import make_rank_node
    from trial_matcher.models.agent_state import TrialCandidate

    _configure_runner(monkeypatch, enabled=True, top_k=3)
    _enable_retrieval_tail(monkeypatch)
    state = _state(
        sample_patient,
        [
            TrialCandidate(nct_id="NCT_VIABLE", score=0.9, rank=1),
            TrialCandidate(nct_id="NCT_TAIL", score=0.8, rank=2),
            TrialCandidate(
                nct_id="NCT_EXCLUDED",
                score=0.7,
                rank=3,
                hard_excluded=True,
                excluded_reason="age 15 below minimum",
            ),
        ],
        {"NCT_VIABLE": _trial_eval("NCT_VIABLE")},
    )

    out = asyncio.run(
        make_rank_node(_Tools(["NCT_VIABLE", "NCT_TAIL", "NCT_EXCLUDED"]))(state)
    )
    ranked = out["ranked_trials"]

    assert [r.nct_id for r in ranked] == ["NCT_VIABLE", "NCT_TAIL", "NCT_EXCLUDED"]
    assert ranked[1].retrieval_tail_fill
    assert ranked[2].hard_excluded_fill
    assert ranked[0].score > ranked[1].score > ranked[2].score
    assert out["retrieval_tail_fill_count"] == 1
    assert out["hard_excluded_fill_count"] == 1


def test_hard_excluded_fill_scores_below_negative_viables(monkeypatch, sample_patient):
    from trial_matcher.agent.nodes import make_rank_node
    from trial_matcher.models.agent_state import TrialCandidate
    from trial_matcher.models.eligibility import TrialEval, TrialLabel

    _configure_runner(monkeypatch, enabled=True, top_k=2)
    veto_eval = TrialEval(
        nct_id="NCT_VIABLE",
        label=TrialLabel.EXCLUDES,
        any_mandatory_inclusion_failed=True,
    )
    state = _state(
        sample_patient,
        [
            TrialCandidate(nct_id="NCT_VIABLE", score=0.9, rank=1),
            TrialCandidate(
                nct_id="NCT_EXCLUDED",
                score=0.8,
                rank=2,
                hard_excluded=True,
                excluded_reason="age 15 below minimum",
            ),
        ],
        {"NCT_VIABLE": veto_eval},
    )

    out = asyncio.run(make_rank_node(_Tools(["NCT_VIABLE", "NCT_EXCLUDED"]))(state))
    ranked = out["ranked_trials"]

    assert ranked[0].nct_id == "NCT_VIABLE"
    assert ranked[1].nct_id == "NCT_EXCLUDED"
    assert ranked[1].score < ranked[0].score


def test_hard_excluded_fill_ignored_in_clinical_active(monkeypatch, sample_patient):
    from trial_matcher.agent.nodes import make_rank_node
    from trial_matcher.models.agent_state import TrialCandidate

    _configure_runner(monkeypatch, enabled=True, mode="clinical_active", top_k=3)
    state = _state(
        sample_patient,
        [
            TrialCandidate(nct_id="NCT_VIABLE", score=0.9, rank=1),
            TrialCandidate(
                nct_id="NCT_EXCLUDED",
                score=0.8,
                rank=2,
                hard_excluded=True,
                excluded_reason="age 15 below minimum",
            ),
        ],
        {"NCT_VIABLE": _trial_eval("NCT_VIABLE")},
    )

    out = asyncio.run(make_rank_node(_Tools(["NCT_VIABLE", "NCT_EXCLUDED"]))(state))

    assert [r.nct_id for r in out["ranked_trials"]] == ["NCT_VIABLE"]
    assert out["hard_excluded_fill_count"] == 0


def test_hard_excluded_fill_skips_trials_missing_from_corpus(monkeypatch, sample_patient):
    from trial_matcher.agent.nodes import make_rank_node
    from trial_matcher.models.agent_state import TrialCandidate

    _configure_runner(monkeypatch, enabled=True, top_k=3)
    state = _state(
        sample_patient,
        [
            TrialCandidate(nct_id="NCT_VIABLE", score=0.9, rank=1),
            TrialCandidate(
                nct_id="NCT_MISSING",
                score=0.8,
                rank=2,
                hard_excluded=True,
                excluded_reason="trial not found in corpus",
            ),
            TrialCandidate(
                nct_id="NCT_EXCLUDED",
                score=0.7,
                rank=3,
                hard_excluded=True,
                excluded_reason="age 15 below minimum",
            ),
        ],
        {"NCT_VIABLE": _trial_eval("NCT_VIABLE")},
    )

    out = asyncio.run(make_rank_node(_Tools(["NCT_VIABLE", "NCT_EXCLUDED"]))(state))

    assert [r.nct_id for r in out["ranked_trials"]] == ["NCT_VIABLE", "NCT_EXCLUDED"]
    assert out["hard_excluded_fill_count"] == 1
    assert out["hard_excluded_fill_skipped_corpus_miss"] == 1


def test_hard_excluded_fill_handles_zero_viables(monkeypatch, sample_patient):
    from trial_matcher.agent.nodes import make_rank_node
    from trial_matcher.models.agent_state import TrialCandidate

    _configure_runner(monkeypatch, enabled=True, top_k=3)
    state = _state(
        sample_patient,
        [
            TrialCandidate(
                nct_id="NCT_EXCLUDED_HIGH",
                score=0.8,
                rank=1,
                hard_excluded=True,
                excluded_reason="age 15 below minimum",
            ),
            TrialCandidate(
                nct_id="NCT_EXCLUDED_LOW",
                score=0.2,
                rank=2,
                hard_excluded=True,
                excluded_reason="age 15 below minimum",
            ),
        ],
        {},
    )

    out = asyncio.run(make_rank_node(_Tools(["NCT_EXCLUDED_HIGH", "NCT_EXCLUDED_LOW"]))(state))

    assert [r.nct_id for r in out["ranked_trials"]] == [
        "NCT_EXCLUDED_HIGH",
        "NCT_EXCLUDED_LOW",
    ]
    assert all(r.hard_excluded_fill for r in out["ranked_trials"])
    assert out["hard_excluded_fill_count"] == 2


def test_judge_disabled_preserves_hard_excluded_fill_provenance(
    monkeypatch,
    sample_patient,
):
    from trial_matcher.agent.nodes import make_judge_node
    from trial_matcher.models.agent_state import initial_state
    from trial_matcher.models.ranking import RankedTrial

    monkeypatch.setenv("TRIAL_MATCHER__RUNNER__USE_LLM_JUDGE", "false")
    from trial_matcher.config import get_settings

    get_settings.cache_clear()
    state = initial_state(patient_raw="patient", run_id="p7")
    state["patient_profile"] = sample_patient
    state["ranked_trials"] = [
        RankedTrial(
            nct_id="NCT_EXCLUDED",
            rank=1,
            score=-1.999,
            eval=_trial_eval("NCT_EXCLUDED"),
            hard_excluded_fill=True,
            excluded_reason="age 15 below minimum",
        )
    ]

    out = asyncio.run(make_judge_node(judge=None, tools=_Tools([]))(state))

    judged = out["judged_top10"][0]
    assert judged.hard_excluded_fill
    assert judged.excluded_reason == "age 15 below minimum"


def test_judge_skips_llm_when_only_hard_excluded_fills(monkeypatch, sample_patient):
    from trial_matcher.agent.nodes import make_judge_node
    from trial_matcher.models.agent_state import initial_state
    from trial_matcher.models.ranking import RankedTrial

    class JudgeShouldNotRun:
        async def judge(self, *args, **kwargs):  # pragma: no cover - failure path
            raise AssertionError("judge should not run for all-fill rankings")

    monkeypatch.setenv("TRIAL_MATCHER__RUNNER__USE_LLM_JUDGE", "true")
    from trial_matcher.config import get_settings

    get_settings.cache_clear()
    state = initial_state(patient_raw="patient", run_id="p7")
    state["patient_profile"] = sample_patient
    state["ranked_trials"] = [
        RankedTrial(
            nct_id="NCT_EXCLUDED",
            rank=1,
            score=-1.999,
            eval=_trial_eval("NCT_EXCLUDED"),
            hard_excluded_fill=True,
            excluded_reason="age 15 below minimum",
        )
    ]

    out = asyncio.run(make_judge_node(JudgeShouldNotRun(), tools=_Tools([]))(state))

    assert out["judged_top10"][0].hard_excluded_fill
    assert "llm_calls" not in out


def test_apply_critique_order_preserves_hard_excluded_fill_provenance():
    from trial_matcher.agent.nodes import make_apply_critique_order_node
    from trial_matcher.models.agent_state import initial_state
    from trial_matcher.models.ranking import JudgedTrial

    state = initial_state(patient_raw="patient", run_id="p7")
    state["judged_top10"] = [
        JudgedTrial(
            nct_id="NCT_EXCLUDED",
            rank=1,
            score=-1.999,
            eval=_trial_eval("NCT_EXCLUDED"),
            hard_excluded_fill=True,
            excluded_reason="age 15 below minimum",
        )
    ]

    out = asyncio.run(make_apply_critique_order_node()(state))

    ranked = out["ranked_trials"][0]
    assert ranked.hard_excluded_fill
    assert ranked.excluded_reason == "age 15 below minimum"


def test_llm_judge_cannot_promote_hard_excluded_fills(sample_patient):
    from trial_matcher.models.ranking import RankedTrial
    from trial_matcher.ranking import llm_judge as judge_mod

    async def fake_structured_complete(*args, **kwargs):
        return judge_mod._JudgeOutput(
            ranking=[
                judge_mod._JudgedItem(nct_id="NCT_FILL", rationale="promote fill"),
                judge_mod._JudgedItem(nct_id="NCT_B", rationale="best real"),
                judge_mod._JudgedItem(nct_id="NCT_A", rationale="second real"),
            ]
        )

    ranked = [
        RankedTrial(nct_id="NCT_A", score=0.9, eval=_trial_eval("NCT_A")),
        RankedTrial(nct_id="NCT_B", score=0.8, eval=_trial_eval("NCT_B")),
        RankedTrial(
            nct_id="NCT_FILL",
            score=-2.0,
            eval=_trial_eval("NCT_FILL"),
            hard_excluded_fill=True,
            excluded_reason="age 15 below minimum",
        ),
    ]

    original = judge_mod.structured_complete
    judge_mod.structured_complete = fake_structured_complete  # type: ignore[assignment]
    try:
        judged = asyncio.run(
            judge_mod.LLMJudge(llm=object()).judge(
                sample_patient,
                ranked,
                trials_meta={},
                top_n=3,
            )
        )
    finally:
        judge_mod.structured_complete = original  # type: ignore[assignment]

    assert [j.nct_id for j in judged] == ["NCT_B", "NCT_A", "NCT_FILL"]
    assert judged[-1].hard_excluded_fill


def test_llm_judge_cannot_promote_retrieval_tail_fills(sample_patient):
    from trial_matcher.models.ranking import RankedTrial
    from trial_matcher.ranking import llm_judge as judge_mod

    async def fake_structured_complete(*args, **kwargs):
        return judge_mod._JudgeOutput(
            ranking=[
                judge_mod._JudgedItem(nct_id="NCT_TAIL", rationale="promote tail"),
                judge_mod._JudgedItem(nct_id="NCT_B", rationale="best real"),
                judge_mod._JudgedItem(nct_id="NCT_A", rationale="second real"),
            ]
        )

    ranked = [
        RankedTrial(nct_id="NCT_A", score=0.9, eval=_trial_eval("NCT_A")),
        RankedTrial(nct_id="NCT_B", score=0.8, eval=_trial_eval("NCT_B")),
        RankedTrial(
            nct_id="NCT_TAIL",
            score=-1.5,
            eval=_trial_eval("NCT_TAIL"),
            retrieval_tail_fill=True,
            excluded_reason="not evaluated; retrieval tail fill",
        ),
    ]

    original = judge_mod.structured_complete
    judge_mod.structured_complete = fake_structured_complete  # type: ignore[assignment]
    try:
        judged = asyncio.run(
            judge_mod.LLMJudge(llm=object()).judge(
                sample_patient,
                ranked,
                trials_meta={},
                top_n=3,
            )
        )
    finally:
        judge_mod.structured_complete = original  # type: ignore[assignment]

    assert [j.nct_id for j in judged] == ["NCT_B", "NCT_A", "NCT_TAIL"]
    assert judged[-1].retrieval_tail_fill
