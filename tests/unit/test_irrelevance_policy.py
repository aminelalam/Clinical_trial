from __future__ import annotations


def _patient():
    from trial_matcher.models.patient import Biomarker, PatientProfile, PriorTreatment

    return PatientProfile(
        topic_id="T1",
        raw_text="Metastatic HER2 positive breast cancer after trastuzumab.",
        primary_diagnosis="breast cancer",
        biomarkers=[Biomarker(name="HER2", status="positive")],
        prior_treatments=[PriorTreatment(name="trastuzumab", category="targeted")],
    )


def _plan():
    from trial_matcher.models.search_plan import SearchPlan

    return SearchPlan(
        primary_disease_query="breast cancer",
        expansion_terms=["HER2"],
        retrieval_priorities=["HER2 targeted therapy"],
    )


def _trial(nct_id: str, condition: str, intervention: str):
    from trial_matcher.models.trial import Eligibility, Trial

    return Trial(
        nct_id=nct_id,
        title=f"{condition} treated with {intervention}",
        brief_summary=f"Study of {intervention}",
        conditions=[condition],
        interventions=[intervention],
        eligibility=Eligibility(raw_text=f"Inclusion Criteria:\n- {condition}"),
    )


def _high_nei_eval(nct_id: str, **updates):
    from trial_matcher.models.eligibility import TrialEval, TrialLabel

    base = TrialEval(
        nct_id=nct_id,
        label=TrialLabel.EXCLUDES,
        n_inclusion=5,
        n_inclusion_met=0,
        n_inclusion_nei=5,
        fraction_nei=1.0,
    )
    return base.model_copy(update=updates)


def _candidate(nct_id: str, rank: int = 25):
    from trial_matcher.models.agent_state import TrialCandidate

    return TrialCandidate(nct_id=nct_id, score=0.1, rank=rank)


def _apply(trial, trial_eval, *, mode: str = "benchmark", retrieval_prior: float = 0.1):
    from trial_matcher.eligibility.irrelevance import apply_multisignal_irrelevance_policy

    return apply_multisignal_irrelevance_policy(
        trial=trial,
        trial_eval=trial_eval,
        candidate=_candidate(trial.nct_id),
        patient=_patient(),
        plan=_plan(),
        enabled=True,
        mode=mode,
        retrieval_prior=retrieval_prior,
        min_nei_fraction=0.8,
        max_inclusion_met=0,
        max_retrieval_prior=0.35,
        min_signal_count=4,
    )


def test_multisignal_irrelevance_converts_only_when_multiple_signals_agree():
    from trial_matcher.models.eligibility import TrialLabel

    trial = _trial("NCT_IRREL", "prostate cancer", "androgen blockade")
    result = _apply(trial, _high_nei_eval("NCT_IRREL"))

    assert result.trial_eval.label == TrialLabel.IRRELEVANT
    assert result.diagnostics["activated"] is True
    assert result.diagnostics["signal_count"] >= 4


def test_multisignal_irrelevance_does_not_convert_explicit_excludes():
    from trial_matcher.models.eligibility import TrialLabel

    trial = _trial("NCT_EXCLUDES", "prostate cancer", "androgen blockade")
    trial_eval = _high_nei_eval(
        "NCT_EXCLUDES",
        any_exclusion_met=True,
        n_exclusion=1,
        n_exclusion_met=1,
    )

    result = _apply(trial, trial_eval)

    assert result.trial_eval.label == TrialLabel.EXCLUDES
    assert result.diagnostics["blocked_reason"] == "explicit_exclusion_met"


def test_multisignal_irrelevance_does_not_convert_good_clinical_support():
    from trial_matcher.models.eligibility import TrialLabel

    trial = _trial("NCT_SUPPORT", "breast cancer", "HER2 therapy")
    result = _apply(trial, _high_nei_eval("NCT_SUPPORT"))

    assert result.trial_eval.label == TrialLabel.EXCLUDES
    assert result.diagnostics["blocked_reason"] == "insufficient_signals"
    assert result.diagnostics["signals"]["low_condition_title_support"] is False


def test_multisignal_irrelevance_requires_low_retrieval_prior():
    from trial_matcher.models.eligibility import TrialLabel

    trial = _trial("NCT_HIGH_PRIOR", "prostate cancer", "androgen blockade")
    result = _apply(
        trial,
        _high_nei_eval("NCT_HIGH_PRIOR"),
        retrieval_prior=0.9,
    )

    assert result.trial_eval.label == TrialLabel.EXCLUDES
    assert result.diagnostics["blocked_reason"] == "missing_core_irrelevance_signals"
    assert result.diagnostics["signals"]["low_retrieval_prior"] is False


def test_multisignal_irrelevance_is_benchmark_only():
    from trial_matcher.models.eligibility import TrialLabel

    trial = _trial("NCT_CLINICAL", "prostate cancer", "androgen blockade")
    result = _apply(trial, _high_nei_eval("NCT_CLINICAL"), mode="clinical_active")

    assert result.trial_eval.label == TrialLabel.EXCLUDES
    assert result.diagnostics["blocked_reason"] == "non_benchmark_mode"
