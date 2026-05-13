"""Scoring function unit tests."""

from __future__ import annotations


def _build_eval(
    nct_id="NCTxxxxxxx",
    n_inc=4,
    n_inc_met=4,
    n_exc=2,
    n_exc_met=0,
    nei_inc=0,
    nei_exc=0,
    mandatory_failed=False,
    exclusion_met=False,
):
    from trial_matcher.models.eligibility import CriterionEval, EligibilityLabel, TrialEval, TrialLabel

    return TrialEval(
        nct_id=nct_id,
        label=TrialLabel.ELIGIBLE if not mandatory_failed else TrialLabel.EXCLUDES,
        criteria=[
            CriterionEval(
                criterion_id=f"x_{i}", label=EligibilityLabel.MET, confidence=0.9
            )
            for i in range(n_inc_met)
        ],
        n_inclusion=n_inc,
        n_exclusion=n_exc,
        n_inclusion_met=n_inc_met,
        n_inclusion_nei=nei_inc,
        n_exclusion_met=n_exc_met,
        n_exclusion_not_met=n_exc - n_exc_met - nei_exc,
        n_exclusion_nei=nei_exc,
        any_mandatory_inclusion_failed=mandatory_failed,
        any_exclusion_met=exclusion_met,
        fraction_nei=(nei_inc + nei_exc) / max(1, n_inc + n_exc),
    )


def test_clinical_active_mandatory_veto_yields_negative_score(sample_trial):
    from trial_matcher.ranking.scorer import score_trial

    eval_ = _build_eval(mandatory_failed=True)
    r = score_trial(sample_trial, eval_, mode="clinical_active")
    assert r.score == -1.0
    assert r.components["mandatory_veto"]


def test_hard_veto_uses_tiny_retrieval_tie_break(sample_trial):
    from trial_matcher.ranking.scorer import score_trial

    eval_ = _build_eval(mandatory_failed=True)
    weak = score_trial(sample_trial, eval_, mode="benchmark", retrieval_prior=0.0)
    strong = score_trial(sample_trial, eval_, mode="benchmark", retrieval_prior=1.0)

    assert weak.score == -1.0
    assert -1.0 < strong.score < -0.99
    assert strong.score > weak.score
    assert strong.components["retrieval_prior"] == 1.0


def test_benchmark_mandatory_veto_is_soft_penalty(sample_trial):
    from trial_matcher.ranking.scorer import score_trial

    clean = score_trial(sample_trial, _build_eval(), mode="benchmark")
    vetoed = score_trial(
        sample_trial,
        _build_eval(mandatory_failed=True),
        mode="benchmark",
        benchmark_soft_veto=True,
    )
    assert vetoed.components["mandatory_veto"]
    assert vetoed.score > -1.0
    assert clean.score > vetoed.score


def test_full_match_score_is_positive(sample_trial):
    from trial_matcher.ranking.scorer import score_trial

    eval_ = _build_eval()
    r = score_trial(sample_trial, eval_)
    assert r.score > 0
    assert not r.components["mandatory_veto"]


def test_retrieval_prior_increases_non_veto_score(sample_trial):
    from trial_matcher.ranking.scorer import score_trial

    eval_ = _build_eval()
    low = score_trial(sample_trial, eval_, mode="benchmark", retrieval_prior=0.0)
    high = score_trial(sample_trial, eval_, mode="benchmark", retrieval_prior=1.0)

    assert high.score > low.score
    assert high.components["retrieval_prior"] == 1.0


def test_high_nei_lowers_score(sample_trial):
    from trial_matcher.ranking.scorer import score_trial

    clean = _build_eval()
    noisy = _build_eval(n_inc_met=2, nei_inc=2)
    r_clean = score_trial(sample_trial, clean)
    r_noisy = score_trial(sample_trial, noisy)
    assert r_clean.score > r_noisy.score
