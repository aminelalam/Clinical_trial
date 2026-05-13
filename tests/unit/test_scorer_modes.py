"""Mode-aware scorer behaviour (B1, B6).

Validates that:
- Benchmark mode does NOT bury COMPLETED/TERMINATED trials.
- Clinical-active mode keeps the existing penalty.
- Geographic scoring no longer matches "Spain" inside "New Spain".
"""

from __future__ import annotations

import pytest


def _eligible_eval():
    from trial_matcher.models.eligibility import TrialEval, TrialLabel

    return TrialEval(
        nct_id="NCT00000001",
        label=TrialLabel.ELIGIBLE,
        criteria=[],
        n_inclusion=2,
        n_exclusion=1,
        n_inclusion_met=2,
        n_exclusion_not_met=1,
        any_mandatory_inclusion_failed=False,
        any_exclusion_met=False,
        fraction_nei=0.0,
    )


def _trial(status, phase=None):
    from trial_matcher.models.trial import (
        AgeRange,
        Eligibility,
        Phase,
        Sex,
        Trial,
    )

    return Trial(
        nct_id="NCT00000001",
        title="t",
        brief_summary="",
        conditions=[],
        phase=phase or Phase.PHASE_2,
        status=status,
        interventional=True,
        eligibility=Eligibility(
            raw_text="",
            inclusion_text="",
            exclusion_text="",
            age_range=AgeRange(),
            sex=Sex.ALL,
        ),
    )


def test_benchmark_mode_does_not_punish_completed():
    from trial_matcher.models.trial import RecruitmentStatus
    from trial_matcher.ranking.scorer import score_trial

    ev = _eligible_eval()
    trial = _trial(RecruitmentStatus.COMPLETED)
    ranked = score_trial(trial, ev, mode="benchmark")
    assert ranked.score > 0, (
        f"Benchmark mode must keep COMPLETED trials competitive, got score={ranked.score}"
    )
    assert ranked.components["recruiting_bonus"] == 0.0


def test_benchmark_mode_active_not_recruiting_gets_partial_bonus():
    from trial_matcher.models.trial import RecruitmentStatus
    from trial_matcher.ranking.scorer import score_trial

    ev = _eligible_eval()
    trial = _trial(RecruitmentStatus.ACTIVE_NOT_RECRUITING)
    ranked = score_trial(trial, ev, mode="benchmark")
    assert ranked.components["recruiting_bonus"] == pytest.approx(0.3)


def test_clinical_active_mode_punishes_completed():
    from trial_matcher.models.trial import RecruitmentStatus
    from trial_matcher.ranking.scorer import score_trial

    ev = _eligible_eval()
    trial = _trial(RecruitmentStatus.COMPLETED)
    ranked = score_trial(trial, ev, mode="clinical_active")
    assert ranked.components["recruiting_bonus"] == -1.0


def test_clinical_active_recruiting_outranks_completed():
    from trial_matcher.models.trial import RecruitmentStatus
    from trial_matcher.ranking.scorer import score_trial

    ev = _eligible_eval()
    rec = _trial(RecruitmentStatus.RECRUITING)
    comp = _trial(RecruitmentStatus.COMPLETED)
    r = score_trial(rec, ev, mode="clinical_active")
    c = score_trial(comp, ev, mode="clinical_active")
    assert r.score > c.score


def test_geography_does_not_match_substring_false_positive():
    from trial_matcher.models.trial import (
        AgeRange,
        Eligibility,
        Location,
        Phase,
        RecruitmentStatus,
        Sex,
        Trial,
    )
    from trial_matcher.ranking.scorer import _geography_score

    # Patient location is "Spain". Trial has a single site whose city is the
    # historical placename "New Spain" and country is Mexico. The old substring
    # logic returned 1.0 here; now it should return 0.5 at most (city/state
    # substring on either side) and never the country-equality 1.0.
    trial = Trial(
        nct_id="NCT00000002",
        title="t",
        brief_summary="",
        conditions=[],
        phase=Phase.PHASE_2,
        status=RecruitmentStatus.RECRUITING,
        interventional=True,
        eligibility=Eligibility(age_range=AgeRange(), sex=Sex.ALL),
        locations=[Location(country="Mexico", state=None, city="New Spain")],
    )
    score = _geography_score(trial, "Spain")
    assert score < 1.0


def test_geography_country_alias_match():
    from trial_matcher.models.trial import (
        AgeRange,
        Eligibility,
        Location,
        Phase,
        RecruitmentStatus,
        Sex,
        Trial,
    )
    from trial_matcher.ranking.scorer import _geography_score

    trial = Trial(
        nct_id="NCT00000003",
        title="t",
        brief_summary="",
        conditions=[],
        phase=Phase.PHASE_2,
        status=RecruitmentStatus.RECRUITING,
        interventional=True,
        eligibility=Eligibility(age_range=AgeRange(), sex=Sex.ALL),
        locations=[Location(country="USA", city="Boston")],
    )
    assert _geography_score(trial, "United States") == 1.0
