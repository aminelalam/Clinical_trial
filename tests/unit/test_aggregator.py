"""Aggregator rules: mapping per-criterion evals to TREC qrels."""

from __future__ import annotations


def test_exclusion_met_marks_trial_excludes(sample_patient):
    from trial_matcher.eligibility.aggregator import aggregate_to_trial_eval
    from trial_matcher.models.criterion import Criterion, CriterionType, Polarity
    from trial_matcher.models.eligibility import CriterionEval, EligibilityLabel, TrialLabel

    incl = Criterion(id="i_1", polarity=Polarity.INCLUSION, raw_text="adult", type=CriterionType.AGE)
    excl = Criterion(id="e_1", polarity=Polarity.EXCLUSION, raw_text="pregnant", type=CriterionType.PREGNANCY)
    evals = [
        CriterionEval(criterion_id="i_1", label=EligibilityLabel.MET, criterion=incl),
        CriterionEval(criterion_id="e_1", label=EligibilityLabel.MET, criterion=excl),
    ]
    aggr = aggregate_to_trial_eval("NCT0001", [incl, excl], evals)
    assert aggr.label == TrialLabel.EXCLUDES
    assert aggr.any_exclusion_met
    assert aggr.trec_qrel == 1


def test_all_met_inclusions_clear_exclusions_eligible():
    from trial_matcher.eligibility.aggregator import aggregate_to_trial_eval
    from trial_matcher.models.criterion import Criterion, CriterionType, Polarity
    from trial_matcher.models.eligibility import CriterionEval, EligibilityLabel, TrialLabel

    incl = Criterion(id="i_1", polarity=Polarity.INCLUSION, raw_text="adult", type=CriterionType.AGE)
    excl = Criterion(id="e_1", polarity=Polarity.EXCLUSION, raw_text="pregnant", type=CriterionType.PREGNANCY)
    evals = [
        CriterionEval(criterion_id="i_1", label=EligibilityLabel.MET, criterion=incl),
        CriterionEval(criterion_id="e_1", label=EligibilityLabel.NOT_MET, criterion=excl),
    ]
    aggr = aggregate_to_trial_eval("NCT0002", [incl, excl], evals)
    assert aggr.label == TrialLabel.ELIGIBLE
    assert aggr.trec_qrel == 2


def test_zero_inclusion_support_high_nei_marks_irrelevant():
    from trial_matcher.eligibility.aggregator import aggregate_to_trial_eval
    from trial_matcher.models.criterion import Criterion, CriterionType, Polarity
    from trial_matcher.models.eligibility import CriterionEval, EligibilityLabel, TrialLabel

    criteria = [
        Criterion(id=f"i_{i}", polarity=Polarity.INCLUSION, raw_text=f"criterion {i}", type=CriterionType.OTHER)
        for i in range(1, 6)
    ]
    evals = [
        CriterionEval(criterion_id=c.id, label=EligibilityLabel.NEI, criterion=c)
        for c in criteria
    ]

    aggr = aggregate_to_trial_eval(
        "NCT_IRREL",
        criteria,
        evals,
        use_irrelevance_heuristic=True,
    )

    assert aggr.label == TrialLabel.IRRELEVANT
    assert aggr.trec_qrel == 0


def test_irrelevance_heuristic_can_be_disabled():
    from trial_matcher.eligibility.aggregator import aggregate_to_trial_eval
    from trial_matcher.models.criterion import Criterion, CriterionType, Polarity
    from trial_matcher.models.eligibility import CriterionEval, EligibilityLabel, TrialLabel

    criteria = [
        Criterion(id=f"i_{i}", polarity=Polarity.INCLUSION, raw_text=f"criterion {i}", type=CriterionType.OTHER)
        for i in range(1, 6)
    ]
    evals = [
        CriterionEval(criterion_id=c.id, label=EligibilityLabel.NEI, criterion=c)
        for c in criteria
    ]

    aggr = aggregate_to_trial_eval(
        "NCT_EXCLUDES",
        criteria,
        evals,
        use_irrelevance_heuristic=False,
    )

    assert aggr.label == TrialLabel.EXCLUDES


def test_irrelevance_heuristic_is_off_by_default(monkeypatch):
    from trial_matcher.config import get_settings
    from trial_matcher.eligibility.aggregator import aggregate_to_trial_eval
    from trial_matcher.models.criterion import Criterion, CriterionType, Polarity
    from trial_matcher.models.eligibility import CriterionEval, EligibilityLabel, TrialLabel

    monkeypatch.delenv("TRIAL_MATCHER__RUNNER__USE_IRRELEVANCE_HEURISTIC", raising=False)
    get_settings.cache_clear()
    criteria = [
        Criterion(id=f"i_{i}", polarity=Polarity.INCLUSION, raw_text=f"criterion {i}", type=CriterionType.OTHER)
        for i in range(1, 6)
    ]
    evals = [
        CriterionEval(criterion_id=c.id, label=EligibilityLabel.NEI, criterion=c)
        for c in criteria
    ]

    aggr = aggregate_to_trial_eval("NCT_DEFAULT", criteria, evals)

    assert aggr.label == TrialLabel.EXCLUDES
    get_settings.cache_clear()


def test_benchmark_mode_uses_calibrated_partial_eligibility_thresholds(monkeypatch):
    from trial_matcher.config import get_settings
    from trial_matcher.eligibility.aggregator import aggregate_to_trial_eval
    from trial_matcher.models.criterion import Criterion, CriterionType, Polarity
    from trial_matcher.models.eligibility import CriterionEval, EligibilityLabel, TrialLabel

    monkeypatch.setenv("TRIAL_MATCHER__RUNNER__MODE", "benchmark")
    get_settings.cache_clear()
    criteria = [
        Criterion(id=f"i_{i}", polarity=Polarity.INCLUSION, raw_text=f"criterion {i}", type=CriterionType.OTHER)
        for i in range(1, 6)
    ]
    evals = [
        CriterionEval(criterion_id=criteria[0].id, label=EligibilityLabel.MET, criterion=criteria[0]),
        *[
            CriterionEval(criterion_id=c.id, label=EligibilityLabel.NEI, criterion=c)
            for c in criteria[1:]
        ],
    ]

    aggr = aggregate_to_trial_eval("NCT_BENCHMARK_CAL", criteria, evals)

    assert aggr.label == TrialLabel.ELIGIBLE
    get_settings.cache_clear()


def test_clinical_mode_keeps_stricter_partial_eligibility_thresholds(monkeypatch):
    from trial_matcher.config import get_settings
    from trial_matcher.eligibility.aggregator import aggregate_to_trial_eval
    from trial_matcher.models.criterion import Criterion, CriterionType, Polarity
    from trial_matcher.models.eligibility import CriterionEval, EligibilityLabel, TrialLabel

    monkeypatch.setenv("TRIAL_MATCHER__RUNNER__MODE", "clinical_active")
    get_settings.cache_clear()
    criteria = [
        Criterion(id=f"i_{i}", polarity=Polarity.INCLUSION, raw_text=f"criterion {i}", type=CriterionType.OTHER)
        for i in range(1, 6)
    ]
    evals = [
        CriterionEval(criterion_id=criteria[0].id, label=EligibilityLabel.MET, criterion=criteria[0]),
        *[
            CriterionEval(criterion_id=c.id, label=EligibilityLabel.NEI, criterion=c)
            for c in criteria[1:]
        ],
    ]

    aggr = aggregate_to_trial_eval("NCT_CLINICAL_STRICT", criteria, evals)

    assert aggr.label == TrialLabel.EXCLUDES
    get_settings.cache_clear()


def test_explicit_exclusion_met_wins_over_irrelevance_heuristic():
    from trial_matcher.eligibility.aggregator import aggregate_to_trial_eval
    from trial_matcher.models.criterion import Criterion, CriterionType, Polarity
    from trial_matcher.models.eligibility import CriterionEval, EligibilityLabel, TrialLabel

    incl = Criterion(id="i_1", polarity=Polarity.INCLUSION, raw_text="rare biomarker", type=CriterionType.BIOMARKER)
    excl = Criterion(id="e_1", polarity=Polarity.EXCLUSION, raw_text="pregnant", type=CriterionType.PREGNANCY)
    evals = [
        CriterionEval(criterion_id="i_1", label=EligibilityLabel.NEI, criterion=incl),
        CriterionEval(criterion_id="e_1", label=EligibilityLabel.MET, criterion=excl),
    ]

    aggr = aggregate_to_trial_eval(
        "NCT_EXPLICIT_EXCLUSION",
        [incl, excl],
        evals,
        use_irrelevance_heuristic=True,
    )

    assert aggr.label == TrialLabel.EXCLUDES
