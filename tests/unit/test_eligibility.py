"""Deterministic eligibility evaluation + aggregation."""

from __future__ import annotations


def _crit(cid, type_, op, value, polarity="inclusion"):
    from trial_matcher.models.criterion import Criterion, CriterionType, Polarity, Predicate

    return Criterion(
        id=cid,
        polarity=Polarity(polarity),
        raw_text="test",
        type=CriterionType(type_),
        predicate=Predicate(op=op, feature=type_, value=value) if value is not None else None,
        is_mandatory=True,
    )


def test_age_inclusion_met(sample_patient):
    from trial_matcher.eligibility.deterministic import evaluate_deterministic
    from trial_matcher.models.eligibility import EligibilityLabel

    c = _crit("i_1", "age", ">=", 18)
    ev = evaluate_deterministic(c, sample_patient)
    assert ev is not None
    assert ev.label == EligibilityLabel.MET


def test_age_inclusion_not_met(sample_patient):
    from trial_matcher.eligibility.deterministic import evaluate_deterministic
    from trial_matcher.models.eligibility import EligibilityLabel

    c = _crit("i_1", "age", ">=", 80)
    ev = evaluate_deterministic(c, sample_patient)
    assert ev is not None
    assert ev.label == EligibilityLabel.NOT_MET


def test_sex_match(sample_patient):
    from trial_matcher.eligibility.deterministic import evaluate_deterministic
    from trial_matcher.models.criterion import Criterion, CriterionType, Polarity, Predicate
    from trial_matcher.models.eligibility import EligibilityLabel

    c = Criterion(
        id="i_2",
        polarity=Polarity.INCLUSION,
        raw_text="Female",
        type=CriterionType.SEX,
        predicate=Predicate(op="=", feature="sex", value="female"),
    )
    ev = evaluate_deterministic(c, sample_patient)
    assert ev is not None
    assert ev.label == EligibilityLabel.MET


def test_aggregator_mandatory_failure_marks_excludes(sample_patient):
    from trial_matcher.eligibility.aggregator import aggregate_to_trial_eval
    from trial_matcher.eligibility.deterministic import evaluate_deterministic
    from trial_matcher.models.eligibility import TrialLabel

    c1 = _crit("i_1", "age", ">=", 18)  # MET
    c2 = _crit("i_2", "age", ">=", 100)  # NOT_MET
    ev1 = evaluate_deterministic(c1, sample_patient)
    ev2 = evaluate_deterministic(c2, sample_patient)
    aggr = aggregate_to_trial_eval("NCTxxxxxxx", [c1, c2], [ev1, ev2])
    assert aggr.label == TrialLabel.EXCLUDES
    assert aggr.any_mandatory_inclusion_failed


def test_aggregator_eligible_when_all_met(sample_patient):
    from trial_matcher.eligibility.aggregator import aggregate_to_trial_eval
    from trial_matcher.eligibility.deterministic import evaluate_deterministic
    from trial_matcher.models.eligibility import TrialLabel

    c1 = _crit("i_1", "age", ">=", 18)
    c2 = _crit("i_2", "age", "<=", 100)
    ev1 = evaluate_deterministic(c1, sample_patient)
    ev2 = evaluate_deterministic(c2, sample_patient)
    aggr = aggregate_to_trial_eval("NCT0001", [c1, c2], [ev1, ev2])
    assert aggr.label == TrialLabel.ELIGIBLE


def test_aggregator_configurable_thresholds():
    from trial_matcher.models.criterion import Criterion, CriterionType, Polarity
    from trial_matcher.models.eligibility import CriterionEval, EligibilityLabel, TrialLabel
    from trial_matcher.eligibility.aggregator import aggregate_to_trial_eval

    c1 = Criterion(id="i_1", polarity=Polarity.INCLUSION, raw_text="x", type=CriterionType.OTHER, is_mandatory=True)
    c2 = Criterion(id="i_2", polarity=Polarity.INCLUSION, raw_text="y", type=CriterionType.OTHER, is_mandatory=False)
    c3 = Criterion(id="e_1", polarity=Polarity.EXCLUSION, raw_text="z", type=CriterionType.OTHER, is_mandatory=True)

    e1 = CriterionEval(criterion_id="i_1", label=EligibilityLabel.MET, confidence=0.9, evidence="", reasoning="", evaluator="llm", criterion=c1)
    e2 = CriterionEval(criterion_id="i_2", label=EligibilityLabel.NEI, confidence=0.3, evidence="", reasoning="", evaluator="llm", criterion=c2)
    e3 = CriterionEval(criterion_id="e_1", label=EligibilityLabel.NOT_MET, confidence=0.9, evidence="", reasoning="", evaluator="llm", criterion=c3)

    # With default thresholds (0.6), 1/2 inclusion met + NEI fraction 1/3 < 0.6 → ELIGIBLE
    aggr_default = aggregate_to_trial_eval("NCT0002", [c1, c2, c3], [e1, e2, e3])
    assert aggr_default.label == TrialLabel.ELIGIBLE

    # With stricter thresholds (min_inc_frac=0.9, max_nei_frac=0.1) → EXCLUDES
    aggr_strict = aggregate_to_trial_eval("NCT0002", [c1, c2, c3], [e1, e2, e3], min_inclusion_fraction=0.9, max_nei_fraction=0.1)
    assert aggr_strict.label == TrialLabel.EXCLUDES
