"""Tests for cascade verifier policy (B10).

Policy: the verifier should be skipped when self-consistency already produced
a confident majority (avg_conf >= 0.7) and when SC produced no majority.
"""

from __future__ import annotations

from trial_matcher.eligibility.cascade import EligibilityCascade
from trial_matcher.models.eligibility import CriterionEval, EligibilityLabel
from trial_matcher.models.criterion import Criterion, CriterionType, Polarity


def _criterion(cid="c1", ctype=CriterionType.BIOMARKER):
    return Criterion(
        id=cid,
        polarity=Polarity.INCLUSION,
        raw_text="HER2 positive",
        type=ctype,
        is_mandatory=True,
    )


def _eval(label=EligibilityLabel.MET, confidence=0.9, evaluator="llm"):
    return CriterionEval(
        criterion_id="c1",
        label=label,
        confidence=confidence,
        evidence="test",
        reasoning="test",
        evaluator=evaluator,
        criterion=_criterion(),
    )


class TestNeedsVerifier:
    def test_skip_verifier_when_sc_confident_majority(self):
        evald = _eval(label=EligibilityLabel.MET, confidence=0.85, evaluator="self_consistency")
        assert EligibilityCascade._needs_verifier(evald, sc_ran=True) is False

    def test_run_verifier_after_sc_low_confidence(self):
        evald = _eval(label=EligibilityLabel.MET, confidence=0.55, evaluator="self_consistency")
        assert EligibilityCascade._needs_verifier(evald, sc_ran=True) is True

    def test_skip_verifier_when_sc_no_majority(self):
        evald = _eval(label=EligibilityLabel.NEI, confidence=0.4, evaluator="self_consistency")
        assert EligibilityCascade._needs_verifier(evald, sc_ran=True) is False

    def test_run_verifier_after_single_shot_low_conf(self):
        evald = _eval(label=EligibilityLabel.NOT_MET, confidence=0.5, evaluator="llm")
        assert EligibilityCascade._needs_verifier(evald, sc_ran=False) is True

    def test_run_verifier_after_single_shot_high_conf(self):
        evald = _eval(label=EligibilityLabel.MET, confidence=0.92, evaluator="llm")
        assert EligibilityCascade._needs_verifier(evald, sc_ran=False) is True

    def test_skip_verifier_sc_confidence_exactly_threshold(self):
        evald = _eval(label=EligibilityLabel.MET, confidence=0.70, evaluator="self_consistency")
        assert EligibilityCascade._needs_verifier(evald, sc_ran=True) is False