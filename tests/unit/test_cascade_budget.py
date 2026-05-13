"""Eligibility cascade budget controls."""

from __future__ import annotations

import asyncio

from trial_matcher.eligibility.cascade import EligibilityCascade
from trial_matcher.models.criterion import Criterion, CriterionType, Polarity
from trial_matcher.models.eligibility import CriterionEval, EligibilityLabel


class _FakeEvaluator:
    def __init__(self):
        self.calls = 0

    async def evaluate(self, criterion, patient, temperature=0.0):
        self.calls += 1
        return CriterionEval(
            criterion_id=criterion.id,
            label=EligibilityLabel.MET,
            confidence=0.8,
            evidence="synthetic",
            reasoning="synthetic",
            evaluator="llm",
            llm_calls=1,
            criterion=criterion,
        )


def _other_criterion() -> Criterion:
    return Criterion(
        id="i_other",
        polarity=Polarity.INCLUSION,
        raw_text="Adequate organ function.",
        type=CriterionType.OTHER,
    )


def test_other_criteria_use_single_shot_when_self_consistency_disabled(sample_patient):
    cascade = EligibilityCascade(
        llm=object(),
        use_verifier=False,
        use_self_consistency=False,
    )
    fake = _FakeEvaluator()
    cascade.evaluator = fake

    out = asyncio.run(cascade.evaluate_one(_other_criterion(), sample_patient))

    assert out.evaluator == "llm"
    assert out.llm_calls == 1
    assert fake.calls == 1


def test_other_criteria_use_self_consistency_by_default(sample_patient):
    cascade = EligibilityCascade(
        llm=object(),
        use_verifier=False,
        use_self_consistency=True,
    )
    fake = _FakeEvaluator()
    cascade.evaluator = fake

    out = asyncio.run(cascade.evaluate_one(_other_criterion(), sample_patient))

    assert out.evaluator == "self_consistency"
    assert out.llm_calls == 3
    assert fake.calls == 3


def test_section_headers_do_not_call_llm(sample_patient):
    cascade = EligibilityCascade(
        llm=object(),
        use_verifier=False,
        use_self_consistency=True,
    )
    fake = _FakeEvaluator()
    cascade.evaluator = fake
    criterion = Criterion(
        id="h_1",
        polarity=Polarity.INCLUSION,
        raw_text="Disease Characteristics:",
        type=CriterionType.SECTION_HEADER,
    )

    out = asyncio.run(cascade.evaluate_one(criterion, sample_patient))

    assert out.label == EligibilityLabel.NEI
    assert out.evaluator == "deterministic"
    assert out.llm_calls == 0
    assert fake.calls == 0
