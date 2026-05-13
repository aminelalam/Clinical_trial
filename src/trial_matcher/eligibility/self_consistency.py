"""Self-consistency: k=3 samples + majority vote, applied conditionally."""

from __future__ import annotations

import asyncio
from collections import Counter

from ..config import get_settings
from ..models.criterion import Criterion
from ..models.eligibility import CriterionEval, EligibilityLabel
from ..models.patient import PatientProfile
from .llm_evaluator import LLMEvaluator


async def self_consistency_eval(
    criterion: Criterion,
    patient: PatientProfile,
    evaluator: LLMEvaluator,
    k: int | None = None,
    temperature: float | None = None,
) -> CriterionEval:
    """Run k LLM evaluations and return the majority decision.

    On a tie or no majority, returns NEI to be conservative.
    """
    s = get_settings()
    k = k or s.llm.sc_k_samples
    temperature = temperature if temperature is not None else s.llm.temperature_sc

    runs: list[CriterionEval] = await asyncio.gather(
        *[evaluator.evaluate(criterion, patient, temperature=temperature) for _ in range(k)]
    )

    counts = Counter(r.label for r in runs)
    most_common = counts.most_common()
    top_label, top_n = most_common[0]
    if top_n <= k // 2:
        # No majority — be conservative.
        return CriterionEval(
            criterion_id=criterion.id,
            label=EligibilityLabel.NEI,
            confidence=0.4,
            evidence="; ".join(r.evidence for r in runs if r.evidence)[:500],
            reasoning=f"Self-consistency (k={k}) produced no majority: {dict(counts)}",
            evaluator="self_consistency",
            llm_calls=k,
            criterion=criterion,
        )

    # Majority found — average confidence across runs that voted with the majority.
    majority_runs = [r for r in runs if r.label == top_label]
    avg_conf = sum(r.confidence for r in majority_runs) / len(majority_runs)
    evidence = next(
        (r.evidence for r in majority_runs if r.evidence),
        majority_runs[0].evidence,
    )
    reasoning = (
        f"Self-consistency (k={k}) majority={top_label.value} "
        f"({top_n}/{k}); evidence: {evidence}"
    )
    return CriterionEval(
        criterion_id=criterion.id,
        label=top_label,
        confidence=avg_conf,
        evidence=evidence,
        reasoning=reasoning,
        evaluator="self_consistency",
        llm_calls=k,
        criterion=criterion,
    )
