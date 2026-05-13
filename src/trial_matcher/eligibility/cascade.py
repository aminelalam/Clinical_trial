"""Eligibility cascade router.

Routes each criterion to:
  1. deterministic (free, exact) — age / sex / numeric labs / ECOG
  2. LLM single-shot — biomarker / prior_treatment / comorbidity / pregnancy
  3. self-consistency (k=3) — type "other" OR confidence < threshold
  4. verifier (devil's advocate) — final pass on borderline cases

The router is the single piece of code where T2 (Micro-F1) is decided, so it
is small on purpose: every branch is explicit and testable.
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterable

from ..config import get_settings
from ..llm.client import UnifiedLLM
from ..llm.few_shot import FewShotBank
from ..models.criterion import Criterion, CriterionType
from ..models.eligibility import CriterionEval, EligibilityLabel
from ..models.patient import PatientProfile
from .deterministic import evaluate_deterministic
from .llm_evaluator import LLMEvaluator
from .self_consistency import self_consistency_eval
from .verifier import EligibilityVerifier


_LLM_TYPES = {
    CriterionType.BIOMARKER,
    CriterionType.PRIOR_TREATMENT,
    CriterionType.COMORBIDITY,
    CriterionType.PREGNANCY,
    CriterionType.DIAGNOSIS,
    CriterionType.CONSENT,
}


class EligibilityCascade:
    """Orchestrates the cascade for a list of criteria."""

    def __init__(
        self,
        llm: UnifiedLLM | None = None,
        use_verifier: bool = True,
        use_self_consistency: bool = True,
        sc_confidence_threshold: float | None = None,
        few_shot_bank: FewShotBank | None = None,
        encoder: Any | None = None,
        few_shot_k: int = 3,
    ):
        s = get_settings()
        self.llm = llm or UnifiedLLM()
        self.evaluator = LLMEvaluator(
            self.llm,
            few_shot_bank=few_shot_bank,
            encoder=encoder,
            few_shot_k=few_shot_k,
        )
        self.verifier = EligibilityVerifier(self.llm) if use_verifier else None
        self.use_self_consistency = use_self_consistency
        self.sc_threshold = (
            sc_confidence_threshold
            if sc_confidence_threshold is not None
            else s.llm.sc_confidence_threshold
        )

    async def evaluate_one(self, criterion: Criterion, patient: PatientProfile) -> CriterionEval:
        if criterion.type == CriterionType.SECTION_HEADER:
            return CriterionEval(
                criterion_id=criterion.id,
                label=EligibilityLabel.NEI,
                confidence=1.0,
                evidence="",
                reasoning="Section header, not an eligibility requirement.",
                evaluator="deterministic",
                llm_calls=0,
                criterion=criterion,
            )

        # 1. Deterministic
        det = evaluate_deterministic(criterion, patient)
        if det is not None and det.label != EligibilityLabel.NEI:
            return det

        # If deterministic returned NEI but the criterion is purely numeric, trust it.
        if det is not None and criterion.type in {CriterionType.AGE, CriterionType.SEX}:
            return det

        # 2. Type-targeted LLM evaluation
        if criterion.type in _LLM_TYPES or criterion.type == CriterionType.PERFORMANCE_STATUS:
            evald = await self.evaluator.evaluate(criterion, patient)
            sc_ran = False
            if self.use_self_consistency and evald.confidence < self.sc_threshold:
                evald = await self._maybe_self_consistency(criterion, patient, evald)
                sc_ran = True
            if self.verifier and self._needs_verifier(evald, sc_ran):
                evald = await self.verifier.verify(criterion, patient, evald)
            return evald

        # 3. Type "other" or LAB without parsable predicate. The default uses
        # self-consistency because these criteria are heterogeneous; evaluation
        # runs can disable it to trade some robustness for a much lower budget.
        if self.use_self_consistency:
            evald = await self_consistency_eval(criterion, patient, self.evaluator)
            sc_ran = True
        else:
            evald = await self.evaluator.evaluate(criterion, patient)
            sc_ran = False
        if self.verifier and self._needs_verifier(evald, sc_ran=sc_ran):
            evald = await self.verifier.verify(criterion, patient, evald)
        return evald

    @staticmethod
    def _needs_verifier(evald: CriterionEval, sc_ran: bool) -> bool:
        """Decide whether the verifier should run after the current evaluation.

        Policy:
        - If SC produced a confident majority (avg_conf >= 0.7), the verifier
          is redundant — skip it to save cost.
        - If SC produced no majority (label=NEI from tie), the verifier
          cannot improve confidence — skip it.
        - Otherwise (SC with avg_conf < 0.7, or single-shot LLM with
          confidence below the verifier threshold), run the verifier.
        """
        if sc_ran and evald.confidence >= 0.7:
            return False
        if sc_ran and evald.evaluator == "self_consistency" and evald.label == EligibilityLabel.NEI:
            return False
        return True

    async def _maybe_self_consistency(
        self,
        criterion: Criterion,
        patient: PatientProfile,
        evald: CriterionEval,
    ) -> CriterionEval:
        if evald.confidence < self.sc_threshold:
            return await self_consistency_eval(criterion, patient, self.evaluator)
        return evald

    async def evaluate_many(
        self,
        criteria: Iterable[Criterion],
        patient: PatientProfile,
        concurrency: int = 8,
        per_criterion_timeout: float = 90.0,
    ) -> list[CriterionEval]:
        """Evaluate a list of criteria with bounded concurrency and a per-criterion
        timeout so a single hung LLM call cannot block the semaphore indefinitely.

        On timeout the criterion is returned as NEI with confidence 0 — a
        conservative default that the verifier may further demote.
        """
        sem = asyncio.Semaphore(concurrency)

        async def _bounded(c: Criterion) -> CriterionEval:
            async with sem:
                try:
                    return await asyncio.wait_for(
                        self.evaluate_one(c, patient),
                        timeout=per_criterion_timeout,
                    )
                except asyncio.TimeoutError:
                    return CriterionEval(
                        criterion_id=c.id,
                        label=EligibilityLabel.NEI,
                        confidence=0.0,
                        evidence="",
                        reasoning=(
                            f"timeout after {per_criterion_timeout}s; "
                            "marked NEI to avoid blocking pipeline"
                        ),
                        evaluator="llm",
                        criterion=c,
                    )

        return await asyncio.gather(*[_bounded(c) for c in criteria])
