"""Devil's advocate verifier — secondary LLM pass that tries to refute the decision.

Implements the upgrade described in the spec doc: when the original decision
has confidence < 0.85 (or any decision), run a verifier that argues for the
opposite. If the rebuttal is moderate or strong, downgrade to NEI to admit
uncertainty.

Empirically this trades a small amount of recall for a noticeable F1 gain,
because the LLM tends to be overconfident on cases that really should be NEI.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..llm.client import UnifiedLLM
from ..llm.prompts import VERIFIER_PROMPT_V1
from ..llm.structured import structured_complete
from ..models.criterion import Criterion
from ..models.eligibility import CriterionEval, EligibilityLabel
from ..models.patient import PatientProfile
from .llm_evaluator import LLMEvaluator


class _VerifierOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    rebuttal_strength: Literal["strong", "moderate", "weak", "none"]
    rebuttal_reason: str
    revised_label: Literal["met", "not_met", "NEI"]


class EligibilityVerifier:
    def __init__(self, llm: UnifiedLLM | None = None, threshold: float = 0.85):
        self.llm = llm or UnifiedLLM()
        self.threshold = threshold

    async def verify(
        self,
        criterion: Criterion,
        patient: PatientProfile,
        original: CriterionEval,
    ) -> CriterionEval:
        """Run verification only when original confidence is below threshold."""
        if original.confidence >= self.threshold:
            return original

        prompt = VERIFIER_PROMPT_V1.format(
            label=original.label.value,
            confidence=original.confidence,
            criterion_annotated=criterion.annotated_text or criterion.raw_text,
            patient_excerpts=LLMEvaluator._patient_excerpts(patient),
            reasoning=original.reasoning or "(no reasoning available)",
        )
        try:
            out = await structured_complete(
                self.llm,
                prompt=prompt,
                response_model=_VerifierOutput,
                model="mini",
                temperature=0.2,
                max_tokens=1200,
                max_retries=1,
                task_name="eligibility_verifier",
            )
        except Exception:
            return original

        # Aggregation rule: if rebuttal is moderate/strong AND revised differs,
        # mark as NEI to admit uncertainty.
        if out.rebuttal_strength in {"moderate", "strong"} and out.revised_label != original.label.value:
            new_label = EligibilityLabel.NEI
            return CriterionEval(
                criterion_id=criterion.id,
                label=new_label,
                confidence=min(original.confidence, 0.5),
                evidence=original.evidence,
                reasoning=original.reasoning + f"\nVERIFIER: {out.rebuttal_reason}",
                evaluator="verifier",
                llm_calls=original.llm_calls + 1,
                rebuttal=out.rebuttal_reason,
                flipped_by_verifier=True,
                criterion=criterion,
            )

        # Weak rebuttal — keep original but record the rebuttal for transparency.
        return CriterionEval(
            criterion_id=criterion.id,
            label=original.label,
            confidence=original.confidence,
            evidence=original.evidence,
            reasoning=original.reasoning + f"\nVERIFIER (weak): {out.rebuttal_reason}",
            evaluator=original.evaluator,
            llm_calls=original.llm_calls + 1,
            rebuttal=out.rebuttal_reason,
            flipped_by_verifier=False,
            criterion=criterion,
        )
