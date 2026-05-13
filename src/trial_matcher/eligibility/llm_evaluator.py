"""LLM-based eligibility evaluator with structured chain-of-thought.

Forces the model to fill all six fields (requirement / negation_check /
temporal_check / evidence_quote / inference_gap / decision) so it cannot
skip any check. This pattern beats vanilla CoT by 5-10 F1 on TREC-style
criterion classification.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..llm.client import UnifiedLLM
from ..llm.few_shot import FewShotBank
from ..llm.prompts import COT_ELIGIBILITY_V1, COT_ELIGIBILITY_V2
from ..llm.structured import structured_complete
from ..models.criterion import Criterion
from ..models.eligibility import CriterionEval, EligibilityLabel
from ..models.patient import PatientProfile


class _CoTOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    requirement: str
    negation_check: str
    temporal_check: str
    evidence_quote: str
    inference_gap: str
    decision: Literal["met", "not_met", "NEI"]
    confidence: float = Field(ge=0.0, le=1.0)


class LLMEvaluator:
    """Single-shot LLM evaluator. Used as the second tier of the cascade.

    When ``few_shot_bank`` is provided, dynamically selects k similar examples
    by criterion text and injects them into the prompt (uses COT_ELIGIBILITY_V2).
    Without a bank, falls back to vanilla COT_ELIGIBILITY_V1.
    """

    def __init__(
        self,
        llm: UnifiedLLM | None = None,
        few_shot_bank: FewShotBank | None = None,
        encoder: Any | None = None,
        few_shot_k: int = 3,
    ):
        self.llm = llm or UnifiedLLM()
        self.few_shot_bank = few_shot_bank
        self.encoder = encoder
        self.few_shot_k = few_shot_k

    async def evaluate(
        self,
        criterion: Criterion,
        patient: PatientProfile,
        temperature: float = 0.0,
    ) -> CriterionEval:
        excerpts = self._patient_excerpts(patient)
        if self.few_shot_bank is not None and self.few_shot_bank.examples:
            examples = self.few_shot_bank.select(
                criterion.raw_text, k=self.few_shot_k, encoder=self.encoder
            )
            block = self.few_shot_bank.to_prompt_block(examples)
            prompt = COT_ELIGIBILITY_V2.format(
                criterion_annotated=criterion.annotated_text or criterion.raw_text,
                patient_excerpts=excerpts,
                few_shot_block=block,
            )
        else:
            prompt = COT_ELIGIBILITY_V1.format(
                criterion_annotated=criterion.annotated_text or criterion.raw_text,
                patient_excerpts=excerpts,
            )
        try:
            cot = await structured_complete(
                self.llm,
                prompt=prompt,
                response_model=_CoTOutput,
                model="mini",
                temperature=temperature,
                max_tokens=1500,
                max_retries=1,
                task_name="eligibility_eval",
            )
            label = EligibilityLabel(cot.decision)
        except Exception:
            return CriterionEval(
                criterion_id=criterion.id,
                label=EligibilityLabel.NEI,
                confidence=0.0,
                evidence="",
                reasoning="LLM evaluator failed",
                evaluator="llm",
                llm_calls=1,
                criterion=criterion,
            )

        reasoning = (
            f"REQUIREMENT: {cot.requirement}\n"
            f"NEGATION: {cot.negation_check}\n"
            f"TEMPORAL: {cot.temporal_check}\n"
            f"GAP: {cot.inference_gap}"
        )
        return CriterionEval(
            criterion_id=criterion.id,
            label=label,
            confidence=cot.confidence,
            evidence=cot.evidence_quote,
            reasoning=reasoning,
            evaluator="llm",
            llm_calls=1,
            criterion=criterion,
        )

    @staticmethod
    def _patient_excerpts(patient: PatientProfile, max_chars: int = 2200) -> str:
        """Render relevant patient excerpts for prompting.

        Prefers structured fields over the raw note when present.
        """
        if not patient:
            return ""
        parts: list[str] = []
        parts.append(f"Summary: {patient.summary()}")
        if patient.biomarkers:
            parts.append(
                "Biomarkers: " + ", ".join(
                    f"{b.name}={b.status}" + (f" ({b.method})" if b.method else "")
                    for b in patient.biomarkers
                )
            )
        if patient.prior_treatments:
            parts.append(
                "Prior treatments: "
                + ", ".join(
                    f"{p.name}" + (f" line {p.line}" if p.line else "")
                    for p in patient.prior_treatments
                )
            )
        if patient.comorbidities:
            parts.append(
                "Comorbidities: " + ", ".join(c.name for c in patient.comorbidities)
            )
        if patient.labs:
            parts.append(
                "Labs: "
                + "; ".join(
                    f"{l.name}={l.value} {l.units or ''}".strip()
                    for l in patient.labs
                    if l.value is not None
                )
            )
        if patient.pregnancy:
            preg = patient.pregnancy
            parts.append(
                "Pregnancy/repro: "
                + ", ".join(
                    f"{k}={v}"
                    for k, v in preg.model_dump().items()
                    if v is not None
                )
            )
        # Always include raw text last, so the model has full context if needed.
        parts.append(f"Patient note (verbatim): {patient.raw_text}")
        joined = "\n".join(parts)
        return joined[:max_chars]
