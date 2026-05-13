"""Self-critique node — final agentic pass over the top-5 ranking."""

from __future__ import annotations

from ..llm.client import UnifiedLLM
from ..llm.prompts import SELF_CRITIQUE_V1
from ..llm.structured import structured_complete
from ..logging import logger
from ..models.critique import Critique, IssueSeverity
from ..models.patient import PatientProfile
from ..models.ranking import JudgedTrial


class SelfCritic:
    """Reviews the top-5 trials, surfaces issues, and may reorder slightly."""

    def __init__(self, llm: UnifiedLLM | None = None):
        self.llm = llm or UnifiedLLM()

    async def critique(
        self,
        patient: PatientProfile,
        top10: list[JudgedTrial],
    ) -> tuple[list[JudgedTrial], Critique]:
        if not top10:
            return top10, Critique()
        top5 = top10[:5]
        block = "\n".join(
            f"{j.rank}. {j.nct_id} | NEI={j.eval.fraction_nei:.0%} | "
            f"incl-met={j.eval.n_inclusion_met}/{j.eval.n_inclusion} | "
            f"rationale: {j.rationale[:200]}"
            for j in top5
        )
        prompt = SELF_CRITIQUE_V1.format(
            patient_summary=patient.summary(),
            top5_block=block,
        )
        try:
            critique = await structured_complete(
                self.llm,
                prompt=prompt,
                response_model=Critique,
                model="large",
                temperature=0.2,
                max_tokens=1500,
                max_retries=1,
                task_name="self_critique",
            )
        except Exception as e:
            logger.warning(f"Self-critique failed: {e!r}; keeping ranking unchanged")
            return top10, Critique()

        if critique.rerank_needed:
            # IssueSeverity is a str-Enum; comparing the raw enum to "high" is
            # always False. Use the enum value (or coerce strings just in case
            # the LLM returned a plain string that bypassed validation).
            high = {
                i.trial_id
                for i in critique.issues_found
                if i.severity == IssueSeverity.HIGH
                or getattr(i.severity, "value", str(i.severity)) == "high"
            }
            # Demote any trial flagged "high" by 2 positions.
            new_order = list(top10)
            for nct in high:
                idx = next((k for k, t in enumerate(new_order) if t.nct_id == nct), None)
                if idx is not None and idx < len(new_order) - 2:
                    new_order.insert(idx + 2, new_order.pop(idx))
            for i, t in enumerate(new_order, start=1):
                t.rank = i
            return new_order, critique
        return top10, critique
