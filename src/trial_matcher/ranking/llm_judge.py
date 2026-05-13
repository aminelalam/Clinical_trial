"""LLM-as-judge final reranker for the top-10 trials."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..llm.client import UnifiedLLM
from ..llm.prompts import LLM_JUDGE_V1
from ..llm.structured import structured_complete
from ..logging import logger
from ..models.patient import PatientProfile
from ..models.ranking import JudgedTrial, RankedTrial


class _JudgedItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    nct_id: str
    rationale: str = ""


class _JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    ranking: list[_JudgedItem] = Field(default_factory=list)


class LLMJudge:
    """Reorder the top-N candidates with a clinical-expert LLM call."""

    def __init__(self, llm: UnifiedLLM | None = None):
        self.llm = llm or UnifiedLLM()

    async def judge(
        self,
        patient: PatientProfile,
        ranked: list[RankedTrial],
        trials_meta: dict[str, dict[str, Any]],
        top_n: int = 10,
    ) -> list[JudgedTrial]:
        """Take the top-N ``ranked`` and re-order via LLM. Returns top-N JudgedTrial.

        ``trials_meta[nct_id]`` is expected to contain title/phase/status for the prompt.
        """
        candidates = ranked[:top_n]
        if not candidates:
            return []
        judgeable = [
            r
            for r in candidates
            if not r.hard_excluded_fill and not r.retrieval_tail_fill
        ]
        fills = [
            r
            for r in candidates
            if r.hard_excluded_fill or r.retrieval_tail_fill
        ]
        if not judgeable:
            return self._to_judged(candidates)

        block = self._render_block(judgeable, trials_meta)
        prompt = LLM_JUDGE_V1.format(
            patient_summary=patient.summary(),
            trials_block=block,
        )
        try:
            out = await structured_complete(
                self.llm,
                prompt=prompt,
                response_model=_JudgeOutput,
                model="large",  # GPT-4o for judging — quality > cost
                temperature=0.2,
                max_tokens=1500,
                max_retries=1,
                task_name="llm_judge",
            )
        except Exception as e:
            logger.warning(f"LLM judge failed: {e!r}; returning original order")
            return self._to_judged(judgeable + fills)

        order = [item.nct_id for item in out.ranking]
        rationales = {item.nct_id: item.rationale for item in out.ranking}

        # Sort only real evaluated trials by the LLM order. Synthetic hard-
        # excluded fills are always appended afterward and cannot be promoted.
        index = {r.nct_id: r for r in judgeable}
        seen = set()
        ordered: list[RankedTrial] = []
        for nct in order:
            if nct in index and nct not in seen:
                seen.add(nct)
                ordered.append(index[nct])
        # Append any judgeable candidates the model omitted, preserving their
        # deterministic order, then append synthetic fills last.
        for r in judgeable:
            if r.nct_id not in seen:
                ordered.append(r)
        judged = self._to_judged(ordered + fills, rationales=rationales)
        return judged[:top_n]

    @staticmethod
    def _to_judged(
        ranked: list[RankedTrial],
        rationales: dict[str, str] | None = None,
    ) -> list[JudgedTrial]:
        rationales = rationales or {}
        return [
            JudgedTrial(
                nct_id=r.nct_id,
                rank=i,
                score=r.score,
                eval=r.eval,
                rationale=rationales.get(r.nct_id, ""),
                components=r.components,
                hard_excluded_fill=r.hard_excluded_fill,
                retrieval_tail_fill=r.retrieval_tail_fill,
                excluded_reason=r.excluded_reason,
            )
            for i, r in enumerate(ranked, start=1)
        ]

    @staticmethod
    def _render_block(
        candidates: list[RankedTrial], trials_meta: dict[str, dict[str, Any]]
    ) -> str:
        lines: list[str] = []
        for i, r in enumerate(candidates, start=1):
            meta = trials_meta.get(r.nct_id, {})
            lines.append(
                f"{i}. {r.nct_id} — {meta.get('title', '?')[:120]} | "
                f"phase={meta.get('phase', 'NA')} | status={meta.get('status', 'UNKNOWN')} | "
                f"incl met/total={r.eval.n_inclusion_met}/{r.eval.n_inclusion}, "
                f"excl clear/total={r.eval.n_exclusion_not_met}/{r.eval.n_exclusion}, "
                f"NEI={r.eval.fraction_nei:.0%}"
            )
        return "\n".join(lines)
