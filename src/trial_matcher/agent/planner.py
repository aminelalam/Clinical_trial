"""Search planner — produces a SearchPlan from a PatientProfile.

The planner is mode-aware: in ``benchmark`` mode it does NOT impose a
mandatory recruitment-status filter (TREC qrels include closed trials);
in ``clinical_active`` mode it requires the trial be open to enrolment.
This applies both to the LLM prompt hint and to the deterministic fallback.
"""

from __future__ import annotations

from typing import Literal

from ..config import get_settings
from ..llm.client import UnifiedLLM
from ..llm.prompts import PLAN_SEARCH_PROMPT
from ..llm.structured import structured_complete
from ..logging import logger
from ..models.patient import PatientProfile
from ..models.search_plan import SearchPlan

PlannerMode = Literal["benchmark", "clinical_active"]


def _resolve_mode(mode: PlannerMode | None) -> PlannerMode:
    if mode is not None:
        return mode
    try:
        configured = get_settings().runner.mode
    except Exception:  # pragma: no cover
        return "clinical_active"
    return configured if configured in ("benchmark", "clinical_active") else "clinical_active"


class SearchPlanner:
    """Wraps the LLM call that produces the agent's initial search plan."""

    def __init__(self, llm: UnifiedLLM | None = None):
        self.llm = llm or UnifiedLLM()

    async def plan(
        self,
        patient: PatientProfile,
        mesh_concepts_summary: str = "",
        relax: bool = False,
        mode: PlannerMode | None = None,
    ) -> SearchPlan:
        resolved = _resolve_mode(mode)
        prompt = PLAN_SEARCH_PROMPT.format(
            patient_profile_json=patient.model_dump_json(indent=2),
            mesh_concepts=mesh_concepts_summary or "(none extracted)",
        )
        if resolved == "benchmark":
            prompt += (
                "\n\nNote: benchmark_mode=true. Do NOT include a mandatory "
                "'status' filter — the evaluation set contains trials in any "
                "recruitment state. Leave mandatory_filters.status unset."
            )
        if relax:
            prompt += "\n\nNote: relax_optional_filters_hint=true (broaden the previous plan)."
        try:
            plan = await structured_complete(
                self.llm,
                prompt=prompt,
                response_model=SearchPlan,
                model="mini",
                temperature=0.2,
                max_tokens=4000,
                max_retries=2,
                task_name="search_plan",
            )
            plan.relax_optional_filters = relax
            plan.source = "llm"
            # Even if the LLM ignored the hint, strip mandatory status in benchmark.
            if resolved == "benchmark" and plan.mandatory_filters:
                plan.mandatory_filters.pop("status", None)
            return plan
        except Exception as e:
            logger.warning(f"Planner failed: {e!r}; using fallback plan (mode={resolved})")
            return self._fallback(patient, relax, resolved)

    @staticmethod
    def _fallback(patient: PatientProfile, relax: bool, mode: PlannerMode) -> SearchPlan:
        """Build a deterministic plan when the LLM call fails.

        In benchmark mode we deliberately omit the recruitment-status filter so
        the downstream hard-filter does not exclude closed trials that the
        TREC qrels may have labelled eligible.
        """
        primary = patient.primary_diagnosis or patient.raw_text[:120]
        mandatory: dict = {}
        if mode == "clinical_active":
            mandatory["status"] = (
                ["RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION"]
                if relax
                else ["RECRUITING", "NOT_YET_RECRUITING"]
            )
        return SearchPlan(
            primary_disease_query=primary,
            expansion_terms=[],
            mandatory_filters=mandatory,
            optional_filters={},
            retrieval_priorities=[],
            risk_flags=[],
            relax_optional_filters=relax,
            source="fallback",
        )
