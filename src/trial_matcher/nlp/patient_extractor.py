"""Extract a structured PatientProfile from a TREC topic / patient note."""

from __future__ import annotations

from ..llm.client import UnifiedLLM
from ..llm.prompts import PATIENT_EXTRACT_V1
from ..llm.structured import structured_complete
from ..logging import logger
from ..models.patient import PatientProfile


class PatientExtractor:
    """LLM-driven structured extractor with one validation retry."""

    def __init__(self, llm: UnifiedLLM | None = None):
        self.llm = llm or UnifiedLLM()

    async def extract(self, topic_id: str, raw_text: str) -> PatientProfile:
        """Run the extractor and return a populated PatientProfile.

        Always returns a valid object — on hard failure, falls back to a profile
        whose only populated field is ``raw_text`` so retrieval can still proceed.
        """
        prompt = PATIENT_EXTRACT_V1.format(patient_text=raw_text)
        try:
            extracted = await structured_complete(
                self.llm,
                prompt=prompt,
                response_model=_RawProfile,
                model="mini",
                temperature=0.0,
                max_tokens=3000,
                max_retries=2,
                task_name="patient_extract",
            )
        except Exception as e:
            logger.warning(f"Patient extraction failed for {topic_id}: {e!r}; using minimal profile")
            return PatientProfile(topic_id=topic_id, raw_text=raw_text)

        return extracted.to_profile(topic_id=topic_id, raw_text=raw_text)


# Internal flexible model — slight relaxations vs the canonical PatientProfile so the LLM
# can populate it without unnecessary validation churn. We map to PatientProfile after.
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..models.patient import (
    Biomarker,
    Comorbidity,
    Lab,
    PatientProfile as _Profile,
    Pregnancy,
    PriorTreatment,
    Sex,
)


class _RawProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    age_years: float | None = None
    sex: str | None = None
    primary_diagnosis: str | None = None
    primary_diagnosis_stage: str | None = None
    secondary_diagnoses: list[str] = Field(default_factory=list)
    ecog: int | None = None
    karnofsky: int | None = None
    biomarkers: list[dict[str, Any]] = Field(default_factory=list)
    prior_treatments: list[dict[str, Any]] = Field(default_factory=list)
    comorbidities: list[dict[str, Any]] = Field(default_factory=list)
    labs: list[dict[str, Any]] = Field(default_factory=list)
    pregnancy: dict[str, Any] | None = None
    location: str | None = None
    free_text_residual: str | None = None

    def to_profile(self, topic_id: str, raw_text: str) -> _Profile:
        sex = Sex.UNKNOWN
        if self.sex and self.sex.lower() in {"male", "female"}:
            sex = Sex(self.sex.lower())

        def _safe(model_cls, items):
            out = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                try:
                    out.append(model_cls.model_validate(item))
                except Exception:
                    continue
            return out

        return _Profile(
            topic_id=topic_id,
            raw_text=raw_text,
            age_years=self.age_years,
            sex=sex,
            primary_diagnosis=self.primary_diagnosis,
            primary_diagnosis_stage=self.primary_diagnosis_stage,
            secondary_diagnoses=self.secondary_diagnoses or [],
            ecog=self.ecog,
            karnofsky=self.karnofsky,
            biomarkers=_safe(Biomarker, self.biomarkers),
            prior_treatments=_safe(PriorTreatment, self.prior_treatments),
            comorbidities=_safe(Comorbidity, self.comorbidities),
            labs=_safe(Lab, self.labs),
            pregnancy=Pregnancy.model_validate(self.pregnancy) if self.pregnancy else None,
            location=self.location,
            free_text_residual=self.free_text_residual,
        )
