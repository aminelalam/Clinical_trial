"""PatientProfile and related typed components."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Sex(str, Enum):
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class Lab(BaseModel):
    """A single lab measurement extracted from the patient note."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="Canonical lab name, e.g. 'ANC', 'creatinine', 'HbA1c'")
    value: float | None = None
    units: str | None = None
    measured_on: date | None = Field(
        default=None, description="Date of measurement if known; None if patient note doesn't say"
    )
    note: str | None = None


class Biomarker(BaseModel):
    """A molecular / biomarker status."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="e.g. 'HER2', 'EGFR', 'BRCA1', 'PD-L1'")
    status: str = Field(
        ..., description="e.g. 'positive', 'negative', 'mutated', 'amplified', 'overexpressed'"
    )
    value: str | None = None
    method: str | None = Field(default=None, description="e.g. 'IHC 3+', 'FISH', 'NGS'")


class PriorTreatment(BaseModel):
    """A prior therapy line / treatment exposure."""

    model_config = ConfigDict(extra="ignore")

    name: str
    category: str | None = Field(
        default=None,
        description="e.g. 'chemotherapy', 'immunotherapy', 'targeted', 'radiation', 'surgery'",
    )
    line: int | None = Field(default=None, description="1=first line, 2=second line, etc.")
    response: str | None = None
    ended_on: date | None = None
    ongoing: bool = False


class Comorbidity(BaseModel):
    """A comorbidity / concurrent condition."""

    model_config = ConfigDict(extra="ignore")

    name: str
    active: bool | None = None
    onset: date | None = None
    note: str | None = None


class Pregnancy(BaseModel):
    """Pregnancy / reproductive status (relevant for many oncology trials)."""

    model_config = ConfigDict(extra="ignore")

    pregnant: bool | None = None
    breastfeeding: bool | None = None
    childbearing_potential: bool | None = None
    contraception: str | None = None


class PatientProfile(BaseModel):
    """Structured representation of a patient note or TREC topic.

    The ``free_text_residual`` field captures any clinically relevant content the
    extractor could not place into a typed slot. It feeds back into LLM prompts
    that need full context but should not pollute deterministic filters.
    """

    model_config = ConfigDict(extra="ignore")

    topic_id: str = Field(..., description="TREC topic id or external patient id")
    raw_text: str = Field(..., description="Original patient note / topic text, verbatim")

    # Demographics
    age_years: float | None = None
    sex: Sex = Sex.UNKNOWN

    # Diagnoses
    primary_diagnosis: str | None = None
    primary_diagnosis_stage: str | None = None
    secondary_diagnoses: list[str] = Field(default_factory=list)

    # Performance status
    ecog: int | None = Field(default=None, ge=0, le=5)
    karnofsky: int | None = Field(default=None, ge=0, le=100)

    # Clinical features
    biomarkers: list[Biomarker] = Field(default_factory=list)
    prior_treatments: list[PriorTreatment] = Field(default_factory=list)
    comorbidities: list[Comorbidity] = Field(default_factory=list)
    labs: list[Lab] = Field(default_factory=list)
    pregnancy: Pregnancy | None = None

    # Geography / preferences
    location: str | None = None

    # Residual context for LLM prompts
    free_text_residual: str | None = None

    # MeSH-normalized concepts (filled by mesh_normalizer)
    mesh_concepts: list[dict[str, Any]] = Field(default_factory=list)

    def has_lab(self, name: str) -> bool:
        n = name.lower().strip()
        return any(lab.name.lower().strip() == n for lab in self.labs)

    def get_lab(self, name: str) -> Lab | None:
        n = name.lower().strip()
        for lab in self.labs:
            if lab.name.lower().strip() == n:
                return lab
        return None

    def summary(self) -> str:
        """Compact textual summary used in LLM prompts."""
        parts: list[str] = []
        if self.age_years is not None:
            parts.append(f"{self.age_years:.0f}-year-old")
        if self.sex != Sex.UNKNOWN:
            parts.append(self.sex.value)
        if self.primary_diagnosis:
            parts.append(f"with {self.primary_diagnosis}")
            if self.primary_diagnosis_stage:
                parts.append(f"({self.primary_diagnosis_stage})")
        if self.ecog is not None:
            parts.append(f"ECOG {self.ecog}")
        if self.biomarkers:
            bm = ", ".join(f"{b.name} {b.status}" for b in self.biomarkers[:3])
            parts.append(f"biomarkers: {bm}")
        return " ".join(parts).strip() or self.raw_text[:200]
