"""TrialDossier — JSON-first structured deliverable for Task T5.

Schema is fixed and validated by Pydantic. The Markdown render in
`dossier.builder` is a pure projection from this object.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .eligibility import EligibilityLabel
from .question import ClinicalQuestion


class FlagSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AttentionFlag(BaseModel):
    """A flag the clinician should review (e.g., far site, many NEI, mandatory failed)."""

    model_config = ConfigDict(extra="ignore")

    severity: FlagSeverity = FlagSeverity.WARNING
    category: Literal[
        "geographic", "eligibility", "trial_status", "data_quality", "clinical"
    ] = "eligibility"
    message: str


class CriterionRow(BaseModel):
    """A row in the per-trial criterion table inside a dossier."""

    model_config = ConfigDict(extra="ignore")

    id: str
    type: str
    polarity: Literal["inclusion", "exclusion"]
    text: str
    label: EligibilityLabel
    evidence: str = ""
    confidence: float = 0.5


class EligibilityCounts(BaseModel):
    model_config = ConfigDict(extra="ignore")

    inclusion_total: int = 0
    inclusion_met: int = 0
    inclusion_not_met: int = 0
    inclusion_nei: int = 0
    exclusion_total: int = 0
    exclusion_met: int = 0
    exclusion_not_met: int = 0
    exclusion_nei: int = 0


class ScoreBreakdown(BaseModel):
    """Auditable breakdown of the score function output."""

    model_config = ConfigDict(extra="ignore")

    total: float
    eligibility_score: float = 0.0
    recruiting_bonus: float = 0.0
    phase_alignment: float = 0.0
    recency: float = 0.0
    geographic_proximity: float = 0.0
    retrieval_prior: float = 0.0
    mandatory_veto: bool = False
    nei_penalty: float = 0.0
    confidence_adjustment: float = 1.0


class DossierMetadata(BaseModel):
    """Static metadata about the trial."""

    model_config = ConfigDict(extra="ignore")

    title: str = ""
    official_title: str | None = None
    phase: str = "NA"
    status: str = "UNKNOWN"
    sponsor: str | None = None
    last_update: date | None = None
    locations_summary: str = ""
    contact_summary: str | None = None
    ctgov_url: str = ""


class TrialDossier(BaseModel):
    """The deliverable for one trial in the per-patient pre-selection package."""

    model_config = ConfigDict(extra="ignore")

    nct_id: str
    rank: int = Field(..., ge=1)
    score: float
    score_breakdown: ScoreBreakdown

    executive_summary: str = Field(
        default="", description="3-5 lines: why this trial fits this patient"
    )

    metadata: DossierMetadata = Field(default_factory=DossierMetadata)

    eligibility_counts: EligibilityCounts = Field(default_factory=EligibilityCounts)
    eligibility_table: list[CriterionRow] = Field(default_factory=list)

    missing_information: list[ClinicalQuestion] = Field(default_factory=list)
    attention_flags: list[AttentionFlag] = Field(default_factory=list)

    # Provenance
    judge_rationale: str | None = None
    critique_notes: list[str] = Field(default_factory=list)
