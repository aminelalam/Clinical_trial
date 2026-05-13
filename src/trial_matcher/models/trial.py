"""Trial — typed representation of a ClinicalTrials.gov study."""

from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Phase(str, Enum):
    EARLY_PHASE_1 = "EARLY_PHASE1"
    PHASE_1 = "PHASE1"
    PHASE_1_2 = "PHASE1_PHASE2"
    PHASE_2 = "PHASE2"
    PHASE_2_3 = "PHASE2_PHASE3"
    PHASE_3 = "PHASE3"
    PHASE_4 = "PHASE4"
    NA = "NA"


class RecruitmentStatus(str, Enum):
    NOT_YET_RECRUITING = "NOT_YET_RECRUITING"
    RECRUITING = "RECRUITING"
    ENROLLING_BY_INVITATION = "ENROLLING_BY_INVITATION"
    ACTIVE_NOT_RECRUITING = "ACTIVE_NOT_RECRUITING"
    SUSPENDED = "SUSPENDED"
    TERMINATED = "TERMINATED"
    COMPLETED = "COMPLETED"
    WITHDRAWN = "WITHDRAWN"
    UNKNOWN = "UNKNOWN"


class Sex(str, Enum):
    ALL = "ALL"
    MALE = "MALE"
    FEMALE = "FEMALE"


class AgeRange(BaseModel):
    """Age limits in days for deterministic filtering. None means unbounded."""

    model_config = ConfigDict(extra="ignore")

    min_days: int | None = None
    max_days: int | None = None
    raw_min: str | None = Field(default=None, description="e.g. '18 Years'")
    raw_max: str | None = Field(default=None, description="e.g. '75 Years'")


class Location(BaseModel):
    model_config = ConfigDict(extra="ignore")

    facility: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None
    status: str | None = None


class Contact(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    role: str | None = None
    email: str | None = None
    phone: str | None = None


class Eligibility(BaseModel):
    """Trial eligibility section."""

    model_config = ConfigDict(extra="ignore")

    raw_text: str = Field(default="", description="Full unparsed criteria text")
    inclusion_text: str = Field(default="")
    exclusion_text: str = Field(default="")
    age_range: AgeRange = Field(default_factory=AgeRange)
    sex: Sex = Sex.ALL
    accepts_healthy_volunteers: bool | None = None


class Trial(BaseModel):
    """ClinicalTrials.gov trial record (subset relevant for matching)."""

    model_config = ConfigDict(extra="ignore")

    nct_id: str = Field(..., description="ClinicalTrials.gov NCT identifier")
    title: str = Field(default="", description="Brief title")
    official_title: str | None = None
    brief_summary: str = Field(default="")
    detailed_description: str | None = None
    conditions: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)

    phase: Phase = Phase.NA
    status: RecruitmentStatus = RecruitmentStatus.UNKNOWN
    study_type: str | None = None
    interventional: bool = True

    eligibility: Eligibility = Field(default_factory=Eligibility)
    locations: list[Location] = Field(default_factory=list)
    contacts: list[Contact] = Field(default_factory=list)

    sponsor: str | None = None
    last_update_date: date | None = None
    enrollment: int | None = None

    # Derived
    @property
    def url(self) -> str:
        return f"https://clinicaltrials.gov/study/{self.nct_id}"

    def primary_text(self) -> str:
        """Concatenated text for indexing / retrieval."""
        parts = [
            self.title,
            self.official_title or "",
            self.brief_summary,
            self.detailed_description or "",
            " ".join(self.conditions),
            " ".join(self.keywords),
            " ".join(self.interventions),
            self.eligibility.inclusion_text,
            self.eligibility.exclusion_text,
        ]
        return "\n".join(p for p in parts if p)

    def text_for_bm25_field(self, field: str) -> str:
        """Return a deterministic text view used by fielded BM25 indexes."""
        match field:
            case "all":
                return self.primary_text()
            case "condition_title":
                parts = [
                    self.title,
                    self.official_title or "",
                    " ".join(self.conditions),
                    " ".join(self.keywords),
                ]
            case "eligibility":
                parts = [
                    self.eligibility.inclusion_text,
                    self.eligibility.exclusion_text,
                    self.eligibility.raw_text,
                ]
            case "intervention":
                parts = [
                    " ".join(self.interventions),
                    self.brief_summary,
                    self.detailed_description or "",
                ]
            case "summary_description":
                parts = [
                    self.brief_summary,
                    self.detailed_description or "",
                    self.title,
                ]
            case _:
                raise ValueError(f"Unknown BM25 field: {field}")
        return "\n".join(p for p in parts if p)
