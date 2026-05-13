"""Structured representation of an eligibility criterion.

A Criterion is the unit of evaluation. It carries both the raw text (for
LLM fallback) and a typed Predicate (for deterministic evaluation when possible).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CriterionType(str, Enum):
    AGE = "age"
    SEX = "sex"
    BIOMARKER = "biomarker"
    PRIOR_TREATMENT = "prior_treatment"
    LAB = "lab"
    COMORBIDITY = "comorbidity"
    PERFORMANCE_STATUS = "performance_status"
    PREGNANCY = "pregnancy"
    DIAGNOSIS = "diagnosis"
    CONSENT = "consent"
    SECTION_HEADER = "section_header"
    OTHER = "other"


class Polarity(str, Enum):
    INCLUSION = "inclusion"
    EXCLUSION = "exclusion"


PredicateOp = Literal[
    ">=", "<=", ">", "<", "=", "!=",
    "in", "not_in",
    "present", "absent", "unknown",
]


class TemporalConstraint(BaseModel):
    """Time window attached to a criterion or predicate.

    ``relation`` summarizes the qualitative relation; ``days`` is the window
    length in days when applicable (e.g., 'within 6 months' → days=180).
    """

    model_config = ConfigDict(extra="ignore")

    relation: Literal[
        "within",  # within last X days
        "before",
        "after",
        "ever",  # any time in patient history
        "current",  # right now
        "ongoing",
        "never",
    ]
    days: int | None = None
    raw: str = Field(default="")


class Predicate(BaseModel):
    """A typed clinical predicate that can be evaluated deterministically.

    Example: ``Predicate(op=">=", feature="age_years", value=18)`` for
    "Age >= 18 years".
    """

    model_config = ConfigDict(extra="ignore")

    op: PredicateOp
    feature: str = Field(..., description="Canonical feature name, e.g. 'age_years', 'ECOG', 'ANC'")
    value: Any = None
    units: str | None = None
    temporal: TemporalConstraint | None = None


class Criterion(BaseModel):
    """A single eligibility criterion extracted from a trial."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(..., description="Local id like 'i_3' (3rd inclusion) or 'e_1' (1st exclusion)")
    polarity: Polarity
    raw_text: str = Field(..., description="Original criterion text from the trial")
    type: CriterionType = CriterionType.OTHER
    predicate: Predicate | None = None
    is_mandatory: bool = True

    # Extraction-time annotations
    has_negation: bool = Field(
        default=False, description="True if NegEx detected explicit negation in the criterion text"
    )
    annotated_text: str | None = Field(
        default=None,
        description="raw_text with negation/temporal markers explicitly tagged for LLM evaluation",
    )
    extraction_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    triage_score: float = Field(
        default=0.0,
        description="Deterministic pre-LLM priority score used when criteria are capped",
    )
    triage_reasons: list[str] = Field(
        default_factory=list,
        description="Short audit tags explaining why the criterion was selected",
    )

    def short(self) -> str:
        prefix = "INCL" if self.polarity == Polarity.INCLUSION else "EXCL"
        return f"[{self.id} {prefix} {self.type.value}] {self.raw_text}"
