"""ClinicalQuestion — output of the question generator (Task T4)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class DataType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATE = "date"
    BOOLEAN = "boolean"
    TEXT = "text"


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ClinicalQuestion(BaseModel):
    """A clinically-formulated question for an indeterminate (NEI) criterion.

    Six required elements (per the spec):
    1. Specific data point requested (with units)
    2. Time window (within what period the data must be valid)
    3. Measurement context (lab type, imaging modality, scoring system)
    4. Why this matters (1-sentence clinical relevance)
    5. Format expected for the answer
    6. Priority level
    """

    model_config = ConfigDict(extra="ignore")

    trial_id: str = Field(..., description="NCT id of the trial whose criterion is undetermined")
    criterion_id: str = Field(..., description="Local criterion id (e.g., 'i_4')")

    question_text: str = Field(..., description="The actual question, max ~50 words")
    data_point: str = Field(..., description="Specific clinical data requested (e.g., 'ANC')")
    units: str | None = None
    time_window: str | None = Field(default=None, description="e.g. 'within last 30 days'")
    measurement_context: str | None = Field(
        default=None, description="e.g. 'CBC with differential', 'CT scan with contrast'"
    )
    rationale: str = Field(..., description="Why this matters for trial X (1 sentence)")
    expected_data_type: DataType = DataType.TEXT
    priority: Priority = Priority.MEDIUM
