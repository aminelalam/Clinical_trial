"""Self-critique output produced by the agentic critique node."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class IssueSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CritiqueIssue(BaseModel):
    model_config = ConfigDict(extra="ignore")

    trial_id: str
    issue: str
    severity: IssueSeverity = IssueSeverity.MEDIUM


class Critique(BaseModel):
    """Output of the self-critique node over top-5 trials."""

    model_config = ConfigDict(extra="ignore")

    issues_found: list[CritiqueIssue] = Field(default_factory=list)
    rerank_needed: bool = False
    rerank_instructions: str = ""
    final_notes: str = ""
