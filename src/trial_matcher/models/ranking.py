"""Ranking outputs: pre-judge ranked trials and post-judge final order."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .eligibility import TrialEval


class RankedTrial(BaseModel):
    """A trial after deterministic scoring but before LLM-judge."""

    model_config = ConfigDict(extra="ignore")

    nct_id: str
    score: float
    rank: int = Field(default=0, ge=0)
    eval: TrialEval

    # Score component breakdown (for auditability and debugging)
    components: dict[str, float] = Field(default_factory=dict)

    # Benchmark-only P7 provenance for synthetic hard-excluded fills.
    hard_excluded_fill: bool = False
    retrieval_tail_fill: bool = False
    excluded_reason: str | None = None


class JudgedTrial(BaseModel):
    """A trial after LLM-judge top-10 reordering."""

    model_config = ConfigDict(extra="ignore")

    nct_id: str
    rank: int
    score: float
    eval: TrialEval
    rationale: str = Field(default="", description="2-line clinical justification by the judge")
    components: dict[str, float] = Field(default_factory=dict)
    hard_excluded_fill: bool = False
    retrieval_tail_fill: bool = False
    excluded_reason: str | None = None
