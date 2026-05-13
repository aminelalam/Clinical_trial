"""Eligibility evaluation outputs at criterion and trial level.

The trial-level labels mirror TREC CT 2021/2022 qrels (eligible / excludes / irrelevant)
so we can submit predictions in the format the benchmark expects.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .criterion import Criterion


class EligibilityLabel(str, Enum):
    """Per-criterion label."""

    MET = "met"
    NOT_MET = "not_met"
    NEI = "NEI"  # Not Enough Information


class TrialLabel(str, Enum):
    """Trial-level label following TREC CT qrels."""

    ELIGIBLE = "eligible"  # qrel 2
    EXCLUDES = "excludes"  # qrel 1 — meets inclusion but fails exclusion (or fails inclusion)
    IRRELEVANT = "irrelevant"  # qrel 0


class CriterionEval(BaseModel):
    """Result of evaluating a single criterion against a patient."""

    model_config = ConfigDict(extra="ignore")

    criterion_id: str
    label: EligibilityLabel
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: str = Field(
        default="",
        description="Quote(s) from the patient note supporting this label",
    )
    reasoning: str = Field(
        default="", description="Step-by-step reasoning produced by the evaluator (CoT)"
    )

    # Provenance
    evaluator: Literal["deterministic", "llm", "self_consistency", "verifier"] = "llm"
    llm_calls: int = 0
    rebuttal: str | None = Field(
        default=None,
        description="Devil's-advocate rebuttal when verifier ran; non-None implies the verifier ran",
    )
    flipped_by_verifier: bool = False

    # Reference to source criterion (denormalized for downstream rendering)
    criterion: Criterion | None = None


class TrialEval(BaseModel):
    """Aggregated trial-level eligibility result."""

    model_config = ConfigDict(extra="ignore")

    nct_id: str
    label: TrialLabel
    criteria: list[CriterionEval] = Field(default_factory=list)

    # Counts (denormalized for fast scoring + rendering)
    n_inclusion: int = 0
    n_exclusion: int = 0
    n_inclusion_met: int = 0
    n_inclusion_not_met: int = 0
    n_inclusion_nei: int = 0
    n_exclusion_met: int = 0
    n_exclusion_not_met: int = 0
    n_exclusion_nei: int = 0

    # Flags
    any_mandatory_inclusion_failed: bool = False
    any_exclusion_met: bool = False
    fraction_nei: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def n_total(self) -> int:
        return self.n_inclusion + self.n_exclusion

    @property
    def n_met(self) -> int:
        return self.n_inclusion_met + self.n_exclusion_met

    @property
    def n_nei(self) -> int:
        return self.n_inclusion_nei + self.n_exclusion_nei

    @property
    def trec_qrel(self) -> int:
        """Map to TREC CT qrel grade: 2=eligible, 1=excludes, 0=irrelevant."""
        return {TrialLabel.ELIGIBLE: 2, TrialLabel.EXCLUDES: 1, TrialLabel.IRRELEVANT: 0}[self.label]
