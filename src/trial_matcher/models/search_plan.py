"""SearchPlan — structured output of the planning agent."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SearchPlan(BaseModel):
    """A structured search plan emitted by the planning LLM.

    Used by retrieval nodes to parameterize BM25 queries, dense retrieval,
    and hard filters. Mutated in place when a re-retrieval is requested
    (``relax_optional_filters``).
    """

    model_config = ConfigDict(extra="ignore")

    primary_disease_query: str = Field(..., description="Main lexical query for BM25")
    expansion_terms: list[str] = Field(
        default_factory=list,
        description="MeSH synonyms and related terms to include in the lexical query",
    )

    mandatory_filters: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Filters that must be applied (e.g., {'sex': 'FEMALE', 'status': ['RECRUITING']})"
        ),
    )
    optional_filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Filters that may be relaxed if recall is insufficient",
    )

    retrieval_priorities: list[str] = Field(
        default_factory=list,
        description="Aspects that should rank higher (e.g., 'biomarker-targeted therapy')",
    )

    risk_flags: list[str] = Field(
        default_factory=list,
        description="Patient features that may exclude many trials (e.g., 'ECOG 3', 'pregnant')",
    )

    # Mutated by the conditional re-retrieval edge
    relax_optional_filters: bool = False

    # Metadata
    source: str = Field(default="llm", description="'llm' or 'fallback' if LLM call failed")
