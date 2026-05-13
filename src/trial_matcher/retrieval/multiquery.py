"""Multi-query reformulation — generate N alternative queries with different aspects.

EXPERIMENTAL — not wired into the agent graph. Kept for future activation if
HyDE + RRF prove insufficient for retrieval diversity. Rationale for deactivation:
(a) HyDE + RRF already covers lexical-semantic diversification, (b) multi-query
adds an LLM call per topic which is not justified with a $100 budget, and (c)
AutoMIR 2024 reports that k>3 medical hypotheticals introduce noise.

Each reformulation emphasizes a different facet:
1. Disease + stage + line of therapy
2. Biomarker / molecular profile
3. Comorbidities and exclusions
4. Geographic / demographic focus
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm.client import UnifiedLLM
from ..llm.prompts import MULTI_QUERY_PROMPT
from ..llm.structured import structured_complete


class MultiQueryOutput(BaseModel):
    queries: list[str] = Field(default_factory=list)


class MultiQueryGenerator:
    """Produces 4 reformulations focused on disease, biomarker, exclusions, demographics."""

    def __init__(self, llm: UnifiedLLM | None = None):
        self.llm = llm or UnifiedLLM()

    async def generate(self, patient_note: str) -> list[str]:
        try:
            out = await structured_complete(
                self.llm,
                prompt=MULTI_QUERY_PROMPT.format(patient_note=patient_note),
                response_model=MultiQueryOutput,
                temperature=0.4,
                max_tokens=400,
                task_name="multi_query",
            )
            return [q.strip() for q in out.queries if q.strip()][:4]
        except Exception:
            return []
