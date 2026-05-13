"""HyDE (Hypothetical Document Embeddings) — medical variant.

Generates k=3 hypothetical "ideal trial criteria" sections for the patient
and averages their dense embeddings with the original patient embedding.
This bridges the lexical gap between patient prose and trial criterion text.

Reference: Gao et al. 2022 (HyDE). Medical role conditioning per AutoMIR 2024.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np

from ..llm.client import UnifiedLLM
from ..llm.prompts import HYDE_PROMPT
from ..logging import logger

if TYPE_CHECKING:  # pragma: no cover
    from .dense import DenseRetriever


class HyDEGenerator:
    """Generate k hypothetical eligibility passages and average with the query."""

    def __init__(self, llm: UnifiedLLM | None = None, k: int = 3, temperature: float = 0.7):
        self.llm = llm or UnifiedLLM()
        self.k = k
        self.temperature = temperature

    async def generate_hypos(self, patient_text: str) -> list[str]:
        """Generate k hypothetical trial-criteria passages."""
        prompts = [HYDE_PROMPT.format(patient_note=patient_text)] * self.k
        try:
            results = await asyncio.gather(
                *[
                    self.llm.acomplete(prompts[i], temperature=self.temperature, max_tokens=350)
                    for i in range(self.k)
                ],
                return_exceptions=True,
            )
            hypos = [r for r in results if isinstance(r, str) and r.strip()]
            if not hypos:
                logger.warning("HyDE produced no hypotheticals; using empty list")
            return hypos
        except Exception as e:  # pragma: no cover
            logger.warning(f"HyDE generation failed: {e!r}")
            return []

    async def compose_query_embedding(
        self, patient_text: str, retriever: "DenseRetriever"
    ) -> list[float]:
        """Return averaged embedding of patient text + k hypotheticals.

        Falls back to plain patient embedding if HyDE fails.
        """
        hypos = await self.generate_hypos(patient_text)
        texts = [patient_text] + hypos
        vectors = retriever.encode_queries(texts)
        if not vectors:
            return retriever.encode_query(patient_text)
        avg = np.mean(np.array(vectors), axis=0)
        return avg.tolist()
