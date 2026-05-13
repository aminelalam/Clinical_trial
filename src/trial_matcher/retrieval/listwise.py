"""Listwise reranker (RankZephyr) — reorders top candidates with sliding-window prompting.

Uses ``rank_llm`` (Castorini) when installed; otherwise falls back to a
no-op (returns the input order). The fallback keeps the runner usable
even on machines without the rank_llm extras.
"""

from __future__ import annotations

from typing import Callable

from ..config import get_settings
from ..logging import logger
from ..models.agent_state import TrialCandidate


class ListwiseReranker:
    """Optional listwise reranker. Set use=False to keep input order unchanged."""

    def __init__(self, model_name: str | None = None, use: bool = True):
        s = get_settings()
        self.model_name = model_name or s.retrieval.listwise_model
        self.use = use
        self._reranker = None

    def _load(self) -> None:
        if self._reranker is not None:
            return
        try:
            from rank_llm.rerank import Reranker
            from rank_llm.rerank.listwise.rank_listwise_os_llm import RankListwiseOSLLM

            logger.info(f"Loading listwise reranker: {self.model_name}")
            agent = RankListwiseOSLLM(
                model=self.model_name,
                context_size=4096,
                num_few_shot_examples=0,
                window_size=20,
                step_size=10,
            )
            self._reranker = Reranker(agent)
        except Exception as e:  # pragma: no cover
            logger.warning(
                f"Listwise reranker unavailable ({e!r}); falling back to no-op order"
            )
            self._reranker = None
            self.use = False

    def rerank(
        self,
        query: str,
        candidates: list[TrialCandidate],
        get_text: Callable[[str], str],
        top_k: int | None = None,
    ) -> list[TrialCandidate]:
        s = get_settings()
        top_k = top_k or s.retrieval.listwise_top_k

        if not self.use or not candidates:
            return [
                TrialCandidate(
                    nct_id=c.nct_id,
                    score=c.score,
                    source="listwise",
                    rank=i + 1,
                    title=c.title,
                    snippet=c.snippet,
                    retrieval_metadata=c.retrieval_metadata,
                )
                for i, c in enumerate(candidates[:top_k])
            ]

        self._load()
        if self._reranker is None:
            # Load failed silently, return identity
            return [
                TrialCandidate(
                    nct_id=c.nct_id,
                    score=c.score,
                    source="listwise",
                    rank=i + 1,
                    title=c.title,
                    snippet=c.snippet,
                    retrieval_metadata=c.retrieval_metadata,
                )
                for i, c in enumerate(candidates[:top_k])
            ]

        # Build rank_llm Request shape
        try:
            from rank_llm.data import Candidate, Query, Request

            rl_candidates = [
                Candidate(docid=c.nct_id, score=c.score, doc={"text": get_text(c.nct_id)})
                for c in candidates
            ]
            request = Request(query=Query(text=query), candidates=rl_candidates)
            result = self._reranker.rerank(request, rank_end=top_k)
            ordered = result.candidates if hasattr(result, "candidates") else result
        except Exception as e:  # pragma: no cover
            logger.warning(f"Listwise reranker call failed ({e!r}); returning identity order")
            ordered = candidates[:top_k]

        out: list[TrialCandidate] = []
        by_id = {c.nct_id: c for c in candidates}
        for i, c in enumerate(ordered[:top_k]):
            nct_id = getattr(c, "docid", None) or getattr(c, "nct_id", None) or ""
            original = by_id.get(str(nct_id))
            out.append(
                TrialCandidate(
                    nct_id=str(nct_id),
                    score=float(getattr(c, "score", 0.0)),
                    source="listwise",
                    rank=i + 1,
                    title=original.title if original else None,
                    snippet=original.snippet if original else None,
                    retrieval_metadata=original.retrieval_metadata if original else {},
                )
            )
        return out
