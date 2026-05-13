"""Pointwise cross-encoder reranker using MedCPT-Cross-Encoder.

Takes (query, candidate_text) pairs, scores each, and re-orders.
Falls back to BGE-reranker-v2-m3 if MedCPT-CE is unavailable.
"""

from __future__ import annotations

from typing import Callable

from ..config import get_settings
from ..logging import logger
from ..models.agent_state import TrialCandidate


class CrossEncoderReranker:
    """Wraps a HuggingFace AutoModelForSequenceClassification cross-encoder."""

    def __init__(self, model_name: str | None = None):
        s = get_settings()
        self.model_name = model_name or s.retrieval.medcpt_cross_encoder
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info(f"Loading cross-encoder: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()
        if torch.cuda.is_available():
            self._model = self._model.cuda()

    def rerank(
        self,
        query: str,
        candidates: list[TrialCandidate],
        get_text: Callable[[str], str],
        top_k: int | None = None,
        batch_size: int = 32,
    ) -> list[TrialCandidate]:
        """Re-rank ``candidates`` and return the new ordered list (length top_k).

        ``get_text`` maps an NCT id to the text to use for scoring (typically
        title + brief_summary + key conditions, truncated to ~400 tokens).
        """
        if not candidates:
            return []

        s = get_settings()
        top_k = top_k or s.retrieval.rerank_top_k
        blend = (
            max(0.0, min(1.0, s.retrieval.fielded_rerank_retrieval_blend))
            if s.retrieval.bm25_mode == "fielded"
            else 0.0
        )
        if blend >= 1.0:
            return [
                TrialCandidate(
                    nct_id=c.nct_id,
                    score=float(c.score or 0.0),
                    source="reranked",
                    rank=i + 1,
                    title=c.title,
                    snippet=c.snippet,
                    retrieval_metadata={
                        **c.retrieval_metadata,
                        "cross_encoder_score": None,
                        "rerank_retrieval_blend": blend,
                        "pre_rerank_score": float(c.score or 0.0),
                        "rerank_strategy": "retrieval_order",
                    },
                )
                for i, c in enumerate(candidates[:top_k])
            ]

        self._load()
        import torch

        pairs = [[query, get_text(c.nct_id)] for c in candidates]
        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i : i + batch_size]
                enc = self._tokenizer(  # type: ignore[union-attr]
                    batch,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=512,
                )
                if torch.cuda.is_available():
                    enc = {k: v.cuda() for k, v in enc.items()}
                logits = self._model(**enc).logits  # type: ignore[union-attr]
                if logits.shape[-1] == 1:
                    scores.extend(logits.squeeze(-1).cpu().tolist())
                else:
                    scores.extend(logits[:, -1].cpu().tolist())

        rank_scores = list(scores)
        if blend > 0.0 and len(candidates) > 1:
            ce_min, ce_max = min(scores), max(scores)
            ret_values = [float(c.score or 0.0) for c in candidates]
            ret_min, ret_max = min(ret_values), max(ret_values)

            def norm(value: float, lo: float, hi: float) -> float:
                return 1.0 if hi <= lo else (value - lo) / (hi - lo)

            rank_scores = [
                ((1.0 - blend) * norm(ce, ce_min, ce_max))
                + (blend * norm(ret, ret_min, ret_max))
                for ce, ret in zip(scores, ret_values)
            ]

        scored = sorted(
            zip(candidates, scores, rank_scores), key=lambda x: x[2], reverse=True
        )[:top_k]
        return [
            TrialCandidate(
                nct_id=c.nct_id,
                score=float(rank_score),
                source="reranked",
                rank=i + 1,
                title=c.title,
                snippet=c.snippet,
                retrieval_metadata={
                    **c.retrieval_metadata,
                    "cross_encoder_score": float(score),
                    "rerank_retrieval_blend": blend,
                    "pre_rerank_score": float(c.score or 0.0),
                },
            )
            for i, (c, score, rank_score) in enumerate(scored)
        ]
