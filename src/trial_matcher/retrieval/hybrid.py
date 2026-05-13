"""Reciprocal Rank Fusion for merging multiple retrieval runs."""

from __future__ import annotations

from collections.abc import Sequence

from ..config import get_settings
from ..models.agent_state import TrialCandidate


def reciprocal_rank_fusion(
    runs: Sequence[Sequence[TrialCandidate]],
    k: int | None = None,
    top_k: int | None = None,
    weights: Sequence[float] | None = None,
) -> list[TrialCandidate]:
    """Fuse multiple ranked runs using RRF.

    RRF score for doc d: sum over runs r of weight_r / (k + rank_r(d)).
    Default k=60 (TREC convention). The optional weights let the caller keep a
    high-recall but noisy retriever as a secondary signal without letting it
    dominate the top of the fused list.
    """
    settings = get_settings()
    rrf_k = k if k is not None else settings.retrieval.rrf_k
    if weights is None:
        run_weights = [1.0 for _ in runs]
    else:
        if len(weights) != len(runs):
            raise ValueError("RRF weights length must match number of runs")
        run_weights = [max(float(w), 0.0) for w in weights]

    scores: dict[str, float] = {}
    seen: dict[str, TrialCandidate] = {}

    for run, weight in zip(runs, run_weights):
        if weight <= 0:
            continue
        for rank, cand in enumerate(run, start=1):
            scores[cand.nct_id] = scores.get(cand.nct_id, 0.0) + weight / (rrf_k + rank)
            # Keep first-seen candidate for metadata (title, snippet, etc.)
            if cand.nct_id not in seen:
                seen[cand.nct_id] = cand

    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    if top_k is not None:
        fused = fused[:top_k]

    return [
        TrialCandidate(
            nct_id=nct_id,
            score=score,
            source="fused",
            rank=i + 1,
            title=seen[nct_id].title,
            snippet=seen[nct_id].snippet,
            retrieval_metadata=seen[nct_id].retrieval_metadata,
        )
        for i, (nct_id, score) in enumerate(fused)
    ]
