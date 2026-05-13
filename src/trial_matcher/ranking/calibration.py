"""Optimize ScoreWeights against a development qrels set (TREC 2021 by default).

We use Nelder-Mead (no gradient required) to maximize NDCG@10 over the dev
topics. Calibration is a one-shot offline procedure: take the predictions
emitted by the runner with default weights, re-score with candidate weights,
and pick the winner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..logging import logger
from .scorer import ScoreWeights, score_trial


@dataclass
class CalibrationSample:
    """One (topic, candidate-trial-list, eval-list, qrel) sample for calibration."""

    topic_id: str
    trials: list  # list[Trial]
    evals: list  # list[TrialEval]
    qrels: dict[str, int]  # nct_id -> grade


def _ndcg_at_10(ranking: list[str], qrels: dict[str, int]) -> float:
    """NDCG@10 with TREC graded relevance (0/1/2)."""
    rels = [qrels.get(nct, 0) for nct in ranking[:10]]
    dcg = sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(rels))
    ideal = sorted(qrels.values(), reverse=True)[:10]
    idcg = sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _negative_ndcg(weights_array: np.ndarray, samples: list[CalibrationSample]) -> float:
    w = ScoreWeights(
        eligibility=float(weights_array[0]),
        recruiting=float(weights_array[1]),
        phase=float(weights_array[2]),
        recency=float(weights_array[3]),
        geography=float(weights_array[4]),
        nei_penalty=float(weights_array[5]),
        confidence_blend=float(np.clip(weights_array[6], 0.0, 1.0)),
    )
    ndcgs = []
    for s in samples:
        scored = [score_trial(t, e, weights=w) for t, e in zip(s.trials, s.evals)]
        scored.sort(key=lambda r: r.score, reverse=True)
        ranking = [r.nct_id for r in scored]
        ndcgs.append(_ndcg_at_10(ranking, s.qrels))
    return -float(np.mean(ndcgs)) if ndcgs else 0.0


def calibrate_weights(samples: list[CalibrationSample]) -> ScoreWeights:
    """Fit ScoreWeights to maximize mean NDCG@10 over the supplied samples."""
    if not samples:
        logger.warning("No calibration samples — returning default weights")
        return ScoreWeights()

    x0 = np.array([1.0, 0.30, 0.20, 0.10, 0.10, 0.20, 0.5])
    result = minimize(
        _negative_ndcg, x0, args=(samples,), method="Nelder-Mead",
        options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-3},
    )
    w = ScoreWeights(
        eligibility=float(result.x[0]),
        recruiting=float(result.x[1]),
        phase=float(result.x[2]),
        recency=float(result.x[3]),
        geography=float(result.x[4]),
        nei_penalty=float(result.x[5]),
        confidence_blend=float(np.clip(result.x[6], 0.0, 1.0)),
    )
    logger.info(
        f"Calibrated weights: NDCG@10={-result.fun:.4f}, weights={w.model_dump()}"
    )
    return w
