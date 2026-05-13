"""Benchmark-only ranking fill for hard-excluded candidates.

The clinical pipeline must not surface hard-excluded trials as enrolment
recommendations. TREC, however, grades partially relevant trials as qrel=1
even when the patient would be excluded by age/sex/status. This module keeps
that benchmark behavior isolated from eligibility evaluation and scoring.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ..models.agent_state import TrialCandidate
from ..models.dossier import ScoreBreakdown
from ..models.eligibility import TrialEval, TrialLabel
from ..models.ranking import RankedTrial
from ..models.trial import Trial


@dataclass(frozen=True)
class HardExcludedFillResult:
    ranked: list[RankedTrial]
    count: int
    retrieval_tail_count: int
    reasons: dict[str, int]
    skipped_corpus_miss: int


def fill_hard_excluded_to_top_k(
    *,
    ranked: Iterable[RankedTrial],
    final_candidates: Iterable[TrialCandidate],
    evaluated_ids: set[str],
    get_trial: Callable[[str], Trial | None],
    output_k: int,
    include_retrieval_tail: bool = False,
) -> HardExcludedFillResult:
    """Append synthetic EXCLUDES results until ``output_k`` is reached.

    Filled trials are ordered by retrieval score but receive scores below every
    already-ranked viable trial. No criteria are extracted and no LLM calls are
    made here; this is strictly a benchmark output-completeness operation.
    """
    out = list(ranked)
    output_k = max(1, int(output_k or 10))
    if len(out) >= output_k:
        return HardExcludedFillResult(out, 0, 0, {}, 0)

    already_ranked_ids = {r.nct_id for r in out}
    tail_count = 0
    skipped_corpus_miss = 0
    min_existing_score = min((r.score for r in out), default=-1.0)
    tail_base_score = min(-1.5, min_existing_score - 0.5)
    if include_retrieval_tail:
        tail = [
            c
            for c in final_candidates
            if not c.hard_excluded
            and c.nct_id not in evaluated_ids
            and c.nct_id not in already_ranked_ids
        ]
        tail.sort(key=lambda c: (c.score is None, -(c.score or 0.0)))
        tail_slots = output_k - len(out)
        for cand in tail:
            if tail_count >= tail_slots:
                break
            if get_trial(cand.nct_id) is None:
                skipped_corpus_miss += 1
                continue
            fill_prior = max(0.0, 1.0 - (tail_count / max(tail_slots, 1)))
            fill_score = tail_base_score + (0.001 * fill_prior)
            fill_breakdown = ScoreBreakdown(
                total=fill_score,
                eligibility_score=0.0,
                recruiting_bonus=0.0,
                phase_alignment=0.0,
                recency=0.0,
                geographic_proximity=0.0,
                retrieval_prior=fill_prior,
                mandatory_veto=False,
                nei_penalty=1.0,
                confidence_adjustment=1.0,
            )
            out.append(
                RankedTrial(
                    nct_id=cand.nct_id,
                    score=fill_score,
                    eval=TrialEval(
                        nct_id=cand.nct_id,
                        label=TrialLabel.EXCLUDES,
                        fraction_nei=1.0,
                    ),
                    components=fill_breakdown.model_dump(),
                    retrieval_tail_fill=True,
                    excluded_reason="not evaluated; retrieval tail fill",
                )
            )
            tail_count += 1

    if len(out) >= output_k:
        return HardExcludedFillResult(out, 0, tail_count, {}, skipped_corpus_miss)

    filled_ids = {r.nct_id for r in out}
    excluded = [
        c
        for c in final_candidates
        if c.hard_excluded
        and c.nct_id not in evaluated_ids
        and c.nct_id not in filled_ids
    ]
    excluded.sort(key=lambda c: (c.score is None, -(c.score or 0.0)))

    slots = output_k - len(out)
    min_existing_score = min((r.score for r in out), default=-1.0)
    base_score = min(-2.0, min_existing_score - 1.0)
    count = 0
    reasons: dict[str, int] = {}

    for cand in excluded:
        if count >= slots:
            break
        if get_trial(cand.nct_id) is None:
            skipped_corpus_miss += 1
            continue

        reason = cand.excluded_reason or "unknown"
        fill_prior = max(0.0, 1.0 - (count / max(slots, 1)))
        fill_score = base_score + (0.001 * fill_prior)
        fill_breakdown = ScoreBreakdown(
            total=fill_score,
            eligibility_score=0.0,
            recruiting_bonus=0.0,
            phase_alignment=0.0,
            recency=0.0,
            geographic_proximity=0.0,
            retrieval_prior=fill_prior,
            mandatory_veto=True,
            nei_penalty=0.0,
            confidence_adjustment=1.0,
        )
        out.append(
            RankedTrial(
                nct_id=cand.nct_id,
                score=fill_score,
                eval=TrialEval(
                    nct_id=cand.nct_id,
                    label=TrialLabel.EXCLUDES,
                    any_mandatory_inclusion_failed=True,
                    fraction_nei=0.0,
                ),
                components=fill_breakdown.model_dump(),
                hard_excluded_fill=True,
                excluded_reason=reason,
            )
        )
        count += 1
        reasons[reason] = reasons.get(reason, 0) + 1

    return HardExcludedFillResult(out, count, tail_count, reasons, skipped_corpus_miss)
