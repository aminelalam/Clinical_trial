"""Aggregate per-criterion CriterionEvals into a trial-level TrialEval.

Mapping rules (TREC CT 2021/2022 qrels):
- ELIGIBLE (qrel 2):     all mandatory inclusion criteria met (or NEI <= threshold)
                         AND no exclusion criterion met.
- EXCLUDES (qrel 1):     any inclusion mandatory not met, OR any exclusion met.
- IRRELEVANT (qrel 0):   no inclusion support and very high NEI, indicating retrieval noise.

We compute a trial-level label that is the closest sensible TREC qrel.
The irrelevance rule is deliberately strict and configurable.
(they passed retrieval+filter), so IRRELEVANT is rarely produced here —
the score function uses NEI fraction to demote noisy candidates further.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..config import get_settings
from ..models.criterion import Criterion, Polarity
from ..models.eligibility import CriterionEval, EligibilityLabel, TrialEval, TrialLabel


def aggregate_to_trial_eval(
    nct_id: str,
    criteria: Iterable[Criterion],
    evals: Iterable[CriterionEval],
    min_inclusion_fraction: float | None = None,
    max_nei_fraction: float | None = None,
    use_irrelevance_heuristic: bool | None = None,
    irrelevant_min_nei_fraction: float | None = None,
    irrelevant_max_inclusion_met: int | None = None,
) -> TrialEval:
    """Aggregate per-criterion evaluations into a trial-level verdict.

    Thresholds ``min_inclusion_fraction`` and ``max_nei_fraction`` control the
    partial-eligibility rule (line 74 of the original). When ``None`` they are
    read from ``RunnerSettings`` so they can be tuned per-run or via env vars.
    """
    s = get_settings()
    default_min_inc = (
        s.runner.benchmark_min_inclusion_fraction
        if s.runner.mode == "benchmark"
        else s.runner.min_inclusion_fraction
    )
    default_max_nei = (
        s.runner.benchmark_max_nei_fraction
        if s.runner.mode == "benchmark"
        else s.runner.max_nei_fraction
    )
    min_inc_frac = min_inclusion_fraction if min_inclusion_fraction is not None else default_min_inc
    max_nei_frac = max_nei_fraction if max_nei_fraction is not None else default_max_nei
    use_irrel = (
        use_irrelevance_heuristic
        if use_irrelevance_heuristic is not None
        else s.runner.use_irrelevance_heuristic
    )
    irrel_min_nei = (
        irrelevant_min_nei_fraction
        if irrelevant_min_nei_fraction is not None
        else s.runner.irrelevant_min_nei_fraction
    )
    irrel_max_inc_met = (
        irrelevant_max_inclusion_met
        if irrelevant_max_inclusion_met is not None
        else s.runner.irrelevant_max_inclusion_met
    )
    by_id = {c.id: c for c in criteria}
    eval_by_id: dict[str, CriterionEval] = {}
    for e in evals:
        # Ensure denormalized criterion is attached
        if e.criterion is None and e.criterion_id in by_id:
            e = e.model_copy(update={"criterion": by_id[e.criterion_id]})
        eval_by_id[e.criterion_id] = e

    n_inc = n_exc = 0
    n_inc_met = n_inc_not = n_inc_nei = 0
    n_exc_met = n_exc_not = n_exc_nei = 0
    any_mand_inc_failed = False
    any_exc_met = False

    for cid, ev in eval_by_id.items():
        crit = ev.criterion or by_id.get(cid)
        if crit is None:
            continue
        if crit.polarity == Polarity.INCLUSION:
            n_inc += 1
            if ev.label == EligibilityLabel.MET:
                n_inc_met += 1
            elif ev.label == EligibilityLabel.NOT_MET:
                n_inc_not += 1
                if crit.is_mandatory:
                    any_mand_inc_failed = True
            else:
                n_inc_nei += 1
        else:
            n_exc += 1
            if ev.label == EligibilityLabel.MET:
                n_exc_met += 1
                any_exc_met = True
            elif ev.label == EligibilityLabel.NOT_MET:
                n_exc_not += 1
            else:
                n_exc_nei += 1

    n_total = n_inc + n_exc
    fraction_nei = ((n_inc_nei + n_exc_nei) / n_total) if n_total else 0.0

    # Apply the rules
    if (
        use_irrel
        and n_inc > 0
        and n_inc_met <= irrel_max_inc_met
        and fraction_nei >= irrel_min_nei
        and not any_exc_met
    ):
        label = TrialLabel.IRRELEVANT
    elif any_exc_met or any_mand_inc_failed:
        label = TrialLabel.EXCLUDES
    elif n_inc > 0 and n_inc_met == n_inc and n_exc_met == 0:
        label = TrialLabel.ELIGIBLE
    elif n_inc_met >= max(1, int(min_inc_frac * n_inc)) and n_exc_met == 0 and fraction_nei < max_nei_frac:
        # Mostly clear with some NEI — still consider eligible from a TREC qrel-2 perspective.
        label = TrialLabel.ELIGIBLE
    else:
        # Cannot conclude eligibility — stays excludes.
        label = TrialLabel.EXCLUDES

    return TrialEval(
        nct_id=nct_id,
        label=label,
        criteria=list(eval_by_id.values()),
        n_inclusion=n_inc,
        n_exclusion=n_exc,
        n_inclusion_met=n_inc_met,
        n_inclusion_not_met=n_inc_not,
        n_inclusion_nei=n_inc_nei,
        n_exclusion_met=n_exc_met,
        n_exclusion_not_met=n_exc_not,
        n_exclusion_nei=n_exc_nei,
        any_mandatory_inclusion_failed=any_mand_inc_failed,
        any_exclusion_met=any_exc_met,
        fraction_nei=fraction_nei,
    )
