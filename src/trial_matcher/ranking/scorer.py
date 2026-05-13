"""Pure deterministic scoring function for ranking trials.

The score is a weighted linear combination of components. In clinical-active
mode, mandatory failures and met exclusions remain hard vetoes. In benchmark
mode, the same safety signal is retained as ``mandatory_veto=True`` but is
applied as a soft penalty, because TREC ranking rewards graded topical
relevance even for trials that are not safely enrolable.

The ``status_bonus`` table is mode-dependent:
- ``clinical_active``: enforces enrollment readiness (RECRUITING > others;
  COMPLETED/TERMINATED penalised). This is the right behaviour for a
  clinician-facing tool.
- ``benchmark``: TREC qrels include trials in any state, so closed trials
  must not be sent to the bottom of the ranking. We give them 0.0 instead
  of -1.0 and slightly reward ACTIVE_NOT_RECRUITING because such trials
  often sit in qrels as eligible.
"""

from __future__ import annotations

from datetime import date
from math import log1p
from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..config import get_settings
from ..models.dossier import ScoreBreakdown
from ..models.eligibility import TrialEval
from ..models.ranking import RankedTrial
from ..models.trial import Phase, RecruitmentStatus, Trial

ScorerMode = Literal["benchmark", "clinical_active"]


class ScoreWeights(BaseModel):
    """Weights for the deterministic score function."""

    model_config = ConfigDict(extra="ignore")

    eligibility: float = 1.0
    recruiting: float = 0.30
    phase: float = 0.20
    recency: float = 0.10
    geography: float = 0.10
    retrieval: float = 0.10
    nei_penalty: float = 0.20
    hard_veto_tie_break: float = 0.001
    soft_veto_penalty: float = 0.35
    confidence_blend: float = 0.5  # 1.0 = full confidence weighting, 0.0 = none

    # In clinical_active mode mandatory_veto is still applied as score=-1.0.


DEFAULT_WEIGHTS = ScoreWeights()

_PHASE_PRIORITY = {
    Phase.PHASE_3: 1.0,
    Phase.PHASE_2_3: 0.95,
    Phase.PHASE_2: 0.85,
    Phase.PHASE_1_2: 0.70,
    Phase.PHASE_1: 0.50,
    Phase.PHASE_4: 0.85,
    Phase.EARLY_PHASE_1: 0.40,
    Phase.NA: 0.5,
}

# Clinical-active mode: privileges trials a patient could enrol in today.
_STATUS_BONUS_CLINICAL = {
    RecruitmentStatus.RECRUITING: 1.0,
    RecruitmentStatus.NOT_YET_RECRUITING: 0.5,
    RecruitmentStatus.ENROLLING_BY_INVITATION: 0.7,
    RecruitmentStatus.ACTIVE_NOT_RECRUITING: 0.0,
    RecruitmentStatus.SUSPENDED: -0.5,
    RecruitmentStatus.TERMINATED: -1.0,
    RecruitmentStatus.COMPLETED: -1.0,
    RecruitmentStatus.WITHDRAWN: -1.0,
    RecruitmentStatus.UNKNOWN: 0.0,
}

# Benchmark mode: TREC qrels can mark COMPLETED/TERMINATED as eligible. Do not
# send them to the bottom of the ranking. Still favour active recruiting trials
# but never below 0 unless the trial was explicitly WITHDRAWN.
_STATUS_BONUS_BENCHMARK = {
    RecruitmentStatus.RECRUITING: 1.0,
    RecruitmentStatus.NOT_YET_RECRUITING: 0.5,
    RecruitmentStatus.ENROLLING_BY_INVITATION: 0.7,
    RecruitmentStatus.ACTIVE_NOT_RECRUITING: 0.3,
    RecruitmentStatus.SUSPENDED: 0.0,
    RecruitmentStatus.TERMINATED: 0.0,
    RecruitmentStatus.COMPLETED: 0.0,
    RecruitmentStatus.WITHDRAWN: -0.3,
    RecruitmentStatus.UNKNOWN: 0.0,
}


def status_bonus(status: RecruitmentStatus, mode: ScorerMode) -> float:
    """Return the recruiting-status bonus for the given mode.

    The two tables are kept explicit (not parameterised by a single float)
    so the choice can be audited and tuned independently per state.
    """
    table = _STATUS_BONUS_BENCHMARK if mode == "benchmark" else _STATUS_BONUS_CLINICAL
    return table.get(status, 0.0)


def _resolve_mode(mode: ScorerMode | None) -> ScorerMode:
    if mode is not None:
        return mode
    try:
        configured = get_settings().runner.mode
    except Exception:  # pragma: no cover — settings unavailable in some unit tests
        return "clinical_active"
    return configured if configured in ("benchmark", "clinical_active") else "clinical_active"


def _eligibility_score(ev: TrialEval) -> float:
    """A 0-1 ranking-friendly score from per-criterion outcomes.

    Higher when more inclusions met and fewer exclusions met.
    """
    if ev.n_inclusion + ev.n_exclusion == 0:
        return 0.0
    inc_match = ev.n_inclusion_met / ev.n_inclusion if ev.n_inclusion else 0.0
    inc_failed = ev.n_inclusion_not_met / ev.n_inclusion if ev.n_inclusion else 0.0
    exc_clear = (
        (ev.n_exclusion_not_met / ev.n_exclusion)
        if ev.n_exclusion
        else 1.0  # No exclusions defined → treat as clear
    )
    nei_factor = 1.0 - 0.5 * ev.fraction_nei
    raw = (inc_match + exc_clear - inc_failed) / 2.0 * nei_factor
    return max(0.0, min(1.0, raw))


def _recency_score(last_update: date | None) -> float:
    if last_update is None:
        return 0.0
    days_old = (date.today() - last_update).days
    if days_old < 0:
        return 1.0
    # log-decay: ~1.0 at 0 days, ~0.5 at 365, ~0.25 at 1825
    return float(1.0 / (1.0 + log1p(days_old / 365.0)))


# Whitespace/punctuation normalised to a token set so substring false positives
# like "spain" matching "new spain" do not happen. We keep the table small and
# focused on TREC-relevant geographies; unknown countries fall back to the old
# substring test on city/state only (where the failure mode is less harmful).
_COUNTRY_ALIASES: dict[str, set[str]] = {
    "united states": {"united states", "usa", "u.s.", "u.s.a.", "us", "america"},
    "united kingdom": {"united kingdom", "uk", "u.k.", "great britain", "england", "scotland", "wales"},
    "spain": {"spain", "españa", "espana"},
    "france": {"france"},
    "germany": {"germany", "deutschland"},
    "italy": {"italy", "italia"},
    "canada": {"canada"},
    "australia": {"australia"},
    "japan": {"japan"},
    "china": {"china", "people's republic of china"},
    "south korea": {"south korea", "korea, republic of", "republic of korea"},
    "netherlands": {"netherlands", "the netherlands", "holland"},
    "belgium": {"belgium"},
    "switzerland": {"switzerland"},
    "austria": {"austria"},
    "sweden": {"sweden"},
    "norway": {"norway"},
    "denmark": {"denmark"},
    "finland": {"finland"},
    "ireland": {"ireland"},
    "portugal": {"portugal"},
    "greece": {"greece"},
    "poland": {"poland"},
    "brazil": {"brazil"},
    "mexico": {"mexico"},
    "argentina": {"argentina"},
    "chile": {"chile"},
    "india": {"india"},
    "israel": {"israel"},
    "turkey": {"turkey", "türkiye"},
    "russia": {"russia", "russian federation"},
    "south africa": {"south africa"},
    "new zealand": {"new zealand"},
    "singapore": {"singapore"},
    "taiwan": {"taiwan"},
}


def _country_canonical(text: str | None) -> str | None:
    if not text:
        return None
    norm = text.strip().lower()
    for canonical, aliases in _COUNTRY_ALIASES.items():
        if norm in aliases or norm == canonical:
            return canonical
    return None


def _geography_score(trial: Trial, patient_location: str | None) -> float:
    """Return a geographic-proximity score in [0, 1].

    The scorer prefers exact country matches via an alias table, then falls
    back to substring matches on city/state only. The previous implementation
    matched substrings on country names too which produced false positives
    ("Spain" inside "New Spain"). City/state strings are diverse enough that
    the substring fallback is acceptable as a soft signal.
    """
    if not patient_location or not trial.locations:
        return 0.0
    pl_norm = patient_location.strip().lower()
    pl_country = _country_canonical(patient_location)

    for loc in trial.locations:
        loc_country = _country_canonical(loc.country)
        if pl_country and loc_country and pl_country == loc_country:
            return 1.0
        # City/state fall back to substring (still useful for "Madrid" → "Madrid, Spain")
        for field in (loc.state, loc.city):
            if field and pl_norm in field.lower():
                return 0.7
            if field and field.lower() in pl_norm:
                return 0.5
    return 0.0


def score_trial(
    trial: Trial,
    eval_: TrialEval,
    *,
    weights: ScoreWeights | None = None,
    patient_location: str | None = None,
    mode: ScorerMode | None = None,
    benchmark_soft_veto: bool | None = None,
    retrieval_prior: float = 0.0,
) -> RankedTrial:
    """Compute a RankedTrial score and breakdown.

    ``mode`` selects the recruiting-status table. When ``None`` (default), the
    runner mode from settings is used; pass an explicit value in unit tests.
    """
    w = weights or DEFAULT_WEIGHTS
    resolved_mode: ScorerMode = _resolve_mode(mode)
    if benchmark_soft_veto is None:
        try:
            benchmark_soft_veto = get_settings().runner.benchmark_soft_veto
        except Exception:  # pragma: no cover — settings unavailable in some unit tests
            benchmark_soft_veto = False
    retrieval_prior = max(0.0, min(1.0, float(retrieval_prior or 0.0)))

    veto_triggered = bool(eval_.any_mandatory_inclusion_failed or eval_.any_exclusion_met)

    # Clinical-active mode privileges patient safety and operational readiness:
    # a known mandatory failure or met exclusion must not be ranked as an
    # enrolment candidate. Benchmark mode handles the same evidence below as a
    # graded penalty so TREC relevance is not collapsed into large -1.0 ties.
    if veto_triggered and (resolved_mode == "clinical_active" or not benchmark_soft_veto):
        total = -1.0 + (w.hard_veto_tie_break * retrieval_prior)
        breakdown = ScoreBreakdown(
            total=total,
            eligibility_score=_eligibility_score(eval_),
            recruiting_bonus=0.0,
            phase_alignment=0.0,
            recency=0.0,
            geographic_proximity=0.0,
            retrieval_prior=retrieval_prior,
            mandatory_veto=True,
            nei_penalty=0.0,
            confidence_adjustment=1.0,
        )
        return RankedTrial(
            nct_id=trial.nct_id,
            score=total,
            eval=eval_,
            components=breakdown.model_dump(),
        )

    elig = _eligibility_score(eval_)
    recruit = status_bonus(trial.status, resolved_mode)
    phase = _PHASE_PRIORITY.get(trial.phase, 0.5)
    recency = _recency_score(trial.last_update_date)
    geo = _geography_score(trial, patient_location)
    nei_pen = eval_.fraction_nei

    avg_conf = 0.7  # Default when no per-criterion confidence available
    if eval_.criteria:
        confs = [c.confidence for c in eval_.criteria]
        avg_conf = sum(confs) / len(confs)
    conf_adj = w.confidence_blend * avg_conf + (1 - w.confidence_blend)

    total = (
        w.eligibility * elig
        + w.recruiting * recruit
        + w.phase * phase
        + w.recency * recency
        + w.geography * geo
        + w.retrieval * retrieval_prior
        - w.nei_penalty * nei_pen
    ) * conf_adj
    if veto_triggered:
        total -= w.soft_veto_penalty

    breakdown = ScoreBreakdown(
        total=total,
        eligibility_score=elig,
        recruiting_bonus=recruit,
        phase_alignment=phase,
        recency=recency,
        geographic_proximity=geo,
        retrieval_prior=retrieval_prior,
        mandatory_veto=veto_triggered,
        nei_penalty=nei_pen,
        confidence_adjustment=conf_adj,
    )
    return RankedTrial(
        nct_id=trial.nct_id,
        score=total,
        eval=eval_,
        components=breakdown.model_dump(),
    )
