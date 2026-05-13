"""Hard deterministic filters applied after reranking and before LLM eligibility.

These cut obvious mismatches without consuming LLM budget:
- Age out of range (when both patient age and trial age limits are known)
- Sex mismatch (when patient sex is known and trial restricts to opposite)
- Status not in allowed set, only when the caller enables clinical-active mode
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ..models.agent_state import TrialCandidate
from ..models.patient import PatientProfile, Sex as PatientSex
from ..models.trial import Sex as TrialSex
from ..models.trial import Trial


def _age_in_range(age_years: float | None, t: Trial) -> bool:
    if age_years is None:
        return True
    days = age_years * 365.0
    if t.eligibility.age_range.min_days is not None and days < t.eligibility.age_range.min_days:
        return False
    if t.eligibility.age_range.max_days is not None and days > t.eligibility.age_range.max_days:
        return False
    return True


def _sex_compatible(patient_sex: PatientSex, t: Trial) -> bool:
    if t.eligibility.sex == TrialSex.ALL:
        return True
    if patient_sex == PatientSex.UNKNOWN:
        return True
    if patient_sex == PatientSex.MALE and t.eligibility.sex == TrialSex.MALE:
        return True
    if patient_sex == PatientSex.FEMALE and t.eligibility.sex == TrialSex.FEMALE:
        return True
    return False


DEFAULT_ACTIVE_STATUSES = frozenset(
    {"RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION"}
)


def active_status_set(allowed_statuses: Iterable[str] | None = None) -> set[str]:
    """Return the status set used by clinical-active filtering."""
    if allowed_statuses is None:
        return set(DEFAULT_ACTIVE_STATUSES)
    return {str(s) for s in allowed_statuses}


def apply_hard_filters(
    candidates: list[TrialCandidate],
    trials: Mapping[str, Trial],
    patient: PatientProfile,
    allowed_statuses: Iterable[str] | None = None,
    filter_status: bool = True,
) -> list[TrialCandidate]:
    """Mark candidates as hard_excluded with a reason, return the same list."""
    statuses = active_status_set(allowed_statuses)
    out: list[TrialCandidate] = []
    for c in candidates:
        c.hard_excluded = False
        c.excluded_reason = None
        t = trials.get(c.nct_id)
        if t is None:
            c.hard_excluded = True
            c.excluded_reason = "trial not found in corpus"
            out.append(c)
            continue
        if not _age_in_range(patient.age_years, t):
            c.hard_excluded = True
            c.excluded_reason = (
                f"age {patient.age_years} outside [{t.eligibility.age_range.raw_min}, "
                f"{t.eligibility.age_range.raw_max}]"
            )
        elif not _sex_compatible(patient.sex, t):
            c.hard_excluded = True
            c.excluded_reason = f"sex {patient.sex.value} vs trial {t.eligibility.sex.value}"
        elif filter_status and t.status.value not in statuses:
            c.hard_excluded = True
            c.excluded_reason = f"status {t.status.value} not in {sorted(statuses)}"
        out.append(c)
    return out


def viable_count(candidates: list[TrialCandidate]) -> int:
    """Count candidates that survived hard filters."""
    return sum(1 for c in candidates if not c.hard_excluded)
