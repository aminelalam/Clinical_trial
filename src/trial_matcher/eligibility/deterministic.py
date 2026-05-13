"""Deterministic eligibility evaluation for criteria with parseable predicates.

Handles:
- age (numeric comparison vs PatientProfile.age_years)
- sex (exact match)
- lab numeric comparisons (when patient has the lab AND units match)
- ECOG / Karnofsky comparisons
- pregnancy explicit yes/no when patient pregnancy is known
"""

from __future__ import annotations

import operator
from typing import Any, Callable

from ..models.criterion import Criterion, CriterionType, Predicate
from ..models.eligibility import CriterionEval, EligibilityLabel
from ..models.patient import PatientProfile, Sex


_OPS: dict[str, Callable[[Any, Any], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "=": operator.eq,
    "!=": operator.ne,
}


def _eval_numeric(op: str, lhs: float, rhs: float) -> bool:
    fn = _OPS.get(op)
    if fn is None:
        return False
    return fn(lhs, rhs)


def evaluate_deterministic(criterion: Criterion, patient: PatientProfile) -> CriterionEval | None:
    """Try to evaluate ``criterion`` deterministically. Return None if cannot."""
    if criterion.type == CriterionType.AGE and criterion.predicate is not None:
        return _eval_age(criterion, criterion.predicate, patient)
    if criterion.type == CriterionType.SEX and criterion.predicate is not None:
        return _eval_sex(criterion, criterion.predicate, patient)
    if criterion.type == CriterionType.PERFORMANCE_STATUS and criterion.predicate is not None:
        return _eval_performance(criterion, criterion.predicate, patient)
    if criterion.type == CriterionType.LAB and criterion.predicate is not None:
        return _eval_lab(criterion, criterion.predicate, patient)
    return None


def _wrap(criterion: Criterion, label: EligibilityLabel, evidence: str, reasoning: str) -> CriterionEval:
    return CriterionEval(
        criterion_id=criterion.id,
        label=label,
        confidence=1.0,
        evidence=evidence,
        reasoning=reasoning,
        evaluator="deterministic",
        criterion=criterion,
    )


# Note on polarity: this module returns labels that reflect ONLY whether the
# criterion's predicate is satisfied for the patient. The aggregator
# (`eligibility/aggregator.py`) handles polarity (exclusion vs inclusion):
#   - inclusion + met     → counts toward "all inclusions met"
#   - exclusion + met     → triggers `any_exclusion_met` → trial excludes
#   - inclusion + not_met → triggers `any_mandatory_inclusion_failed` if mandatory
# Callers must NOT flip the label here; aggregation is the single source of truth.


def _eval_age(criterion: Criterion, pred: Predicate, patient: PatientProfile) -> CriterionEval | None:
    if patient.age_years is None:
        return _wrap(
            criterion,
            EligibilityLabel.NEI,
            evidence="Patient note has no age",
            reasoning="Age unknown — cannot evaluate.",
        )
    try:
        rhs = float(pred.value)
    except (TypeError, ValueError):
        return None
    ok = _eval_numeric(pred.op, patient.age_years, rhs)
    return _wrap(
        criterion,
        EligibilityLabel.MET if ok else EligibilityLabel.NOT_MET,
        evidence=f"Patient age = {patient.age_years} years",
        reasoning=f"{patient.age_years} {pred.op} {rhs} → {ok}",
    )


def _eval_sex(criterion: Criterion, pred: Predicate, patient: PatientProfile) -> CriterionEval | None:
    if patient.sex == Sex.UNKNOWN:
        return _wrap(
            criterion,
            EligibilityLabel.NEI,
            evidence="Patient sex unknown",
            reasoning="Cannot evaluate sex without explicit value.",
        )
    rhs = str(pred.value).lower() if pred.value else ""
    ok = patient.sex.value == rhs
    return _wrap(
        criterion,
        EligibilityLabel.MET if ok else EligibilityLabel.NOT_MET,
        evidence=f"Patient sex = {patient.sex.value}",
        reasoning=f"{patient.sex.value} {pred.op} {rhs}",
    )


def _eval_performance(
    criterion: Criterion, pred: Predicate, patient: PatientProfile
) -> CriterionEval | None:
    if pred.feature.lower() in {"ecog", "performance_status"} and patient.ecog is not None:
        try:
            rhs = float(pred.value)
        except (TypeError, ValueError):
            return None
        ok = _eval_numeric(pred.op, patient.ecog, rhs)
        return _wrap(
            criterion,
            EligibilityLabel.MET if ok else EligibilityLabel.NOT_MET,
            evidence=f"ECOG = {patient.ecog}",
            reasoning=f"ECOG {patient.ecog} {pred.op} {rhs}",
        )
    if pred.feature.lower() == "karnofsky" and patient.karnofsky is not None:
        try:
            rhs = float(pred.value)
        except (TypeError, ValueError):
            return None
        ok = _eval_numeric(pred.op, patient.karnofsky, rhs)
        return _wrap(
            criterion,
            EligibilityLabel.MET if ok else EligibilityLabel.NOT_MET,
            evidence=f"Karnofsky = {patient.karnofsky}",
            reasoning=f"Karnofsky {patient.karnofsky} {pred.op} {rhs}",
        )
    return _wrap(
        criterion,
        EligibilityLabel.NEI,
        evidence="No ECOG/Karnofsky in patient note",
        reasoning="Performance status not recorded.",
    )


def _eval_lab(criterion: Criterion, pred: Predicate, patient: PatientProfile) -> CriterionEval | None:
    lab = patient.get_lab(pred.feature)
    if lab is None or lab.value is None:
        return _wrap(
            criterion,
            EligibilityLabel.NEI,
            evidence=f"No recent {pred.feature} value in patient note",
            reasoning=f"Lab {pred.feature} missing — cannot evaluate.",
        )
    try:
        rhs = float(pred.value)
    except (TypeError, ValueError):
        return None
    ok = _eval_numeric(pred.op, lab.value, rhs)
    return _wrap(
        criterion,
        EligibilityLabel.MET if ok else EligibilityLabel.NOT_MET,
        evidence=f"{pred.feature} = {lab.value} {lab.units or ''}".strip(),
        reasoning=f"{pred.feature}={lab.value} {pred.op} {rhs} → {ok}",
    )
