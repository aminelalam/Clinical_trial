"""Criterion-level evidence scoring for benchmark ranking experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..models.criterion import CriterionType, Polarity
from ..models.eligibility import EligibilityLabel, TrialLabel
from ..models.ranking import RankedTrial


_TYPE_WEIGHTS = {
    CriterionType.AGE: 0.65,
    CriterionType.SEX: 0.55,
    CriterionType.BIOMARKER: 1.20,
    CriterionType.PRIOR_TREATMENT: 1.10,
    CriterionType.LAB: 1.00,
    CriterionType.COMORBIDITY: 1.00,
    CriterionType.PERFORMANCE_STATUS: 1.00,
    CriterionType.PREGNANCY: 1.10,
    CriterionType.DIAGNOSIS: 1.20,
    CriterionType.CONSENT: 0.15,
    CriterionType.SECTION_HEADER: 0.0,
    CriterionType.OTHER: 0.80,
}

_SUBSTANTIVE_TYPES = {
    CriterionType.BIOMARKER,
    CriterionType.PRIOR_TREATMENT,
    CriterionType.LAB,
    CriterionType.COMORBIDITY,
    CriterionType.PERFORMANCE_STATUS,
    CriterionType.PREGNANCY,
    CriterionType.DIAGNOSIS,
    CriterionType.OTHER,
}


@dataclass(frozen=True)
class CriterionEvidenceResult:
    ranked: RankedTrial
    diagnostics: dict[str, Any]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def criterion_evidence_components(ranked: RankedTrial) -> dict[str, Any]:
    """Summarize per-criterion evidence into auditable numeric components."""
    ev = ranked.eval
    inc_total = inc_met = inc_failed = inc_missing = 0.0
    substantive_inc_total = substantive_inc_met = 0.0
    exc_total = exc_clear = exc_hit = exc_unknown = 0.0
    mandatory_missing = 0.0
    table: list[dict[str, Any]] = []

    for row in ev.criteria:
        crit = row.criterion
        if crit is None or crit.type == CriterionType.SECTION_HEADER:
            continue
        type_weight = _TYPE_WEIGHTS.get(crit.type, 0.80)
        weight = type_weight * (1.0 if crit.is_mandatory else 0.65)
        confidence = _clamp(row.confidence)
        if weight <= 0:
            continue

        penalty = 0.0
        status = row.label.value
        if crit.polarity == Polarity.INCLUSION:
            inc_total += weight
            if crit.type in _SUBSTANTIVE_TYPES:
                substantive_inc_total += weight
            if row.label == EligibilityLabel.MET:
                inc_met += weight * confidence
                if crit.type in _SUBSTANTIVE_TYPES:
                    substantive_inc_met += weight * confidence
            elif row.label == EligibilityLabel.NOT_MET:
                inc_failed += weight * confidence
                penalty = weight * confidence
            else:
                missing = weight * (0.75 if crit.is_mandatory else 0.45)
                inc_missing += missing
                mandatory_missing += missing if crit.is_mandatory else 0.0
                penalty = missing
        else:
            exc_total += weight
            if row.label == EligibilityLabel.NOT_MET:
                exc_clear += weight * confidence
            elif row.label == EligibilityLabel.MET:
                exc_hit += weight * confidence
                penalty = 1.35 * weight * confidence
                status = "exclusion_hit"
            else:
                # Unknown exclusion evidence is weakly informative in benchmark
                # ranking. Penalize explicit exclusion hits, not every missing
                # exclusion statement.
                unknown = weight * (0.25 if crit.is_mandatory else 0.15)
                exc_unknown += unknown
                penalty = unknown

        table.append(
            {
                "criterion_id": row.criterion_id,
                "polarity": crit.polarity.value,
                "type": crit.type.value,
                "label": row.label.value,
                "status": status,
                "confidence": round(confidence, 4),
                "weight": round(weight, 4),
                "penalty": round(penalty, 4),
                "evidence": (row.evidence or "")[:180],
            }
        )

    inclusion_support = _clamp(inc_met / inc_total) if inc_total else 0.0
    substantive_inclusion_support = (
        _clamp(substantive_inc_met / substantive_inc_total)
        if substantive_inc_total
        else 0.0
    )
    inclusion_failure = _clamp(inc_failed / inc_total) if inc_total else 0.0
    inclusion_missing = _clamp(inc_missing / inc_total) if inc_total else 0.0
    exclusion_clear = _clamp(exc_clear / exc_total) if exc_total else 1.0
    exclusion_penalty = _clamp(exc_hit / exc_total) if exc_total else 0.0
    exclusion_uncertainty = _clamp(exc_unknown / exc_total) if exc_total else 0.0
    missing_required_penalty = _clamp(
        mandatory_missing / max(inc_total + (0.45 * exc_total), 1e-9)
    )

    thin_evidence_penalty = 0.0
    if ev.label == TrialLabel.ELIGIBLE:
        if ev.n_inclusion == 0 or ev.n_inclusion_met == 0:
            thin_evidence_penalty += 0.12
        elif ev.n_inclusion_met <= 1 and substantive_inclusion_support < 0.35:
            thin_evidence_penalty += 0.10
        elif ev.n_inclusion_met <= 1:
            thin_evidence_penalty += 0.04
        if ev.fraction_nei >= 0.65:
            thin_evidence_penalty += 0.04

    evidence_score = _clamp(
        0.38 * inclusion_support
        + 0.30 * substantive_inclusion_support
        + 0.12 * exclusion_clear
        + 0.10 * (1.0 - ev.fraction_nei)
        - 0.45 * inclusion_failure
        - 0.72 * exclusion_penalty
        - 0.06 * exclusion_uncertainty
        - 0.42 * missing_required_penalty
        - thin_evidence_penalty
    )

    return {
        "criterion_evidence_score": round(evidence_score, 4),
        "inclusion_support": round(inclusion_support, 4),
        "substantive_inclusion_support": round(substantive_inclusion_support, 4),
        "inclusion_failure": round(inclusion_failure, 4),
        "inclusion_missing": round(inclusion_missing, 4),
        "exclusion_clear": round(exclusion_clear, 4),
        "exclusion_penalty": round(exclusion_penalty, 4),
        "exclusion_uncertainty": round(exclusion_uncertainty, 4),
        "missing_required_penalty": round(missing_required_penalty, 4),
        "thin_evidence_penalty": round(thin_evidence_penalty, 4),
        "criterion_rows": table,
    }


def apply_criterion_evidence_adjustment(
    ranked: RankedTrial,
    *,
    mode: str,
    policy: str,
    weight: float,
) -> CriterionEvidenceResult:
    """Adjust a RankedTrial score using criterion-level evidence.

    This is benchmark-only and label-preserving. It uses already-computed
    criterion evaluations, so it adds no LLM cost.
    """
    if mode != "benchmark" or policy != "score_adjust":
        return CriterionEvidenceResult(
            ranked=ranked,
            diagnostics={
                "policy": policy,
                "effective_policy": "off",
                "mode": mode,
            },
        )

    components = criterion_evidence_components(ranked)
    weight = _clamp(weight)
    evidence_score = float(components["criterion_evidence_score"])

    # Keep hard-vetoed trials mostly ordered by the existing safety logic. The
    # criterion table is still recorded, but score movement is tiny in that band.
    hard_veto = bool((ranked.components or {}).get("mandatory_veto", False))
    effective_weight = weight * (0.25 if hard_veto and ranked.score <= -0.95 else 1.0)
    adjustment = effective_weight * (evidence_score - 0.5)

    new_components = dict(ranked.components or {})
    new_components.update(
        {
            "criterion_evidence_score": evidence_score,
            "criterion_evidence_adjustment": round(adjustment, 6),
            "criterion_inclusion_support": float(components["inclusion_support"]),
            "criterion_substantive_inclusion_support": float(
                components["substantive_inclusion_support"]
            ),
            "criterion_inclusion_failure": float(components["inclusion_failure"]),
            "criterion_inclusion_missing": float(components["inclusion_missing"]),
            "criterion_exclusion_clear": float(components["exclusion_clear"]),
            "criterion_exclusion_penalty": float(components["exclusion_penalty"]),
            "criterion_exclusion_uncertainty": float(components["exclusion_uncertainty"]),
            "criterion_missing_required_penalty": float(
                components["missing_required_penalty"]
            ),
            "criterion_thin_evidence_penalty": float(
                components["thin_evidence_penalty"]
            ),
        }
    )
    adjusted = ranked.model_copy(
        update={
            "score": float(ranked.score + adjustment),
            "components": new_components,
        }
    )
    return CriterionEvidenceResult(
        ranked=adjusted,
        diagnostics={
            "policy": policy,
            "effective_policy": "score_adjust",
            "mode": mode,
            "weight": weight,
            "effective_weight": round(effective_weight, 4),
            "original_score": round(float(ranked.score), 6),
            "adjusted_score": round(float(adjusted.score), 6),
            "adjustment": round(adjustment, 6),
            **components,
        },
    )
