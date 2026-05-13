"""Criterion-level evidence scoring tests."""

from __future__ import annotations

from trial_matcher.models.criterion import Criterion, CriterionType, Polarity
from trial_matcher.models.eligibility import CriterionEval, EligibilityLabel, TrialEval, TrialLabel
from trial_matcher.models.ranking import RankedTrial
from trial_matcher.ranking.criterion_evidence import apply_criterion_evidence_adjustment


def _criterion(
    criterion_id: str,
    *,
    polarity: Polarity,
    criterion_type: CriterionType,
    mandatory: bool = True,
) -> Criterion:
    return Criterion(
        id=criterion_id,
        polarity=polarity,
        raw_text=criterion_id,
        type=criterion_type,
        is_mandatory=mandatory,
    )


def _eval_row(
    criterion: Criterion,
    label: EligibilityLabel,
    *,
    confidence: float = 0.9,
) -> CriterionEval:
    return CriterionEval(
        criterion_id=criterion.id,
        criterion=criterion,
        label=label,
        confidence=confidence,
        evidence=f"evidence for {criterion.id}",
    )


def _ranked(
    rows: list[CriterionEval],
    *,
    label: TrialLabel = TrialLabel.ELIGIBLE,
    score: float = 0.5,
    components: dict[str, float] | None = None,
) -> RankedTrial:
    return RankedTrial(
        nct_id="NCT1",
        score=score,
        eval=TrialEval(
            nct_id="NCT1",
            label=label,
            criteria=rows,
            n_inclusion=sum(1 for r in rows if r.criterion.polarity == Polarity.INCLUSION),
            n_exclusion=sum(1 for r in rows if r.criterion.polarity == Polarity.EXCLUSION),
            n_inclusion_met=sum(
                1
                for r in rows
                if r.criterion.polarity == Polarity.INCLUSION
                and r.label == EligibilityLabel.MET
            ),
            n_inclusion_not_met=sum(
                1
                for r in rows
                if r.criterion.polarity == Polarity.INCLUSION
                and r.label == EligibilityLabel.NOT_MET
            ),
            n_inclusion_nei=sum(
                1
                for r in rows
                if r.criterion.polarity == Polarity.INCLUSION
                and r.label == EligibilityLabel.NEI
            ),
            n_exclusion_met=sum(
                1
                for r in rows
                if r.criterion.polarity == Polarity.EXCLUSION
                and r.label == EligibilityLabel.MET
            ),
            n_exclusion_not_met=sum(
                1
                for r in rows
                if r.criterion.polarity == Polarity.EXCLUSION
                and r.label == EligibilityLabel.NOT_MET
            ),
            n_exclusion_nei=sum(
                1
                for r in rows
                if r.criterion.polarity == Polarity.EXCLUSION
                and r.label == EligibilityLabel.NEI
            ),
            any_mandatory_inclusion_failed=any(
                r.criterion.polarity == Polarity.INCLUSION
                and r.criterion.is_mandatory
                and r.label == EligibilityLabel.NOT_MET
                for r in rows
            ),
            any_exclusion_met=any(
                r.criterion.polarity == Polarity.EXCLUSION
                and r.label == EligibilityLabel.MET
                for r in rows
            ),
            fraction_nei=sum(1 for r in rows if r.label == EligibilityLabel.NEI)
            / max(len(rows), 1),
        ),
        components=components or {},
    )


def test_criterion_evidence_boosts_strong_inclusion_and_clear_exclusion():
    diagnosis = _criterion(
        "metastatic breast cancer",
        polarity=Polarity.INCLUSION,
        criterion_type=CriterionType.DIAGNOSIS,
    )
    biomarker = _criterion(
        "HER2 positive",
        polarity=Polarity.INCLUSION,
        criterion_type=CriterionType.BIOMARKER,
    )
    prior_chemo = _criterion(
        "prior chemotherapy",
        polarity=Polarity.EXCLUSION,
        criterion_type=CriterionType.PRIOR_TREATMENT,
    )
    ranked = _ranked(
        [
            _eval_row(diagnosis, EligibilityLabel.MET),
            _eval_row(biomarker, EligibilityLabel.MET),
            _eval_row(prior_chemo, EligibilityLabel.NOT_MET),
        ]
    )

    result = apply_criterion_evidence_adjustment(
        ranked,
        mode="benchmark",
        policy="score_adjust",
        weight=0.18,
    )

    assert result.ranked.score > ranked.score
    assert result.diagnostics["criterion_evidence_score"] > 0.5
    assert result.ranked.components["criterion_inclusion_support"] > 0.8
    assert result.ranked.components["criterion_exclusion_penalty"] == 0.0


def test_criterion_evidence_penalizes_exclusion_hits_and_missing_required_support():
    diagnosis = _criterion(
        "metastatic disease",
        polarity=Polarity.INCLUSION,
        criterion_type=CriterionType.DIAGNOSIS,
    )
    ecog = _criterion(
        "ECOG <= 1",
        polarity=Polarity.INCLUSION,
        criterion_type=CriterionType.PERFORMANCE_STATUS,
    )
    prior_chemo = _criterion(
        "prior chemotherapy",
        polarity=Polarity.EXCLUSION,
        criterion_type=CriterionType.PRIOR_TREATMENT,
    )
    ranked = _ranked(
        [
            _eval_row(diagnosis, EligibilityLabel.MET),
            _eval_row(ecog, EligibilityLabel.NEI),
            _eval_row(prior_chemo, EligibilityLabel.MET),
        ],
        label=TrialLabel.EXCLUDES,
    )

    result = apply_criterion_evidence_adjustment(
        ranked,
        mode="benchmark",
        policy="score_adjust",
        weight=0.18,
    )

    assert result.ranked.score < ranked.score
    assert result.ranked.components["criterion_exclusion_penalty"] > 0.0
    assert result.ranked.components["criterion_missing_required_penalty"] > 0.0


def test_criterion_evidence_is_benchmark_only_and_off_by_default():
    criterion = _criterion(
        "adult patient",
        polarity=Polarity.INCLUSION,
        criterion_type=CriterionType.AGE,
    )
    ranked = _ranked([_eval_row(criterion, EligibilityLabel.MET)])

    clinical = apply_criterion_evidence_adjustment(
        ranked,
        mode="clinical_active",
        policy="score_adjust",
        weight=0.18,
    )
    off = apply_criterion_evidence_adjustment(
        ranked,
        mode="benchmark",
        policy="off",
        weight=0.18,
    )

    assert clinical.ranked == ranked
    assert clinical.diagnostics["effective_policy"] == "off"
    assert off.ranked == ranked
    assert off.diagnostics["effective_policy"] == "off"


def test_criterion_evidence_components_stay_numeric_for_rank_output():
    criterion = _criterion(
        "adult patient",
        polarity=Polarity.INCLUSION,
        criterion_type=CriterionType.AGE,
    )
    ranked = _ranked([_eval_row(criterion, EligibilityLabel.MET)])

    result = apply_criterion_evidence_adjustment(
        ranked,
        mode="benchmark",
        policy="score_adjust",
        weight=0.18,
    )

    assert result.diagnostics["criterion_rows"]
    assert all(isinstance(value, (int, float, bool)) for value in result.ranked.components.values())
