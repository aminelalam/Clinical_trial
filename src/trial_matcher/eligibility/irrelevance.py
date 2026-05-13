"""Multi-signal benchmark policy for mapping retrieval noise to class 0."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..models.agent_state import TrialCandidate
from ..models.eligibility import TrialEval, TrialLabel
from ..models.patient import PatientProfile
from ..models.search_plan import SearchPlan
from ..models.trial import Trial


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.-]{1,}")
_GENERIC_CLINICAL = {
    "advanced",
    "cancer",
    "carcinoma",
    "disease",
    "malignancy",
    "malignant",
    "metastatic",
    "neoplasm",
    "neoplasms",
    "patients",
    "recurrent",
    "solid",
    "stage",
    "study",
    "therapy",
    "trial",
    "tumor",
    "tumors",
    "with",
}
_STOPWORDS = _GENERIC_CLINICAL | {
    "about",
    "after",
    "all",
    "and",
    "any",
    "are",
    "been",
    "but",
    "by",
    "can",
    "due",
    "has",
    "had",
    "for",
    "from",
    "have",
    "her",
    "his",
    "how",
    "into",
    "its",
    "may",
    "new",
    "nor",
    "not",
    "of",
    "off",
    "old",
    "one",
    "only",
    "or",
    "our",
    "out",
    "per",
    "than",
    "that",
    "the",
    "this",
    "to",
    "two",
    "was",
    "were",
    "who",
    "will",
}


@dataclass(frozen=True)
class IrrelevancePolicyResult:
    trial_eval: TrialEval
    diagnostics: dict[str, Any]


def _tokens(text: str | None, *, drop_generic: bool = False) -> set[str]:
    if not text:
        return set()
    stop = _STOPWORDS if drop_generic else (_STOPWORDS - _GENERIC_CLINICAL)
    return {tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in stop}


def _patient_condition_tokens(patient: PatientProfile, plan: SearchPlan | None) -> set[str]:
    parts: list[str] = []
    if patient.primary_diagnosis:
        parts.append(patient.primary_diagnosis)
    if patient.primary_diagnosis_stage:
        parts.append(patient.primary_diagnosis_stage)
    parts.extend(patient.secondary_diagnoses)
    for concept in patient.mesh_concepts or []:
        if isinstance(concept, dict):
            parts.append(str(concept.get("name") or ""))
            parts.extend(str(s) for s in (concept.get("synonyms") or [])[:4])
    if plan is not None:
        parts.append(plan.primary_disease_query)
        parts.extend(plan.expansion_terms[:8])
    return _tokens(" ".join(parts), drop_generic=True)


def _trial_condition_title_tokens(trial: Trial) -> set[str]:
    parts = [
        trial.title,
        trial.official_title or "",
        " ".join(trial.conditions),
        " ".join(trial.keywords),
    ]
    return _tokens(" ".join(parts), drop_generic=True)


def _patient_biomarker_intervention_tokens(
    patient: PatientProfile,
    plan: SearchPlan | None,
) -> set[str]:
    parts: list[str] = []
    for biomarker in patient.biomarkers:
        parts.extend([biomarker.name, biomarker.status, biomarker.value or ""])
    for treatment in patient.prior_treatments:
        parts.extend([treatment.name, treatment.category or "", treatment.response or ""])
    if plan is not None:
        parts.extend(plan.retrieval_priorities)
        parts.extend(plan.risk_flags)
    return _tokens(" ".join(parts), drop_generic=True)


def _trial_biomarker_intervention_tokens(trial: Trial) -> set[str]:
    parts = [
        trial.title,
        trial.brief_summary,
        trial.detailed_description or "",
        " ".join(trial.interventions),
        trial.eligibility.raw_text,
    ]
    return _tokens(" ".join(parts), drop_generic=True)


def retrieval_prior_for_candidate(
    candidate: TrialCandidate,
    *,
    max_rank: int,
    fallback_rank: int = 0,
) -> float:
    rank = candidate.rank if candidate.rank > 0 else fallback_rank
    if rank <= 0:
        return 0.0
    max_rank = max(max_rank, rank, 1)
    if max_rank <= 1:
        return 1.0
    return max(0.0, min(1.0, 1.0 - ((rank - 1) / (max_rank - 1))))


def apply_multisignal_irrelevance_policy(
    *,
    trial: Trial,
    trial_eval: TrialEval,
    candidate: TrialCandidate,
    patient: PatientProfile,
    plan: SearchPlan | None,
    enabled: bool,
    mode: str,
    retrieval_prior: float,
    min_nei_fraction: float,
    max_inclusion_met: int,
    max_retrieval_prior: float,
    min_signal_count: int,
) -> IrrelevancePolicyResult:
    """Convert clear retrieval noise to qrel-0 only when several signals agree."""
    diagnostics: dict[str, Any] = {
        "enabled": bool(enabled),
        "mode": mode,
        "original_label": trial_eval.label.value,
        "final_label": trial_eval.label.value,
        "retrieval_prior": round(float(retrieval_prior), 4),
        "activated": False,
        "blocked_reason": None,
    }
    if not enabled:
        diagnostics["blocked_reason"] = "disabled"
        return IrrelevancePolicyResult(trial_eval=trial_eval, diagnostics=diagnostics)
    if mode != "benchmark":
        diagnostics["blocked_reason"] = "non_benchmark_mode"
        return IrrelevancePolicyResult(trial_eval=trial_eval, diagnostics=diagnostics)
    if trial_eval.label == TrialLabel.ELIGIBLE:
        diagnostics["blocked_reason"] = "eligible_label"
        return IrrelevancePolicyResult(trial_eval=trial_eval, diagnostics=diagnostics)
    if trial_eval.any_exclusion_met:
        diagnostics["blocked_reason"] = "explicit_exclusion_met"
        return IrrelevancePolicyResult(trial_eval=trial_eval, diagnostics=diagnostics)
    if trial_eval.any_mandatory_inclusion_failed:
        diagnostics["blocked_reason"] = "mandatory_inclusion_failed"
        return IrrelevancePolicyResult(trial_eval=trial_eval, diagnostics=diagnostics)

    patient_condition = _patient_condition_tokens(patient, plan)
    trial_condition = _trial_condition_title_tokens(trial)
    condition_overlap = sorted(patient_condition & trial_condition)

    patient_bio_int = _patient_biomarker_intervention_tokens(patient, plan)
    trial_bio_int = _trial_biomarker_intervention_tokens(trial)
    bio_int_overlap = sorted(patient_bio_int & trial_bio_int)

    signals = {
        "high_nei": trial_eval.fraction_nei >= min_nei_fraction,
        "zero_inclusions_met": trial_eval.n_inclusion_met <= max_inclusion_met,
        "low_retrieval_prior": retrieval_prior <= max_retrieval_prior,
        "low_condition_title_support": bool(patient_condition) and not condition_overlap,
        "no_biomarker_intervention_match": bool(patient_bio_int) and not bio_int_overlap,
    }
    signal_count = sum(1 for value in signals.values() if value)

    diagnostics.update(
        {
            "signals": signals,
            "signal_count": signal_count,
            "min_signal_count": int(min_signal_count),
            "condition_title_overlap": condition_overlap[:10],
            "condition_title_patient_terms": sorted(patient_condition)[:20],
            "biomarker_intervention_overlap": bio_int_overlap[:10],
            "biomarker_intervention_patient_terms": sorted(patient_bio_int)[:20],
        }
    )

    core_ready = (
        signals["high_nei"]
        and signals["zero_inclusions_met"]
        and signals["low_retrieval_prior"]
    )
    if not core_ready:
        diagnostics["blocked_reason"] = "missing_core_irrelevance_signals"
        return IrrelevancePolicyResult(trial_eval=trial_eval, diagnostics=diagnostics)

    if signal_count < max(1, min_signal_count):
        diagnostics["blocked_reason"] = "insufficient_signals"
        return IrrelevancePolicyResult(trial_eval=trial_eval, diagnostics=diagnostics)

    converted = trial_eval.model_copy(update={"label": TrialLabel.IRRELEVANT})
    diagnostics["activated"] = True
    diagnostics["final_label"] = converted.label.value
    return IrrelevancePolicyResult(trial_eval=converted, diagnostics=diagnostics)
