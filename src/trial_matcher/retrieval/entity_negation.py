"""Lightweight entity/negation features for benchmark candidate reranking."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ..models.agent_state import TrialCandidate
from ..models.patient import PatientProfile, Sex as PatientSex
from ..models.search_plan import SearchPlan
from ..models.trial import Sex as TrialSex
from ..models.trial import Trial


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+./-]{1,}")
_SPLIT_RE = re.compile(r"[.;:\n\r]+")
_NEGATION_CUES = {
    "absence",
    "denied",
    "denies",
    "free",
    "no",
    "none",
    "not",
    "without",
}
_NEGATION_BIGRAMS = {
    ("absence", "of"),
    ("free", "of"),
    ("negative", "for"),
    ("no", "evidence"),
    ("no", "history"),
}
_STOPWORDS = {
    "about",
    "after",
    "all",
    "also",
    "and",
    "as",
    "any",
    "are",
    "at",
    "be",
    "been",
    "but",
    "by",
    "can",
    "care",
    "case",
    "clinical",
    "cohort",
    "due",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "history",
    "in",
    "into",
    "is",
    "its",
    "may",
    "new",
    "negative",
    "nor",
    "not",
    "of",
    "off",
    "old",
    "only",
    "or",
    "our",
    "out",
    "patient",
    "patients",
    "per",
    "prior",
    "she",
    "study",
    "that",
    "the",
    "their",
    "therapy",
    "this",
    "to",
    "trial",
    "two",
    "was",
    "were",
    "who",
    "will",
    "with",
    "within",
    "year",
    "years",
}
_GENERIC_CLINICAL = {
    "advanced",
    "cancer",
    "carcinoma",
    "condition",
    "disease",
    "disorder",
    "malignancy",
    "malignant",
    "metastatic",
    "neoplasm",
    "neoplasms",
    "recurrent",
    "solid",
    "stage",
    "syndrome",
    "tumor",
    "tumors",
}


@dataclass(frozen=True)
class EntityNegationResult:
    candidates: list[TrialCandidate]
    diagnostics: dict[str, Any]


def _normalize_token(token: str) -> str:
    token = token.lower().strip("-_.,;:()[]{}")
    if token.endswith("'s"):
        token = token[:-2]
    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token


def _tokens(text: str | None, *, drop_generic: bool = True) -> set[str]:
    if not text:
        return set()
    stop = set(_STOPWORDS)
    if drop_generic:
        stop |= _GENERIC_CLINICAL
    return {
        normalized
        for raw in _TOKEN_RE.findall(text.lower())
        if (normalized := _normalize_token(raw)) and normalized not in stop
    }


def _entity_tokens(values: Iterable[str | None], *, drop_generic: bool = True) -> set[str]:
    return _tokens(" ".join(v or "" for v in values), drop_generic=drop_generic)


def _negated_terms(text: str | None, *, window: int = 7) -> set[str]:
    if not text:
        return set()
    out: set[str] = set()
    for segment in _SPLIT_RE.split(text.lower()):
        toks = [_normalize_token(tok) for tok in _TOKEN_RE.findall(segment)]
        toks = [tok for tok in toks if tok]
        for index, tok in enumerate(toks):
            cue = tok in _NEGATION_CUES
            if not cue and index + 1 < len(toks):
                cue = (tok, toks[index + 1]) in _NEGATION_BIGRAMS
            if not cue:
                continue
            start = index + 1
            if index + 1 < len(toks) and (tok, toks[index + 1]) in _NEGATION_BIGRAMS:
                start = index + 2
            for neg_tok in toks[start : start + window]:
                if neg_tok in _STOPWORDS or neg_tok in _GENERIC_CLINICAL:
                    continue
                out.add(neg_tok)
    return out


def _overlap_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = left & right
    if not overlap:
        return 0.0
    return max(0.0, min(1.0, len(overlap) / math.sqrt(len(left) * len(right))))


def _patient_features(patient: PatientProfile, plan: SearchPlan | None) -> dict[str, set[str]]:
    disease_parts: list[str] = [
        patient.primary_diagnosis or "",
        patient.primary_diagnosis_stage or "",
        *patient.secondary_diagnoses,
    ]
    for concept in patient.mesh_concepts or []:
        if isinstance(concept, dict):
            disease_parts.append(str(concept.get("name") or ""))
            disease_parts.extend(str(s) for s in (concept.get("synonyms") or [])[:8])
    if plan is not None:
        disease_parts.extend([plan.primary_disease_query, *plan.expansion_terms[:12]])

    biomarker_parts = [
        f"{b.name} {b.status} {b.value or ''}"
        for b in patient.biomarkers
    ]
    treatment_parts = [
        f"{t.name} {t.category or ''} {t.response or ''}"
        for t in patient.prior_treatments
    ]
    comorbidity_parts = [
        c.name for c in patient.comorbidities if c.active is not False
    ]
    raw_parts = [patient.raw_text, patient.free_text_residual or ""]
    if plan is not None:
        treatment_parts.extend(plan.retrieval_priorities[:12])
        comorbidity_parts.extend(plan.risk_flags[:12])

    disease = _entity_tokens(disease_parts)
    biomarker = _entity_tokens(biomarker_parts, drop_generic=False)
    treatment = _entity_tokens(treatment_parts)
    comorbidity = _entity_tokens(comorbidity_parts)
    negated = _negated_terms("\n".join(raw_parts))
    positive = (
        disease
        | biomarker
        | treatment
        | comorbidity
        | (_tokens(patient.raw_text) - negated)
    )
    return {
        "disease": disease,
        "biomarker": biomarker,
        "treatment": treatment,
        "comorbidity": comorbidity,
        "positive": positive,
        "negated": negated,
    }


def _trial_features(trial: Trial, candidate: TrialCandidate) -> dict[str, set[str]]:
    title_condition = _entity_tokens(
        [
            candidate.title,
            candidate.snippet,
            trial.title,
            trial.official_title or "",
            " ".join(trial.conditions),
            " ".join(trial.keywords),
        ]
    )
    intervention = _entity_tokens(
        [
            " ".join(trial.interventions),
            trial.brief_summary,
            trial.detailed_description or "",
        ],
        drop_generic=False,
    )
    inclusion = _tokens(trial.eligibility.inclusion_text or trial.eligibility.raw_text)
    exclusion = _tokens(trial.eligibility.exclusion_text)
    all_terms = title_condition | intervention | inclusion | exclusion
    negated = _negated_terms(
        "\n".join(
            [
                trial.title,
                trial.official_title or "",
                trial.brief_summary,
                trial.detailed_description or "",
                trial.eligibility.raw_text,
            ]
        )
    )
    return {
        "title_condition": title_condition,
        "intervention": intervention,
        "inclusion": inclusion,
        "exclusion": exclusion,
        "all": all_terms,
        "negated": negated,
    }


def _sex_alignment(patient: PatientProfile, trial: Trial) -> float:
    if patient.sex == PatientSex.UNKNOWN or trial.eligibility.sex == TrialSex.ALL:
        return 0.5
    if patient.sex == PatientSex.MALE and trial.eligibility.sex == TrialSex.MALE:
        return 1.0
    if patient.sex == PatientSex.FEMALE and trial.eligibility.sex == TrialSex.FEMALE:
        return 1.0
    return 0.0


def entity_negation_components(
    *,
    candidate: TrialCandidate,
    trial: Trial,
    patient: PatientProfile,
    plan: SearchPlan | None,
) -> dict[str, Any]:
    """Compute deterministic entity/negation evidence for one candidate."""
    patient_terms = _patient_features(patient, plan)
    trial_terms = _trial_features(trial, candidate)

    disease_support = _overlap_score(
        patient_terms["disease"] | patient_terms["comorbidity"],
        trial_terms["title_condition"],
    )
    biomarker_support = _overlap_score(
        patient_terms["biomarker"],
        trial_terms["all"],
    )
    intervention_support = _overlap_score(
        patient_terms["treatment"],
        trial_terms["intervention"] | trial_terms["inclusion"],
    )
    inclusion_support = _overlap_score(
        patient_terms["positive"],
        trial_terms["inclusion"],
    )
    exclusion_overlap = _overlap_score(
        patient_terms["positive"],
        trial_terms["exclusion"],
    )
    negated_trial_match = sorted(
        patient_terms["negated"]
        & (trial_terms["title_condition"] | trial_terms["intervention"] | trial_terms["inclusion"])
    )
    trial_negates_patient = sorted(trial_terms["negated"] & patient_terms["positive"])
    negation_penalty = min(1.0, 0.25 * (len(negated_trial_match) + len(trial_negates_patient)))
    positive_overlap = sorted(patient_terms["positive"] & trial_terms["all"])
    specific_overlap = sorted(
        (
            patient_terms["disease"]
            | patient_terms["biomarker"]
            | patient_terms["treatment"]
            | patient_terms["comorbidity"]
        )
        & trial_terms["all"]
    )
    generic_only_match = bool(positive_overlap) and not specific_overlap
    sex_alignment = _sex_alignment(patient, trial)
    clinical_support = max(
        disease_support,
        biomarker_support,
        intervention_support,
        inclusion_support,
    )
    sex_bonus = 0.08 * sex_alignment if clinical_support >= 0.05 else 0.0

    entity_score = (
        0.4 * disease_support
        + 0.2 * inclusion_support
        + 0.18 * max(biomarker_support, intervention_support)
        + sex_bonus
        - 0.18 * exclusion_overlap
        - 0.24 * negation_penalty
        - (0.08 if generic_only_match else 0.0)
    )
    entity_score = max(0.0, min(1.0, entity_score))
    return {
        "entity_score": round(entity_score, 4),
        "disease_support": round(disease_support, 4),
        "biomarker_support": round(biomarker_support, 4),
        "intervention_support": round(intervention_support, 4),
        "inclusion_support": round(inclusion_support, 4),
        "exclusion_overlap": round(exclusion_overlap, 4),
        "negation_penalty": round(negation_penalty, 4),
        "sex_alignment": round(sex_alignment, 4),
        "generic_only_match": generic_only_match,
        "positive_overlap_terms": positive_overlap[:12],
        "specific_overlap_terms": specific_overlap[:12],
        "patient_negated_trial_terms": negated_trial_match[:12],
        "trial_negated_patient_terms": trial_negates_patient[:12],
    }


def _retrieval_norms(candidates: list[TrialCandidate]) -> dict[str, float]:
    scores = [float(c.score or 0.0) for c in candidates]
    min_score = min(scores, default=0.0)
    max_score = max(scores, default=0.0)
    spread = max_score - min_score
    out: dict[str, float] = {}
    for index, cand in enumerate(candidates):
        if spread > 0:
            value = (float(cand.score or 0.0) - min_score) / spread
        elif len(candidates) > 1:
            value = 1.0 - (index / (len(candidates) - 1))
        else:
            value = 1.0
        out[cand.nct_id] = max(0.0, min(1.0, value))
    return out


def apply_entity_negation_rerank(
    candidates: Iterable[TrialCandidate],
    *,
    get_trial: Callable[[str], Trial | None],
    patient: PatientProfile | None,
    plan: SearchPlan | None,
    mode: str,
    policy: str,
    weight: float,
    protect_top: int,
) -> EntityNegationResult:
    """Audit or rerank final candidates using entity/negation evidence.

    The policy is benchmark-only and never reads qrels. ``audit`` records
    features but returns candidates unchanged; ``rerank_final`` blends the
    original retrieval/reranker score with the deterministic entity score.
    """
    original = list(candidates)
    effective_policy = policy if mode == "benchmark" and patient is not None else "off"
    if effective_policy not in {"audit", "rerank_final"}:
        return EntityNegationResult(
            candidates=original,
            diagnostics={
                "policy": policy,
                "effective_policy": "off",
                "mode": mode,
                "candidate_count": len(original),
            },
        )

    norms = _retrieval_norms(original)
    weight = max(0.0, min(1.0, float(weight)))
    protect_top = max(0, int(protect_top or 0))
    scored: list[TrialCandidate] = []
    by_id: dict[str, Any] = {}
    original_order = {cand.nct_id: index for index, cand in enumerate(original)}
    original_scores = [float(c.score or 0.0) for c in original]
    min_score = min(original_scores, default=0.0)
    max_score = max(original_scores, default=0.0)
    spread = max(max_score - min_score, 1.0)

    for index, cand in enumerate(original):
        trial = get_trial(cand.nct_id)
        if trial is None or patient is None:
            scored.append(cand)
            by_id[cand.nct_id] = {"trial_found": False}
            continue
        components = entity_negation_components(
            candidate=cand,
            trial=trial,
            patient=patient,
            plan=plan,
        )
        retrieval_norm = norms.get(cand.nct_id, 0.0)
        entity_score = float(components["entity_score"])
        protected = index < protect_top
        blended_norm = (1.0 - weight) * retrieval_norm + weight * entity_score
        new_score = min_score + (blended_norm * spread)
        if protected and new_score < float(cand.score or 0.0):
            new_score = float(cand.score or 0.0)
        metadata = dict(cand.retrieval_metadata or {})
        metadata["entity_negation"] = {
            **components,
            "original_score": float(cand.score or 0.0),
            "input_position": index + 1,
            "retrieval_norm": round(retrieval_norm, 4),
            "blended_norm": round(blended_norm, 4),
            "protected_top": protected,
        }
        by_id[cand.nct_id] = metadata["entity_negation"]
        if effective_policy == "rerank_final":
            scored.append(
                cand.model_copy(
                    update={
                        "score": float(new_score),
                        "retrieval_metadata": metadata,
                    }
                )
            )
        else:
            scored.append(cand.model_copy(update={"retrieval_metadata": metadata}))

    if effective_policy == "rerank_final":
        scored.sort(
            key=lambda cand: (
                cand.score is None,
                -(cand.score or 0.0),
                original_order.get(cand.nct_id, 10**9),
                cand.nct_id,
            )
        )
    for output_index, cand in enumerate(scored):
        metadata = dict(cand.retrieval_metadata or {})
        entity_metadata = metadata.get("entity_negation")
        if not isinstance(entity_metadata, dict):
            continue
        entity_metadata = {**entity_metadata, "output_position": output_index + 1}
        metadata["entity_negation"] = entity_metadata
        scored[output_index] = cand.model_copy(update={"retrieval_metadata": metadata})
        by_id[cand.nct_id] = entity_metadata

    return EntityNegationResult(
        candidates=scored,
        diagnostics={
            "policy": policy,
            "effective_policy": effective_policy,
            "mode": mode,
            "candidate_count": len(original),
            "weight": weight,
            "protect_top": protect_top,
            "input_ids": [c.nct_id for c in original[:30]],
            "output_ids": [c.nct_id for c in scored[:30]],
            "changed_order": [c.nct_id for c in original] != [c.nct_id for c in scored],
            "by_id": by_id,
        },
    )
