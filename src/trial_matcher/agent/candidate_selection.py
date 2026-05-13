"""Benchmark-only candidate selection policies before eligibility evaluation."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ..models.agent_state import TrialCandidate
from ..models.patient import PatientProfile
from ..models.search_plan import SearchPlan
from ..models.trial import Trial


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+.-]{1,}")
_STOPWORDS = {
    "advanced",
    "and",
    "are",
    "cancer",
    "carcinoma",
    "disease",
    "for",
    "from",
    "have",
    "into",
    "malignancy",
    "metastatic",
    "neoplasm",
    "neoplasms",
    "not",
    "patients",
    "phase",
    "recurrent",
    "solid",
    "stage",
    "study",
    "that",
    "the",
    "this",
    "trial",
    "with",
}


@dataclass(frozen=True)
class CandidateSelectionResult:
    selected: list[TrialCandidate]
    diagnostics: dict[str, Any]


def _candidate_sort_key(candidate: TrialCandidate) -> tuple[bool, float, int, str]:
    return (
        candidate.score is None,
        -(candidate.score or 0.0),
        candidate.rank if candidate.rank > 0 else 10**9,
        candidate.nct_id,
    )


def _original_score_sort_key(candidate: TrialCandidate) -> tuple[bool, float, int, str]:
    entity = (candidate.retrieval_metadata or {}).get("entity_negation") or {}
    original_score = (
        entity.get("original_score")
        if isinstance(entity, dict) and entity.get("original_score") is not None
        else candidate.score
    )
    return (
        original_score is None,
        -(float(original_score or 0.0)),
        candidate.rank if candidate.rank > 0 else 10**9,
        candidate.nct_id,
    )


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return {tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS}


def _trial_feature_tokens(trial: Trial | None, candidate: TrialCandidate) -> set[str]:
    """Return the text features used by the MMR diversity selector."""
    parts: list[str] = [
        candidate.title or "",
        candidate.snippet or "",
    ]
    if trial is not None:
        parts.extend(
            [
                trial.title,
                trial.official_title or "",
                " ".join(trial.conditions),
                " ".join(trial.keywords),
                " ".join(trial.interventions),
                getattr(trial.phase, "value", str(trial.phase)),
                getattr(trial.status, "value", str(trial.status)),
                trial.eligibility.raw_text[:4000],
            ]
        )
    return _tokens("\n".join(parts))


def _patient_tokens(patient: PatientProfile | None, plan: SearchPlan | None) -> set[str]:
    parts: list[str] = []
    if patient is not None:
        parts.extend(
            [
                patient.primary_diagnosis or "",
                patient.primary_diagnosis_stage or "",
                " ".join(patient.secondary_diagnoses),
                " ".join(f"{b.name} {b.status} {b.value or ''}" for b in patient.biomarkers),
                " ".join(
                    f"{t.name} {t.category or ''} {t.response or ''}"
                    for t in patient.prior_treatments
                ),
            ]
        )
        for concept in patient.mesh_concepts or []:
            if isinstance(concept, dict):
                parts.append(str(concept.get("name") or ""))
                parts.extend(str(s) for s in (concept.get("synonyms") or [])[:6])
        parts.append(patient.free_text_residual or "")
    if plan is not None:
        parts.extend(
            [
                plan.primary_disease_query,
                " ".join(plan.expansion_terms),
                " ".join(plan.retrieval_priorities),
                " ".join(plan.risk_flags),
            ]
        )
    return _tokens(" ".join(parts))


def _support_score(candidate_tokens: set[str], patient_tokens: set[str]) -> float:
    if not patient_tokens:
        return 0.5
    overlap = candidate_tokens & patient_tokens
    return max(0.0, min(1.0, len(overlap) / max(1, min(8, len(patient_tokens)))))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _retrieval_values(pool: list[TrialCandidate]) -> dict[str, float]:
    scores = [float(c.score or 0.0) for c in pool]
    min_score = min(scores, default=0.0)
    max_score = max(scores, default=0.0)
    spread = max_score - min_score
    out: dict[str, float] = {}
    for index, cand in enumerate(pool):
        if spread > 0:
            value = (float(cand.score or 0.0) - min_score) / spread
        elif len(pool) > 1:
            value = 1.0 - (index / (len(pool) - 1))
        else:
            value = 1.0
        out[cand.nct_id] = max(0.0, min(1.0, value))
    return out


def _top_score_result(
    sorted_candidates: list[TrialCandidate],
    *,
    policy: str,
    cap: int,
) -> CandidateSelectionResult:
    effective_cap = cap if cap > 0 else len(sorted_candidates)
    selected = list(sorted_candidates[:effective_cap])
    selected_ids = [c.nct_id for c in selected]
    dropped = [c.nct_id for c in sorted_candidates[effective_cap:]]
    original_top_score_ids = [
        c.nct_id for c in sorted(sorted_candidates, key=_original_score_sort_key)[:effective_cap]
    ]
    original_top_score_set = set(original_top_score_ids)
    selection_source_by_id: dict[str, str] = {}
    outside_top_score_added: list[str] = []
    for cand in selected:
        if cand.nct_id not in original_top_score_set:
            selection_source_by_id[cand.nct_id] = "entity_outside_top_score"
            outside_top_score_added.append(cand.nct_id)
        else:
            selection_source_by_id[cand.nct_id] = "top_score"
    return CandidateSelectionResult(
        selected=selected,
        diagnostics={
            "policy": policy,
            "effective_policy": "top_score",
            "requested_cap": cap,
            "effective_cap": effective_cap,
            "input_viable_count": len(sorted_candidates),
            "pool_size": len(sorted_candidates),
            "selected_count": len(selected),
            "selected_ids": selected_ids,
            "top_score_ids": selected_ids,
            "original_top_score_ids": original_top_score_ids,
            "diverse_added_ids": [],
            "outside_top_score_added_ids": outside_top_score_added,
            "selection_source_by_id": selection_source_by_id,
            "dropped_by_selection_count": 0,
            "dropped_by_cap_count": len(dropped),
            "dropped_by_cap_ids": dropped[:20],
        },
    )


def select_viable_candidates(
    candidates: Iterable[TrialCandidate],
    *,
    get_trial: Callable[[str], Trial | None],
    patient: PatientProfile | None = None,
    plan: SearchPlan | None = None,
    policy: str = "top_score",
    mode: str = "benchmark",
    cap: int = 0,
    keep_top: int = 8,
    select_total: int = 10,
    pool_size: int = 20,
) -> CandidateSelectionResult:
    """Select viable candidates to send to expensive eligibility evaluation.

    ``diverse_top10`` is benchmark-only and intentionally uses only trial text
    plus retrieval scores. It never reads qrels/gold labels.
    """
    sorted_candidates = sorted(list(candidates), key=_candidate_sort_key)
    if mode != "benchmark" or policy != "diverse_top10":
        return _top_score_result(sorted_candidates, policy=policy, cap=cap)

    if not sorted_candidates:
        return _top_score_result(sorted_candidates, policy=policy, cap=cap)

    configured_total = max(1, int(select_total or 10))
    effective_cap = configured_total if cap <= 0 else min(max(1, cap), configured_total)
    effective_cap = min(effective_cap, len(sorted_candidates))
    keep = min(max(0, int(keep_top or 0)), effective_cap)
    pool = sorted_candidates[: min(max(pool_size, effective_cap), len(sorted_candidates))]

    selected = list(pool[:keep])
    selected_ids = {c.nct_id for c in selected}
    diverse_added: list[TrialCandidate] = []
    support: dict[str, float] = {}
    if len(selected) < effective_cap:
        feature_cache: dict[str, set[str]] = {
            cand.nct_id: _trial_feature_tokens(get_trial(cand.nct_id), cand)
            for cand in pool
        }
        patient_feature_tokens = _patient_tokens(patient, plan)
        support = {
            cand.nct_id: _support_score(
                feature_cache.get(cand.nct_id, set()),
                patient_feature_tokens,
            )
            for cand in pool
        }
        retrieval = _retrieval_values(pool)
        remaining = [c for c in pool if c.nct_id not in selected_ids]
        while remaining and len(selected) < effective_cap:
            best = max(
                remaining,
                key=lambda cand: (
                    0.70 * retrieval.get(cand.nct_id, 0.0)
                    + 0.20 * support.get(cand.nct_id, 0.0)
                    + 0.10
                    * (
                        1.0
                        - max(
                            (
                                _jaccard(
                                    feature_cache.get(cand.nct_id, set()),
                                    feature_cache.get(sel.nct_id, set()),
                                )
                                for sel in selected
                            ),
                            default=0.0,
                        )
                    ),
                    float(cand.score or 0.0),
                    -(cand.rank if cand.rank > 0 else 10**9),
                    cand.nct_id,
                ),
            )
            selected.append(best)
            diverse_added.append(best)
            selected_ids.add(best.nct_id)
            remaining = [c for c in remaining if c.nct_id != best.nct_id]

    selected_order = [c.nct_id for c in selected]
    top_score_ids = [c.nct_id for c in sorted_candidates[:effective_cap]]
    selected_set = set(selected_order)
    dropped_by_selection = [nct_id for nct_id in top_score_ids if nct_id not in selected_set]
    dropped_by_cap = [c.nct_id for c in sorted_candidates if c.nct_id not in selected_set][
        len(dropped_by_selection) :
    ]
    top_score_set = set(top_score_ids)
    outside_top_score_added = [
        c.nct_id for c in diverse_added if c.nct_id not in top_score_set
    ]
    selection_source_by_id = {
        nct_id: (
            "top_score"
            if nct_id in top_score_set
            else "diverse_outside_top_score"
        )
        for nct_id in selected_order
    }

    return CandidateSelectionResult(
        selected=selected,
        diagnostics={
            "policy": policy,
            "effective_policy": "diverse_top10",
            "requested_cap": cap,
            "effective_cap": effective_cap,
            "input_viable_count": len(sorted_candidates),
            "pool_size": len(pool),
            "selected_count": len(selected),
            "kept_top_score_count": keep,
            "diverse_added_count": len(diverse_added),
            "selected_ids": selected_order,
            "top_score_ids": top_score_ids,
            "diverse_added_ids": [c.nct_id for c in diverse_added],
            "outside_top_score_added_ids": outside_top_score_added,
            "selection_source_by_id": selection_source_by_id,
            "diverse_added_support": {
                c.nct_id: round(support.get(c.nct_id, 0.0), 4)
                for c in diverse_added
            },
            "dropped_by_selection_count": len(dropped_by_selection),
            "dropped_by_selection_ids": dropped_by_selection,
            "dropped_by_cap_count": len(dropped_by_cap),
            "dropped_by_cap_ids": dropped_by_cap[:20],
        },
    )
