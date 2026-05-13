"""Fielded BM25 retrieval over multiple persisted BM25S sub-indexes."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import get_settings
from ..models.agent_state import TrialCandidate
from ..models.patient import PatientProfile
from ..models.search_plan import SearchPlan
from ..models.trial import Trial
from .bm25 import BM25Retriever

BM25_FIELD_NAMES = (
    "all",
    "condition_title",
    "eligibility",
    "intervention",
    "summary_description",
)

DEFAULT_FIELD_WEIGHTS: dict[str, float] = {
    "all": 1.0,
    "condition_title": 1.4,
    "eligibility": 0.5,
    "intervention": 0.2,
    "summary_description": 0.5,
}


def trial_text_for_bm25_field(trial: Trial, field: str) -> str:
    """Return the text indexed for a fielded BM25 sub-index."""
    if field not in BM25_FIELD_NAMES:
        raise ValueError(f"Unknown BM25 field: {field}")
    return trial.text_for_bm25_field(field)


def _join_terms(values: list[str]) -> str:
    return " ".join(v.strip() for v in values if v and v.strip())


def _patient_feature_terms(patient: PatientProfile | None) -> list[str]:
    if patient is None:
        return []
    terms: list[str] = []
    if patient.primary_diagnosis:
        terms.append(patient.primary_diagnosis)
    if patient.primary_diagnosis_stage:
        terms.append(patient.primary_diagnosis_stage)
    terms.extend(patient.secondary_diagnoses[:5])
    terms.extend(f"{b.name} {b.status} {b.value or ''}" for b in patient.biomarkers[:8])
    terms.extend(t.name for t in patient.prior_treatments[:8])
    terms.extend(c.name for c in patient.comorbidities[:6])
    if patient.ecog is not None:
        terms.append(f"ECOG {patient.ecog}")
    if patient.karnofsky is not None:
        terms.append(f"Karnofsky {patient.karnofsky}")
    return terms


def fielded_queries(
    *,
    query: str,
    mesh_terms: list[str] | None = None,
    plan: SearchPlan | None = None,
    patient: PatientProfile | None = None,
) -> dict[str, str]:
    """Build deterministic per-field lexical queries."""
    mesh_terms = mesh_terms or []
    priorities = list(plan.retrieval_priorities if plan is not None else [])
    risk_flags = list(plan.risk_flags if plan is not None else [])
    patient_terms = _patient_feature_terms(patient)

    disease_terms = _join_terms([query, *mesh_terms])
    broad_terms = _join_terms([query, *mesh_terms, *priorities, *patient_terms])
    eligibility_terms = _join_terms([query, *patient_terms, *risk_flags, *priorities])
    intervention_terms = _join_terms([query, *priorities])
    if patient is not None:
        intervention_terms = _join_terms(
            [
                intervention_terms,
                *[t.name for t in patient.prior_treatments[:8]],
                *[b.name for b in patient.biomarkers[:8]],
            ]
        )

    return {
        "all": broad_terms or query,
        "condition_title": disease_terms or query,
        "eligibility": eligibility_terms or query,
        "intervention": intervention_terms or query,
        "summary_description": broad_terms or query,
    }


class FieldedBM25Retriever:
    """Load several BM25S indexes and fuse their ranked lists with weighted RRF."""

    def __init__(
        self,
        index_dir: Path | str | None = None,
        *,
        fields: tuple[str, ...] = BM25_FIELD_NAMES,
        weights: dict[str, float] | None = None,
    ):
        settings = get_settings()
        configured_dir = settings.retrieval.fielded_bm25_index_dir
        self.index_dir = Path(
            index_dir
            or configured_dir
            or settings.paths.indices_dir / "bm25_trec2021_fielded"
        )
        self.fields = fields
        self.weights = dict(DEFAULT_FIELD_WEIGHTS)
        manifest_path = self.index_dir / "index_manifest.json"
        if weights is None and manifest_path.exists():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                subindices = payload.get("subindices") or {}
                for field, field_payload in subindices.items():
                    if field in BM25_FIELD_NAMES and "weight" in field_payload:
                        self.weights[field] = float(field_payload["weight"])
            except Exception:
                pass
        if weights:
            self.weights.update(weights)
        self._retrievers: dict[str, BM25Retriever] = {}

    def _retriever(self, field: str) -> BM25Retriever:
        if field not in BM25_FIELD_NAMES:
            raise ValueError(f"Unknown BM25 field: {field}")
        if field not in self._retrievers:
            self._retrievers[field] = BM25Retriever(self.index_dir / field)
        return self._retrievers[field]

    @property
    def nct_ids(self) -> list[str]:
        return self._retriever("all").nct_ids

    def retrieve(
        self,
        query: str,
        *,
        mesh_terms: list[str] | None = None,
        plan: SearchPlan | None = None,
        patient: PatientProfile | None = None,
        k: int | None = None,
        per_field_k: int | None = None,
    ) -> list[TrialCandidate]:
        settings = get_settings()
        final_k = k or settings.retrieval.bm25_top_k
        field_k = per_field_k or max(final_k, settings.retrieval.fielded_bm25_per_field_k)
        queries = fielded_queries(
            query=query,
            mesh_terms=mesh_terms,
            plan=plan,
            patient=patient,
        )
        return self.fuse_field_runs(queries, final_k=final_k, field_k=field_k)

    def retrieve_by_field(
        self,
        query: str,
        *,
        mesh_terms: list[str] | None = None,
        plan: SearchPlan | None = None,
        patient: PatientProfile | None = None,
        k: int | None = None,
    ) -> dict[str, list[TrialCandidate]]:
        settings = get_settings()
        field_k = k or settings.retrieval.fielded_bm25_per_field_k
        queries = fielded_queries(
            query=query,
            mesh_terms=mesh_terms,
            plan=plan,
            patient=patient,
        )
        return {
            field: self._retriever(field).retrieve(queries.get(field) or query, k=field_k)
            for field in self.fields
        }

    def fuse_field_runs(
        self,
        queries: dict[str, str],
        *,
        final_k: int,
        field_k: int,
    ) -> list[TrialCandidate]:
        settings = get_settings()
        scores: dict[str, float] = {}
        seen: dict[str, TrialCandidate] = {}
        field_ranks: dict[str, dict[str, int]] = {}
        field_scores: dict[str, dict[str, float]] = {}
        rrf_k = settings.retrieval.rrf_k

        for field in self.fields:
            weight = max(float(self.weights.get(field, 1.0)), 0.0)
            if weight <= 0:
                continue
            run = self._retriever(field).retrieve(queries.get(field) or "", k=field_k)
            for cand in run:
                rank = cand.rank if cand.rank > 0 else len(field_ranks.get(cand.nct_id, {})) + 1
                scores[cand.nct_id] = scores.get(cand.nct_id, 0.0) + weight / (rrf_k + rank)
                field_ranks.setdefault(cand.nct_id, {})[field] = rank
                field_scores.setdefault(cand.nct_id, {})[field] = float(cand.score)
                if cand.nct_id not in seen:
                    seen[cand.nct_id] = cand

        fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:final_k]
        out: list[TrialCandidate] = []
        for i, (nct_id, score) in enumerate(fused, start=1):
            first = seen[nct_id]
            out.append(
                TrialCandidate(
                    nct_id=nct_id,
                    score=float(score),
                    source="bm25_fielded",
                    rank=i,
                    title=first.title,
                    snippet=first.snippet,
                    retrieval_metadata={
                        "bm25_mode": "fielded",
                        "field_ranks": field_ranks.get(nct_id, {}),
                        "field_scores": field_scores.get(nct_id, {}),
                        "field_weights": {f: self.weights.get(f, 1.0) for f in self.fields},
                    },
                )
            )
        return out
