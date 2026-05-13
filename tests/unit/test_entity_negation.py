"""Entity/negation candidate reranking tests."""

from __future__ import annotations

from trial_matcher.models.agent_state import TrialCandidate
from trial_matcher.models.patient import Biomarker, PatientProfile, PriorTreatment, Sex
from trial_matcher.models.search_plan import SearchPlan
from trial_matcher.models.trial import Eligibility, Sex as TrialSex, Trial
from trial_matcher.retrieval.entity_negation import apply_entity_negation_rerank


def _patient() -> PatientProfile:
    return PatientProfile(
        topic_id="1",
        raw_text=(
            "55-year-old male with EGFR mutated lung adenocarcinoma treated with "
            "osimertinib. No history of melanoma."
        ),
        age_years=55,
        sex=Sex.MALE,
        primary_diagnosis="lung adenocarcinoma",
        biomarkers=[Biomarker(name="EGFR", status="mutated")],
        prior_treatments=[PriorTreatment(name="osimertinib", category="targeted")],
    )


def _plan() -> SearchPlan:
    return SearchPlan(
        primary_disease_query="lung adenocarcinoma",
        expansion_terms=["non-small cell lung cancer", "EGFR mutation"],
        retrieval_priorities=["EGFR targeted therapy", "osimertinib"],
    )


def _trials() -> dict[str, Trial]:
    return {
        "GOOD": Trial(
            nct_id="GOOD",
            title="EGFR mutated lung adenocarcinoma study",
            conditions=["lung adenocarcinoma"],
            interventions=["osimertinib"],
            brief_summary="Treatment study for EGFR mutated lung cancer.",
            eligibility=Eligibility(
                inclusion_text="Patients with EGFR mutated lung adenocarcinoma.",
                exclusion_text="Uncontrolled cardiac disease.",
                sex=TrialSex.MALE,
            ),
        ),
        "BAD": Trial(
            nct_id="BAD",
            title="Melanoma immunotherapy study",
            conditions=["melanoma"],
            interventions=["nivolumab"],
            brief_summary="Immunotherapy study for melanoma.",
            eligibility=Eligibility(
                inclusion_text="Patients with melanoma.",
                exclusion_text="Prior EGFR targeted therapy.",
                sex=TrialSex.ALL,
            ),
        ),
    }


def test_entity_negation_audit_preserves_order_and_scores():
    trials = _trials()
    candidates = [
        TrialCandidate(nct_id="BAD", score=10.0, rank=1),
        TrialCandidate(nct_id="GOOD", score=9.0, rank=2),
    ]

    out = apply_entity_negation_rerank(
        candidates,
        get_trial=trials.get,
        patient=_patient(),
        plan=_plan(),
        mode="benchmark",
        policy="audit",
        weight=0.9,
        protect_top=0,
    )

    assert [c.nct_id for c in out.candidates] == ["BAD", "GOOD"]
    assert [c.score for c in out.candidates] == [10.0, 9.0]
    assert out.diagnostics["effective_policy"] == "audit"
    assert "entity_negation" in out.candidates[0].retrieval_metadata
    assert "entity_negation" in out.candidates[1].retrieval_metadata


def test_entity_negation_rerank_can_promote_supported_candidate():
    trials = _trials()
    candidates = [
        TrialCandidate(nct_id="BAD", score=10.0, rank=1),
        TrialCandidate(nct_id="GOOD", score=9.0, rank=2),
    ]

    out = apply_entity_negation_rerank(
        candidates,
        get_trial=trials.get,
        patient=_patient(),
        plan=_plan(),
        mode="benchmark",
        policy="rerank_final",
        weight=0.9,
        protect_top=0,
    )

    assert [c.nct_id for c in out.candidates] == ["GOOD", "BAD"]
    assert out.diagnostics["changed_order"] is True
    good_meta = out.diagnostics["by_id"]["GOOD"]
    bad_meta = out.diagnostics["by_id"]["BAD"]
    assert good_meta["entity_score"] > bad_meta["entity_score"]
    assert "melanoma" in bad_meta["patient_negated_trial_terms"]


def test_entity_negation_is_benchmark_only():
    trials = _trials()
    candidates = [
        TrialCandidate(nct_id="BAD", score=10.0, rank=1),
        TrialCandidate(nct_id="GOOD", score=9.0, rank=2),
    ]

    out = apply_entity_negation_rerank(
        candidates,
        get_trial=trials.get,
        patient=_patient(),
        plan=_plan(),
        mode="clinical_active",
        policy="rerank_final",
        weight=0.9,
        protect_top=0,
    )

    assert out.candidates == candidates
    assert out.diagnostics["effective_policy"] == "off"


def test_entity_negation_rerank_does_not_read_qrel_like_metadata():
    trials = _trials()
    base = [
        TrialCandidate(nct_id="BAD", score=10.0, rank=1, retrieval_metadata={"qrel": 2}),
        TrialCandidate(nct_id="GOOD", score=9.0, rank=2, retrieval_metadata={"qrel": 0}),
    ]
    flipped = [
        TrialCandidate(nct_id="BAD", score=10.0, rank=1, retrieval_metadata={"qrel": 0}),
        TrialCandidate(nct_id="GOOD", score=9.0, rank=2, retrieval_metadata={"qrel": 2}),
    ]

    kwargs = dict(
        get_trial=trials.get,
        patient=_patient(),
        plan=_plan(),
        mode="benchmark",
        policy="rerank_final",
        weight=0.9,
        protect_top=0,
    )

    first = apply_entity_negation_rerank(base, **kwargs)
    second = apply_entity_negation_rerank(flipped, **kwargs)

    assert [c.nct_id for c in first.candidates] == [c.nct_id for c in second.candidates]
