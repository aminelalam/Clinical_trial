"""Sanity checks on the Pydantic models — the system's contract."""

from __future__ import annotations

import pytest


def test_patient_profile_minimal():
    from trial_matcher.models.patient import PatientProfile, Sex

    p = PatientProfile(topic_id="x", raw_text="patient note")
    assert p.sex == Sex.UNKNOWN
    assert p.age_years is None
    assert p.summary().startswith("patient note") or "patient" in p.summary()


def test_trial_serialization_roundtrip(sample_trial):
    payload = sample_trial.model_dump(mode="json")
    from trial_matcher.models.trial import Trial

    rebuilt = Trial.model_validate(payload)
    assert rebuilt.nct_id == sample_trial.nct_id
    assert rebuilt.eligibility.raw_text == sample_trial.eligibility.raw_text


def test_criterion_polarity_and_id():
    from trial_matcher.models.criterion import Criterion, CriterionType, Polarity

    c = Criterion(id="i_1", polarity=Polarity.INCLUSION, raw_text="Age >= 18", type=CriterionType.AGE)
    assert c.short().startswith("[i_1 INCL age]")


def test_trial_eval_qrel_mapping():
    from trial_matcher.models.eligibility import TrialEval, TrialLabel

    e = TrialEval(nct_id="NCT123", label=TrialLabel.ELIGIBLE)
    assert e.trec_qrel == 2
    e2 = TrialEval(nct_id="NCT123", label=TrialLabel.EXCLUDES)
    assert e2.trec_qrel == 1
    e3 = TrialEval(nct_id="NCT123", label=TrialLabel.IRRELEVANT)
    assert e3.trec_qrel == 0


def test_dossier_construction(sample_trial):
    from trial_matcher.models.dossier import (
        DossierMetadata,
        ScoreBreakdown,
        TrialDossier,
    )

    d = TrialDossier(
        nct_id=sample_trial.nct_id,
        rank=1,
        score=0.7,
        score_breakdown=ScoreBreakdown(total=0.7),
        metadata=DossierMetadata(
            title=sample_trial.title,
            phase=sample_trial.phase.value,
            status=sample_trial.status.value,
            ctgov_url=sample_trial.url,
        ),
    )
    assert d.rank == 1
    assert d.metadata.ctgov_url.endswith(sample_trial.nct_id)


def test_agent_state_construction():
    from trial_matcher.models.agent_state import AgentState

    s = AgentState(patient_raw="hi", run_id="abc")
    s.add_timing("test_node", 0.123)
    assert len(s.node_timings) == 1
    assert s.node_timings[0].seconds == pytest.approx(0.123)


def test_search_plan_relax_flag():
    from trial_matcher.models.search_plan import SearchPlan

    p = SearchPlan(primary_disease_query="lung cancer")
    assert not p.relax_optional_filters
    p.relax_optional_filters = True
    assert p.relax_optional_filters
