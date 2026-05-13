"""Planner is mode-aware: benchmark omits status mandatory filter (B2)."""

from __future__ import annotations

import pytest


@pytest.fixture
def patient_min(sample_patient):
    return sample_patient


def test_fallback_benchmark_mode_omits_status(patient_min):
    from trial_matcher.agent.planner import SearchPlanner

    plan = SearchPlanner._fallback(patient_min, relax=False, mode="benchmark")
    assert "status" not in plan.mandatory_filters
    assert plan.source == "fallback"


def test_fallback_clinical_active_keeps_status(patient_min):
    from trial_matcher.agent.planner import SearchPlanner

    plan = SearchPlanner._fallback(patient_min, relax=False, mode="clinical_active")
    assert "status" in plan.mandatory_filters
    assert "RECRUITING" in plan.mandatory_filters["status"]


def test_fallback_clinical_active_relax_adds_invitation(patient_min):
    from trial_matcher.agent.planner import SearchPlanner

    plan = SearchPlanner._fallback(patient_min, relax=True, mode="clinical_active")
    assert "ENROLLING_BY_INVITATION" in plan.mandatory_filters["status"]
