"""Dossier rendering — Markdown projection of TrialDossier."""

from __future__ import annotations


def test_render_markdown_with_minimal_dossier():
    from trial_matcher.dossier.builder import DossierBuilder
    from trial_matcher.models.dossier import (
        DossierMetadata,
        ScoreBreakdown,
        TrialDossier,
    )

    d = TrialDossier(
        nct_id="NCT12345678",
        rank=1,
        score=0.8,
        score_breakdown=ScoreBreakdown(total=0.8),
        executive_summary="Patient is a strong candidate.",
        metadata=DossierMetadata(
            title="Phase 2 trial",
            phase="PHASE2",
            status="RECRUITING",
            ctgov_url="https://clinicaltrials.gov/study/NCT12345678",
        ),
    )
    md = DossierBuilder.render_markdown(d)
    assert "NCT12345678" in md
    assert "Patient is a strong candidate" in md
    assert "Score breakdown" in md
