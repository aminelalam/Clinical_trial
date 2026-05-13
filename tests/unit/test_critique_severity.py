"""SelfCritic correctly compares IssueSeverity (B3)."""

from __future__ import annotations

import asyncio
from typing import Any


def _judged_top10():
    from trial_matcher.models.eligibility import TrialEval, TrialLabel
    from trial_matcher.models.ranking import JudgedTrial

    out = []
    for i in range(10):
        out.append(
            JudgedTrial(
                nct_id=f"NCT{i:08d}",
                rank=i + 1,
                score=1.0 - 0.1 * i,
                eval=TrialEval(
                    nct_id=f"NCT{i:08d}",
                    label=TrialLabel.ELIGIBLE,
                    n_inclusion=2,
                    n_exclusion=1,
                    n_inclusion_met=2,
                    n_exclusion_not_met=1,
                ),
                rationale="r",
            )
        )
    return out


def test_high_severity_triggers_demotion(sample_patient):
    """Reproduces B3. With severity == "high" string-compare the demotion
    branch never fired. After the enum fix, the high-severity trial moves
    down by exactly two positions."""
    import trial_matcher.ranking.critique as crit_mod
    from trial_matcher.models.critique import Critique, CritiqueIssue, IssueSeverity
    from trial_matcher.ranking.critique import SelfCritic

    async def fake_structured_complete(*args: Any, **kwargs: Any) -> Critique:
        return Critique(
            issues_found=[
                CritiqueIssue(
                    trial_id="NCT00000000",
                    issue="phase mismatch",
                    severity=IssueSeverity.HIGH,
                )
            ],
            rerank_needed=True,
            rerank_instructions="demote 1",
            final_notes="",
        )

    original = crit_mod.structured_complete
    crit_mod.structured_complete = fake_structured_complete  # type: ignore[assignment]
    try:
        critic = SelfCritic(llm=None)  # llm unused because structured_complete is stubbed
        # SelfCritic.__init__ requires a UnifiedLLM; pass a dummy with the methods
        # we actually call (none, since structured_complete is stubbed).
        critic.llm = object()  # type: ignore[assignment]
        new_order, critique = asyncio.run(critic.critique(sample_patient, _judged_top10()))
    finally:
        crit_mod.structured_complete = original  # type: ignore[assignment]

    nct0_rank = next(j.rank for j in new_order if j.nct_id == "NCT00000000")
    assert nct0_rank == 3
    assert critique.rerank_needed is True
