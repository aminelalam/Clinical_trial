from __future__ import annotations


def _candidate(nct_id: str, score: float, rank: int):
    from trial_matcher.models.agent_state import TrialCandidate

    return TrialCandidate(nct_id=nct_id, score=score, rank=rank, source="fused")


def _trial(nct_id: str, condition: str, intervention: str):
    from trial_matcher.models.trial import Eligibility, Trial

    return Trial(
        nct_id=nct_id,
        title=f"{condition} {intervention}",
        brief_summary=f"Study of {intervention} in {condition}",
        conditions=[condition],
        interventions=[intervention],
        eligibility=Eligibility(raw_text=f"Inclusion Criteria:\n- {condition}"),
    )


def test_diverse_top10_keeps_top6_and_adds_mmr_diversity():
    from trial_matcher.agent.candidate_selection import select_viable_candidates

    cands = [
        _candidate("NCT01", 0.96, 1),
        _candidate("NCT02", 0.95, 2),
        _candidate("NCT03", 0.94, 3),
        _candidate("NCT04", 0.93, 4),
        _candidate("NCT05", 0.92, 5),
        _candidate("NCT06", 0.91, 6),
        _candidate("NCT07", 0.89, 7),
        _candidate("NCT08", 0.88, 8),
        _candidate("NCT09", 0.87, 9),
        _candidate("NCT10", 0.86, 10),
        _candidate("NCT11", 0.85, 11),
        _candidate("NCT12", 0.84, 12),
        _candidate("NCT13", 0.83, 13),
        _candidate("NCT14", 0.82, 14),
    ]
    trials = {
        **{
            f"NCT{i:02d}": _trial(f"NCT{i:02d}", "breast cancer", "HER2 therapy")
            for i in range(1, 11)
        },
        "NCT11": _trial("NCT11", "lung cancer", "EGFR inhibitor"),
        "NCT12": _trial("NCT12", "melanoma", "BRAF inhibitor"),
        "NCT13": _trial("NCT13", "leukemia", "FLT3 inhibitor"),
        "NCT14": _trial("NCT14", "prostate cancer", "androgen blockade"),
    }

    first = select_viable_candidates(
        cands,
        get_trial=trials.get,
        policy="diverse_top10",
        mode="benchmark",
        cap=10,
        keep_top=6,
        select_total=10,
    )
    second = select_viable_candidates(
        cands,
        get_trial=trials.get,
        policy="diverse_top10",
        mode="benchmark",
        cap=10,
        keep_top=6,
        select_total=10,
    )

    selected_ids = [c.nct_id for c in first.selected]
    assert selected_ids == [c.nct_id for c in second.selected]
    assert selected_ids[:6] == ["NCT01", "NCT02", "NCT03", "NCT04", "NCT05", "NCT06"]
    assert len(selected_ids) == 10
    assert "NCT11" in selected_ids
    assert first.diagnostics["dropped_by_selection_count"] >= 1
    assert first.diagnostics["selection_source_by_id"]["NCT11"] == (
        "diverse_outside_top_score"
    )
    assert first.diagnostics["selection_source_by_id"]["NCT07"] == "top_score"


def test_selector_ignores_qrel_like_retrieval_metadata():
    from trial_matcher.agent.candidate_selection import select_viable_candidates

    cands = [
        _candidate("NCT_QREL_ZERO", 1.0, 1).model_copy(
            update={"retrieval_metadata": {"qrel": 0, "gold": 0}}
        ),
        _candidate("NCT_QREL_TWO", 1.0, 2).model_copy(
            update={"retrieval_metadata": {"qrel": 2, "gold": 2}}
        ),
    ]

    out = select_viable_candidates(
        cands,
        get_trial=lambda _nct_id: None,
        policy="diverse_top10",
        mode="benchmark",
        cap=1,
        keep_top=0,
        select_total=1,
    )

    assert [c.nct_id for c in out.selected] == ["NCT_QREL_ZERO"]


def test_top_score_marks_entity_promotions_against_original_score():
    from trial_matcher.agent.candidate_selection import select_viable_candidates

    original_top = _candidate("NCT_ORIGINAL_TOP", 0.8, 1).model_copy(
        update={
            "retrieval_metadata": {
                "entity_negation": {"original_score": 0.8},
            }
        }
    )
    entity_promoted = _candidate("NCT_ENTITY_PROMOTED", 0.9, 2).model_copy(
        update={
            "retrieval_metadata": {
                "entity_negation": {"original_score": 0.1},
            }
        }
    )

    out = select_viable_candidates(
        [original_top, entity_promoted],
        get_trial=lambda _nct_id: None,
        policy="top_score",
        mode="benchmark",
        cap=1,
    )

    assert [c.nct_id for c in out.selected] == ["NCT_ENTITY_PROMOTED"]
    assert out.diagnostics["original_top_score_ids"] == ["NCT_ORIGINAL_TOP"]
    assert out.diagnostics["outside_top_score_added_ids"] == ["NCT_ENTITY_PROMOTED"]
    assert out.diagnostics["selection_source_by_id"]["NCT_ENTITY_PROMOTED"] == (
        "entity_outside_top_score"
    )


def test_diverse_policy_is_benchmark_only():
    from trial_matcher.agent.candidate_selection import select_viable_candidates

    cands = [
        _candidate("NCT_LOW", 0.1, 1),
        _candidate("NCT_HIGH", 0.9, 2),
        _candidate("NCT_MID", 0.5, 3),
    ]

    out = select_viable_candidates(
        cands,
        get_trial=lambda _nct_id: None,
        policy="diverse_top10",
        mode="clinical_active",
        cap=2,
        keep_top=0,
        select_total=10,
    )

    assert [c.nct_id for c in out.selected] == ["NCT_HIGH", "NCT_MID"]
    assert out.diagnostics["effective_policy"] == "top_score"
