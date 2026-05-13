from __future__ import annotations


def test_calibration_sweep_preserves_veto_before_thresholds():
    from eval.eligibility_calibration_sweep import _row_pred_qrel

    row = {
        "n_inclusion_total": 5,
        "n_inclusion_met": 4,
        "n_exclusion_met": 0,
        "fraction_nei": 0.0,
        "components": {"mandatory_veto": True},
        "score": -0.999,
    }

    assert _row_pred_qrel(row, min_inclusion_fraction=0.1, max_nei_fraction=1.0) == 1


def test_calibration_sweep_allows_benchmark_partial_support():
    from eval.eligibility_calibration_sweep import _row_pred_qrel

    row = {
        "n_inclusion_total": 5,
        "n_inclusion_met": 1,
        "n_exclusion_met": 0,
        "fraction_nei": 0.8,
        "components": {"mandatory_veto": False},
        "score": 0.1,
    }

    assert _row_pred_qrel(row, min_inclusion_fraction=0.1, max_nei_fraction=1.0) == 2
