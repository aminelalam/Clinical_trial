"""Temporal extraction behavior."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "text,expected_relation,expected_days",
    [
        ("within 6 months", "within", 180),
        ("within the past 30 days", "within", 30),
        ("3 weeks prior", "within", 21),
        ("ongoing", "ongoing", None),
        ("currently", "current", None),
        ("history of stroke", "ever", None),
        (">= 6 months since transplant", "after", 180),
    ],
)
def test_temporal_patterns(text, expected_relation, expected_days):
    from trial_matcher.nlp.temporal import extract_temporal

    t = extract_temporal(text)
    assert t is not None, f"Failed to extract from: {text}"
    assert t.relation == expected_relation
    assert t.days == expected_days


def test_no_temporal():
    from trial_matcher.nlp.temporal import extract_temporal

    assert extract_temporal("Age >= 18") is None


def test_annotate_temporal_inserts_marker():
    from trial_matcher.nlp.temporal import annotate_temporal

    out = annotate_temporal("MI within 6 months excluded")
    assert "[TEMPORAL:" in out
