"""NegEx / negation regex behavior."""

from __future__ import annotations


def test_simple_negation_detected():
    from trial_matcher.nlp.negation import has_negation

    assert has_negation("no history of myocardial infarction")
    assert has_negation("denies chest pain")
    assert has_negation("never smoked")


def test_no_negation_in_positive_statement():
    from trial_matcher.nlp.negation import has_negation

    assert not has_negation("the patient has a history of diabetes")
    assert not has_negation("ECOG 1")


def test_annotate_runs_without_scispacy():
    """Even when scispaCy isn't installed, annotate should not raise."""
    from trial_matcher.nlp.negation import annotate_negations

    text = "Inclusion: no history of brain metastases."
    out = annotate_negations(text)
    assert isinstance(out, str)
    assert "no history" in out.lower() or "[CONTAINS_NEGATION]" in out or "NEGATED" in out
