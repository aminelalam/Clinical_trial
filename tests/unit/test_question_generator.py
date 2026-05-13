"""Question-generation robustness helpers."""

from __future__ import annotations

from trial_matcher.models.criterion import Criterion, Polarity
from trial_matcher.models.eligibility import CriterionEval, EligibilityLabel
from trial_matcher.models.question import DataType, Priority
from trial_matcher.models.trial import Trial
from trial_matcher.nlp.question_generator import (
    _QuestionPayload,
    _clean_prompt_text,
    _fallback_question,
    _missing_reason_for_prompt,
)


def test_question_payload_normalizes_common_llm_enum_aliases():
    payload = _QuestionPayload.model_validate(
        {
            "question_text": "What is the latest creatinine value?",
            "data_point": "Creatinine",
            "rationale": "Required for renal eligibility.",
            "expected_data_type": "lab_value",
            "priority": "urgent",
        }
    )

    assert payload.expected_data_type is DataType.NUMERIC
    assert payload.priority is Priority.CRITICAL


def test_question_payload_treats_mixed_data_type_as_text():
    payload = _QuestionPayload.model_validate(
        {
            "question_text": "Which clinical findings support this criterion?",
            "data_point": "Composite clinical status",
            "rationale": "The criterion requires several clinical facts.",
            "expected_data_type": "mixed",
            "priority": "medium",
        }
    )

    assert payload.expected_data_type is DataType.TEXT


def test_clean_prompt_text_removes_deidentified_markers_and_control_chars():
    cleaned = _clean_prompt_text("Patient [**Name**]\x03 has\nmetastatic disease.", 200)

    assert "[**Name**]" not in cleaned
    assert "\x03" not in cleaned
    assert "\n" not in cleaned


def test_fallback_question_infers_boolean_type_for_history_criterion():
    criterion = Criterion(
        id="e_1",
        polarity=Polarity.EXCLUSION,
        raw_text="History of uncontrolled congestive heart failure within 6 months.",
    )
    question = _fallback_question(
        criterion=criterion,
        trial=Trial(nct_id="NCT00000001", title="Test trial"),
        reason="Heart failure status was not available.",
    )

    assert question.trial_id == "NCT00000001"
    assert question.criterion_id == "e_1"
    assert question.expected_data_type is DataType.DATE
    assert question.priority is Priority.HIGH


def test_missing_reason_for_prompt_omits_long_patient_evidence_for_nei():
    eval_ = CriterionEval(
        criterion_id="i_1",
        label=EligibilityLabel.NEI,
        reasoning='Self-consistency majority=NEI; evidence: "very long patient excerpt"',
    )

    reason = _missing_reason_for_prompt(eval_)

    assert "patient excerpt" not in reason
    assert "not contain enough information" in reason
