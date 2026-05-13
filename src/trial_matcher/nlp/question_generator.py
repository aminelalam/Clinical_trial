"""Generate a clinically formulated question for a NEI criterion."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, field_validator

from ..llm.client import UnifiedLLM
from ..llm.prompts import QUESTION_GEN_V1
from ..llm.structured import structured_complete
from ..logging import logger
from ..models.criterion import Criterion
from ..models.eligibility import CriterionEval
from ..models.patient import PatientProfile
from ..models.question import ClinicalQuestion, DataType, Priority
from ..models.trial import Trial


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_DEIDENTIFIED = re.compile(r"\[\*\*.*?\*\*\]")
_WHITESPACE = re.compile(r"\s+")


class _QuestionPayload(BaseModel):
    """Lenient LLM payload; stable IDs are filled from local context."""

    model_config = ConfigDict(extra="ignore")

    trial_id: str | None = None
    criterion_id: str | None = None
    question_text: str
    data_point: str
    units: str | None = None
    time_window: str | None = None
    measurement_context: str | None = None
    rationale: str
    expected_data_type: DataType = DataType.TEXT
    priority: Priority = Priority.MEDIUM

    @field_validator("expected_data_type", mode="before")
    @classmethod
    def _normalize_expected_data_type(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "number": DataType.NUMERIC,
            "float": DataType.NUMERIC,
            "integer": DataType.NUMERIC,
            "int": DataType.NUMERIC,
            "lab": DataType.NUMERIC,
            "lab_value": DataType.NUMERIC,
            "measurement": DataType.NUMERIC,
            "yes_no": DataType.BOOLEAN,
            "true_false": DataType.BOOLEAN,
            "binary": DataType.BOOLEAN,
            "choice": DataType.CATEGORICAL,
            "category": DataType.CATEGORICAL,
            "datetime": DataType.DATE,
            "free_text": DataType.TEXT,
            "clinical_note": DataType.TEXT,
            "mixed": DataType.TEXT,
            "multiple": DataType.TEXT,
            "composite": DataType.TEXT,
            "unknown": DataType.TEXT,
        }
        return aliases.get(normalized, normalized)

    @field_validator("priority", mode="before")
    @classmethod
    def _normalize_priority(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "urgent": Priority.CRITICAL,
            "very_high": Priority.CRITICAL,
            "required": Priority.HIGH,
            "important": Priority.HIGH,
            "normal": Priority.MEDIUM,
            "unknown": Priority.MEDIUM,
            "optional": Priority.LOW,
        }
        return aliases.get(normalized, normalized)


def _clean_prompt_text(text: str | None, max_chars: int) -> str:
    """Remove brittle prompt content without changing the clinical meaning."""
    if not text:
        return ""
    cleaned = _DEIDENTIFIED.sub("[redacted]", text)
    cleaned = _CONTROL_CHARS.sub(" ", cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


_NUMERIC_HINTS = (
    "bilirubin",
    "creatinine",
    "hemoglobin",
    "haemoglobin",
    "platelet",
    "neutrophil",
    "anc",
    "ast",
    "alt",
    "alkaline phosphatase",
    "ecog",
    "karnofsky",
    "lvef",
    "qt",
    "inr",
)
_DATE_HINTS = ("within", "prior to", "before", "after", "days", "weeks", "months", "years")
_BOOLEAN_HINTS = (
    "history of",
    "active",
    "uncontrolled",
    "concurrent",
    "prior",
    "previous",
    "received",
    "requires",
    "known",
    "pregnant",
    "breastfeeding",
)


def _infer_question_type(criterion_text: str) -> DataType:
    text = criterion_text.lower()
    if any(hint in text for hint in _NUMERIC_HINTS):
        return DataType.NUMERIC
    if any(hint in text for hint in _DATE_HINTS):
        return DataType.DATE
    if any(hint in text for hint in _BOOLEAN_HINTS):
        return DataType.BOOLEAN
    return DataType.TEXT


def _fallback_question(
    *,
    criterion: Criterion,
    trial: Trial,
    reason: str = "Information missing from the patient note.",
) -> ClinicalQuestion:
    criterion_text = _clean_prompt_text(criterion.raw_text, 220)
    data_type = _infer_question_type(criterion_text)
    priority = Priority.HIGH if getattr(criterion, "is_mandatory", True) else Priority.MEDIUM
    question_text = (
        f"For trial {trial.nct_id}, what patient information confirms this criterion: "
        f"{criterion_text}"
    )
    if len(question_text) > 260:
        question_text = question_text[:257].rstrip() + "..."
    return ClinicalQuestion(
        trial_id=trial.nct_id,
        criterion_id=criterion.id,
        question_text=question_text,
        data_point=criterion_text[:120],
        time_window="Use the time window specified by the trial criterion, if any.",
        measurement_context="Clinical record, laboratory report, imaging report, or treatment history as applicable.",
        rationale=_clean_prompt_text(reason, 240) or f"Required by trial {trial.nct_id} criterion {criterion.id}.",
        expected_data_type=data_type,
        priority=priority,
    )


def _missing_reason_for_prompt(eval_: CriterionEval) -> str:
    """Keep question prompts focused on the missing datum, not patient evidence."""
    if not eval_.reasoning:
        return "The available patient note does not contain enough information to determine this criterion confidently."
    if eval_.label.value == "NEI":
        return "The available patient note does not contain enough information to determine this criterion confidently."
    return "The criterion could not be converted into a stable screening question from the available structured data."


class QuestionGenerator:
    """Produces a ClinicalQuestion for a (criterion, patient, trial) triple."""

    def __init__(self, llm: UnifiedLLM | None = None):
        self.llm = llm or UnifiedLLM()

    async def generate(
        self,
        criterion: Criterion,
        patient: PatientProfile,
        trial: Trial,
        eval_: CriterionEval,
    ) -> ClinicalQuestion:
        safe_title = _clean_prompt_text(trial.title, 180)
        safe_criterion = _clean_prompt_text(criterion.raw_text, 700)
        safe_summary = _clean_prompt_text(patient.summary(), 500)
        safe_missing_for_prompt = _missing_reason_for_prompt(eval_)
        safe_missing_for_fallback = _clean_prompt_text(
            eval_.reasoning or safe_missing_for_prompt,
            240,
        )
        prompt = QUESTION_GEN_V1.format(
            nct_id=trial.nct_id,
            trial_title=safe_title,
            criterion_text=safe_criterion,
            criterion_id=criterion.id,
            patient_summary=safe_summary,
            what_is_missing=safe_missing_for_prompt,
        )
        try:
            payload = await structured_complete(
                self.llm,
                prompt=prompt,
                response_model=_QuestionPayload,
                model="mini",
                temperature=0.0,
                max_tokens=800,
                max_retries=2,
                task_name="question_generate",
            )
            return ClinicalQuestion(
                trial_id=payload.trial_id or trial.nct_id,
                criterion_id=payload.criterion_id or criterion.id,
                question_text=payload.question_text,
                data_point=payload.data_point,
                units=payload.units,
                time_window=payload.time_window,
                measurement_context=payload.measurement_context,
                rationale=payload.rationale,
                expected_data_type=payload.expected_data_type,
                priority=payload.priority,
            )
        except Exception as e:
            logger.warning(
                f"Question generation failed for {trial.nct_id}/{criterion.id}: {e!r}; "
                f"using fallback question"
            )
            return _fallback_question(
                criterion=criterion,
                trial=trial,
                reason=safe_missing_for_fallback,
            )
