import json

from trial_matcher.llm.client import LLMResponse
from trial_matcher.models.criterion import Criterion, CriterionType, Polarity
from trial_matcher.models.patient import Biomarker, PatientProfile
from trial_matcher.models.trial import Eligibility, Trial
from trial_matcher.nlp.criterion_extractor import CriterionExtractor, _balanced_limit
from trial_matcher.nlp.criterion_triage import (
    infer_criterion_type,
    is_section_header,
    section_header_action,
    select_criteria,
)


def _criterion(cid: str, polarity: Polarity) -> Criterion:
    return Criterion(id=cid, polarity=polarity, raw_text=f"{cid} text")


def test_balanced_limit_keeps_inclusions_and_exclusions():
    criteria = [
        *[_criterion(f"i_{i}", Polarity.INCLUSION) for i in range(1, 8)],
        *[_criterion(f"e_{i}", Polarity.EXCLUSION) for i in range(1, 5)],
    ]

    limited = _balanced_limit(criteria, 6)

    assert len(limited) == 6
    assert sum(c.polarity == Polarity.INCLUSION for c in limited) == 3
    assert sum(c.polarity == Polarity.EXCLUSION for c in limited) == 3


def test_triage_prefers_clinical_signal_over_administrative_text():
    criteria = [
        Criterion(
            id="i_1",
            polarity=Polarity.INCLUSION,
            raw_text="Able and willing to sign written informed consent",
        ),
        Criterion(
            id="i_2",
            polarity=Polarity.INCLUSION,
            raw_text="Histologically confirmed metastatic non-small cell lung cancer",
        ),
        Criterion(
            id="i_3",
            polarity=Polarity.INCLUSION,
            raw_text="EGFR exon 19 deletion or L858R mutation positive disease",
        ),
        Criterion(
            id="e_1",
            polarity=Polarity.EXCLUSION,
            raw_text="Active uncontrolled CNS metastases",
        ),
    ]
    patient = PatientProfile(
        topic_id="1",
        raw_text="EGFR mutated metastatic NSCLC",
        primary_diagnosis="metastatic non-small cell lung cancer",
        biomarkers=[Biomarker(name="EGFR", status="mutated")],
    )

    out = select_criteria(criteria, 3, patient=patient)
    selected_ids = {c.id for c in out.selected}

    assert selected_ids == {"i_2", "i_3", "e_1"}
    assert "i_1" not in selected_ids
    assert out.diagnostics["dropped_by_type"]["consent"] == 1
    assert any("patient_match" in r for c in out.selected for r in c.triage_reasons)


def test_triage_patient_match_ignores_generic_words():
    criteria = [
        Criterion(
            id="i_1",
            polarity=Polarity.INCLUSION,
            raw_text="Prior systemic therapy is required",
        ),
        Criterion(
            id="i_2",
            polarity=Polarity.INCLUSION,
            raw_text="HER2 positive breast carcinoma",
        ),
    ]
    patient = PatientProfile(
        topic_id="1",
        raw_text="HER2-positive breast cancer after prior therapy",
        primary_diagnosis="breast cancer",
        biomarkers=[Biomarker(name="HER2", status="positive")],
    )

    out = select_criteria(criteria, 2, patient=patient)
    reasons = {c.id: c.triage_reasons for c in out.selected}

    assert not any("patient_match:prior" in r for r in reasons["i_1"])
    assert any("patient_match" in r and "her2" in r for r in reasons["i_2"])


def test_infer_criterion_type_before_llm_classification():
    assert infer_criterion_type("ECOG performance status 0 or 1") == CriterionType.PERFORMANCE_STATUS
    assert infer_criterion_type("Prior platinum chemotherapy is required") == CriterionType.PRIOR_TREATMENT
    assert infer_criterion_type("Adequate creatinine and bilirubin") == CriterionType.LAB
    assert infer_criterion_type("Disease Characteristics:") == CriterionType.SECTION_HEADER
    assert is_section_header("For control participants:")
    assert section_header_action("Radiotherapy:") == "merge"
    assert section_header_action("See Disease Characteristics") == "drop"
    assert section_header_action("OR:") == "drop"


def test_section_headers_are_merged_or_dropped_before_classification():
    trial = Trial(
        nct_id="NCT00000009",
        title="Header segmentation test",
        eligibility=Eligibility(
            raw_text="Eligibility",
            inclusion_text="\n".join(
                [
                    "Disease Characteristics:",
                    "- Histologically confirmed non-small cell lung cancer",
                    "Radiotherapy:",
                    "- No prior brain radiotherapy",
                    "",
                    "See Disease Characteristics",
                    "",
                    "Surgery:",
                ]
            ),
            exclusion_text="\n".join(
                [
                    "For control participants:",
                    "- Active uncontrolled infection",
                ]
            ),
        ),
    )

    criteria, diagnostics = CriterionExtractor._fallback_split_with_diagnostics(trial)

    raw_texts = [c.raw_text for c in criteria]
    assert raw_texts == [
        "Disease Characteristics: Histologically confirmed non-small cell lung cancer",
        "Radiotherapy: No prior brain radiotherapy",
        "For control participants: Active uncontrolled infection",
    ]
    assert all(c.type != CriterionType.SECTION_HEADER for c in criteria)
    assert diagnostics["section_headers_merged"] == 3
    assert diagnostics["section_headers_dropped"] == 2


class _FakeLLM:
    primary = "azure"

    def __init__(self):
        self.prompts: list[str] = []

    def is_reasoning_model(self, model: str = "mini") -> bool:
        return False

    def _cache_model_identity(self, provider: str, model: str) -> str:
        return f"fake-{id(self)}"

    async def achat(self, messages, **kwargs):
        self.prompts.append(messages[-1]["content"])
        return LLMResponse(
            text=json.dumps(
                {
                    "criteria": [
                        {
                            "id": "i_1",
                            "polarity": "inclusion",
                            "raw_text": "Age > 18 years",
                            "type": "age",
                            "predicate": {
                                "op": ">",
                                "feature": "age_years",
                                "value": 18,
                                "units": "years",
                            },
                            "is_mandatory": True,
                            "has_negation": False,
                        }
                    ]
                }
            ),
            provider="azure",
            model="fake",
        )


def test_extractor_classifies_only_bounded_segmented_criteria():
    import asyncio

    llm = _FakeLLM()
    trial = Trial(
        nct_id="NCT00000001",
        title="Bounded extraction test",
        eligibility=Eligibility(
            raw_text="Eligibility",
            inclusion_text="\n".join(
                [
                    "- Age > 18 years",
                    "- Confirmed diagnosis",
                    "- ECOG 0-1",
                    "- Adequate organ function",
                ]
            ),
            exclusion_text="\n".join(
                [
                    "- Active infection",
                    "- Prior severe reaction",
                    "- Pregnancy",
                ]
            ),
        ),
    )

    criteria = asyncio.run(CriterionExtractor(llm).extract(trial, max_criteria=4))

    assert len(criteria) == 4
    assert {c.polarity for c in criteria} == {Polarity.INCLUSION, Polarity.EXCLUSION}
    assert criteria[0].id == "i_1"
    assert criteria[0].type.value == "age"
    assert "i_4" not in llm.prompts[0]


def test_extractor_triages_before_prompting_llm():
    import asyncio

    llm = _FakeLLM()
    trial = Trial(
        nct_id="NCT00000002",
        title="Triage extraction test",
        eligibility=Eligibility(
            raw_text="Eligibility",
            inclusion_text="\n".join(
                [
                    "- Able and willing to sign written informed consent",
                    "- Histologically confirmed metastatic lung adenocarcinoma",
                    "- EGFR mutation positive disease",
                    "- Available for follow-up visits",
                ]
            ),
            exclusion_text="\n".join(
                [
                    "- Active uncontrolled CNS metastases",
                    "- Pregnancy or breastfeeding",
                ]
            ),
        ),
    )
    patient = PatientProfile(
        topic_id="1",
        raw_text="EGFR mutated metastatic lung adenocarcinoma",
        primary_diagnosis="metastatic lung adenocarcinoma",
        biomarkers=[Biomarker(name="EGFR", status="mutated")],
    )

    result = asyncio.run(
        CriterionExtractor(llm).extract_with_diagnostics(
            trial,
            max_criteria=4,
            patient=patient,
            use_triage=True,
        )
    )

    prompt = llm.prompts[0]
    assert len(result.criteria) == 4
    assert "EGFR mutation positive" in prompt
    assert "Histologically confirmed" in prompt
    assert "Active uncontrolled CNS" in prompt
    assert "Available for follow-up" not in prompt
    assert result.diagnostics["dropped"] == 2
    assert result.diagnostics["dropped_by_type"]["consent"] >= 1
