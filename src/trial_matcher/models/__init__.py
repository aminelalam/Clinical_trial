"""All Pydantic models — the system's typed contract.

Importing from this package returns flat names so other modules can do:
    from trial_matcher.models import PatientProfile, Trial, Criterion, ...
"""

from .agent_state import AgentState, NodeTiming
from .criterion import Criterion, CriterionType, Polarity, Predicate, TemporalConstraint
from .critique import Critique, CritiqueIssue, IssueSeverity
from .dossier import (
    AttentionFlag,
    CriterionRow,
    DossierMetadata,
    EligibilityCounts,
    ScoreBreakdown,
    TrialDossier,
)
from .eligibility import CriterionEval, EligibilityLabel, TrialEval, TrialLabel
from .patient import (
    Biomarker,
    Comorbidity,
    Lab,
    PatientProfile,
    Pregnancy,
    PriorTreatment,
    Sex,
)
from .question import ClinicalQuestion, DataType, Priority
from .ranking import JudgedTrial, RankedTrial
from .search_plan import SearchPlan
from .trial import (
    AgeRange,
    Contact,
    Eligibility,
    Location,
    Phase,
    RecruitmentStatus,
    Trial,
)

__all__ = [
    # patient
    "PatientProfile",
    "Sex",
    "Lab",
    "Biomarker",
    "PriorTreatment",
    "Comorbidity",
    "Pregnancy",
    # trial
    "Trial",
    "Eligibility",
    "Location",
    "Contact",
    "Phase",
    "RecruitmentStatus",
    "AgeRange",
    # criterion
    "Criterion",
    "CriterionType",
    "Polarity",
    "Predicate",
    "TemporalConstraint",
    # eligibility
    "CriterionEval",
    "EligibilityLabel",
    "TrialEval",
    "TrialLabel",
    # search plan
    "SearchPlan",
    # ranking
    "RankedTrial",
    "JudgedTrial",
    # question
    "ClinicalQuestion",
    "DataType",
    "Priority",
    # dossier
    "TrialDossier",
    "ScoreBreakdown",
    "AttentionFlag",
    "EligibilityCounts",
    "CriterionRow",
    "DossierMetadata",
    # critique
    "Critique",
    "CritiqueIssue",
    "IssueSeverity",
    # agent state
    "AgentState",
    "NodeTiming",
]
