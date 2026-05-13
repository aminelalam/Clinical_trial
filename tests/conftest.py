"""Shared pytest fixtures.

These fixtures avoid network calls — LLM clients are mocked, indices are not
loaded. Real-data integration tests live behind the @pytest.mark.integration
marker and are skipped by default.

We deliberately avoid pytest's built-in ``tmp_path`` fixture because some
Windows installations have ACL issues on the global Temp folder. Instead we
create a project-local ``.test_tmp/`` directory which is fully under the
repo and gitignored.
"""

from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

# Ensure src/ is importable
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_TEST_TMP_ROOT = Path(__file__).resolve().parent.parent / ".test_tmp"


@pytest.fixture()
def project_tmp_path() -> Path:
    """A per-test scratch directory under the project (not the system Temp).

    Avoids Windows ACL issues with ``C:\\Users\\...\\AppData\\Local\\Temp``
    that break pytest's built-in ``tmp_path`` fixture on some machines.
    """
    _TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    p = _TEST_TMP_ROOT / f"t_{uuid.uuid4().hex[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    yield p
    # Best-effort cleanup; ignore errors so a locked file doesn't fail the test.
    shutil.rmtree(p, ignore_errors=True)


@pytest.fixture(autouse=True)
def isolated_settings(project_tmp_path, monkeypatch):
    """Redirect cache/data paths to a project-local tmp dir for every test."""
    monkeypatch.setenv("TRIAL_MATCHER__PATHS__CACHE_DIR", str(project_tmp_path / ".cache"))
    monkeypatch.setenv("TRIAL_MATCHER__PATHS__DATA_DIR", str(project_tmp_path / "data"))
    monkeypatch.setenv("TRIAL_MATCHER__PATHS__INDICES_DIR", str(project_tmp_path / "indices"))
    monkeypatch.setenv("TRIAL_MATCHER__LLM__DEFAULT_PROVIDER", "azure")
    # Clear the lru_cache so fresh settings are read
    from trial_matcher.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture()
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture()
def sample_trial(fixtures_dir):
    from trial_matcher.models.trial import (
        AgeRange,
        Eligibility,
        Phase,
        RecruitmentStatus,
        Sex,
        Trial,
    )

    return Trial(
        nct_id="NCT12345678",
        title="Phase 2 trial of drug X in HER2+ metastatic breast cancer",
        brief_summary="Study evaluates drug X in adults with HER2+ MBC.",
        conditions=["Breast Neoplasms", "HER2+ Breast Cancer"],
        phase=Phase.PHASE_2,
        status=RecruitmentStatus.RECRUITING,
        interventional=True,
        eligibility=Eligibility(
            raw_text=(
                "Inclusion Criteria:\n"
                "- Age >= 18\n"
                "- HER2 positive metastatic breast cancer\n"
                "- ECOG 0-1\n"
                "Exclusion Criteria:\n"
                "- Pregnant or breastfeeding\n"
                "- Active brain metastases within 6 months"
            ),
            inclusion_text="Age >= 18\nHER2 positive metastatic breast cancer\nECOG 0-1",
            exclusion_text="Pregnant or breastfeeding\nActive brain metastases within 6 months",
            age_range=AgeRange(min_days=18 * 365, max_days=None, raw_min="18 Years"),
            sex=Sex.ALL,
        ),
    )


@pytest.fixture()
def sample_patient():
    from trial_matcher.models.patient import (
        Biomarker,
        PatientProfile,
        Pregnancy,
        Sex,
    )

    return PatientProfile(
        topic_id="t1",
        raw_text=(
            "47-year-old woman with metastatic HER2-positive breast cancer, "
            "ECOG 1, no prior trastuzumab. No history of brain metastases."
        ),
        age_years=47.0,
        sex=Sex.FEMALE,
        primary_diagnosis="metastatic breast cancer",
        primary_diagnosis_stage="metastatic",
        ecog=1,
        biomarkers=[Biomarker(name="HER2", status="positive")],
        pregnancy=Pregnancy(pregnant=False, breastfeeding=False),
    )


class MockLLM:
    """A minimal stand-in for UnifiedLLM used in unit tests."""

    def __init__(self, canned: dict[str, Any] | None = None):
        self.canned = canned or {}
        self.call_log: list[dict] = []

    async def acomplete(self, prompt, **kwargs):
        self.call_log.append({"prompt": prompt[:200], **kwargs})
        # Default: echo a JSON object the structured helper can parse
        for substring, response in self.canned.items():
            if substring in prompt:
                return response
        return '{"decision": "NEI", "confidence": 0.5}'

    async def achat(self, messages, **kwargs):
        return await self.acomplete(messages[-1]["content"], **kwargs)


@pytest.fixture()
def mock_llm():
    return MockLLM()
