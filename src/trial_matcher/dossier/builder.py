"""Build a TrialDossier (JSON) and render its Markdown projection."""

from __future__ import annotations

from pathlib import Path

import jinja2

from ..llm.client import UnifiedLLM
from ..llm.prompts import DOSSIER_SUMMARY_V1
from ..logging import logger
from ..models.dossier import (
    AttentionFlag,
    CriterionRow,
    DossierMetadata,
    EligibilityCounts,
    FlagSeverity,
    ScoreBreakdown,
    TrialDossier,
)
from ..models.eligibility import EligibilityLabel
from ..models.patient import PatientProfile
from ..models.question import ClinicalQuestion
from ..models.ranking import JudgedTrial
from ..models.trial import Trial


_TEMPLATE_DIR = Path(__file__).parent / "templates"
_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(_TEMPLATE_DIR)),
    autoescape=False,
    undefined=jinja2.StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


class DossierBuilder:
    """Builds a TrialDossier object and renders its Markdown."""

    def __init__(self, llm: UnifiedLLM | None = None):
        self.llm = llm or UnifiedLLM()

    async def build(
        self,
        trial: Trial,
        judged: JudgedTrial,
        patient: PatientProfile,
        questions: list[ClinicalQuestion],
        critique_notes: list[str] | None = None,
    ) -> TrialDossier:
        rows = self._criterion_rows(judged)
        counts = self._counts_from_rows(rows)
        flags = self._flags(trial, judged, patient)
        breakdown = self._score_breakdown(judged)

        meta = DossierMetadata(
            title=trial.title or trial.official_title or "",
            official_title=trial.official_title,
            phase=trial.phase.value,
            status=trial.status.value,
            sponsor=trial.sponsor,
            last_update=trial.last_update_date,
            locations_summary=self._locations_summary(trial),
            contact_summary=self._contact_summary(trial),
            ctgov_url=trial.url,
        )

        summary = await self._executive_summary(trial, judged, patient)

        return TrialDossier(
            nct_id=trial.nct_id,
            rank=judged.rank,
            score=judged.score,
            score_breakdown=breakdown,
            executive_summary=summary,
            metadata=meta,
            eligibility_counts=counts,
            eligibility_table=rows,
            missing_information=sorted(
                questions,
                key=lambda q: ["critical", "high", "medium", "low"].index(q.priority.value),
            ),
            attention_flags=flags,
            judge_rationale=judged.rationale,
            critique_notes=list(critique_notes or []),
        )

    @staticmethod
    def render_markdown(dossier: TrialDossier) -> str:
        template = _ENV.get_template("trial.md.j2")
        return template.render(d=dossier)

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------
    @staticmethod
    def _criterion_rows(j: JudgedTrial) -> list[CriterionRow]:
        rows = []
        for ce in j.eval.criteria:
            if ce.criterion is None:
                continue
            rows.append(
                CriterionRow(
                    id=ce.criterion.id,
                    type=ce.criterion.type.value,
                    polarity=ce.criterion.polarity.value,  # type: ignore[arg-type]
                    text=ce.criterion.raw_text,
                    label=ce.label,
                    evidence=ce.evidence,
                    confidence=ce.confidence,
                )
            )
        return rows

    @staticmethod
    def _counts_from_rows(rows: list[CriterionRow]) -> EligibilityCounts:
        c = EligibilityCounts()
        for r in rows:
            if r.polarity == "inclusion":
                c.inclusion_total += 1
                if r.label == EligibilityLabel.MET:
                    c.inclusion_met += 1
                elif r.label == EligibilityLabel.NOT_MET:
                    c.inclusion_not_met += 1
                else:
                    c.inclusion_nei += 1
            else:
                c.exclusion_total += 1
                if r.label == EligibilityLabel.MET:
                    c.exclusion_met += 1
                elif r.label == EligibilityLabel.NOT_MET:
                    c.exclusion_not_met += 1
                else:
                    c.exclusion_nei += 1
        return c

    @staticmethod
    def _flags(trial: Trial, j: JudgedTrial, patient: PatientProfile) -> list[AttentionFlag]:
        flags: list[AttentionFlag] = []
        if j.eval.fraction_nei >= 0.4:
            flags.append(
                AttentionFlag(
                    severity=FlagSeverity.WARNING,
                    category="data_quality",
                    message=f"{j.eval.fraction_nei:.0%} of criteria are NEI — review questions before screening.",
                )
            )
        if j.eval.any_mandatory_inclusion_failed:
            flags.append(
                AttentionFlag(
                    severity=FlagSeverity.CRITICAL,
                    category="eligibility",
                    message="At least one mandatory inclusion criterion is not met.",
                )
            )
        if j.eval.any_exclusion_met:
            flags.append(
                AttentionFlag(
                    severity=FlagSeverity.CRITICAL,
                    category="eligibility",
                    message="At least one exclusion criterion is met by the patient.",
                )
            )
        if patient.location and trial.locations:
            in_country = any(
                loc.country and patient.location.lower() in loc.country.lower()
                for loc in trial.locations
            )
            if not in_country:
                flags.append(
                    AttentionFlag(
                        severity=FlagSeverity.WARNING,
                        category="geographic",
                        message=f"No site in patient location ({patient.location}); travel may be required.",
                    )
                )
        if trial.status.value not in {"RECRUITING", "NOT_YET_RECRUITING"}:
            flags.append(
                AttentionFlag(
                    severity=FlagSeverity.WARNING,
                    category="trial_status",
                    message=f"Trial status is {trial.status.value}.",
                )
            )
        return flags

    @staticmethod
    def _score_breakdown(j: JudgedTrial) -> ScoreBreakdown:
        comp = j.components or {}
        return ScoreBreakdown(
            total=j.score,
            eligibility_score=float(comp.get("eligibility_score", 0.0)),
            recruiting_bonus=float(comp.get("recruiting_bonus", 0.0)),
            phase_alignment=float(comp.get("phase_alignment", 0.0)),
            recency=float(comp.get("recency", 0.0)),
            geographic_proximity=float(comp.get("geographic_proximity", 0.0)),
            mandatory_veto=bool(comp.get("mandatory_veto", False)),
            nei_penalty=float(comp.get("nei_penalty", 0.0)),
            confidence_adjustment=float(comp.get("confidence_adjustment", 1.0)),
        )

    @staticmethod
    def _locations_summary(trial: Trial) -> str:
        if not trial.locations:
            return "(no locations listed)"
        cities = []
        for loc in trial.locations[:6]:
            label = ", ".join(p for p in [loc.city, loc.country] if p)
            if label:
                cities.append(label)
        return "; ".join(cities) or "(no locations listed)"

    @staticmethod
    def _contact_summary(trial: Trial) -> str | None:
        if not trial.contacts:
            return None
        c = trial.contacts[0]
        parts = [c.name, c.email, c.phone]
        return " | ".join(p for p in parts if p) or None

    async def _executive_summary(
        self, trial: Trial, j: JudgedTrial, patient: PatientProfile
    ) -> str:
        top_evidence = ""
        for ce in j.eval.criteria:
            if ce.label == EligibilityLabel.MET and ce.evidence:
                top_evidence = ce.evidence[:200]
                break
        prompt = DOSSIER_SUMMARY_V1.format(
            patient_summary=patient.summary(),
            trial_title=trial.title or "(unknown title)",
            phase=trial.phase.value,
            status=trial.status.value,
            n_inc_met=j.eval.n_inclusion_met,
            n_inc=j.eval.n_inclusion,
            n_inc_nei=j.eval.n_inclusion_nei,
            n_exc_not_met=j.eval.n_exclusion_not_met,
            n_exc=j.eval.n_exclusion,
            n_exc_nei=j.eval.n_exclusion_nei,
            top_evidence=top_evidence or "(none)",
        )
        try:
            text = await self.llm.acomplete(
                prompt, model="mini", temperature=0.2, max_tokens=800
            )
            return text.strip()
        except Exception as e:
            logger.warning(f"Dossier summary generation failed: {e!r}")
            return j.rationale or trial.brief_summary[:300]
