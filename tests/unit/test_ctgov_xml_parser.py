"""ClinicalTrials.gov legacy XML parser tests."""

from __future__ import annotations


def test_parse_ctgov_xml_root_maps_legacy_fields():
    import xml.etree.ElementTree as ET

    from trial_matcher.ingestion.ctgov_xml_parser import parse_ctgov_xml_root
    from trial_matcher.models.trial import RecruitmentStatus, Sex

    root = ET.fromstring(
        """
        <clinical_study>
          <id_info><nct_id>NCT00000001</nct_id></id_info>
          <brief_title>EGFR lung cancer study</brief_title>
          <official_title>Official EGFR NSCLC Study</official_title>
          <brief_summary><textblock>Short summary.</textblock></brief_summary>
          <detailed_description><textblock>Detailed text.</textblock></detailed_description>
          <condition>Non Small Cell Lung Cancer</condition>
          <keyword>EGFR</keyword>
          <intervention><intervention_name>Osimertinib</intervention_name></intervention>
          <phase>Phase 2</phase>
          <overall_status>Completed</overall_status>
          <study_type>Observational</study_type>
          <eligibility>
            <criteria><textblock>Inclusion Criteria: Adults with EGFR NSCLC.
            Exclusion Criteria: Prior drug X.</textblock></criteria>
            <gender>Female</gender>
            <minimum_age>18 Years</minimum_age>
            <maximum_age>N/A</maximum_age>
            <healthy_volunteers>No</healthy_volunteers>
          </eligibility>
          <sponsors><lead_sponsor><agency>NCI</agency></lead_sponsor></sponsors>
          <last_update_posted>April 27, 2021</last_update_posted>
        </clinical_study>
        """
    )

    trial = parse_ctgov_xml_root(root)

    assert trial is not None
    assert trial.nct_id == "NCT00000001"
    assert trial.status == RecruitmentStatus.COMPLETED
    assert trial.interventional is False
    assert trial.eligibility.sex == Sex.FEMALE
    assert trial.interventions == ["Osimertinib"]
    assert "Detailed text" in trial.primary_text()
    assert "Adults with EGFR" in trial.eligibility.inclusion_text
