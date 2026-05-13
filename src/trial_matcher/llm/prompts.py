"""All LLM prompts in one place, versioned and documented.

Each prompt is a constant. When a prompt changes meaningfully, bump the
``_V<N>`` suffix so caches don't return stale outputs.

Style notes:
- We force structured outputs for everything except HyDE.
- We pre-process inputs (NegEx tags, temporal extraction) so prompts are short.
- We use clinical role conditioning ("You are a senior oncology coordinator…")
  when it noticeably improves quality (HyDE, judge, critique).
"""

# =============================================================================
# Patient profile extraction
# =============================================================================

PATIENT_EXTRACT_V1 = """You are a clinical NLP system. Extract a structured patient \
profile from the note below.

Patient note:
{patient_text}

Extract these fields. Use null when unknown.
- age_years: float, in years
- sex: "male" | "female" | "unknown"
- primary_diagnosis: primary clinical diagnosis (string)
- primary_diagnosis_stage: stage if mentioned (e.g., "IV", "metastatic"), else null
- secondary_diagnoses: list of strings
- ecog: integer 0-5, else null
- karnofsky: integer 0-100, else null
- biomarkers: list of objects with {{name, status, value?, method?}}
- prior_treatments: list of objects with {{name, category?, line?, response?, ongoing}}
- comorbidities: list of objects with {{name, active?, note?}}
- labs: list of objects with {{name, value?, units?, measured_on?, note?}}
- pregnancy: object with {{pregnant?, breastfeeding?, childbearing_potential?}} or null
- location: city or country if mentioned, else null
- free_text_residual: any clinically relevant content not placed above

Quote evidence verbatim where possible. Do not invent values."""


# =============================================================================
# HyDE — Hypothetical Document Embeddings (medical role conditioning)
# =============================================================================

HYDE_PROMPT = """You are a senior oncology clinical trial coordinator.
Given the following patient case, write a brief eligibility-criteria section \
(5-10 sentences) for a clinical trial that would be a perfect match for this \
patient. Be specific about disease, biomarkers, line of therapy, and key labs. \
Do NOT mention the patient — write only the eligibility criteria as they would \
appear on a real trial.

Patient: {patient_note}

Hypothetical eligibility criteria:"""


# =============================================================================
# Multi-query reformulation
# =============================================================================

MULTI_QUERY_PROMPT = """Generate 4 distinct search queries to find clinical \
trials for this patient. Each query should emphasize a different aspect:
1. Disease + stage + line of therapy
2. Biomarker / molecular profile
3. Comorbidities and exclusions to consider
4. Geographic / demographic focus

Patient: {patient_note}

Output JSON: {{"queries": ["...", "...", "...", "..."]}}"""


# =============================================================================
# Search planning (agentic)
# =============================================================================

PLAN_SEARCH_PROMPT = """You are planning a clinical-trial search for a specific patient.

Patient profile (JSON):
{patient_profile_json}

MeSH concepts identified (top synonyms):
{mesh_concepts}

Produce a SearchPlan as JSON with these fields:
1. primary_disease_query (str): main disease query for lexical retrieval.
2. expansion_terms (list[str]): MeSH synonyms / related terms to add.
3. mandatory_filters: dict with optional keys among:
   - phase (list of "PHASE1"|"PHASE2"|"PHASE3"|"PHASE4")
   - status (list, default ["RECRUITING","NOT_YET_RECRUITING"])
   - sex (str if relevant)
   - min_age_days (int)
   - max_age_days (int)
4. optional_filters: dict (relaxable if recall is low).
5. retrieval_priorities: list[str] explaining what should rank high \
(e.g., "biomarker-targeted therapy", "earlier line of treatment").
6. risk_flags: list[str] of features that may exclude many trials \
(e.g., "elderly", "ECOG 3", "prior immunotherapy").

Be specific. If patient has a rare biomarker, set retrieval_priorities accordingly. \
If pediatric, restrict mandatory phase. If pregnant, exclude reproductive-toxicity drugs. \
If relax_optional_filters_hint=true is passed, produce a more permissive plan than your \
previous attempt."""


# =============================================================================
# Criterion extraction (per trial)
# =============================================================================

CRITERION_EXTRACT_V1 = """You are an expert clinical trial coordinator. Parse \
the eligibility section below into a JSON list of structured criteria.

Trial NCT id: {nct_id}
Trial title: {trial_title}

Pre-annotated eligibility text (negation/temporal already tagged):
{annotated_text}

Output JSON: {{"criteria": [...]}}, where each criterion has:
- id: "i_<n>" (inclusion) or "e_<n>" (exclusion)
- polarity: "inclusion" | "exclusion"
- raw_text: original criterion sentence (verbatim)
- type: "age"|"sex"|"biomarker"|"prior_treatment"|"lab"|"comorbidity"|\
"performance_status"|"pregnancy"|"diagnosis"|"consent"|"other"
- predicate: object with {{op, feature, value, units?, temporal?}} or null when not deterministic
- is_mandatory: boolean (true unless the criterion text says "preferred", "if possible")
- has_negation: boolean (true if the criterion logic involves negation)

Each individual criterion (e.g., one bullet point) becomes one entry. \
Use null instead of guessing when a slot is unclear."""


CRITERION_CLASSIFY_V1 = """You are an expert clinical trial coordinator. Classify \
pre-segmented eligibility criteria for one trial.

Trial NCT id: {nct_id}
Trial title: {trial_title}

Pre-segmented criteria. Each row is:
id | polarity | annotated criterion text
{criteria_block}

Return JSON: {{"criteria": [...]}}, with exactly one object per input id.
For each object:
- id: copy the input id exactly
- polarity: copy the input polarity exactly: "inclusion" or "exclusion"
- raw_text: copy the criterion text, without the annotation tags if possible
- type: "age"|"sex"|"biomarker"|"prior_treatment"|"lab"|"comorbidity"|\
"performance_status"|"pregnancy"|"diagnosis"|"consent"|"other"
- predicate: object with {{op, feature, value, units?, temporal?}} or null when not deterministic
- is_mandatory: boolean, true unless the text explicitly says optional/preferred
- has_negation: boolean, true if the criterion logic involves negation

Do not merge, split, invent, omit, or reorder criteria. Use null instead of guessing."""


# =============================================================================
# Eligibility evaluation — structured CoT, six required fields
# =============================================================================

COT_ELIGIBILITY_V1 = """Evaluate whether this patient meets the criterion.

Criterion (annotated):
{criterion_annotated}

Patient note (relevant excerpts):
{patient_excerpts}

Reason step-by-step. Fill ALL six fields below — use "N/A" only when \
truly not applicable, never to skip thinking.

1. requirement: What exactly does the criterion require?
2. negation_check: Is the criterion negated? Are key terms in the patient note negated?
3. temporal_check: Is there a time constraint? Does the patient's relevant event fall within it?
4. evidence_quote: The exact sentence(s) in the patient note that are relevant. \
If none, state "no relevant evidence in patient note".
5. inference_gap: Is there missing information that prevents a confident decision? What specifically?
6. decision: One of "met" | "not_met" | "NEI". Default to NEI when info is genuinely missing.

Also output:
- confidence: 0.0 to 1.0

Output JSON with fields: requirement, negation_check, temporal_check, evidence_quote, \
inference_gap, decision, confidence."""


# Variant V2 — same as V1 but with a few-shot block injected between patient
# excerpts and the reasoning instructions. The runner uses V2 when a
# FewShotBank is provided to the cascade; otherwise V1 stays in effect for
# backward compat (and ablation studies).

COT_ELIGIBILITY_V2 = """Evaluate whether this patient meets the criterion.

Criterion (annotated):
{criterion_annotated}

Patient note (relevant excerpts):
{patient_excerpts}

Below are correctly-resolved analogous examples. Use them as a guide for \
how to reason; do not blindly copy decisions — your decision must follow \
from the actual criterion and patient above.

{few_shot_block}

Now reason step-by-step about the criterion above. Fill ALL six fields below \
— use "N/A" only when truly not applicable, never to skip thinking.

1. requirement: What exactly does the criterion require?
2. negation_check: Is the criterion negated? Are key terms in the patient note negated?
3. temporal_check: Is there a time constraint? Does the patient's relevant event fall within it?
4. evidence_quote: The exact sentence(s) in the patient note that are relevant. \
If none, state "no relevant evidence in patient note".
5. inference_gap: Is there missing information that prevents a confident decision? What specifically?
6. decision: One of "met" | "not_met" | "NEI". Default to NEI when info is genuinely missing.

Also output:
- confidence: 0.0 to 1.0

Output JSON with fields: requirement, negation_check, temporal_check, evidence_quote, \
inference_gap, decision, confidence."""


# =============================================================================
# Verifier — devil's advocate (T2 quality booster)
# =============================================================================

VERIFIER_PROMPT_V1 = """You previously decided this criterion has label "{label}" with \
confidence {confidence:.2f}.

Now play devil's advocate. Find the strongest argument that this decision is WRONG, \
considering:
- Negation in the criterion or patient note
- Temporal constraints (did the event happen within the required window?)
- Implicit information that may have been missed
- Edge cases (e.g., "recent" vs "current")

Criterion: {criterion_annotated}
Patient excerpts: {patient_excerpts}
Original decision: {label}
Original reasoning: {reasoning}

Output JSON:
{{
  "rebuttal_strength": "strong" | "moderate" | "weak" | "none",
  "rebuttal_reason": str,
  "revised_label": "met" | "not_met" | "NEI"
}}"""


# =============================================================================
# LLM-judge for top-10 ranking refinement
# =============================================================================

LLM_JUDGE_V1 = """You are a senior oncology physician asked to rank 10 trials for \
a specific patient.

Patient summary: {patient_summary}

Top 10 candidate trials with eligibility breakdown:
{trials_block}

Re-order the trials from most to least promising. For each, justify with TWO \
clinically-grounded sentences. Penalize trials with critical NEI criteria \
or many failed inclusion criteria. Reward trials with strong inclusion match \
and active recruitment.

Output JSON:
{{"ranking": [{{"nct_id": "...", "rationale": "..."}}, ...]}}

The list MUST contain all 10 NCT ids in the new order."""


# =============================================================================
# Self-critique
# =============================================================================

SELF_CRITIQUE_V1 = """You are a senior oncology physician reviewing the top-5 \
trial recommendations for this patient.

Patient: {patient_summary}

Top 5 recommendations with rationale:
{top5_block}

Critically review:
1. Are there any obvious clinical mismatches in the top-3?
2. Is any trial in the top-5 that the patient would clearly find unacceptable \
(geography, phase, prior treatment requirement)?
3. Is the order between #1 and #2 defensible?
4. Are critical NEI criteria being underweighted?

Output JSON:
{{
  "issues_found": [{{"trial_id": "...", "issue": "...", "severity": "low|medium|high"}}],
  "rerank_needed": true|false,
  "rerank_instructions": "...",
  "final_notes": "..."
}}"""


# =============================================================================
# Question generator (T4)
# =============================================================================

QUESTION_GEN_V1 = """You are a clinical research coordinator preparing a pre-screening \
checklist for a physician. The patient is being considered for trial {nct_id}: \
"{trial_title}".

This criterion is currently UNDETERMINED for this patient:
"{criterion_text}" (criterion ID: {criterion_id})

Available patient info: {patient_summary}
Missing info preventing decision: {what_is_missing}

Generate ONE clinical question for the physician with these 6 elements (all required):
1. data_point: specific clinical data requested (with units if applicable)
2. time_window: within what period the data must be valid (e.g., "within last 30 days")
3. measurement_context: lab type, imaging modality, scoring system
4. rationale: why this matters (1 sentence on clinical relevance, citing the trial)
5. expected_data_type: "numeric" | "categorical" | "date" | "boolean" | "text"
6. priority: "critical" | "high" | "medium" | "low"

Also provide:
- question_text: the actual question (max 50 words, specific, actionable)
- units: optional unit string

Output JSON with all fields populated."""


# =============================================================================
# Executive summary for the dossier (T5)
# =============================================================================

DOSSIER_SUMMARY_V1 = """Write a 3-5 line executive summary explaining why this \
trial fits this patient. Mention disease alignment, key biomarker matches, \
phase appropriateness, and one concrete reason for caution if any. \
Tone: clinical, concise, no marketing language.

Patient: {patient_summary}
Trial title: {trial_title}
Phase: {phase}
Status: {status}
Inclusion match: {n_inc_met}/{n_inc} met, {n_inc_nei} NEI
Exclusion clear: {n_exc_not_met}/{n_exc} clean, {n_exc_nei} NEI
Top inclusion evidence: {top_evidence}

Output: plain text, no JSON."""
