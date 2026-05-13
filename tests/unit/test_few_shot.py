"""Tests for the curated few-shot bank and the FewShotBank loader.

These tests run against the real banco_few_shot/ directory at the repo root
(not a fixture) so the v1 sizing/category contract is enforced as code.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BANK_DIR = REPO_ROOT / "banco_few_shot"


def test_bank_loads_with_expected_size_and_files():
    from trial_matcher.llm.few_shot import FewShotBank

    bank = FewShotBank.from_jsonl_dir(BANK_DIR)
    assert len(bank.examples) == 42, f"expected 42 v1 examples, got {len(bank.examples)}"

    # Four files, each non-empty
    expected_stems = {"negation", "temporal", "biomarker_synonym", "generic"}
    found_stems = {p.stem for p in BANK_DIR.glob("*.jsonl")}
    assert found_stems == expected_stems, f"unexpected files: {found_stems ^ expected_stems}"


def test_decision_distribution_balanced_per_file():
    """Each category file has at least 1 met, 1 not_met, and 1 NEI example."""
    from trial_matcher.llm.few_shot import FewShotExample

    for path in BANK_DIR.glob("*.jsonl"):
        examples = [
            FewShotExample.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        decisions = Counter(e.decision for e in examples)
        # All three labels must be present so the model doesn't learn a default.
        assert decisions.get("met", 0) >= 1, f"{path.name}: 0 'met' examples"
        assert decisions.get("not_met", 0) >= 1, f"{path.name}: 0 'not_met' examples"
        assert decisions.get("NEI", 0) >= 1, f"{path.name}: 0 'NEI' examples"


def test_to_prompt_block_renders_expected_format():
    from trial_matcher.llm.few_shot import FewShotBank, FewShotExample

    ex = FewShotExample(
        id="t_001",
        category="test_cat",
        criterion_text="dummy criterion",
        patient_excerpt="dummy patient",
        decision="NEI",
        reasoning="dummy reasoning",
    )
    bank = FewShotBank()
    block = bank.to_prompt_block([ex])
    assert "### Example (test_cat)" in block
    assert "Criterion: dummy criterion" in block
    assert "Decision: NEI" in block


def test_select_with_fake_encoder_returns_k_examples():
    """End-to-end select() flow with a deterministic fake encoder."""
    from trial_matcher.llm.few_shot import FewShotBank, FewShotExample

    class HashEncoder:
        """Deterministic encoder that maps text length → fixed vector for tests."""

        @staticmethod
        def _vec(text: str) -> list[float]:
            rng = np.random.default_rng(seed=hash(text) % (2**32))
            v = rng.normal(size=8).astype(np.float32)
            return v.tolist()

        def encode_queries(self, texts):
            return [self._vec(t) for t in texts]

        def encode_query(self, text):
            return self._vec(text)

    bank = FewShotBank()
    bank.examples = [
        FewShotExample(
            id=f"x_{i}",
            category="test",
            criterion_text=f"criterion variant {i}",
            patient_excerpt="…",
            decision="met",
            reasoning="…",
        )
        for i in range(8)
    ]
    bank.index(HashEncoder())
    out = bank.select("query about variant 3", k=3, encoder=HashEncoder())
    assert len(out) == 3
    assert all(o.criterion_text.startswith("criterion variant") for o in out)


def test_select_uses_index_encoder_when_encoder_not_passed():
    """The dynamic bank must not silently degrade to examples[:k] after index()."""
    from trial_matcher.llm.few_shot import FewShotBank, FewShotExample

    class KeywordEncoder:
        def _vec(self, text: str) -> list[float]:
            t = text.lower()
            return [
                1.0 if "egfr" in t else 0.0,
                1.0 if "her2" in t else 0.0,
                1.0 if "renal" in t else 0.0,
            ]

        def encode_queries(self, texts):
            return [self._vec(t) for t in texts]

        def encode_query(self, text):
            return self._vec(text)

    bank = FewShotBank()
    bank.examples = [
        FewShotExample(
            id="first",
            category="test",
            criterion_text="adequate renal function",
            patient_excerpt="",
            decision="met",
            reasoning="",
        ),
        FewShotExample(
            id="target",
            category="test",
            criterion_text="EGFR exon 19 deletion required",
            patient_excerpt="",
            decision="met",
            reasoning="",
        ),
        FewShotExample(
            id="other",
            category="test",
            criterion_text="HER2 positive disease",
            patient_excerpt="",
            decision="met",
            reasoning="",
        ),
    ]
    bank.index(KeywordEncoder())

    out = bank.select("patient has EGFR mutation", k=1)

    assert out[0].id == "target"


def test_v2_prompt_has_few_shot_placeholder():
    """Sanity check: COT_ELIGIBILITY_V2 must template the few-shot block."""
    from trial_matcher.llm.prompts import COT_ELIGIBILITY_V1, COT_ELIGIBILITY_V2

    assert "{few_shot_block}" in COT_ELIGIBILITY_V2
    assert "{few_shot_block}" not in COT_ELIGIBILITY_V1
    # Both V1 and V2 must keep these two placeholders
    for tpl in (COT_ELIGIBILITY_V1, COT_ELIGIBILITY_V2):
        assert "{criterion_annotated}" in tpl
        assert "{patient_excerpts}" in tpl


def test_evaluator_accepts_optional_bank():
    """The LLMEvaluator constructor should accept few_shot_bank/encoder/k kwargs."""
    from trial_matcher.eligibility.llm_evaluator import LLMEvaluator
    from trial_matcher.llm.few_shot import FewShotBank

    bank = FewShotBank()  # empty bank; valid
    ev = LLMEvaluator(few_shot_bank=bank, encoder=None, few_shot_k=3)
    assert ev.few_shot_bank is bank
    assert ev.few_shot_k == 3
