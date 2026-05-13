from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from scripts.run_final_delivery import build_command


def _args(**overrides):
    base = {
        "year": "2021",
        "topic_limit": 75,
        "topic_ids": "",
        "run_name": "FINAL_DELIVERY_TREC2021_P12_FULL_75",
        "output_root": Path("results/experiments"),
        "concurrency": 4,
        "overwrite": False,
        "metric_only": False,
        "dry_run": False,
    }
    base.update(overrides)
    return Namespace(**base)


def test_final_delivery_wrapper_uses_accepted_p12_profile():
    cmd = build_command(_args())

    assert "scripts\\run_mini_eval.py" in " ".join(cmd) or "scripts/run_mini_eval.py" in " ".join(cmd)
    assert cmd[cmd.index("--max-trials-per-topic") + 1] == "10"
    assert cmd[cmd.index("--benchmark-candidate-selection-policy") + 1] == "top_score"
    assert cmd[cmd.index("--benchmark-entity-rerank-policy") + 1] == "rerank_final"
    assert cmd[cmd.index("--benchmark-entity-rerank-weight") + 1] == "0.09"
    assert cmd[cmd.index("--benchmark-entity-protect-top") + 1] == "3"
    assert cmd[cmd.index("--benchmark-criterion-evidence-policy") + 1] == "score_adjust"
    assert cmd[cmd.index("--benchmark-criterion-evidence-weight") + 1] == "0.50"
    assert "--no-dense" in cmd
    assert "--enable-few-shot" in cmd
    assert "--include-retrieval-traces" in cmd
    assert "--enable-multisignal-irrel-heuristic" in cmd


def test_final_delivery_metric_only_topic_check_skips_t4_t5_only():
    cmd = build_command(
        _args(
            topic_ids="75",
            run_name="CHECK_TOPIC_75",
            overwrite=True,
            metric_only=True,
        )
    )

    assert cmd[cmd.index("--topic-ids") + 1] == "75"
    assert cmd[cmd.index("--run-name") + 1] == "CHECK_TOPIC_75"
    assert "--overwrite" in cmd
    assert "--no-questions" in cmd
    assert "--no-dossiers" in cmd
    assert "--no-llm-judge" in cmd
