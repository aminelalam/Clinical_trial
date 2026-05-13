"""Analyze predictions from a mini-eval run."""
import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/experiments/02_quality_pass_v1/predictions.json")
p = json.loads(path.read_text(encoding="utf-8"))
m = p["metadata"]
print(f"=== Run Metadata ===")
print(f"Topics: {m['n_topics']}")
print(f"Mode: {m['settings']['runner_mode']}")
print(f"max_trials: {m['settings']['max_trials_per_topic']}")
print(f"max_criteria: {m['settings']['max_criteria_per_trial']}")
print(f"few_shot: {m['settings']['use_few_shot']}")
print(f"verifier: {m['settings']['use_verifier']}")
print(f"hyde: {m['settings']['use_hyde']}")
print(f"listwise: {m['settings']['use_listwise']}")
print(f"mesh_loaded: {m['settings']['mesh_loaded']}")
print(f"mesh_concepts: {m['settings']['mesh_concepts']}")
print(f"retrieval_attempts: {m['settings'].get('max_retrieval_attempts', 'N/A')}")
print()

for t in p["topics"]:
    tid = t["topic_id"]
    n_ranked = len(t.get("ranked_trials", []))
    has_error = t.get("error")
    diag = t.get("diagnostics", {})
    cc = diag.get("candidate_counts", {})
    print(f"--- Topic {tid} ---")
    print(f"  Ranked: {n_ranked}, Error: {has_error}")
    print(f"  BM25={cc.get('bm25',0)}, Dense={cc.get('dense',0)}, Fused={cc.get('fused',0)}, "
          f"Reranked={cc.get('reranked',0)}, Final={cc.get('final',0)}, Viable={cc.get('viable',0)}, "
          f"Evaled={cc.get('trial_evals',0)}")
    filter_reasons = diag.get("filter_reasons", {})
    if filter_reasons:
        print(f"  Filter reasons: {filter_reasons}")
    for rt in t.get("ranked_trials", [])[:5]:
        predicted_qrel = rt.get("predicted_trec_qrel", rt.get("trec_qrel"))
        print(f"  {rt['nct_id']}: score={rt['score']:.4f}, label={rt['label']}, "
              f"predicted_trec_qrel={predicted_qrel}, "
              f"inc={rt['n_inclusion_met']}/{rt['n_inclusion_total']}, "
              f"exc={rt['n_exclusion_met']}/{rt['n_exclusion_total']}, "
              f"nei={rt['fraction_nei']:.3f}")
    print()

# Summary stats
total_ranked = sum(len(t.get("ranked_trials", [])) for t in p["topics"])
excludes_count = sum(1 for t in p["topics"] for rt in t.get("ranked_trials", [])
                     if rt["label"] == "excludes")
eligible_count = sum(1 for t in p["topics"] for rt in t.get("ranked_trials", [])
                     if rt["label"] == "eligible")
veto_count = sum(1 for t in p["topics"] for rt in t.get("ranked_trials", [])
                 if rt["score"] <= -0.99)
print(f"=== Summary ===")
print(f"Total ranked trials: {total_ranked}")
print(f"Eligible: {eligible_count}, Excludes: {excludes_count}")
print(f"Veto (score <= -0.99): {veto_count}")
