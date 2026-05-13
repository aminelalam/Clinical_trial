"""Quick P0 analysis."""
import json
from pathlib import Path

for name in ["mini_trec2021_p0", "01_bug_fixes", "02_quality_pass_v1"]:
    p = Path(f"results/experiments/{name}/predictions.json")
    if not p.exists():
        print(f"  {name}: NOT FOUND")
        continue
    d = json.loads(p.read_text(encoding="utf-8"))
    m = d["metadata"]
    s = m.get("settings", {})
    labels = []
    scores = []
    veto = 0
    for t in d["topics"]:
        for rt in t.get("ranked_trials", []):
            labels.append(rt["label"])
            scores.append(rt["score"])
            if rt["score"] <= -0.99:
                veto += 1
    eligible = labels.count("eligible")
    excludes = labels.count("excludes")
    total = len(labels)
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"=== {name} ===")
    print(f"  Mode: {s.get('runner_mode','?')}, trials: {s.get('max_trials_per_topic','?')}, criteria: {s.get('max_criteria_per_trial','?')}")
    print(f"  Total ranked: {total}, Eligible: {eligible}, Excludes: {excludes}, Veto: {veto}")
    print(f"  Avg score: {avg_score:.4f}")