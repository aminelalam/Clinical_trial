import json

with open("results/smoke.json") as f:
    d = json.load(f)

print(f"Topics: {len(d['topics'])}")
for t in d["topics"]:
    topic_id = t["topic_id"]
    n_trials = len(t["ranked_trials"])
    error = t.get("error", "none")
    n_q = len(t.get("questions", []))
    n_d = len(t.get("dossiers", []))
    print(f"  {topic_id}: trials={n_trials}, questions={n_q}, dossiers={n_d}, error={error[:120]}")
    for trial in t["ranked_trials"][:3]:
        print(f"    rank={trial['rank']} {trial['nct_id']} score={trial['score']} label={trial['label']}")
