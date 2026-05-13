"""Compare metrics across runs."""
import json
from pathlib import Path

for name in ["mini_trec2021_p0", "01_bug_fixes", "02_quality_pass_v1"]:
    tm = Path(f"results/experiments/{name}/trec_metrics.json")
    em = Path(f"results/experiments/{name}/eligibility_metrics.json")
    if not tm.exists():
        print(f"{name}: no trec_metrics")
        continue
    t = json.loads(tm.read_text(encoding="utf-8"))
    e = json.loads(em.read_text(encoding="utf-8")) if em.exists() else {}
    el = e.get("trial_level", {})
    ndcg = t.get("ndcg_cut_10", 0)
    recall = t.get("recall_20", 0)
    p10 = t.get("P_10", 0)
    mapp = t.get("map", 0)
    rr = t.get("recip_rank", 0)
    mf1 = el.get("micro_f1", 0)
    macrof1 = el.get("macro_f1", 0)
    print(f"=== {name} ===")
    print(f"  NDCG@10={ndcg:.4f}, Recall@20={recall:.4f}, P@10={p10:.4f}, MAP={mapp:.4f}, RR={rr:.4f}")
    print(f"  Micro-F1={mf1:.4f}, Macro-F1={macrof1:.4f}")
    print()