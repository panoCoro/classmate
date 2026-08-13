import json
from pathlib import Path
from src.retrieve import dense_search, bm25_search, hybrid_candidates, retrieve

qa = json.loads(Path("evaluation/qa_set.json").read_text(encoding="utf-8"))
answerable = [x for x in qa if x.get("expected_sources")] 

def first_rank(top_srcs, expected):
    for i, s in enumerate(top_srcs, start=1):
        if s in expected:
            return i
    return 0   # miss

rows = []
for item in answerable:
    q, exp = item["question"], item["expected_sources"]
    
    conds = {
        "dense": [doc.metadata["source"] for doc, score in dense_search(q)[:5]],
        "bm25": [meta["source"] for text, meta in bm25_search(q)[:5]],
        "fused": [doc.metadata["source"] for doc, score in hybrid_candidates(q)[:5]],
        "full": [doc.metadata["source"] for doc, score in retrieve(q)[:5]]
    }
    
    row = {"id": item["id"], "category": item["category"]}
    for k, srcs in conds.items():
        r = first_rank(srcs, exp)
        row[f"{k}_rank"] = r
        row[f"{k}_hit"] = r > 0
    rows.append(row)

for cond in ["dense", "bm25", "fused", "full"]:
    ranks = [r[f"{cond}_rank"] for r in rows]
    hits = sum(r[f"{cond}_hit"] for r in rows)
    mrr = sum(1/r if r else 0 for r in ranks) / len(ranks)
    print(f"{cond}: hits {hits}/{len(rows)} | source-level MRR@5 = {mrr:.2f}")

json.dump(rows, open("evaluation/retrieval_eval.json", "w"), indent=2)