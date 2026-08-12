import json
import time
from datetime import datetime
from pathlib import Path
from src.generate import generate_answer, is_refusal
from src.verify import faithfulness

QA_PATH = Path("evaluation/qa_set.json")
RESULTS_PATH = Path(f"evaluation/results_{datetime.now():%Y%m%d_%H%M}.jsonl")

def load_qa_set():
    with open(QA_PATH, encoding="utf-8") as f:
        return json.load(f)
    
def evaluate_one(item):
    t0 = time.perf_counter()
    answer, sources, results = generate_answer(item["question"])
    elapsed = time.perf_counter() - t0
    refused = is_refusal(answer)
    retrieval = [{"source": doc.metadata["source"], "score": float(score)} for doc, score in results]
    contexts = [doc.page_content for doc, _score in results]
    record = {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "answer": answer,
        "refused": refused,
        "sources": sources,
        "retrieval": retrieval,
        "top1_score": retrieval[0]["score"] if retrieval else None,
        "contexts": contexts,
        "seconds": round(elapsed, 2),}
    if refused:
        record["faithfulness_min"] = None
        record["faithfulness_mean"] = None
        record["sentences"] = []
    else:
        overall, details = faithfulness(answer, results)
        record["faithfulness_min"] = round(overall, 2) if overall is not None else None
        record["faithfulness_mean"] = round(sum(s for _t, s in details) / len(details), 4) if details else None
        record["sentences"] = [{"text": t, "support": round(s, 4)} for t, s in details]
    return record

if __name__ == "__main__":
    qa = load_qa_set()
    print(f"Running evaluation on {len(qa)} questions...")  
    with open(RESULTS_PATH, "a", encoding="utf-8") as out:
        for i, item in  enumerate (qa, start = 1):
            print(f"[{i}/{len(qa)}]{item['id']}: {item['question'][:60]}")
            record = evaluate_one(item)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            status = "REFUSED" if record["refused"] else f"faith_min={record['faithfulness_min']}"
            top1 = record["top1_score"]
            top1_str = f"top1={top1:.2f}" if top1 is not None else "n/a"
            print(f"  -> {record['seconds']}s, | {top1_str} | {status}")
            
        