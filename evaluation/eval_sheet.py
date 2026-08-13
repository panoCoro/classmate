import csv
import json
import sys
from pathlib import Path

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(Path("evaluation").glob("results_*.jsonl"))[0]

print(f"Building sheet from: {RESULTS}")

SHEET = Path("evaluation/eval_sheet.csv")

with open(RESULTS, encoding="utf-8") as f:
    records = [json.loads(line) for line in f]
with open(SHEET, "w", newline="", encoding="utf-8") as out:
    writer = csv.writer(out)
    writer.writerow(["id", "category", "question", "answer", "sources", "correct (y/partial/n)", "right_sources (y/n)", "should_have_refused (y/n)", "comments"])
    for r in records:
        writer.writerow([r["id"], r["category"], r["question"], r["answer"],"; ".join(r["sources"]), "", "", "", ""])
print(f"Labelling sheet written to {SHEET}")

