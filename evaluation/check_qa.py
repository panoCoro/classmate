import json
from pathlib import Path

VALID_CATEGORIES = {'admin', 'factual', 'code', 'oos_zero', 'oos_echo', 'in_scope_unanswerable'}
REFUSAL_CATEGORIES = {'oos_zero', 'oos_echo', 'in_scope_unanswerable'}
VALID_SOURCES = {'Key module information.pdf', 'Lab Sheet Week 1.pdf', 'Lab Sheet Week 2.pdf', 'Lab Sheet Week 3.pdf', 'Lab Sheet Week 4.pdf', 'Lab Sheet Week 5.pdf', 'Lab Sheet Week 6.pdf', 'README.md', 'Week 1 admin.pdf', 'Week 1.pdf', 'Week 2.pdf', 'Week 3 benepar-example.py', 'Week 3 syntax-examples.py', 'Week 3.pdf', 'Week 4.pdf', 'Week 5.pdf', 'Week 6 lab-scripttest.py', 'Week 6 nli.py', 'Week 6 query-test.py', 'Week 6 summarisation-test.py', 'Week 6.pdf', 'Week 7.pdf', 'Week 8.pdf', 'Week 9.pdf'}

qa = json.loads(Path("evaluation/qa_set.json").read_text(encoding="utf-8"))
ids = [item["id"] for item in qa]
problems = []
if len(ids) != len(set(ids)):
    problems.append("duplicate ids present")

for item in qa:
    if item["category"] not in VALID_CATEGORIES:
        problems.append(f"{item['id']}: unknown category '{item['category']}'")
    if not item["question"].strip() or not item["expected_answer"].strip():
        problems.append(f"{item['id']}: question or expected_answer is empty")
    for src in item["expected_sources"]:
        if src not in VALID_SOURCES:
            problems.append(f"{item['id']}: source '{src}' not in corpus")
    if item["paraphrase_of"] is not None and item["paraphrase_of"] not in ids:
        problems.append(f"{item['id']}: paraphrase_of points to missing id")
    if item["category"] in REFUSAL_CATEGORIES and item["expected_answer"] != "REFUSAL":
        problems.append(f"{item['id']}: refusal-category entries must have expected_answer REFUSAL")
    if item["expected_answer"] == "REFUSAL" and item["expected_sources"]:
        problems.append(f"{item['id']}: REFUSAL entries should have empty sources")
        
counts = {}
for item in qa:
    counts[item["category"]] = counts.get(item["category"], 0) + 1
print(f"{len(qa)} questions | per category: {counts}")
print("ALL CHECKS PASSED" if not problems else "\n".join(problems))