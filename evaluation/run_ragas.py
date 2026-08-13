import json
import sys
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI
from ragas import evaluate
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics._faithfulness import faithfulness
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision

load_dotenv()

client = OpenAI()
llm = llm_factory("gpt-4.1", client=client)
embeddings = embedding_factory("text-embedding-3-small")

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(Path("evaluation").glob("results_*.jsonl"))[0]
print(f"Reading: {RESULTS}")

qa_lookup = {x["id"]: x["expected_answer"]
             for x in json.loads(Path("evaluation/qa_set.json").read_text(encoding="utf-8"))}

rows = []
with open(RESULTS, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["refused"]:
            continue
        rows.append({"question": r["question"], "answer": r["answer"],
                     "contexts": r["contexts"], "ground_truth": qa_lookup[r["id"]]})

dataset = Dataset.from_list(rows)
scores = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=llm,
    embeddings=embeddings,
)

print(scores)
scores.to_pandas().to_csv("evaluation/ragas_scores.csv", index=False)
print("Saved to evaluation/ragas_scores.csv")