from langchain_ollama import ChatOllama
from src.retrieve import retrieve

LLM_MODEL = "llama3.2:3b"
TEMPERATURE = 0.2
NUM_CTX = 4096

PROMT_TEMPLATE = """You are Classmate chatbot, a teaching assistant
for the Natural Language Processing module. 

1. You are given the excerpts from the corpus below.
2. You are also given a question from a student.
3. Answer based on the excerpts and the question. If the excerpts do not contain enough information, say "I don't know" instead of making up an answer.
4. Do not assume and do not use any information that is not present in the excerpts.
5. Keep the answer concise and to the point. Avoid unnecessary details.

Corpus Excerpts:
{context}
Student Question: {question}

Answer:"""

def build_prompt(context, question):
    return PROMT_TEMPLATE.format(context=context, question=question)

def build_context(results):
    parts = []
    for chunk, score in results:
        parts.append(f"Source: {chunk.metadata['source']}\nContent: {chunk.page_content}")
    return "\n\n".join(parts)

def generate_answer(question):
    results = retrieve(question)
    context = build_context(results)
    prompt = build_prompt(context, question)
    llm = ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE, num_ctx=NUM_CTX)
    response = llm.invoke(prompt)
    sources = []
    for chunk, score in results:
        src = chunk.metadata["source"]
        if src not in sources:
            sources.append(src)
    return response.content, sources, results

if __name__ == "__main__":
    question = input("Ask Classmate: ") 
    answer, sources, results = generate_answer(question) 
    print(f"\nAnswer: {answer}\n")
    print("Sources:")
    for s in sources:
        print(f"- {s}")