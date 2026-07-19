from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

EMBDED_MODEL = "nomic-embed-text"
CHROMA_DIR = "chroma_db"
TOP_K = 5   

def get_db():
    embeddings = OllamaEmbeddings(model=EMBDED_MODEL)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    
def retrieve(question, top_k=TOP_K):
    db = get_db()
    results = db.similarity_search_with_score(question, k=top_k)
    return results

if __name__ == "__main__":
    question = input("Enter your question: ")
    results = retrieve(question)
    for i, (chunk, score) in enumerate(results):
        print(f"\nResult {i+1} | Score: {score:.4f}")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Content: {chunk.page_content[:500]}...")  