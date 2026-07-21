import re
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

EMBED_MODEL = "nomic-embed-text"
CHROMA_DIR = "chroma_db"
TOP_K = 5   
CANDIDATES = 20
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 20
RRF_K = 60

_DB = None
_BM25 = None
_RERANKER = None

def get_reranker():
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoder(RERANKER_MODEL)
    return _RERANKER

def get_db():
    global _DB
    if _DB is None:
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        _DB = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    return _DB

def tokenize(text):
    return re.findall(r'\w+', text.lower())

def get_bm25():
    global _BM25
    if _BM25 is None:
        data = get_db().get()
        texts = data['documents']
        metadatas = data['metadatas']
        index = BM25Okapi([tokenize(t) for t in texts])
        _BM25 = (index, texts, metadatas)
    return _BM25

def dense_search(question, n = CANDIDATES):
    return get_db().similarity_search_with_score(question, k= n)

def bm25_search(question, n = CANDIDATES):
    index, texts, metadatas = get_bm25()
    scores = index.get_scores(tokenize(question))
    ranked = sorted(range(len(texts)), key=lambda i: scores[i], reverse=True)
    return [(texts[i], metadatas[i]) for i in ranked[:n]]
    
def hybrid_candidates(question, n= RERANK_CANDIDATES):
    fused = {}
    docs = {}
    
    for rank, (doc, _dist) in enumerate(dense_search(question)):
        key = doc.page_content 
        fused[key] = fused.get(key, 0) + 1 / (RRF_K + rank + 1)
        docs[key] = doc
        
    for rank, (text, meta) in enumerate(bm25_search(question)):
        fused[text] = fused.get(text, 0) + 1 / (RRF_K + rank + 1)
        if text not in docs:
            docs[text] = Document(page_content=text, metadata=meta)
            
    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
    return [(docs[key], score) for key, score in ranked[:n]]

def rerank(question, candidates, k= TOP_K):
    pairs = [(question, doc.page_content) for doc, _rrf in candidates]
    scores = get_reranker().predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
    return [(doc, score) for (doc, _rrf), score in ranked[:k]]

def retrieve(question, k=TOP_K):
    candidates = hybrid_candidates(question)
    return rerank(question, candidates, k)





if __name__ == "__main__":
    question = input("Enter your question: ")
    results = retrieve(question)
    for i, (chunk, score) in enumerate(results):
        print(f"\nResult {i+1} | Score: {score:.2f}")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Content: {chunk.page_content[:300]}...")  