from langchain_core.documents import Document
from src.ingest import chunk_documents

def make_docs():
    long_text = "Natural Language Processing content for testing. " * 60
    return [Document(page_content=long_text, metadata={"source": "test_source1"}),
            Document(page_content=long_text, metadata={"source": "test_source2"})]

def test_chunks_are_produced():
    assert len(chunk_documents(make_docs())) > 1
    
def test_metadata_survives_chunking():
    chunks = chunk_documents(make_docs())
    assert all("source" in c.metadata for c in chunks)
    assert {c.metadata["source"] for c in chunks} == {"test_source1", "test_source2"}
    
def test_chunking_is_deterministic():
    a = [c.page_content for c in chunk_documents(make_docs())]
    b = [c.page_content for c in chunk_documents(make_docs())]
    assert a == b
    
def test_chunk_size():
    assert all(len(c.page_content) <= 800 for c in chunk_documents(make_docs()))