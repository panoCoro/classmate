from git import Repo
import os
from pathlib import Path
import pymupdf4llm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CORPUS_URL = "https://github.com/panoCoro/nlp-course-corpus"
CORPUS_DIR = "data/corpus"
TEXT_SUFFIXES = [".txt", ".md", ".py"]
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

def get_corpus():
    if not os.path.exists(CORPUS_DIR):
        print(f"Cloning corpus from {CORPUS_URL} to {CORPUS_DIR}...")
        Repo.clone_from(CORPUS_URL, CORPUS_DIR)
        print("Corpus cloned successfully.")
    else:
        print(f"Corpus already exists at {CORPUS_DIR}. Pulling latest changes...")
        Repo(CORPUS_DIR).remotes.origin.pull()
        print("Corpus updated successfully.")

def load_corpus():
    docs = []
    skipped_files = []
    for path in Path(CORPUS_DIR).rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        try:
            if path.suffix.lower() == ".pdf":
                print(f" parsing {path.name} ...")
                text = pymupdf4llm.to_markdown(str(path))
            elif path.suffix.lower() in TEXT_SUFFIXES:
                text = path.read_text(encoding="utf-8", errors="ignore")
            else:
                skipped_files.append(path.name)
                continue               
        except Exception as e:
            print(f"Error reading {path.name}: {e}")
            skipped_files.append(path.name)
            continue
        docs.append(Document(page_content=text, metadata={"source": str(path.relative_to(CORPUS_DIR))}))
    print(f"Loaded {len(docs)} documents; skipped {len(skipped_files)}: {skipped_files}")
    return docs 

def chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    get_corpus()
    documents = load_corpus()
    chunks = chunk_documents(documents)