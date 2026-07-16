from git import Repo
import os

CORPUS_URL = "https://github.com/panoCoro/nlp-course-corpus"
CORPUS_DIR = "data/corpus"

def get_corpus():
    if not os.path.exists(CORPUS_DIR):
        print(f"Cloning corpus from {CORPUS_URL} to {CORPUS_DIR}...")
        Repo.clone_from(CORPUS_URL, CORPUS_DIR)
        print("Corpus cloned successfully.")
    else:
        print(f"Corpus already exists at {CORPUS_DIR}. Pulling latest changes...")
        Repo(CORPUS_DIR).remotes.origin.pull()
        print("Corpus updated successfully.")
get_corpus()