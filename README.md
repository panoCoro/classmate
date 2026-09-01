# Classmate: ask your course repo 

Classmate is a local RAG chatbot over the corpus of an NLP module at Birkbeck, University of London (2026).

# Corpus
The module corpus is in a separate repository: https://github.com/panoCoro/nlp-course-corpus (will remain available only for the marking period)

Used with written permission of the module convenor, Dr. Paul Nulty. 

# Architecture

Hybrid retrieval (dense + BM25 + RRF), cross-encoder reranking.
Llama 3.2 3b reader, NLI faithfulness score per sentence, sources from metadata.
Runs entirely offline on 8GB RAM.  

# Requirements / Setup

Python 3.13 and Ollama

                python -m venv .venv
                source .venv/bin/activate
                pip install -r requirements.txt
                ollama pull llama3.2:3b
                ollama pull nomic-embed-text
                python -m src.ingest  # downloads the corpus automatically 

# UI

                streamlit run app.py

# Testing 

                python -m pytest

# Evaluation 

Results in Chapter 7 of the Report

                python -m evaluation.retrieval_eval # comparison
                python -m evaluation.run_eval # QA set evaluation 

RAGAS requires OpenAI API: scores in evaluation/ragas_scores.csv

# Structure

src/ingest.py
src/retrieve.py
src/generate.py
src/verify.py
app.py
evaluation/
tests/ 