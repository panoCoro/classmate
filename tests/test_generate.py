from langchain_core.documents import Document
from src.generate import build_context, build_prompt

TEST = [(Document(page_content= "TF-IDF weighs terms by rarity.", metadata={"source": "Week 2.pdf"}), -1.0)]

def test_context_names_and_numbers_sources():
    ctx = build_context(TEST)
    assert "[1]" in ctx
    assert "Week 2.pdf" in ctx
    assert "TF-IDF weighs terms by rarity." in ctx
    
def test_prompt_question_and_verbatim_refusal():
    prompt = build_prompt(build_context(TEST), "what is tf-idf?")
    assert "what is tf-idf?" in prompt
    assert "could not find this in the course materials" in prompt.lower()

def test_refusal_needle():
    from src.generate import REFUSAL_NEEDLE, PROMPT_TEMPLATE
    assert REFUSAL_NEEDLE in PROMPT_TEMPLATE.lower()