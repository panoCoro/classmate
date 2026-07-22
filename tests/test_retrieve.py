from src.retrieve import tokenize

def test_tokenize_lowercase_and_no_punctuation():
    assert tokenize("What's RAG?") == ["what", "s", "rag"]
    
def test_tokenize_keeps_code():
    assert tokenize("use porter_tokenizer()") == ["use", "porter_tokenizer"]    
    

