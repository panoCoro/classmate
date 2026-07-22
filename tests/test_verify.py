from src.verify import split_sentences

def test_short_sentences():
    result = split_sentences("First test sentence here. Second one here. Tiny.")
    assert result == ["First test sentence here.", "Second one here."]
    
def test_punctuation_remains():
    result = split_sentences("Is this a question? Yes, it is.")
    assert all(s[-1] in ".!?" for s in result)
    
def test_empty_input():
    assert split_sentences("") == []
    