import re
from sentence_transformers import CrossEncoder

NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
MIN_SENTENCE_CHARS = 10

_VERIFIER = None

def get_verifier():
    global _VERIFIER
    if _VERIFIER is None:
        _VERIFIER = CrossEncoder(NLI_MODEL)
    return _VERIFIER

def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= MIN_SENTENCE_CHARS]

def faithfulness(answer, results):
    verifier = get_verifier()
    labels = {name.lower(): idx for name, idx in verifier.model.config.label2id.items()}
    entail_idx = labels["entailment"]
    
    chunks = [doc.page_content for doc, _score in results]
    sentences = split_sentences(answer)
    if not sentences:
        return None, []
    
    details = []
    for sentence in sentences:
        pairs = [(chunk, sentence) for chunk in chunks]
        probs = verifier.predict(pairs, apply_softmax=True)
        best = float(max(row[entail_idx] for row in probs))
        details.append((sentence, best))
        
    overall = min(score for _s, score in details)
    return overall, details