"""Step 5 of RAG: find the chunks most relevant to a question.

We use BM25 (the ranking algorithm behind many search engines). It needs no
heavy ML libraries, so installation is easy. In Version 3 you can swap this for
embeddings + FAISS for semantic search.
"""
import math
import re
from collections import Counter

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "to", "in", "on", "for",
    "and", "or", "what", "which", "who", "how", "does", "do", "did", "this",
    "that", "it", "with", "as", "by", "at", "be", "from", "about", "me", "tell",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, chunks, k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [tokenize(c.text) for c in chunks]
        self.tf = [Counter(d) for d in self.docs]
        n = len(self.docs)
        self.avgdl = (sum(len(d) for d in self.docs) / n) if n else 1.0
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int = 5):
        """Return [(chunk, score), ...] best first. Empty if nothing matches."""
        q_tokens = set(tokenize(query))
        scored = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i]) or 1
            score = 0.0
            for t in q_tokens:
                f = tf.get(t)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                    score += self.idf[t] * f * (self.k1 + 1) / denom
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [(self.chunks[i], s) for s, i in scored[:k]]
