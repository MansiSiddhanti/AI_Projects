# Plain BM25 (Okapi). Small enough that pulling in rank_bm25 wasn't worth it.
import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    "a an and are as at be been but by can do does for from has have how in into is "
    "it its of on or such that the their them then there these they this to was were "
    "what when where which who why will with would".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.term_freqs = [Counter(tokenize(doc)) for doc in documents]
        self.doc_lengths = [sum(tf.values()) for tf in self.term_freqs]
        self.n_docs = len(documents)
        self.avg_length = (sum(self.doc_lengths) / self.n_docs) if self.n_docs else 0.0

        doc_freq: Counter[str] = Counter()
        for tf in self.term_freqs:
            doc_freq.update(tf.keys())
        self.idf = {
            term: math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def __len__(self) -> int:
        return self.n_docs

    def scores(self, query: str) -> list[float]:
        terms = [t for t in tokenize(query) if t in self.idf]
        results = [0.0] * self.n_docs
        if not terms or not self.avg_length:
            return results
        for i, tf in enumerate(self.term_freqs):
            norm = self.k1 * (1 - self.b + self.b * self.doc_lengths[i] / self.avg_length)
            score = 0.0
            for term in terms:
                freq = tf.get(term)
                if freq:
                    score += self.idf[term] * freq * (self.k1 + 1) / (freq + norm)
            results[i] = score
        return results

    def top_n(self, query: str, n: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        ranked = [
            (i, s)
            for i, s in enumerate(self.scores(query))
            if s > 0 and (allowed is None or i in allowed)
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:n]
