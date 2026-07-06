"""Sparse lexical encoder for hybrid retrieval (architecture SS7.4).

Client side we produce BM25-style term-frequency weights over hashed token
indices; Qdrant applies the IDF component (sparse vector Modifier.IDF).
Tokenization deliberately preserves entity-ish tokens — tickers, part numbers,
vessel names ("cl", "ab-1234", "ever-given") — because exact lexical matching
is where niche-domain recall is won (SS7.4).
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-\._/]*")

_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    this to was were will with not but they their there been being over under
    after before between into through during""".split()
)


def tokenize(text: str) -> list[str]:
    tokens = []
    for match in _TOKEN_RE.finditer(text.lower()):
        token = match.group(0).strip("-._/")
        if len(token) >= 2 and token not in _STOPWORDS:
            tokens.append(token)
    return tokens


def token_index(token: str) -> int:
    return int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "big")


class Bm25SparseEncoder:
    """BM25 TF saturation with length normalization (k1/b); IDF is Qdrant's job."""

    name = "bm25-hash32"

    def __init__(self, k1: float = 1.5, b: float = 0.75, avg_doc_len: int = 400) -> None:
        self.k1 = k1
        self.b = b
        self.avg_doc_len = avg_doc_len

    def encode(self, text: str) -> tuple[list[int], list[float]]:
        tokens = tokenize(text)
        if not tokens:
            return [], []
        doc_len = len(tokens)
        counts = Counter(tokens)
        weights: dict[int, float] = {}
        norm = self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
        for token, tf in counts.items():
            weight = tf * (self.k1 + 1) / (tf + norm)
            idx = token_index(token)
            weights[idx] = weights.get(idx, 0.0) + weight  # hash collisions merge
        indices = sorted(weights)
        return indices, [weights[i] for i in indices]
