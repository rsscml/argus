"""Dense embedders (architecture SS7.3 stage 6, AD-10).

`AzureEmbedder` is production (langchain-openai). `HashingEmbedder` is a
deterministic, dependency-free stand-in for dev and tests — NOT for
production retrieval quality. Selection: settings.embedder ("azure"|"hashing").

The (name, dimension) pair is recorded at collection creation; a mismatch on
a later run raises — re-embedding is an explicit index-version bump (SS12.1, R5).
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


class Embedder(Protocol):
    name: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic bag-of-hashed-tokens embedding (cosine-comparable)."""

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension
        self.name = f"hashing-{dimension}"

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode()).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class AzureEmbedder:
    """Azure OpenAI embeddings via the AD-10 factory."""

    def __init__(self) -> None:
        from argus.llm.factory import get_azure_embeddings, model_fingerprint

        self._inner = get_azure_embeddings()
        fp = model_fingerprint()
        self.name = f"azure:{fp['embedding_deployment']}"
        self.dimension = len(self._inner.embed_query("dimension probe"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)


def make_embedder(kind: str, hashing_dim: int = 256) -> Embedder:
    if kind == "azure":
        return AzureEmbedder()
    if kind == "hashing":
        return HashingEmbedder(hashing_dim)
    raise ValueError(f"unknown embedder {kind!r} (expected 'azure' or 'hashing')")
