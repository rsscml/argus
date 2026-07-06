"""Syndication clustering (SS7.3 stage 3).

MinHash over word 5-shingles; LSH at a high Jaccard threshold groups
near-duplicate documents (wire copy reprints) under one story_cluster_id, so
corroboration later counts them as ONE independent voice (SS8.4).

Signatures persist in doc_signatures so clustering works across daily runs
within a rolling window.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
from datasketch import MinHash, MinHashLSH
from sqlalchemy import select
from sqlalchemy.orm import Session

from argus.ingest.sparse import tokenize
from argus.snapshots.db import DocSignatureRow, SnapshotRow
from argus.util import utcnow

NUM_PERM = 128
THRESHOLD = 0.8
SHINGLE = 5


def minhash_for(text: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    tokens = tokenize(text)
    if len(tokens) < SHINGLE:
        shingles = [" ".join(tokens)] if tokens else []
    else:
        shingles = [" ".join(tokens[i : i + SHINGLE]) for i in range(len(tokens) - SHINGLE + 1)]
    for shingle in shingles:
        m.update(shingle.encode("utf-8"))
    return m


class SyndicationClusterer:
    """LSH index seeded from recent persisted signatures; assigns cluster ids."""

    def __init__(self, session: Session, window_days: int = 14) -> None:
        self.session = session
        self.lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
        self._cluster_of: dict[str, str] = {}
        cutoff = utcnow() - timedelta(days=window_days)
        rows = self.session.execute(
            select(DocSignatureRow, SnapshotRow.cluster_id)
            .join(SnapshotRow, SnapshotRow.content_hash == DocSignatureRow.content_hash)
            .where(DocSignatureRow.created_at >= cutoff)
        ).all()
        for sig, cluster_id in rows:
            m = MinHash(num_perm=NUM_PERM,
                        hashvalues=np.frombuffer(sig.minhash, dtype=np.uint64))
            key = sig.content_hash
            if key not in self.lsh:
                self.lsh.insert(key, m)
            self._cluster_of[key] = cluster_id or key[:16]

    def assign(self, content_hash: str, text: str) -> str:
        """Return the cluster id for this document, registering its signature."""
        m = minhash_for(text)
        matches = self.lsh.query(m)
        cluster_id = next(
            (self._cluster_of[h] for h in matches if h in self._cluster_of),
            None,
        ) or content_hash[:16]

        if content_hash not in self.lsh:
            self.lsh.insert(content_hash, m)
        self._cluster_of[content_hash] = cluster_id
        if self.session.get(DocSignatureRow, content_hash) is None:
            self.session.add(
                DocSignatureRow(
                    content_hash=content_hash,
                    minhash=np.asarray(m.hashvalues, dtype=np.uint64).tobytes(),
                )
            )
        return cluster_id
