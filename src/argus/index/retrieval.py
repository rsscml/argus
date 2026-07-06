"""Retrieval API (architecture SS7.4).

    retrieve(query, *, domain, window=None, ...)

Invariant G1 at read time: results are filtered to snapshot sources inside the
domain's registry scope — indexing something never makes it citable.

Ranking is relevance only (dense + sparse fused with RRF). Tier is attached to
each result for downstream corroboration (SS8.4) but NEVER boosts ranking:
keeping ranking and trust orthogonal avoids starving the corroborator of the
disagreeing evidence it needs.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from qdrant_client import QdrantClient, models

from argus.config.profile import DomainProfile
from argus.config.registry import Registry
from argus.ingest.embed import Embedder
from argus.ingest.entities import QueryExpander
from argus.ingest.sparse import Bm25SparseEncoder
from argus.util import utcnow


@dataclass
class RetrievedChunk:
    chunk_id: str
    content_hash: str
    source_id: str
    tier: int | None  # min tier among the domain's include_tags for this source
    score: float
    text: str
    title: str | None
    url: str
    published_at: str | None
    cluster_id: str | None
    entity_ids: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.entity_ids is None:
            self.entity_ids = []


def _tier_in_domain(registry: Registry, profile: DomainProfile, source_id: str) -> int | None:
    try:
        source = registry.get(source_id)
    except KeyError:
        return None
    tiers = [
        t for tag in profile.registry_scope.include_tags
        if (t := source.tier_for(tag)) is not None
    ]
    return min(tiers) if tiers else None


def window_scan(
    *,
    registry: Registry,
    profile: DomainProfile,
    client: QdrantClient,
    collection: str,
    window_hours: int,
    limit: int = 500,
) -> list[RetrievedChunk]:
    """Query-less retrieval for the daily brief (SS8.2): every in-scope chunk
    published inside the window. No ranking — the brief is deterministic;
    story grouping and ordering happen downstream."""
    scope = profile.registry_scope
    allowed = [s.id for s in registry.in_scope(scope.include_tags, scope.min_tier)]
    if not allowed or not client.collection_exists(collection):
        return []  # nothing ingested yet -> empty window, not an error
    cutoff = utcnow() - timedelta(hours=window_hours)
    flt = models.Filter(must=[
        models.FieldCondition(key="source_id", match=models.MatchAny(any=allowed)),
        models.FieldCondition(key="published_ts",
                              range=models.Range(gte=int(cutoff.timestamp()))),
    ])
    points, _ = client.scroll(
        collection_name=collection, scroll_filter=flt, limit=limit, with_payload=True
    )
    chunks: list[RetrievedChunk] = []
    for point in points:
        payload = point.payload or {}
        if payload.get("kind") == "embedder_mark":
            continue
        source_id = payload.get("source_id", "")
        chunks.append(RetrievedChunk(
            chunk_id=payload.get("chunk_id", ""),
            content_hash=payload.get("content_hash", ""),
            source_id=source_id,
            tier=_tier_in_domain(registry, profile, source_id),
            score=0.0,
            text=payload.get("text", ""),
            title=payload.get("title"),
            url=payload.get("url", ""),
            published_at=payload.get("published_at"),
            cluster_id=payload.get("cluster_id"),
            entity_ids=payload.get("entity_ids") or [],
        ))
    chunks.sort(key=lambda c: c.published_at or "", reverse=True)
    return chunks


def retrieve(
    query: str,
    *,
    registry: Registry,
    profile: DomainProfile,
    client: QdrantClient,
    collection: str,
    embedder: Embedder,
    sparse: Bm25SparseEncoder,
    expander: QueryExpander | None = None,
    window_hours: int | None = None,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    top_k = top_k or profile.retrieval.top_k
    scope = profile.registry_scope
    allowed = [s.id for s in registry.in_scope(scope.include_tags, scope.min_tier)]
    if not allowed:
        return []

    must: list[models.Condition] = [
        models.FieldCondition(key="source_id", match=models.MatchAny(any=allowed))
    ]
    if window_hours is not None:
        cutoff = utcnow() - timedelta(hours=window_hours)
        must.append(models.FieldCondition(
            key="published_ts", range=models.Range(gte=int(cutoff.timestamp()))
        ))
    flt = models.Filter(must=must)

    expanded = expander.expand(query) if expander else query
    sparse_idx, sparse_val = sparse.encode(expanded)
    prefetch = [
        models.Prefetch(query=embedder.embed_query(query), using="dense",
                        limit=top_k * 4, filter=flt),
    ]
    if sparse_idx:
        prefetch.append(models.Prefetch(
            query=models.SparseVector(indices=sparse_idx, values=sparse_val),
            using="sparse", limit=top_k * 4, filter=flt,
        ))

    response = client.query_points(
        collection_name=collection,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        query_filter=flt,
        with_payload=True,
    )

    results: list[RetrievedChunk] = []
    for point in response.points:
        payload = point.payload or {}
        if payload.get("kind") == "embedder_mark":
            continue
        source_id = payload.get("source_id", "")
        results.append(
            RetrievedChunk(
                chunk_id=payload.get("chunk_id", ""),
                content_hash=payload.get("content_hash", ""),
                source_id=source_id,
                tier=_tier_in_domain(registry, profile, source_id),
                score=point.score or 0.0,
                text=payload.get("text", ""),
                title=payload.get("title"),
                url=payload.get("url", ""),
                published_at=payload.get("published_at"),
                cluster_id=payload.get("cluster_id"),
                entity_ids=payload.get("entity_ids") or [],
            )
        )
    return results
