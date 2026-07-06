"""Ingestion pipeline as a LangGraph StateGraph (architecture SS7.3, AD-9).

Linear, deterministic graph over the SS7.3 stages. Using LangGraph here keeps
one orchestration model across the system and lets M3 attach the Postgres
checkpointer to ingestion runs unchanged (SS8.1).

Idempotency (no framework magic): only extraction_status='pending' snapshots
are selected; chunk IDs are deterministic '<content_hash>:<n>' so re-upserts
overwrite; terminal statuses close the loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from argus.config.registry import Registry
from argus.ingest.cluster import SyndicationClusterer
from argus.ingest.embed import Embedder
from argus.ingest.entities import EntityTagger
from argus.ingest.extract import extract_text
from argus.ingest.sparse import Bm25SparseEncoder
from argus.index.qdrant import chunk_point, ensure_collection
from argus.snapshots.blob import BlobStore
from argus.snapshots.db import SnapshotRow


class ExtractedDoc(BaseModel):
    content_hash: str
    source_id: str
    text: str
    title: str | None = None
    url: str = ""
    published_at: str | None = None  # ISO, for payload display
    published_ts: int = 0
    cluster_id: str | None = None
    entity_ids: list[str] = Field(default_factory=list)


class IngestStats(BaseModel):
    selected: int = 0
    extracted: int = 0
    failed: int = 0
    unsupported: int = 0
    empty: int = 0
    chunks_upserted: int = 0
    clusters_joined: int = 0  # docs that landed in a pre-existing cluster


class IngestState(BaseModel):
    limit: int = 500
    pending: list[str] = Field(default_factory=list)
    docs: dict[str, ExtractedDoc] = Field(default_factory=dict)
    stats: IngestStats = Field(default_factory=IngestStats)


@dataclass
class IngestDeps:
    session: Session
    blob: BlobStore
    client: QdrantClient
    collection: str
    embedder: Embedder
    sparse: Bm25SparseEncoder
    tagger: EntityTagger
    registry: Registry
    chunk_size: int = 1600
    chunk_overlap: int = 200
    cluster_window_days: int = 14


def build_ingest_graph(deps: IngestDeps):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=deps.chunk_size, chunk_overlap=deps.chunk_overlap
    )

    def select_pending(state: IngestState) -> dict:
        rows = deps.session.scalars(
            select(SnapshotRow)
            .where(SnapshotRow.extraction_status == "pending")
            .order_by(SnapshotRow.fetched_at)
            .limit(state.limit)
        ).all()
        return {"pending": [r.content_hash for r in rows],
                "stats": state.stats.model_copy(update={"selected": len(rows)})}

    def extract(state: IngestState) -> dict:
        docs: dict[str, ExtractedDoc] = {}
        stats = state.stats.model_copy()
        for content_hash in state.pending:
            row = deps.session.get(SnapshotRow, content_hash)
            text, status = extract_text(deps.blob.get(content_hash), row.media_type)
            if status != "ok":
                row.extraction_status = status
                setattr(stats, status, getattr(stats, status) + 1)
                continue
            published = row.published_at or row.fetched_at
            docs[content_hash] = ExtractedDoc(
                content_hash=content_hash,
                source_id=row.source_id,
                text=text,
                title=row.title,
                url=row.url,
                published_at=published.isoformat(),
                published_ts=int(published.timestamp()),
            )
            stats.extracted += 1
        deps.session.commit()
        return {"docs": docs, "stats": stats}

    def cluster(state: IngestState) -> dict:
        clusterer = SyndicationClusterer(deps.session, deps.cluster_window_days)
        stats = state.stats.model_copy()
        docs = dict(state.docs)
        for content_hash, doc in docs.items():
            cluster_id = clusterer.assign(content_hash, doc.text)
            if cluster_id != content_hash[:16]:
                stats.clusters_joined += 1
            doc.cluster_id = cluster_id
            deps.session.get(SnapshotRow, content_hash).cluster_id = cluster_id
        deps.session.commit()
        return {"docs": docs, "stats": stats}

    def tag_entities(state: IngestState) -> dict:
        docs = dict(state.docs)
        for doc in docs.values():
            doc.entity_ids = deps.tagger.tag(doc.text)
        return {"docs": docs}

    def chunk_embed_upsert(state: IngestState) -> dict:
        ensure_collection(deps.client, deps.collection,
                          deps.embedder.dimension, deps.embedder.name)
        stats = state.stats.model_copy()
        for content_hash, doc in state.docs.items():
            source = deps.registry.get(doc.source_id)
            domain_tags = sorted(source.tags)
            chunks = splitter.split_text(doc.text) or [doc.text]
            dense_vectors = deps.embedder.embed_documents(chunks)
            points = []
            for n, (chunk_text, dense) in enumerate(zip(chunks, dense_vectors)):
                indices, values = deps.sparse.encode(chunk_text)
                points.append(chunk_point(
                    chunk_id=f"{content_hash}:{n}",
                    dense=dense, sparse_indices=indices, sparse_values=values,
                    payload={
                        "content_hash": content_hash,
                        "source_id": doc.source_id,
                        "domain_tags": domain_tags,
                        "title": doc.title,
                        "url": doc.url,
                        "published_at": doc.published_at,
                        "published_ts": doc.published_ts,
                        "cluster_id": doc.cluster_id,
                        "entity_ids": doc.entity_ids,
                        "text": chunk_text,
                    },
                ))
            deps.client.upsert(deps.collection, points)
            deps.session.get(SnapshotRow, content_hash).extraction_status = "done"
            stats.chunks_upserted += len(points)
        deps.session.commit()
        return {"stats": stats}

    def has_pending(state: IngestState) -> str:
        return "extract" if state.pending else END

    graph = StateGraph(IngestState)
    graph.add_node("select_pending", select_pending)
    graph.add_node("extract", extract)
    graph.add_node("cluster", cluster)
    graph.add_node("tag_entities", tag_entities)
    graph.add_node("chunk_embed_upsert", chunk_embed_upsert)
    graph.add_edge(START, "select_pending")
    graph.add_conditional_edges("select_pending", has_pending,
                                {"extract": "extract", END: END})
    graph.add_edge("extract", "cluster")
    graph.add_edge("cluster", "tag_entities")
    graph.add_edge("tag_entities", "chunk_embed_upsert")
    graph.add_edge("chunk_embed_upsert", END)
    return graph.compile()


def run_ingest(deps: IngestDeps, limit: int = 500) -> IngestStats:
    compiled = build_ingest_graph(deps)
    final = compiled.invoke(IngestState(limit=limit))
    return IngestStats.model_validate(final["stats"])
