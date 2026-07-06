"""Qdrant index management (architecture SS7.4).

One collection, named vectors: "dense" (Azure embeddings; cosine) and
"sparse" (BM25-style TF from argus.ingest.sparse; Qdrant applies IDF).
Payload carries the SS7.4 metadata; `published_ts` (epoch seconds) supports
window filters, `published_at` (ISO) is for display.

Embedder identity is stamped on the collection at creation; a later mismatch
raises — re-embedding is an explicit index-version bump (R5).
"""
from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from argus.settings import Settings

EMBEDDER_MARK_ID = uuid.uuid5(uuid.NAMESPACE_URL, "argus:embedder-mark")


def make_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    path = settings.data_dir / "qdrant"
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))  # embedded local mode (dev)


def _mark_point(dense_dim: int, embedder_name: str) -> models.PointStruct:
    return models.PointStruct(
        id=str(EMBEDDER_MARK_ID),
        vector={"dense": [0.0] * dense_dim},
        payload={"kind": "embedder_mark", "embedder": embedder_name,
                 "source_id": "__meta__", "published_ts": 0},
    )


def ensure_collection(
    client: QdrantClient, name: str, dense_dim: int, embedder_name: str
) -> None:
    if client.collection_exists(name):
        info = client.get_collection(name)
        size = info.config.params.vectors["dense"].size  # type: ignore[index]
        if size != dense_dim:
            raise RuntimeError(
                f"collection {name!r} has dense dim {size}, embedder produces "
                f"{dense_dim}. Re-embedding is an explicit index-version bump "
                f"(architecture R5): use a new collection name or delete this one."
            )
        # Dimension equality is NOT identity: two embedders can share a dim yet
        # produce incomparable spaces (R5). Enforce the stamp written at creation.
        marks = client.retrieve(name, ids=[str(EMBEDDER_MARK_ID)], with_payload=True)
        stamped = marks[0].payload.get("embedder") if marks and marks[0].payload else None
        if stamped is None:
            # Mark missing (creation interrupted before the stamp landed, or a
            # pre-stamp collection): adopt the current embedder so the guard
            # holds from here on.
            client.upsert(name, [_mark_point(dense_dim, embedder_name)])
        elif stamped != embedder_name:
            raise RuntimeError(
                f"collection {name!r} was built with embedder {stamped!r}; the "
                f"current embedder is {embedder_name!r}. Mixing embedding spaces "
                f"corrupts retrieval silently (architecture R5): re-embedding is "
                f"an explicit index-version bump — use a new collection name or "
                f"delete this one."
            )
        return

    try:
        sparse_params = models.SparseVectorParams(modifier=models.Modifier.IDF)
        client.create_collection(
            collection_name=name,
            vectors_config={"dense": models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": sparse_params},
        )
    except Exception:
        # older servers / local-mode gaps: fall back to sparse without IDF
        client.create_collection(
            collection_name=name,
            vectors_config={"dense": models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )

    client.create_payload_index(name, "source_id", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "published_ts", models.PayloadSchemaType.INTEGER)
    client.upsert(name, [_mark_point(dense_dim, embedder_name)])


def chunk_point(
    *, chunk_id: str, dense: list[float], sparse_indices: list[int],
    sparse_values: list[float], payload: dict[str, Any],
) -> models.PointStruct:
    return models.PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"argus:chunk:{chunk_id}")),
        vector={
            "dense": dense,
            "sparse": models.SparseVector(indices=sparse_indices, values=sparse_values),
        },
        payload={"chunk_id": chunk_id, **payload},
    )
