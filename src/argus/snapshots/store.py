"""Snapshot store facade (architecture SS7.2).

put() freezes a RawItem: blob first, then the DB row. Rows are immutable —
a second put of identical bytes is a dedup hit and returns the original row
untouched, even if it arrived from a different source (byte-identical
syndication; semantic clustering is M2's job, SS7.3, and the duplicate
sighting is preserved in the fetch-event counters).
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from argus.fetchers.base import RawItem
from argus.snapshots.blob import BlobStore
from argus.snapshots.db import SnapshotRow


@dataclass
class PutResult:
    content_hash: str
    created: bool
    row: SnapshotRow


class SnapshotStore:
    def __init__(self, blob: BlobStore, session: Session) -> None:
        self.blob = blob
        self.session = session

    def put(self, item: RawItem) -> PutResult:
        content_hash, blob_uri, _ = self.blob.put(item.content)
        existing = self.session.get(SnapshotRow, content_hash)
        if existing is not None:
            return PutResult(content_hash, False, existing)
        row = SnapshotRow(
            content_hash=content_hash,
            source_id=item.source_id,
            url=item.url,
            title=item.title,
            media_type=item.media_type,
            published_at=item.published_at,
            fetched_at=item.fetched_at,
            blob_uri=blob_uri,
            meta=item.meta,
        )
        self.session.add(row)
        return PutResult(content_hash, True, row)

    def get_bytes(self, content_hash: str) -> bytes:
        return self.blob.get(content_hash)
