"""Content-addressed blob store (architecture SS7.2, AD-3).

Blobs live at <root>/<sha256[:2]>/<sha256>. Written once, never mutated:
a publisher edit produces a *new* hash and a *new* blob. Local filesystem in
M1; the path scheme maps 1:1 onto S3 keys later (SS10).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


class BlobStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()  # as_uri() requires absolute paths
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, content_hash: str) -> Path:
        return self.root / content_hash[:2] / content_hash

    def put(self, content: bytes) -> tuple[str, str, bool]:
        """Store bytes; return (sha256, uri, created).

        Idempotent: existing blobs are never rewritten (immutability).
        """
        content_hash = hashlib.sha256(content).hexdigest()
        path = self.path_for(content_hash)
        if path.exists():
            return content_hash, path.as_uri(), False
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(content)
        os.replace(tmp, path)  # atomic on POSIX
        return content_hash, path.as_uri(), True

    def get(self, content_hash: str) -> bytes:
        return self.path_for(content_hash).read_bytes()

    def exists(self, content_hash: str) -> bool:
        return self.path_for(content_hash).exists()
