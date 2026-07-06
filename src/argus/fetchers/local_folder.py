"""Local-folder fetcher adapter.

The manually curated corpus is just another registry source (SS6.1, G4):
it flows through the same snapshot store and, in M2, the same index.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from argus.config.registry import Source
from argus.fetchers.base import FetchContext, Fetcher, FetchResult, RawItem

_MEDIA_TYPES = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".html": "text/html",
    ".htm": "text/html",
    ".pdf": "application/pdf",
    ".json": "application/json",
    ".csv": "text/csv",
}


class LocalFolderFetcher(Fetcher):
    name = "local_folder"

    def fetch(self, source: Source, ctx: FetchContext) -> FetchResult:
        items: list[RawItem] = []
        for endpoint in source.fetch.endpoints:
            root = Path(endpoint)
            if not root.exists():
                raise FileNotFoundError(f"local corpus root does not exist: {root}")
            for path in sorted(p for p in root.rglob("*") if p.is_file()):
                rel = path.relative_to(root)
                # Skip hidden files AND anything under hidden directories:
                # rglob descends into .git/, .cache/, editor state, etc., whose
                # contents must never become Tier-1 "evidence".
                if any(part.startswith(".") for part in rel.parts):
                    continue
                mtime = datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).replace(tzinfo=None)
                if ctx.since is not None and mtime <= ctx.since:
                    continue  # unchanged since last poll
                items.append(
                    RawItem(
                        source_id=source.id,
                        url=path.resolve().as_uri(),
                        content=path.read_bytes(),
                        media_type=_MEDIA_TYPES.get(path.suffix.lower(),
                                                    "application/octet-stream"),
                        title=path.stem,
                        published_at=mtime,
                        meta={"relpath": str(path.relative_to(root)), "root": str(root)},
                    )
                )
        return FetchResult(items=items)
