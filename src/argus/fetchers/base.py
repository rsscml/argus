"""Fetcher layer contracts (architecture SS7.1).

Adapters are deliberately dumb: fetch, don't judge. Trust decisions live in
the registry (SS6.1) and the corroboration engine (SS8.4), never here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from argus.config.registry import Source
from argus.util import utcnow


class RawItem(BaseModel):
    """One fetched item, frozen as-is before any processing (SS7.2)."""

    source_id: str
    url: str  # canonical URL, or synthetic URI (file://...) for local corpora
    content: bytes
    media_type: str = "text/html"
    title: str | None = None
    published_at: datetime | None = None  # declared by source; naive UTC
    fetched_at: datetime = Field(default_factory=utcnow)
    meta: dict[str, Any] = Field(default_factory=dict)


class FetchContext(BaseModel):
    """Per-source watermark state handed to adapters."""

    since: datetime | None = None  # last successful poll (naive UTC)
    etag: str | None = None
    last_modified: str | None = None


class FetchResult(BaseModel):
    items: list[RawItem] = Field(default_factory=list)
    not_modified: bool = False  # conditional GET said nothing changed
    etag: str | None = None
    last_modified: str | None = None


class Fetcher(ABC):
    """Adapter interface. Implementations register in argus.fetchers.ADAPTERS."""

    name: ClassVar[str]

    @abstractmethod
    def fetch(self, source: Source, ctx: FetchContext) -> FetchResult:
        """Pull items newer than ctx.since where the transport allows knowing.

        Must be side-effect free beyond network IO; must raise on transport
        failure (the poll loop records it and isolates the source, SS12.3).
        """


class UnknownAdapterError(KeyError):
    pass
