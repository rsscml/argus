"""RSS/Atom fetcher adapter (architecture SS7.1, AD-1).

Snapshot unit is the feed *entry* (headline + summary), matching the typical
RSS license posture in the registry (SS6.1). If — and only if — the source's
FetchSpec sets fetch_full_text (i.e. its license permits), the linked page is
fetched instead, gated by robots.txt (SS12.4).
"""
from __future__ import annotations

import time
from datetime import datetime

import feedparser
import httpx

from argus.config.registry import Source
from argus.fetchers.base import FetchContext, Fetcher, FetchResult, RawItem
from argus.fetchers.robots import USER_AGENT, RobotsGate


def _struct_to_dt(t: time.struct_time | None) -> datetime | None:
    if t is None:
        return None
    return datetime(*t[:6])  # feedparser struct_times are UTC -> naive UTC


class RssFetcher(Fetcher):
    name = "rss"

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(
            timeout=20, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        )
        self._robots = RobotsGate(self._client)

    def fetch(self, source: Source, ctx: FetchContext) -> FetchResult:
        # Conditional GET headers only make sense with a single endpoint,
        # because the watermark stores one etag/last-modified per source.
        conditional = len(source.fetch.endpoints) == 1
        items: list[RawItem] = []
        new_etag = ctx.etag
        new_lm = ctx.last_modified
        any_content = False

        for endpoint in source.fetch.endpoints:
            headers: dict[str, str] = {}
            if conditional and ctx.etag:
                headers["If-None-Match"] = ctx.etag
            if conditional and ctx.last_modified:
                headers["If-Modified-Since"] = ctx.last_modified

            resp = self._client.get(endpoint, headers=headers)
            if resp.status_code == 304:
                continue
            resp.raise_for_status()
            any_content = True
            if conditional:
                new_etag = resp.headers.get("ETag", new_etag)
                new_lm = resp.headers.get("Last-Modified", new_lm)

            feed = feedparser.parse(resp.content)
            for entry in feed.entries:
                item = self._entry_to_item(source, endpoint, entry, ctx.since)
                if item is not None:
                    items.append(item)

        return FetchResult(
            items=items,
            not_modified=(not any_content and not items),
            etag=new_etag,
            last_modified=new_lm,
        )

    def _entry_to_item(
        self, source: Source, endpoint: str, entry, since: datetime | None
    ) -> RawItem | None:
        published = _struct_to_dt(
            getattr(entry, "published_parsed", None)
            or getattr(entry, "updated_parsed", None)
        )
        # Watermark filter: skip clearly-old entries; keep undated ones —
        # content-hash dedup in the snapshot store is the real guard (SS7.2).
        if since is not None and published is not None and published <= since:
            return None

        link = getattr(entry, "link", None) or endpoint
        title = getattr(entry, "title", None)

        if source.fetch.fetch_full_text and self._robots.allowed(link):
            try:
                page = self._client.get(link)
                page.raise_for_status()
                return RawItem(
                    source_id=source.id,
                    url=link,
                    content=page.content,
                    media_type=page.headers.get("Content-Type", "text/html").split(";")[0],
                    title=title,
                    published_at=published,
                    meta={"feed": endpoint, "kind": "full_page"},
                )
            except Exception as exc:  # fall back to the entry itself
                fallback_meta = {"feed": endpoint, "kind": "rss_entry",
                                 "full_text_error": str(exc)}
                return self._summary_item(source, entry, link, title, published, fallback_meta)

        return self._summary_item(
            source, entry, link, title, published, {"feed": endpoint, "kind": "rss_entry"}
        )

    @staticmethod
    def _summary_item(source, entry, link, title, published, meta) -> RawItem:
        if getattr(entry, "content", None):
            body = entry.content[0].get("value", "")
        else:
            body = getattr(entry, "summary", "") or (title or "")
        return RawItem(
            source_id=source.id,
            url=link,
            content=body.encode("utf-8"),
            media_type="text/html",
            title=title,
            published_at=published,
            meta=meta,
        )
