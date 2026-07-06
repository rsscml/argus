"""Generic HTTP JSON API fetcher adapter (architecture SS5.3).

Maps a JSON endpoint into RawItems via FetchSpec.options, e.g.:

    options:
      items_path: "data.articles"        # dotted path to the item list
      fields:
        url: "link"
        title: "headline"
        published_at: "publishedAt"      # ISO-8601 expected
        content: "body"                  # optional; falls back to raw item JSON

Domain-specific structured feeds (exchange data, statistical releases) that
outgrow this mapper get their own bespoke adapter — engine untouched (SS5.3).
"""
from __future__ import annotations

import json
from datetime import datetime

import httpx

from argus.config.registry import Source
from argus.fetchers.base import FetchContext, Fetcher, FetchResult, RawItem
from argus.fetchers.robots import USER_AGENT
from argus.util import as_naive_utc, dot_get


def _parse_dt(value) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return as_naive_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


class HttpApiFetcher(Fetcher):
    name = "http_api"

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(
            timeout=20, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        )

    def fetch(self, source: Source, ctx: FetchContext) -> FetchResult:
        opts = source.fetch.options
        items_path = opts.get("items_path", "")
        fields: dict[str, str] = opts.get("fields", {})
        items: list[RawItem] = []

        for endpoint in source.fetch.endpoints:
            resp = self._client.get(endpoint)
            resp.raise_for_status()
            payload = resp.json()
            raw_items = dot_get(payload, items_path, payload) if items_path else payload
            if not isinstance(raw_items, list):
                raise ValueError(
                    f"{source.id}: items_path {items_path!r} did not yield a list"
                )
            for raw in raw_items:
                url = dot_get(raw, fields.get("url", "url"), endpoint)
                published = _parse_dt(dot_get(raw, fields.get("published_at", "")))
                if ctx.since is not None and published is not None and published <= ctx.since:
                    continue
                content_field = fields.get("content")
                content = (
                    str(dot_get(raw, content_field, "")) if content_field else ""
                ) or json.dumps(raw, sort_keys=True)
                items.append(
                    RawItem(
                        source_id=source.id,
                        url=str(url),
                        content=content.encode("utf-8"),
                        media_type="application/json" if not content_field else "text/plain",
                        title=dot_get(raw, fields.get("title", "title")),
                        published_at=published,
                        meta={"endpoint": endpoint, "kind": "api_item"},
                    )
                )
        return FetchResult(items=items)
