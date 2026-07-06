"""Fetcher adapter registry (architecture SS5.3, SS7.1)."""
from __future__ import annotations

from argus.fetchers.base import (
    FetchContext,
    Fetcher,
    FetchResult,
    RawItem,
    UnknownAdapterError,
)
from argus.fetchers.http_api import HttpApiFetcher
from argus.fetchers.local_folder import LocalFolderFetcher
from argus.fetchers.rss import RssFetcher
from argus.fetchers.site_scraper import SiteScraperFetcher

ADAPTERS: dict[str, type[Fetcher]] = {
    RssFetcher.name: RssFetcher,
    LocalFolderFetcher.name: LocalFolderFetcher,
    HttpApiFetcher.name: HttpApiFetcher,
    SiteScraperFetcher.name: SiteScraperFetcher,
    # "sitemap": planned (SS5.3); registry entries referencing it fail validation loudly.
}


def get_fetcher(name: str, **kwargs) -> Fetcher:
    try:
        return ADAPTERS[name](**kwargs)
    except KeyError:
        raise UnknownAdapterError(
            f"unknown fetch adapter {name!r}; known: {sorted(ADAPTERS)}"
        ) from None


__all__ = [
    "ADAPTERS",
    "FetchContext",
    "Fetcher",
    "FetchResult",
    "RawItem",
    "UnknownAdapterError",
    "get_fetcher",
]
