"""Public-site scraper fetcher adapter (architecture §5.3 plugin point, §7.1, §12.4).

This is NOT part of the base engine — it is a bespoke `Fetcher` you register by
name, exactly as §5.3 anticipates ("some domains need a custom adapter … engine
untouched"). It lets a broad news *portal* (one that has no usable RSS, or mixes
many subjects) act as a controlled source that only yields pages relevant to a
given domain.

TOPIC SCOPING — two layers, coarse then fine:

  1. URL scope (cheap, polite): only same-host links under your seed/section
     pages are considered, filtered by `url_include` / `url_exclude` regexes.
     Point the seeds at the site's topical sections (e.g. /markets/, /stocks/)
     so the fetch budget is spent where the relevant articles live.

  2. Relevance vocabulary (accurate): an article is kept only if its extracted
     text mentions at least `min_matches` distinct entries from a relevance
     vocabulary — normally the SAME entity dictionary your domain profile already
     uses for tagging and query expansion (§5.1/§5.2). For a "Nifty50" domain,
     that dictionary is the 50 constituents + tickers + aliases, so only articles
     actually about those companies survive. Inline `keywords` can extend it.

ADS / BOILERPLATE: the snapshot is stored as raw HTML by design (§7.2, immutable
raw bytes → reproducible citations). Ad and navigation stripping happens in the
ingestion extractor (trafilatura, §7.3), the same path every other source uses —
so ads never reach the index. The scraper additionally emits *article* pages
only (never section/listing pages full of teasers), and uses the extracted text
(not the raw page) for the relevance decision, so what it filters on matches what
gets indexed.

COMPLIANCE (§12.4) — non-negotiable, matching the rest of the system:
  * robots.txt is honored for EVERY fetch (there is no off-switch here);
  * requests are rate-limited (`crawl_delay_seconds`) and bounded
    (`max_listing_pages`, `max_articles`);
  * crawling never leaves the registered host and never recurses into a spider;
  * it does NOT bypass paywalls, logins, or any access control.
  Respecting each site's Terms of Service is the operator's responsibility; say
  so honestly in the registry `license` field before you activate the source.

CONFIG lives in the registry entry's `fetch.options` (adapter-specific, like the
http_api adapter). Everything has a sensible default; a minimal config is just
`endpoints` + a `relevance.dictionary`. Full option reference at the bottom.
"""
from __future__ import annotations

import re
import time
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlsplit
from xml.etree import ElementTree

import httpx
from pydantic import BaseModel, Field

from argus.config.registry import Source
from argus.fetchers.base import FetchContext, Fetcher, FetchResult, RawItem
from argus.fetchers.robots import USER_AGENT, RobotsGate
from argus.util import as_naive_utc

# Links we never treat as articles regardless of user config (assets, actions).
_HARD_SKIP = re.compile(
    r"^(?:mailto:|tel:|javascript:|#)|\.(?:jpg|jpeg|png|gif|webp|svg|ico|css|js|"
    r"mp4|mp3|zip|gz|rss|xml)(?:\?|$)",
    re.IGNORECASE,
)

# Best-effort published-date signals, in priority order.
_DATE_PATTERNS = [
    re.compile(r'"datePublished"\s*:\s*"([^"]+)"'),                       # JSON-LD
    re.compile(r'<meta[^>]+property=["\']article:published_time["\']'
               r'[^>]+content=["\']([^"\']+)', re.IGNORECASE),
    re.compile(r'<time[^>]+datetime=["\']([^"\']+)', re.IGNORECASE),
]
_OG_TITLE = re.compile(r'<meta[^>]+property=["\']og:title["\']'
                       r'[^>]+content=["\']([^"\']+)', re.IGNORECASE)
_TITLE = re.compile(r"<title[^>]*>([^<]+)</title>", re.IGNORECASE)


# --------------------------------------------------------------------------
# options
# --------------------------------------------------------------------------

class _Relevance(BaseModel):
    dictionary: str | None = None          # CSV path (CWD/repo-root relative or absolute)
    keywords: list[str] = Field(default_factory=list)
    min_matches: int = Field(default=1, ge=1)
    prefilter_url: bool = False            # cheap pre-gate on the de-slugified URL


class _ScraperOptions(BaseModel):
    follow_links: bool = True              # harvest article links from the seed pages
    sitemap: str | None = None             # optional sitemap(-index) URL to enumerate
    sitemap_max_urls: int = Field(default=2000, ge=1)
    max_listing_pages: int = Field(default=8, ge=1)
    max_articles: int = Field(default=40, ge=1)   # caps article-page FETCHES per poll
    crawl_delay_seconds: float = Field(default=1.0, ge=0)
    same_host_only: bool = True
    url_include: list[str] = Field(default_factory=list)   # keep if ANY matches (when set)
    url_exclude: list[str] = Field(default_factory=list)   # drop if ANY matches
    min_text_chars: int = Field(default=500, ge=0)
    relevance: _Relevance = Field(default_factory=_Relevance)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

class _LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            for key, value in attrs:
                if key == "href" and value:
                    self.hrefs.append(value)


def _hosts(endpoints: list[str]) -> set[str]:
    hosts: set[str] = set()
    for endpoint in endpoints:
        netloc = urlsplit(endpoint).netloc.lower()
        if netloc:
            hosts.add(netloc)
            hosts.add(netloc.removeprefix("www."))
    return hosts


def _build_relevance(rel: _Relevance):
    """An EntityTagger over dictionary + inline keywords, or None (URL-only scope).

    Reuses the engine's own dictionary loader and matcher so 'relevant' means
    exactly what it means at ingestion-time entity tagging (§5.2)."""
    from argus.ingest.entities import Entity, EntityTagger, load_dictionary

    entities: list[Entity] = []
    if rel.dictionary:
        path = Path(rel.dictionary)
        if not path.exists():
            raise FileNotFoundError(
                f"relevance dictionary not found: {path} (paths resolve relative "
                f"to the working directory — run from the repo root, as the CLI does)"
            )
        entities.extend(load_dictionary(path))
    entities.extend(Entity(surface=k, canonical_id=k) for k in rel.keywords)
    return EntityTagger(entities) if entities else None


def _extract_published(html: str) -> datetime | None:
    for pattern in _DATE_PATTERNS:
        m = pattern.search(html)
        if not m:
            continue
        try:
            return as_naive_utc(datetime.fromisoformat(m.group(1).replace("Z", "+00:00")))
        except ValueError:
            continue
    return None


def _extract_title(html: str) -> str | None:
    for pattern in (_OG_TITLE, _TITLE):
        m = pattern.search(html)
        if m:
            return m.group(1).strip() or None
    return None


# --------------------------------------------------------------------------
# adapter
# --------------------------------------------------------------------------

class SiteScraperFetcher(Fetcher):
    name = "site_scraper"

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(
            timeout=25, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        )
        self._robots = RobotsGate(self._client)
        self._last_get = 0.0
        self._delay = 0.0

    # -- polite, robots-gated GET ------------------------------------------
    def _get(self, url: str) -> httpx.Response | None:
        if not self._robots.allowed(url):
            return None  # robots.txt disallows — skip silently (§12.4)
        wait = self._delay - (time.monotonic() - self._last_get)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = self._client.get(url)
            self._last_get = time.monotonic()
            resp.raise_for_status()
            return resp
        except Exception:
            self._last_get = time.monotonic()
            return None  # one bad URL never aborts the source (poll loop isolates sources)

    # -- URL candidate discovery -------------------------------------------
    def _candidates(self, source: Source, opts: _ScraperOptions,
                    includes: list[re.Pattern], excludes: list[re.Pattern]) -> list[str]:
        allowed_hosts = _hosts(source.fetch.endpoints)
        seen: set[str] = set()
        ordered: list[str] = []

        def consider(raw_url: str, base: str) -> None:
            url = urldefrag(urljoin(base, raw_url.strip()))[0]
            if not url or url in seen:
                return
            if _HARD_SKIP.search(url):
                return
            if opts.same_host_only and urlsplit(url).netloc.lower() not in allowed_hosts:
                return
            if includes and not any(p.search(url) for p in includes):
                return
            if any(p.search(url) for p in excludes):
                return
            seen.add(url)
            ordered.append(url)

        if opts.sitemap:
            for url in self._sitemap_urls(opts.sitemap, opts.sitemap_max_urls):
                consider(url, opts.sitemap)

        if opts.follow_links:
            for endpoint in source.fetch.endpoints[: opts.max_listing_pages]:
                resp = self._get(endpoint)
                if resp is None:
                    continue
                extractor = _LinkExtractor()
                extractor.feed(resp.text)
                for href in extractor.hrefs:
                    consider(href, str(resp.url))

        return ordered

    def _sitemap_urls(self, sitemap_url: str, cap: int) -> list[str]:
        resp = self._get(sitemap_url)
        if resp is None:
            return []
        try:
            root = ElementTree.fromstring(resp.content)
        except ElementTree.ParseError:
            return []
        # `.//{*}loc` honors the sitemap namespace via ElementPath's wildcard;
        # `Element.iter("{*}loc")` would NOT (iter does exact tag-string matching).
        locs = [el.text.strip() for el in root.findall(".//{*}loc") if el.text]
        if root.tag.rsplit("}", 1)[-1].lower() == "sitemapindex":
            urls: list[str] = []
            for child in locs[:20]:                       # bounded index expansion
                child_resp = self._get(child)
                if child_resp is None:
                    continue
                try:
                    child_root = ElementTree.fromstring(child_resp.content)
                except ElementTree.ParseError:
                    continue
                urls += [el.text.strip() for el in child_root.findall(".//{*}loc") if el.text]
                if len(urls) >= cap:
                    break
            return urls[:cap]
        return locs[:cap]

    # -- relevance ----------------------------------------------------------
    @staticmethod
    def _slug_text(url: str) -> str:
        return re.sub(r"[-_/]+", " ", urlsplit(url).path)

    # -- main ---------------------------------------------------------------
    def fetch(self, source: Source, ctx: FetchContext) -> FetchResult:
        import trafilatura  # lazy (heavy): kept off the adapter-registry import path

        opts = _ScraperOptions.model_validate(source.fetch.options)
        self._delay = opts.crawl_delay_seconds
        try:
            includes = [re.compile(p, re.IGNORECASE) for p in opts.url_include]
            excludes = [re.compile(p, re.IGNORECASE) for p in opts.url_exclude]
        except re.error as exc:
            raise ValueError(f"{source.id}: bad url_include/url_exclude regex: {exc}")

        tagger = _build_relevance(opts.relevance)
        min_matches = opts.relevance.min_matches

        candidates = self._candidates(source, opts, includes, excludes)

        items: list[RawItem] = []
        fetches = 0
        for url in candidates:
            if fetches >= opts.max_articles:
                break

            # Cheap pre-gate: skip fetching pages whose slug shows no vocabulary
            # hit (opt-in; a slug can legitimately omit the entity, so default off).
            if tagger is not None and opts.relevance.prefilter_url:
                if len(tagger.tag(self._slug_text(url))) < 1:
                    continue

            resp = self._get(url)
            fetches += 1
            if resp is None:
                continue

            html = resp.text
            # Extract the SAME way ingestion will, so the relevance check matches
            # what gets indexed (§7.3). Ads/nav are stripped here for the decision.
            body = trafilatura.extract(html, include_comments=False) or ""
            if len(body.strip()) < opts.min_text_chars:
                continue  # teaser/listing/thin page — not a real article

            matches: list[str] = []
            if tagger is not None:
                title_text = _extract_title(html) or ""
                matches = tagger.tag(f"{title_text}\n{body}")
                if len(matches) < min_matches:
                    continue  # off-topic for this domain — drop

            published = _extract_published(html)
            if ctx.since is not None and published is not None and published <= ctx.since:
                continue  # older than the watermark (content-hash dedup is the real guard)

            content_type = resp.headers.get("Content-Type", "text/html").split(";")[0]
            items.append(RawItem(
                source_id=source.id,
                url=urldefrag(str(resp.url))[0],
                content=resp.content,                     # raw HTML bytes (§7.2)
                media_type=content_type or "text/html",
                title=_extract_title(html),
                published_at=published,
                meta={
                    "kind": "scraped_article",
                    "adapter": self.name,
                    "relevance_matches": matches,          # audit which entities matched
                    "http_status": resp.status_code,
                },
            ))

        # No conditional-GET here: a multi-URL crawl has no single validator to
        # store in the per-source watermark, so we rely on `since` + the snapshot
        # store's content-hash dedup (unchanged articles re-hash identically and
        # are recorded as duplicates, never re-indexed).
        return FetchResult(items=items)


# =============================================================================
# OPTION REFERENCE (place under a registry entry's `fetch.options`)
# -----------------------------------------------------------------------------
# follow_links: true            # harvest <a href> article links from the seeds
# sitemap: null                 # optional sitemap or sitemap-index URL to enumerate
# sitemap_max_urls: 2000
# max_listing_pages: 8          # how many seed/section pages to scan for links
# max_articles: 40              # hard cap on article-page fetches per poll (politeness)
# crawl_delay_seconds: 1.0      # minimum gap between requests
# same_host_only: true          # never leave the registered host
# url_include: []               # regex list; a candidate must match >=1 (when non-empty)
# url_exclude: []               # regex list; drop a candidate matching any
# min_text_chars: 500           # skip thin/teaser pages after extraction
# relevance:
#   dictionary: "domains/nifty50/entities/nifty50.csv"   # normally your domain's own CSV
#   keywords: ["Nifty 50", "Nifty50", "NSE"]             # extends the dictionary
#   min_matches: 1              # >= N distinct vocabulary hits in title+body to keep
#   prefilter_url: false        # also require a vocabulary hit in the URL slug (fewer fetches)
# =============================================================================
