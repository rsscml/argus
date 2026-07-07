"""Sitemap-driven site scraper (architecture §5.3 plugin point, §7.1, §12.4).

Register a broad site as a controlled source by giving ONLY its domain URL. The
adapter reads the site's robots.txt to discover the sitemaps it declares, walks
those sitemap trees RECURSIVELY (index -> index -> ... -> urlset) down to article
URLs, and fetches up to `max_pages` of them — no section seeds, no link crawling,
no hand-listed URLs.

DISCOVERY, in order:
  1. robots.txt `Sitemap:` lines (via RobotsGate.site_maps — one fetch also gives
     the Disallow rules honored on every article fetch);
  2. any `options.sitemaps` you list — the fallback when robots declares none;
  3. well-known paths (/sitemap.xml, …) only if 1 and 2 are both empty.
If none of these yield a sitemap, fetch() raises so the poll loop records the
source as failing (SS12.3) rather than silently producing nothing.

MINIMAL registry entry — just the domain in `endpoints`:
    fetch:
      adapter: site_scraper
      endpoints: ["https://www.example.com/"]
      options: {}

OPTIONAL scoping (see full reference at the bottom):
  - relevance.dictionary / keywords : keep only articles that mention your
    domain's entities. STRONGLY recommended for a broad portal — without it every
    article the site publishes enters the domain and floods the daily brief, since
    window_scan ingests everything from a tagged source. Reuses the domain's own
    entity CSV, so 'relevant' means what it means at ingestion-time tagging (SS5.2).
  - url_include / url_exclude : cheap regex scope applied WHILE walking, so the
    page budget fills with in-scope URLs instead of junk.
Ads/boilerplate are stripped downstream by the ingestion extractor (SS7.3); the
snapshot stays raw HTML (SS7.2). Incremental polls skip sitemap URLs whose
<lastmod> predates the watermark, so only new/changed articles are fetched.

COMPLIANCE (SS12.4): robots.txt honored, rate-limited (`crawl_delay_seconds`),
bounded (`max_pages`, `max_sitemap_docs`, `max_sitemap_depth`), same-host only, no
paywall/login bypass. Presents as a normal browser (BROWSER_HEADERS) so a crude
edge User-Agent filter can't block reading robots.txt or public article pages —
that is not a challenge bypass; a site behind a real JS/Cloudflare challenge
should be ingested via its RSS/API (the rss / http_api adapters), not scraped.
"""
from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlsplit
from xml.etree import ElementTree

import httpx
from pydantic import BaseModel, Field

from argus.config.registry import Source
from argus.fetchers.base import FetchContext, Fetcher, FetchResult, RawItem
from argus.fetchers.robots import BROWSER_HEADERS, RobotsGate
from argus.util import as_naive_utc

# Tried only when robots.txt / options.sitemaps yield nothing.
_WELL_KNOWN = ["/sitemap.xml", "/sitemap_index.xml", "/sitemap-index.xml", "/sitemap/sitemap.xml"]

_OG_TITLE = re.compile(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)', re.I)
_TITLE = re.compile(r"<title[^>]*>([^<]+)</title>", re.I)
_DATE_PATTERNS = [
    re.compile(r'"datePublished"\s*:\s*"([^"]+)"'),
    re.compile(r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)', re.I),
    re.compile(r'<time[^>]+datetime=["\']([^"\']+)', re.I),
]


# --------------------------------------------------------------------------
# options
# --------------------------------------------------------------------------

class _Relevance(BaseModel):
    dictionary: str | None = None          # CSV path (repo-root relative or absolute)
    keywords: list[str] = Field(default_factory=list)
    min_matches: int = Field(default=1, ge=1)


class _ScraperOptions(BaseModel):
    sitemaps: list[str] = Field(default_factory=list)      # fallback if robots declares none
    max_pages: int = Field(default=50, ge=1)               # article-page FETCH budget
    max_sitemap_docs: int = Field(default=50, ge=1)        # sitemap documents fetched
    max_sitemap_depth: int = Field(default=8, ge=1)        # nested-index depth cap
    crawl_delay_seconds: float = Field(default=1.0, ge=0)
    url_include: list[str] = Field(default_factory=list)   # regex; keep matching page URLs
    url_exclude: list[str] = Field(default_factory=list)   # regex; drop matching page URLs
    min_text_chars: int = Field(default=400, ge=0)         # skip thin/teaser pages
    respect_lastmod: bool = True                           # skip URLs older than the watermark
    relevance: _Relevance = Field(default_factory=_Relevance)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _skey(url: str) -> str:
    return urldefrag(url)[0].rstrip("/")


def _normalize_site(s: str) -> str:
    if not re.match(r"^https?://", s, re.I):
        s = "https://" + s
    parts = urlsplit(s)
    return f"{parts.scheme}://{parts.netloc}/"


def _parse_dt(s: str) -> datetime | None:
    try:
        return as_naive_utc(datetime.fromisoformat(s.replace("Z", "+00:00")))
    except (ValueError, AttributeError):
        return None


def _extract_title(html: str) -> str | None:
    for pat in (_OG_TITLE, _TITLE):
        m = pat.search(html)
        if m:
            return m.group(1).strip() or None
    return None


def _extract_published(html: str) -> datetime | None:
    for pat in _DATE_PATTERNS:
        m = pat.search(html)
        if m:
            dt = _parse_dt(m.group(1))
            if dt is not None:
                return dt
    return None


def _looks_like_sitemap(resp: httpx.Response | None) -> bool:
    if resp is None:
        return False
    head = resp.text[:1000].lower()
    return "<sitemapindex" in head or "<urlset" in head


def _build_relevance(rel: _Relevance):
    """EntityTagger over dictionary + inline keywords, or None (no topic filter).
    Reuses the engine's own dictionary loader/matcher (SS5.2)."""
    if not rel.dictionary and not rel.keywords:
        return None
    from argus.ingest.entities import Entity, EntityTagger, load_dictionary

    entities: list[Entity] = []
    if rel.dictionary:
        path = Path(rel.dictionary)
        if not path.exists():
            raise FileNotFoundError(
                f"relevance dictionary not found: {path} (paths resolve relative to "
                f"the working directory — run from the repo root, as the CLI does)")
        entities.extend(load_dictionary(path))
    entities.extend(Entity(surface=k, canonical_id=k) for k in rel.keywords)
    return EntityTagger(entities) if entities else None


def _fetch_xml(get, url: str):
    resp = get(url)
    if resp is None:
        return None
    data = resp.content
    if url.endswith(".gz") or data[:2] == b"\x1f\x8b":     # gzipped sitemap
        import gzip
        try:
            data = gzip.decompress(data)
        except Exception:  # noqa: BLE001
            pass
    try:
        return ElementTree.fromstring(data)
    except ElementTree.ParseError:
        return None


def _walk_sitemaps(get, roots: list[str], cap: int, inc: list, exc: list,
                   since: datetime | None, *, max_docs: int, max_depth: int) -> list[dict]:
    """Walk one or more sitemap trees of ANY depth to page URLs. All roots share
    one budget and one visited-set. Bounded by cap (page URLs), max_docs (sitemap
    documents fetched), max_depth, and the visited-set (cycles/diamonds). Filters
    page URLs by inc/exc while collecting, and drops URLs whose <lastmod> is at or
    before `since` (unchanged since the last poll)."""
    from collections import deque

    def keep(u: str) -> bool:
        if inc and not any(p.search(u) for p in inc):
            return False
        return not any(p.search(u) for p in exc)

    pages: list[dict] = []
    seen: set[str] = set()
    queue: deque[tuple[str, int]] = deque((u, 0) for u in roots)
    docs = 0
    while queue and len(pages) < cap and docs < max_docs:
        sm_url, depth = queue.popleft()
        if _skey(sm_url) in seen:
            continue
        seen.add(_skey(sm_url))
        root = _fetch_xml(get, sm_url)
        docs += 1
        if root is None:
            continue
        if _local(root.tag) == "sitemapindex":
            if depth < max_depth:
                for loc in root.findall(".//{*}loc"):
                    if loc.text and _skey(loc.text.strip()) not in seen:
                        queue.append((loc.text.strip(), depth + 1))
            continue
        for url_el in root.iter():                         # a <urlset>
            if _local(url_el.tag) != "url":
                continue
            loc_el = url_el.find("{*}loc")                 # direct child (not image:loc)
            if loc_el is None or not loc_el.text:
                continue
            loc = loc_el.text.strip()
            if not keep(loc):
                continue
            lastmod_el = url_el.find("{*}lastmod")
            lastmod = (lastmod_el.text or "").strip() if lastmod_el is not None else ""
            if since is not None and lastmod:
                dt = _parse_dt(lastmod)
                if dt is not None and dt <= since:
                    continue                               # unchanged since last poll
            title_el = url_el.find(".//{*}title")          # news:title if present
            pages.append({
                "url": loc, "lastmod": lastmod,
                "title": (title_el.text or "").strip() if title_el is not None else "",
            })
            if len(pages) >= cap:
                break
    return pages[:cap]


# --------------------------------------------------------------------------
# adapter
# --------------------------------------------------------------------------

class SiteScraperFetcher(Fetcher):
    name = "site_scraper"

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(
            timeout=25, follow_redirects=True, headers=BROWSER_HEADERS)
        self._robots = RobotsGate(self._client)
        self._last_get = 0.0
        self._delay = 0.0

    def _get(self, url: str) -> httpx.Response | None:
        """Throttled GET (used for sitemaps AND articles). Swallows per-URL errors —
        one bad URL never aborts the source; total failure surfaces as an empty walk
        or the no-sitemaps raise."""
        wait = self._delay - (time.monotonic() - self._last_get)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = self._client.get(url)
            self._last_get = time.monotonic()
            resp.raise_for_status()
            return resp
        except Exception:  # noqa: BLE001
            self._last_get = time.monotonic()
            return None

    def _discover_sitemaps(self, site_roots: list[str], explicit: list[str]) -> list[str]:
        declared: list[str] = []
        for root in site_roots:
            declared += self._robots.site_maps(root)
        sitemaps = list(dict.fromkeys(explicit + declared))    # explicit first, deduped
        if not sitemaps:                                       # well-known fallback
            for root in site_roots:
                for path in _WELL_KNOWN:
                    cand = urljoin(root, path)
                    if _looks_like_sitemap(self._get(cand)):
                        sitemaps.append(cand)
        return sitemaps

    def fetch(self, source: Source, ctx: FetchContext) -> FetchResult:
        import trafilatura  # lazy (heavy): off the adapter-registry import path

        opts = _ScraperOptions.model_validate(source.fetch.options)
        self._delay = opts.crawl_delay_seconds
        try:
            inc = [re.compile(p, re.I) for p in opts.url_include]
            exc = [re.compile(p, re.I) for p in opts.url_exclude]
        except re.error as exc_err:
            raise ValueError(f"{source.id}: bad url_include/url_exclude regex: {exc_err}")

        tagger = _build_relevance(opts.relevance)
        site_roots = [_normalize_site(e) for e in source.fetch.endpoints]
        allowed_hosts = {urlsplit(r).netloc.lower().removeprefix("www.") for r in site_roots}

        sitemaps = self._discover_sitemaps(site_roots, opts.sitemaps)
        if not sitemaps:
            raise ValueError(
                f"{source.id}: no sitemaps found in robots.txt, options.sitemaps, or "
                f"well-known paths for {sorted(allowed_hosts)}. List one under "
                f"options.sitemaps, or use the rss/http_api adapter for this source.")

        since = ctx.since if opts.respect_lastmod else None
        candidates = _walk_sitemaps(self._get, sitemaps, opts.max_pages, inc, exc, since,
                                    max_docs=opts.max_sitemap_docs, max_depth=opts.max_sitemap_depth)

        items: list[RawItem] = []
        fetches = 0
        for cand in candidates:
            if fetches >= opts.max_pages:
                break
            url = cand["url"]
            if urlsplit(url).netloc.lower().removeprefix("www.") not in allowed_hosts:
                continue                                   # sitemap listed an off-site URL
            if not self._robots.allowed(url):
                continue                                   # robots.txt disallows

            resp = self._get(url)
            fetches += 1
            if resp is None:
                continue
            content_type = resp.headers.get("Content-Type", "text/html").split(";")[0].strip()
            if content_type and "html" not in content_type and content_type != "text/plain":
                continue                                   # skip binaries / non-articles

            html = resp.text
            body = trafilatura.extract(html, include_comments=False) or ""
            if len(body.strip()) < opts.min_text_chars:
                continue                                   # teaser/listing/thin page

            matches: list[str] = []
            if tagger is not None:
                matches = tagger.tag(f"{_extract_title(html) or ''}\n{body}")
                if len(matches) < opts.relevance.min_matches:
                    continue                               # off-topic for this domain

            published = _parse_dt(cand.get("lastmod", "")) or _extract_published(html)
            items.append(RawItem(
                source_id=source.id,
                url=urldefrag(str(resp.url))[0],
                content=resp.content,                      # raw HTML bytes (SS7.2)
                media_type=content_type or "text/html",
                title=_extract_title(html) or (cand.get("title") or None),
                published_at=published,
                meta={
                    "kind": "scraped_article",
                    "adapter": self.name,
                    "relevance_matches": matches,
                    "sitemap_lastmod": cand.get("lastmod", ""),
                    "http_status": resp.status_code,
                },
            ))

        # No conditional GET: a multi-URL crawl has no single validator for the
        # per-source watermark. Incremental freshness comes from <lastmod> vs
        # ctx.since above, backed by the snapshot store's content-hash dedup.
        return FetchResult(items=items)


# =============================================================================
# OPTION REFERENCE (place under a registry entry's `fetch.options`; all optional)
# -----------------------------------------------------------------------------
# The registry entry's `endpoints` holds the domain URL(s) — that is the only
# required input. A source scraping one topical site needs no options at all.
#
# sitemaps: []                  # explicit sitemap URL(s); used when robots.txt
#                               #   declares none (also the place to point at a
#                               #   site's dedicated news-sitemap for freshness)
# max_pages: 50                 # article-page FETCH budget per poll (relevance
#                               #   filtering may yield fewer items than this)
# max_sitemap_docs: 50          # cap on sitemap DOCUMENTS fetched while walking
# max_sitemap_depth: 8          # nested sitemap-index depth cap
# crawl_delay_seconds: 1.0      # minimum gap between requests
# url_include: []               # regex list; keep only matching page URLs
# url_exclude: []               # regex list; drop matching page URLs
# min_text_chars: 400           # skip pages with less extracted text than this
# respect_lastmod: true         # skip sitemap URLs whose <lastmod> <= watermark
# relevance:                    # topic scoping (recommended for broad portals)
#   dictionary: "domains/<d>/entities/<d>.csv"   # usually your domain's own CSV
#   keywords: ["..."]                            # extends the dictionary
#   min_matches: 1              # >= N distinct entity hits in title+body to keep
# =============================================================================
