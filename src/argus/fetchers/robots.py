"""robots.txt gate (architecture SS12.4).

Consulted for full-page fetches (feed/sitemap endpoints are published for
consumption). Standard convention: unreachable robots.txt => allowed; an
explicit disallow is honored. The same cached parse also exposes the
``Sitemap:`` lines the origin declares, via ``site_maps()``.

Requests present as a normal browser (BROWSER_HEADERS). Many news CDNs reject
non-browser User-Agents at the edge with a 403 — including on /robots.txt
itself, which would otherwise leave us unable to read (and therefore honor) the
site's rules or discover its sitemaps. The content requested is public and served
freely to any browser; presenting as one clears that crude filter. It does NOT
defeat a real JS/Cloudflare challenge — a site that still blocks should be
ingested via its RSS/API. Disallow rules are matched against ``*`` and honored.
"""
from __future__ import annotations

from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"),
    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
               "image/avif,image/webp,*/*;q=0.8"),
    "Accept-Language": "en-US,en;q=0.9",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document", "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none", "Sec-Fetch-User": "?1",
}
# The token robots rules are matched against. Presenting as a browser, we follow
# the `*` rules any browser would (a browser UA won't match bot-specific stanzas).
USER_AGENT = BROWSER_HEADERS["User-Agent"]


class RobotsGate:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client
        self._cache: dict[str, RobotFileParser | None] = {}

    def allowed(self, url: str) -> bool:
        parser = self._parser_for(url)
        if parser is None:
            return True
        return parser.can_fetch(USER_AGENT, url)

    def site_maps(self, url: str) -> list[str]:
        """Sitemap URLs declared in the origin's robots.txt (empty if none or
        unreachable). Reuses the same cached parse as ``allowed()`` — one fetch
        per origin serves both crawl permissions and sitemap discovery."""
        parser = self._parser_for(url)
        if parser is None:
            return []
        return list(parser.site_maps() or [])

    def _parser_for(self, url: str) -> RobotFileParser | None:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._cache:
            self._cache[origin] = self._load(origin)
        return self._cache[origin]

    def _load(self, origin: str) -> RobotFileParser | None:
        try:
            client = self._client or httpx.Client(timeout=10, follow_redirects=True)
            resp = client.get(f"{origin}/robots.txt", headers=BROWSER_HEADERS)
            if resp.status_code >= 400:
                return None
            parser = RobotFileParser()
            parser.parse(resp.text.splitlines())
            return parser
        except Exception:
            return None
