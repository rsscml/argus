"""robots.txt gate (architecture SS12.4).

Consulted only for full-page fetches (feed endpoints are published for
consumption). Standard convention: unreachable robots.txt => allowed;
an explicit disallow is honored.
"""
from __future__ import annotations

from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

USER_AGENT = "argus-research-agent/0.1"


class RobotsGate:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client
        self._cache: dict[str, RobotFileParser | None] = {}

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._cache:
            self._cache[origin] = self._load(origin)
        parser = self._cache[origin]
        if parser is None:
            return True
        return parser.can_fetch(USER_AGENT, url)

    def _load(self, origin: str) -> RobotFileParser | None:
        try:
            client = self._client or httpx.Client(timeout=10, follow_redirects=True)
            resp = client.get(f"{origin}/robots.txt", headers={"User-Agent": USER_AGENT})
            if resp.status_code >= 400:
                return None
            parser = RobotFileParser()
            parser.parse(resp.text.splitlines())
            return parser
        except Exception:
            return None
