"""Gated discovery seam (architecture SS7.1, SS6.2, invariant 1).

A DiscoveryProvider observes the open web; its results can ONLY become
candidate-queue proposals — never snapshots, never citations. NullDiscovery
is the default; operators plug a real provider (e.g. an Azure AI Search or
Bing wrapper) behind this interface without touching the engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class DiscoveryHit:
    url: str
    title: str = ""
    snippet: str = ""


class DiscoveryProvider(Protocol):
    name: str

    def search(self, query: str, k: int = 5) -> list[DiscoveryHit]: ...


class NullDiscovery:
    name = "null"

    def search(self, query: str, k: int = 5) -> list[DiscoveryHit]:
        return []
