"""Entity dictionaries (architecture SS5.1/SS5.2).

Plain CSVs (surface_form, canonical_id, type) shipped inside domain profiles.
Two consumers:
  - EntityTagger: annotates canonical ids into chunk metadata at ingestion.
  - QueryExpander: rewrites the sparse query with sibling surface forms at
    retrieval ("WTI" <-> "West Texas Intermediate"), where niche recall is won.

Matching is word-boundary, case-insensitive regex over all surfaces. Fine for
dictionaries up to a few thousand entries; swap in Aho-Corasick behind the
same interface if a domain outgrows that.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from argus.config.profile import list_domains, load_profile


@dataclass(frozen=True)
class Entity:
    surface: str
    canonical_id: str
    type: str = ""


def load_dictionary(path: Path) -> list[Entity]:
    entities: list[Entity] = []
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row or row[0].strip().lower() in ("surface_form", "#"):
                continue
            surface = row[0].strip()
            canonical = row[1].strip() if len(row) > 1 else surface
            etype = row[2].strip() if len(row) > 2 else ""
            if surface:
                entities.append(Entity(surface, canonical, etype))
    return entities


def load_all_dictionaries(domains_dir: Path) -> list[Entity]:
    """Union of every domain's dictionaries. Ingestion is domain-agnostic —
    snapshots aren't owned by a domain — so tagging uses all vocabularies;
    retrieval scoping happens later via the registry (SS7.4)."""
    entities: list[Entity] = []
    for domain in list_domains(domains_dir):
        profile = load_profile(domains_dir, domain)
        for spec in profile.entities.dictionaries:
            path = spec.path if spec.path.is_absolute() else domains_dir / domain / spec.path
            if path.exists():
                entities.extend(load_dictionary(path))
    return entities


class EntityTagger:
    def __init__(self, entities: list[Entity]) -> None:
        self._by_surface = { }
        surfaces = sorted({e.surface for e in entities}, key=len, reverse=True)
        for e in entities:
            self._by_surface.setdefault(e.surface.lower(), e.canonical_id)
        self._pattern = (
            re.compile(
                r"\b(?:" + "|".join(re.escape(s) for s in surfaces) + r")\b",
                re.IGNORECASE,
            )
            if surfaces
            else None
        )

    def tag(self, text: str) -> list[str]:
        if self._pattern is None:
            return []
        found = {self._by_surface[m.group(0).lower()] for m in self._pattern.finditer(text)}
        return sorted(found)


class QueryExpander:
    def __init__(self, entities: list[Entity]) -> None:
        self._tagger = EntityTagger(entities)
        self._surfaces_of: dict[str, set[str]] = {}
        for e in entities:
            self._surfaces_of.setdefault(e.canonical_id, set()).add(e.surface)

    def expand(self, query: str) -> str:
        extra: set[str] = set()
        for canonical in self._tagger.tag(query):
            extra.add(canonical)
            extra.update(self._surfaces_of.get(canonical, ()))
        if not extra:
            return query
        return query + " " + " ".join(sorted(extra))
