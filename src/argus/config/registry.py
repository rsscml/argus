"""Source registry models and loader (architecture SS6.1).

The registry is the ONLY place trust is assigned (G1). It is a git-versioned
YAML file; every change is a reviewed commit, and the commit hash in force is
recorded with every run (SS12.1).
"""
from __future__ import annotations

import subprocess
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class SourceStatus(str, Enum):
    active = "active"
    paused = "paused"
    retired = "retired"


class FetchSpec(BaseModel):
    """How a source is pulled. Adapters are looked up by name (SS5.3)."""

    adapter: str
    endpoints: list[str] = Field(min_length=1)
    fetch_full_text: bool = False  # only set True if `license` permits (SS6.1)
    options: dict[str, Any] = Field(default_factory=dict)  # adapter-specific


class TagPolicy(BaseModel):
    """Per-tag authority tier: 1 primary/wire/official, 2 editorial, 3 supplementary."""

    tier: int = Field(ge=1, le=3)


class Source(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9_\-]+$")
    name: str
    homepage: str | None = None
    language: str = "en"
    fetch: FetchSpec
    license: str = Field(min_length=1)  # mandatory human-readable terms (SS6.1)
    tags: dict[str, TagPolicy] = Field(min_length=1)
    independence_group: str | None = None  # syndication family (SS8.4)
    status: SourceStatus = SourceStatus.active
    added: date | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _default_independence_group(self) -> "Source":
        if self.independence_group is None:
            self.independence_group = self.id
        return self

    def tier_for(self, tag: str) -> int | None:
        policy = self.tags.get(tag)
        return policy.tier if policy else None


class Registry(BaseModel):
    sources: list[Source]

    @field_validator("sources")
    @classmethod
    def _unique_ids(cls, v: list[Source]) -> list[Source]:
        seen: set[str] = set()
        for s in v:
            if s.id in seen:
                raise ValueError(f"duplicate source id: {s.id!r}")
            seen.add(s.id)
        return v

    def get(self, source_id: str) -> Source:
        for s in self.sources:
            if s.id == source_id:
                return s
        raise KeyError(source_id)

    def in_scope(self, include_tags: list[str], min_tier: int = 3) -> list[Source]:
        """Active sources matching a domain's registry_scope (SS5.1/SS5.2).

        A source is in scope if ANY of its tags is in `include_tags` with a
        tier value <= `min_tier` (tier 1 is highest authority).
        """
        selected: list[Source] = []
        for s in self.sources:
            if s.status is not SourceStatus.active:
                continue
            if any(
                (t := s.tier_for(tag)) is not None and t <= min_tier
                for tag in include_tags
            ):
                selected.append(s)
        return selected

    def all_tags(self) -> set[str]:
        return {tag for s in self.sources for tag in s.tags}


def load_registry(path: Path) -> Registry:
    """Load and validate the registry YAML (top-level list or {'sources': [...]})."""
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        raise ValueError(f"registry file is empty: {path}")
    if isinstance(raw, list):
        raw = {"sources": raw}
    try:
        return Registry.model_validate(raw)
    except Exception as exc:  # re-raise with file context
        raise ValueError(f"invalid registry at {path}: {exc}") from exc


def registry_commit(path: Path) -> str | None:
    """Git commit hash last touching the registry file, '-dirty' suffixed if it
    has uncommitted changes. None outside a git repo (SS12.1)."""
    try:
        wd = path.resolve().parent
        commit = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", path.name],
            cwd=wd, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        if not commit:
            return None
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", path.name],
            cwd=wd, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except Exception:
        return None
