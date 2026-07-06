"""Domain profile models (architecture SS5).

The full SS5.1 schema is modeled now, even though M1 only consumes
`registry_scope` and `schedule` — profiles are the genericity contract (G5)
and defining them early prevents drift in M2-M5.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class RegistryScope(BaseModel):
    include_tags: list[str] = Field(min_length=1)
    min_tier: int = Field(default=3, ge=1, le=3)


class EntityDictionary(BaseModel):
    path: Path  # CSV: surface_form, canonical_id, type (SS5.1)


class EntitiesConfig(BaseModel):
    dictionaries: list[EntityDictionary] = Field(default_factory=list)


class QuantTolerance(BaseModel):
    relative: float | None = Field(default=None, ge=0)
    absolute: float | None = Field(default=None, ge=0)


class QuantClaimsConfig(BaseModel):
    unit_normalization: bool = True
    tolerance: QuantTolerance = Field(default_factory=QuantTolerance)


class ClaimsConfig(BaseModel):
    schemas: list[Literal["event", "quantitative"]] = Field(default=["event"])
    quantitative: QuantClaimsConfig = Field(default_factory=QuantClaimsConfig)


class CorroborationRule(BaseModel):
    min_sources: int = Field(default=2, ge=1)
    min_tier1: int = Field(default=1, ge=0)
    independence: Literal["story_cluster", "independence_group"] = "story_cluster"


class QuantCorroborationRule(CorroborationRule):
    match: list[str] = Field(default=["entity", "metric", "as_of_window"])
    value_tolerance: float | Literal["inherit"] = "inherit"


class CorroborationConfig(BaseModel):
    event: CorroborationRule = Field(default_factory=CorroborationRule)
    quantitative: QuantCorroborationRule = Field(default_factory=QuantCorroborationRule)
    # Below-threshold behavior: publish as labeled attribution, never silence
    # or false confidence (SS5.2, R4).
    fallback: Literal["attribute", "flag", "drop"] = "attribute"


class RetrievalConfig(BaseModel):
    brief_window_hours: int = Field(default=24, ge=1)
    top_k: int = Field(default=24, ge=1)
    rerank: bool = True
    max_iterations: int = Field(default=4, ge=1)  # bounded loops only (SS8.1)


class OutputConfig(BaseModel):
    brief_template: Path | None = None
    report_template: Path | None = None


class ScheduleConfig(BaseModel):
    brief_cron: str | None = None


class DomainProfile(BaseModel):
    domain: str = Field(pattern=r"^[a-z0-9_\-]+$")
    description: str = ""
    registry_scope: RegistryScope
    entities: EntitiesConfig = Field(default_factory=EntitiesConfig)
    claims: ClaimsConfig = Field(default_factory=ClaimsConfig)
    corroboration: CorroborationConfig = Field(default_factory=CorroborationConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)


def load_profile(domains_dir: Path, domain: str) -> DomainProfile:
    path = domains_dir / domain / "profile.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no profile at {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    try:
        profile = DomainProfile.model_validate(raw)
    except Exception as exc:
        raise ValueError(f"invalid profile at {path}: {exc}") from exc
    if profile.domain != domain:
        raise ValueError(
            f"profile.domain {profile.domain!r} does not match directory {domain!r}"
        )
    return profile


def list_domains(domains_dir: Path) -> list[str]:
    if not domains_dir.exists():
        return []
    return sorted(
        p.name for p in domains_dir.iterdir() if (p / "profile.yaml").exists()
    )
