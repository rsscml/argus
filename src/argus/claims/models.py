"""Claim and verdict models (architecture SS8.3, SS8.4, SS9)."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    confirmed = "confirmed"
    attributed = "attributed"
    contradicted = "contradicted"
    insufficient = "insufficient"


class ClaimEvidence(BaseModel):
    """One supporting snippet: the SS9 CLAIM_EVIDENCE link."""

    sid: str                      # evidence id in the brief's citable universe
    chunk_id: str
    content_hash: str
    source_id: str
    tier: int | None = None
    cluster_id: str | None = None
    independence_group: str = ""
    published_at: str | None = None


class Claim(BaseModel):
    """A structured assertion extracted from evidence (SS8.3)."""

    claim_type: Literal["event", "quantitative"]
    text: str
    modality: Literal["asserted", "denied", "speculated"] = "asserted"
    # quantitative fields (SS8.3 QuantClaim): normalized tuple
    entity_id: str | None = None
    metric: str | None = None
    value: float | None = None
    unit: str | None = None
    as_of: str | None = None      # ISO date
    direction: str | None = None  # rose | fell | flat
    evidence: list[ClaimEvidence] = Field(default_factory=list)


class CorroboratedClaim(BaseModel):
    """Engine output: a merged claim group with its verdict (SS8.4)."""

    text: str
    claim_type: Literal["event", "quantitative"]
    verdict: Verdict
    independent_voices: int
    tier1_voices: int
    sids: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    story_cluster: str | None = None
    conflict: str | None = None   # human-readable description when contradicted
    dropped: bool = False         # fallback == 'drop': excluded from synthesis,
                                  # preserved in the manifest for audit
