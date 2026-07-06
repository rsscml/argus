"""Shared graph state (architecture SS8.1/SS8.2).

TailState carries everything the shared corroborate -> synthesize -> verify
tail reads and writes; workload states extend it. Keeping the tail's state in
one class is what lets both graphs share the same node functions.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from argus.claims.models import CorroboratedClaim
from argus.index.retrieval import RetrievedChunk


class TailState(BaseModel):
    run_id: str
    domain: str
    evidence: dict[str, RetrievedChunk] = Field(default_factory=dict)
    claims: list[CorroboratedClaim] = Field(default_factory=list)
    verdict_counts: dict[str, int] = Field(default_factory=dict)
    body_md: str = ""
    cited_sids: list[str] = Field(default_factory=list)
    uncited_sentences: int = 0
    invalid_citations_stripped: int = 0
    verify_attempts: int = 0
    needs_retry: bool = False
    verifier_feedback: list[str] = Field(default_factory=list)
    sentences_checked: int = 0
    sentences_dropped: int = 0
    citation_precision: float = 1.0

    model_config = {"arbitrary_types_allowed": True}
