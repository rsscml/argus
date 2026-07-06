"""Claim extraction (architecture SS8.3).

Extraction runs per evidence chunk, so every claim is born citing the snapshot
hash + chunk it came from. Two implementations behind one seam:
  - LlmClaimExtractor: AzureChatOpenAI structured output, temperature 0.
  - StubClaimExtractor: deterministic sentence/regex extraction for dev, tests,
    and the golden harness.
"""
from __future__ import annotations

import re
from typing import Protocol

from pydantic import BaseModel, Field

from argus.claims.models import Claim, ClaimEvidence
from argus.index.retrieval import RetrievedChunk
from argus.synthesis.sentences import split_sentences


def evidence_from_chunk(sid: str, chunk: RetrievedChunk, independence_group: str) -> ClaimEvidence:
    return ClaimEvidence(
        sid=sid, chunk_id=chunk.chunk_id, content_hash=chunk.content_hash,
        source_id=chunk.source_id, tier=chunk.tier, cluster_id=chunk.cluster_id,
        independence_group=independence_group or chunk.source_id,
        published_at=chunk.published_at,
    )


class ClaimExtractor(Protocol):
    name: str

    def extract(self, sid: str, chunk: RetrievedChunk, independence_group: str) -> list[Claim]: ...


_QUANT_RE = re.compile(
    r"(?i)\b(?P<metric>price|prices|premium|premiums|rate|rates|output|production|volume)\b"
    r"[^.\n]*?\b(?:to|at|of|reached)\s+\$?(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>percent|%|per barrel|bbl|usd|dollars|mb/d)?"
)
_DENIAL_RE = re.compile(r"(?i)\b(denied|denies|dismissed|rejected reports|no evidence)\b")
_DIRECTION_RE = re.compile(r"(?i)\b(rose|climbed|increased|fell|dropped|declined|eased)\b")


class StubClaimExtractor:
    """One claim per substantive sentence; a quantitative claim supersedes the
    event claim for its sentence (SS8.3 QuantClaim normalization is minimal
    here — real unit normalization rides on the LLM extractor)."""

    name = "stub-claims"

    def __init__(self, min_sentence_len: int = 25) -> None:
        self.min_len = min_sentence_len

    def extract(self, sid, chunk, independence_group) -> list[Claim]:
        ev = evidence_from_chunk(sid, chunk, independence_group)
        claims: list[Claim] = []
        for line in chunk.text.splitlines():
            for sentence in split_sentences(line.strip()):
                sentence = " ".join(sentence.split())
                if len(sentence) < self.min_len:
                    continue
                quant = _QUANT_RE.search(sentence)
                if quant:
                    direction = _DIRECTION_RE.search(sentence)
                    entity = (chunk.entity_ids[0] if chunk.entity_ids else None)
                    claims.append(Claim(
                        claim_type="quantitative", text=sentence,
                        entity_id=entity or quant.group("metric").lower(),
                        metric=quant.group("metric").lower().rstrip("s"),
                        value=float(quant.group("value")),
                        unit=(quant.group("unit") or "").lower() or None,
                        as_of=(chunk.published_at or "")[:10] or None,
                        direction=(direction.group(1).lower() if direction else None),
                        evidence=[ev],
                    ))
                    continue
                claims.append(Claim(
                    claim_type="event", text=sentence,
                    modality="denied" if _DENIAL_RE.search(sentence) else "asserted",
                    evidence=[ev],
                ))
        return claims


class _ExtractedClaim(BaseModel):
    claim_type: str = Field(description="'event' or 'quantitative'")
    text: str = Field(description="single-sentence restatement of the claim")
    modality: str = Field(default="asserted", description="asserted|denied|speculated")
    entity_id: str | None = None
    metric: str | None = None
    value: float | None = None
    unit: str | None = Field(default=None, description="normalized unit")
    as_of: str | None = Field(default=None, description="ISO date the value is as-of")
    direction: str | None = None


class _ClaimBatch(BaseModel):
    claims: list[_ExtractedClaim] = Field(default_factory=list)


_EXTRACT_PROMPT = """Extract discrete factual claims from the evidence below.
Rules: one claim per assertion, single-sentence text, no interpretation.
Use claim_type='quantitative' for any numeric fact (normalize value+unit and
set entity_id/metric/as_of); 'event' otherwise. Set modality='denied' when the
text denies the event.

EVIDENCE (published {published}):
{text}"""


class LlmClaimExtractor:
    name = "azure-claims"

    def __init__(self, chat_model=None) -> None:
        if chat_model is None:
            from argus.llm.factory import get_chat_model

            chat_model = get_chat_model("utility", temperature=0.0)
        self._model = chat_model.with_structured_output(_ClaimBatch)

    def extract(self, sid, chunk, independence_group) -> list[Claim]:
        ev = evidence_from_chunk(sid, chunk, independence_group)
        batch = self._model.invoke(_EXTRACT_PROMPT.format(
            published=chunk.published_at or "unknown", text=chunk.text[:4000]
        ))
        claims = []
        for item in batch.claims:
            if item.claim_type not in ("event", "quantitative"):
                continue
            modality = item.modality if item.modality in ("asserted", "denied", "speculated") else "asserted"
            claims.append(Claim(
                claim_type=item.claim_type, text=item.text, modality=modality,
                entity_id=item.entity_id, metric=item.metric, value=item.value,
                unit=item.unit, as_of=item.as_of or (chunk.published_at or "")[:10] or None,
                direction=item.direction, evidence=[ev],
            ))
        return claims
