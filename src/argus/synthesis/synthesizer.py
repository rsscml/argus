"""Grounded synthesis (architecture SS8.5 — completed at M4).

The synthesizer receives ONLY verdict-annotated claims and their evidence
snippets (SS8.4 output) — never the open index. Every sentence must cite
[S#] markers; fabricated markers are stripped; uncited sentences are counted;
the verifier (argus.synthesis.verify) closes the loop with capped retries.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from argus.claims.models import CorroboratedClaim, Verdict
from argus.index.retrieval import RetrievedChunk
from argus.synthesis.sentences import CITATION_RE, split_sentences


@dataclass
class Story:
    cluster_id: str
    headline: str
    chunks: list[RetrievedChunk] = field(default_factory=list)

    @property
    def newest(self) -> str:
        return max((c.published_at or "" for c in self.chunks), default="")


@dataclass
class SynthesisResult:
    body_md: str
    cited_sids: list[str]
    uncited_sentences: int
    invalid_citations_stripped: int


def build_stories(
    chunks: list[RetrievedChunk], *, max_stories: int, chunks_per_story: int
) -> tuple[list[Story], dict[str, RetrievedChunk]]:
    """Group window chunks into stories by syndication cluster (SS8.2 d3) and
    assign stable evidence ids S1..Sn. Returns (stories, sid -> chunk map)."""
    by_cluster: dict[str, list[RetrievedChunk]] = {}
    for chunk in chunks:
        by_cluster.setdefault(chunk.cluster_id or chunk.content_hash[:16], []).append(chunk)

    stories = []
    for cluster_id, members in by_cluster.items():
        members.sort(key=lambda c: ((c.published_at or ""), c.chunk_id), reverse=True)
        headline = next((m.title for m in members if m.title), None) or (
            members[0].text.strip().splitlines()[0][:90] if members else cluster_id
        )
        stories.append(Story(cluster_id=cluster_id, headline=headline,
                             chunks=members[:chunks_per_story]))
    stories.sort(key=lambda s: s.newest, reverse=True)
    stories = stories[:max_stories]

    evidence: dict[str, RetrievedChunk] = {}
    n = 0
    for story in stories:
        for chunk in story.chunks:
            n += 1
            evidence[f"S{n}"] = chunk
    return stories, evidence


def verdict_label(claim: CorroboratedClaim) -> str:
    if claim.verdict is Verdict.confirmed:
        n = claim.independent_voices
        return f"confirmed by {n} independent source" + ("s" if n != 1 else "")
    if claim.verdict is Verdict.attributed:
        return f"reported by {', '.join(claim.sources)}, unconfirmed"
    if claim.verdict is Verdict.contradicted:
        return f"CONFLICTING REPORTS — {claim.conflict or 'sources disagree'}"
    return "unverified"


class Synthesizer(Protocol):
    name: str

    def write_brief(
        self,
        domain: str,
        claims: list[CorroboratedClaim],
        evidence: dict[str, RetrievedChunk],
        feedback: list[str] | None = None,
    ) -> SynthesisResult: ...


def _validate_citations(body: str, evidence: dict[str, RetrievedChunk]) -> SynthesisResult:
    invalid = 0

    def _check(match: re.Match) -> str:
        nonlocal invalid
        sid = match.group(0)[1:-1]
        if sid in evidence:
            return match.group(0)
        invalid += 1
        return ""  # unknown evidence id: strip — never ship a fabricated citation

    cleaned = CITATION_RE.sub(_check, body)
    cited = sorted(
        {m[1:-1] for m in CITATION_RE.findall(cleaned)},
        key=lambda s: int(s[1:]),
    )
    uncited = 0
    for line in cleaned.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sentence in split_sentences(line):
            if sentence.strip() and not CITATION_RE.search(sentence):
                uncited += 1
    return SynthesisResult(cleaned.strip(), cited, uncited, invalid)


def _story_headlines(
    claims: list[CorroboratedClaim], evidence: dict[str, RetrievedChunk]
) -> dict[str | None, str]:
    headlines: dict[str | None, str] = {}
    for claim in claims:
        if claim.story_cluster in headlines:
            continue
        for sid in claim.sids:
            chunk = evidence.get(sid)
            if chunk is not None:
                headlines[claim.story_cluster] = chunk.title or chunk.text.splitlines()[0][:90]
                break
        headlines.setdefault(claim.story_cluster, claim.story_cluster or "Untitled")
    return headlines


class StubSynthesizer:
    """Deterministic, LLM-free: one line per claim, verdict label inline,
    every sentence cited by construction. Dev/test and degraded mode."""

    name = "stub-extractive"

    def write_brief(self, domain, claims, evidence, feedback=None) -> SynthesisResult:
        live = [c for c in claims if not c.dropped]
        headlines = _story_headlines(live, evidence)
        by_story: dict[str | None, list[CorroboratedClaim]] = {}
        for claim in live:
            by_story.setdefault(claim.story_cluster, []).append(claim)

        sections = []
        for story, story_claims in by_story.items():
            lines = [f"### {headlines[story]}"]
            for claim in story_claims:
                markers = "".join(f"[{s}]" for s in claim.sids)
                text = " ".join(claim.text.split()).rstrip(".")
                lines.append(f"{text} ({verdict_label(claim)}). {markers}")
            sections.append("\n".join(lines))
        return _validate_citations("\n\n".join(sections), evidence)


_PROMPT = """You are writing the daily monitoring brief for the '{domain}' domain.

STRICT RULES:
- Use ONLY the corroborated claims and evidence blocks below. No outside facts.
- Every sentence MUST end with citation markers like [S3] drawn from the claim's evidence.
- Reflect each claim's verdict wording faithfully: say "confirmed by N independent
  sources" only when the verdict is confirmed; attribute unconfirmed claims to their
  source; surface CONFLICTING REPORTS explicitly with both positions cited.
- One '### ' heading per story, then 1-3 sentences per claim group.
- Neutral, factual tone. No advice, no speculation.
{feedback}
CORROBORATED CLAIMS:
{claims}

EVIDENCE:
{evidence}

Write the brief now."""


class LlmSynthesizer:
    """AzureChatOpenAI via the AD-10 factory, temperature 0 (SS8.1)."""

    def __init__(self, chat_model=None) -> None:
        if chat_model is None:
            from argus.llm.factory import get_chat_model

            chat_model = get_chat_model("synthesis", temperature=0.0)
        self._model = chat_model
        self.name = "azure-llm"

    def write_brief(self, domain, claims, evidence, feedback=None) -> SynthesisResult:
        live = [c for c in claims if not c.dropped]
        claim_lines = []
        for i, claim in enumerate(live, 1):
            markers = "".join(f"[{s}]" for s in claim.sids)
            claim_lines.append(
                f"{i}. [{claim.verdict.value}] ({verdict_label(claim)}) "
                f"{claim.text} — evidence: {markers}"
            )
        evidence_blocks = []
        for sid in sorted({s for c in live for s in c.sids}, key=lambda s: int(s[1:])):
            chunk = evidence.get(sid)
            if chunk is None:
                continue
            evidence_blocks.append(
                f"[{sid}] source={chunk.source_id} tier={chunk.tier} "
                f"published={chunk.published_at or '?'}\n{chunk.text[:1200]}"
            )
        feedback_block = ""
        if feedback:
            failed = "\n".join(f"- {s}" for s in feedback)
            feedback_block = (
                "\nREVISION REQUIRED — these sentences failed evidence verification; "
                f"rewrite them strictly from the evidence or remove them:\n{failed}\n"
            )
        prompt = _PROMPT.format(
            domain=domain, feedback=feedback_block,
            claims="\n".join(claim_lines), evidence="\n\n".join(evidence_blocks),
        )
        response = self._model.invoke(prompt)
        body = response.content if hasattr(response, "content") else str(response)
        return _validate_citations(str(body), evidence)


def make_synthesizer(kind: str) -> Synthesizer:
    if kind == "azure":
        return LlmSynthesizer()
    if kind == "stub":
        return StubSynthesizer()
    raise ValueError(f"unknown synthesizer {kind!r} (expected 'azure' or 'stub')")
