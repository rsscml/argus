"""Citation verification (architecture SS8.5, completed at M4).

Every sentence in the synthesized body must carry >=1 [S#] citation and be
entailed by the snippets it cites (SS8.5, AD-8). Cited sentences are checked
against their snippets; UNCITED sentences fail by definition — they are
unverifiable, and because fabricated citation markers are stripped upstream
(_validate_citations), an uncited sentence is exactly what a hallucinated
citation degrades into. Failures loop back to the synthesizer with the
offending sentences flagged, up to the retry cap; after the cap, unsupported
content is DROPPED — never shipped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from argus.index.retrieval import RetrievedChunk
from argus.ingest.sparse import tokenize
from argus.synthesis.sentences import CITATION_RE, sids_in, split_sentences

# Verdict labels are engine-generated annotations (SS8.4), not factual claims —
# strip them before entailment so "(confirmed by 2 independent sources)" never
# fails verification against evidence that obviously doesn't contain it.
_VERDICT_LABEL_RES = [
    re.compile(r"\(CONFLICTING REPORTS.*", re.S),
    re.compile(r"\((?:confirmed by \d+ independent sources?|reported by [^)]*, unconfirmed|unverified)\)"),
]


def strip_verdict_labels(sentence: str) -> str:
    for pattern in _VERDICT_LABEL_RES:
        sentence = pattern.sub("", sentence)
    return sentence.strip()


# Markdown structure, not prose: headings and horizontal rules carry no factual
# claim, so they are neither checked nor droppable.
_STRUCTURAL_LINE_RE = re.compile(r"^(?:#|(?:-{3,}|\*{3,}|_{3,})\s*$)")


def _is_structural(line: str) -> bool:
    return bool(_STRUCTURAL_LINE_RE.match(line.strip()))


@dataclass
class SentenceCheck:
    sentence: str
    sids: list[str]
    supported: bool


@dataclass
class VerificationReport:
    checks: list[SentenceCheck] = field(default_factory=list)

    @property
    def checked(self) -> int:
        return len(self.checks)

    @property
    def failed(self) -> list[SentenceCheck]:
        return [c for c in self.checks if not c.supported]

    @property
    def precision(self) -> float:
        return 1.0 if not self.checks else (self.checked - len(self.failed)) / self.checked


class Verifier(Protocol):
    name: str

    def supported(self, sentence: str, evidence_texts: list[str]) -> bool: ...


class HeuristicVerifier:
    """Lexical entailment proxy: the share of the sentence's content tokens
    present in its cited snippets. Deterministic; dev/offline and the cheap
    first gate. The LLM verifier is the production judge."""

    name = "heuristic-overlap"

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def supported(self, sentence: str, evidence_texts: list[str]) -> bool:
        stripped = CITATION_RE.sub("", sentence)
        tokens = set(tokenize(stripped))
        if not tokens:
            return True
        pool = set()
        for text in evidence_texts:
            pool.update(tokenize(text))
        return len(tokens & pool) / len(tokens) >= self.threshold


class LlmVerifier:
    """Azure-backed entailment at temperature 0 on the utility deployment (SS10)."""

    name = "azure-entailment"

    def __init__(self, chat_model=None) -> None:
        if chat_model is None:
            from argus.llm.factory import get_chat_model

            chat_model = get_chat_model("utility", temperature=0.0)
        self._model = chat_model

    def supported(self, sentence: str, evidence_texts: list[str]) -> bool:
        joined = "\n---\n".join(t[:1500] for t in evidence_texts)
        prompt = (
            "Is the SENTENCE fully supported by the EVIDENCE (no added facts, "
            "numbers, or attributions)? Answer strictly YES or NO.\n"
            f"SENTENCE: {CITATION_RE.sub('', sentence).strip()}\nEVIDENCE:\n{joined}"
        )
        response = self._model.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        return "YES" in str(content).upper()


def check_body(
    body: str, evidence: dict[str, RetrievedChunk], verifier: Verifier
) -> VerificationReport:
    report = VerificationReport()
    for line in body.splitlines():
        line = line.strip()
        if not line or _is_structural(line):
            continue
        for sentence in split_sentences(line):
            sids = sids_in(sentence)
            if not sids:
                # SS8.5: every sentence must carry >=1 citation. Uncited prose is
                # unverifiable — and it is what a fabricated [S#] becomes after
                # _validate_citations strips the marker — so it FAILS and rides
                # the same retry-then-drop loop as any unsupported sentence.
                report.checks.append(SentenceCheck(
                    sentence=sentence, sids=[], supported=False,
                ))
                continue
            texts = [evidence[s].text for s in sids if s in evidence]
            checkable = strip_verdict_labels(sentence)
            report.checks.append(SentenceCheck(
                sentence=sentence, sids=sids,
                supported=verifier.supported(checkable, texts),
            ))
    return report


def drop_unsupported(body: str, report: VerificationReport) -> tuple[str, int]:
    """Remove failed sentences after the retry cap. Unsupported content is
    never shipped (SS8.5)."""
    bad = {c.sentence.strip() for c in report.failed}
    if not bad:
        return body, 0
    dropped = 0
    lines_out: list[str] = []
    for line in body.splitlines():
        if not line.strip() or _is_structural(line):
            lines_out.append(line)
            continue
        kept = []
        for sentence in split_sentences(line.strip()):
            if sentence.strip() in bad:
                dropped += 1
            else:
                kept.append(sentence.strip())
        if kept:
            lines_out.append(" ".join(kept))
    return "\n".join(lines_out).strip(), dropped
