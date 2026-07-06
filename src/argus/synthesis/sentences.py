"""Sentence + citation parsing shared by synthesis and verification (SS8.5)."""
from __future__ import annotations

import re

CITATION_RE = re.compile(r"\[S\d+\]")
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
_LEADING_CITATIONS_RE = re.compile(r"^((?:\[S\d+\]\s*)+)(.*)$", re.S)


def split_sentences(line: str) -> list[str]:
    """Split a line into sentences, binding 'sentence. [S1]'-style trailing
    citation markers back to the sentence they cite."""
    merged: list[str] = []
    for part in SENTENCE_END_RE.split(line):
        match = _LEADING_CITATIONS_RE.match(part)
        if match and merged:
            merged[-1] += " " + match.group(1).strip()
            rest = match.group(2).strip()
            if rest:
                merged.append(rest)
        else:
            merged.append(part)
    return [m for m in merged if m.strip()]


def sids_in(text: str) -> list[str]:
    return sorted({m[1:-1] for m in CITATION_RE.findall(text)}, key=lambda s: int(s[1:]))
