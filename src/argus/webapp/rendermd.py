"""Brief/report markdown → HTML with citation chips.

The synthesizer emits sentences carrying `[S12]` snapshot citations and a
Sources section whose entries start `**[S12]**`. After the markdown pass we:
1. give each Sources entry an anchor (`id="src-S12"`), then
2. turn every remaining `[S12]` into a superscript link to that anchor.
The chips are the UI's visible thread from any sentence back to its immutable
snapshot — the product's whole point, so they get first-class treatment.
"""
from __future__ import annotations

import re

import markdown as md

_SOURCE_ANCHOR = re.compile(r"<strong>\[(S\d+)\]</strong>")
_CITE = re.compile(r"\[(S\d+)\]")


def render(markdown_text: str) -> str:
    html = md.markdown(
        markdown_text,
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html",
    )
    html = _SOURCE_ANCHOR.sub(r'<strong id="src-\1" class="src-anchor">[\1]</strong>', html)
    html = _CITE.sub(r'<sup class="cite"><a href="#src-\1">\1</a></sup>', html)
    return html
