"""Extraction (SS7.3 stage 1): snapshot bytes -> clean text.

Returns (text, status). Statuses map to SnapshotRow.extraction_status:
'ok' (caller sets 'done' after indexing), 'empty', 'unsupported', 'failed'.
PDF extraction is deferred (tracked as 'unsupported' with a meta note).
"""
from __future__ import annotations

import json

import trafilatura

_TEXT_TYPES = {"text/markdown", "text/plain", "text/csv"}


def extract_text(content: bytes, media_type: str) -> tuple[str, str]:
    try:
        if media_type == "text/html":
            decoded = content.decode("utf-8", errors="replace")
            text = trafilatura.extract(decoded, include_comments=False) or ""
            if not text.strip():
                # RSS summaries are often bare fragments trafilatura rejects;
                # fall back to a naive tag strip so headline+summary survive.
                text = _strip_tags(decoded)
            return (text.strip(), "ok" if text.strip() else "empty")
        if media_type in _TEXT_TYPES:
            text = content.decode("utf-8", errors="replace").strip()
            return (text, "ok" if text else "empty")
        if media_type == "application/json":
            payload = json.loads(content.decode("utf-8", errors="replace"))
            text = json.dumps(payload, indent=1, ensure_ascii=False)
            return (text, "ok")
        if media_type == "application/pdf":
            return _extract_pdf(content)
        return ("", "unsupported")
    except Exception:
        return ("", "failed")


def _extract_pdf(content: bytes) -> tuple[str, str]:
    """Text-layer extraction via pypdf (SS7.3 stage 1).

    Scanned/image-only PDFs yield no text layer and surface as 'empty' —
    deliberately visible in ingest stats rather than silently indexed blank.
    OCR is tracked as an open question (architecture Q3).
    """
    from io import BytesIO

    from pypdf import PdfReader

    reader = PdfReader(BytesIO(content))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            return ("", "failed")
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    text = "\n\n".join(p.strip() for p in pages if p.strip()).strip()
    return (text, "ok" if len(text) >= 20 else "empty")


def _strip_tags(html: str) -> str:
    import re

    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()
