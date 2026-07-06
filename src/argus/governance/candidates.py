"""Candidate queue (architecture SS6.2).

Discovery proposals wait here for a human. Approval does NOT edit the registry
programmatically — the registry changes only via reviewed git commits (AD-2) —
so `approve` marks the row and emits a ready-to-paste YAML stanza.
"""
from __future__ import annotations

from urllib.parse import urlsplit

from sqlalchemy import select
from sqlalchemy.orm import Session

from argus.config.registry import Registry
from argus.snapshots.db import CandidateRow


def _netloc(url: str) -> str:
    return urlsplit(url).netloc.lower().removeprefix("www.")


def registry_netlocs(registry: Registry) -> set[str]:
    locs: set[str] = set()
    for source in registry.sources:
        for url in [source.homepage or "", *source.fetch.endpoints]:
            if url.startswith("http"):
                locs.add(_netloc(url))
    return locs


def propose(
    session: Session, registry: Registry, *, url: str, title: str = "",
    rationale: str = "", run_id: str | None = None,
) -> bool:
    """Queue an off-registry discovery hit. Returns True if newly queued.

    Skips: registry domains (their content arrives via feeds), and domains
    already proposed in any status (rejected domains are never re-proposed,
    SS6.2)."""
    netloc = _netloc(url)
    if not netloc or netloc in registry_netlocs(registry):
        return False
    exists = session.scalar(
        select(CandidateRow).where(CandidateRow.netloc == netloc).limit(1)
    )
    if exists is not None:
        return False
    session.add(CandidateRow(url=url, netloc=netloc, title=title,
                             rationale=rationale, run_id=run_id))
    session.commit()
    return True


def pending(session: Session) -> list[CandidateRow]:
    return list(session.scalars(
        select(CandidateRow).where(CandidateRow.status == "pending")
        .order_by(CandidateRow.created_at)
    ))


def approve(session: Session, candidate_id: int) -> CandidateRow:
    row = session.get(CandidateRow, candidate_id)
    if row is None:
        raise KeyError(candidate_id)
    row.status = "approved"
    session.commit()
    return row


def reject(session: Session, candidate_id: int, reason: str) -> CandidateRow:
    row = session.get(CandidateRow, candidate_id)
    if row is None:
        raise KeyError(candidate_id)
    row.status = "rejected"
    row.reason = reason
    session.commit()
    return row


def yaml_stanza(row: CandidateRow) -> str:
    """Registry entry template for an approved candidate — paste, edit tiers
    and license, and commit (AD-2)."""
    return f"""  - id: {row.netloc.replace('.', '_').replace('-', '_')}
    name: {row.title or row.netloc}
    homepage: https://{row.netloc}
    fetch:
      adapter: rss
      endpoints:
        - <FILL: feed url on {row.netloc}>
    license: "<FILL: verify terms before activating>"
    tags:
      <FILL_TAG>: {{ tier: 3 }}   # start supplementary; promote deliberately
    status: paused                # activate only after license review
    notes: "Proposed by run {row.run_id or 'n/a'}: {row.rationale or 'discovery hit'}"
"""
