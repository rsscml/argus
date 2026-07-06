"""Per-source health report (architecture SS12.5).

Aggregates fetch events into the signal the operator needs: is each source
alive, when did it last succeed, and is it actually yielding new material?
A source whose snapshots_new collapses to zero for days is the early warning
for a feed change or site redesign (drift monitoring).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from argus.config.registry import Registry
from argus.snapshots.db import FetchEventRow
from argus.util import utcnow


@dataclass
class SourceHealth:
    source_id: str
    status: str  # ok | degraded | failing | unknown
    last_status: str | None
    last_ok_at: datetime | None
    consecutive_failures: int
    events_in_window: int
    items_in_window: int
    new_in_window: int
    last_error: str | None


def health_report(
    session: Session,
    registry: Registry,
    *,
    window_hours: int = 24,
    domain: str | None = None,
) -> list[SourceHealth]:
    cutoff = utcnow() - timedelta(hours=window_hours)
    report: list[SourceHealth] = []

    for source in registry.sources:
        stmt = (
            select(FetchEventRow)
            .where(FetchEventRow.source_id == source.id)
            .order_by(FetchEventRow.started_at.desc())
        )
        if domain:
            stmt = stmt.where(FetchEventRow.domain == domain)
        events = list(session.scalars(stmt))

        if not events:
            report.append(SourceHealth(source.id, "unknown", None, None, 0, 0, 0, 0, None))
            continue

        consecutive_failures = 0
        for e in events:
            if e.status == "error":
                consecutive_failures += 1
            else:
                break

        last_ok = next((e.started_at for e in events if e.status != "error"), None)
        in_window = [e for e in events if e.started_at >= cutoff]
        last_error = next((e.error for e in events if e.error), None)

        if consecutive_failures == 0:
            status = "ok"
        elif consecutive_failures < 3:
            status = "degraded"
        else:
            status = "failing"

        report.append(
            SourceHealth(
                source_id=source.id,
                status=status,
                last_status=events[0].status,
                last_ok_at=last_ok,
                consecutive_failures=consecutive_failures,
                events_in_window=len(in_window),
                items_in_window=sum(e.items_seen for e in in_window),
                new_in_window=sum(e.snapshots_new for e in in_window),
                last_error=last_error,
            )
        )
    return report
