"""Poll loop (architecture SS7.1, SS12.3).

Invariant G1 is enforced here structurally: the only iterable of sources is
`registry.in_scope(profile.registry_scope)` — there is no code path that
fetches anything else.

Failure isolation: each source is fetched and committed independently; an
exception is recorded as an error fetch-event and the loop continues.
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from argus.config.profile import DomainProfile
from argus.config.registry import Registry
from argus.fetchers import get_fetcher
from argus.fetchers.base import FetchContext, Fetcher
from argus.snapshots.db import FetchEventRow, WatermarkRow
from argus.snapshots.store import SnapshotStore
from argus.util import utcnow


@dataclass
class SourcePollResult:
    source_id: str
    status: str  # ok | not_modified | error | skipped_dry_run
    items_seen: int = 0
    snapshots_new: int = 0
    snapshots_dup: int = 0
    error: str | None = None


@dataclass
class PollSummary:
    domain: str
    registry_commit: str | None
    results: list[SourcePollResult] = field(default_factory=list)

    @property
    def totals(self) -> tuple[int, int, int]:
        return (
            sum(r.items_seen for r in self.results),
            sum(r.snapshots_new for r in self.results),
            sum(r.snapshots_dup for r in self.results),
        )


def poll_domain(
    registry: Registry,
    profile: DomainProfile,
    session: Session,
    store: SnapshotStore,
    *,
    registry_commit: str | None = None,
    only_source: str | None = None,
    dry_run: bool = False,
    fetcher_overrides: dict[str, Fetcher] | None = None,
) -> PollSummary:
    scope = profile.registry_scope
    sources = registry.in_scope(scope.include_tags, scope.min_tier)
    if only_source is not None:
        sources = [s for s in sources if s.id == only_source]
        if not sources:
            raise KeyError(
                f"source {only_source!r} is not in scope for domain {profile.domain!r}"
            )

    summary = PollSummary(domain=profile.domain, registry_commit=registry_commit)

    for source in sources:
        started = utcnow()
        result = SourcePollResult(source_id=source.id, status="ok")
        try:
            fetcher = (fetcher_overrides or {}).get(
                source.fetch.adapter
            ) or get_fetcher(source.fetch.adapter)

            wm = session.get(WatermarkRow, source.id)
            ctx = FetchContext(
                since=wm.last_polled_at if wm else None,
                etag=wm.etag if wm else None,
                last_modified=wm.last_modified if wm else None,
            )

            fetch_result = fetcher.fetch(source, ctx)
            result.items_seen = len(fetch_result.items)

            if dry_run:
                result.status = "skipped_dry_run"
            else:
                for item in fetch_result.items:
                    put = store.put(item)
                    if put.created:
                        result.snapshots_new += 1
                    else:
                        result.snapshots_dup += 1

                if wm is None:
                    wm = WatermarkRow(source_id=source.id)
                    session.add(wm)
                wm.last_polled_at = started
                wm.etag = fetch_result.etag
                wm.last_modified = fetch_result.last_modified

                if fetch_result.not_modified and not fetch_result.items:
                    result.status = "not_modified"

        except Exception as exc:
            session.rollback()  # isolate: discard this source's partial work only
            result.status = "error"
            result.error = f"{type(exc).__name__}: {exc}"
            result.error_trace = traceback.format_exc()  # type: ignore[attr-defined]

        if not dry_run:
            session.add(
                FetchEventRow(
                    source_id=source.id,
                    domain=profile.domain,
                    started_at=started,
                    finished_at=utcnow(),
                    status=result.status if result.status != "skipped_dry_run" else "ok",
                    items_seen=result.items_seen,
                    snapshots_new=result.snapshots_new,
                    snapshots_dup=result.snapshots_dup,
                    error=result.error,
                    registry_commit=registry_commit,
                )
            )
            session.commit()

        summary.results.append(result)

    return summary
