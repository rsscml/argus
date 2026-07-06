"""Argus CLI (M1): validate | poll | health | snapshots.

Later milestones add: research, replay, review (architecture SS13).
"""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from argus.config.profile import list_domains, load_profile
from argus.config.registry import load_registry, registry_commit
from argus.fetchers import ADAPTERS
from argus.observe.health import health_report
from argus.settings import get_settings
from argus.snapshots.blob import BlobStore
from argus.snapshots.db import SnapshotRow, init_db, make_engine, make_session_factory
from argus.snapshots.store import SnapshotStore

app = typer.Typer(add_completion=False, help="Argus — controlled-source research agent (M1)")
console = Console()


def _session():
    settings = get_settings()
    engine = make_engine(settings.resolved_db_url)
    init_db(engine)
    return settings, make_session_factory(engine)()


@app.command()
def validate() -> None:
    """Validate the registry, adapters, and all domain profiles."""
    settings = get_settings()
    registry = load_registry(settings.registry_path)
    commit = registry_commit(settings.registry_path)
    console.print(
        f"[green]registry ok[/green] — {len(registry.sources)} sources, "
        f"commit {commit or '(not in git — commit it: the registry is trust policy, SS6.1)'}"
    )

    problems = 0
    for source in registry.sources:
        if source.fetch.adapter not in ADAPTERS:
            console.print(
                f"[red]  {source.id}: unknown adapter {source.fetch.adapter!r}[/red]"
            )
            problems += 1

    domains = list_domains(settings.domains_dir)
    if not domains:
        console.print("[yellow]no domain profiles found[/yellow]")
    for domain in domains:
        profile = load_profile(settings.domains_dir, domain)
        scope = registry.in_scope(
            profile.registry_scope.include_tags, profile.registry_scope.min_tier
        )
        missing = set(profile.registry_scope.include_tags) - registry.all_tags()
        note = f" [yellow](tags with no source: {sorted(missing)})[/yellow]" if missing else ""
        console.print(
            f"[green]profile ok[/green] — {domain}: {len(scope)} sources in scope{note}"
        )
        if not scope:
            console.print(f"[red]  {domain}: scope selects zero sources[/red]")
            problems += 1

    raise typer.Exit(code=1 if problems else 0)


@app.command()
def poll(
    domain: str = typer.Option(..., help="Domain profile to poll"),
    source: str | None = typer.Option(None, help="Restrict to one source id"),
    dry_run: bool = typer.Option(False, help="Fetch but do not persist"),
) -> None:
    """Fetch all in-scope sources for a domain and snapshot new material."""
    from argus.fetchers.poll import poll_domain  # local import keeps CLI startup fast

    settings, session = _session()
    registry = load_registry(settings.registry_path)
    profile = load_profile(settings.domains_dir, domain)
    store = SnapshotStore(BlobStore(settings.blob_root), session)

    summary = poll_domain(
        registry,
        profile,
        session,
        store,
        registry_commit=registry_commit(settings.registry_path),
        only_source=source,
        dry_run=dry_run,
    )

    table = Table(title=f"poll — {domain} (registry {summary.registry_commit or 'uncommitted'})")
    for col in ("source", "status", "items", "new", "dup", "error"):
        table.add_column(col)
    for r in summary.results:
        style = {"ok": "green", "not_modified": "cyan", "error": "red"}.get(r.status, "")
        table.add_row(
            r.source_id, f"[{style}]{r.status}[/{style}]" if style else r.status,
            str(r.items_seen), str(r.snapshots_new), str(r.snapshots_dup),
            (r.error or "")[:80],
        )
    seen, new, dup = summary.totals
    console.print(table)
    console.print(f"totals: {seen} items seen, {new} new snapshots, {dup} duplicates")


@app.command()
def health(
    window_hours: int = typer.Option(24, help="Aggregation window"),
    domain: str | None = typer.Option(None, help="Restrict to one domain's events"),
) -> None:
    """Per-source health over the recent window (SS12.5)."""
    settings, session = _session()
    registry = load_registry(settings.registry_path)
    rows = health_report(session, registry, window_hours=window_hours, domain=domain)

    table = Table(title=f"source health — last {window_hours}h")
    for col in ("source", "status", "last ok", "consec. failures", "events", "items", "new", "last error"):
        table.add_column(col)
    style = {"ok": "green", "degraded": "yellow", "failing": "red", "unknown": "dim"}
    for h in rows:
        table.add_row(
            h.source_id,
            f"[{style[h.status]}]{h.status}[/{style[h.status]}]",
            h.last_ok_at.strftime("%Y-%m-%d %H:%M") if h.last_ok_at else "—",
            str(h.consecutive_failures),
            str(h.events_in_window),
            str(h.items_in_window),
            str(h.new_in_window),
            (h.last_error or "")[:60],
        )
    console.print(table)


@app.command()
def snapshots(
    limit: int = typer.Option(20),
    source: str | None = typer.Option(None),
) -> None:
    """List the most recent snapshots."""
    from sqlalchemy import select

    _, session = _session()
    stmt = select(SnapshotRow).order_by(SnapshotRow.fetched_at.desc()).limit(limit)
    if source:
        stmt = stmt.where(SnapshotRow.source_id == source)

    table = Table(title="snapshots")
    for col in ("hash", "source", "published", "fetched", "title/url"):
        table.add_column(col)
    for row in session.scalars(stmt):
        table.add_row(
            row.content_hash[:12],
            row.source_id,
            row.published_at.strftime("%Y-%m-%d %H:%M") if row.published_at else "—",
            row.fetched_at.strftime("%Y-%m-%d %H:%M"),
            (row.title or row.url)[:60],
        )
    console.print(table)


@app.command()
def ingest(limit: int = typer.Option(500, help="Max pending snapshots this run")) -> None:
    """Run the ingestion graph over pending snapshots (SS7.3)."""
    from argus.index.qdrant import make_client
    from argus.ingest.embed import make_embedder
    from argus.ingest.entities import EntityTagger, load_all_dictionaries
    from argus.ingest.pipeline import IngestDeps, run_ingest
    from argus.ingest.sparse import Bm25SparseEncoder

    settings, session = _session()
    registry = load_registry(settings.registry_path)
    stats = run_ingest(
        IngestDeps(
            session=session,
            blob=BlobStore(settings.blob_root),
            client=make_client(settings),
            collection=settings.collection,
            embedder=make_embedder(settings.embedder, settings.hashing_dim),
            sparse=Bm25SparseEncoder(),
            tagger=EntityTagger(load_all_dictionaries(settings.domains_dir)),
            registry=registry,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
        limit=limit,
    )
    console.print(
        f"selected {stats.selected} | extracted {stats.extracted} | "
        f"chunks upserted {stats.chunks_upserted} | joined existing clusters "
        f"{stats.clusters_joined} | failed {stats.failed} | "
        f"unsupported {stats.unsupported} | empty {stats.empty}"
    )


@app.command()
def search(
    query: str = typer.Argument(...),
    domain: str = typer.Option(...),
    window_hours: int | None = typer.Option(None, help="Recency filter"),
    top_k: int = typer.Option(8),
) -> None:
    """Hybrid retrieval over the index, scoped by the domain registry (SS7.4)."""
    from argus.index.qdrant import make_client
    from argus.index.retrieval import retrieve
    from argus.ingest.embed import make_embedder
    from argus.ingest.entities import QueryExpander, load_all_dictionaries
    from argus.ingest.sparse import Bm25SparseEncoder

    settings, _ = _session()
    registry = load_registry(settings.registry_path)
    profile = load_profile(settings.domains_dir, domain)
    results = retrieve(
        query,
        registry=registry,
        profile=profile,
        client=make_client(settings),
        collection=settings.collection,
        embedder=make_embedder(settings.embedder, settings.hashing_dim),
        sparse=Bm25SparseEncoder(),
        expander=QueryExpander(load_all_dictionaries(settings.domains_dir)),
        window_hours=window_hours,
        top_k=top_k,
    )
    table = Table(title=f"search — {domain!s}: {query!r}")
    for col in ("score", "tier", "source", "published", "cluster", "snapshot", "text"):
        table.add_column(col)
    for r in results:
        table.add_row(
            f"{r.score:.3f}", str(r.tier or "—"), r.source_id,
            (r.published_at or "—")[:16], (r.cluster_id or "—")[:8],
            r.content_hash[:10], (r.title or r.text)[:70],
        )
    console.print(table)


@app.command()
def brief(
    domain: str = typer.Option(...),
    window_hours: int | None = typer.Option(None, help="Override profile window"),
) -> None:
    """Run the daily brief graph end-to-end: poll -> ingest -> retrieve -> synthesize (SS8.2)."""
    from argus.graphs.checkpoint import make_checkpointer
    from argus.graphs.daily import run_daily_brief
    from argus.wiring import build_brief_deps

    settings, session = _session()
    deps = build_brief_deps(settings, session, domain)
    run = run_daily_brief(deps, window_hours=window_hours,
                          checkpointer=make_checkpointer(settings))
    console.print(f"[green]run {run.run_id} done[/green]")
    console.print(f"brief:    {run.output_path}")
    console.print(f"manifest: {run.manifest_path}")


@app.command()
def runs(limit: int = typer.Option(10)) -> None:
    """List recent runs (SS9)."""
    from sqlalchemy import select

    from argus.snapshots.db import RunRow

    _, session = _session()
    table = Table(title="runs")
    for col in ("run_id", "domain", "workload", "status", "started", "output"):
        table.add_column(col)
    for r in session.scalars(select(RunRow).order_by(RunRow.started_at.desc()).limit(limit)):
        style = {"done": "green", "failed": "red", "running": "yellow"}.get(r.status, "")
        table.add_row(r.run_id, r.domain, r.workload,
                      f"[{style}]{r.status}[/{style}]" if style else r.status,
                      r.started_at.strftime("%Y-%m-%d %H:%M"), r.output_path or "—")
    console.print(table)


@app.command()
def replay(run_id: str = typer.Argument(...)) -> None:
    """Evidence-identical replay from a run's manifest (SS11.3)."""
    from pathlib import Path

    from argus.graphs.replay import replay_run
    from argus.snapshots.db import RunRow
    from argus.wiring import build_brief_deps

    settings, session = _session()
    run = session.get(RunRow, run_id)
    if run is None or not run.manifest_path:
        console.print(f"[red]no manifest for run {run_id!r}[/red]")
        raise typer.Exit(1)
    deps = build_brief_deps(settings, session, run.domain)
    out_path, comparison = replay_run(deps, Path(run.manifest_path))
    console.print(f"[green]replayed[/green] -> {out_path}")
    console.print(f"snapshot set identical: {comparison['snapshot_set_identical']}")
    console.print(f"original citations: {comparison['original_cited_sids']}")
    console.print(f"replay citations:   {comparison['replay_cited_sids']}")


@app.command()
def golden(domain: str = typer.Option(...)) -> None:
    """Run the domain's golden set (SS12.5): verdict expectations + citation precision."""
    from pathlib import Path

    from argus.observe.golden import run_golden
    from argus.synthesis.synthesizer import make_synthesizer
    from argus.wiring import make_claim_stack

    settings = get_settings()
    profile = load_profile(settings.domains_dir, domain)
    cases = Path("tests/golden") / domain / "cases.yaml"
    if not cases.exists():
        console.print(f"[red]no golden set at {cases}[/red]")
        raise typer.Exit(1)
    extractor, matcher, verifier = make_claim_stack(settings)
    report = run_golden(profile, cases, extractor=extractor, matcher=matcher,
                        verifier=verifier, synthesizer=make_synthesizer(settings.llm))

    table = Table(title=f"golden — {domain}")
    for col in ("case", "passed", "failed", "citation precision"):
        table.add_column(col)
    for o in report.outcomes:
        table.add_row(
            o.case, "\n".join(o.passed) or "—",
            f"[red]{chr(10).join(o.failed)}[/red]" if o.failed else "—",
            f"{o.citation_precision:.2f}",
        )
    console.print(table)
    verdict = "[green]PASS[/green]" if report.ok else "[red]FAIL[/red]"
    console.print(
        f"{verdict} — expectations {report.expectation_pass_rate:.0%} "
        f"(target {report.targets.get('expectation_pass_rate', 1.0):.0%}), "
        f"citation precision {report.citation_precision:.2f} "
        f"(target {report.targets.get('citation_precision', 0.95):.2f})"
    )
    raise typer.Exit(0 if report.ok else 1)


@app.command()
def research(
    question: str = typer.Argument(...),
    domain: str = typer.Option(...),
) -> None:
    """Run the deep research graph on demand (SS8.2 right column)."""
    from argus.graphs.checkpoint import make_checkpointer
    from argus.graphs.research import run_research
    from argus.wiring import build_research_deps

    settings, session = _session()
    deps = build_research_deps(settings, session, domain)
    run = run_research(deps, question, checkpointer=make_checkpointer(settings))
    console.print(f"[green]run {run.run_id} done[/green]")
    console.print(f"report:   {run.output_path}")
    console.print(f"manifest: {run.manifest_path}")


@app.command()
def review(
    approve: int | None = typer.Option(None, help="Candidate id to approve"),
    reject: int | None = typer.Option(None, help="Candidate id to reject"),
    reason: str = typer.Option("", help="Rejection reason"),
) -> None:
    """Review the candidate source queue (SS6.2)."""
    from argus.governance import candidates as cq

    settings, session = _session()
    if approve is not None:
        row = cq.approve(session, approve)
        console.print(f"[green]approved[/green] {row.netloc} — paste into "
                      f"{settings.registry_path}, fill tiers + license, and COMMIT (AD-2):")
        console.print(cq.yaml_stanza(row))
        return
    if reject is not None:
        if not reason:
            console.print("[red]--reason is required when rejecting[/red]")
            raise typer.Exit(1)
        row = cq.reject(session, reject, reason)
        console.print(f"[yellow]rejected[/yellow] {row.netloc}: {reason} "
                      f"(will not be re-proposed)")
        return
    rows = cq.pending(session)
    if not rows:
        console.print("no pending candidates")
        return
    table = Table(title="candidate queue — pending human review")
    for col in ("id", "domain", "title", "proposed by", "rationale"):
        table.add_column(col)
    for r in rows:
        table.add_row(str(r.id), r.netloc, (r.title or "—")[:40],
                      r.run_id or "—", (r.rationale or "")[:60])
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
