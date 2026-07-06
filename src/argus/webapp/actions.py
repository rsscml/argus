"""Job bodies: thin wrappers over the core entry points.

Each function opens its own DB session (jobs run on the worker thread), pulls
fresh settings (so settings-page changes apply to the *next* run), and reuses
the process-shared Qdrant client via webapp.wiring. No pipeline logic here —
that all stays in the engine, exactly where the CLI finds it.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from argus.config.profile import load_profile
from argus.config.registry import load_registry, registry_commit
from argus.graphs.checkpoint import make_checkpointer
from argus.settings import Settings
from argus.snapshots.blob import BlobStore
from argus.snapshots.db import RunRow
from argus.snapshots.store import SnapshotStore
from argus.webapp import wiring
from argus.webapp.jobs import Job
from argus.webapp.state import AppState, fresh_settings


def _poll_result_dicts(summary) -> list[dict]:
    rows = []
    for r in summary.results:
        rows.append({
            "source_id": r.source_id, "status": r.status,
            "items_seen": r.items_seen, "snapshots_new": r.snapshots_new,
            "snapshots_dup": r.snapshots_dup, "error": r.error,
        })
    return rows


def poll_job(app: AppState, domain: str, source: str | None = None):
    def run(job: Job) -> dict:
        settings = fresh_settings()
        job.note = f"polling sources for {domain}"
        with app.session() as session:
            registry = load_registry(settings.registry_path)
            profile = load_profile(settings.domains_dir, domain)
            store = SnapshotStore(BlobStore(settings.blob_root), session)
            from argus.fetchers.poll import poll_domain

            summary = poll_domain(
                registry, profile, session, store,
                registry_commit=registry_commit(settings.registry_path),
                only_source=source,
            )
            seen, new, dup = summary.totals
            return {
                "domain": domain, "items_seen": seen, "snapshots_new": new,
                "snapshots_dup": dup, "sources": _poll_result_dicts(summary),
            }
    return run


def ingest_job(app: AppState, limit: int = 500):
    def run(job: Job) -> dict:
        settings = fresh_settings()
        job.note = "extract → cluster → tag → chunk → embed"
        with app.session() as session:
            from argus.ingest.pipeline import run_ingest

            deps = wiring.build_ingest_deps(settings, session, app.qdrant())
            stats = run_ingest(deps, limit=limit)
            return stats.model_dump()
    return run


def brief_job(app: AppState, domain: str, window_hours: int | None,
              triggered_by: str = "manual"):
    def run(job: Job) -> dict:
        settings = fresh_settings()
        job.note = f"daily brief for {domain} (poll → ingest → synthesize)"
        with app.session() as session:
            deps = wiring.build_brief_deps(settings, session, domain, app.qdrant())
            from argus.graphs.daily import run_daily_brief

            run_row = run_daily_brief(
                deps, window_hours=window_hours,
                checkpointer=make_checkpointer(settings),
            )
            return _run_row_result(run_row) | {"triggered_by": triggered_by}
    return run


def research_job(app: AppState, domain: str, question: str):
    def run(job: Job) -> dict:
        settings = fresh_settings()
        job.note = f"deep research in {domain}"
        with app.session() as session:
            deps = wiring.build_research_deps(settings, session, domain, app.qdrant())
            from argus.graphs.research import run_research

            run_row = run_research(
                deps, question, checkpointer=make_checkpointer(settings)
            )
            return _run_row_result(run_row) | {"question": question}
    return run


def replay_job(app: AppState, run_id: str):
    def run(job: Job) -> dict:
        settings = fresh_settings()
        with app.session() as session:
            row = session.get(RunRow, run_id)
            if row is None or not row.manifest_path:
                raise ValueError(f"no manifest recorded for run {run_id!r}")
            job.note = f"evidence-identical replay of {run_id}"
            deps = wiring.build_brief_deps(settings, session, row.domain, app.qdrant())
            from argus.graphs.replay import replay_run

            out_path, comparison = replay_run(deps, Path(row.manifest_path))
            return {"run_id": run_id, "output_path": str(out_path),
                    "comparison": comparison}
    return run


def golden_job(app: AppState, domain: str):
    def run(job: Job) -> dict:
        settings = fresh_settings()
        profile = load_profile(settings.domains_dir, domain)
        cases = Path("tests/golden") / domain / "cases.yaml"
        if not cases.exists():
            raise FileNotFoundError(
                f"no golden set at {cases} — add cases.yaml to enable quality "
                f"checks for this domain (architecture SS12.5)"
            )
        job.note = f"golden set for {domain}"
        from argus.observe.golden import run_golden
        from argus.synthesis.synthesizer import make_synthesizer
        from argus.wiring import make_claim_stack

        extractor, matcher, verifier = make_claim_stack(settings)
        report = run_golden(profile, cases, extractor=extractor, matcher=matcher,
                            verifier=verifier,
                            synthesizer=make_synthesizer(settings.llm))
        return {
            "domain": domain,
            "ok": report.ok,
            "expectation_pass_rate": report.expectation_pass_rate,
            "citation_precision": report.citation_precision,
            "targets": report.targets,
            "outcomes": [asdict(o) for o in report.outcomes],
        }
    return run


def _run_row_result(run_row: RunRow) -> dict:
    return {
        "run_id": run_row.run_id,
        "domain": run_row.domain,
        "workload": run_row.workload,
        "status": run_row.status,
        "output_path": run_row.output_path,
        "manifest_path": run_row.manifest_path,
    }
