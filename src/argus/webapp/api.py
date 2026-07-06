"""REST API for the Argus web layer.

Thin by design: every handler resolves fresh settings, opens a short-lived
session, and calls the same functions the CLI calls. Long work is submitted
to the JobManager and polled via /api/jobs/{id}.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from argus.config.profile import DomainProfile, list_domains, load_profile
from argus.config.registry import (Registry, Source, SourceStatus, load_registry,
                                   registry_commit)
from argus.fetchers import ADAPTERS
from argus.governance import candidates as cq
from argus.ingest.extract import extract_text
from argus.observe.health import health_report
from argus.snapshots.blob import BlobStore
from argus.snapshots.db import CandidateRow, RunRow, SnapshotRow
from argus.webapp import actions, config_store, rendermd
from argus.webapp.scheduler import validate_cron
from argus.webapp.state import AppState, fresh_settings

router = APIRouter(prefix="/api")


def _app(request: Request) -> AppState:
    return request.app.state.argus


def _iso(value: datetime | date | None) -> str | None:
    return value.isoformat(timespec="seconds") if isinstance(value, datetime) \
        else (value.isoformat() if value else None)


def _run_dict(r: RunRow) -> dict:
    return {
        "run_id": r.run_id, "domain": r.domain, "workload": r.workload,
        "status": r.status, "registry_commit": r.registry_commit,
        "models": r.models, "started_at": _iso(r.started_at),
        "finished_at": _iso(r.finished_at),
        "output_path": r.output_path, "manifest_path": r.manifest_path,
    }


# ===========================================================================
# overview
# ===========================================================================

@router.get("/overview")
def overview(request: Request):
    app = _app(request)
    settings = fresh_settings()
    try:
        registry = load_registry(settings.registry_path)
        registry_error = None
    except Exception as exc:
        registry, registry_error = None, str(exc)

    domains = []
    schedule_by_domain = {row["domain"]: row for row in app.scheduler.refresh()}
    with app.session() as session:
        last_runs = {
            row.domain: row for row in session.scalars(
                select(RunRow).order_by(RunRow.started_at.asc())
            )
        }  # ascending scan: the dict keeps the latest per domain
        snapshot_count = session.scalar(select(func.count(SnapshotRow.content_hash)))
        run_count = session.scalar(select(func.count(RunRow.run_id)))
        pending_candidates = session.scalar(
            select(func.count(CandidateRow.id)).where(CandidateRow.status == "pending")
        )
        health_summary: dict[str, int] = {}
        if registry is not None:
            for h in health_report(session, registry, window_hours=24):
                health_summary[h.status] = health_summary.get(h.status, 0) + 1

    for name in list_domains(settings.domains_dir):
        entry: dict = {"name": name}
        try:
            profile = load_profile(settings.domains_dir, name)
            entry["description"] = profile.description
            entry["window_hours"] = profile.retrieval.brief_window_hours
            entry["cron"] = profile.schedule.brief_cron
            if registry is not None:
                entry["sources_in_scope"] = len(registry.in_scope(
                    profile.registry_scope.include_tags,
                    profile.registry_scope.min_tier))
        except Exception as exc:
            entry["error"] = str(exc)
        sched = schedule_by_domain.get(name, {})
        entry["schedule_enabled"] = sched.get("enabled", False)
        entry["next_run"] = sched.get("next_run")
        last = last_runs.get(name)
        entry["last_run"] = _run_dict(last) if last else None
        domains.append(entry)

    azure_missing = []
    if settings.llm == "azure" or settings.embedder == "azure":
        import os
        needed = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
                  "AZURE_OPENAI_API_VERSION", "ARGUS_AZURE_EMBEDDING_DEPLOYMENT",
                  "ARGUS_AZURE_SYNTHESIS_DEPLOYMENT",
                  "ARGUS_AZURE_UTILITY_DEPLOYMENT"]
        azure_missing = [k for k in needed if not os.environ.get(k)]

    return {
        "domains": domains,
        "counts": {"snapshots": snapshot_count, "runs": run_count,
                   "pending_candidates": pending_candidates},
        "health": health_summary,
        "jobs_active": [j.to_dict() for j in app.jobs.active()],
        "mode": {"llm": settings.llm, "embedder": settings.embedder},
        "registry": {
            "commit": registry_commit(settings.registry_path),
            "error": registry_error,
            "sources": len(registry.sources) if registry else 0,
        },
        "azure_missing": azure_missing,
        "restart_required": app.restart_required(),
    }


# ===========================================================================
# runs + outputs
# ===========================================================================

@router.get("/runs")
def runs(request: Request, limit: int = 25, domain: str | None = None,
         workload: str | None = None):
    with _app(request).session() as session:
        stmt = select(RunRow).order_by(RunRow.started_at.desc()).limit(limit)
        if domain:
            stmt = stmt.where(RunRow.domain == domain)
        if workload:
            stmt = stmt.where(RunRow.workload == workload)
        return [_run_dict(r) for r in session.scalars(stmt)]


@router.get("/runs/{run_id}")
def run_detail(request: Request, run_id: str):
    with _app(request).session() as session:
        row = session.get(RunRow, run_id)
    if row is None:
        raise HTTPException(404, f"no run {run_id!r}")
    manifest = None
    if row.manifest_path and Path(row.manifest_path).exists():
        try:
            manifest = json.loads(Path(row.manifest_path).read_text())
        except Exception as exc:
            manifest = {"error": f"manifest unreadable: {exc}"}
    return {"run": _run_dict(row), "manifest": manifest}


@router.get("/runs/{run_id}/output")
def run_output(request: Request, run_id: str):
    with _app(request).session() as session:
        row = session.get(RunRow, run_id)
    if row is None:
        raise HTTPException(404, f"no run {run_id!r}")
    if not row.output_path or not Path(row.output_path).exists():
        raise HTTPException(404, "this run has no output file (it may have "
                                 "failed before synthesis)")
    text = Path(row.output_path).read_text()
    return {"run_id": run_id, "path": row.output_path,
            "markdown": text, "html": rendermd.render(text)}


# ===========================================================================
# actions (jobs)
# ===========================================================================

class BriefRequest(BaseModel):
    domain: str
    window_hours: int | None = Field(default=None, ge=1, le=24 * 30)


class ResearchRequest(BaseModel):
    domain: str
    question: str = Field(min_length=3)


class PollRequest(BaseModel):
    domain: str
    source: str | None = None


class IngestRequest(BaseModel):
    limit: int = Field(default=500, ge=1, le=10_000)


class DomainOnly(BaseModel):
    domain: str


def _require_domain(domain: str) -> None:
    settings = fresh_settings()
    if domain not in list_domains(settings.domains_dir):
        raise HTTPException(404, f"no domain {domain!r}")


@router.post("/actions/brief")
def action_brief(request: Request, body: BriefRequest):
    app = _app(request)
    _require_domain(body.domain)
    if app.jobs.active(kind="brief", domain=body.domain):
        raise HTTPException(409, f"a brief for {body.domain!r} is already "
                                 "queued or running")
    job = app.jobs.submit(
        "brief", f"Daily brief — {body.domain}",
        {"domain": body.domain, "window_hours": body.window_hours,
         "triggered_by": "manual"},
        actions.brief_job(app, body.domain, body.window_hours),
    )
    return {"job_id": job.id}


@router.post("/actions/research")
def action_research(request: Request, body: ResearchRequest):
    app = _app(request)
    _require_domain(body.domain)
    job = app.jobs.submit(
        "research", f"Deep research — {body.domain}",
        {"domain": body.domain, "question": body.question},
        actions.research_job(app, body.domain, body.question),
    )
    return {"job_id": job.id}


@router.post("/actions/poll")
def action_poll(request: Request, body: PollRequest):
    app = _app(request)
    _require_domain(body.domain)
    job = app.jobs.submit(
        "poll", f"Poll sources — {body.domain}",
        {"domain": body.domain, "source": body.source},
        actions.poll_job(app, body.domain, body.source),
    )
    return {"job_id": job.id}


@router.post("/actions/ingest")
def action_ingest(request: Request, body: IngestRequest):
    app = _app(request)
    job = app.jobs.submit(
        "ingest", "Ingest pending snapshots", {"limit": body.limit},
        actions.ingest_job(app, body.limit),
    )
    return {"job_id": job.id}


@router.post("/actions/replay/{run_id}")
def action_replay(request: Request, run_id: str):
    app = _app(request)
    job = app.jobs.submit(
        "replay", f"Replay — {run_id}", {"run_id": run_id},
        actions.replay_job(app, run_id),
    )
    return {"job_id": job.id}


@router.post("/actions/golden")
def action_golden(request: Request, body: DomainOnly):
    app = _app(request)
    _require_domain(body.domain)
    job = app.jobs.submit(
        "golden", f"Golden set — {body.domain}", {"domain": body.domain},
        actions.golden_job(app, body.domain),
    )
    return {"job_id": job.id}


@router.post("/actions/validate")
def action_validate(request: Request):
    """Fast, synchronous: registry + adapters + every profile (CLI `validate`)."""
    settings = fresh_settings()
    messages: list[dict] = []
    ok = True
    try:
        registry = load_registry(settings.registry_path)
        messages.append({"level": "ok", "text":
                         f"registry ok — {len(registry.sources)} sources, commit "
                         f"{registry_commit(settings.registry_path) or 'uncommitted'}"})
    except Exception as exc:
        return {"ok": False, "messages": [{"level": "error",
                                           "text": f"registry invalid: {exc}"}]}
    for source in registry.sources:
        if source.fetch.adapter not in ADAPTERS:
            ok = False
            messages.append({"level": "error", "text":
                             f"{source.id}: unknown adapter "
                             f"{source.fetch.adapter!r} (known: {sorted(ADAPTERS)})"})
    for domain in list_domains(settings.domains_dir):
        try:
            profile = load_profile(settings.domains_dir, domain)
        except Exception as exc:
            ok = False
            messages.append({"level": "error",
                             "text": f"profile {domain}: {exc}"})
            continue
        scope = registry.in_scope(profile.registry_scope.include_tags,
                                  profile.registry_scope.min_tier)
        missing = set(profile.registry_scope.include_tags) - registry.all_tags()
        text = f"profile ok — {domain}: {len(scope)} sources in scope"
        if missing:
            text += f" (tags with no source: {sorted(missing)})"
        messages.append({"level": "ok" if scope else "error", "text": text})
        if not scope:
            ok = False
            messages.append({"level": "error",
                             "text": f"{domain}: scope selects zero sources"})
        if profile.schedule.brief_cron:
            try:
                validate_cron(profile.schedule.brief_cron)
            except ValueError as exc:
                ok = False
                messages.append({"level": "error", "text": f"{domain}: {exc}"})
    return {"ok": ok, "messages": messages}


# ===========================================================================
# jobs
# ===========================================================================

@router.get("/jobs")
def jobs_list(request: Request, limit: int = 40):
    return [j.to_dict() for j in _app(request).jobs.list(limit)]


@router.get("/jobs/{job_id}")
def job_detail(request: Request, job_id: str):
    app = _app(request)
    job = app.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id!r}")
    d = job.to_dict(with_trace=True)
    # let the scheduler remember the outcome of the run it triggered
    if (job.status in ("done", "failed")
            and job.kind == "brief"
            and job.params.get("triggered_by") == "schedule"):
        app.scheduler.record_job_outcome(job.params.get("domain", ""), d)
    return d


# ===========================================================================
# registry (sources)
# ===========================================================================

@router.get("/registry")
def registry_view(request: Request):
    settings = fresh_settings()
    registry = load_registry(settings.registry_path)
    commit = registry_commit(settings.registry_path)
    return {
        "commit": commit,
        "dirty": bool(commit and commit.endswith("-dirty")),
        "path": str(settings.registry_path),
        "adapters": sorted(ADAPTERS),
        "tags": sorted(registry.all_tags()),
        "sources": [s.model_dump(mode="json") for s in registry.sources],
    }


class SourceBody(BaseModel):
    source: dict
    note: str = ""


@router.post("/registry/sources")
def registry_add(request: Request, body: SourceBody):
    return _registry_upsert(body, creating=True)


@router.put("/registry/sources/{source_id}")
def registry_update(request: Request, source_id: str, body: SourceBody):
    if body.source.get("id") != source_id:
        raise HTTPException(400, "source id in body must match the URL")
    return _registry_upsert(body, creating=False)


def _registry_upsert(body: SourceBody, creating: bool) -> dict:
    settings = fresh_settings()
    try:
        source = Source.model_validate(body.source)
    except Exception as exc:
        raise HTTPException(422, f"invalid source: {exc}")
    registry = load_registry(settings.registry_path)
    exists = any(s.id == source.id for s in registry.sources)
    if creating and exists:
        raise HTTPException(409, f"source {source.id!r} already exists")
    if not creating and not exists:
        raise HTTPException(404, f"no source {source.id!r}")
    verb = "add" if creating else "update"
    note = f" — {body.note}" if body.note else ""
    result = config_store.upsert_source(
        settings.registry_path, source,
        f"webapp: {verb} source '{source.id}'{note}",
    )
    return result


class StatusBody(BaseModel):
    status: str = Field(pattern="^(active|paused|retired)$")


@router.post("/registry/sources/{source_id}/status")
def registry_status(request: Request, source_id: str, body: StatusBody):
    settings = fresh_settings()
    registry = load_registry(settings.registry_path)
    try:
        source = registry.get(source_id)
    except KeyError:
        raise HTTPException(404, f"no source {source_id!r}")
    updated = source.model_copy(update={"status": SourceStatus(body.status)})
    return config_store.upsert_source(
        settings.registry_path, Source.model_validate(updated.model_dump()),
        f"webapp: set source '{source_id}' {body.status}",
    )


# ===========================================================================
# domains (profiles + entity dictionaries)
# ===========================================================================

@router.get("/domains")
def domains_list(request: Request):
    settings = fresh_settings()
    try:
        registry = load_registry(settings.registry_path)
    except Exception:
        registry = None
    out = []
    for name in list_domains(settings.domains_dir):
        entry: dict = {"name": name}
        try:
            profile = load_profile(settings.domains_dir, name)
            entry["profile"] = profile.model_dump(mode="json")
            if registry:
                entry["sources_in_scope"] = [
                    s.id for s in registry.in_scope(
                        profile.registry_scope.include_tags,
                        profile.registry_scope.min_tier)
                ]
        except Exception as exc:
            entry["error"] = str(exc)
        out.append(entry)
    return {"domains": out}


class CreateDomainBody(BaseModel):
    name: str = Field(pattern=r"^[a-z0-9_\-]+$")
    description: str = ""
    include_tags: list[str] = Field(min_length=1)
    min_tier: int = Field(default=3, ge=1, le=3)
    brief_cron: str | None = None


@router.post("/domains")
def domains_create(request: Request, body: CreateDomainBody):
    settings = fresh_settings()
    if body.brief_cron:
        try:
            validate_cron(body.brief_cron)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
    try:
        result = config_store.create_domain(
            settings.domains_dir, name=body.name, description=body.description,
            include_tags=body.include_tags, min_tier=body.min_tier,
            brief_cron=body.brief_cron,
        )
    except FileExistsError as exc:
        raise HTTPException(409, str(exc))
    _app(request).scheduler.refresh()
    return result


class ProfileBody(BaseModel):
    profile: dict


@router.put("/domains/{name}")
def domains_update(request: Request, name: str, body: ProfileBody):
    settings = fresh_settings()
    if name not in list_domains(settings.domains_dir):
        raise HTTPException(404, f"no domain {name!r}")
    try:
        profile = DomainProfile.model_validate(body.profile)
    except Exception as exc:
        raise HTTPException(422, f"invalid profile: {exc}")
    if profile.domain != name:
        raise HTTPException(400, "profile.domain must match the URL")
    if profile.schedule.brief_cron:
        try:
            validate_cron(profile.schedule.brief_cron)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
    result = config_store.save_profile(
        settings.domains_dir, profile, f"webapp: update domain '{name}'"
    )
    _app(request).scheduler.refresh()
    return result


@router.get("/domains/{name}/entities")
def entities_get(request: Request, name: str):
    settings = fresh_settings()
    if name not in list_domains(settings.domains_dir):
        raise HTTPException(404, f"no domain {name!r}")
    return config_store.read_entities(settings.domains_dir, name)


class EntitiesBody(BaseModel):
    path: str
    rows: list[dict]


@router.put("/domains/{name}/entities")
def entities_put(request: Request, name: str, body: EntitiesBody):
    settings = fresh_settings()
    if name not in list_domains(settings.domains_dir):
        raise HTTPException(404, f"no domain {name!r}")
    try:
        return config_store.write_entities(settings.domains_dir, name,
                                           body.path, body.rows)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


# ===========================================================================
# candidates (governance queue)
# ===========================================================================

@router.get("/candidates")
def candidates_list(request: Request, status: str = "pending", limit: int = 100):
    with _app(request).session() as session:
        stmt = select(CandidateRow).order_by(CandidateRow.created_at.desc()) \
                                   .limit(limit)
        if status != "all":
            stmt = stmt.where(CandidateRow.status == status)
        rows = list(session.scalars(stmt))
        return [{
            "id": r.id, "url": r.url, "netloc": r.netloc, "title": r.title,
            "rationale": r.rationale, "run_id": r.run_id, "status": r.status,
            "reason": r.reason, "created_at": _iso(r.created_at),
        } for r in rows]


class ApproveBody(BaseModel):
    add_to_registry: bool = False
    tag: str = ""
    tier: int = Field(default=3, ge=1, le=3)


@router.post("/candidates/{candidate_id}/approve")
def candidate_approve(request: Request, candidate_id: int, body: ApproveBody):
    settings = fresh_settings()
    app = _app(request)
    with app.session() as session:
        try:
            row = cq.approve(session, candidate_id)
        except KeyError:
            raise HTTPException(404, f"no candidate {candidate_id}")
        stanza = cq.yaml_stanza(row)
        response: dict = {"id": row.id, "netloc": row.netloc, "stanza": stanza,
                          "added_to_registry": False}
        if body.add_to_registry:
            if not body.tag:
                raise HTTPException(422, "a domain tag is required when adding "
                                         "to the registry")
            source = Source.model_validate({
                "id": row.netloc.replace(".", "_").replace("-", "_"),
                "name": row.title or row.netloc,
                "homepage": f"https://{row.netloc}",
                "fetch": {"adapter": "rss",
                          "endpoints": [f"https://{row.netloc}/FILL-feed-url"]},
                "license": "UNREVIEWED — verify the source's terms, then edit "
                           "this field before activating.",
                "tags": {body.tag: {"tier": body.tier}},
                "status": "paused",
                "notes": f"Proposed by run {row.run_id or 'n/a'}: "
                         f"{row.rationale or 'discovery hit'}",
            })
            result = config_store.upsert_source(
                settings.registry_path, source,
                f"webapp: add candidate '{row.netloc}' as paused source",
            )
            response["added_to_registry"] = True
            response["source_id"] = source.id
            response["registry"] = result
        return response


class RejectBody(BaseModel):
    reason: str = Field(min_length=2)


@router.post("/candidates/{candidate_id}/reject")
def candidate_reject(request: Request, candidate_id: int, body: RejectBody):
    with _app(request).session() as session:
        try:
            row = cq.reject(session, candidate_id, body.reason)
        except KeyError:
            raise HTTPException(404, f"no candidate {candidate_id}")
        return {"id": row.id, "netloc": row.netloc, "status": row.status,
                "reason": row.reason}


# ===========================================================================
# operations (health, snapshots)
# ===========================================================================

@router.get("/health")
def health(request: Request, window_hours: int = 24, domain: str | None = None):
    settings = fresh_settings()
    registry = load_registry(settings.registry_path)
    with _app(request).session() as session:
        rows = health_report(session, registry, window_hours=window_hours,
                             domain=domain)
    out = []
    for h in rows:
        d = asdict(h)
        d["last_ok_at"] = _iso(h.last_ok_at)
        d["status_by_registry"] = registry.get(h.source_id).status.value
        out.append(d)
    return {"window_hours": window_hours, "sources": out}


@router.get("/snapshots")
def snapshots(request: Request, limit: int = 30, source: str | None = None,
              q: str | None = None):
    with _app(request).session() as session:
        stmt = select(SnapshotRow).order_by(SnapshotRow.fetched_at.desc()) \
                                  .limit(limit)
        if source:
            stmt = stmt.where(SnapshotRow.source_id == source)
        if q:
            like = f"%{q}%"
            stmt = stmt.where(SnapshotRow.title.ilike(like)
                              | SnapshotRow.url.ilike(like))
        rows = list(session.scalars(stmt))
    return [{
        "content_hash": r.content_hash, "source_id": r.source_id,
        "url": r.url, "title": r.title, "media_type": r.media_type,
        "published_at": _iso(r.published_at), "fetched_at": _iso(r.fetched_at),
        "cluster_id": r.cluster_id, "extraction_status": r.extraction_status,
    } for r in rows]


@router.get("/snapshots/{content_hash}")
def snapshot_detail(request: Request, content_hash: str):
    settings = fresh_settings()
    with _app(request).session() as session:
        row = session.get(SnapshotRow, content_hash)
    if row is None:
        raise HTTPException(404, "no such snapshot")
    preview, preview_status = "", "unavailable"
    blob = BlobStore(settings.blob_root)
    if blob.exists(content_hash):
        text, status = extract_text(blob.get(content_hash), row.media_type)
        preview_status = status
        preview = text[:4000]
    return {
        "content_hash": row.content_hash, "source_id": row.source_id,
        "url": row.url, "title": row.title, "media_type": row.media_type,
        "published_at": _iso(row.published_at), "fetched_at": _iso(row.fetched_at),
        "cluster_id": row.cluster_id, "extraction_status": row.extraction_status,
        "meta": row.meta, "preview": preview, "preview_status": preview_status,
    }


# ===========================================================================
# settings + scheduler
# ===========================================================================

@router.get("/settings")
def settings_get(request: Request):
    return config_store.settings_view(_app(request).restart_required())


class SettingsBody(BaseModel):
    changes: dict[str, str | None]


@router.put("/settings")
def settings_put(request: Request, body: SettingsBody):
    try:
        result = config_store.update_settings(body.changes)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    result["restart_pending"] = _app(request).restart_required()
    return result


@router.get("/scheduler")
def scheduler_get(request: Request):
    return {"domains": _app(request).scheduler.refresh()}


class EnableBody(BaseModel):
    enabled: bool


@router.post("/scheduler/{domain}")
def scheduler_set(request: Request, domain: str, body: EnableBody):
    try:
        return {"domains": _app(request).scheduler.set_enabled(domain,
                                                               body.enabled)}
    except KeyError:
        raise HTTPException(404, f"no domain {domain!r}")
