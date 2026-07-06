"""Daily brief graph (architecture SS8.2 left column + shared tail at M3 level).

Deterministic pipeline: poll -> ingest -> window retrieval -> story grouping
-> grounded synthesis. No planner, no agentic search (that's deep research).
Corroboration is STUBBED at M3 (roadmap): the corroborate node is a pass-
through placeholder so the M4 engine drops into an existing socket.

Runtime conventions (SS8.1): checkpointer on every run (thread_id = run_id,
so checkpoint history + the runs row = the manifest backbone, AD-6); model
identity recorded in the run row; bounded, linear control flow.
"""
from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from argus.claims.engine import EventMatcher
from argus.claims.extract import ClaimExtractor
from argus.config.profile import DomainProfile
from argus.config.registry import Registry
from argus.graphs.shared import attach_tail, serialize_evidence, tail_manifest_fields
from argus.graphs.state import TailState
from argus.index.retrieval import window_scan
from argus.ingest.pipeline import IngestDeps, run_ingest
from argus.settings import Settings
from argus.snapshots.blob import BlobStore
from argus.snapshots.db import RunRow
from argus.snapshots.store import SnapshotStore
from argus.synthesis.render import render_brief
from argus.synthesis.synthesizer import Synthesizer, build_stories
from argus.synthesis.verify import Verifier
from argus.util import utcnow


class BriefState(TailState):
    window_hours: int = 24
    polled_new: int = 0
    ingested: int = 0
    story_count: int = 0


@dataclass
class BriefDeps:
    settings: Settings
    session: Session
    registry: Registry
    profile: DomainProfile
    client: QdrantClient
    ingest_deps: IngestDeps
    synthesizer: Synthesizer
    extractor: ClaimExtractor
    matcher: EventMatcher
    verifier: Verifier
    registry_commit: str | None
    skip_poll: bool = False  # replay/offline runs


def build_daily_graph(deps: BriefDeps, checkpointer=None):
    def poll(state: BriefState) -> dict:
        if deps.skip_poll:
            return {}
        from argus.fetchers.poll import poll_domain

        store = SnapshotStore(BlobStore(deps.settings.blob_root), deps.session)
        summary = poll_domain(
            deps.registry, deps.profile, deps.session, store,
            registry_commit=deps.registry_commit,
        )
        _, new, _ = summary.totals
        return {"polled_new": new}

    def ingest(state: BriefState) -> dict:
        stats = run_ingest(deps.ingest_deps)
        return {"ingested": stats.extracted}

    def retrieve_window(state: BriefState) -> dict:
        chunks = window_scan(
            registry=deps.registry, profile=deps.profile, client=deps.client,
            collection=deps.ingest_deps.collection,  # single source of truth
            window_hours=state.window_hours,
        )
        stories, evidence = build_stories(
            chunks,
            max_stories=deps.settings.brief_max_stories,
            chunks_per_story=deps.settings.brief_chunks_per_story,
        )
        # stories are recomputed deterministically from evidence in synthesize;
        # only the evidence map (the citable universe) lives in state.
        return {"evidence": evidence, "story_count": len(stories)}

    graph = StateGraph(BriefState)
    graph.add_node("poll", poll)
    graph.add_node("ingest", ingest)
    graph.add_node("retrieve_window", retrieve_window)
    graph.add_edge(START, "poll")
    graph.add_edge("poll", "ingest")
    graph.add_edge("ingest", "retrieve_window")
    attach_tail(graph, deps, after="retrieve_window")  # SS4 invariant 2
    return graph.compile(checkpointer=checkpointer)


def _profile_hash(domains_dir: Path, domain: str) -> str:
    path = domains_dir / domain / "profile.yaml"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else ""


def _model_fingerprints(settings: Settings) -> dict:
    fingerprints = {"synthesizer": settings.llm, "embedder": settings.embedder}
    if settings.llm == "azure" or settings.embedder == "azure":
        from argus.llm.factory import model_fingerprint

        fingerprints["azure"] = model_fingerprint("synthesis")
    return fingerprints


def run_daily_brief(
    deps: BriefDeps, *, window_hours: int | None = None, checkpointer=None
) -> RunRow:
    settings = deps.settings
    window = window_hours or deps.profile.retrieval.brief_window_hours
    run_id = f"{utcnow():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"

    run = RunRow(
        run_id=run_id, domain=deps.profile.domain, workload="daily_brief",
        registry_commit=deps.registry_commit, models=_model_fingerprints(settings),
    )
    deps.session.add(run)
    deps.session.commit()

    try:
        graph = build_daily_graph(deps, checkpointer=checkpointer)
        final = graph.invoke(
            BriefState(run_id=run_id, domain=deps.profile.domain, window_hours=window),
            config={"configurable": {"thread_id": run_id}},
        )
        state = BriefState.model_validate(final)

        out_dir = settings.briefs_dir / deps.profile.domain
        out_dir.mkdir(parents=True, exist_ok=True)
        generated_at = utcnow().isoformat(timespec="seconds")
        brief_md = render_brief(
            profile=deps.profile, domains_dir=settings.domains_dir,
            domain=deps.profile.domain, run_id=run_id, generated_at=generated_at,
            window_hours=window, body_md=state.body_md, evidence=state.evidence,
            registry_commit=deps.registry_commit,
            synthesizer=deps.synthesizer.name, embedder=deps.ingest_deps.embedder.name,
        )
        brief_path = out_dir / f"{run_id}.md"
        brief_path.write_text(brief_md)

        manifest = {
            "run_id": run_id,
            "workload": "daily_brief",
            "domain": deps.profile.domain,
            "window_hours": window,
            "registry_commit": deps.registry_commit,
            "profile_hash": _profile_hash(settings.domains_dir, deps.profile.domain),
            "models": run.models,
            "snapshot_set": sorted({c.content_hash for c in state.evidence.values()}),
            "evidence": serialize_evidence(state.evidence),
            **tail_manifest_fields(state),
            "stats": {
                "polled_new": state.polled_new,
                "ingested": state.ingested,
                "stories": state.story_count,
                "cited_sids": state.cited_sids,
                "uncited_sentences": state.uncited_sentences,
                "invalid_citations_stripped": state.invalid_citations_stripped,
            },
            "output": str(brief_path),
            "generated_at": generated_at,
        }
        manifest_path = out_dir / f"{run_id}.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        run.status = "done"
        run.finished_at = utcnow()
        run.manifest_path = str(manifest_path)
        run.output_path = str(brief_path)
        deps.session.commit()
        return run
    except Exception:
        # A failed node may have left the shared session rollback-required
        # (nodes add/commit on it); without the rollback the status write
        # raises PendingRollbackError, masking the real error and leaving the
        # run 'running' forever. Bookkeeping is best-effort: never mask.
        try:
            deps.session.rollback()
            run.status = "failed"
            run.finished_at = utcnow()
            deps.session.commit()
        except Exception:
            deps.session.rollback()
        raise


def replay_brief(deps: BriefDeps, manifest_path: Path) -> tuple[Path, dict]:
    """Back-compat alias — see argus.graphs.replay.replay_run (SS11.3)."""
    from argus.graphs.replay import replay_run

    return replay_run(deps, manifest_path)
