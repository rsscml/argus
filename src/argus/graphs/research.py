"""Deep research graph (architecture SS8.2 right column + shared tail).

plan -> discover -> retrieve <-> reflect (bounded loop) -> assemble ->
[shared tail: corroborate -> synthesize -> verify] -> report + manifest.

Bounded loops only (SS8.1): the retrieve-reflect cycle is capped by
profile.retrieval.max_iterations. Gated discovery (SS7.1/SS6.2, invariant 1):
off-registry hits become candidate-queue proposals and are NEVER retrieved,
snapshotted, or cited in the current run.
"""
from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from langgraph.graph import END, START, StateGraph
from pydantic import Field
from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from argus.claims.engine import EventMatcher
from argus.claims.extract import ClaimExtractor
from argus.config.profile import DomainProfile
from argus.config.registry import Registry
from argus.governance.candidates import propose
from argus.governance.discovery import DiscoveryProvider
from argus.graphs.shared import attach_tail, serialize_evidence, tail_manifest_fields
from argus.graphs.state import TailState
from argus.index.retrieval import RetrievedChunk, retrieve
from argus.ingest.entities import QueryExpander
from argus.ingest.pipeline import IngestDeps
from argus.ingest.sparse import tokenize
from argus.settings import Settings
from argus.snapshots.db import RunRow
from argus.synthesis.render import render_report
from argus.synthesis.synthesizer import Synthesizer
from argus.synthesis.verify import Verifier
from argus.util import utcnow


# ---------- planner / reflector seams ----------


class Planner(Protocol):
    name: str

    def plan(self, question: str, max_sub_questions: int) -> list[str]: ...


class Reflector(Protocol):
    name: str

    def refine(self, sub_question: str) -> str | None: ...


class StubPlanner:
    """Deterministic: the question itself, plus ' and '-split parts."""

    name = "stub-planner"

    def plan(self, question: str, max_sub_questions: int) -> list[str]:
        parts = [p.strip(" ?.") for p in question.split(" and ") if p.strip(" ?.")]
        subs = [question.strip()] if len(parts) <= 1 else parts
        return subs[:max_sub_questions]


class StubReflector:
    """Deterministic: retry with the query's first content tokens (broader
    lexical net); give up (None) if that changes nothing."""

    name = "stub-reflector"

    def refine(self, sub_question: str) -> str | None:
        tokens = tokenize(sub_question)[:4]
        refined = " ".join(tokens)
        return refined if refined and refined != sub_question.lower().strip(" ?.") else None


_PLAN_PROMPT = """Decompose this research question into at most {n} focused,
independently-searchable sub-questions (one per line, no numbering, no prose):

{question}"""


class LlmPlanner:
    name = "azure-planner"

    def __init__(self, chat_model=None) -> None:
        if chat_model is None:
            from argus.llm.factory import get_chat_model

            chat_model = get_chat_model("synthesis", temperature=0.0)
        self._model = chat_model

    def plan(self, question: str, max_sub_questions: int) -> list[str]:
        response = self._model.invoke(
            _PLAN_PROMPT.format(n=max_sub_questions, question=question)
        )
        content = response.content if hasattr(response, "content") else str(response)
        subs = [line.strip(" -•") for line in str(content).splitlines() if line.strip()]
        return (subs or [question])[:max_sub_questions]


class LlmReflector:
    name = "azure-reflector"

    def __init__(self, chat_model=None) -> None:
        if chat_model is None:
            from argus.llm.factory import get_chat_model

            chat_model = get_chat_model("utility", temperature=0.0)
        self._model = chat_model

    def refine(self, sub_question: str) -> str | None:
        response = self._model.invoke(
            "This search query returned no results in a curated news index. "
            "Rewrite it as a broader keyword query (reply with the query only, "
            f"or NONE to give up): {sub_question}"
        )
        content = str(getattr(response, "content", response)).strip()
        return None if not content or content.upper() == "NONE" else content


# ---------- state & deps ----------


class ResearchState(TailState):
    question: str = ""
    sub_questions: list[str] = Field(default_factory=list)
    pending_queries: list[str] = Field(default_factory=list)
    hits: dict[str, int] = Field(default_factory=dict)
    collected: dict[str, RetrievedChunk] = Field(default_factory=dict)
    iterations: int = 0
    candidates_queued: int = 0


@dataclass
class ResearchDeps:
    settings: Settings
    session: Session
    registry: Registry
    profile: DomainProfile
    client: QdrantClient
    ingest_deps: IngestDeps
    expander: QueryExpander
    synthesizer: Synthesizer
    extractor: ClaimExtractor
    matcher: EventMatcher
    verifier: Verifier
    planner: Planner
    reflector: Reflector
    discovery: DiscoveryProvider
    registry_commit: str | None


# ---------- graph ----------


def build_research_graph(deps: ResearchDeps, checkpointer=None):
    profile = deps.profile

    def plan(state: ResearchState) -> dict:
        subs = deps.planner.plan(state.question, deps.settings.research_sub_questions_max)
        return {"sub_questions": subs, "pending_queries": list(subs)}

    def discover(state: ResearchState) -> dict:
        queued = 0
        for query in state.sub_questions:
            for hit in deps.discovery.search(query):
                if propose(deps.session, deps.registry, url=hit.url, title=hit.title,
                           rationale=f"hit for {query!r}: {hit.snippet[:120]}",
                           run_id=state.run_id):
                    queued += 1
        # invariant 1: hits are proposals only; nothing here enters evidence.
        return {"candidates_queued": queued}

    def retrieve_round(state: ResearchState) -> dict:
        collected = dict(state.collected)
        hits = dict(state.hits)
        for query in state.pending_queries:
            results = retrieve(
                query, registry=deps.registry, profile=profile, client=deps.client,
                collection=deps.ingest_deps.collection,
                embedder=deps.ingest_deps.embedder, sparse=deps.ingest_deps.sparse,
                expander=deps.expander, top_k=profile.retrieval.top_k,
            )
            # Coverage gate: dense ANN always returns nearest neighbors, so raw
            # result count is meaningless. A result covers its query only if it
            # shares >=1 content token with the entity-expanded query; zero-
            # anchor hits are discarded — in a curated index they are noise far
            # more often than paraphrase (revisit for cross-lingual recall, R3).
            expanded = deps.expander.expand(query) if deps.expander else query
            query_tokens = set(tokenize(expanded))
            covering = [
                c for c in results if query_tokens & set(tokenize(c.text))
            ]
            hits[query] = len(covering)
            for chunk in covering:
                kept = collected.get(chunk.chunk_id)
                if kept is None or chunk.score > kept.score:
                    collected[chunk.chunk_id] = chunk
        return {"collected": collected, "hits": hits, "pending_queries": []}

    def reflect(state: ResearchState) -> dict:
        uncovered = [q for q, n in state.hits.items() if n == 0]
        if uncovered and state.iterations < profile.retrieval.max_iterations:
            refined = []
            for query in uncovered:
                new_query = deps.reflector.refine(query)
                if new_query and new_query not in state.hits:
                    refined.append(new_query)
            if refined:
                return {"pending_queries": refined, "iterations": state.iterations + 1}
        return {}

    def assemble(state: ResearchState) -> dict:
        ranked = sorted(state.collected.values(), key=lambda c: c.score, reverse=True)
        ranked = ranked[: deps.settings.research_max_evidence]
        evidence = {f"S{i}": chunk for i, chunk in enumerate(ranked, 1)}
        return {"evidence": evidence}

    graph = StateGraph(ResearchState)
    graph.add_node("plan", plan)
    graph.add_node("discover", discover)
    graph.add_node("retrieve", retrieve_round)
    graph.add_node("reflect", reflect)
    graph.add_node("assemble", assemble)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "discover")
    graph.add_edge("discover", "retrieve")
    graph.add_edge("retrieve", "reflect")
    graph.add_conditional_edges(
        "reflect",
        lambda state: "retrieve" if state.pending_queries else "assemble",
        {"retrieve": "retrieve", "assemble": "assemble"},
    )
    attach_tail(graph, deps, after="assemble")  # SS4 invariant 2
    return graph.compile(checkpointer=checkpointer)


def _profile_hash(domains_dir: Path, domain: str) -> str:
    path = domains_dir / domain / "profile.yaml"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else ""


def run_research(
    deps: ResearchDeps, question: str, *, checkpointer=None
) -> RunRow:
    settings = deps.settings
    run_id = f"{utcnow():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
    models = {"synthesizer": settings.llm, "embedder": settings.embedder}
    if settings.llm == "azure" or settings.embedder == "azure":
        from argus.llm.factory import model_fingerprint

        models["azure"] = model_fingerprint("synthesis")

    run = RunRow(run_id=run_id, domain=deps.profile.domain, workload="deep_research",
                 registry_commit=deps.registry_commit, models=models)
    deps.session.add(run)
    deps.session.commit()

    try:
        graph = build_research_graph(deps, checkpointer=checkpointer)
        final = graph.invoke(
            ResearchState(run_id=run_id, domain=deps.profile.domain, question=question),
            config={"configurable": {"thread_id": run_id}},
        )
        state = ResearchState.model_validate(final)

        out_dir = settings.briefs_dir / deps.profile.domain
        out_dir.mkdir(parents=True, exist_ok=True)
        generated_at = utcnow().isoformat(timespec="seconds")
        report_md = render_report(
            profile=deps.profile, domains_dir=settings.domains_dir,
            domain=deps.profile.domain, run_id=run_id, generated_at=generated_at,
            question=question, body_md=state.body_md, evidence=state.evidence,
            registry_commit=deps.registry_commit,
            synthesizer=deps.synthesizer.name, embedder=deps.ingest_deps.embedder.name,
        )
        report_path = out_dir / f"{run_id}.md"
        report_path.write_text(report_md)

        manifest = {
            "run_id": run_id,
            "workload": "deep_research",
            "domain": deps.profile.domain,
            "question": question,
            "sub_questions": state.sub_questions,
            "iterations": state.iterations,
            "hits": state.hits,
            "candidates_queued": state.candidates_queued,
            "registry_commit": deps.registry_commit,
            "profile_hash": _profile_hash(settings.domains_dir, deps.profile.domain),
            "models": models,
            "snapshot_set": sorted({c.content_hash for c in state.evidence.values()}),
            "evidence": serialize_evidence(state.evidence),
            **tail_manifest_fields(state),
            "stats": {
                "cited_sids": state.cited_sids,
                "uncited_sentences": state.uncited_sentences,
                "invalid_citations_stripped": state.invalid_citations_stripped,
            },
            "output": str(report_path),
            "generated_at": generated_at,
        }
        manifest_path = out_dir / f"{run_id}.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        run.status = "done"
        run.finished_at = utcnow()
        run.manifest_path = str(manifest_path)
        run.output_path = str(report_path)
        deps.session.commit()
        return run
    except Exception:
        # See run_daily_brief: rollback first so the status write cannot raise
        # PendingRollbackError; bookkeeping must never mask the original error.
        try:
            deps.session.rollback()
            run.status = "failed"
            run.finished_at = utcnow()
            deps.session.commit()
        except Exception:
            deps.session.rollback()
        raise


def make_planner_stack(settings: Settings) -> tuple[Planner, Reflector]:
    if settings.llm == "azure":
        return LlmPlanner(), LlmReflector()
    return StubPlanner(), StubReflector()
