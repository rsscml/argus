"""Web-layer wiring: same construction as argus.wiring, one difference.

argus.wiring builds a *new* QdrantClient per call, which is correct for a
one-shot CLI process but fatal in a server: embedded local mode holds a file
lock, so the second construction in the same process raises. These builders
take the process's shared client (state.AppState.qdrant()) and are otherwise
line-for-line the core wiring — the engine modules stay untouched.
"""
from __future__ import annotations

from qdrant_client import QdrantClient
from sqlalchemy.orm import Session

from argus.config.profile import load_profile
from argus.config.registry import load_registry, registry_commit
from argus.governance.discovery import NullDiscovery
from argus.graphs.daily import BriefDeps
from argus.graphs.research import ResearchDeps, make_planner_stack
from argus.ingest.embed import make_embedder
from argus.ingest.entities import EntityTagger, QueryExpander, load_all_dictionaries
from argus.ingest.pipeline import IngestDeps
from argus.ingest.sparse import Bm25SparseEncoder
from argus.settings import Settings
from argus.snapshots.blob import BlobStore
from argus.wiring import make_claim_stack
from argus.synthesis.synthesizer import make_synthesizer


def build_ingest_deps(
    settings: Settings, session: Session, client: QdrantClient
) -> IngestDeps:
    registry = load_registry(settings.registry_path)
    return IngestDeps(
        session=session,
        blob=BlobStore(settings.blob_root),
        client=client,
        collection=settings.collection,
        embedder=make_embedder(settings.embedder, settings.hashing_dim),
        sparse=Bm25SparseEncoder(),
        tagger=EntityTagger(load_all_dictionaries(settings.domains_dir)),
        registry=registry,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def build_brief_deps(
    settings: Settings, session: Session, domain: str, client: QdrantClient
) -> BriefDeps:
    registry = load_registry(settings.registry_path)
    profile = load_profile(settings.domains_dir, domain)
    ingest_deps = build_ingest_deps(settings, session, client)
    extractor, matcher, verifier = make_claim_stack(settings)
    return BriefDeps(
        settings=settings,
        session=session,
        registry=registry,
        profile=profile,
        client=client,
        ingest_deps=ingest_deps,
        synthesizer=make_synthesizer(settings.llm),
        extractor=extractor,
        matcher=matcher,
        verifier=verifier,
        registry_commit=registry_commit(settings.registry_path),
    )


def build_research_deps(
    settings: Settings, session: Session, domain: str, client: QdrantClient
) -> ResearchDeps:
    registry = load_registry(settings.registry_path)
    profile = load_profile(settings.domains_dir, domain)
    dictionaries = load_all_dictionaries(settings.domains_dir)
    ingest_deps = build_ingest_deps(settings, session, client)
    extractor, matcher, verifier = make_claim_stack(settings)
    planner, reflector = make_planner_stack(settings)
    return ResearchDeps(
        settings=settings, session=session, registry=registry, profile=profile,
        client=client, ingest_deps=ingest_deps,
        expander=QueryExpander(dictionaries),
        synthesizer=make_synthesizer(settings.llm), extractor=extractor,
        matcher=matcher, verifier=verifier, planner=planner, reflector=reflector,
        discovery=NullDiscovery(), registry_commit=registry_commit(settings.registry_path),
    )
