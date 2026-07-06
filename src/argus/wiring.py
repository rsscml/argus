"""Dependency wiring shared by the CLI and schedulers.

Keeps construction in one place so graphs receive explicit deps (testable)
while operators get a one-liner.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from argus.config.profile import load_profile
from argus.config.registry import load_registry, registry_commit
from argus.governance.discovery import NullDiscovery
from argus.graphs.daily import BriefDeps
from argus.graphs.research import ResearchDeps, make_planner_stack
from argus.index.qdrant import make_client
from argus.ingest.embed import make_embedder
from argus.ingest.entities import EntityTagger, QueryExpander, load_all_dictionaries
from argus.ingest.pipeline import IngestDeps
from argus.ingest.sparse import Bm25SparseEncoder
from argus.settings import Settings
from argus.snapshots.blob import BlobStore
from argus.claims.engine import HeuristicEventMatcher, LlmEventMatcher
from argus.claims.extract import LlmClaimExtractor, StubClaimExtractor
from argus.synthesis.synthesizer import make_synthesizer
from argus.synthesis.verify import HeuristicVerifier, LlmVerifier


def make_claim_stack(settings: Settings):
    """(extractor, matcher, verifier) per settings.llm — Azure in prod, deterministic offline."""
    if settings.llm == "azure":
        return (LlmClaimExtractor(), LlmEventMatcher(), LlmVerifier())
    return (
        StubClaimExtractor(),
        HeuristicEventMatcher(settings.event_match_jaccard),
        HeuristicVerifier(),
    )


def build_brief_deps(settings: Settings, session: Session, domain: str) -> BriefDeps:
    registry = load_registry(settings.registry_path)
    profile = load_profile(settings.domains_dir, domain)
    client = make_client(settings)
    embedder = make_embedder(settings.embedder, settings.hashing_dim)
    ingest_deps = IngestDeps(
        session=session,
        blob=BlobStore(settings.blob_root),
        client=client,
        collection=settings.collection,
        embedder=embedder,
        sparse=Bm25SparseEncoder(),
        tagger=EntityTagger(load_all_dictionaries(settings.domains_dir)),
        registry=registry,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
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


def build_research_deps(settings: Settings, session: Session, domain: str) -> ResearchDeps:
    registry = load_registry(settings.registry_path)
    profile = load_profile(settings.domains_dir, domain)
    client = make_client(settings)
    embedder = make_embedder(settings.embedder, settings.hashing_dim)
    dictionaries = load_all_dictionaries(settings.domains_dir)
    ingest_deps = IngestDeps(
        session=session, blob=BlobStore(settings.blob_root), client=client,
        collection=settings.collection, embedder=embedder, sparse=Bm25SparseEncoder(),
        tagger=EntityTagger(dictionaries), registry=registry,
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
    )
    extractor, matcher, verifier = make_claim_stack(settings)
    planner, reflector = make_planner_stack(settings)
    return ResearchDeps(
        settings=settings, session=session, registry=registry, profile=profile,
        client=client, ingest_deps=ingest_deps, expander=QueryExpander(dictionaries),
        synthesizer=make_synthesizer(settings.llm), extractor=extractor,
        matcher=matcher, verifier=verifier, planner=planner, reflector=reflector,
        discovery=NullDiscovery(), registry_commit=registry_commit(settings.registry_path),
    )
