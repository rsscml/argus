"""Relational schema for M1 (architecture SS7.2, SS9, SS12.5).

SQLite for dev, Postgres in production (SS10) — same SQLAlchemy models.
Datetimes are naive UTC throughout (see argus.util).

M2+ columns (cluster_id, extraction_status) exist now so the snapshot table
never needs a breaking migration when ingestion lands.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Integer, LargeBinary, String, Text, create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from argus.util import utcnow


class Base(DeclarativeBase):
    pass


class SnapshotRow(Base):
    """One immutable snapshot of fetched material — the unit of citation (SS7.2)."""

    __tablename__ = "snapshots"

    content_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_id: Mapped[str] = mapped_column(String(64), index=True)
    url: Mapped[str] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    media_type: Mapped[str] = mapped_column(String(64), default="text/html")
    published_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    blob_uri: Mapped[str] = mapped_column(Text)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    # Reserved for M2 (SS7.3):
    cluster_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    extraction_status: Mapped[str] = mapped_column(String(16), default="pending")


class WatermarkRow(Base):
    """Per-source fetch watermark: 'since' + conditional-GET validators (SS7.1)."""

    __tablename__ = "watermarks"

    source_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    last_polled_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    etag: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_modified: Mapped[str | None] = mapped_column(Text, nullable=True)


class FetchEventRow(Base):
    """One poll attempt against one source — the health record (SS12.5).

    Statuses: ok | not_modified | error.
    """

    __tablename__ = "fetch_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[str] = mapped_column(String(64), index=True)
    domain: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(16))
    items_seen: Mapped[int] = mapped_column(Integer, default=0)
    snapshots_new: Mapped[int] = mapped_column(Integer, default=0)
    snapshots_dup: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    registry_commit: Mapped[str | None] = mapped_column(String(80), nullable=True)


class DocSignatureRow(Base):
    """MinHash signature per document for cross-run syndication clustering (SS7.3)."""

    __tablename__ = "doc_signatures"

    content_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    minhash: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class RunRow(Base):
    """One graph run: the manifest backbone joins checkpoints by run_id (SS9, AD-6)."""

    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String(48), primary_key=True)
    domain: Mapped[str] = mapped_column(String(64), index=True)
    workload: Mapped[str] = mapped_column(String(32))  # daily_brief | deep_research
    status: Mapped[str] = mapped_column(String(16), default="running")
    registry_commit: Mapped[str | None] = mapped_column(String(80), nullable=True)
    models: Mapped[dict] = mapped_column(JSON, default=dict)  # deployments + api_version (SS12.1)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    manifest_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_path: Mapped[str | None] = mapped_column(Text, nullable=True)


class CandidateRow(Base):
    """Off-registry source proposal awaiting human review (SS6.2)."""

    __tablename__ = "candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(Text)
    netloc: Mapped[str] = mapped_column(String(255), index=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    run_id: Mapped[str | None] = mapped_column(String(48), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|approved|rejected
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


def make_engine(db_url: str) -> Engine:
    engine = create_engine(db_url, future=True)
    if db_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn, _):  # pragma: no cover - trivial
            dbapi_conn.execute("PRAGMA foreign_keys=ON")
    return engine


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
