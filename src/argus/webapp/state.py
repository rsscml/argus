"""Process-wide state for the web layer.

Three shared things live for the whole server process:

1. **Settings overrides** — a `KEY=value` file the settings page writes.
   Values are applied to ``os.environ`` at startup and on every save, and
   ``argus.settings.get_settings()`` is constructed fresh per request/job, so
   most changes take effect on the *next* action without a restart. Keys that
   are bound at process start (data dir, DB URL, Qdrant URL) are flagged as
   restart-required by the API.
2. **SQLAlchemy engine** — one engine, NullPool + a generous SQLite timeout so
   API threads and the single job worker can share the dev database safely.
3. **Qdrant client** — exactly one per process. Embedded local mode
   (``QdrantClient(path=...)``) holds a file lock; constructing a second
   client against the same path fails, which is why the core wiring cannot be
   called concurrently and why jobs are serialized (see jobs.py).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from argus.settings import Settings, get_settings
from argus.snapshots.db import init_db

# --------------------------------------------------------------------------
# settings overrides (the settings page's persistence)
# --------------------------------------------------------------------------

def overrides_path() -> Path:
    root = Path(os.environ.get("ARGUS_WEBAPP_STATE", "data/webapp"))
    return root / "overrides.env"


def read_overrides() -> dict[str, str]:
    path = overrides_path()
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def write_overrides(values: dict[str, str]) -> None:
    path = overrides_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Argus web settings overrides — managed by the settings page.",
        "# Values here are applied to the environment at server start and on",
        "# every save. Delete a line (or the file) to fall back to the shell",
        "# environment / built-in defaults. Keep this file private: it may",
        "# hold the Azure OpenAI key (use a real secret manager in prod).",
    ]
    lines += [f"{k}={v}" for k, v in sorted(values.items())]
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, path)
    try:  # best-effort: the file may hold a secret
        os.chmod(path, 0o600)
    except OSError:
        pass


def apply_overrides() -> dict[str, str]:
    """Load the overrides file into os.environ (UI-saved values win)."""
    values = read_overrides()
    for key, value in values.items():
        os.environ[key] = value
    return values


# --------------------------------------------------------------------------
# shared engine / sessions
# --------------------------------------------------------------------------

class AppState:
    def __init__(self) -> None:
        apply_overrides()
        settings = get_settings()
        self.started_db_url = settings.resolved_db_url
        self.started_data_dir = str(settings.data_dir)
        self.started_qdrant_url = settings.qdrant_url
        self.engine: Engine = _make_web_engine(self.started_db_url)
        init_db(self.engine)
        self._session_factory = sessionmaker(
            bind=self.engine, expire_on_commit=False, future=True
        )
        self._qdrant = None
        self._qdrant_lock = threading.Lock()
        # populated in app.create_app():
        self.jobs = None       # jobs.JobManager
        self.scheduler = None  # scheduler.SchedulerService

    # -- sessions ----------------------------------------------------------
    def session(self) -> Session:
        return self._session_factory()

    # -- qdrant (one client per process; embedded mode is file-locked) ------
    def qdrant(self):
        with self._qdrant_lock:
            if self._qdrant is None:
                from argus.index.qdrant import make_client

                self._qdrant = make_client(get_settings())
            return self._qdrant

    # -- restart detection ---------------------------------------------------
    def restart_required(self) -> list[str]:
        """Boot-bound settings whose current value differs from boot value."""
        settings = get_settings()
        stale = []
        if settings.resolved_db_url != self.started_db_url:
            stale.append("ARGUS_DB_URL")
        if str(settings.data_dir) != self.started_data_dir:
            stale.append("ARGUS_DATA_DIR")
        if settings.qdrant_url != self.started_qdrant_url:
            stale.append("ARGUS_QDRANT_URL")
        return stale


def _make_web_engine(db_url: str) -> Engine:
    """Like snapshots.db.make_engine, but safe for a multithreaded server.

    NullPool means a connection never migrates between threads via the pool;
    check_same_thread=False + a 30s busy timeout let the API thread and the
    job worker share the SQLite file without spurious lock errors.
    """
    kwargs: dict = {"future": True, "poolclass": NullPool}
    if db_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    engine = create_engine(db_url, **kwargs)
    if db_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn, _):  # pragma: no cover - trivial
            dbapi_conn.execute("PRAGMA foreign_keys=ON")
    return engine


def fresh_settings() -> Settings:
    """Settings snapshot for one request/job (env may have changed)."""
    return get_settings()
