"""Checkpointer factory (architecture SS8.1, AD-6).

SQLite for dev (default, under the data dir); Postgres in production via
ARGUS_CHECKPOINT_URL=postgresql://... (requires the 'postgres' extra).
Thread id == run_id, so `runs` rows join checkpoint history directly.
"""
from __future__ import annotations

import sqlite3

from argus.settings import Settings


def make_checkpointer(settings: Settings):
    url = settings.checkpoint_url
    if url.startswith("postgres"):
        # NOTE: PostgresSaver.from_conn_string is a @contextmanager — calling it
        # bare returns a _GeneratorContextManager, not a saver (and closes the
        # connection on __exit__). Own the connection for the process lifetime
        # instead, with the kwargs the checkpointer requires (autocommit +
        # dict_row; prepare_threshold=0 keeps pgbouncer-style poolers happy).
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg import Connection
        from psycopg.rows import dict_row

        conn = Connection.connect(
            url, autocommit=True, prepare_threshold=0, row_factory=dict_row
        )
        saver = PostgresSaver(conn)
        saver.setup()  # idempotent migrations
        return saver
    from langgraph.checkpoint.sqlite import SqliteSaver

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.checkpoint_sqlite_path), check_same_thread=False)
    return SqliteSaver(conn)
