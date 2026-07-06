"""Small shared utilities.

Datetime convention: all timestamps are stored and compared as *naive UTC*
(converted at the boundary). This keeps SQLite (dev) and Postgres (prod, SS10)
behavior identical.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utcnow() -> datetime:
    """Current time as naive UTC."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def as_naive_utc(dt: datetime | None) -> datetime | None:
    """Normalize any datetime to naive UTC."""
    if dt is None or dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def dot_get(obj: Any, path: str, default: Any = None) -> Any:
    """Traverse nested dicts/lists with a dotted path ('data.items.0.title')."""
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return default
    return cur
