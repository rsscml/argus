"""Background jobs for the web layer.

Everything that touches the pipeline (poll, ingest, brief, research, replay,
golden) runs here rather than inside a request: runs can take minutes, and the
UI polls job status instead of holding a connection open.

Why max_workers=1: in the default dev deployment all jobs share one SQLite
database and one *embedded* Qdrant store. Serializing jobs makes concurrent-
writer bugs structurally impossible and matches the architecture's sizing
("one process is fine at this scale", SS10). Scheduled runs and button-presses
go through the same queue, so they can never trample each other either.
"""
from __future__ import annotations

import secrets
import threading
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from argus.util import utcnow

JobFn = Callable[["Job"], dict]


@dataclass
class Job:
    id: str
    kind: str            # brief | research | poll | ingest | replay | golden
    title: str
    params: dict = field(default_factory=dict)
    status: str = "queued"        # queued | running | done | failed
    note: str = ""                # one-line progress note, set by the job fn
    created_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    result: dict | None = None
    error: str | None = None
    trace: str | None = None

    def to_dict(self, *, with_trace: bool = False) -> dict:
        d = {
            "id": self.id, "kind": self.kind, "title": self.title,
            "params": self.params, "status": self.status, "note": self.note,
            "created_at": self.created_at, "started_at": self.started_at,
            "finished_at": self.finished_at, "result": self.result,
            "error": self.error,
        }
        if with_trace:
            d["trace"] = self.trace
        return d


class JobManager:
    def __init__(self, max_history: int = 300) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="argus-job"
        )
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._max_history = max_history

    # -- API -----------------------------------------------------------------
    def submit(self, kind: str, title: str, params: dict, fn: JobFn) -> Job:
        job = Job(
            id=f"{utcnow():%H%M%S}-{secrets.token_hex(3)}",
            kind=kind, title=title, params=params,
            created_at=utcnow().isoformat(timespec="seconds"),
        )
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > self._max_history:
                oldest_id, oldest = next(iter(self._jobs.items()))
                if oldest.status in ("queued", "running"):
                    break  # never evict live jobs
                self._jobs.pop(oldest_id)
        self._executor.submit(self._run, job, fn)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())[-limit:][::-1]

    def active(self, kind: str | None = None, domain: str | None = None) -> list[Job]:
        with self._lock:
            live = [j for j in self._jobs.values() if j.status in ("queued", "running")]
        if kind:
            live = [j for j in live if j.kind == kind]
        if domain:
            live = [j for j in live if j.params.get("domain") == domain]
        return live

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # -- worker ----------------------------------------------------------------
    def _run(self, job: Job, fn: JobFn) -> None:
        job.status = "running"
        job.started_at = utcnow().isoformat(timespec="seconds")
        try:
            job.result = fn(job) or {}
            job.status = "done"
        except Exception as exc:  # surfaced to the UI, never swallowed
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.trace = traceback.format_exc()
        finally:
            job.finished_at = utcnow().isoformat(timespec="seconds")
