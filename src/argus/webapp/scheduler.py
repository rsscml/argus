"""Periodic runs (architecture SS10: cron / APScheduler).

Each domain profile carries its own `schedule.brief_cron`; this service turns
those into APScheduler cron jobs that submit a brief through the same
JobManager as button presses — one queue, no overlap. Scheduling is OFF per
domain until the operator turns it on (no surprise runs on a fresh install);
toggles and the last scheduled outcome persist in data/webapp/scheduler.json.

If a brief for a domain is still queued/running when its cron fires, the tick
is recorded as skipped rather than stacking a duplicate run.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from argus.config.profile import list_domains, load_profile
from argus.settings import get_settings
from argus.util import utcnow
from argus.webapp import actions
from argus.webapp.state import AppState


def _state_path() -> Path:
    root = Path(os.environ.get("ARGUS_WEBAPP_STATE", "data/webapp"))
    return root / "scheduler.json"


def validate_cron(expr: str) -> None:
    """Raise ValueError with a friendly message on a bad cron expression."""
    try:
        CronTrigger.from_crontab(expr)
    except ValueError as exc:
        raise ValueError(f"invalid cron expression {expr!r}: {exc}") from exc


class SchedulerService:
    def __init__(self, app: AppState) -> None:
        self.app = app
        self._scheduler = BackgroundScheduler()
        self._lock = threading.Lock()
        self._state = self._load_state()

    # -- persistence -----------------------------------------------------------
    def _load_state(self) -> dict:
        path = _state_path()
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
        return {"domains": {}, "last": {}}

    def _save_state(self) -> None:
        path = _state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, indent=2))
        os.replace(tmp, path)

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        self._scheduler.start()
        self.refresh()

    def shutdown(self) -> None:
        try:
            self._scheduler.shutdown(wait=False)
        except Exception:
            pass

    # -- scheduling ------------------------------------------------------------
    def refresh(self) -> list[dict]:
        """Rebuild jobs from profiles + enabled flags; report per-domain state."""
        with self._lock:
            settings = get_settings()
            report: list[dict] = []
            wanted: set[str] = set()
            for domain in list_domains(settings.domains_dir):
                entry: dict = {"domain": domain, "cron": None, "enabled": False,
                               "next_run": None, "error": None,
                               "last": self._state["last"].get(domain)}
                try:
                    profile = load_profile(settings.domains_dir, domain)
                    entry["cron"] = profile.schedule.brief_cron
                except Exception as exc:
                    entry["error"] = f"profile failed to load: {exc}"
                    report.append(entry)
                    continue

                enabled = bool(
                    self._state["domains"].get(domain, {}).get("enabled", False)
                )
                entry["enabled"] = enabled
                job_id = f"brief:{domain}"
                if enabled and entry["cron"]:
                    try:
                        trigger = CronTrigger.from_crontab(entry["cron"])
                    except ValueError as exc:
                        entry["error"] = f"invalid cron: {exc}"
                        report.append(entry)
                        continue
                    job = self._scheduler.get_job(job_id)
                    if job is None:
                        job = self._scheduler.add_job(
                            self._fire, trigger=trigger, id=job_id,
                            args=[domain], replace_existing=True,
                            misfire_grace_time=3600, coalesce=True,
                        )
                    else:
                        job.reschedule(trigger=trigger)
                        job = self._scheduler.get_job(job_id)
                    wanted.add(job_id)
                    if job.next_run_time:
                        entry["next_run"] = job.next_run_time.isoformat(
                            timespec="seconds")
                elif enabled and not entry["cron"]:
                    entry["error"] = ("no schedule set — add a cron expression "
                                      "to this domain's profile")
                report.append(entry)

            for job in self._scheduler.get_jobs():
                if job.id not in wanted:
                    self._scheduler.remove_job(job.id)
            return report

    def set_enabled(self, domain: str, enabled: bool) -> list[dict]:
        settings = get_settings()
        if domain not in list_domains(settings.domains_dir):
            raise KeyError(domain)
        with self._lock:
            self._state["domains"].setdefault(domain, {})["enabled"] = enabled
            self._save_state()
        return self.refresh()

    # -- trigger ---------------------------------------------------------------
    def _fire(self, domain: str) -> None:
        fired_at = utcnow().isoformat(timespec="seconds")
        if self.app.jobs.active(kind="brief", domain=domain):
            self._record_last(domain, {
                "fired_at": fired_at, "outcome": "skipped",
                "detail": "previous brief for this domain still running",
            })
            return
        job = self.app.jobs.submit(
            "brief", f"Scheduled brief — {domain}",
            {"domain": domain, "triggered_by": "schedule"},
            actions.brief_job(self.app, domain, None, triggered_by="schedule"),
        )
        self._record_last(domain, {"fired_at": fired_at, "outcome": "submitted",
                                   "job_id": job.id})

    def record_job_outcome(self, domain: str, job_dict: dict) -> None:
        """Called by the API layer when it notices a scheduled job finished."""
        self._record_last(domain, {
            "fired_at": job_dict.get("created_at"),
            "outcome": job_dict.get("status"),
            "job_id": job_dict.get("id"),
            "run_id": (job_dict.get("result") or {}).get("run_id"),
            "detail": job_dict.get("error"),
        })

    def _record_last(self, domain: str, payload: dict) -> None:
        with self._lock:
            self._state["last"][domain] = payload
            self._save_state()
