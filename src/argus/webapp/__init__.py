"""Argus web layer — a FastAPI app over the existing engine.

Design constraints, mirroring the architecture document:
- This package is a *layer on top*: it calls the same entry points as the CLI
  (poll_domain, run_ingest, run_daily_brief, run_research, replay_run,
  health_report, run_golden, candidate queue) and never re-implements them.
- Registry and profiles remain the only place trust/config lives (AD-2/AD-5);
  the webapp edits those YAML files with validation, timestamped backups, and
  best-effort git commits so the "reviewed commit" trail survives UI edits.
- One worker thread executes all runs, because the dev deployment shares a
  SQLite database and an embedded (file-locked) Qdrant instance.
"""
