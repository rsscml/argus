# Argus web app — operator guide

The web app is a layer over the existing engine: it calls the same entry
points as the CLI and adds nothing to the trust path. Three pages:

| Page | URL | What it's for |
|------|-----|---------------|
| Research desk | `/` | Read the latest brief per domain, run one now, ask deep-research questions, see recent runs and a health summary. |
| Admin | `/admin` | Sources (the trust registry), Domains (profiles + vocabulary), Candidates (proposed sources awaiting review), Settings. |
| Maintenance | `/maintenance` | Source health, schedules, run history with manifests and replay, the snapshot archive, manual data actions, the job log. |

## Start it

```bash
pip install -e ".[webapp]"       # adds fastapi, uvicorn, apscheduler, markdown
argus-web                        # http://127.0.0.1:8765
```

`ARGUS_WEB_HOST` / `ARGUS_WEB_PORT` change the bind address. To require a
password, set one under Admin → Settings → Web access (applies immediately),
or export `ARGUS_WEB_PASSWORD` before starting. This gate is meant for a
trusted machine or LAN; put a reverse proxy with TLS in front of anything
exposed further.

**Try it with no Azure account:** set Engine → Language model to `stub` and
Embeddings to `hashing` in Settings (or export `ARGUS_LLM=stub
ARGUS_EMBEDDER=hashing`). Everything runs offline with a deterministic
extractive synthesizer — the full pipeline, real verdict labels, real
citations. Switch both to `azure` and fill in the Azure section when ready.

Run the server from the repository root: registry, domain, and local-corpus
paths are resolved relative to the working directory, same as the CLI.

## The provenance strip

The mono line under the masthead is the app's certification stamp, on every
page: the registry commit in force, synthesizer and embedder mode, snapshot
count, and whether the job queue is busy. If it shows `·dirty`, the registry
file has changes not yet committed; if it shows `restart required`, a
storage-level setting changed and needs a server restart.

## How edits are saved

Registry and profile edits are the product's trust policy, so every save is:
validated (Pydantic) → backed up (`registry/.backups/`, timestamped) →
written atomically → re-loaded as proof → committed to git as `argus-web`.
The toast shows the commit hash. Outside a git repository everything still
works; saves are just reported as unversioned. Hand-written YAML comments are
not preserved by UI saves — the backup keeps the pre-edit text.

Settings are different by design (secrets never live in config files): the
settings page writes `data/webapp/overrides.env` (mode 0600) and applies
values to the running server. Values saved there override your shell
environment. Most settings apply to the *next* run; the data directory,
database URL, and Qdrant URL are bound at startup and flagged as
restart-required.

## Schedules

Each domain's profile carries a cron expression (`schedule.brief_cron`, edited
in Admin → Domains). Maintenance → Schedules turns each domain's schedule on
or off — everything is **off** on a fresh install, so nothing runs until you
say so. Scheduled and manual runs share one queue: they can never overlap, and
a tick that fires while the previous brief is still running is skipped and
recorded.

## Runs, jobs, and the queue

Briefs, research, polls, ingests, replays, and golden checks run as background
jobs, one at a time (the dev deployment shares one SQLite database and one
embedded, file-locked Qdrant index — serializing jobs makes concurrent-writer
bugs impossible and matches the architecture's single-worker sizing). Watch
progress in the tray at the bottom right or in Maintenance → Jobs; finished
runs land in Maintenance → Runs with their manifest and a Replay button.

## Candidate sources

Deep research may propose off-registry outlets it noticed; they are never
fetched or cited. Approving one from Admin → Candidates can add it to the
registry immediately — but always **paused**, with an `UNREVIEWED` license
placeholder and a filler endpoint. Fill in the real feed URL and the license
terms in Admin → Sources, then activate. Paused sources are never polled, so
nothing enters the trust path before you've reviewed it.

## When something looks wrong

- **A source stopped yielding:** Maintenance → Health. "Failing" = 3+
  consecutive errors; a "new" count stuck at zero usually means the feed moved
  — check its endpoint in Admin → Sources. One bad source never blocks a run.
- **A brief said something surprising:** open the run in Maintenance → Runs →
  Manifest to see exactly which snapshots and verdicts produced it, or Replay
  it against the frozen evidence.
- **You hand-edited YAML:** Maintenance → Data actions → Validate checks the
  registry, every profile, every adapter name, and every cron expression.
