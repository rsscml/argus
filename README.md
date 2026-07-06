# Argus — all milestones (M1–M6)

Milestone 1 of the controlled-source research agent (see `docs/architecture.md`,
v0.2). Delivers: the git-versioned **source registry** (§6.1), the pull-first
**fetcher layer** with watermarks and failure isolation (§7.1), the immutable
content-addressed **snapshot store** (§7.2), and the per-source **health
report** (§12.5) — the M1 exit criterion.

M2 adds the **ingestion pipeline as a LangGraph graph** (§7.3, AD-9 v0.3):
extraction (trafilatura), MinHash **syndication clustering**, **entity
tagging** from domain dictionaries, chunking (`langchain-text-splitters`), and
the **Qdrant hybrid index** (dense Azure embeddings + in-repo BM25 sparse,
RRF fusion) with the §7.4 retrieval API (`argus search`). No LlamaIndex.

M3 adds the **daily brief graph** (§8.2): poll → ingest → window retrieval →
story grouping → **grounded, citation-enforced synthesis** (Azure via the
AD-10 factory, or `ARGUS_LLM=stub` for a deterministic extractive brief), with
a **checkpointer** on every run (SQLite dev / Postgres via the `postgres`
extra), a **run manifest** (§12.1), and **evidence-identical replay**
(`argus replay <run_id>`, §11.3). **PDF extraction** is in (pypdf):
text-layer PDFs flow end-to-end; scanned/image-only PDFs surface as `empty`
in ingest stats (OCR tracked as Q3). Try: `argus brief --domain general_news`.

M4 completes the trust core: **claim extraction** (§8.3, event + quantitative),
the **corroboration engine** (§8.4) with syndication/independence-group voice
counting, per-tag tier rules, tolerance-based quantitative matching, and
explicit contradiction surfacing; the **citation verifier loop** (§8.5) with
capped retries and drop-never-ship; **verdict labels in briefs** ("confirmed
by N independent sources", "reported by X, unconfirmed", "CONFLICTING
REPORTS"); and the **golden-set harness**: `argus golden --domain
general_news` (also in CI via pytest). All LLM stages (extraction, event
pairing, entailment, synthesis) run on Azure OpenAI in prod and deterministic
offline implementations with `ARGUS_LLM=stub`.

M5 adds the **deep research graph** (§8.2 right): planner → bounded
retrieve-reflect loop (lexical coverage gate, `max_iterations` cap) →
**the same shared tail** as the daily brief (§4 invariant 2: trust policy runs
in one place — `src/argus/graphs/shared.py`), plus **gated discovery** behind
a provider seam and the **live candidate queue**: `argus research "..."
--domain X`, `argus review [--approve N | --reject N --reason "..."]`.
Approval emits a paste-ready registry stanza (paused, license-unfilled) —
the registry changes only via reviewed git commits (AD-2).

M6 is the G5 acceptance: the **commodities domain** onboarded with **zero
engine changes** — only `registry/sources.yaml` (per-tag tiers: Al Jazeera is
tier 2 for general_news but tier 3 for commodities), `domains/commodities/`
(profile + entity dictionary), `corpora/`, and `tests/golden/commodities/`.
Verified by `tests/test_m6_commodities.py` and `argus golden --domain
commodities`.

## Quickstart

```bash
pip install -e ".[dev]"
argus validate                          # registry + profiles + adapters
argus poll --domain general_news        # fetch -> snapshot for all in-scope sources
argus health                            # per-source health over the last 24h
argus snapshots --limit 10              # most recent snapshots
```

## Configuration (env vars, prefix ARGUS_)

| Var | Default | Meaning |
|-----|---------|---------|
| `ARGUS_DATA_DIR` | `data` | Blob store + default SQLite DB location |
| `ARGUS_DB_URL` | `sqlite:///<data_dir>/argus.db` | SQLAlchemy URL (Postgres in prod, §10) |
| `ARGUS_REGISTRY_PATH` | `registry/sources.yaml` | Trust policy (§6.1) |
| `ARGUS_DOMAINS_DIR` | `domains` | Domain profiles (§5) |

## Invariants enforced here

- Only `active` registry sources within the domain's `registry_scope` are ever
  fetched (G1). There is no other fetch path.
- Snapshots are content-addressed (sha256) and never mutated; a publisher edit
  produces a *new* snapshot (G2/G3, §7.2).
- One failing source never aborts a poll run; it is recorded as a fetch event
  and surfaces in `argus health` (§12.3).

## Web app (settings, runs, and maintenance without the CLI)

A FastAPI layer over the same entry points the CLI uses — nothing is added to
the trust path. Three pages: the **research desk** (`/`) shows the latest
cited brief per domain, runs one on demand, and answers deep-research
questions; **admin** (`/admin`) manages sources, domain profiles, entity
vocabularies, the candidate queue, and settings; **maintenance**
(`/maintenance`) covers source health, cron schedules, run history with
manifests and evidence-identical replay, the snapshot archive, and manual data
actions.

```bash
pip install -e ".[webapp]"
argus-web                      # http://127.0.0.1:8765
```

Registry and profile saves are validated, backed up, and git-committed so the
reviewed-commit trail (AD-2) survives UI edits; settings persist to an env
overrides file, never to config files (SS12.4). Long actions run through a
single-worker job queue (the dev deployment shares SQLite and an embedded,
file-locked Qdrant). Scheduling uses APScheduler over each profile's
`brief_cron`, off by default per domain. Optional password gate via
`ARGUS_WEB_PASSWORD` or the settings page. Full guide: `docs/webapp.md`. Regression suite: `pytest tests/test_webapp.py` (offline, ~10 s).
