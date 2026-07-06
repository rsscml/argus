"""Persistence for the things the admin page edits.

The registry and domain profiles ARE the product's trust policy (AD-2, G5),
so UI edits follow a strict sequence:

    validate (Pydantic) → timestamped backup → atomic write →
    re-load round-trip → best-effort `git commit`

If the re-load fails the backup is restored, so a bad write can never leave
the system unloadable. Git commits keep the "registry changes are reviewed
commits" property meaningful for UI-driven changes: every save is attributable
and diffable. YAML comments in hand-edited files are not preserved (noted in
the UI); backups keep the pre-edit text.

Settings are different: they are environment-driven by design (SS12.4), so the
settings page persists a `KEY=value` overrides file (state.py) instead of
touching any config file. SETTING_FIELDS below is the single catalog of what
the page exposes.
"""
from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

from argus.config.profile import DomainProfile, load_profile
from argus.config.registry import Registry, Source, load_registry, registry_commit

# --------------------------------------------------------------------------
# git helpers
# --------------------------------------------------------------------------

def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                          text=True, timeout=15)


def git_commit(path: Path, message: str) -> dict:
    """Best-effort commit of one file. Never raises; reports what happened."""
    try:
        cwd = path.resolve().parent
        top = _git(["rev-parse", "--show-toplevel"], cwd)
        if top.returncode != 0:
            return {"committed": False,
                    "detail": "not a git repository — changes saved but unversioned"}
        add = _git(["add", "--", str(path.resolve())], cwd)
        if add.returncode != 0:
            return {"committed": False, "detail": add.stderr.strip()[:200]}
        commit = _git(
            ["-c", "user.name=argus-web", "-c", "user.email=argus-web@local",
             "commit", "-m", message, "--", str(path.resolve())],
            cwd,
        )
        if commit.returncode != 0:
            out = (commit.stdout + commit.stderr).strip()
            if "nothing to commit" in out:
                return {"committed": False, "detail": "no changes to commit"}
            return {"committed": False, "detail": out[:200]}
        sha = _git(["rev-parse", "--short", "HEAD"], cwd).stdout.strip()
        return {"committed": True, "commit": sha, "detail": f"committed {sha}"}
    except Exception as exc:  # pragma: no cover - defensive
        return {"committed": False, "detail": f"{type(exc).__name__}: {exc}"}


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = path.parent / ".backups"
    backup_dir.mkdir(exist_ok=True)
    dest = backup_dir / f"{path.stem}.{stamp}{path.suffix}"
    shutil.copy2(path, dest)
    return dest


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

_REGISTRY_HEADER = (
    "# Argus source registry — trust policy (architecture SS6.1).\n"
    "# Managed via the Argus admin page; every save is validated, backed up\n"
    "# to registry/.backups/, and committed so the trust trail stays auditable.\n"
)


def _source_to_yaml_dict(source: Source) -> dict:
    d = source.model_dump(mode="json", exclude_none=True)
    # the model validator defaults independence_group to the source id;
    # keep the file free of that noise so real syndication families stand out.
    if d.get("independence_group") == source.id:
        d.pop("independence_group", None)
    if not d.get("fetch", {}).get("options"):
        d.get("fetch", {}).pop("options", None)
    if d.get("fetch", {}).get("fetch_full_text") is False:
        d.get("fetch", {}).pop("fetch_full_text", None)
    return d


def dump_registry(registry: Registry) -> str:
    body = yaml.safe_dump(
        {"sources": [_source_to_yaml_dict(s) for s in registry.sources]},
        sort_keys=False, allow_unicode=True, default_flow_style=False, width=100,
    )
    return _REGISTRY_HEADER + body


def save_registry(path: Path, registry: Registry, message: str) -> dict:
    Registry.model_validate(registry.model_dump())  # belt and braces
    backup = _backup(path)
    _atomic_write(path, dump_registry(registry))
    try:
        load_registry(path)  # round-trip proof
    except Exception:
        if backup is not None:
            shutil.copy2(backup, path)
        raise
    result = git_commit(path, message)
    result["backup"] = str(backup) if backup else None
    result["registry_commit"] = registry_commit(path)
    return result


def upsert_source(path: Path, source: Source, message: str) -> dict:
    registry = load_registry(path)
    sources = [s for s in registry.sources if s.id != source.id]
    existed = len(sources) != len(registry.sources)
    sources.append(source) if not existed else sources.insert(
        next(i for i, s in enumerate(registry.sources) if s.id == source.id), source
    )
    registry = Registry(sources=sources)
    out = save_registry(path, registry, message)
    out["updated" if existed else "created"] = source.id
    return out


# --------------------------------------------------------------------------
# domain profiles
# --------------------------------------------------------------------------

_PROFILE_HEADER = (
    "# Domain profile (architecture SS5.1) — everything that makes this a\n"
    "# domain lives here; the engine is domain-agnostic (G5). Managed via the\n"
    "# Argus admin page; saves are validated, backed up, and committed.\n"
)


def dump_profile(profile: DomainProfile) -> str:
    d = profile.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
    d = {"domain": profile.domain, "description": profile.description,
         **{k: v for k, v in d.items() if k not in ("domain", "description")}}
    # registry_scope is mandatory even when it matches defaults
    d["registry_scope"] = profile.registry_scope.model_dump(mode="json")
    body = yaml.safe_dump(d, sort_keys=False, allow_unicode=True,
                          default_flow_style=False, width=100)
    return _PROFILE_HEADER + body


def save_profile(domains_dir: Path, profile: DomainProfile, message: str) -> dict:
    path = domains_dir / profile.domain / "profile.yaml"
    backup = _backup(path)
    _atomic_write(path, dump_profile(profile))
    try:
        load_profile(domains_dir, profile.domain)
    except Exception:
        if backup is not None:
            shutil.copy2(backup, path)
        raise
    result = git_commit(path, message)
    result["backup"] = str(backup) if backup else None
    return result


def create_domain(domains_dir: Path, *, name: str, description: str,
                  include_tags: list[str], min_tier: int,
                  brief_cron: str | None) -> dict:
    if (domains_dir / name / "profile.yaml").exists():
        raise FileExistsError(f"domain {name!r} already exists")
    profile = DomainProfile.model_validate({
        "domain": name,
        "description": description,
        "registry_scope": {"include_tags": include_tags, "min_tier": min_tier},
        "entities": {"dictionaries": [{"path": "entities/entities.csv"}]},
        "schedule": {"brief_cron": brief_cron} if brief_cron else {},
    })
    entities = domains_dir / name / "entities" / "entities.csv"
    entities.parent.mkdir(parents=True, exist_ok=True)
    if not entities.exists():
        entities.write_text("surface_form,canonical_id,type\n")
    result = save_profile(domains_dir, profile, f"webapp: create domain '{name}'")
    git_commit(entities, f"webapp: scaffold entities for '{name}'")
    return result


# --------------------------------------------------------------------------
# entity dictionaries
# --------------------------------------------------------------------------

def read_entities(domains_dir: Path, domain: str) -> dict:
    profile = load_profile(domains_dir, domain)
    out = []
    for dictionary in profile.entities.dictionaries:
        path = dictionary.path
        if not path.is_absolute():
            path = domains_dir / domain / path
        rows: list[dict] = []
        if path.exists():
            with path.open(newline="") as fh:
                rows = list(csv.DictReader(fh))
        out.append({"path": str(dictionary.path), "rows": rows})
    return {"dictionaries": out}


def write_entities(domains_dir: Path, domain: str, dict_path: str,
                   rows: list[dict]) -> dict:
    profile = load_profile(domains_dir, domain)
    declared = {str(d.path) for d in profile.entities.dictionaries}
    if dict_path not in declared:
        raise ValueError(f"{dict_path!r} is not declared in the profile "
                         f"(declared: {sorted(declared)})")
    path = Path(dict_path)
    if not path.is_absolute():
        path = domains_dir / domain / path
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["surface_form", "canonical_id", "type"])
    writer.writeheader()
    for row in rows:
        surface = (row.get("surface_form") or "").strip()
        canonical = (row.get("canonical_id") or "").strip()
        if not surface or not canonical:
            continue
        writer.writerow({"surface_form": surface, "canonical_id": canonical,
                         "type": (row.get("type") or "").strip()})
    _backup(path)
    _atomic_write(path, buf.getvalue())
    return git_commit(path, f"webapp: update entity dictionary for '{domain}'")


# --------------------------------------------------------------------------
# settings catalog (what the settings page exposes)
# --------------------------------------------------------------------------

# (env key, group, label, kind, restart_required, secret, help)
SETTING_FIELDS: list[tuple[str, str, str, str, bool, bool, str]] = [
    ("ARGUS_LLM", "engine", "Language model", "choice:azure,stub", False, False,
     "azure = real synthesis via Azure OpenAI. stub = deterministic offline "
     "mode (extractive briefs, no API calls) — good for trying things out."),
    ("ARGUS_EMBEDDER", "engine", "Embeddings", "choice:azure,hashing", False, False,
     "azure = Azure OpenAI embeddings (production). hashing = offline dev "
     "embeddings. Changing this against an existing index is refused by the "
     "engine — use a new collection name to re-embed."),
    ("ARGUS_COLLECTION", "engine", "Index collection name", "text", False, False,
     "Qdrant collection holding the hybrid index. Bump this to start a fresh "
     "index (e.g. after switching embedders)."),

    ("AZURE_OPENAI_ENDPOINT", "azure", "Azure OpenAI endpoint", "text", False, False,
     "https://<resource>.openai.azure.com"),
    ("AZURE_OPENAI_API_KEY", "azure", "Azure OpenAI API key", "secret", False, True,
     "Stored in data/webapp/overrides.env (0600). Prefer a managed identity "
     "or secret manager in production (architecture SS12.4)."),
    ("AZURE_OPENAI_API_VERSION", "azure", "API version", "text", False, False,
     "Pinned per run into every manifest (SS12.1)."),
    ("ARGUS_AZURE_SYNTHESIS_DEPLOYMENT", "azure", "Synthesis deployment", "text",
     False, False, "Strong model deployment — synthesis and planning."),
    ("ARGUS_AZURE_UTILITY_DEPLOYMENT", "azure", "Utility deployment", "text",
     False, False, "Cheaper deployment — claim pairing and citation checks."),
    ("ARGUS_AZURE_EMBEDDING_DEPLOYMENT", "azure", "Embedding deployment", "text",
     False, False, "e.g. a text-embedding-3-large deployment."),

    ("ARGUS_BRIEF_MAX_STORIES", "tuning", "Max stories per brief", "int", False,
     False, "Upper bound on stories in a daily brief."),
    ("ARGUS_BRIEF_CHUNKS_PER_STORY", "tuning", "Evidence chunks per story", "int",
     False, False, "How much evidence each story carries into synthesis."),
    ("ARGUS_VERIFY_MAX_RETRIES", "tuning", "Citation-verifier retries", "int",
     False, False, "Bounded rewrite loop before unsupported sentences are "
     "dropped (never shipped)."),
    ("ARGUS_RESEARCH_MAX_EVIDENCE", "tuning", "Research evidence cap", "int",
     False, False, "Max evidence chunks assembled for a deep-research report."),
    ("ARGUS_RESEARCH_SUB_QUESTIONS_MAX", "tuning", "Max sub-questions", "int",
     False, False, "Planner decomposition cap for deep research."),
    ("ARGUS_CHUNK_SIZE", "tuning", "Chunk size (chars)", "int", False, False,
     "Ingestion chunking; applies to newly ingested documents."),
    ("ARGUS_CHUNK_OVERLAP", "tuning", "Chunk overlap (chars)", "int", False, False,
     "Ingestion chunk overlap."),

    ("ARGUS_DATA_DIR", "storage", "Data directory", "text", True, False,
     "Snapshots, index, briefs, checkpoints. Restart required."),
    ("ARGUS_DB_URL", "storage", "Database URL", "text", True, False,
     "Blank = SQLite under the data directory. Postgres in production. "
     "Restart required."),
    ("ARGUS_QDRANT_URL", "storage", "Qdrant server URL", "text", True, False,
     "Blank = embedded local index under the data directory. Restart required."),
    ("ARGUS_CHECKPOINT_URL", "storage", "Checkpointer URL", "text", False, False,
     "Blank = SQLite checkpoints. postgresql://… in production (needs the "
     "'postgres' extra)."),

    ("ARGUS_WEB_PASSWORD", "webapp", "Web password", "secret", False, True,
     "When set, every page requires sign-in. Applies immediately. Leave blank "
     "on a trusted machine."),
]

SETTING_KEYS = {f[0] for f in SETTING_FIELDS}


def settings_view(started_restart_keys: list[str]) -> dict:
    """Current effective settings, grouped, secrets masked."""
    from argus.settings import Settings

    defaults = Settings.model_construct()  # defaults without env resolution
    default_map = {
        "ARGUS_LLM": defaults.llm, "ARGUS_EMBEDDER": defaults.embedder,
        "ARGUS_COLLECTION": defaults.collection,
        "ARGUS_BRIEF_MAX_STORIES": defaults.brief_max_stories,
        "ARGUS_BRIEF_CHUNKS_PER_STORY": defaults.brief_chunks_per_story,
        "ARGUS_VERIFY_MAX_RETRIES": defaults.verify_max_retries,
        "ARGUS_RESEARCH_MAX_EVIDENCE": defaults.research_max_evidence,
        "ARGUS_RESEARCH_SUB_QUESTIONS_MAX": defaults.research_sub_questions_max,
        "ARGUS_CHUNK_SIZE": defaults.chunk_size,
        "ARGUS_CHUNK_OVERLAP": defaults.chunk_overlap,
        "ARGUS_DATA_DIR": str(defaults.data_dir),
    }
    from argus.webapp.state import read_overrides

    overrides = read_overrides()
    fields = []
    for key, group, label, kind, restart, secret, help_text in SETTING_FIELDS:
        env_value = os.environ.get(key)
        field = {
            "key": key, "group": group, "label": label, "kind": kind,
            "restart_required": restart, "secret": secret, "help": help_text,
            "overridden": key in overrides,
            "default": str(default_map.get(key, "")),
        }
        if secret:
            field["set"] = bool(env_value)
        else:
            field["value"] = env_value if env_value is not None else ""
        fields.append(field)
    return {"fields": fields, "restart_pending": started_restart_keys,
            "overrides_file": str(_overrides_display())}


def _overrides_display() -> Path:
    from argus.webapp.state import overrides_path

    return overrides_path()


def update_settings(changes: dict[str, str | None]) -> dict:
    """Apply changes to the overrides file and the live environment."""
    unknown = sorted(set(changes) - SETTING_KEYS)
    if unknown:
        raise ValueError(f"unknown setting keys: {unknown}")
    from argus.webapp.state import read_overrides, write_overrides

    overrides = read_overrides()
    restart_keys = {f[0] for f in SETTING_FIELDS if f[4]}
    touched_restart: list[str] = []
    for key, value in changes.items():
        if value is None or value == "":
            overrides.pop(key, None)
            os.environ.pop(key, None)
        else:
            overrides[key] = str(value)
            os.environ[key] = str(value)
        if key in restart_keys:
            touched_restart.append(key)
    write_overrides(overrides)
    return {"saved": sorted(k for k in changes), "restart_required": touched_restart}
