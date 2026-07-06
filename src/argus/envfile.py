"""Environment-file loading — the one place the KEY=value format is defined
(parsing delegated to python-dotenv; interpolation disabled so secrets are
taken literally).

Argus reads configuration in four layers. ``load_process_env()`` applies the
file layers to ``os.environ`` once per process, at the first
``get_settings()`` call, so every entry point (CLI, web console, scheduler,
tests) parses configuration through the same door. Later layers win:

    1. built-in defaults      field defaults on ``argus.settings.Settings``
    2. ./.env                 project root, gitignored — fills gaps only,
                              never overrides an already-exported shell var
                              (conventional dotenv semantics; see .env.example)
    3. shell environment      whatever the operator exported
    4. <data>/webapp/overrides.env
                              written by the web console's Settings page.
                              Deliberately overrides the shell, so a value
                              saved in the UI always takes effect — for
                              ``argus brief`` under cron too, not just the
                              console.

Who parses what, downstream of this module: keys prefixed ``ARGUS_`` are read
by pydantic-settings (``argus.settings.Settings``, ``env_prefix="ARGUS_"``);
the Azure OpenAI variables (``AZURE_OPENAI_ENDPOINT / _API_KEY /
_API_VERSION``) and the role deployments are read straight from
``os.environ`` by ``argus.llm.factory``. Loading both files *into the
environment* — rather than into a settings object — is what feeds both
consumers at once.
"""
from __future__ import annotations

import os
import re
from io import StringIO
from pathlib import Path

from dotenv import dotenv_values

__all__ = [
    "parse_env_file", "load_env_file", "load_process_env",
    "overrides_path", "read_overrides", "write_overrides", "reset_for_tests",
]

_loaded = False  # load_process_env ran in this process

# `KEY=   # note` — an empty value followed by an inline comment. python-dotenv
# parses the comment AS the value here (it only strips inline comments after a
# non-empty value), which would silently poison settings like ARGUS_DB_URL=.
# Rewrite just that shape to a plain empty value before dotenv sees it; the
# space requirement (=\s+#) keeps deliberate values like KEY=#abc and quoted
# KEY="#tag" untouched.
_EMPTY_THEN_COMMENT = re.compile(r"^(\s*(?:export\s+)?[^=\s#]+\s*=)\s+#.*$")


# --------------------------------------------------------------------------
# the format
# --------------------------------------------------------------------------

def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a KEY=value file with python-dotenv semantics: ``#`` full-line
    comments, inline comments after a space (``DIM=256  # note`` yields
    ``256``; quote values that must contain ``␣#``), ``export`` prefixes,
    single/double quotes, and quoted multi-line values. Two deliberate
    deviations from stock dotenv: no ``${VAR}`` interpolation (secrets are
    taken literally — quote style makes no difference), and
    ``KEY=   # comment`` yields an empty value rather than the comment text.
    Bare keys without ``=`` are dropped. Returns {} for a missing file."""
    if not path.exists():
        return {}
    text = "\n".join(
        _EMPTY_THEN_COMMENT.sub(r"\1", line)
        for line in path.read_text().splitlines()
    )
    parsed = dotenv_values(stream=StringIO(text), interpolate=False)
    return {k: v for k, v in parsed.items() if v is not None}


def load_env_file(path: Path, *, override: bool = False) -> dict[str, str]:
    """Apply a file's values to ``os.environ``; returns what was applied."""
    applied: dict[str, str] = {}
    for key, value in parse_env_file(path).items():
        if override or key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


# --------------------------------------------------------------------------
# the console's overrides file
# --------------------------------------------------------------------------

def overrides_path() -> Path:
    """``<ARGUS_DATA_DIR|data>/webapp/overrides.env`` unless relocated via
    ``ARGUS_WEBAPP_STATE``. Computed after ``.env`` loads, so ``.env`` may
    set the data directory."""
    state = os.environ.get("ARGUS_WEBAPP_STATE")
    root = Path(state) if state else Path(os.environ.get("ARGUS_DATA_DIR", "data")) / "webapp"
    return root / "overrides.env"


def read_overrides() -> dict[str, str]:
    return parse_env_file(overrides_path())


def write_overrides(values: dict[str, str]) -> None:
    path = overrides_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Argus web settings overrides — managed by the console's Settings page.",
        "# Highest-precedence config layer: applied over .env AND the shell at",
        "# every process start (CLI included). Delete a line (or the file) to",
        "# fall back to the shell / .env / built-in defaults. Keep this file",
        "# private: it may hold the Azure OpenAI key (use a real secret",
        "# manager in production).",
    ]
    lines += [f"{k}={v}" for k, v in sorted(values.items())]
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, path)
    try:  # best-effort: the file may hold a secret
        os.chmod(path, 0o600)
    except OSError:
        pass


# --------------------------------------------------------------------------
# the process-start chokepoint
# --------------------------------------------------------------------------

def load_process_env(*, force: bool = False) -> dict[str, dict[str, str]]:
    """Apply layer 2 (./.env, non-overriding) then layer 4 (console
    overrides, overriding). Runs once per process unless ``force``;
    ``get_settings()`` calls this, so no entry point can miss it."""
    global _loaded
    if _loaded and not force:
        return {}
    applied = {
        "dotenv": load_env_file(Path(".env"), override=False),
        "overrides": load_env_file(overrides_path(), override=True),
    }
    _loaded = True
    return applied


def reset_for_tests() -> None:
    global _loaded
    _loaded = False
