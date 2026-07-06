"""Tests for argus.envfile — the configuration layering.

Precedence under test (lowest -> highest):
    Settings defaults < ./.env < shell environment < console overrides.env
"""
from __future__ import annotations

from pathlib import Path

import pytest

from argus import envfile

KEY = "ARGUS_BRIEF_MAX_STORIES"  # int-typed Settings field, safe to play with


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    """Every test runs in an empty cwd with the loader guard reset and the
    keys it plays with absent from the environment."""
    monkeypatch.chdir(tmp_path)
    for key in (KEY, "ARGUS_DATA_DIR", "ARGUS_WEBAPP_STATE", "DEMO_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    envfile.reset_for_tests()
    yield
    envfile.reset_for_tests()


def test_parse_format(tmp_path):
    f = tmp_path / "x.env"
    f.write_text(
        "# comment\n"
        "\n"
        "PLAIN = value with spaces  \n"
        "export EXPORTED=yes\n"
        'QUOTED="a # not-a-comment"\n'
        "SINGLE='  padded  '\n"
        "NOEQUALS ignored\n"
        "=noname ignored\n"
    )
    assert envfile.parse_env_file(f) == {
        "PLAIN": "value with spaces",
        "EXPORTED": "yes",
        "QUOTED": "a # not-a-comment",
        "SINGLE": "  padded  ",
    }
    assert envfile.parse_env_file(tmp_path / "missing.env") == {}


def test_dotenv_fills_but_never_overrides_shell(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(f"{KEY}=5\nDEMO_TOKEN=from-dotenv\n")
    monkeypatch.setenv(KEY, "8")  # operator export wins over .env
    applied = envfile.load_process_env()
    import os
    assert os.environ[KEY] == "8"
    assert os.environ["DEMO_TOKEN"] == "from-dotenv"
    assert KEY not in applied["dotenv"] and "DEMO_TOKEN" in applied["dotenv"]


def test_overrides_beat_shell_and_dotenv(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(f"{KEY}=5\n")
    monkeypatch.setenv(KEY, "8")
    ov = tmp_path / "data" / "webapp" / "overrides.env"
    ov.parent.mkdir(parents=True)
    ov.write_text(f"{KEY}=9\n")
    envfile.load_process_env()
    import os
    assert os.environ[KEY] == "9"  # console decision wins everywhere


def test_loader_runs_once_unless_forced(tmp_path):
    import os
    (tmp_path / ".env").write_text("DEMO_TOKEN=first\n")
    envfile.load_process_env()
    assert os.environ["DEMO_TOKEN"] == "first"
    (tmp_path / ".env").write_text("DEMO_TOKEN=second\n")
    envfile.load_process_env()               # guarded: no re-read
    assert os.environ["DEMO_TOKEN"] == "first"
    os.environ.pop("DEMO_TOKEN")
    envfile.load_process_env(force=True)     # webapp startup path
    assert os.environ["DEMO_TOKEN"] == "second"


def test_get_settings_reads_dotenv(tmp_path):
    (tmp_path / ".env").write_text(f"{KEY}=7\n")
    from argus.settings import get_settings
    assert get_settings().brief_max_stories == 7


def test_full_precedence_chain(monkeypatch, tmp_path):
    from argus.settings import get_settings
    assert get_settings().brief_max_stories == 20      # 1. field default
    envfile.reset_for_tests()
    (tmp_path / ".env").write_text(f"{KEY}=5\n")
    assert get_settings().brief_max_stories == 5       # 2. .env
    envfile.reset_for_tests()
    monkeypatch.setenv(KEY, "8")
    assert get_settings().brief_max_stories == 8       # 3. shell
    envfile.reset_for_tests()
    ov = envfile.overrides_path()
    ov.parent.mkdir(parents=True)
    ov.write_text(f"{KEY}=9\n")
    assert get_settings().brief_max_stories == 9       # 4. console overrides


def test_overrides_path_honors_data_dir_and_state(monkeypatch):
    assert envfile.overrides_path() == Path("data/webapp/overrides.env")
    monkeypatch.setenv("ARGUS_DATA_DIR", "/srv/argus-data")
    assert envfile.overrides_path() == Path("/srv/argus-data/webapp/overrides.env")
    monkeypatch.setenv("ARGUS_WEBAPP_STATE", "/etc/argus")
    assert envfile.overrides_path() == Path("/etc/argus/overrides.env")
