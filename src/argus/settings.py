"""Runtime settings (env-driven, prefix ARGUS_).

Secrets never live in config files (architecture SS12.4); connection strings
and paths come from the environment.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ARGUS_")

    data_dir: Path = Path("data")
    db_url: str = ""  # empty -> derived SQLite URL under data_dir
    registry_path: Path = Path("registry/sources.yaml")
    domains_dir: Path = Path("domains")

    # M2 — index & embeddings (architecture SS7.3/SS7.4, AD-10)
    embedder: str = "azure"      # "azure" (prod) | "hashing" (dev/tests only)
    hashing_dim: int = 256
    qdrant_url: str = ""         # empty -> embedded local mode under data_dir
    collection: str = "argus_chunks"
    chunk_size: int = 1600
    chunk_overlap: int = 200

    # M3 — runtime plane (architecture SS8, SS11)
    llm: str = "azure"           # "azure" (prod) | "stub" (dev: extractive, no LLM)
    checkpoint_url: str = ""     # empty -> sqlite under data_dir; postgres URL in prod
    brief_max_stories: int = 20
    brief_chunks_per_story: int = 3
    verify_max_retries: int = 2
    event_match_jaccard: float = 0.5
    research_max_evidence: int = 40
    research_sub_questions_max: int = 4

    @property
    def resolved_db_url(self) -> str:
        if self.db_url:
            return self.db_url
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{self.data_dir / 'argus.db'}"

    @property
    def blob_root(self) -> Path:
        return self.data_dir / "snapshots"

    @property
    def briefs_dir(self) -> Path:
        return self.data_dir / "briefs"

    @property
    def checkpoint_sqlite_path(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"


def get_settings() -> Settings:
    return Settings()
