import os
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Конфігурація веб-сервісу з оточення (INDEX_PATH, LOG_LEVEL)."""

    index_path: Path = Field(default=Path("index.json"))
    log_level: str = Field(default="INFO")
    app_title: str = Field(default="findex — Пошуковий рушій")


def load_settings() -> Settings:
    raw_index = os.getenv("INDEX_PATH", "index.json").strip() or "index.json"
    raw_log = os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO"
    raw_title = os.getenv("FINDEX_WEB_TITLE", "findex — Пошуковий рушій").strip()
    return Settings(
        index_path=Path(raw_index),
        log_level=raw_log,
        app_title=raw_title or "findex — Пошуковий рушій",
    )
