from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PROCEX_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://procex:procex@localhost:5432/procex"

    # LLM. "claude-code" runs through the Claude Code CLI on a Claude subscription (evaluation without API
    # billing); "fixture" replays stored outputs for the eval documents (demo without any account)
    extractor: Literal["claude", "claude-code", "fixture"] = "claude"
    claude_code_model: str | None = None
    fixtures_dir: str = "eval/fixtures"
    model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_tokens: int = 16000

    # Fields whose final confidence falls below this go to the human-review queue.
    review_threshold: float = 0.8
    # Our own company names: a "supplier" matching one of these is the bill-to party, not the supplier.
    buyer_names: list[str] = []
    max_upload_bytes: int = 20 * 1024 * 1024

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
