"""Central config loader.

All secrets and tunables come from environment variables (loaded from a
local .env via python-dotenv, which is gitignored). Nothing here is ever
hardcoded, and the config is validated eagerly so a missing key fails at
startup rather than mid-run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {val!r}") from exc


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_model: str
    openai_api_key: str | None
    anthropic_api_key: str | None

    gmail_credentials_path: Path
    gmail_token_path: Path

    max_llm_calls_per_hour: int
    max_emails_per_run: int
    require_human_approval: bool
    audit_log_dir: Path

    def validate(self) -> None:
        if self.llm_provider not in {"anthropic", "openai"}:
            raise ConfigError(
                f"LLM_PROVIDER must be 'anthropic' or 'openai', got {self.llm_provider!r}"
            )
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ConfigError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ConfigError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if self.max_llm_calls_per_hour <= 0:
            raise ConfigError("MAX_LLM_CALLS_PER_HOUR must be positive")
        if self.max_emails_per_run <= 0:
            raise ConfigError("MAX_EMAILS_PER_RUN must be positive")
        if not self.require_human_approval:
            # This is a deliberate, loud guard: auto-send is the one mode
            # that can actually hurt a client's inbox/reputation, so we
            # never silently allow it.
            raise ConfigError(
                "REQUIRE_HUMAN_APPROVAL=false is not supported yet. "
                "This agent only ever creates drafts for human review."
            )


def load_settings() -> Settings:
    settings = Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "anthropic"),
        llm_model=os.getenv("LLM_MODEL", "claude-sonnet-5"),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        gmail_credentials_path=Path(os.getenv("GMAIL_CREDENTIALS_PATH", "credentials.json")),
        gmail_token_path=Path(os.getenv("GMAIL_TOKEN_PATH", "token.json")),
        max_llm_calls_per_hour=_env_int("MAX_LLM_CALLS_PER_HOUR", 100),
        max_emails_per_run=_env_int("MAX_EMAILS_PER_RUN", 25),
        require_human_approval=_env_bool("REQUIRE_HUMAN_APPROVAL", True),
        audit_log_dir=Path(os.getenv("AUDIT_LOG_DIR", "audit_logs")),
    )
    settings.validate()
    return settings
