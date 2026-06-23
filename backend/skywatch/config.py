"""Environment-driven settings (single place that reads the environment).

Everything else imports :func:`get_settings` — never read ``os.environ`` directly.
Values come from the repo-root ``.env`` (see ``.env.example``) or real environment
variables, which take precedence.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/skywatch/config.py -> parents[2] is the repo root that holds .env
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"

# Placeholder values shipped in .env.example; treated as "not configured".
_PLACEHOLDERS = {"", "your_client_id", "your_client_secret"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- OpenSky OAuth2 (client-credentials) ---
    opensky_client_id: str = ""
    opensky_client_secret: str = ""
    opensky_token_url: str = (
        "https://auth.opensky-network.org/auth/realms/"
        "opensky-network/protocol/openid-connect/token"
    )
    opensky_states_url: str = "https://opensky-network.org/api/states/all"

    # --- Bounding box: lamin,lomin,lamax,lomax (defaults = central Europe) ---
    bbox_lamin: float = 47.0
    bbox_lomin: float = 5.0
    bbox_lamax: float = 55.0
    bbox_lomax: float = 15.0

    # --- Collector ---
    poll_interval_seconds: int = Field(default=8, ge=1)

    # --- Model / scoring (used at train and serve time) ---
    model_window_size: int = Field(default=20, ge=2)
    anomaly_threshold_percentile: float = Field(default=99.0, gt=0.0, lt=100.0)

    # --- Live scoring service (Phase 6) ---
    scoring_mode: str = "replay"          # "demo" (bundled file, no DB) | "replay" (DB) | "live"
    replay_minutes: int = Field(default=20, ge=1)
    replay_interval_seconds: float = Field(default=1.0, gt=0.0)
    cors_origins: str = "*"               # comma-separated allowed origins ("*" = any)

    # --- Database ---
    database_url: str = "postgresql+asyncpg://skywatch:skywatch@localhost:5432/skywatch"

    @model_validator(mode="after")
    def _check_bbox(self) -> "Settings":
        if self.bbox_lamin >= self.bbox_lamax:
            raise ValueError("BBOX_LAMIN must be < BBOX_LAMAX")
        if self.bbox_lomin >= self.bbox_lomax:
            raise ValueError("BBOX_LOMIN must be < BBOX_LOMAX")
        return self

    @property
    def has_credentials(self) -> bool:
        """True only when real (non-placeholder) OpenSky credentials are set."""
        return (
            self.opensky_client_id not in _PLACEHOLDERS
            and self.opensky_client_secret not in _PLACEHOLDERS
        )

    @property
    def bbox_params(self) -> dict[str, float]:
        """Query params for the OpenSky ``/states/all`` bounding box."""
        return {
            "lamin": self.bbox_lamin,
            "lomin": self.bbox_lomin,
            "lamax": self.bbox_lamax,
            "lomax": self.bbox_lomax,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
