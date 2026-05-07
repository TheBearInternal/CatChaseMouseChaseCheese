"""Configuration loader for hft_lab.

Reads all required environment variables at import time, validates that none
are missing, and exports a single frozen ``Config`` dataclass instance named
``config``.  Every value is typed correctly (str / int / float / bool).

Raises ``ConfigurationError`` — naming the absent key — if any required
variable is unset or blank.  API keys and secret values are never printed,
logged, or exposed anywhere in this module.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LIVE_BASE_URL: str = "https://api.alpaca.markets"

REQUIRED_KEYS: tuple[str, ...] = (
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "ALPACA_BASE_URL",
    "ALPACA_DATA_URL",
    "NEWS_API_KEY",
    "PRIMARY_SYMBOL",
    "BENCHMARK_SYMBOL",
    "BEHAVIOR_SEED",
    "ACCOUNT_LIMIT",
    "MAX_RISK_PER_TRADE_PCT",
    "POSITION_MODE",
    "FIXED_POSITION_SIZE",
    "MIN_POSITION_SIZE",
    "SPREAD_ADJUSTMENT_ENABLED",
    "SLIPPAGE_FACTOR",
    "MIN_PROFIT_SPREAD_MULTIPLE",
)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class ConfigurationError(Exception):
    """Raised when a required environment variable is missing or malformed.

    The message always names the specific key that caused the failure so the
    operator knows exactly what to add to their ``.env`` file.
    """


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_env(key: str) -> str:
    """Return the stripped value for *key* or raise ``ConfigurationError``.

    Args:
        key: Environment variable name to look up.

    Returns:
        Non-empty string value.

    Raises:
        ConfigurationError: If the variable is absent or contains only whitespace.
    """
    value = os.getenv(key)
    if value is None or value.strip() == "":
        raise ConfigurationError(
            f"Required environment variable '{key}' is not set. "
            f"Copy .env.template to .env and supply a value for '{key}'."
        )
    return value.strip()


def _parse_bool(raw: str, key: str) -> bool:
    """Parse a string representation of a boolean.

    Accepts: ``true``, ``1``, ``yes`` → True; ``false``, ``0``, ``no`` → False.

    Args:
        raw: Raw string value from the environment.
        key: Variable name included in any error message.

    Returns:
        Parsed boolean.

    Raises:
        ConfigurationError: If the value does not match any recognised pattern.
    """
    normalized = raw.strip().lower()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no"):
        return False
    raise ConfigurationError(
        f"Environment variable '{key}' must be a boolean "
        f"(true/false/yes/no/1/0), got: {raw!r}"
    )


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Immutable, typed snapshot of all hft_lab configuration values.

    Constructed once at module import time from environment variables.
    Downstream modules should import the module-level ``config`` singleton
    rather than constructing this class directly.
    """

    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_base_url: str
    alpaca_data_url: str
    news_api_key: str
    primary_symbol: str
    benchmark_symbol: str
    behavior_seed: int
    account_limit: float
    max_risk_per_trade_pct: float
    position_mode: str
    fixed_position_size: int
    min_position_size: int
    spread_adjustment_enabled: bool
    slippage_factor: float
    min_profit_spread_multiple: float

    def is_live(self) -> bool:
        """Return ``True`` when connected to the live (real-money) Alpaca endpoint.

        Downstream components use this to apply stricter safeguards — e.g. extra
        confirmation checks or reduced position caps — whenever real funds are
        at risk.

        Returns:
            ``True`` if ``alpaca_base_url`` matches the live trading URL.
        """
        return self.alpaca_base_url.rstrip("/") == LIVE_BASE_URL.rstrip("/")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _load_config() -> Config:
    """Read ``.env``, validate all required keys, and build a ``Config`` instance.

    Returns:
        Populated ``Config`` dataclass.

    Raises:
        ConfigurationError: On the first missing or malformed required variable.
    """
    load_dotenv()

    # Validate presence of every required key before attempting type conversions
    # so the error message identifies the missing key rather than raising a
    # generic KeyError or ValueError.
    for key in REQUIRED_KEYS:
        _require_env(key)

    return Config(
        alpaca_api_key=_require_env("ALPACA_API_KEY"),
        alpaca_secret_key=_require_env("ALPACA_SECRET_KEY"),
        alpaca_base_url=_require_env("ALPACA_BASE_URL"),
        alpaca_data_url=_require_env("ALPACA_DATA_URL"),
        news_api_key=_require_env("NEWS_API_KEY"),
        primary_symbol=_require_env("PRIMARY_SYMBOL"),
        benchmark_symbol=_require_env("BENCHMARK_SYMBOL"),
        behavior_seed=int(_require_env("BEHAVIOR_SEED")),
        account_limit=float(_require_env("ACCOUNT_LIMIT")),
        max_risk_per_trade_pct=float(_require_env("MAX_RISK_PER_TRADE_PCT")),
        position_mode=_require_env("POSITION_MODE"),
        fixed_position_size=int(_require_env("FIXED_POSITION_SIZE")),
        min_position_size=int(_require_env("MIN_POSITION_SIZE")),
        spread_adjustment_enabled=_parse_bool(
            _require_env("SPREAD_ADJUSTMENT_ENABLED"), "SPREAD_ADJUSTMENT_ENABLED"
        ),
        slippage_factor=float(_require_env("SLIPPAGE_FACTOR")),
        min_profit_spread_multiple=float(_require_env("MIN_PROFIT_SPREAD_MULTIPLE")),
    )


# ---------------------------------------------------------------------------
# Module-level singleton — imported by all other modules
# ---------------------------------------------------------------------------

config: Config = _load_config()
