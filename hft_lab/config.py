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
    # Phase 2 keys
    "DEV_MODE",
    "HISTORICAL_BARS",
    "BAR_TIMEFRAME",
    "TRADE_LOG_PATH",
    "SESSION_STATE_PATH",
    "ENSEMBLE_THRESHOLD",
    "ENSEMBLE_THRESHOLD_AMBIGUOUS",
    "ENSEMBLE_THRESHOLD_RANDOM",
    "MAX_OPEN_POSITIONS",
    "NEWS_POLL_INTERVAL",
    "KALMAN_PROCESS_NOISE",
    "KALMAN_MEASUREMENT_NOISE",
    "IC_WINDOW_EQUITY",
    "IC_WINDOW_CRYPTO",
    "IC_WINDOW_FOREX",
    "GARCH_UPDATE_INTERVAL",
    "ATR_PERIOD",
    "ATR_STOP_MULTIPLIER",
    "RISK_REWARD_RATIO",
    # Phase 6 keys
    "MARKET_TYPE",
    "CRYPTO_BENCHMARK",
    "OANDA_API_KEY",
    "OANDA_ACCOUNT_ID",
    "OANDA_ENVIRONMENT",
    "FOREX_BENCHMARK",
    # News strategy keys
    "REDDIT_SENTIMENT_ENABLED",
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
    min_position_size: float
    spread_adjustment_enabled: bool
    slippage_factor: float
    min_profit_spread_multiple: float
    # Phase 2 fields
    dev_mode: bool
    historical_bars: int
    bar_timeframe: str
    trade_log_path: str
    session_state_path: str
    ensemble_threshold: float
    ensemble_threshold_ambiguous: float
    ensemble_threshold_random: float
    max_open_positions: int
    news_poll_interval: int
    kalman_process_noise: float
    kalman_measurement_noise: float
    ic_window_equity: int
    ic_window_crypto: int
    ic_window_forex: int
    garch_update_interval: int
    atr_period: int
    atr_stop_multiplier: float
    risk_reward_ratio: float
    # Phase 6 fields
    market_type: str
    crypto_benchmark: str
    oanda_api_key: str
    oanda_account_id: str
    oanda_environment: str
    forex_benchmark: str
    # Forex ATR floor and spread protection
    min_atr_forex: float
    min_stop_spread_multiple: float
    max_entry_spread: float
    # Continuous bar-level IC scoring
    ic_forward_bars: int
    ic_warm_start_enabled: bool
    # Entry gating
    tradeable_regimes: frozenset
    trading_sessions: str
    post_close_cooldown_s: float
    # News strategy fields
    reddit_sentiment_enabled: bool
    reddit_client_id: str
    reddit_client_secret: str

    def is_live(self) -> bool:
        """Return ``True`` when connected to the live (real-money) Alpaca endpoint.

        Downstream components use this to apply stricter safeguards — e.g. extra
        confirmation checks or reduced position caps — whenever real funds are
        at risk.

        Returns:
            ``True`` if ``alpaca_base_url`` matches the live trading URL.
        """
        return self.alpaca_base_url.rstrip("/") == LIVE_BASE_URL.rstrip("/")

    def bar_timeframe_minutes(self) -> int:
        """Return the configured bar timeframe in whole minutes.

        Parses BAR_TIMEFRAME strings like ``1Min``, ``5Min``, ``15Min``,
        ``1Hour``, ``4H`` or a bare number of minutes.  Falls back to 1 on
        any parse failure so the engine degrades to 1-minute bars rather
        than crashing.
        """
        raw = self.bar_timeframe.strip().lower()
        try:
            if raw.endswith("min"):
                return max(1, int(raw[:-3]))
            if raw.endswith("hour"):
                return max(1, int(raw[:-4])) * 60
            if raw.endswith("h"):
                return max(1, int(raw[:-1])) * 60
            if raw.endswith("m"):
                return max(1, int(raw[:-1]))
            return max(1, int(raw))
        except ValueError:
            return 1

    def oanda_granularity(self) -> str:
        """Return the OANDA candle granularity string for BAR_TIMEFRAME.

        e.g. 1 → ``M1``, 15 → ``M15``, 60 → ``H1``, 240 → ``H4``.
        """
        minutes = self.bar_timeframe_minutes()
        if minutes >= 60 and minutes % 60 == 0:
            return f"H{minutes // 60}"
        return f"M{minutes}"

    def is_crypto(self) -> bool:
        """Return ``True`` when the engine is configured for cryptocurrency markets.

        Crypto mode enables 24/7 trading via Alpaca's crypto data feed, disables
        equity market-hours logic, and switches to rolling-window VWAP.

        Returns:
            ``True`` when ``market_type`` is ``"crypto"``.
        """
        return self.market_type.lower() == "crypto"

    def is_forex(self) -> bool:
        """Return ``True`` when the engine is configured for forex markets.

        Forex mode routes data through the OANDA v20 API, applies 24/5
        weekend-aware market-hours logic, and uses tick-count VWAP.

        Returns:
            ``True`` when ``market_type`` is ``"forex"``.
        """
        return self.market_type.lower() == "forex"


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

    # Validate MARKET_TYPE is a recognised value
    market_type_raw = _require_env("MARKET_TYPE").lower()
    if market_type_raw not in ("equity", "crypto", "forex"):
        raise ConfigurationError(
            f"MARKET_TYPE must be 'equity', 'crypto', or 'forex', "
            f"got: {market_type_raw!r}"
        )

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
        min_position_size=float(_require_env("MIN_POSITION_SIZE")),
        spread_adjustment_enabled=_parse_bool(
            _require_env("SPREAD_ADJUSTMENT_ENABLED"), "SPREAD_ADJUSTMENT_ENABLED"
        ),
        slippage_factor=float(_require_env("SLIPPAGE_FACTOR")),
        min_profit_spread_multiple=float(_require_env("MIN_PROFIT_SPREAD_MULTIPLE")),
        # Phase 2
        dev_mode=_parse_bool(_require_env("DEV_MODE"), "DEV_MODE"),
        historical_bars=int(_require_env("HISTORICAL_BARS")),
        bar_timeframe=_require_env("BAR_TIMEFRAME"),
        trade_log_path=_require_env("TRADE_LOG_PATH"),
        session_state_path=_require_env("SESSION_STATE_PATH"),
        ensemble_threshold=float(_require_env("ENSEMBLE_THRESHOLD")),
        ensemble_threshold_ambiguous=float(_require_env("ENSEMBLE_THRESHOLD_AMBIGUOUS")),
        ensemble_threshold_random=float(_require_env("ENSEMBLE_THRESHOLD_RANDOM")),
        max_open_positions=int(_require_env("MAX_OPEN_POSITIONS")),
        news_poll_interval=int(_require_env("NEWS_POLL_INTERVAL")),
        kalman_process_noise=float(_require_env("KALMAN_PROCESS_NOISE")),
        kalman_measurement_noise=float(_require_env("KALMAN_MEASUREMENT_NOISE")),
        ic_window_equity=int(_require_env("IC_WINDOW_EQUITY")),
        ic_window_crypto=int(_require_env("IC_WINDOW_CRYPTO")),
        ic_window_forex=int(_require_env("IC_WINDOW_FOREX")),
        garch_update_interval=int(_require_env("GARCH_UPDATE_INTERVAL")),
        atr_period=int(_require_env("ATR_PERIOD")),
        atr_stop_multiplier=float(_require_env("ATR_STOP_MULTIPLIER")),
        risk_reward_ratio=float(_require_env("RISK_REWARD_RATIO")),
        # Phase 6
        market_type=market_type_raw,
        crypto_benchmark=_require_env("CRYPTO_BENCHMARK"),
        oanda_api_key=_require_env("OANDA_API_KEY"),
        oanda_account_id=_require_env("OANDA_ACCOUNT_ID"),
        oanda_environment=_require_env("OANDA_ENVIRONMENT"),
        forex_benchmark=_require_env("FOREX_BENCHMARK"),
        # News strategy — Reddit is optional; credentials default to empty string
        reddit_sentiment_enabled=_parse_bool(
            _require_env("REDDIT_SENTIMENT_ENABLED"), "REDDIT_SENTIMENT_ENABLED"
        ),
        min_atr_forex=float(os.getenv("MIN_ATR_FOREX", "0.0005")),
        min_stop_spread_multiple=float(os.getenv("MIN_STOP_SPREAD_MULTIPLE", "3.0")),
        max_entry_spread=float(os.getenv("MAX_ENTRY_SPREAD", "0.00040")),
        ic_forward_bars=int(os.getenv("IC_FORWARD_BARS", "5")),
        ic_warm_start_enabled=_parse_bool(
            os.getenv("IC_WARM_START_ENABLED", "true"), "IC_WARM_START_ENABLED"
        ),
        # Empty TRADEABLE_REGIMES disables the regime gate entirely
        tradeable_regimes=frozenset(
            s.strip().upper()
            for s in os.getenv(
                "TRADEABLE_REGIMES", "TRENDING,MEAN_REVERTING"
            ).split(",")
            if s.strip()
        ),
        trading_sessions=os.getenv("TRADING_SESSIONS", "08:00-12:00"),
        post_close_cooldown_s=float(os.getenv("POST_CLOSE_COOLDOWN_S", "300")),
        reddit_client_id=os.getenv("REDDIT_CLIENT_ID", ""),
        reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
    )


# ---------------------------------------------------------------------------
# Module-level singleton — imported by all other modules
# ---------------------------------------------------------------------------

config: Config = _load_config()
