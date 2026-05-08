"""Multi-scale memory and data management for the trading engine.

Three rolling buffers operate simultaneously:

* **FastBuffer** — tick-level deque (up to 200 entries).
* **MediumBuffer** — 90-minute rolling window of 1-minute bar aggregations.
* **SlowBuffer** — single session-level JSON snapshot persisted between sessions.

``DataManager`` coordinates all buffers for both the primary symbol and the
benchmark, handles historical warm-up from the Alpaca REST API, and exposes
derived metrics (VWAP, relative strength).
"""
from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import config
from logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAST_BUFFER_MAXLEN: int = 200
MEDIUM_BUFFER_WINDOW_MINUTES: int = 90
SHOCK_WINDOW_MINUTES: int = 10
SHOCK_VOLATILITY_MULTIPLIER: float = 2.0
MIN_BARS_NORMAL: int = 50
MIN_BARS_DEV: int = 30
RS_LOOKBACK_TICKS: int = 20


# ---------------------------------------------------------------------------
# FastBuffer
# ---------------------------------------------------------------------------


class FastBuffer:
    """Tick-level rolling buffer backed by a fixed-length deque.

    Each entry is a dict with keys: timestamp, symbol, price, bid, ask,
    bid_size, ask_size, volume.  Oldest entries are automatically evicted
    when the buffer reaches ``maxlen``.
    """

    def __init__(self, maxlen: int = FAST_BUFFER_MAXLEN) -> None:
        """Initialise the buffer.

        Args:
            maxlen: Maximum number of ticks retained.
        """
        self._data: deque[Dict[str, Any]] = deque(maxlen=maxlen)

    def append(self, tick: Dict[str, Any]) -> None:
        """Append a new tick to the buffer.

        Args:
            tick: Dict containing tick fields (price, bid, ask, etc.).
        """
        self._data.append(tick)

    def to_dataframe(self) -> pd.DataFrame:
        """Return buffer contents as a pandas DataFrame.

        Returns:
            DataFrame with one row per tick, or empty DataFrame if buffer
            is empty.
        """
        if not self._data:
            return pd.DataFrame()
        return pd.DataFrame(list(self._data))

    def latest_price(self) -> Optional[float]:
        """Return the most recent non-None price in the buffer.

        Returns:
            Latest price as float, or ``None`` if buffer is empty.
        """
        for tick in reversed(self._data):
            if tick.get("price") is not None:
                return float(tick["price"])
        return None

    def latest_bid(self) -> Optional[float]:
        """Return the most recent non-None bid price.

        Returns:
            Latest bid as float, or ``None`` if unavailable.
        """
        for tick in reversed(self._data):
            if tick.get("bid") is not None:
                return float(tick["bid"])
        return None

    def latest_ask(self) -> Optional[float]:
        """Return the most recent non-None ask price.

        Returns:
            Latest ask as float, or ``None`` if unavailable.
        """
        for tick in reversed(self._data):
            if tick.get("ask") is not None:
                return float(tick["ask"])
        return None

    def __len__(self) -> int:
        return len(self._data)


# ---------------------------------------------------------------------------
# MediumBuffer
# ---------------------------------------------------------------------------


class MediumBuffer:
    """Time-bounded rolling window of 1-minute bar aggregations.

    Entries older than ``MEDIUM_BUFFER_WINDOW_MINUTES`` are evicted on each
    ``append`` call.  Each bar is a dict with keys: timestamp, open, high,
    low, close, volume, vwap_contribution.
    """

    def __init__(self) -> None:
        self._bars: List[Dict[str, Any]] = []

    def append(self, bar: Dict[str, Any]) -> None:
        """Append a new bar and evict entries outside the rolling window.

        Args:
            bar: Dict with OHLCV data and a ``timestamp`` key.
        """
        self._bars.append(bar)
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=MEDIUM_BUFFER_WINDOW_MINUTES)
        self._bars = [
            b for b in self._bars
            if b["timestamp"].tzinfo is not None and b["timestamp"] >= cutoff
            or b["timestamp"].tzinfo is None
        ]

    def to_dataframe(self) -> pd.DataFrame:
        """Return buffer contents as a pandas DataFrame.

        Returns:
            DataFrame with one row per bar, or empty DataFrame if empty.
        """
        if not self._bars:
            return pd.DataFrame()
        return pd.DataFrame(self._bars)

    def volatility(self) -> float:
        """Return rolling standard deviation of close-to-close log returns.

        Returns:
            Standard deviation of returns, or 0.0 if insufficient data.
        """
        df = self.to_dataframe()
        if len(df) < 2:
            return 0.0
        closes = df["close"].values.astype(float)
        returns = np.diff(np.log(closes + 1e-10))
        return float(np.std(returns))

    def is_post_shock(self) -> bool:
        """Return True if recent volatility exceeds 2× the 90-minute average.

        Uses the last ``SHOCK_WINDOW_MINUTES`` of bars versus the full window.

        Returns:
            ``True`` when in a post-shock high-volatility state.
        """
        df = self.to_dataframe()
        if len(df) < SHOCK_WINDOW_MINUTES + 2:
            return False
        closes = df["close"].values.astype(float)
        returns = np.diff(np.log(closes + 1e-10))
        full_vol = float(np.std(returns))
        recent_returns = returns[-SHOCK_WINDOW_MINUTES:]
        recent_vol = float(np.std(recent_returns)) if len(recent_returns) >= 2 else 0.0
        return recent_vol > SHOCK_VOLATILITY_MULTIPLIER * full_vol and full_vol > 0

    def __len__(self) -> int:
        return len(self._bars)


# ---------------------------------------------------------------------------
# SlowBuffer
# ---------------------------------------------------------------------------


class SlowBuffer:
    """Single session snapshot persisted as JSON between sessions.

    Stores: session_date, session_pnl, signal_weights, regime_counts,
    trade_count.  Used by KalmanEnsemble to restore weight state.
    """

    def __init__(self) -> None:
        self.data: Dict[str, Any] = {
            "session_date": None,
            "session_pnl": 0.0,
            "signal_weights": None,
            "regime_counts": {},
            "trade_count": 0,
        }

    def save(self, path: str) -> None:
        """Persist the current snapshot to disk as JSON.

        Args:
            path: File path to write. Parent directory is created if needed.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = dict(self.data)
        # Serialise datetime to string if present
        if isinstance(payload.get("session_date"), datetime):
            payload["session_date"] = payload["session_date"].isoformat()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.debug(f"SlowBuffer saved → {path}")

    def load(self, path: str) -> None:
        """Load a previously saved snapshot from disk.

        Silently no-ops if the file does not exist.

        Args:
            path: File path to read.
        """
        if not os.path.exists(path):
            logger.debug(f"No session state found at {path} — starting fresh")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            logger.info(f"SlowBuffer loaded from {path}")
        except Exception as exc:
            logger.warning(f"Failed to load session state from {path}: {exc}")


# ---------------------------------------------------------------------------
# DataManager
# ---------------------------------------------------------------------------


class DataManager:
    """Coordinates all data buffers and derived metrics for the trading engine.

    Holds:
    * ``fast_primary``     — FastBuffer for PRIMARY_SYMBOL ticks.
    * ``fast_benchmark``   — FastBuffer for BENCHMARK_SYMBOL ticks.
    * ``medium_primary``   — MediumBuffer for PRIMARY_SYMBOL 1-min bars.
    * ``slow_buffer``      — SlowBuffer for session persistence.
    """

    def __init__(self) -> None:
        self.fast_primary: FastBuffer = FastBuffer()
        self.fast_benchmark: FastBuffer = FastBuffer()
        self.medium_primary: MediumBuffer = MediumBuffer()
        self.slow_buffer: SlowBuffer = SlowBuffer()
        self._session_open_ts: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    async def ingest_tick(self, symbol: str, tick_data: Dict[str, Any]) -> None:
        """Route an incoming tick to the appropriate FastBuffer.

        Args:
            symbol:    Symbol string matching PRIMARY_SYMBOL or BENCHMARK_SYMBOL.
            tick_data: Dict with tick fields (price, bid, ask, etc.).
        """
        if symbol == config.primary_symbol:
            self.fast_primary.append(tick_data)
            if self._session_open_ts is None:
                self._session_open_ts = datetime.now(timezone.utc)
        elif symbol == config.benchmark_symbol:
            self.fast_benchmark.append(tick_data)

    async def ingest_bar(self, bar_data: Dict[str, Any]) -> None:
        """Append a completed 1-minute bar to the MediumBuffer.

        Args:
            bar_data: Dict with OHLCV fields and a ``timestamp`` key.
        """
        self.medium_primary.append(bar_data)

    # ------------------------------------------------------------------
    # Historical warm-up
    # ------------------------------------------------------------------

    async def warm_up(
        self,
        alpaca_client: Any,
        symbol: str,
        benchmark_symbol: str,
        n_bars: int,
    ) -> bool:
        """Pull historical minute bars and populate MediumBuffer.

        Runs the synchronous Alpaca REST call in a thread executor to remain
        non-blocking.

        Args:
            alpaca_client:    ``StockHistoricalDataClient`` instance.
            symbol:           Primary symbol to warm up.
            benchmark_symbol: Benchmark symbol to warm up.
            n_bars:           Number of bars to request.

        Returns:
            ``True`` when ``is_ready`` is satisfied after loading.
        """
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        def _fetch() -> Any:
            start = datetime.now(timezone.utc) - timedelta(days=4)
            request = StockBarsRequest(
                symbol_or_symbols=[symbol, benchmark_symbol],
                timeframe=TimeFrame.Minute,
                start=start,
                limit=n_bars,
            )
            return alpaca_client.get_stock_bars(request)

        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, _fetch)

            # Populate primary MediumBuffer
            try:
                primary_bars = list(response[symbol])
            except (KeyError, TypeError):
                primary_bars = []

            for bar in primary_bars:
                ts = bar.timestamp
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                vwap_val = float(bar.vwap) if getattr(bar, "vwap", None) else float(bar.close)
                self.medium_primary.append({
                    "timestamp": ts,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                    "vwap_contribution": vwap_val * float(bar.volume),
                })

            # Populate benchmark FastBuffer from recent closes
            try:
                bench_bars = list(response[benchmark_symbol])
            except (KeyError, TypeError):
                bench_bars = []

            for bar in bench_bars[-RS_LOOKBACK_TICKS * 2:]:
                ts = bar.timestamp
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                self.fast_benchmark.append({
                    "timestamp": ts,
                    "symbol": benchmark_symbol,
                    "price": float(bar.close),
                    "bid": None,
                    "ask": None,
                    "bid_size": 0,
                    "ask_size": 0,
                    "volume": float(bar.volume),
                })

            logger.info(
                f"Warm-up: loaded {len(primary_bars)} {symbol} bars, "
                f"{len(bench_bars)} {benchmark_symbol} bars — ready={self.is_ready}"
            )
            return self.is_ready

        except Exception as exc:
            logger.error(f"Warm-up failed: {exc!r}")
            return False

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    def compute_vwap(self) -> Optional[float]:
        """Return cumulative intraday VWAP from session open using FastBuffer data.

        Returns:
            VWAP as float, or ``None`` if insufficient data.
        """
        df = self.fast_primary.to_dataframe()
        if df.empty or "price" not in df.columns or "volume" not in df.columns:
            return None
        valid = df.dropna(subset=["price", "volume"])
        if valid.empty:
            return None
        total_volume = valid["volume"].sum()
        if total_volume == 0:
            return None
        return float((valid["price"] * valid["volume"]).sum() / total_volume)

    def compute_relative_strength(self) -> Optional[float]:
        """Return (primary_return - benchmark_return) over the last N ticks.

        Returns:
            Relative strength as float, or ``None`` if benchmark data unavailable.
        """
        prim_df = self.fast_primary.to_dataframe()
        bench_df = self.fast_benchmark.to_dataframe()
        if prim_df.empty or bench_df.empty:
            return None
        prim_prices = prim_df["price"].dropna().values
        bench_prices = bench_df["price"].dropna().values
        if len(prim_prices) < 2 or len(bench_prices) < 2:
            return None
        n = min(RS_LOOKBACK_TICKS, len(prim_prices), len(bench_prices))
        prim_ret = (prim_prices[-1] / prim_prices[-n]) - 1.0
        bench_ret = (bench_prices[-1] / bench_prices[-n]) - 1.0
        return float(prim_ret - bench_ret)

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """Return True when enough bars are loaded for signal computation.

        In DEV_MODE the requirement is reduced to ``MIN_BARS_DEV``.
        """
        min_bars = MIN_BARS_DEV if config.dev_mode else MIN_BARS_NORMAL
        return len(self.medium_primary) >= min_bars
