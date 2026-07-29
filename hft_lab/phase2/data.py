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
TICK_HISTORY_MAXLEN: int = 100   # rolling window for Lee-Ready tick-rule proxy


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
        # Window scales with the configured bar timeframe so capacity stays
        # ~MEDIUM_BUFFER_WINDOW_MINUTES bars regardless of BAR_TIMEFRAME
        cutoff = datetime.now(timezone.utc) - timedelta(
            minutes=MEDIUM_BUFFER_WINDOW_MINUTES * config.bar_timeframe_minutes()
        )
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
            # Positions are always sourced from OANDA on startup; never restore from file
            self.data.pop("open_positions", None)
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
        self._hist_bars: deque[Dict[str, Any]] = deque(maxlen=FAST_BUFFER_MAXLEN)
        self._tick_history: deque[tuple[float, int]] = deque(maxlen=TICK_HISTORY_MAXLEN)
        self._last_tick_price: Optional[float] = None
        # Bar-close snapshots awaiting forward-return maturity for IC scoring:
        # (timestamp, mid_price, signal_scores_at_t, regime_at_t)
        self.ic_snapshots: deque[tuple] = deque()
        # Live 1-minute bar aggregation from ticks (fills MediumBuffer after
        # warm-up bars age out; nothing else feeds live bars)
        self._current_bar: Optional[Dict[str, Any]] = None
        self._current_bar_minute: Optional[datetime] = None
        self._is_crypto: bool = config.is_crypto()
        self._is_forex: bool = config.is_forex()
        self._vwap_proxy_logged: bool = False  # log forex tick-volume note once
        _mode = (
            "CRYPTO (24/7 rolling VWAP)" if self._is_crypto else
            "FOREX (24h rolling VWAP, tick-count volume)" if self._is_forex else
            "EQUITY (session VWAP)"
        )
        logger.info(f"DataManager initialised | mode={_mode}")

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    async def ingest_tick(self, symbol: str, tick_data: Dict[str, Any]) -> None:
        """Route an incoming tick to the appropriate FastBuffer.

        For the primary symbol, also records a Lee-Ready tick direction into
        ``_tick_history`` as a ``(price, direction)`` tuple where direction is
        ``+1`` (uptick), ``-1`` (downtick), or ``0`` (no change).

        Args:
            symbol:    Symbol string matching PRIMARY_SYMBOL or BENCHMARK_SYMBOL.
            tick_data: Dict with tick fields (price, bid, ask, etc.).
        """
        if symbol == config.primary_symbol:
            self.fast_primary.append(tick_data)
            if self._session_open_ts is None:
                self._session_open_ts = datetime.now(timezone.utc)

            price: Optional[float] = tick_data.get("price")
            if price is not None:
                if self._last_tick_price is None:
                    direction = 0
                elif price > self._last_tick_price:
                    direction = 1
                elif price < self._last_tick_price:
                    direction = -1
                else:
                    direction = 0
                self._tick_history.append((float(price), direction))
                self._last_tick_price = float(price)
                self._aggregate_tick_into_bar(tick_data, float(price))

        elif symbol == config.benchmark_symbol:
            self.fast_benchmark.append(tick_data)

    def _aggregate_tick_into_bar(
        self, tick_data: Dict[str, Any], price: float
    ) -> None:
        """Roll primary-symbol ticks into live 1-minute OHLCV bars.

        On each minute rollover the completed bar is appended to
        ``medium_primary`` and ``_hist_bars`` — the only live source of new
        bars for the bar-based signals and bar-close IC scoring.

        Args:
            tick_data: Raw tick dict (timestamp/volume fields used).
            price:     Tick price, already validated non-None.
        """
        ts = tick_data.get("timestamp") or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        # Floor to the configured bar timeframe (epoch-aligned buckets)
        bucket_s = config.bar_timeframe_minutes() * 60
        epoch = int(ts.timestamp())
        minute = datetime.fromtimestamp(epoch - (epoch % bucket_s), tz=timezone.utc)
        volume = float(tick_data.get("volume") or 1.0)

        if self._current_bar is None or self._current_bar_minute is None:
            self._current_bar_minute = minute
            self._current_bar = {
                "timestamp": minute,
                "open": price, "high": price, "low": price, "close": price,
                "volume": volume,
                "vwap_contribution": price * volume,
            }
            return

        if minute > self._current_bar_minute:
            completed = self._current_bar
            self.medium_primary.append(completed)
            self._hist_bars.append(completed)
            self._current_bar_minute = minute
            self._current_bar = {
                "timestamp": minute,
                "open": price, "high": price, "low": price, "close": price,
                "volume": volume,
                "vwap_contribution": price * volume,
            }
            return

        bar = self._current_bar
        bar["high"] = max(bar["high"], price)
        bar["low"] = min(bar["low"], price)
        bar["close"] = price
        bar["volume"] += volume
        bar["vwap_contribution"] += price * volume

    async def ingest_bar(self, bar_data: Dict[str, Any]) -> None:
        """Append a completed 1-minute bar to the MediumBuffer and _hist_bars.

        Args:
            bar_data: Dict with OHLCV fields and a ``timestamp`` key.
        """
        self.medium_primary.append(bar_data)
        self._hist_bars.append(bar_data)

    def record_ic_snapshot(
        self, price: float, signal_scores: Dict[str, float], regime: Any
    ) -> List[tuple]:
        """Store a bar-close snapshot and return matured forward-return events.

        Each snapshot holds the signal scores and regime *as computed at that
        bar* — they are never recomputed later.  Once a snapshot is
        ``config.ic_forward_bars`` bars old (one snapshot is pushed per bar
        close), its realized forward return is computed against the current
        price and the snapshot is emitted for scoring.

        Args:
            price:         Bar-close mid price at time t.
            signal_scores: Signal scores computed at time t.
            regime:        Regime classified at time t (opaque to DataManager).

        Returns:
            List of ``(signal_scores_at_t, regime_at_t, forward_return)``
            events that matured with this push (usually 0 or 1).
        """
        events: List[tuple] = []
        self.ic_snapshots.append(
            (datetime.now(timezone.utc), float(price), dict(signal_scores), regime)
        )
        while len(self.ic_snapshots) > config.ic_forward_bars:
            _ts, p0, s0, r0 = self.ic_snapshots.popleft()
            if p0 > 1e-9:
                events.append((s0, r0, (float(price) - p0) / p0))
        return events

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
        if config.is_forex():
            return await self._warm_up_forex(symbol, benchmark_symbol, n_bars)

        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        tf_minutes = config.bar_timeframe_minutes()
        if tf_minutes >= 60 and tf_minutes % 60 == 0:
            bar_tf = TimeFrame(tf_minutes // 60, TimeFrameUnit.Hour)
        else:
            bar_tf = TimeFrame(tf_minutes, TimeFrameUnit.Minute)
        # Lookback must cover n_bars at the configured timeframe (×3 margin
        # for weekends/closed hours)
        lookback_days = max(4, (n_bars * tf_minutes * 3) // (60 * 24) + 1)

        def _fetch() -> Any:
            start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            if config.is_crypto():
                from alpaca.data.historical import CryptoHistoricalDataClient
                from alpaca.data.requests import CryptoBarsRequest
                client = CryptoHistoricalDataClient()  # no auth required
                request = CryptoBarsRequest(
                    symbol_or_symbols=[symbol, benchmark_symbol],
                    timeframe=bar_tf,
                    start=start,
                    limit=n_bars,
                )
                return client.get_crypto_bars(request)
            else:
                from alpaca.data.historical import StockHistoricalDataClient
                from alpaca.data.requests import StockBarsRequest
                client = StockHistoricalDataClient(
                    api_key=config.alpaca_api_key,
                    secret_key=config.alpaca_secret_key,
                )
                request = StockBarsRequest(
                    symbol_or_symbols=[symbol, benchmark_symbol],
                    timeframe=bar_tf,
                    start=start,
                    limit=n_bars,
                )
                return client.get_stock_bars(request)

        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, _fetch)

            # Extract raw bar lists from the multi-symbol BarSet.
            # alpaca-py exposes a `.data` dict keyed by symbol; fall back to
            # direct __getitem__ access for older SDK versions.
            try:
                raw_data: dict = dict(response.data)
            except (AttributeError, TypeError):
                raw_data = {}

            def _extract(sym: str) -> list:
                if sym in raw_data:
                    return list(raw_data[sym])
                try:
                    return list(response[sym])
                except (KeyError, TypeError):
                    return []

            primary_bars = _extract(symbol)
            bench_bars_all = _extract(benchmark_symbol)

            # Populate _hist_bars (persists regardless of timestamp age) and
            # medium_primary (live rolling window — historical bars may be evicted).
            for bar in primary_bars:
                ts = bar.timestamp
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                vwap_val = float(bar.vwap) if getattr(bar, "vwap", None) else float(bar.close)
                bar_dict = {
                    "timestamp": ts,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                    "vwap_contribution": vwap_val * float(bar.volume),
                }
                self._hist_bars.append(bar_dict)
                self.medium_primary.append(bar_dict)

            # Populate benchmark FastBuffer from recent closes
            if not bench_bars_all:
                logger.warning(
                    f"0 {benchmark_symbol} bars loaded — benchmark symbol may be "
                    "unavailable on the IEX free data feed (NYSE Arca not covered)"
                )
            bench_bars = bench_bars_all
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

            bench_note = (
                f"{len(bench_bars)} {benchmark_symbol} bars"
                if bench_bars
                else f"0 {benchmark_symbol} bars (benchmark unavailable on IEX feed)"
            )
            logger.info(
                f"Warm-up: loaded {len(primary_bars)} {symbol} bars, "
                f"{bench_note} — ready={self.is_ready}"
            )
            return self.is_ready

        except Exception as exc:
            logger.error(f"Warm-up failed: {exc!r}")
            return False

    async def _warm_up_forex(
        self, symbol: str, benchmark_symbol: str, n_bars: int
    ) -> bool:
        """Warm up from OANDA REST API historical candles (forex mode).

        Args:
            symbol:           Primary instrument, e.g. ``"EUR_USD"``.
            benchmark_symbol: Benchmark instrument, e.g. ``"GBP_USD"``.
            n_bars:           Number of candles to request.

        Returns:
            ``True`` when ``is_ready`` is satisfied after loading.
        """
        from phase2.forex import OANDAHistoricalFetcher

        def _fetch() -> tuple:
            fetcher = OANDAHistoricalFetcher(
                config.oanda_api_key,
                config.oanda_account_id,
                config.oanda_environment,
            )
            granularity = config.oanda_granularity()
            return (
                fetcher.fetch_candles(symbol, count=n_bars, granularity=granularity),
                fetcher.fetch_candles(
                    benchmark_symbol, count=n_bars, granularity=granularity
                ),
            )

        try:
            loop = asyncio.get_running_loop()
            primary_bars, bench_bars = await loop.run_in_executor(None, _fetch)

            for bar in primary_bars:
                self._hist_bars.append(bar)
                self.medium_primary.append(bar)

            if not bench_bars:
                logger.warning(
                    f"0 {benchmark_symbol} forex bars loaded — "
                    "check OANDA instrument availability"
                )

            for bar in bench_bars[-RS_LOOKBACK_TICKS * 2:]:
                self.fast_benchmark.append({
                    "timestamp": bar["timestamp"],
                    "symbol": benchmark_symbol,
                    "price": bar["close"],
                    "bid": None,
                    "ask": None,
                    "bid_size": 0,
                    "ask_size": 0,
                    "volume": bar["volume"],
                })

            bench_note = (
                f"{len(bench_bars)} {benchmark_symbol} bars"
                if bench_bars
                else f"0 {benchmark_symbol} bars (unavailable)"
            )
            logger.info(
                f"FOREX Warm-up: loaded {len(primary_bars)} {symbol} bars, "
                f"{bench_note} — ready={self.is_ready}"
            )
            return self.is_ready

        except Exception as exc:
            logger.error(f"FOREX warm-up failed: {exc!r}")
            return False

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    def compute_vwap(self) -> Optional[float]:
        """Return VWAP from tick data.

        * Equity: cumulative from session open (resets daily at 09:30 EST).
        * Crypto / Forex: rolling 24-hour window.  For forex, tick count
          (``volume=1`` per tick) is used as the volume proxy; a one-time
          DEBUG message is logged on first call to make this explicit.

        Returns:
            VWAP as float, or ``None`` if insufficient data.
        """
        df = self.fast_primary.to_dataframe()
        if df.empty or "price" not in df.columns or "volume" not in df.columns:
            return None

        if "timestamp" in df.columns:
            if self._is_crypto or self._is_forex:
                if self._is_forex and not self._vwap_proxy_logged:
                    logger.debug(
                        "FOREX mode: using tick count as VWAP volume proxy"
                    )
                    self._vwap_proxy_logged = True
                cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
                df = df[df["timestamp"] >= cutoff]
            elif self._session_open_ts is not None:
                df = df[df["timestamp"] >= self._session_open_ts]

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

    def to_dataframe(self) -> pd.DataFrame:
        """Return bar data as a DataFrame, preferring live MediumBuffer.

        Falls back to ``_hist_bars`` when the MediumBuffer has been time-evicted
        below the readiness threshold (e.g. at session start before new bars arrive).

        Returns:
            DataFrame with one row per bar, or empty DataFrame if no data.
        """
        min_bars = MIN_BARS_DEV if config.dev_mode else MIN_BARS_NORMAL
        if len(self.medium_primary) >= min_bars:
            return self.medium_primary.to_dataframe()
        if self._hist_bars:
            return pd.DataFrame(list(self._hist_bars))
        return pd.DataFrame()

    @property
    def is_ready(self) -> bool:
        """Return True when enough primary bars are loaded for signal computation.

        Checks ``_hist_bars`` (timestamp-independent) so warm-up bars are counted
        even after MediumBuffer's 90-minute window evicts them.  Benchmark data
        is supplementary and does not gate readiness.

        In DEV_MODE the requirement is reduced to ``MIN_BARS_DEV``.
        """
        min_bars = MIN_BARS_DEV if config.dev_mode else MIN_BARS_NORMAL
        return len(self._hist_bars) >= min_bars
