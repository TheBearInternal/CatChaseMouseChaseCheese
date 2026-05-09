"""Six-dimensional signal engine for hft_lab Phase 2.

Each signal class ingests a ``DataManager`` instance and produces a score in
the range ``[-1.0, +1.0]`` where positive values are bullish and negative
values are bearish.  ``0.0`` is always returned when there is insufficient data
to produce a meaningful estimate.

Signal classes
--------------
MACDSignal            : MACD histogram normalized via tanh.
RSISignal             : RSI(14) normalized to [-1, +1] with exhaustion damping.
BollingerSignal       : %B mean-reversion score, polarity inverts in trends.
VWAPSignal            : Intraday VWAP deviation via tanh, polarity inverts in trends.
OrderBookSignal       : Bid/ask size imbalance (L2) or Lee-Ready tick-rule proxy.
RelativeStrengthSignal: Asset return minus benchmark return via tanh.

``SignalEngine`` runs all six and returns a dict of scores.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import config
from logger import get_logger
from phase2.data import DataManager

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
MACD_NORMALIZER_WINDOW: int = 20   # recent histogram range for adaptive normalizer

RSI_PERIOD: int = 14
RSI_OVERBOUGHT: float = 80.0
RSI_OVERSOLD: float = 20.0
RSI_EXHAUSTION_DAMP: float = 0.30

BOLLINGER_PERIOD: int = 20
BOLLINGER_STD: float = 2.0

VWAP_TANH_SCALE: float = 0.005      # fraction of price used to normalize VWAP deviation

RS_TANH_SCALE: float = 0.002        # fraction used to normalize relative strength


# ---------------------------------------------------------------------------
# MACD Signal
# ---------------------------------------------------------------------------


class MACDSignal:
    """MACD histogram normalized to [-1, +1] via tanh with an adaptive normalizer."""

    def __init__(self, data_manager: DataManager) -> None:
        """Initialise with a reference to the shared DataManager.

        Args:
            data_manager: Shared DataManager instance.
        """
        self._dm = data_manager

    def compute(self, regime_state: Optional[object] = None) -> float:
        """Compute and return the MACD signal score.

        Args:
            regime_state: Unused; present for uniform interface.

        Returns:
            Float in [-1.0, +1.0], or 0.0 if insufficient data.
        """
        df = self._dm.medium_primary.to_dataframe()
        if len(df) < MACD_SLOW + MACD_SIGNAL + 2:
            return 0.0
        closes = df["close"].astype(float)
        ema_fast = closes.ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = closes.ewm(span=MACD_SLOW, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
        histogram = macd_line - signal_line

        # Adaptive normalizer: std of recent histogram values
        recent = histogram.iloc[-MACD_NORMALIZER_WINDOW:]
        normalizer = float(recent.std()) if len(recent) >= 4 else 1.0
        if normalizer < 1e-8:
            normalizer = 1e-8

        score = math.tanh(float(histogram.iloc[-1]) / normalizer)
        return float(np.clip(score, -1.0, 1.0))


# ---------------------------------------------------------------------------
# RSI Signal
# ---------------------------------------------------------------------------


class RSISignal:
    """Standard RSI(14) normalized to [-1, +1] with exhaustion dampening."""

    def __init__(self, data_manager: DataManager) -> None:
        """Initialise with a reference to the shared DataManager.

        Args:
            data_manager: Shared DataManager instance.
        """
        self._dm = data_manager

    def compute(self, regime_state: Optional[object] = None) -> float:
        """Compute and return the RSI signal score.

        RSI > 80 or RSI < 20 triggers a 30% magnitude reduction to account for
        momentum exhaustion.

        Args:
            regime_state: Unused; present for uniform interface.

        Returns:
            Float in [-1.0, +1.0], or 0.0 if insufficient data.
        """
        df = self._dm.medium_primary.to_dataframe()
        if len(df) < RSI_PERIOD + 2:
            return 0.0
        closes = df["close"].astype(float)
        delta = closes.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(span=RSI_PERIOD, adjust=False).mean()
        avg_loss = loss.ewm(span=RSI_PERIOD, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        current_rsi = float(rsi.iloc[-1])

        score = (current_rsi - 50.0) / 50.0
        if current_rsi > RSI_OVERBOUGHT or current_rsi < RSI_OVERSOLD:
            score *= (1.0 - RSI_EXHAUSTION_DAMP)
        return float(np.clip(score, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Bollinger Signal
# ---------------------------------------------------------------------------


class BollingerSignal:
    """Bollinger %B mean-reversion score; polarity inverts in trending regime."""

    def __init__(self, data_manager: DataManager) -> None:
        """Initialise with a reference to the shared DataManager.

        Args:
            data_manager: Shared DataManager instance.
        """
        self._dm = data_manager

    def compute(self, regime_state: Optional[object] = None) -> float:
        """Compute and return the Bollinger Band signal score.

        In mean-reversion framing: price near lower band → bullish (+1),
        price near upper band → bearish (-1).  In a trending regime the
        polarity is inverted so that breakouts are followed rather than faded.

        Args:
            regime_state: Current ``RegimeState`` enum value; used to invert
                          polarity when trending.

        Returns:
            Float in [-1.0, +1.0], or 0.0 if insufficient data.
        """
        df = self._dm.medium_primary.to_dataframe()
        if len(df) < BOLLINGER_PERIOD + 2:
            return 0.0
        closes = df["close"].astype(float)
        sma = closes.rolling(BOLLINGER_PERIOD).mean()
        std = closes.rolling(BOLLINGER_PERIOD).std()
        upper = sma + BOLLINGER_STD * std
        lower = sma - BOLLINGER_STD * std
        band_range = float(upper.iloc[-1]) - float(lower.iloc[-1])
        if band_range < 1e-8:
            return 0.0
        pct_b = (float(closes.iloc[-1]) - float(lower.iloc[-1])) / band_range
        score = 1.0 - (2.0 * pct_b)   # mean-reversion: above mid → bearish

        if regime_state is not None:
            from phase2.regime import RegimeState
            if regime_state == RegimeState.TRENDING:
                score = -score  # follow the trend instead
        return float(np.clip(score, -1.0, 1.0))


# ---------------------------------------------------------------------------
# VWAP Signal
# ---------------------------------------------------------------------------


class VWAPSignal:
    """Intraday VWAP deviation via tanh; polarity inverts in strong trending regime."""

    def __init__(self, data_manager: DataManager) -> None:
        """Initialise with a reference to the shared DataManager.

        Args:
            data_manager: Shared DataManager instance.
        """
        self._dm = data_manager

    def compute(self, regime_state: Optional[object] = None) -> float:
        """Compute and return the VWAP signal score.

        Mean-reversion framing: price above VWAP → bearish pull, price below →
        bullish pull.  Polarity inverts in TRENDING regime.

        Args:
            regime_state: Current ``RegimeState``; triggers polarity inversion
                          when TRENDING.

        Returns:
            Float in [-1.0, +1.0], or 0.0 if insufficient intraday data.
        """
        vwap = self._dm.compute_vwap()
        price = self._dm.fast_primary.latest_price()
        if vwap is None or price is None or vwap < 1e-8:
            return 0.0
        deviation = (price - vwap) / vwap
        score = -math.tanh(deviation / VWAP_TANH_SCALE)  # above VWAP = negative

        if regime_state is not None:
            from phase2.regime import RegimeState
            if regime_state == RegimeState.TRENDING:
                score = -score
        return float(np.clip(score, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Order Book Signal
# ---------------------------------------------------------------------------


TICK_RULE_WINDOW: int = 50    # number of recent tick directions to sum
TICK_RULE_NORMALIZER: float = float(TICK_RULE_WINDOW)


class OrderBookSignal:
    """Bid/ask size imbalance with Lee-Ready tick-rule fallback.

    When Level 2 book depth is available (bid_size > 0 and ask_size > 0),
    the standard imbalance formula is used::

        score = (bid_size - ask_size) / (bid_size + ask_size)

    On free data feeds where bid_size and ask_size are always 0, the signal
    falls back to a tick-rule proxy: the sum of the last 50 tick directions
    (+1 uptick / -1 downtick / 0 no change) divided by 50.  This gives a
    genuine volume-flow pressure reading from trade prints alone.
    """

    def __init__(self, data_manager: DataManager) -> None:
        """Initialise with a reference to the shared DataManager.

        Args:
            data_manager: Shared DataManager instance.
        """
        self._dm = data_manager
        self._tick_rule_logged: bool = False

    def compute(self, regime_state: Optional[object] = None) -> float:
        """Compute and return the order book imbalance score.

        Tries Level 2 bid/ask sizes first.  Falls back to the Lee-Ready
        tick-rule proxy when L2 data is unavailable.

        Args:
            regime_state: Unused; present for uniform interface.

        Returns:
            Float in [-1.0, +1.0], or 0.0 if no tick history exists yet.
        """
        df = self._dm.fast_primary.to_dataframe()
        if not df.empty and "bid_size" in df.columns and "ask_size" in df.columns:
            latest = df.dropna(subset=["bid_size", "ask_size"]).tail(1)
            if not latest.empty:
                bid_sz = float(latest["bid_size"].iloc[-1])
                ask_sz = float(latest["ask_size"].iloc[-1])
                total = bid_sz + ask_sz
                if total >= 1e-8:
                    return float(np.clip((bid_sz - ask_sz) / total, -1.0, 1.0))

        # L2 data unavailable — use tick-rule proxy
        if not self._tick_rule_logged:
            logger.debug("OrderBook | using tick-rule proxy (no L2 data)")
            self._tick_rule_logged = True

        history = self._dm._tick_history
        if not history:
            return 0.0
        recent = list(history)[-TICK_RULE_WINDOW:]
        direction_sum = sum(d for _, d in recent)
        return float(np.clip(direction_sum / TICK_RULE_NORMALIZER, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Relative Strength Signal
# ---------------------------------------------------------------------------


class RelativeStrengthSignal:
    """Asset return minus benchmark return, normalized via tanh."""

    def __init__(self, data_manager: DataManager) -> None:
        """Initialise with a reference to the shared DataManager.

        Args:
            data_manager: Shared DataManager instance.
        """
        self._dm = data_manager

    def compute(self, regime_state: Optional[object] = None) -> float:
        """Compute and return the relative strength signal score.

        Args:
            regime_state: Unused; present for uniform interface.

        Returns:
            Float in [-1.0, +1.0], or 0.0 if benchmark data unavailable.
        """
        rs = self._dm.compute_relative_strength()
        if rs is None:
            return 0.0
        return float(np.clip(math.tanh(rs / RS_TANH_SCALE), -1.0, 1.0))


# ---------------------------------------------------------------------------
# Signal Engine
# ---------------------------------------------------------------------------

SIGNAL_NAMES: tuple[str, ...] = (
    "macd",
    "rsi",
    "bollinger",
    "vwap",
    "order_book",
    "relative_strength",
)


class SignalEngine:
    """Instantiates and runs all six signal classes.

    All signals share the same ``DataManager`` reference so they read from
    consistent buffers on each call.
    """

    def __init__(self, data_manager: DataManager) -> None:
        """Initialise all signal instances.

        Args:
            data_manager: Shared DataManager instance.
        """
        self._macd = MACDSignal(data_manager)
        self._rsi = RSISignal(data_manager)
        self._bollinger = BollingerSignal(data_manager)
        self._vwap = VWAPSignal(data_manager)
        self._order_book = OrderBookSignal(data_manager)
        self._rs = RelativeStrengthSignal(data_manager)

    def compute_all(self, regime_state: Optional[object] = None) -> Dict[str, float]:
        """Compute all six signal scores and return them as a named dict.

        All scores are logged at DEBUG level.

        Args:
            regime_state: Current ``RegimeState`` forwarded to signals that
                          use it for polarity decisions.

        Returns:
            Dict mapping signal name to score in [-1.0, +1.0].
        """
        scores: Dict[str, float] = {
            "macd":             self._macd.compute(regime_state),
            "rsi":              self._rsi.compute(regime_state),
            "bollinger":        self._bollinger.compute(regime_state),
            "vwap":             self._vwap.compute(regime_state),
            "order_book":       self._order_book.compute(regime_state),
            "relative_strength": self._rs.compute(regime_state),
        }
        logger.debug(
            "Signal scores | " +
            " ".join(f"{k}={v:+.3f}" for k, v in scores.items())
        )
        return scores
