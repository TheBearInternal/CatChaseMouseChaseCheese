"""Market regime detection for hft_lab Phase 2.

Combines two independent regime indicators:

* **ADXRegime** — Average Directional Index measures trend strength.
  ADX > 25 signals a trending market regardless of direction.
* **HurstRegime** — Hurst exponent via R/S analysis measures long-range
  memory in log returns.  H > 0.55 trending, H < 0.45 mean-reverting,
  0.45–0.55 random walk.

``RegimeClassifier`` fuses both indicators into one of four ``RegimeState``
values.  The random-walk classification takes priority over ADX because
confirmed random-walk behaviour makes both trend-following and mean-reversion
strategies unreliable.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from config import config
from logger import get_logger
from phase2.data import DataManager

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ADX_PERIOD: int = 14
ADX_TRENDING_THRESHOLD: float = 25.0

HURST_MIN_OBSERVATIONS: int = 50
HURST_TRENDING_THRESHOLD: float = 0.55
HURST_MEAN_REVERT_THRESHOLD: float = 0.45
HURST_RANDOM_WALK_LOW: float = 0.45
HURST_RANDOM_WALK_HIGH: float = 0.55


# ---------------------------------------------------------------------------
# RegimeState enum
# ---------------------------------------------------------------------------


class RegimeState(Enum):
    """Four mutually-exclusive market regime classifications."""

    TRENDING = "TRENDING"
    MEAN_REVERTING = "MEAN_REVERTING"
    RANDOM_WALK = "RANDOM_WALK"
    AMBIGUOUS = "AMBIGUOUS"


# ---------------------------------------------------------------------------
# ADX Regime
# ---------------------------------------------------------------------------


class ADXRegime:
    """Average Directional Index regime indicator.

    Computes full ADX(14) from the MediumBuffer OHLC data using Wilder's
    smoothing method.
    """

    def __init__(self) -> None:
        self._adx_value: float = 0.0

    def update(self, data_manager: DataManager) -> None:
        """Recompute ADX from current MediumBuffer data.

        Args:
            data_manager: DataManager holding the OHLC bars.
        """
        df = data_manager.medium_primary.to_dataframe()
        if len(df) < ADX_PERIOD * 2 + 1:
            self._adx_value = 0.0
            return
        self._adx_value = self._compute_adx(df)

    @staticmethod
    def _compute_adx(df: pd.DataFrame) -> float:
        """Compute ADX(14) from an OHLC DataFrame.

        Args:
            df: DataFrame with columns open, high, low, close.

        Returns:
            Latest ADX value.
        """
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        closes = df["close"].values.astype(float)
        n = len(highs)
        period = ADX_PERIOD

        tr = np.empty(n)
        plus_dm = np.empty(n)
        minus_dm = np.empty(n)
        tr[0] = highs[0] - lows[0]
        plus_dm[0] = 0.0
        minus_dm[0] = 0.0

        for i in range(1, n):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i - 1])
            lc = abs(lows[i] - closes[i - 1])
            tr[i] = max(hl, hc, lc)

            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]
            plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0.0
            minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0.0

        # Wilder smoothing: EWM with alpha = 1 / period
        alpha = 1.0 / period
        atr = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().values
        pdm_s = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().values
        mdm_s = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().values

        with np.errstate(divide="ignore", invalid="ignore"):
            plus_di = np.where(atr > 0, 100.0 * pdm_s / atr, 0.0)
            minus_di = np.where(atr > 0, 100.0 * mdm_s / atr, 0.0)
            dx_denom = plus_di + minus_di
            dx = np.where(dx_denom > 0, 100.0 * np.abs(plus_di - minus_di) / dx_denom, 0.0)

        adx = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().values
        return float(adx[-1])

    @property
    def adx_value(self) -> float:
        """Most recently computed ADX value."""
        return self._adx_value

    @property
    def is_trending(self) -> bool:
        """True when ADX exceeds ``ADX_TRENDING_THRESHOLD`` (25)."""
        return self._adx_value > ADX_TRENDING_THRESHOLD


# ---------------------------------------------------------------------------
# Hurst Regime
# ---------------------------------------------------------------------------


class HurstRegime:
    """Hurst exponent via Rescaled Range (R/S) analysis on log returns.

    Requires at least ``HURST_MIN_OBSERVATIONS`` data points to produce a
    valid estimate.
    """

    def __init__(self) -> None:
        self._hurst_value: float = 0.5

    def update(self, data_manager: DataManager) -> None:
        """Recompute the Hurst exponent from current MediumBuffer data.

        Args:
            data_manager: DataManager holding the bar data.
        """
        df = data_manager.medium_primary.to_dataframe()
        if len(df) < HURST_MIN_OBSERVATIONS:
            self._hurst_value = 0.5
            return
        closes = df["close"].values.astype(float)
        log_returns = np.diff(np.log(closes + 1e-10))
        self._hurst_value = self._compute_hurst(log_returns)

    @staticmethod
    def _compute_hurst(returns: np.ndarray) -> float:
        """Estimate Hurst exponent using the R/S method.

        Divides the returns series into sub-windows of decreasing size,
        computes R/S for each, and fits a log-log regression.

        Args:
            returns: 1-D array of log returns.

        Returns:
            Estimated Hurst exponent, clamped to [0.01, 0.99].
        """
        n = len(returns)
        window_sizes = []
        rs_means = []

        for divisor in [2, 4, 8, 16]:
            w = n // divisor
            if w < 10:
                continue
            rs_list = []
            for start in range(0, n - w + 1, w):
                seg = returns[start: start + w]
                mean_seg = np.mean(seg)
                deviations = np.cumsum(seg - mean_seg)
                R = np.max(deviations) - np.min(deviations)
                S = np.std(seg, ddof=1)
                if S > 0:
                    rs_list.append(R / S)
            if rs_list:
                rs_means.append(float(np.mean(rs_list)))
                window_sizes.append(w)

        if len(window_sizes) < 2:
            return 0.5

        log_n = np.log(np.array(window_sizes, dtype=float))
        log_rs = np.log(np.array(rs_means, dtype=float))
        H = float(np.polyfit(log_n, log_rs, 1)[0])
        return float(np.clip(H, 0.01, 0.99))

    @property
    def hurst_value(self) -> float:
        """Most recently computed Hurst exponent."""
        return self._hurst_value


# ---------------------------------------------------------------------------
# RegimeClassifier
# ---------------------------------------------------------------------------


class RegimeClassifier:
    """Fuses ADX and Hurst indicators into a single ``RegimeState``.

    Classification logic
    --------------------
    1. If Hurst in [0.45, 0.55] → RANDOM_WALK (overrides ADX).
    2. ADX trending AND Hurst > 0.55 → TRENDING.
    3. ADX not trending AND Hurst < 0.45 → MEAN_REVERTING.
    4. All other combinations → AMBIGUOUS.
    """

    def __init__(self) -> None:
        self._adx = ADXRegime()
        self._hurst = HurstRegime()
        self._last_state: Optional[RegimeState] = None

    def classify(self, data_manager: DataManager) -> RegimeState:
        """Classify the current market regime.

        Logs at INFO level whenever the state changes.

        Args:
            data_manager: DataManager holding the latest OHLC data.

        Returns:
            Current ``RegimeState``.
        """
        self._adx.update(data_manager)
        self._hurst.update(data_manager)

        h = self._hurst.hurst_value
        adx_trending = self._adx.is_trending

        if HURST_RANDOM_WALK_LOW <= h <= HURST_RANDOM_WALK_HIGH:
            state = RegimeState.RANDOM_WALK
        elif adx_trending and h > HURST_TRENDING_THRESHOLD:
            state = RegimeState.TRENDING
        elif not adx_trending and h < HURST_MEAN_REVERT_THRESHOLD:
            state = RegimeState.MEAN_REVERTING
        else:
            state = RegimeState.AMBIGUOUS

        if state != self._last_state:
            logger.info(
                f"Regime changed: {self._last_state} → {state.value} "
                f"(ADX={self._adx.adx_value:.1f} H={h:.3f})"
            )
            self._last_state = state

        return state

    @property
    def adx(self) -> ADXRegime:
        """Underlying ADX indicator."""
        return self._adx

    @property
    def hurst(self) -> HurstRegime:
        """Underlying Hurst indicator."""
        return self._hurst
