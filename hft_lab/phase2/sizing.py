"""Position sizing and risk management for hft_lab Phase 2.

Three classes work together to determine how large a position to take and
whether the trade economics justify entering at all:

``GARCHSizer``    — GARCH(1,1) volatility model; periodic MLE re-fit via scipy.
``ATRSizer``      — ATR(14) stop and take-profit distances.
``SpreadAdjuster``— Spread cost analysis and viability check.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import config
from logger import get_logger
from phase2.data import DataManager

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# GARCH initial parameters (sum < 1 ensures stationarity)
GARCH_OMEGA_INIT: float = 0.00001
GARCH_ALPHA_INIT: float = 0.10
GARCH_BETA_INIT: float = 0.85

# Bounds for MLE optimization
GARCH_OMEGA_BOUNDS: tuple = (1e-9, 0.01)
GARCH_ALPHA_BOUNDS: tuple = (0.0, 0.50)
GARCH_BETA_BOUNDS: tuple = (0.0, 0.99)
GARCH_STATIONARITY_MAX: float = 0.9999

# Confidence scalar ceiling
CONFIDENCE_SCALAR_MAX: float = 1.5

# Minimum variance floor to prevent division by zero
VARIANCE_FLOOR: float = 1e-10


# ---------------------------------------------------------------------------
# GARCHSizer
# ---------------------------------------------------------------------------


class GARCHSizer:
    """GARCH(1,1) volatility model for adaptive position sizing.

    The model is updated recursively on each new return observation via
    ``update()``.  Periodically, ``fit()`` re-estimates ω, α, β via MLE on
    the full return history from MediumBuffer to correct for parameter drift.
    """

    def __init__(self, data_manager: DataManager) -> None:
        """Initialise with GARCH parameters and a DataManager reference.

        Args:
            data_manager: DataManager supplying OHLC returns for fitting.
        """
        self._dm = data_manager
        self._omega: float = GARCH_OMEGA_INIT
        self._alpha: float = GARCH_ALPHA_INIT
        self._beta: float = GARCH_BETA_INIT
        self._sigma2: float = GARCH_OMEGA_INIT / (1.0 - GARCH_ALPHA_INIT - GARCH_BETA_INIT)
        self._last_fit_ts: float = 0.0

    def update(self, return_value: float) -> None:
        """Update the GARCH variance estimate with one new return observation.

        σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}

        Args:
            return_value: Most recent log return or percent return.
        """
        self._sigma2 = (
            self._omega
            + self._alpha * return_value ** 2
            + self._beta * self._sigma2
        )
        self._sigma2 = max(self._sigma2, VARIANCE_FLOOR)
        logger.debug(f"GARCH update: σ²={self._sigma2:.8f}")

    def fit(self, returns_series: pd.Series) -> None:
        """Re-fit ω, α, β via MLE on a returns series.

        Uses scipy L-BFGS-B with stationarity constraint α + β < 1.  Keeps
        current parameters if the optimizer fails or produces a non-stationary
        solution.

        Args:
            returns_series: Pandas Series of return values.
        """
        returns = returns_series.dropna().values
        if len(returns) < 30:
            logger.debug("GARCH fit: insufficient data — keeping current parameters")
            return

        def _neg_loglik(params: np.ndarray) -> float:
            omega, alpha, beta = params
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= GARCH_STATIONARITY_MAX:
                return 1e12
            sigma2 = np.var(returns)
            total = 0.0
            for i in range(1, len(returns)):
                sigma2 = omega + alpha * returns[i - 1] ** 2 + beta * sigma2
                if sigma2 <= 0:
                    return 1e12
                total += np.log(sigma2) + returns[i] ** 2 / sigma2
            return total

        try:
            result = minimize(
                _neg_loglik,
                x0=np.array([self._omega, self._alpha, self._beta]),
                method="L-BFGS-B",
                bounds=[GARCH_OMEGA_BOUNDS, GARCH_ALPHA_BOUNDS, GARCH_BETA_BOUNDS],
                options={"maxiter": 300, "ftol": 1e-8},
            )
            if result.success:
                omega, alpha, beta = result.x
                if alpha + beta < GARCH_STATIONARITY_MAX:
                    self._omega = float(omega)
                    self._alpha = float(alpha)
                    self._beta = float(beta)
                    logger.info(
                        f"GARCH fit: ω={omega:.2e} α={alpha:.4f} β={beta:.4f}"
                    )
                else:
                    logger.warning("GARCH fit: stationarity violated — keeping prior params")
            else:
                logger.warning(f"GARCH fit did not converge: {result.message}")
        except Exception as exc:
            logger.warning(f"GARCH fit error: {exc!r}")

        self._last_fit_ts = time.time()

    def forecast_volatility(self) -> float:
        """Return one-step-ahead volatility forecast (σ, not σ²).

        Returns:
            Standard deviation forecast.
        """
        return float(np.sqrt(max(self._sigma2, VARIANCE_FLOOR)))

    def compute_position_size(
        self,
        account_equity: float,
        max_risk_pct: float,
        confidence: float,
        current_price: float,
        profile_variance: float,
    ) -> int:
        """Return an integer share count sized to current volatility and confidence.

        Formula
        -------
        base_risk            = account_equity × max_risk_pct
        volatility_scalar    = 1 / (1 + forecast_volatility × 100)
        confidence_scalar    = min(confidence / threshold, 1.5)
        raw_size             = base_risk × volatility_scalar × confidence_scalar / price
        Applies randomize_quantity from behavior with profile_variance.

        Args:
            account_equity:    Current account equity in dollars.
            max_risk_pct:      Maximum fraction of equity to risk.
            confidence:        Decision confidence from EnsembleDecision.
            current_price:     Current asset price.
            profile_variance:  order_size_variance from BehaviorProfile.

        Returns:
            Final integer share count, minimum MIN_POSITION_SIZE.
        """
        from phase1.behavior import randomize_quantity

        if current_price < 1e-6:
            return config.min_position_size

        effective_equity = min(account_equity, config.account_limit)
        base_risk = effective_equity * max_risk_pct
        vol = self.forecast_volatility()
        volatility_scalar = 1.0 / (1.0 + vol * 100.0)
        confidence_scalar = min(confidence / config.ensemble_threshold, CONFIDENCE_SCALAR_MAX)

        raw_size = (base_risk * volatility_scalar * confidence_scalar) / current_price
        candidate = max(config.min_position_size, round(raw_size))

        # Enforce account exposure limit
        max_shares = int(config.account_limit / current_price)
        candidate = min(candidate, max(config.min_position_size, max_shares))

        # Apply behavioral randomization
        candidate = randomize_quantity(candidate, variance_pct=profile_variance)
        logger.debug(
            f"PositionSize: equity={effective_equity:.0f} "
            f"(capped from {account_equity:.0f}) vol={vol:.4f} "
            f"conf={confidence:.3f} → {candidate} shares"
        )
        return candidate

    @property
    def should_refit(self) -> bool:
        """True when GARCH_UPDATE_INTERVAL seconds have elapsed since last fit."""
        return (time.time() - self._last_fit_ts) >= config.garch_update_interval


# ---------------------------------------------------------------------------
# ATRSizer
# ---------------------------------------------------------------------------


class ATRSizer:
    """Average True Range indicator for stop-loss and take-profit distances.

    ATR is computed from MediumBuffer OHLC using Wilder's exponential
    smoothing.
    """

    def __init__(self, data_manager: DataManager) -> None:
        """Initialise with a DataManager reference.

        Args:
            data_manager: DataManager holding OHLC bars.
        """
        self._dm = data_manager

    def _compute_atr(self) -> float:
        """Compute ATR(ATR_PERIOD) from the MediumBuffer.

        Returns:
            ATR value in price units, or 0.0 if insufficient data.
        """
        df = self._dm.medium_primary.to_dataframe()
        if len(df) < config.atr_period + 1:
            return 0.0
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)
        closes = df["close"].values.astype(float)
        n = len(highs)
        tr = np.empty(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        alpha = 1.0 / config.atr_period
        atr = float(pd.Series(tr).ewm(alpha=alpha, adjust=False).mean().values[-1])
        if config.is_forex() and atr < config.min_atr_forex:
            raw_atr = atr
            atr = config.min_atr_forex
            logger.warning(
                f"ATR floored | raw={raw_atr:.6f} → floor={atr:.5f} (spread protection)"
            )
        return atr

    def compute_stop_distance(self) -> float:
        """Return stop-loss distance = ATR_STOP_MULTIPLIER × current ATR.

        Returns:
            Stop distance in dollars.
        """
        return self._compute_atr() * config.atr_stop_multiplier

    def compute_take_profit_distance(self, stop_distance: float) -> float:
        """Return take-profit distance = stop_distance × RISK_REWARD_RATIO.

        Args:
            stop_distance: Stop-loss distance in dollars.

        Returns:
            Take-profit distance in dollars.
        """
        return stop_distance * config.risk_reward_ratio


# ---------------------------------------------------------------------------
# SpreadAdjuster
# ---------------------------------------------------------------------------


class SpreadAdjuster:
    """Bid/ask spread cost analysis and trade viability filter.

    All spread and slippage values are read live from the FastBuffer so they
    reflect current market conditions rather than stale estimates.
    """

    def __init__(self, data_manager: DataManager) -> None:
        """Initialise with a DataManager reference.

        Args:
            data_manager: DataManager holding live tick data.
        """
        self._dm = data_manager

    @property
    def current_spread_dollars(self) -> float:
        """Current bid-ask spread in dollars.

        Returns:
            ask - bid, or 0.0 if either is unavailable.
        """
        bid = self._dm.fast_primary.latest_bid()
        ask = self._dm.fast_primary.latest_ask()
        if bid is None or ask is None:
            return 0.0
        return max(0.0, ask - bid)

    @property
    def current_spread_pct(self) -> float:
        """Current bid-ask spread as a fraction of the mid price.

        Returns:
            Spread as decimal fraction, or 0.0 if prices unavailable.
        """
        bid = self._dm.fast_primary.latest_bid()
        ask = self._dm.fast_primary.latest_ask()
        if bid is None or ask is None or bid + ask < 1e-8:
            return 0.0
        mid = (bid + ask) / 2.0
        return (ask - bid) / mid if mid > 0 else 0.0

    def adjust_fill_price(self, side: str) -> Optional[float]:
        """Return the realistic fill price for a given order side.

        Buys fill at ask, sells fill at bid.

        Args:
            side: "LONG" for a buy, "SHORT" for a sell.

        Returns:
            Fill price as float, or ``None`` if prices are unavailable.
        """
        if side == "LONG":
            return self._dm.fast_primary.latest_ask()
        return self._dm.fast_primary.latest_bid()

    def is_trade_worth_it(
        self, expected_profit_dollars: float, position_size: int
    ) -> bool:
        """Return True if expected profit exceeds costs by the required multiple.

        Cost = spread × MIN_PROFIT_SPREAD_MULTIPLE + slippage

        Args:
            expected_profit_dollars: Expected gross profit in dollars.
            position_size:           Number of shares in the proposed trade.

        Returns:
            ``True`` when the trade economics are viable.
        """
        price = self._dm.fast_primary.latest_price() or 0.0
        spread = self.current_spread_dollars
        slippage = position_size * price * config.slippage_factor
        min_profit = spread * config.min_profit_spread_multiple + slippage
        return expected_profit_dollars > min_profit

    def net_pnl(self, gross_pnl: float, position_size: int) -> float:
        """Subtract spread and slippage costs from gross PnL.

        Args:
            gross_pnl:     Gross profit/loss in dollars.
            position_size: Number of shares.

        Returns:
            Net PnL in dollars.
        """
        price = self._dm.fast_primary.latest_price() or 0.0
        spread_cost = self.current_spread_dollars * position_size
        slippage_cost = position_size * price * config.slippage_factor
        return gross_pnl - spread_cost - slippage_cost
