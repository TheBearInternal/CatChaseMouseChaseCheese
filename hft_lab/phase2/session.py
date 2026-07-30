"""Market calendar and session lifecycle management for hft_lab Phase 2.

``MarketCalendar`` provides timezone-aware market-hours logic using the
standard library ``zoneinfo`` module (Python 3.9+).

``SessionManager`` owns the main trading loop.  It wakes on each tick,
drives the full signal → regime → ensemble → execution pipeline, and handles
all session lifecycle transitions:

  closed → warm-up → active trading → pre-close → session close → sleep

Crash safety: any exception inside the per-tick handler is caught, logged at
ERROR, and the loop resumes after a brief backoff sleep rather than crashing.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from config import config
from logger import get_logger
from phase1.behavior import BehaviorProfile, human_pause
from phase2.data import DataManager
from phase2.ensemble import EnsembleDecision, InformationCoefficient, KalmanEnsemble
from phase2.execution import OrderExecutor, TradeLogger
from phase2.regime import RegimeClassifier, RegimeState
from phase2.news_calendar import EventCalendar
from phase2.sentiment import SentimentAnalyzer
from phase2.signals import SignalEngine
from phase2.sizing import ATRSizer, GARCHSizer, SpreadAdjuster

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EST = ZoneInfo("America/New_York")
MARKET_OPEN_TIME: dtime = dtime(9, 30, 0)
MARKET_CLOSE_TIME: dtime = dtime(16, 0, 0)
PRE_CLOSE_MINUTES: int = 15

TICK_INTERVAL_S: float = 1.0       # main loop cadence
ERROR_BACKOFF_S: float = 5.0       # sleep after a per-tick exception

# Daily FX rollover window (America/New_York) — spreads blow out around the
# 17:00 swap, so no new forex entries are opened inside it
ROLLOVER_START: dtime = dtime(16, 55, 0)
ROLLOVER_END: dtime = dtime(17, 15, 0)


def _parse_trading_sessions(raw: str) -> List[tuple]:
    """Parse TRADING_SESSIONS into (start, end) time tuples.

    Accepts comma-separated ``HH:MM-HH:MM`` ranges (America/New_York).
    Malformed entries are skipped with a warning; an empty string yields an
    empty list, which disables the session filter.
    """
    sessions: List[tuple] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            start_s, end_s = part.split("-")
            sh, sm = start_s.strip().split(":")
            eh, em = end_s.strip().split(":")
            sessions.append((dtime(int(sh), int(sm)), dtime(int(eh), int(em))))
        except (ValueError, AttributeError):
            logger.warning(f"Ignoring malformed trading session {part!r}")
    return sessions


# ---------------------------------------------------------------------------
# MarketCalendar
# ---------------------------------------------------------------------------


class MarketCalendar:
    """Timezone-aware NYSE/NASDAQ market hours logic.

    All computations use the America/New_York timezone so DST transitions
    are handled automatically via ``zoneinfo``.
    """

    def is_market_open(self) -> bool:
        """Return True if the current moment falls within active market hours.

        * Crypto: always ``True`` (24/7).
        * Forex: ``True`` from Sunday 17:00 EST through Friday 17:00 EST;
          Saturday is fully closed; Sunday before 17:00 EST is closed.
        * Equity: ``True`` Monday–Friday 09:30–16:00 EST.

        Returns:
            ``True`` when trading is active for the configured market type.
        """
        if config.is_crypto():
            return True
        if config.is_forex():
            now = datetime.now(EST)
            weekday = now.weekday()
            if weekday == 5:          # Saturday — always closed
                return False
            if weekday == 6:          # Sunday — open after 17:00
                return now.hour >= 17
            if weekday == 4:          # Friday — closes at 17:00
                return now.hour < 17
            return True               # Monday–Thursday always open
        now = datetime.now(EST)
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        t = now.time()
        return MARKET_OPEN_TIME <= t < MARKET_CLOSE_TIME

    def time_to_open(self) -> float:
        """Return seconds until the next market open.

        * Crypto: always ``0`` (never closes).
        * Forex: ``0`` if already open, otherwise seconds until next
          Sunday 17:00 EST (the weekly open).
        * Equity: seconds until the next 09:30 EST weekday open.

        Returns:
            Seconds as float.
        """
        if config.is_crypto():
            return 0.0
        if config.is_forex():
            if self.is_market_open():
                return 0.0
            now = datetime.now(EST)
            # Advance day-by-day until we land on a Sunday at 17:00
            candidate = now.replace(hour=17, minute=0, second=0, microsecond=0)
            if now >= candidate:
                candidate = candidate + timedelta(days=1)
            while candidate.weekday() != 6:  # 6 = Sunday
                candidate = candidate + timedelta(days=1)
            return max(0.0, (candidate - now).total_seconds())
        now = datetime.now(EST)
        # Find next 09:30 on a weekday
        candidate = now.replace(
            hour=MARKET_OPEN_TIME.hour,
            minute=MARKET_OPEN_TIME.minute,
            second=0,
            microsecond=0,
        )
        if now >= candidate:
            candidate = candidate.replace(day=candidate.day + 1)
        # Skip weekends
        while candidate.weekday() >= 5:
            candidate = candidate.replace(day=candidate.day + 1)
        delta = (candidate - now).total_seconds()
        return max(0.0, delta)

    def time_to_close(self) -> float:
        """Return seconds until today's market close.

        Returns:
            Seconds remaining, or 0 if market is closed.
        """
        now = datetime.now(EST)
        close = now.replace(
            hour=MARKET_CLOSE_TIME.hour,
            minute=MARKET_CLOSE_TIME.minute,
            second=0,
            microsecond=0,
        )
        delta = (close - now).total_seconds()
        return max(0.0, delta)

    def is_approaching_close(self, minutes: int = PRE_CLOSE_MINUTES) -> bool:
        """Return True if within ``minutes`` of the market close.

        Args:
            minutes: Lookahead window in minutes.

        Returns:
            ``True`` when time_to_close() < minutes × 60.
        """
        return 0 < self.time_to_close() < minutes * 60

    def is_rollover_blackout(self) -> bool:
        """Return True during the daily FX rollover window (16:55–17:15 EST).

        Spreads widen sharply around the 17:00 New York swap.  Callers use this
        to suppress *new* entries; positions already open are unaffected.

        Returns:
            ``True`` when the current New York time is inside the window.
        """
        now_t = datetime.now(EST).time()
        return ROLLOVER_START <= now_t < ROLLOVER_END

    def current_minute(self) -> int:
        """Return minutes elapsed since today's market open (0-based).

        Returns:
            Integer in [0, 389], or -1 outside market hours.
        """
        now = datetime.now(EST)
        open_today = now.replace(
            hour=MARKET_OPEN_TIME.hour,
            minute=MARKET_OPEN_TIME.minute,
            second=0,
            microsecond=0,
        )
        delta = (now - open_today).total_seconds()
        minute = int(delta // 60)
        return minute if 0 <= minute < 390 else -1


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------


class SessionManager:
    """Orchestrates the full autonomous trading lifecycle for one session.

    Responsibilities
    ----------------
    * Session open: warm up data, fit GARCH, restore Kalman weights.
    * Active trading: per-tick signal→regime→ensemble→execution pipeline.
    * Pre-close: stop opening new positions, begin closing existing ones.
    * Session close: flatten all positions, save SlowBuffer snapshot.
    * Between sessions: sleep until next market open.
    """

    def __init__(
        self,
        data_manager: DataManager,
        signal_engine: SignalEngine,
        regime_classifier: RegimeClassifier,
        kalman_ensemble: KalmanEnsemble,
        ic_tracker: InformationCoefficient,
        ensemble_decision: EnsembleDecision,
        sentiment_analyzer: SentimentAnalyzer,
        garch_sizer: GARCHSizer,
        atr_sizer: ATRSizer,
        spread_adjuster: SpreadAdjuster,
        order_executor: OrderExecutor,
        trade_logger: TradeLogger,
        market_calendar: MarketCalendar,
        profile: BehaviorProfile,
        historical_client: Any,
    ) -> None:
        self._dm = data_manager
        self._signals = signal_engine
        self._regime = regime_classifier
        self._kalman = kalman_ensemble
        self._ic = ic_tracker
        self._ensemble = ensemble_decision
        self._sentiment = sentiment_analyzer
        self._garch = garch_sizer
        self._atr = atr_sizer
        self._spread = spread_adjuster
        self._executor = order_executor
        self._trade_logger = trade_logger
        self._calendar = market_calendar
        self._profile = profile
        self._historical_client = historical_client
        self._stop_event: asyncio.Event = asyncio.Event()
        self._in_pre_close: bool = False
        self._last_state_save_ts: float = 0.0
        self._event_calendar: EventCalendar = EventCalendar(silent=False)
        self._last_calendar_log_ts: float = 0.0
        self._last_summary_ts: float = 0.0
        self._last_suppressed_minute: Optional[int] = None
        self._in_rollover_blackout: bool = False
        self._last_ic_bar_ts: Optional[Any] = None
        self._regime_gate_logged: Optional[str] = None
        self._trading_sessions: List[tuple] = _parse_trading_sessions(
            config.trading_sessions
        )
        self._session_window_open: Optional[bool] = None
        self._post_close_log_key: Optional[float] = None
        # Broker-side closes discovered by the position sync feed the same
        # IC/Kalman learning path the equity flow uses
        self._executor.trade_close_listener = self._handle_broker_close

    def stop(self) -> None:
        """Signal the session loop to exit after the current iteration."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Main async loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main session management loop — runs until ``stop()`` is called.

        Handles market open/close transitions and delegates each tick to
        ``_trading_tick()``.  Exceptions inside the tick handler are caught
        and logged without crashing the loop.
        """
        logger.info("SessionManager started")

        while not self._stop_event.is_set():
            market_open = self._calendar.is_market_open() or config.dev_mode

            if not market_open:
                wait_s = self._calendar.time_to_open()
                logger.info(f"Market closed — sleeping {wait_s / 3600:.1f}h until open")
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=min(wait_s, 60.0)
                    )
                    break
                except asyncio.TimeoutError:
                    continue

            # --- Session open ---
            await self._session_start()

            # --- Active trading ---
            while (self._calendar.is_market_open() or config.dev_mode) \
                    and not self._stop_event.is_set():

                # Approaching-close wind-down is equity-only.
                # Crypto (24/7) and forex (weekend-aware) have no intraday close.
                if not config.is_crypto() and not config.is_forex() \
                        and not self._in_pre_close \
                        and self._calendar.is_approaching_close() \
                        and not config.dev_mode:
                    await self._pre_close()

                try:
                    await self._trading_tick()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error(
                        f"Trading tick error: {type(exc).__name__}: {exc}",
                        exc_info=True,
                    )
                    await asyncio.sleep(ERROR_BACKOFF_S)

                # Rolling 24-hour state persistence for crypto and forex
                # (neither has a scheduled session close).
                if (config.is_crypto() or config.is_forex()) \
                        and time.time() - self._last_state_save_ts >= 86400:
                    await self._save_state()
                    self._last_state_save_ts = time.time()

                await asyncio.sleep(TICK_INTERVAL_S)

            # --- Session close ---
            await self._session_close()

        logger.info("SessionManager stopped")

    # ------------------------------------------------------------------
    # Session lifecycle handlers
    # ------------------------------------------------------------------

    async def _session_start(self) -> None:
        """Initialise state at the beginning of a trading session."""
        if config.is_crypto():
            logger.info(
                "=== Session starting (24/7 CRYPTO MODE — no market-hours logic, "
                "no session close, state saved every 24h) ==="
            )
        elif config.is_forex():
            logger.info(
                "=== Session starting (FOREX MODE — 24/5 weekend-aware, "
                "no session-close position flattening, state saved every 24h) ==="
            )
        else:
            logger.info("=== Session starting ===")
        self._in_pre_close = False
        self._trade_logger.reset_session_pnl()

        # Restore Kalman weights from previous session if available
        self._dm.slow_buffer.load(config.session_state_path)
        weights_state = self._dm.slow_buffer.data.get("signal_weights")
        if weights_state:
            self._kalman.load_state(weights_state)

        # Reconcile trades the broker closed while the engine was offline
        if config.is_forex():
            await self._executor.reconcile_offline_closes()

        # Select the correct benchmark symbol for the active market type
        if config.is_crypto():
            benchmark = config.crypto_benchmark
        elif config.is_forex():
            benchmark = config.forex_benchmark
        else:
            benchmark = config.benchmark_symbol

        if config.is_forex() and config.primary_symbol == benchmark:
            logger.warning(
                f"primary_symbol == forex_benchmark ({config.primary_symbol}) — "
                "relative_strength signal will be zero. "
                "Set FOREX_BENCHMARK to a different pair."
            )

        # Warm up data buffers from historical API
        warmed = await self._dm.warm_up(
            self._historical_client,
            config.primary_symbol,
            benchmark,
            config.historical_bars,
        )
        if not warmed:
            logger.warning(
                "Warm-up did not reach is_ready threshold — signals will be "
                "suppressed until sufficient data accumulates"
            )

        # Timeframe visibility: effective bar size + median ATR of history
        hist_df = self._dm.to_dataframe()
        if len(hist_df) > config.atr_period + 1:
            highs = hist_df["high"].values.astype(float)
            lows = hist_df["low"].values.astype(float)
            closes = hist_df["close"].values.astype(float)
            tr = np.maximum(
                highs[1:] - lows[1:],
                np.maximum(
                    np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:] - closes[:-1]),
                ),
            )
            atr_series = pd.Series(tr).ewm(
                alpha=1.0 / config.atr_period, adjust=False
            ).mean()
            logger.info(
                f"Timeframe | {config.bar_timeframe} | "
                f"median ATR over {len(hist_df)} bars = "
                f"{float(atr_series.median()):.5f}"
            )

        # Fit GARCH on available historical returns
        await self._fit_garch()

        # Warm-start IC/Kalman from history (no-op if weights already trained)
        await self._warm_start_ic()

        # Kalman prediction step at session open
        self._kalman.predict()

        # Launch the forex position monitor alongside the trading loop
        if config.is_forex():
            asyncio.create_task(
                self._monitor_positions(), name="forex-position-monitor"
            )

        logger.info("Session ready — trading loop active")

    async def _pre_close(self) -> None:
        """Handle pre-close wind-down: stop new entries, begin closing positions."""
        logger.info(
            f"Approaching market close ({PRE_CLOSE_MINUTES} min) — "
            "stopping new entries, closing positions"
        )
        self._in_pre_close = True
        await self._executor.close_all_positions("session_close")

    async def _session_close(self) -> None:
        """Flatten remaining positions (equity only) and persist session state."""
        logger.info("=== Session closing ===")
        # Crypto and forex have no scheduled session close; the engine shutdown
        # handler manages position flattening.  Equity flattens on calendar close.
        if not config.is_crypto() and not config.is_forex() and not self._in_pre_close:
            await self._executor.close_all_positions("session_close")

        session_pnl = self._trade_logger.get_session_pnl()
        logger.info(f"Session PnL — gross (approximate): ${session_pnl:.2f}")
        await self._save_state()

    async def _save_state(self) -> None:
        """Persist Kalman weights and session metrics to SlowBuffer."""
        self._dm.slow_buffer.data.update({
            "session_date": datetime.now(timezone.utc).isoformat(),
            "session_pnl": self._trade_logger.get_session_pnl(),
            "signal_weights": self._kalman.save_state(),
            "trade_count": self._dm.slow_buffer.data.get("trade_count", 0),
        })
        self._dm.slow_buffer.save(config.session_state_path)
        logger.info("Session state saved")

    # ------------------------------------------------------------------
    # Per-tick trading pipeline
    # ------------------------------------------------------------------

    async def _trading_tick(self) -> None:
        """Execute one iteration of the signal→decision→execution pipeline."""
        if not self._dm.is_ready:
            return

        if self._in_pre_close:
            return

        # 1. Event calendar risk check (logs transitions; 15-min summary log)
        in_event_window, event = self._event_calendar.is_high_impact_window()
        risk_mult = self._event_calendar.get_risk_multiplier()
        threshold_mult = self._event_calendar.get_threshold_multiplier()
        now_ts = time.time()
        if now_ts - self._last_calendar_log_ts >= 900:
            upcoming = self._event_calendar.get_upcoming_events(hours_ahead=24)
            if upcoming:
                logger.info(f"CALENDAR | {self._event_calendar.next_event_summary()}")
            self._last_calendar_log_ts = now_ts
        if in_event_window and event is not None:
            logger.debug(
                f"CALENDAR | Window active: {event.name} ({event.impact}) "
                f"risk_mult={risk_mult:.2f} threshold_mult={threshold_mult:.2f}"
            )

        # 1b. FX rollover blackout state transition (log once each way)
        if config.is_forex():
            in_blackout = self._calendar.is_rollover_blackout()
            if in_blackout != self._in_rollover_blackout:
                if in_blackout:
                    logger.info(
                        "Rollover blackout | 16:55–17:15 EST — new entries blocked"
                    )
                else:
                    logger.info(
                        "Rollover blackout ended — new entries allowed"
                    )
                self._in_rollover_blackout = in_blackout

            # 1c. Trading-session window transition (log once each way)
            in_window = self._in_trading_session()
            if in_window != self._session_window_open:
                if in_window:
                    logger.info("Session | trading window open")
                else:
                    logger.info(
                        f"Session | outside trading window "
                        f"({config.trading_sessions}) — entries blocked"
                    )
                self._session_window_open = in_window

        # 2. Regime classification
        regime = self._regime.classify(self._dm)

        # 3. Signal scores
        signal_scores = self._signals.compute_all(regime)

        # 3b. Bar-close IC scoring — learning decoupled from trade outcomes.
        # On each new bar, snapshot the scores/regime computed at this bar;
        # snapshots ic_forward_bars old are scored on their realized forward
        # return and update IC + the Kalman vector of the regime at time t.
        bars_df = self._dm.medium_primary.to_dataframe()
        if len(bars_df) > 0:
            last_bar_ts = bars_df["timestamp"].iloc[-1]
            if last_bar_ts != self._last_ic_bar_ts:
                self._last_ic_bar_ts = last_bar_ts
                bar_close = float(bars_df["close"].iloc[-1])
                for s0, r0, fwd in self._dm.record_ic_snapshot(
                    bar_close, signal_scores, regime
                ):
                    self._score_ic_observation(s0, r0, fwd)

        # 4. News sentiment
        sentiment = self._sentiment.current_sentiment

        # 5. Kalman weights for this regime's vector
        weights = self._kalman.get_weights(regime)

        # 6. Ensemble decision
        decision = self._ensemble.decide(signal_scores, weights, regime, sentiment)

        # 7. Act on non-HOLD decision
        if decision.action != "HOLD":
            await self._handle_signal(
                decision, signal_scores, weights, regime,
                risk_mult=risk_mult, threshold_mult=threshold_mult,
            )

        # 8. Check existing positions for fills and exits
        closed_trades = await self._executor.check_positions()
        for trade in closed_trades:
            await self._on_trade_closed(trade, signal_scores, regime)

        # 9. Periodic GARCH re-fit
        if self._garch.should_refit:
            await self._fit_garch()

        # 10. 15-second heartbeat — replaces per-tick noise with a readable summary
        if now_ts - self._last_summary_ts >= 15:
            regime_thresholds = {
                "RANDOM_WALK": config.ensemble_threshold_random,
                "AMBIGUOUS": config.ensemble_threshold_ambiguous,
            }
            threshold = regime_thresholds.get(regime.value, config.ensemble_threshold)
            logger.info(
                f"Status | regime={regime.value} "
                f"conf={decision.confidence:.4f} "
                f"sentiment={sentiment:+.3f} "
                f"positions={len(self._executor.open_positions)} "
                f"threshold={threshold:.2f}"
            )
            self._last_summary_ts = now_ts

    async def _handle_signal(
        self,
        decision: Any,
        signal_scores: Dict[str, float],
        weights: Dict[str, float],
        regime: RegimeState,
        risk_mult: float = 1.0,
        threshold_mult: float = 1.0,
    ) -> None:
        """Evaluate and potentially execute a LONG or SHORT signal.

        Checks position limits, applies calendar threshold gate, sizes the
        trade (with event risk scaling), validates spread economics, applies
        behavioral filters, then submits the bracket order.

        Args:
            decision:       EnsembleDecision Decision namedtuple.
            signal_scores:  Signal scores at this moment.
            weights:        Kalman weights at this moment.
            regime:         Current RegimeState.
            risk_mult:      Position-size multiplier from EventCalendar (≤1.0).
            threshold_mult: Confidence-threshold multiplier from EventCalendar (≥1.0).
        """
        # Calendar-adjusted confidence threshold gate
        regime_thresholds = {
            "RANDOM_WALK": config.ensemble_threshold_random,
            "AMBIGUOUS": config.ensemble_threshold_ambiguous,
        }
        base_thresh = regime_thresholds.get(regime.value, config.ensemble_threshold)
        adjusted_thresh = base_thresh * threshold_mult
        if decision.confidence < adjusted_thresh:
            logger.debug(
                f"Signal gated by calendar: confidence={decision.confidence:.3f} "
                f"< threshold={adjusted_thresh:.3f} "
                f"(base={base_thresh:.3f} × mult={threshold_mult:.2f})"
            )
            return

        # Enforce position limit
        if len(self._executor.open_positions) >= config.max_open_positions:
            logger.debug("Position limit reached — skipping signal")
            return

        # Forex FIFO guard: one open position per symbol; signal-reversal exit
        if config.is_forex():
            symbol = config.primary_symbol
            open_pos = next(
                (p for p in self._executor._open_positions.values()
                 if p.symbol == symbol),
                None,
            )
            if open_pos is not None:
                opposite = {"LONG": "SHORT", "SHORT": "LONG"}
                if (decision.action == opposite.get(open_pos.side)
                        and decision.confidence >= config.ensemble_threshold):
                    logger.info(
                        f"Signal reversal exit | {symbol} {open_pos.side} closed "
                        f"— signal flipped to {decision.action} "
                        f"conf={decision.confidence:.4f}"
                    )
                    await self._executor.close_position_by_symbol(
                        symbol, reason="signal_reversal"
                    )
                else:
                    logger.debug(f"FIFO | Position already open for {symbol} — skipping")
                return

            # Per-symbol submission cooldown
            last_ts = self._executor._last_order_time.get(symbol, 0.0)
            if time.time() - last_ts < 30.0:
                logger.debug(
                    f"FIFO | Submission cooldown active for {symbol} "
                    f"({30.0 - (time.time() - last_ts):.0f}s remaining) — skipping"
                )
                return

        # Rollover blackout: block new entries only — the FIFO block above
        # already handled (and allowed) reversal exits on open positions
        if config.is_forex() and self._calendar.is_rollover_blackout():
            logger.debug("Rollover blackout active — skipping new entry")
            return

        # Regime gate: only enter in configured regimes (empty set disables)
        if config.tradeable_regimes and regime.value not in config.tradeable_regimes:
            if self._regime_gate_logged != regime.value:
                logger.info(f"Regime gate | {regime.value} not tradeable — holding")
                self._regime_gate_logged = regime.value
            return
        self._regime_gate_logged = None

        # Trading-session filter: forex entries only inside configured windows
        if config.is_forex() and not self._in_trading_session():
            return

        # Post-close cooldown: no re-entry into just-closed conditions
        # (signal_reversal closes are exempt and never anchor the cooldown)
        last_close = self._executor.last_close_time(config.primary_symbol)
        if last_close > 0:
            elapsed = time.time() - last_close
            if elapsed < config.post_close_cooldown_s:
                if self._post_close_log_key != last_close:
                    remaining = int(config.post_close_cooldown_s - elapsed)
                    logger.info(
                        f"Post-close cooldown | {config.primary_symbol} | "
                        f"{remaining}s remaining"
                    )
                    self._post_close_log_key = last_close
                return

        # Behavioral activity filter — log once per suppressed minute
        current_min = self._calendar.current_minute()
        if current_min >= 0 and not self._profile.should_act(current_min):
            if current_min != self._last_suppressed_minute:
                logger.debug(
                    f"BehaviorProfile suppressed action at minute {current_min}"
                )
                self._last_suppressed_minute = current_min
            return
        self._last_suppressed_minute = None

        # Fetch account equity for sizing — skip Alpaca for forex
        price = self._dm.fast_primary.latest_price()
        if price is None:
            return

        if config.is_forex():
            equity = config.account_limit
        else:
            try:
                loop = asyncio.get_running_loop()
                account = await loop.run_in_executor(
                    None, self._executor._client.get_account
                )
                equity = float(getattr(account, "equity", config.account_limit))
            except Exception:
                equity = config.account_limit

        # Compute base position size then apply event risk multiplier
        quantity = self._garch.compute_position_size(
            account_equity=equity,
            max_risk_pct=config.max_risk_per_trade_pct,
            confidence=decision.confidence,
            current_price=price,
            profile_variance=self._profile.order_size_variance,
        )
        quantity = max(config.min_position_size, int(quantity * risk_mult))
        if config.is_forex():
            quantity = max(1, round(quantity))

        # Compute stop and take-profit distances; derive raw ATR for order log
        stop_dist = self._atr.compute_stop_distance()
        tp_dist = self._atr.compute_take_profit_distance(stop_dist)
        atr_value = stop_dist / config.atr_stop_multiplier if config.atr_stop_multiplier > 0 else stop_dist

        # Spread viability check
        expected_profit = tp_dist * quantity
        if not self._spread.is_trade_worth_it(expected_profit, quantity):
            logger.debug(
                f"Trade not worth it: expected_profit={expected_profit:.2f} "
                f"spread={self._spread.current_spread_dollars:.4f}"
            )
            return

        # Human-pause timing randomization
        await human_pause()

        # Submit bracket order
        await self._executor.submit_bracket_order(
            side=decision.action,
            quantity=quantity,
            stop_distance=stop_dist,
            take_profit_distance=tp_dist,
            signal_scores=signal_scores,
            weights=weights,
            regime_name=regime.value,
            atr_value=atr_value,
            confidence=decision.confidence,
        )

    async def _on_trade_closed(
        self,
        trade: Any,
        current_signal_scores: Dict[str, float],
        regime: RegimeState,
    ) -> None:
        """Update IC and run Kalman correction after a position closes.

        The Kalman update targets only *regime*'s weight vector so each
        regime accumulates its own signal performance history.

        Args:
            trade:                 Closed TradeRecord.
            current_signal_scores: Signal scores at the time of closure check.
            regime:                RegimeState active when the position closed.
        """
        if trade.entry_price < 1e-6:
            return

        actual_return = (trade.exit_price - trade.entry_price) / trade.entry_price
        if trade.side == "SHORT":
            actual_return = -actual_return

        from phase2.signals import SIGNAL_NAMES
        for name in SIGNAL_NAMES:
            pred = trade.signal_scores_at_entry.get(name, 0.0)
            self._ic.update(name, pred, actual_return)

        # Kalman correction step — update only this regime's weight vector
        ic_vector = np.array([self._ic.get_ic(n) for n in SIGNAL_NAMES])
        self._kalman.update(ic_vector, regime)

        logger.info(
            f"Trade closed: {trade.side} {trade.symbol} "
            f"gross={trade.gross_pnl:+.2f} net={trade.net_pnl:+.2f} "
            f"reason={trade.exit_reason} | "
            f"session_pnl={self._trade_logger.get_session_pnl():+.2f}"
        )

    # ------------------------------------------------------------------
    # Forex position monitor
    # ------------------------------------------------------------------

    async def _monitor_positions(self) -> None:
        """Every 30 seconds, reconcile the local tracker against OANDA's live state.

        Runs unconditionally — even with an empty tracker — so positions that
        exist on OANDA but are missing locally are rediscovered.  The executor
        applies the sync on the event loop and defers it while an order
        submission is in flight.
        """
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=30.0)
                break
            except asyncio.TimeoutError:
                pass

            before_count = len(self._executor._open_positions)
            closed_records = await self._executor.sync_positions_from_oanda()
            if closed_records is None:
                continue
            after_count = len(self._executor._open_positions)

            if after_count != before_count:
                if after_count == 0:
                    logger.info(
                        "Position monitor | OANDA reports 0 open — clearing tracker "
                        "(TP/SL hit or externally closed)"
                    )
                else:
                    logger.info(
                        f"Position monitor | reconciled {before_count} → "
                        f"{after_count} tracked position(s)"
                    )

    def _in_trading_session(self) -> bool:
        """Return True when the current New York time is inside a trading window.

        An empty TRADING_SESSIONS config disables the filter (always True).
        Ranges where start > end are treated as crossing midnight.
        """
        if not self._trading_sessions:
            return True
        now_t = datetime.now(EST).time()
        for start, end in self._trading_sessions:
            if start <= end:
                if start <= now_t < end:
                    return True
            else:  # crosses midnight, e.g. 22:00-02:00
                if now_t >= start or now_t < end:
                    return True
        return False

    def _score_ic_observation(
        self,
        scores: Dict[str, float],
        regime: RegimeState,
        forward_return: float,
    ) -> None:
        """Update IC and the regime-conditional Kalman filter from one bar event.

        This is the bar-level learning path — distinct from trade-level
        TradeRecords, which remain the cost-aware ground truth.

        Args:
            scores:         Signal scores as computed at time t (never recomputed).
            regime:         Regime classified at time t (not the current regime).
            forward_return: Realized return over the following ic_forward_bars bars.
        """
        from phase2.signals import SIGNAL_NAMES

        for name in SIGNAL_NAMES:
            self._ic.update(name, scores.get(name, 0.0), forward_return)
        ic_vector = np.array([self._ic.get_ic(n) for n in SIGNAL_NAMES])
        self._kalman.update(ic_vector, regime)
        logger.debug(
            f"IC bar-score | regime={regime.value} fwd_ret={forward_return:+.6f}"
        )

    def _log_kalman_weights(self, context: str) -> None:
        """Log all four regime weight vectors at INFO for visibility.

        Signs are always shown explicitly — a negative weight means the filter
        learned to fade that signal, which must be readable at a glance.
        """
        for r in RegimeState:
            weights = self._kalman.get_weights(r)
            formatted = " ".join(f"{k}={v:+.4f}" for k, v in weights.items())
            net = sum(weights.values())
            logger.info(
                f"Kalman weights [{context}] | {r.value}: {formatted} "
                f"| net={net:+.4f}"
            )

    async def _warm_start_ic(self) -> None:
        """Warm-start IC/Kalman from historical bars with strict no-lookahead.

        Replays the warm-up history bar by bar through a throwaway
        ``DataManager`` + ``SignalEngine`` + ``RegimeClassifier`` so every
        signal and the regime classification see only bars up to the one
        being scored, then applies the same forward-return scoring used
        live.  Runs only when the Kalman weights are still at their uniform
        priors, so restored or already-trained state is never double-counted.

        Bars are replayed with naive timestamps so the MediumBuffer's
        90-minute wall-clock eviction cannot silently drop them mid-replay.
        """
        if not config.ic_warm_start_enabled:
            return

        from phase2.ensemble import INITIAL_WEIGHT

        for r in RegimeState:
            if any(
                abs(v - INITIAL_WEIGHT) > 1e-9
                for v in self._kalman.get_weights(r).values()
            ):
                logger.info(
                    "IC warm start skipped — weights already trained "
                    "(restored from state or scored this session)"
                )
                return

        df = self._dm.to_dataframe()
        if df.empty or len(df) < 40:
            logger.info("IC warm start skipped — insufficient history")
            return

        import logging as _logging
        from collections import deque as _deque

        replay_dm = DataManager.__new__(DataManager)
        # Minimal init: only the attributes signal/regime computation touches
        replay_dm.fast_primary = type(self._dm.fast_primary)()
        replay_dm.fast_benchmark = type(self._dm.fast_benchmark)()
        replay_dm.medium_primary = type(self._dm.medium_primary)()
        replay_dm.slow_buffer = type(self._dm.slow_buffer)()
        replay_dm._session_open_ts = None
        replay_dm._hist_bars = _deque(maxlen=200)
        replay_dm._tick_history = _deque(maxlen=100)
        replay_dm._last_tick_price = None
        replay_dm._is_crypto = config.is_crypto()
        replay_dm._is_forex = config.is_forex()
        replay_dm._vwap_proxy_logged = True
        replay_dm.ic_snapshots = _deque()

        replay_engine = SignalEngine(replay_dm)
        replay_regime = RegimeClassifier()

        # Silence per-bar DEBUG/INFO chatter from the replay components
        muted = ["phase2.signals", "phase2.regime", "phase2.data"]
        saved_levels = {n: _logging.getLogger(n).level for n in muted}
        for n in muted:
            _logging.getLogger(n).setLevel(_logging.WARNING)

        scored = 0
        try:
            pending: _deque = _deque()
            fwd = max(1, config.ic_forward_bars)
            for row in df.to_dict("records"):
                ts = row.get("timestamp")
                naive_ts = (
                    ts.replace(tzinfo=None) if getattr(ts, "tzinfo", None) else ts
                )
                bar = {**row, "timestamp": naive_ts}
                replay_dm.medium_primary.append(bar)
                replay_dm._hist_bars.append(bar)
                replay_dm.fast_primary.append({
                    "timestamp": ts,
                    "symbol": config.primary_symbol,
                    "price": float(row["close"]),
                    "bid": None,
                    "ask": None,
                    "bid_size": 0,
                    "ask_size": 0,
                    "volume": float(row.get("volume", 1.0)),
                })

                bar_regime = replay_regime.classify(replay_dm)
                bar_scores = replay_engine.compute_all(bar_regime)
                price = float(row["close"])

                pending.append((price, bar_scores, bar_regime))
                if len(pending) > fwd:
                    p0, s0, r0 = pending.popleft()
                    if p0 > 1e-9:
                        self._score_ic_observation(s0, r0, (price - p0) / p0)
                        scored += 1
        except Exception as exc:
            logger.warning(f"IC warm start aborted after {scored} bars: {exc!r}")
        finally:
            for n, lvl in saved_levels.items():
                _logging.getLogger(n).setLevel(lvl)

        if scored > 0:
            logger.info(
                f"IC warm start | scored {scored} bars | "
                "weights initialized from history"
            )
            self._log_kalman_weights("warm-start")

    async def _handle_broker_close(self, record: Any) -> None:
        """Feed a broker-side close (TP/SL hit on OANDA) into the learning loop.

        Called by the executor's position sync for every TradeRecord it builds
        from OANDA's closing ORDER_FILL transaction.  Uses the regime recorded
        at entry so the IC observations update the Kalman weight vector that
        actually produced the trade.
        """
        try:
            entry_regime = RegimeState(record.regime_at_entry)
        except ValueError:
            logger.debug(
                f"Broker close for {record.symbol}: unknown entry regime "
                f"{record.regime_at_entry!r} — skipping learning update"
            )
            return
        await self._on_trade_closed(record, {}, entry_regime)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _fit_garch(self) -> None:
        """Fit GARCH parameters on current MediumBuffer returns."""
        df = self._dm.medium_primary.to_dataframe()
        if len(df) < 30:
            return
        closes = df["close"].astype(float)
        returns = np.log(closes / closes.shift(1)).dropna()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._garch.fit, returns)
