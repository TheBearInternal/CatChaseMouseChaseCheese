"""Main orchestrator for the Phase 2 trading engine.

Wires all Phase 1 infrastructure (WebSocket streaming, behavioral
randomization) to all Phase 2 components (signals, regime, ensemble,
sentiment, sizing, execution, session management) and launches the
fully autonomous trading loop.

Entry point::

    python -m phase2.engine

Startup sequence
----------------
1. Load config and logger.
2. Log DEV_MODE and live-account warnings if applicable.
3. Apply BehaviorProfile market_open_delay (skipped in DEV_MODE).
4. Initialise all components.
5. Subclass Phase 1 ConnectionManager to feed DataManager on every tick.
6. Start SentimentAnalyzer as background async task.
7. Start WebSocket connection as background async task.
8. Start SessionManager as main async task.
9. On KeyboardInterrupt / SIGTERM: close all positions, save state, shut down.
"""
from __future__ import annotations

import asyncio
import signal
from datetime import datetime, timezone
from typing import Any, Optional

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

from config import config
from logger import get_logger
from phase1.behavior import BehaviorProfile, market_open_delay
from phase1.connection import ConnectionManager
from phase2.data import DataManager
from phase2.ensemble import EnsembleDecision, InformationCoefficient, KalmanEnsemble
from phase2.execution import OrderExecutor, TradeLogger
from phase2.regime import RegimeClassifier
from phase2.sentiment import SentimentAnalyzer
from phase2.session import MarketCalendar, SessionManager
from phase2.signals import SignalEngine
from phase2.sizing import ATRSizer, GARCHSizer, SpreadAdjuster

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# WebSocket bridge: routes Phase 1 stream events into DataManager
# ---------------------------------------------------------------------------


class TradingConnectionManager(ConnectionManager):
    """ConnectionManager subclass that feeds DataManager on every message.

    Overrides ``_handle_trade`` and ``_handle_quote`` so all incoming tick
    data is routed into the shared DataManager rather than only logged.
    The Phase 1 health check, reconnection, and signal handling are
    fully inherited.
    """

    def __init__(self, profile: BehaviorProfile, data_manager: DataManager) -> None:
        """Initialise with the shared DataManager wired in.

        Args:
            profile:      BehaviorProfile for startup delay and jitter.
            data_manager: DataManager to receive all incoming tick data.
        """
        super().__init__(profile=profile)
        self._data_manager = data_manager

    async def _handle_trade(self, data: object) -> None:
        """Forward trade tick to DataManager and log at DEBUG level.

        Args:
            data: Trade object from alpaca-py.
        """
        self._last_message_ts = __import__("time").time()
        symbol: str = getattr(data, "symbol", "UNKNOWN")
        price = getattr(data, "price", None)
        volume = getattr(data, "size", 0)
        timestamp = getattr(data, "timestamp", datetime.now(timezone.utc))

        tick = {
            "timestamp": timestamp,
            "symbol": symbol,
            "price": float(price) if price is not None else None,
            "bid": None,
            "ask": None,
            "bid_size": 0,
            "ask_size": 0,
            "volume": float(volume) if volume else 0.0,
        }
        await self._data_manager.ingest_tick(symbol, tick)
        logger.debug(f"TRADE | symbol={symbol} price={price} volume={volume}")

    async def _handle_quote(self, data: object) -> None:
        """Forward quote tick to DataManager and log at DEBUG level.

        Args:
            data: Quote object from alpaca-py.
        """
        self._last_message_ts = __import__("time").time()
        symbol: str = getattr(data, "symbol", "UNKNOWN")
        bid = getattr(data, "bid_price", None)
        ask = getattr(data, "ask_price", None)
        bid_size = getattr(data, "bid_size", 0)
        ask_size = getattr(data, "ask_size", 0)
        timestamp = getattr(data, "timestamp", datetime.now(timezone.utc))

        mid: Optional[float] = None
        if bid is not None and ask is not None:
            mid = (float(bid) + float(ask)) / 2.0

        tick = {
            "timestamp": timestamp,
            "symbol": symbol,
            "price": mid,
            "bid": float(bid) if bid is not None else None,
            "ask": float(ask) if ask is not None else None,
            "bid_size": int(bid_size) if bid_size else 0,
            "ask_size": int(ask_size) if ask_size else 0,
            "volume": 0.0,
        }
        await self._data_manager.ingest_tick(symbol, tick)
        logger.debug(f"QUOTE | symbol={symbol} bid={bid} ask={ask}")


# ---------------------------------------------------------------------------
# Engine entry point
# ---------------------------------------------------------------------------


async def _main() -> None:
    """Async entry point: initialise, wire, and run the trading engine."""

    # 1. Warnings ----------------------------------------------------------
    if config.dev_mode:
        logger.warning(
            "=== DEV MODE ACTIVE === "
            "Startup delay bypassed, 30-bar warm-up, market hours ignored."
        )
    if config.is_live():
        logger.warning(
            "=== LIVE ACCOUNT DETECTED === "
            "This engine is connected to a real funded Alpaca account. "
            "All risk limits are active. Proceed with extreme caution."
        )

    # 2. Phase 1 behavioral profile ----------------------------------------
    profile = BehaviorProfile(seed=config.behavior_seed)
    if not config.dev_mode:
        await market_open_delay(profile)

    # 3. Alpaca clients ----------------------------------------------------
    is_paper = not config.is_live()
    trading_client = TradingClient(
        api_key=config.alpaca_api_key,
        secret_key=config.alpaca_secret_key,
        paper=is_paper,
    )
    historical_client = StockHistoricalDataClient(
        api_key=config.alpaca_api_key,
        secret_key=config.alpaca_secret_key,
    )

    # 4. Phase 2 components ------------------------------------------------
    data_manager = DataManager()
    signal_engine = SignalEngine(data_manager)
    regime_classifier = RegimeClassifier()
    ic_tracker = InformationCoefficient()
    kalman_ensemble = KalmanEnsemble()
    ensemble_decision = EnsembleDecision()
    sentiment_analyzer = SentimentAnalyzer()
    garch_sizer = GARCHSizer(data_manager)
    atr_sizer = ATRSizer(data_manager)
    spread_adjuster = SpreadAdjuster(data_manager)
    trade_logger = TradeLogger()
    order_executor = OrderExecutor(
        trading_client=trading_client,
        data_manager=data_manager,
        spread_adjuster=spread_adjuster,
        trade_logger=trade_logger,
        profile=profile,
    )
    market_calendar = MarketCalendar()
    session_manager = SessionManager(
        data_manager=data_manager,
        signal_engine=signal_engine,
        regime_classifier=regime_classifier,
        kalman_ensemble=kalman_ensemble,
        ic_tracker=ic_tracker,
        ensemble_decision=ensemble_decision,
        sentiment_analyzer=sentiment_analyzer,
        garch_sizer=garch_sizer,
        atr_sizer=atr_sizer,
        spread_adjuster=spread_adjuster,
        order_executor=order_executor,
        trade_logger=trade_logger,
        market_calendar=market_calendar,
        profile=profile,
        historical_client=historical_client,
    )

    # 5. Phase 1 WebSocket connection wired to DataManager -----------------
    connection_manager = TradingConnectionManager(
        profile=profile,
        data_manager=data_manager,
    )

    # 6. Graceful shutdown event -------------------------------------------
    shutdown_event = asyncio.Event()

    def _request_shutdown(sig_name: str) -> None:
        logger.info(f"Engine received {sig_name} — initiating shutdown")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, lambda: _request_shutdown("SIGINT"))
        loop.add_signal_handler(signal.SIGTERM, lambda: _request_shutdown("SIGTERM"))
    except NotImplementedError:
        signal.signal(signal.SIGINT, lambda s, f: shutdown_event.set())

    # 7. Launch tasks ------------------------------------------------------
    sentiment_task = asyncio.create_task(
        sentiment_analyzer.run_poll_loop(), name="sentiment-poller"
    )
    ws_task = asyncio.create_task(
        connection_manager.run(), name="websocket-stream"
    )
    session_task = asyncio.create_task(
        session_manager.run(), name="session-manager"
    )
    shutdown_task = asyncio.create_task(
        shutdown_event.wait(), name="shutdown-watcher"
    )

    logger.info(
        f"Engine running | symbol={config.primary_symbol} "
        f"benchmark={config.benchmark_symbol} "
        f"live={config.is_live()} dev={config.dev_mode}"
    )

    try:
        done, pending = await asyncio.wait(
            [sentiment_task, ws_task, session_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Surface any unexpected exceptions from completed tasks
        for task in done:
            if task != shutdown_task and not task.cancelled():
                exc = task.exception()
                if exc is not None:
                    logger.critical(
                        f"Engine task {task.get_name()} raised: {exc!r}",
                        exc_info=exc,
                    )

    except asyncio.CancelledError:
        logger.info("Engine main task cancelled")

    finally:
        logger.info("Engine shutting down — closing positions and saving state")

        # Cancel background tasks
        for task in [sentiment_task, ws_task, session_task, shutdown_task]:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        # Flatten all positions
        try:
            await order_executor.close_all_positions("engine_shutdown")
        except Exception as exc:
            logger.error(f"Error closing positions on shutdown: {exc!r}")

        # Save session state
        try:
            session_pnl = trade_logger.get_session_pnl()
            data_manager.slow_buffer.data.update({
                "session_date": datetime.now(timezone.utc).isoformat(),
                "session_pnl": session_pnl,
                "signal_weights": kalman_ensemble.save_state(),
            })
            data_manager.slow_buffer.save(config.session_state_path)
        except Exception as exc:
            logger.error(f"Error saving session state on shutdown: {exc!r}")

        # Shut down WebSocket
        try:
            await connection_manager.shutdown()
        except Exception:
            pass

        logger.info("Engine stopped.")


if __name__ == "__main__":
    asyncio.run(_main())
