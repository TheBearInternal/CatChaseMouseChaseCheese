"""Alpaca WebSocket market data connection manager.

Manages the full lifecycle of a streaming connection to Alpaca's real-time
market data feed:

* Randomized startup delay sourced from :class:`~phase1.behavior.BehaviorProfile`
  so connections do not align mechanically with market open.
* Simultaneous trade and quote subscriptions for ``PRIMARY_SYMBOL`` and
  ``BENCHMARK_SYMBOL``.
* Exponential backoff reconnection on dropout: 1 s → 2 s → 4 s … capped at
  60 s, maximum 10 consecutive failures before raising a critical alert.
* Background health-check coroutine that logs connection status every 30 s.
* Clean SIGINT / SIGTERM shutdown with logged initiation and completion.
* Visible WARNING at startup when ``Config.is_live()`` is True.

Entry point::

    python -m phase1.connection
"""
from __future__ import annotations

import asyncio
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from alpaca.data.live import CryptoDataStream, StockDataStream

from config import config
from logger import get_logger
from phase1.behavior import BehaviorProfile

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

RECONNECT_INITIAL_DELAY_S: float = 1.0   # starting backoff delay in seconds
RECONNECT_MAX_DELAY_S: float = 60.0      # ceiling for backoff delay
RECONNECT_MAX_ATTEMPTS: int = 10         # consecutive failures before halt
HEALTH_CHECK_INTERVAL_S: int = 30        # seconds between health log entries


# ---------------------------------------------------------------------------
# Connection manager
# ---------------------------------------------------------------------------


class ConnectionManager:
    """Lifecycle manager for the Alpaca real-time market data WebSocket stream.

    Applies session-level behavioral randomization to startup timing and
    implements exponential backoff on any dropout, ensuring the connection is
    resilient without hammering the Alpaca endpoint.

    Typical usage::

        profile = BehaviorProfile(seed=config.behavior_seed)
        manager = ConnectionManager(profile)
        await manager.run()   # blocks until SIGINT/SIGTERM or max failures
    """

    def __init__(self, profile: BehaviorProfile) -> None:
        """Initialise the manager.

        Args:
            profile: Session behavioral profile supplying the startup delay and
                     any future per-session tuning parameters.
        """
        self._profile: BehaviorProfile = profile
        self._stream: Optional[StockDataStream] = None
        self._last_message_ts: float = 0.0
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._connected: bool = False
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="alpaca-stream"
        )

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    async def _handle_trade(self, data: object) -> None:
        """Process one incoming trade update from the stream.

        Extracts symbol, price, and timestamp then logs at INFO level.

        Args:
            data: Trade data object provided by alpaca-py.
        """
        self._last_message_ts = time.time()
        symbol: str = getattr(data, "symbol", "UNKNOWN")
        price: object = getattr(data, "price", None)
        timestamp: object = getattr(data, "timestamp", None)
        logger.info(f"TRADE | symbol={symbol} price={price} timestamp={timestamp}")

    async def _handle_quote(self, data: object) -> None:
        """Process one incoming quote update from the stream.

        Extracts symbol, bid, ask, and timestamp then logs at INFO level.

        Args:
            data: Quote data object provided by alpaca-py.
        """
        self._last_message_ts = time.time()
        symbol: str = getattr(data, "symbol", "UNKNOWN")
        bid: object = getattr(data, "bid_price", None)
        ask: object = getattr(data, "ask_price", None)
        timestamp: object = getattr(data, "timestamp", None)
        logger.info(
            f"QUOTE | symbol={symbol} bid={bid} ask={ask} timestamp={timestamp}"
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def _health_check_loop(self) -> None:
        """Log connection status and message receipt every HEALTH_CHECK_INTERVAL_S.

        Runs as a background task alongside the active stream.  Exits cleanly
        when ``_shutdown_event`` is set.
        """
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=float(HEALTH_CHECK_INTERVAL_S),
                )
                break  # shutdown event fired within the timeout
            except asyncio.TimeoutError:
                pass  # normal case — interval elapsed without shutdown

            if self._last_message_ts > 0:
                age_s = time.time() - self._last_message_ts
                logger.info(
                    f"Health | connected={self._connected} "
                    f"last_message={age_s:.1f}s ago"
                )
            else:
                logger.info(
                    f"Health | connected={self._connected} no messages received yet"
                )

    # ------------------------------------------------------------------
    # Stream construction
    # ------------------------------------------------------------------

    def _build_stream(self) -> object:
        """Construct and subscribe a fresh stream instance.

        Selects ``CryptoDataStream`` or ``StockDataStream`` based on
        ``config.is_crypto()``.  A new instance is created on every connection
        attempt so stale internal state from a previous dropout does not carry
        over.

        Returns:
            Stream object (CryptoDataStream or StockDataStream) ready to run.
        """
        benchmark = (
            config.crypto_benchmark if config.is_crypto() else config.benchmark_symbol
        )

        if config.is_crypto():
            stream = CryptoDataStream(
                api_key=config.alpaca_api_key,
                secret_key=config.alpaca_secret_key,
            )
            logger.info("Stream type: CryptoDataStream (24/7 digital assets)")
        else:
            stream = StockDataStream(
                api_key=config.alpaca_api_key,
                secret_key=config.alpaca_secret_key,
            )
            logger.info("Stream type: StockDataStream (US equities)")

        stream.subscribe_trades(
            self._handle_trade,
            config.primary_symbol,
            benchmark,
        )
        stream.subscribe_quotes(
            self._handle_quote,
            config.primary_symbol,
            benchmark,
        )
        return stream

    # ------------------------------------------------------------------
    # Reconnection loop
    # ------------------------------------------------------------------

    async def _connect_with_backoff(self) -> None:
        """Drive connection attempts with exponential backoff.

        Resets the attempt counter on any successful connection.  Raises once
        ``RECONNECT_MAX_ATTEMPTS`` consecutive failures have occurred.

        Raises:
            RuntimeError: When the maximum consecutive failure count is exceeded.
        """
        attempt: int = 0
        backoff_s: float = RECONNECT_INITIAL_DELAY_S

        while not self._shutdown_event.is_set():
            if attempt >= RECONNECT_MAX_ATTEMPTS:
                logger.critical(
                    f"HALT — exceeded {RECONNECT_MAX_ATTEMPTS} consecutive "
                    "reconnection attempts without a successful connection."
                )
                raise RuntimeError(
                    f"Maximum reconnection attempts ({RECONNECT_MAX_ATTEMPTS}) exceeded."
                )

            if attempt > 0:
                logger.warning(
                    f"Reconnect attempt {attempt}/{RECONNECT_MAX_ATTEMPTS} "
                    f"— backoff delay {backoff_s:.1f}s"
                )
                # Honour shutdown requests received during the backoff wait.
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=backoff_s
                    )
                    logger.info("Shutdown requested during backoff — aborting reconnect")
                    return
                except asyncio.TimeoutError:
                    pass
                backoff_s = min(backoff_s * 2.0, RECONNECT_MAX_DELAY_S)

            try:
                logger.info(
                    f"Connecting to Alpaca stream | "
                    f"base_url={config.alpaca_base_url} "
                    f"symbols=[{config.primary_symbol}, {config.benchmark_symbol}]"
                )
                self._stream = self._build_stream()

                self._connected = True
                logger.info(
                    f"Stream active | subscribed to trades+quotes for "
                    f"{config.primary_symbol} and {config.benchmark_symbol}"
                )

                # StockDataStream.run() calls asyncio.run() internally, so it
                # must execute in a separate thread to avoid nested event-loop
                # conflicts.
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self._executor, self._stream.run)

                if self._shutdown_event.is_set():
                    logger.info("Stream exited cleanly — shutdown in progress")
                    return

                # Stream returned without an exception and without shutdown being
                # requested — treat as an unexpected dropout.
                logger.warning(
                    "Stream exited without error — scheduling reconnect"
                )
                self._connected = False
                attempt += 1
                backoff_s = RECONNECT_INITIAL_DELAY_S  # reset backoff on clean exit

            except asyncio.CancelledError:
                logger.info("Connection task cancelled — shutting down")
                self._connected = False
                raise

            except Exception as exc:
                self._connected = False
                attempt += 1
                logger.error(
                    f"Stream error (attempt {attempt}/{RECONNECT_MAX_ATTEMPTS}): "
                    f"{type(exc).__name__}: {exc}"
                )

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _schedule_shutdown(self, sig_name: str) -> None:
        """Schedule the async shutdown coroutine from a synchronous signal context.

        Required on Windows where ``signal.signal()`` callbacks are synchronous
        and cannot directly await coroutines.  Uses ``call_soon_threadsafe`` so
        it is safe to call from any thread.

        Args:
            sig_name: Signal name string used in the log message.
        """
        logger.info(f"Received {sig_name} — initiating graceful shutdown")
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(loop.create_task, self.shutdown())

    def _register_signal_handlers(self) -> None:
        """Register SIGINT and SIGTERM handlers for graceful shutdown.

        Attempts the Unix-style ``loop.add_signal_handler`` first.  Falls back
        to ``signal.signal`` on Windows where ``add_signal_handler`` raises
        ``NotImplementedError``; in that case only SIGINT is registered because
        SIGTERM is not reliably available on Windows.
        """
        loop = asyncio.get_running_loop()

        try:
            loop.add_signal_handler(signal.SIGINT, lambda: self._schedule_shutdown("SIGINT"))
            loop.add_signal_handler(signal.SIGTERM, lambda: self._schedule_shutdown("SIGTERM"))
            logger.debug("Signal handlers registered via loop.add_signal_handler (Unix)")
        except NotImplementedError:
            # Windows fallback — synchronous signal.signal; SIGTERM omitted as it
            # is not reliably available on Windows.
            signal.signal(signal.SIGINT, lambda sig, frame: self._schedule_shutdown("SIGINT"))
            logger.debug("Signal handlers registered via signal.signal (Windows fallback)")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start the connection manager.

        Steps performed, in order:
        1. Log a live-account warning if ``Config.is_live()`` is True.
        2. Register SIGINT / SIGTERM handlers.
        3. Apply randomized startup delay from the behavior profile.
        4. Launch background health-check coroutine.
        5. Enter the reconnection loop — blocks until shutdown or max failures.
        6. Cancel the health-check task and clean up on exit.
        """
        if config.is_live():
            logger.warning(
                "=== LIVE ACCOUNT DETECTED === "
                "This system is connected to a real funded Alpaca account. "
                "All safeguards are active. Proceed with extreme caution."
            )

        self._register_signal_handlers()

        if not config.dev_mode:
            startup_delay = self._profile.get_connection_delay()
            logger.info(
                f"Applying randomized startup delay: {startup_delay:.2f}s "
                "(sourced from BehaviorProfile)"
            )
            await asyncio.sleep(startup_delay)
        else:
            logger.debug("DEV_MODE: startup delay bypassed")

        health_task = asyncio.create_task(
            self._health_check_loop(), name="health-check"
        )

        try:
            await self._connect_with_backoff()
        finally:
            self._shutdown_event.set()
            health_task.cancel()
            try:
                await health_task
            except asyncio.CancelledError:
                pass

    async def shutdown(self) -> None:
        """Gracefully stop the stream and signal all background loops to exit.

        Safe to call multiple times — idempotent via ``_shutdown_event``.
        """
        logger.info("Shutdown initiated")
        self._shutdown_event.set()
        self._connected = False

        if self._stream is not None:
            try:
                self._stream.stop()
                logger.info("Stream stopped cleanly")
            except Exception as exc:
                logger.warning(f"Error while stopping stream: {type(exc).__name__}: {exc}")

        self._executor.shutdown(wait=False, cancel_futures=True)
        logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------


async def _main() -> None:
    """Async entry point for ``python -m phase1.connection``."""
    profile = BehaviorProfile(seed=config.behavior_seed)
    manager = ConnectionManager(profile=profile)
    await manager.run()


if __name__ == "__main__":
    asyncio.run(_main())
