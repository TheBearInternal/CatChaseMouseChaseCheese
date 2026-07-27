"""OANDA v20 data adapter for the Universal Engine.

Provides two classes that replace Alpaca's data infrastructure when
``MARKET_TYPE=forex``:

* :class:`OANDAHistoricalFetcher` — fetches M1 candles from the OANDA REST API.
* :class:`OANDAStreamingConnection` — streams live price ticks and feeds
  :class:`~phase2.data.DataManager` on each update.

Signal logic, ensemble, Kalman Filter, GARCH, and execution architecture are
completely unmodified — only the data source changes.

Usage::

    # Warm-up (called automatically by DataManager.warm_up in forex mode)
    fetcher = OANDAHistoricalFetcher(api_key, account_id, environment)
    bars = fetcher.fetch_candles("EUR_USD", count=200)

    # Live streaming (wired by engine.py when MARKET_TYPE=forex)
    conn = OANDAStreamingConnection(api_key, account_id, environment,
                                    instruments=["EUR_USD", "GBP_USD"],
                                    data_manager=dm)
    await conn.run()
"""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from oandapyV20 import API
from oandapyV20.endpoints import instruments as oanda_instruments
from oandapyV20.endpoints import pricing

from logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants — mirror Phase 1 ConnectionManager reconnect settings
# ---------------------------------------------------------------------------

RECONNECT_INITIAL_DELAY_S: float = 1.0
RECONNECT_MAX_DELAY_S: float = 60.0
RECONNECT_MAX_ATTEMPTS: int = 10
HEALTH_CHECK_INTERVAL_S: float = 30.0


# ---------------------------------------------------------------------------
# OANDAHistoricalFetcher
# ---------------------------------------------------------------------------


class OANDAHistoricalFetcher:
    """Fetches historical candles from the OANDA v20 REST API.

    Used by :class:`~phase2.data.DataManager` during session warm-up to
    pre-populate ``_hist_bars`` and ``MediumBuffer`` with recent price history.
    All API errors are caught and logged; the caller receives an empty list
    rather than an exception.
    """

    def __init__(self, api_key: str, account_id: str, environment: str) -> None:
        """Initialise the fetcher.

        Args:
            api_key:     OANDA personal access token.  Never logged or printed.
            account_id:  V20 account ID (e.g. ``"001-001-XXXXXXX-XXX"``).
            environment: ``"live"`` or ``"practice"``.
        """
        self._client: API = API(access_token=api_key, environment=environment)
        self._account_id: str = account_id

    def fetch_candles(
        self,
        instrument: str,
        count: int = 200,
        granularity: str = "M1",
    ) -> List[Dict[str, Any]]:
        """Pull historical candles for one instrument from the OANDA REST API.

        Only complete (closed) candles are included.  The in-progress candle
        returned by OANDA is discarded.

        Args:
            instrument:  OANDA instrument string, e.g. ``"EUR_USD"``.
            count:       Number of candles to request (includes the incomplete
                         candle, so the returned list may have one fewer entry).
            granularity: OANDA granularity string (``"M1"``, ``"H1"`` …).

        Returns:
            List of bar dicts with keys: ``timestamp``, ``open``, ``high``,
            ``low``, ``close``, ``volume``, ``vwap_contribution``.
            Returns an empty list on any API or network error.
        """
        try:
            endpoint = oanda_instruments.InstrumentsCandles(
                instrument,
                params={
                    "count": str(count),
                    "granularity": granularity,
                    "price": "M",  # mid-point candles only
                },
            )
            response: Dict[str, Any] = self._client.request(endpoint)
            raw_candles: List[Dict[str, Any]] = response.get("candles", [])

            bars: List[Dict[str, Any]] = []
            for c in raw_candles:
                if not c.get("complete", True):
                    continue  # discard in-progress candle
                mid: Dict[str, str] = c.get("mid", {})
                ts_str: str = c.get("time", "")
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    ts = datetime.now(timezone.utc)

                close_price = float(mid.get("c", 0.0))
                volume = float(c.get("volume", 1))
                bars.append({
                    "timestamp": ts,
                    "open": float(mid.get("o", 0.0)),
                    "high": float(mid.get("h", 0.0)),
                    "low": float(mid.get("l", 0.0)),
                    "close": close_price,
                    "volume": volume,
                    "vwap_contribution": close_price * volume,
                })

            logger.info(
                f"OANDA historical: fetched {len(bars)} {instrument} "
                f"{granularity} candles"
            )
            return bars

        except Exception as exc:
            logger.error(
                f"OANDAHistoricalFetcher.fetch_candles({instrument}): {exc!r}"
            )
            return []


# ---------------------------------------------------------------------------
# OANDAStreamingConnection
# ---------------------------------------------------------------------------


class OANDAStreamingConnection:
    """Drives a live OANDA v20 price stream and routes each tick into DataManager.

    The OANDA streaming API is synchronous; it runs inside a
    :class:`~concurrent.futures.ThreadPoolExecutor` so the asyncio event loop
    remains unblocked.  Reconnection follows the same exponential backoff
    pattern as :class:`~phase1.connection.ConnectionManager`.

    Heartbeat messages (``type="HEARTBEAT"``) are silently discarded.
    PRICE messages are parsed and forwarded to
    :meth:`~phase2.data.DataManager.ingest_tick` via
    :func:`asyncio.run_coroutine_threadsafe`.
    """

    def __init__(
        self,
        api_key: str,
        account_id: str,
        environment: str,
        instruments: List[str],
        data_manager: Any,
        order_executor: Optional[Any] = None,
    ) -> None:
        """Initialise the streaming connection.

        Args:
            api_key:        OANDA personal access token.  Never logged or printed.
            account_id:     V20 account ID.
            environment:    ``"live"`` or ``"practice"``.
            instruments:    OANDA instrument strings to subscribe, e.g.
                            ``["EUR_USD", "GBP_USD"]``.
            data_manager:   :class:`~phase2.data.DataManager` that receives ticks.
            order_executor: Optional :class:`~phase2.execution.OrderExecutor`;
                            when set, position state is synced from OANDA after
                            each successful reconnect.
        """
        self._api_key: str = api_key
        self._account_id: str = account_id
        self._environment: str = environment
        self._instruments: List[str] = instruments
        self._dm: Any = data_manager
        self._order_executor: Optional[Any] = order_executor
        self._connected: bool = False
        self._shutdown_event: asyncio.Event = asyncio.Event()
        self._last_message_ts: float = 0.0
        self._thread_executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="oanda-stream"
        )

    @property
    def is_connected(self) -> bool:
        """Return ``True`` when the pricing stream is active."""
        return self._connected

    # ------------------------------------------------------------------
    # Synchronous stream (runs in executor thread)
    # ------------------------------------------------------------------

    def _stream_sync(self, loop: asyncio.AbstractEventLoop) -> None:
        """Synchronous price-stream loop — executes inside ThreadPoolExecutor.

        Iterates over the OANDA ``PricingStream`` generator and schedules
        :meth:`~phase2.data.DataManager.ingest_tick` on *loop* for each
        ``PRICE`` message.  Exits when ``_shutdown_event`` is set or the
        generator raises.

        Args:
            loop: The running asyncio event loop, captured in :meth:`run` and
                  passed here so :func:`asyncio.run_coroutine_threadsafe` has a
                  valid target without calling deprecated ``get_event_loop()``.
        """
        client = API(access_token=self._api_key, environment=self._environment)
        endpoint = pricing.PricingStream(
            self._account_id,
            params={"instruments": ",".join(self._instruments)},
        )
        for raw in client.request(endpoint):
            if self._shutdown_event.is_set():
                break

            msg_type: str = raw.get("type", "")
            if msg_type == "HEARTBEAT":
                continue
            if msg_type != "PRICE":
                continue

            instrument: str = raw.get("instrument", "UNKNOWN")
            bids: List[Dict[str, Any]] = raw.get("bids", [])
            asks: List[Dict[str, Any]] = raw.get("asks", [])
            bid = float(bids[0]["price"]) if bids else 0.0
            ask = float(asks[0]["price"]) if asks else 0.0
            mid = (bid + ask) / 2.0 if (bid and ask) else (bid or ask)

            ts_str: str = raw.get("time", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                ts = datetime.now(timezone.utc)

            tick: Dict[str, Any] = {
                "timestamp": ts,
                "symbol": instrument,
                "price": mid,
                "bid": bid,
                "ask": ask,
                "bid_size": 0,
                "ask_size": 0,
                "volume": 1.0,  # tick count used as volume proxy
            }
            self._last_message_ts = time.time()
            asyncio.run_coroutine_threadsafe(
                self._dm.ingest_tick(instrument, tick), loop
            )
            logger.debug(
                f"FOREX TICK | symbol={instrument} bid={bid:.5f} ask={ask:.5f}"
            )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def _health_check_loop(self) -> None:
        """Log connection status and message age every HEALTH_CHECK_INTERVAL_S.

        Exits cleanly when ``_shutdown_event`` is set.
        """
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=HEALTH_CHECK_INTERVAL_S,
                )
                break
            except asyncio.TimeoutError:
                pass

            if self._last_message_ts > 0:
                age_s = time.time() - self._last_message_ts
                logger.info(
                    f"FOREX Health | connected={self._connected} "
                    f"last_message={age_s:.1f}s ago"
                )
            else:
                logger.info(
                    f"FOREX Health | connected={self._connected} "
                    "no messages received yet"
                )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Drive the pricing stream with exponential backoff reconnection.

        Blocks until ``_shutdown_event`` is set or
        ``RECONNECT_MAX_ATTEMPTS`` consecutive failures are exceeded.
        """
        health_task = asyncio.create_task(
            self._health_check_loop(), name="forex-health-check"
        )
        loop = asyncio.get_running_loop()
        attempt: int = 0
        backoff_s: float = RECONNECT_INITIAL_DELAY_S

        try:
            while not self._shutdown_event.is_set():
                if attempt >= RECONNECT_MAX_ATTEMPTS:
                    logger.critical(
                        f"FOREX HALT — exceeded {RECONNECT_MAX_ATTEMPTS} "
                        "consecutive reconnection attempts without success."
                    )
                    break

                if attempt > 0:
                    logger.warning(
                        f"FOREX reconnect attempt {attempt}/{RECONNECT_MAX_ATTEMPTS} "
                        f"— backoff {backoff_s:.1f}s"
                    )
                    try:
                        await asyncio.wait_for(
                            self._shutdown_event.wait(), timeout=backoff_s
                        )
                        break  # shutdown requested during backoff
                    except asyncio.TimeoutError:
                        pass
                    backoff_s = min(backoff_s * 2.0, RECONNECT_MAX_DELAY_S)

                try:
                    logger.info(
                        f"FOREX connecting | env={self._environment} "
                        f"instruments={self._instruments}"
                    )
                    self._connected = True

                    # On reconnect, sync positions before resuming the stream
                    if attempt > 0 and self._order_executor is not None:
                        logger.info("Stream reconnect — syncing positions from OANDA")
                        await self._order_executor.sync_positions_from_oanda()

                    await loop.run_in_executor(
                        self._thread_executor,
                        lambda: self._stream_sync(loop),
                    )

                    if self._shutdown_event.is_set():
                        logger.info(
                            "FOREX stream exited cleanly — shutdown in progress"
                        )
                        break

                    logger.warning(
                        "FOREX stream exited without error — scheduling reconnect"
                    )
                    self._connected = False
                    attempt += 1
                    backoff_s = RECONNECT_INITIAL_DELAY_S  # reset on clean exit

                except asyncio.CancelledError:
                    self._connected = False
                    raise
                except Exception as exc:
                    self._connected = False
                    attempt += 1
                    logger.error(
                        f"FOREX stream error "
                        f"(attempt {attempt}/{RECONNECT_MAX_ATTEMPTS}): "
                        f"{type(exc).__name__}: {exc}"
                    )

        finally:
            self._connected = False
            self._shutdown_event.set()
            health_task.cancel()
            try:
                await health_task
            except asyncio.CancelledError:
                pass

    async def shutdown(self) -> None:
        """Stop the price stream and clean up resources.

        Safe to call multiple times — idempotent via ``_shutdown_event``.
        """
        logger.info("FOREX stream shutdown initiated")
        self._shutdown_event.set()
        self._connected = False
        self._thread_executor.shutdown(wait=False, cancel_futures=True)
        logger.info("FOREX stream shutdown complete")
