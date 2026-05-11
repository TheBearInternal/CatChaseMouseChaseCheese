"""Order execution, position tracking, and trade logging for hft_lab Phase 2.

Classes
-------
Position      : Open position dataclass — entry metadata only.
TradeRecord   : Closed trade dataclass — full round-trip metadata.
TradeLogger   : Appends TradeRecord rows to a CSV file.
OrderExecutor : Submits bracket orders via alpaca-py, tracks open positions,
                detects fills and closures, enforces MAX_OPEN_POSITIONS.

All API calls to Alpaca are routed through ``PacedAPIClient`` (Phase 1
behavior module) for timing randomization and rate limiting.  Synchronous
alpaca-py calls run in a thread executor to keep the event loop free.
"""
from __future__ import annotations

import asyncio
import csv
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import config
from logger import get_logger
from phase2.data import DataManager
from phase2.sizing import SpreadAdjuster

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STOP_LIMIT_BUFFER_PCT: float = 0.002   # stop-limit price offset from stop trigger
ORDER_TIME_IN_FORCE: str = "day"
POSITION_CHECK_TIMEOUT_S: float = 5.0  # executor timeout for Alpaca REST calls
ORDER_COOLDOWN_S: float = 30.0         # cooldown after a failed order submission


# ---------------------------------------------------------------------------
# Position dataclass
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """Represents one open position submitted to Alpaca.

    All dollar amounts are in USD.
    """

    symbol: str
    side: str                          # "LONG" or "SHORT"
    entry_price: float
    quantity: int
    stop_price: float
    take_profit_price: float
    entry_time: datetime
    order_id: str
    signal_scores_at_entry: Dict[str, float]
    weights_at_entry: Dict[str, float]
    regime_at_entry: str


# ---------------------------------------------------------------------------
# TradeRecord dataclass
# ---------------------------------------------------------------------------


@dataclass
class TradeRecord:
    """Immutable record of a completed round-trip trade.

    Extends the fields from ``Position`` with exit metadata and PnL figures.
    """

    symbol: str
    side: str
    entry_price: float
    quantity: int
    stop_price: float
    take_profit_price: float
    entry_time: datetime
    order_id: str
    signal_scores_at_entry: Dict[str, float]
    weights_at_entry: Dict[str, float]
    regime_at_entry: str
    exit_price: float
    exit_time: datetime
    exit_reason: str               # "stop_loss" | "take_profit" | "session_close" | "manual"
    gross_pnl: float
    net_pnl: float
    spread_cost: float
    duration_seconds: float


# ---------------------------------------------------------------------------
# TradeLogger
# ---------------------------------------------------------------------------

_CSV_COLUMNS: tuple[str, ...] = (
    "symbol", "side", "entry_price", "quantity", "stop_price",
    "take_profit_price", "entry_time", "order_id", "regime_at_entry",
    "exit_price", "exit_time", "exit_reason",
    "gross_pnl", "net_pnl", "spread_cost", "duration_seconds",
)


class TradeLogger:
    """Appends closed trade records to a CSV file.

    Writes the header row on first use if the file does not yet exist.
    """

    def __init__(self) -> None:
        self._path: str = config.trade_log_path
        self._session_pnl: float = 0.0
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        if not os.path.exists(self._path):
            with open(self._path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(_CSV_COLUMNS)

    def log_trade(self, record: TradeRecord) -> None:
        """Append one TradeRecord row to the CSV.

        Args:
            record: Completed trade to log.
        """
        self._session_pnl += record.net_pnl
        row = [
            record.symbol,
            record.side,
            f"{record.entry_price:.4f}",
            record.quantity,
            f"{record.stop_price:.4f}",
            f"{record.take_profit_price:.4f}",
            record.entry_time.isoformat(),
            record.order_id,
            record.regime_at_entry,
            f"{record.exit_price:.4f}",
            record.exit_time.isoformat(),
            record.exit_reason,
            f"{record.gross_pnl:.4f}",
            f"{record.net_pnl:.4f}",
            f"{record.spread_cost:.4f}",
            f"{record.duration_seconds:.1f}",
        ]
        with open(self._path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        logger.info(
            f"Trade logged | {record.side} {record.symbol} "
            f"entry={record.entry_price:.2f} exit={record.exit_price:.2f} "
            f"gross={record.gross_pnl:+.2f} net={record.net_pnl:+.2f} "
            f"reason={record.exit_reason}"
        )

    def get_session_pnl(self) -> float:
        """Return cumulative net PnL for this session.

        Returns:
            Session net PnL in dollars.
        """
        return self._session_pnl

    def reset_session_pnl(self) -> None:
        """Reset the session PnL counter (call at session open)."""
        self._session_pnl = 0.0


# ---------------------------------------------------------------------------
# OrderExecutor
# ---------------------------------------------------------------------------


class OrderExecutor:
    """Submits bracket orders and manages the open-position lifecycle.

    All Alpaca REST calls are synchronous; they are dispatched to a thread
    executor so the async trading loop is never blocked.  ``PacedAPIClient``
    wraps each executor call with jitter and token-bucket rate limiting.
    """

    def __init__(
        self,
        trading_client: Any,
        data_manager: DataManager,
        spread_adjuster: SpreadAdjuster,
        trade_logger: TradeLogger,
        profile: Any,
    ) -> None:
        """Initialise the executor.

        Args:
            trading_client:  alpaca-py ``TradingClient`` instance.
            data_manager:    DataManager for live price access.
            spread_adjuster: SpreadAdjuster for fill price and cost checks.
            trade_logger:    TradeLogger for recording closed trades.
            profile:         BehaviorProfile for gaussian_jitter on orders.
        """
        self._client = trading_client
        self._dm = data_manager
        self._spread = spread_adjuster
        self._logger = trade_logger
        self._profile = profile
        self._open_positions: Dict[str, Position] = {}  # keyed by order_id
        self._last_order_attempt_ts: float = 0.0        # cooldown after failed orders
        self._last_order_time: Dict[str, float] = {}    # per-symbol submission timestamp

        from phase1.behavior import PacedAPIClient, default_rate_limiter
        self._paced = PacedAPIClient(
            callable_fn=self._noop_async,
            rate_limiter=default_rate_limiter,
            profile=profile,
        )

        self._oanda_client: Optional[Any] = None
        if config.is_forex():
            from oandapyV20 import API as OandaAPI
            self._oanda_client = OandaAPI(
                access_token=config.oanda_api_key,
                environment=config.oanda_environment,
            )
            self._sync_positions_from_oanda()

    def _sync_positions_from_oanda(self) -> None:
        """Populate _open_positions from OANDA open positions on startup.

        Restores position state so the session can correctly detect exits after
        a restart without creating ghost positions or missing close events.
        """
        try:
            from oandapyV20.endpoints.positions import OpenPositions

            r = OpenPositions(config.oanda_account_id)
            response: Dict[str, Any] = self._oanda_client.request(r)
            positions = response.get("positions", [])
            for pos in positions:
                instrument = pos.get("instrument", "")
                long_units = float(pos.get("long", {}).get("units", 0))
                short_units = float(pos.get("short", {}).get("units", 0))
                if abs(long_units) < 1 and abs(short_units) < 1:
                    continue
                side = "LONG" if long_units > 0 else "SHORT"
                qty = int(abs(long_units if long_units != 0 else short_units))
                avg_price = float(
                    pos.get("long" if side == "LONG" else "short", {}).get(
                        "averagePrice", 0.0
                    )
                )
                synthetic_id = f"oanda_sync_{instrument}_{int(time.time())}"
                self._open_positions[synthetic_id] = Position(
                    symbol=instrument,
                    side=side,
                    entry_price=avg_price,
                    quantity=qty,
                    stop_price=0.0,
                    take_profit_price=0.0,
                    entry_time=datetime.now(timezone.utc),
                    order_id=synthetic_id,
                    signal_scores_at_entry={},
                    weights_at_entry={},
                    regime_at_entry="unknown",
                )
            logger.info(
                f"Position sync | found {len(self._open_positions)} open "
                f"position(s) from OANDA"
            )
        except Exception as exc:
            logger.warning(f"OANDA position sync failed: {exc!r} — starting with empty state")

    def _count_oanda_positions(self) -> Optional[int]:
        """Return the number of non-zero positions OANDA currently holds, or None on error.

        Used for post-fill verification without modifying _open_positions.
        """
        try:
            from oandapyV20.endpoints.positions import OpenPositions

            r = OpenPositions(config.oanda_account_id)
            response: Dict[str, Any] = self._oanda_client.request(r)
            count = sum(
                1
                for pos in response.get("positions", [])
                if abs(float(pos.get("long", {}).get("units", 0))) >= 1
                or abs(float(pos.get("short", {}).get("units", 0))) >= 1
            )
            return count
        except Exception as exc:
            logger.warning(f"OANDA position count check failed: {exc!r}")
            return None

    @staticmethod
    async def _noop_async() -> None:
        pass

    def _run_sync(self, fn, *args, **kwargs):
        """Schedule a synchronous alpaca-py call for execution in a thread."""
        loop = asyncio.get_event_loop()
        return loop.run_in_executor(None, lambda: fn(*args, **kwargs))

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    async def _submit_forex_bracket(
        self,
        side: str,
        qty: int,
        entry_price: float,
        stop_price: float,
        tp_price: float,
        atr_value: float,
        signal_scores: Dict[str, float],
        weights: Dict[str, float],
        regime_name: str,
    ) -> Optional[Position]:
        """Submit a market bracket order to OANDA and return a tracked Position.

        OANDA handles stop-loss and take-profit server-side.  Positive units
        indicate a long (buy); negative units indicate a short (sell).

        Args:
            side:          "LONG" or "SHORT".
            qty:           OANDA units (base currency, e.g. 1000 = 1 micro-lot).
            entry_price:   Mid price at submission time (for tracking only).
            stop_price:    Stop-loss price in instrument quote currency.
            tp_price:      Take-profit price in instrument quote currency.
            atr_value:     Raw ATR used for stop/tp calculation (logged).
            signal_scores: Signal scores at decision time.
            weights:       Ensemble weights at decision time.
            regime_name:   RegimeState name string.

        Returns:
            ``Position`` on success, ``None`` on any error.
        """
        from oandapyV20.endpoints import orders as oanda_orders

        instrument = config.primary_symbol
        units = str(qty) if side == "LONG" else str(-qty)
        order_data = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": units,
                "takeProfitOnFill": {
                    "price": f"{tp_price:.5f}",
                },
                "stopLossOnFill": {
                    "price": f"{stop_price:.5f}",
                    "timeInForce": "GTC",
                },
            }
        }

        logger.info(
            f"Order | {side} {qty} {instrument} @ {entry_price:.5f} | "
            f"stop={stop_price:.5f} tp={tp_price:.5f} | ATR={atr_value:.5f}"
        )

        try:
            loop = asyncio.get_running_loop()
            r = oanda_orders.OrderCreate(config.oanda_account_id, data=order_data)
            response: Dict[str, Any] = await loop.run_in_executor(
                None, lambda: self._oanda_client.request(r)
            )

            fill_tx = response.get("orderFillTransaction")
            if fill_tx is None:
                # Order was queued or rejected — no position opened
                order_id = (
                    response.get("orderCreateTransaction", {}).get("id")
                    or f"oanda_{int(time.time())}"
                )
                logger.warning(
                    f"Order id={order_id} did not result in immediate fill "
                    "— not counting as open position"
                )
                self._last_order_attempt_ts = 0.0
                return None

            order_id = fill_tx.get("id") or f"oanda_{int(time.time())}"
            position = Position(
                symbol=instrument,
                side=side,
                entry_price=entry_price,
                quantity=qty,
                stop_price=stop_price,
                take_profit_price=tp_price,
                entry_time=datetime.now(timezone.utc),
                order_id=str(order_id),
                signal_scores_at_entry=dict(signal_scores),
                weights_at_entry=dict(weights),
                regime_at_entry=regime_name,
            )
            self._open_positions[str(order_id)] = position
            logger.info(f"OANDA order submitted: id={order_id} regime={regime_name}")
            self._last_order_attempt_ts = 0.0  # clear cooldown on success
            self._last_order_time[instrument] = time.time()

            # Post-fill verification: compare local count to OANDA without overwriting metadata
            oanda_count = self._count_oanda_positions()
            local_count = len(self._open_positions)
            if oanda_count is not None and oanda_count != local_count:
                logger.warning(
                    f"Position count mismatch after fill: local={local_count}, "
                    f"OANDA reports {oanda_count} — trusting OANDA"
                )

            return position

        except Exception as exc:
            logger.error(f"OANDA order submission failed: {exc!r}")
            self._last_order_attempt_ts = time.time()
            return None

    async def submit_bracket_order(
        self,
        side: str,
        quantity: int,
        stop_distance: float,
        take_profit_distance: float,
        signal_scores: Dict[str, float],
        weights: Dict[str, float],
        regime_name: str,
        atr_value: float = 0.0,
    ) -> Optional[Position]:
        """Submit a bracket order and return an open Position on success.

        Routes to OANDA when ``config.is_forex()``, otherwise submits a
        Alpaca bracket order.  Enforces a 30-second retry cooldown after
        any failed submission.

        Args:
            side:                  "LONG" or "SHORT".
            quantity:              Shares (equity/crypto) or units (forex).
            stop_distance:         Stop-loss distance in price units.
            take_profit_distance:  Take-profit distance in price units.
            signal_scores:         Signal scores at decision time.
            weights:               Ensemble weights at decision time.
            regime_name:           RegimeState name string.
            atr_value:             Raw ATR used for stop/tp (logged only).

        Returns:
            ``Position`` on successful submission, ``None`` on any error.
        """
        from phase1.behavior import gaussian_jitter

        # --- Cooldown gate ------------------------------------------------
        elapsed = time.time() - self._last_order_attempt_ts
        if self._last_order_attempt_ts > 0 and elapsed < ORDER_COOLDOWN_S:
            remaining = int(ORDER_COOLDOWN_S - elapsed)
            logger.warning(
                f"Order cooldown active | {remaining}s remaining — skipping signal"
            )
            return None

        # --- Position limit -----------------------------------------------
        if len(self._open_positions) >= config.max_open_positions:
            logger.debug(
                f"MAX_OPEN_POSITIONS ({config.max_open_positions}) reached — skipping"
            )
            return None

        # --- Entry price (mid) and bracket levels -------------------------
        bid = self._dm.fast_primary.latest_bid()
        ask = self._dm.fast_primary.latest_ask()
        if bid is not None and ask is not None:
            entry_price = (bid + ask) / 2.0
        else:
            entry_price = self._dm.fast_primary.latest_price()
        if entry_price is None:
            logger.warning("Cannot submit order: price unavailable")
            return None

        if side == "LONG":
            stop_price = entry_price - stop_distance
            tp_price = entry_price + take_profit_distance
        else:
            stop_price = entry_price + stop_distance
            tp_price = entry_price - take_profit_distance

        # --- Pre-submission jitter ----------------------------------------
        jitter_ms = gaussian_jitter(0.0, self._profile.polling_jitter_sigma)
        await asyncio.sleep(jitter_ms / 1000.0)

        # --- Forex path (OANDA) -------------------------------------------
        if config.is_forex():
            return await self._submit_forex_bracket(
                side=side,
                qty=quantity,
                entry_price=entry_price,
                stop_price=round(stop_price, 5),
                tp_price=round(tp_price, 5),
                atr_value=atr_value,
                signal_scores=signal_scores,
                weights=weights,
                regime_name=regime_name,
            )

        # --- Equity / Crypto path (Alpaca) --------------------------------
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
        from alpaca.trading.requests import (
            LimitOrderRequest,
            TakeProfitRequest,
            StopLossRequest,
        )

        fill_price = self._spread.adjust_fill_price(side)
        if fill_price is None:
            logger.warning("Cannot submit order: fill price unavailable")
            return None

        alpaca_side = OrderSide.BUY if side == "LONG" else OrderSide.SELL
        stop_price_r = round(stop_price, 2)
        tp_price_r = round(tp_price, 2)
        fill_price_r = round(fill_price, 2)

        # Live account extra validation
        if config.is_live():
            if quantity * fill_price_r > config.account_limit * 0.5:
                logger.warning(
                    f"Live account: order value ${quantity * fill_price_r:.0f} exceeds "
                    "50% of ACCOUNT_LIMIT — rejected"
                )
                return None

        order_data = LimitOrderRequest(
            symbol=config.primary_symbol,
            qty=quantity,
            side=alpaca_side,
            time_in_force=TimeInForce.DAY,
            limit_price=fill_price_r,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=tp_price_r),
            stop_loss=StopLossRequest(
                stop_price=stop_price_r,
                limit_price=round(
                    stop_price_r * (0.998 if side == "LONG" else 1.002), 2
                ),
            ),
        )

        logger.info(
            f"Order | {side} {quantity} {config.primary_symbol} "
            f"@ {entry_price:.2f} | "
            f"stop={stop_price_r:.2f} tp={tp_price_r:.2f} | "
            f"ATR={atr_value:.4f} regime={regime_name}"
        )

        try:
            order = await self._run_sync(
                self._client.submit_order, order_data=order_data
            )
            position = Position(
                symbol=config.primary_symbol,
                side=side,
                entry_price=entry_price,
                quantity=quantity,
                stop_price=stop_price_r,
                take_profit_price=tp_price_r,
                entry_time=datetime.now(timezone.utc),
                order_id=str(order.id),
                signal_scores_at_entry=dict(signal_scores),
                weights_at_entry=dict(weights),
                regime_at_entry=regime_name,
            )
            self._open_positions[str(order.id)] = position
            logger.info(f"Alpaca order submitted: id={order.id} status={order.status}")
            self._last_order_attempt_ts = 0.0  # clear cooldown on success
            return position

        except Exception as exc:
            logger.error(f"Order submission failed: {exc!r}")
            self._last_order_attempt_ts = time.time()
            return None

    # ------------------------------------------------------------------
    # Position monitoring
    # ------------------------------------------------------------------

    async def check_positions(self) -> List[TradeRecord]:
        """Poll for position updates and detect newly closed positions.

        For equity/crypto, queries Alpaca.  For forex, OANDA manages stop-loss
        and take-profit server-side, so this method returns an empty list
        (closures are reported via OANDA's streaming transaction feed, not polled).

        Returns:
            List of ``TradeRecord`` instances for every position that closed
            since the last call (stop hit, take-profit hit, or manually closed).
        """
        if not self._open_positions:
            return []

        if config.is_forex():
            return []

        try:
            alpaca_positions = await self._run_sync(self._client.get_all_positions)
            live_ids = {str(p.asset_id) for p in alpaca_positions}
        except Exception as exc:
            logger.error(f"check_positions: failed to fetch Alpaca positions — {exc!r}")
            return []

        try:
            all_orders = await self._run_sync(
                self._client.get_orders,
                filter=None,
            )
            # Build map from order id to order status
            order_map = {str(o.id): o for o in (all_orders or [])}
        except Exception:
            order_map = {}

        closed_trades: List[TradeRecord] = []
        still_open: Dict[str, Position] = {}

        for oid, pos in self._open_positions.items():
            order = order_map.get(oid)
            filled = order is not None and str(order.status) in (
                "filled", "partially_filled"
            )
            # Check if position symbol is still in live Alpaca positions
            symbol_held = any(
                getattr(p, "symbol", None) == pos.symbol
                for p in (alpaca_positions or [])
            )

            if filled and not symbol_held:
                # Position closed — determine exit price and reason
                exit_price = self._spread.adjust_fill_price(
                    "SHORT" if pos.side == "LONG" else "LONG"
                ) or pos.entry_price
                exit_time = datetime.now(timezone.utc)
                duration = (exit_time - pos.entry_time).total_seconds()

                if pos.side == "LONG":
                    gross = (exit_price - pos.entry_price) * pos.quantity
                else:
                    gross = (pos.entry_price - exit_price) * pos.quantity

                net = self._spread.net_pnl(gross, pos.quantity)
                spread_cost = gross - net

                # Infer reason from exit price proximity
                if abs(exit_price - pos.stop_price) < abs(exit_price - pos.take_profit_price):
                    reason = "stop_loss"
                else:
                    reason = "take_profit"

                record = TradeRecord(
                    symbol=pos.symbol,
                    side=pos.side,
                    entry_price=pos.entry_price,
                    quantity=pos.quantity,
                    stop_price=pos.stop_price,
                    take_profit_price=pos.take_profit_price,
                    entry_time=pos.entry_time,
                    order_id=pos.order_id,
                    signal_scores_at_entry=pos.signal_scores_at_entry,
                    weights_at_entry=pos.weights_at_entry,
                    regime_at_entry=pos.regime_at_entry,
                    exit_price=exit_price,
                    exit_time=exit_time,
                    exit_reason=reason,
                    gross_pnl=gross,
                    net_pnl=net,
                    spread_cost=spread_cost,
                    duration_seconds=duration,
                )
                self._logger.log_trade(record)
                closed_trades.append(record)
            else:
                still_open[oid] = pos

        self._open_positions = still_open
        return closed_trades

    async def close_all_positions(self, reason: str) -> None:
        """Emergency flatten: submit market orders to close every open position.

        For forex, issues OANDA position-close requests for each tracked
        position.  For equity/crypto, calls Alpaca's bulk close endpoint.

        Args:
            reason: Exit reason string recorded in TradeRecord.
        """
        if not self._open_positions:
            return
        logger.info(f"Closing all positions — reason: {reason}")

        if config.is_forex():
            from oandapyV20.endpoints import positions as oanda_positions
            now = datetime.now(timezone.utc)
            closed: List[str] = []
            for oid, pos in self._open_positions.items():
                try:
                    data = {
                        "longUnits": "ALL" if pos.side == "LONG" else "NONE",
                        "shortUnits": "ALL" if pos.side == "SHORT" else "NONE",
                    }
                    r = oanda_positions.PositionClose(
                        config.oanda_account_id,
                        instrument=pos.symbol,
                        data=data,
                    )
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None, lambda: self._oanda_client.request(r)
                    )
                    price = self._dm.fast_primary.latest_price() or pos.entry_price
                    gross = (
                        (price - pos.entry_price) * pos.quantity
                        if pos.side == "LONG"
                        else (pos.entry_price - price) * pos.quantity
                    )
                    net = self._spread.net_pnl(gross, pos.quantity)
                    self._logger.log_trade(TradeRecord(
                        symbol=pos.symbol, side=pos.side,
                        entry_price=pos.entry_price, quantity=pos.quantity,
                        stop_price=pos.stop_price,
                        take_profit_price=pos.take_profit_price,
                        entry_time=pos.entry_time, order_id=pos.order_id,
                        signal_scores_at_entry=pos.signal_scores_at_entry,
                        weights_at_entry=pos.weights_at_entry,
                        regime_at_entry=pos.regime_at_entry,
                        exit_price=price, exit_time=now, exit_reason=reason,
                        gross_pnl=gross, net_pnl=net, spread_cost=gross - net,
                        duration_seconds=(now - pos.entry_time).total_seconds(),
                    ))
                    closed.append(oid)
                except Exception as exc:
                    exc_str = str(exc)
                    if "CLOSEOUT_POSITION_DOESNT_EXIST" in exc_str:
                        logger.warning(
                            f"Position {oid} not found on OANDA — removing from tracker"
                        )
                        closed.append(oid)
                    else:
                        logger.error(f"OANDA close position {oid} failed: {exc!r}")
            for oid in closed:
                del self._open_positions[oid]
            return

        try:
            await self._run_sync(self._client.close_all_positions, cancel_orders=True)
            now = datetime.now(timezone.utc)
            for pos in self._open_positions.values():
                price = self._spread.adjust_fill_price(
                    "SHORT" if pos.side == "LONG" else "LONG"
                ) or pos.entry_price
                if pos.side == "LONG":
                    gross = (price - pos.entry_price) * pos.quantity
                else:
                    gross = (pos.entry_price - price) * pos.quantity
                net = self._spread.net_pnl(gross, pos.quantity)
                self._logger.log_trade(TradeRecord(
                    symbol=pos.symbol,
                    side=pos.side,
                    entry_price=pos.entry_price,
                    quantity=pos.quantity,
                    stop_price=pos.stop_price,
                    take_profit_price=pos.take_profit_price,
                    entry_time=pos.entry_time,
                    order_id=pos.order_id,
                    signal_scores_at_entry=pos.signal_scores_at_entry,
                    weights_at_entry=pos.weights_at_entry,
                    regime_at_entry=pos.regime_at_entry,
                    exit_price=price,
                    exit_time=now,
                    exit_reason=reason,
                    gross_pnl=gross,
                    net_pnl=net,
                    spread_cost=gross - net,
                    duration_seconds=(now - pos.entry_time).total_seconds(),
                ))
            self._open_positions.clear()
        except Exception as exc:
            logger.error(f"close_all_positions failed: {exc!r}")

    def update_positions(self, positions_list: List[Any]) -> None:
        """Sync internal position state with Alpaca's live position list.

        Removes any tracked positions whose symbols no longer appear in the
        live list, preventing ghost positions from accumulating.

        Args:
            positions_list: List of Alpaca position objects.
        """
        live_symbols = {getattr(p, "symbol", None) for p in positions_list}
        stale = [
            oid for oid, pos in self._open_positions.items()
            if pos.symbol not in live_symbols
        ]
        for oid in stale:
            logger.warning(f"Removing stale position tracking for order {oid}")
            del self._open_positions[oid]

    @property
    def open_positions(self) -> List[Position]:
        """Current list of tracked open positions."""
        return list(self._open_positions.values())

    @property
    def total_exposure(self) -> float:
        """Sum of (entry_price × quantity) for all open positions."""
        price = self._dm.fast_primary.latest_price() or 0.0
        return sum(pos.quantity * price for pos in self._open_positions.values())
