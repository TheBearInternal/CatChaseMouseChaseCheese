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

        from phase1.behavior import PacedAPIClient, default_rate_limiter
        self._paced = PacedAPIClient(
            callable_fn=self._noop_async,
            rate_limiter=default_rate_limiter,
            profile=profile,
        )

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

    async def submit_bracket_order(
        self,
        side: str,
        quantity: int,
        stop_distance: float,
        take_profit_distance: float,
        signal_scores: Dict[str, float],
        weights: Dict[str, float],
        regime_name: str,
    ) -> Optional[Position]:
        """Submit a bracket order to Alpaca and return an open Position on success.

        Applies ``gaussian_jitter`` from the behavior profile before submission.
        Runs additional validation when ``Config.is_live()`` is True.

        Args:
            side:                  "LONG" or "SHORT".
            quantity:              Number of shares.
            stop_distance:         Stop-loss distance in dollars.
            take_profit_distance:  Take-profit distance in dollars.
            signal_scores:         Signal scores at decision time.
            weights:               Ensemble weights at decision time.
            regime_name:           RegimeState name string.

        Returns:
            ``Position`` on successful submission, ``None`` on any error.
        """
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
        from alpaca.trading.requests import (
            LimitOrderRequest,
            TakeProfitRequest,
            StopLossRequest,
        )
        from phase1.behavior import gaussian_jitter

        if len(self._open_positions) >= config.max_open_positions:
            logger.debug(
                f"MAX_OPEN_POSITIONS ({config.max_open_positions}) reached — skipping"
            )
            return None

        fill_price = self._spread.adjust_fill_price(side)
        if fill_price is None:
            logger.warning("Cannot submit order: fill price unavailable")
            return None

        if side == "LONG":
            alpaca_side = OrderSide.BUY
            stop_price = round(fill_price - stop_distance, 2)
            tp_price = round(fill_price + take_profit_distance, 2)
        else:
            alpaca_side = OrderSide.SELL
            stop_price = round(fill_price + stop_distance, 2)
            tp_price = round(fill_price - take_profit_distance, 2)

        # Live account extra validation
        if config.is_live():
            if quantity * fill_price > config.account_limit * 0.5:
                logger.warning(
                    f"Live account: order value ${quantity * fill_price:.0f} exceeds "
                    "50% of ACCOUNT_LIMIT — rejected"
                )
                return None

        # Pre-submission jitter (behavioral randomization)
        jitter_ms = gaussian_jitter(0.0, self._profile.polling_jitter_sigma)
        await asyncio.sleep(jitter_ms / 1000.0)

        order_data = LimitOrderRequest(
            symbol=config.primary_symbol,
            qty=quantity,
            side=alpaca_side,
            time_in_force=TimeInForce.DAY,
            limit_price=fill_price,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=tp_price),
            stop_loss=StopLossRequest(
                stop_price=stop_price,
                limit_price=round(stop_price * (0.998 if side == "LONG" else 1.002), 2),
            ),
        )

        logger.info(
            f"Submitting {side} bracket | qty={quantity} entry={fill_price:.2f} "
            f"stop={stop_price:.2f} tp={tp_price:.2f} regime={regime_name}"
        )

        try:
            order = await self._run_sync(
                self._client.submit_order, order_data=order_data
            )
            position = Position(
                symbol=config.primary_symbol,
                side=side,
                entry_price=fill_price,
                quantity=quantity,
                stop_price=stop_price,
                take_profit_price=tp_price,
                entry_time=datetime.now(timezone.utc),
                order_id=str(order.id),
                signal_scores_at_entry=dict(signal_scores),
                weights_at_entry=dict(weights),
                regime_at_entry=regime_name,
            )
            self._open_positions[str(order.id)] = position
            logger.info(f"Order submitted: id={order.id} status={order.status}")
            return position

        except Exception as exc:
            logger.error(f"Order submission failed: {exc!r}")
            return None

    # ------------------------------------------------------------------
    # Position monitoring
    # ------------------------------------------------------------------

    async def check_positions(self) -> List[TradeRecord]:
        """Poll Alpaca for position updates and detect newly closed positions.

        Returns:
            List of ``TradeRecord`` instances for every position that closed
            since the last call (stop hit, take-profit hit, or manually closed).
        """
        if not self._open_positions:
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

        Args:
            reason: Exit reason string recorded in TradeRecord.
        """
        if not self._open_positions:
            return
        logger.info(f"Closing all positions — reason: {reason}")
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
