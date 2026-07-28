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
from logger import get_logger, summarize_broker_error
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


def _parse_oanda_time(ts: str) -> datetime:
    """Parse an OANDA RFC3339 timestamp (nanosecond precision) safely.

    OANDA reports times like ``2026-07-28T12:34:56.123456789Z``;
    ``datetime.fromisoformat`` only accepts up to microseconds, so the
    fractional part is trimmed to 6 digits.  Falls back to now() on any
    parse failure.
    """
    try:
        ts = ts.replace("Z", "+00:00")
        if "." in ts:
            head, rest = ts.split(".", 1)
            tz_idx = max(rest.find("+"), rest.find("-"))
            frac, tz = (rest[:tz_idx], rest[tz_idx:]) if tz_idx >= 0 else (rest, "")
            ts = f"{head}.{frac[:6]}{tz}"
        return datetime.fromisoformat(ts)
    except Exception:
        return datetime.now(timezone.utc)


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
    trade_id: str = ""                 # OANDA trade ID (forex fills only)
    confidence_at_entry: float = 0.0   # ensemble confidence at decision time


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
        # Forex prices need 5 decimals; equity/crypto keep the coarser format
        prec = 5 if config.is_forex() else 4
        row = [
            record.symbol,
            record.side,
            f"{record.entry_price:.{prec}f}",
            record.quantity,
            f"{record.stop_price:.{prec}f}",
            f"{record.take_profit_price:.{prec}f}",
            record.entry_time.isoformat(),
            record.order_id,
            record.regime_at_entry,
            f"{record.exit_price:.{prec}f}",
            record.exit_time.isoformat(),
            record.exit_reason,
            f"{record.gross_pnl:.4f}",
            f"{record.net_pnl:.4f}",
            f"{record.spread_cost:.4f}",
            f"{record.duration_seconds:.1f}",
        ]
        with open(self._path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
        log_prec = 5 if config.is_forex() else 2
        logger.info(
            f"Trade logged | {record.side} {record.symbol} "
            f"entry={record.entry_price:.{log_prec}f} "
            f"exit={record.exit_price:.{log_prec}f} "
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
        self._order_in_flight: bool = False             # defers tracker syncs mid-submission
        self._positions_lock: asyncio.Lock = asyncio.Lock()
        self._tracker_version: int = 0                  # bumped on every tracker mutation
        self._spread_gate_blocked: bool = False         # log-once flag for the spread gate
        # Async callback(TradeRecord) invoked for each broker-side close the
        # sync discovers — SessionManager wires this to the IC/Kalman update
        self.trade_close_listener: Optional[Any] = None

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
                request_params={"timeout": 10},
            )
            self._initial_position_sync()

    def _fetch_oanda_open_positions(self) -> Optional[List[Dict[str, Any]]]:
        """Blocking OANDA OpenPositions request.

        Returns the raw positions list, or ``None`` on any error so callers
        can distinguish "no positions" from "fetch failed".
        """
        try:
            from oandapyV20.endpoints.positions import OpenPositions

            r = OpenPositions(config.oanda_account_id)
            response: Dict[str, Any] = self._oanda_client.request(r)
            if not isinstance(response, dict):
                return None
            return response.get("positions", [])
        except Exception as exc:
            logger.warning(f"OANDA position fetch failed: {summarize_broker_error(exc)}")
            return None

    def _build_tracker_from_snapshot(
        self, positions: List[Dict[str, Any]]
    ) -> Dict[str, Position]:
        """Convert a raw OANDA positions snapshot into tracker entries (no I/O)."""
        tracker: Dict[str, Position] = {}
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
            tracker[synthetic_id] = Position(
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
        return tracker

    def _initial_position_sync(self) -> None:
        """Populate the tracker from OANDA at startup (runs before the event loop)."""
        snapshot = self._fetch_oanda_open_positions()
        if snapshot is None:
            logger.warning("OANDA startup position sync failed — tracker starts empty")
            return
        self._open_positions = self._build_tracker_from_snapshot(snapshot)
        logger.info(
            f"Position sync | found {len(self._open_positions)} open "
            f"position(s) from OANDA"
        )

    _CLOSE_REASON_MAP: Dict[str, str] = {
        "TAKE_PROFIT_ORDER": "take_profit",
        "STOP_LOSS_ORDER": "stop_loss",
        "MARKET_ORDER": "market_close",
        "TRAILING_STOP_LOSS_ORDER": "trailing_stop",
        "MARKET_ORDER_POSITION_CLOSEOUT": "position_closeout",
    }

    # ORDER_FILL reasons that represent a broker-side close of an open trade
    _OFFLINE_CLOSE_REASONS: frozenset = frozenset({
        "TAKE_PROFIT_ORDER",
        "STOP_LOSS_ORDER",
        "TRAILING_STOP_LOSS_ORDER",
        "MARKET_ORDER_POSITION_CLOSEOUT",
    })

    # ------------------------------------------------------------------
    # Offline-close reconciliation state (persisted via SlowBuffer)
    # ------------------------------------------------------------------

    def _record_trade_metadata(self, position: Position) -> None:
        """Persist the entry snapshot for offline-close reconciliation.

        This is a lookup table keyed by OANDA trade ID — never a source of
        truth about what is open (OANDA remains authoritative).
        """
        if not position.trade_id:
            return
        try:
            meta = self._dm.slow_buffer.data.setdefault("open_trade_metadata", {})
            meta[str(position.trade_id)] = {
                "symbol": position.symbol,
                "side": position.side,
                "units": position.quantity,
                "entry_price": position.entry_price,
                "entry_time": position.entry_time.isoformat(),
                "signal_scores": dict(position.signal_scores_at_entry),
                "regime": position.regime_at_entry,
                "confidence": position.confidence_at_entry,
            }
            self._dm.slow_buffer.save(config.session_state_path)
        except Exception as exc:
            logger.warning(f"Failed to persist trade metadata: {exc!r}")

    def _clear_trade_metadata(self, trade_id: str) -> None:
        """Drop a reconciled trade's entry snapshot and persist the change."""
        if not trade_id:
            return
        try:
            meta = self._dm.slow_buffer.data.get("open_trade_metadata") or {}
            if str(trade_id) in meta:
                meta.pop(str(trade_id), None)
                self._dm.slow_buffer.save(config.session_state_path)
        except Exception as exc:
            logger.warning(f"Failed to clear trade metadata {trade_id}: {exc!r}")

    def _note_last_transaction_id(self, last_id: Optional[str]) -> None:
        """Track OANDA's most recently seen transaction ID (monotonic)."""
        if not last_id:
            return
        data = self._dm.slow_buffer.data
        try:
            if int(last_id) > int(data.get("last_transaction_id") or 0):
                data["last_transaction_id"] = str(last_id)
        except (TypeError, ValueError):
            data["last_transaction_id"] = str(last_id)

    def _fetch_transactions_since(
        self, since_id: str
    ) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        """Blocking fetch of all account transactions after *since_id*.

        Returns:
            ``(transactions, lastTransactionID)`` or ``(None, None)`` on error.
        """
        try:
            from oandapyV20.endpoints.transactions import TransactionsSinceID

            r = TransactionsSinceID(
                config.oanda_account_id, params={"id": str(since_id)}
            )
            response: Dict[str, Any] = self._oanda_client.request(r)
            if not isinstance(response, dict):
                return None, None
            return response.get("transactions", []), response.get("lastTransactionID")
        except Exception as exc:
            logger.warning(
                f"Transaction fetch since {since_id} failed: "
                f"{summarize_broker_error(exc)}"
            )
            return None, None

    async def reconcile_offline_closes(self) -> List[TradeRecord]:
        """Detect and record trades the broker closed while the engine was down.

        Scans the OANDA transaction stream since the persisted
        ``last_transaction_id`` for ORDER_FILL transactions whose reason is a
        broker-side close (TP / SL / trailing stop / margin closeout), matches
        ``tradesClosed[].tradeID`` against the persisted entry snapshots, and
        emits full ``TradeRecord``s through the same trade-close path used for
        live closes so IC and Kalman update normally.  Closes with no stored
        entry snapshot are logged to the trade CSV but not scored.

        Returns:
            List of reconciled TradeRecords (empty when nothing was missed).
        """
        if not config.is_forex() or self._oanda_client is None:
            return []
        data = self._dm.slow_buffer.data
        last_tx = data.get("last_transaction_id")
        if not last_tx:
            return []
        meta_map: Dict[str, Any] = data.get("open_trade_metadata") or {}

        loop = asyncio.get_running_loop()
        txs, new_last = await loop.run_in_executor(
            None, lambda: self._fetch_transactions_since(last_tx)
        )
        if txs is None:
            return []

        records: List[TradeRecord] = []
        summary_lines: List[str] = []
        for tx in txs:
            if tx.get("type") != "ORDER_FILL":
                continue
            raw_reason = str(tx.get("reason", ""))
            if raw_reason not in self._OFFLINE_CLOSE_REASONS:
                continue
            exit_time = _parse_oanda_time(str(tx.get("time", "")))
            for reduced in (tx.get("tradesClosed") or []):
                trade_id = str(reduced.get("tradeID", ""))
                exit_price = float(reduced.get("price") or tx.get("price") or 0.0)
                realized = float(reduced.get("realizedPL", 0.0))
                mapped_reason = self._CLOSE_REASON_MAP.get(
                    raw_reason, raw_reason.lower()
                )
                snap = meta_map.get(trade_id)

                if snap:
                    try:
                        entry_time = datetime.fromisoformat(snap["entry_time"])
                    except Exception:
                        entry_time = exit_time
                    side = str(snap.get("side", "LONG"))
                    qty = int(snap.get("units", 0)) or 1
                    entry_price = float(snap.get("entry_price", 0.0))
                    gross = (
                        (exit_price - entry_price) if side == "LONG"
                        else (entry_price - exit_price)
                    ) * qty
                    record = TradeRecord(
                        symbol=str(snap.get("symbol") or tx.get("instrument", "UNKNOWN")),
                        side=side,
                        entry_price=entry_price,
                        quantity=qty,
                        stop_price=0.0,
                        take_profit_price=0.0,
                        entry_time=entry_time,
                        order_id=f"offline_{trade_id}",
                        signal_scores_at_entry=dict(snap.get("signal_scores") or {}),
                        weights_at_entry={},
                        regime_at_entry=str(snap.get("regime", "unknown")),
                        exit_price=exit_price,
                        exit_time=exit_time,
                        exit_reason=mapped_reason,
                        gross_pnl=gross,
                        net_pnl=realized,
                        spread_cost=gross - realized,
                        duration_seconds=(exit_time - entry_time).total_seconds(),
                    )
                    self._logger.log_trade(record)
                    if self.trade_close_listener is not None:
                        try:
                            await self.trade_close_listener(record)
                        except Exception as exc:
                            logger.error(f"trade_close_listener error: {exc!r}")
                    meta_map.pop(trade_id, None)
                else:
                    logger.warning(
                        f"Offline close {trade_id} has no entry snapshot "
                        "— logged but not scored"
                    )
                    closed_units = float(reduced.get("units", 0) or 0)
                    side = "LONG" if closed_units < 0 else "SHORT"
                    record = TradeRecord(
                        symbol=str(tx.get("instrument", "UNKNOWN")),
                        side=side,
                        entry_price=0.0,
                        quantity=int(abs(closed_units)) or 1,
                        stop_price=0.0,
                        take_profit_price=0.0,
                        entry_time=exit_time,
                        order_id=f"offline_{trade_id}",
                        signal_scores_at_entry={},
                        weights_at_entry={},
                        regime_at_entry="unknown",
                        exit_price=exit_price,
                        exit_time=exit_time,
                        exit_reason=mapped_reason,
                        gross_pnl=realized,
                        net_pnl=realized,
                        spread_cost=0.0,
                        duration_seconds=0.0,
                    )
                    self._logger.log_trade(record)

                records.append(record)
                summary_lines.append(
                    f"  {record.symbol} {record.side:<5} {record.quantity}u | "
                    f"entry={record.entry_price:.5f} exit={record.exit_price:.5f} | "
                    f"pnl={record.net_pnl:+.2f} | {raw_reason} | "
                    f"{exit_time.strftime('%Y-%m-%d %H:%M:%S')}"
                )

        self._note_last_transaction_id(new_last)
        try:
            self._dm.slow_buffer.save(config.session_state_path)
        except Exception as exc:
            logger.warning(f"Failed to persist reconciliation state: {exc!r}")

        if records:
            net = sum(r.net_pnl for r in records)
            logger.info(
                f"Offline reconciliation | {len(records)} trade(s) closed while "
                "engine was down:\n"
                + "\n".join(summary_lines)
                + f"\nNet offline P&L: {net:+.2f} over {len(records)} trade(s)"
            )
        return records

    def _fetch_closing_fill(
        self, trade_id: str, from_tx_id: str
    ) -> Optional[Dict[str, Any]]:
        """Blocking lookup of the ORDER_FILL transaction that closed *trade_id*.

        Scans the account transaction stream since the entry fill transaction
        for a fill whose ``tradesClosed`` references the trade, giving the
        broker's actual exit price, realized P&L, and close reason.

        Returns:
            Dict with ``price``, ``realized_pl``, ``reason``, ``time`` keys,
            or ``None`` when the closing fill cannot be found.
        """
        try:
            from oandapyV20.endpoints.transactions import TransactionsSinceID

            r = TransactionsSinceID(
                config.oanda_account_id, params={"id": str(from_tx_id)}
            )
            response: Dict[str, Any] = self._oanda_client.request(r)
            for tx in response.get("transactions", []):
                if tx.get("type") != "ORDER_FILL":
                    continue
                for reduced in (tx.get("tradesClosed") or []):
                    if str(reduced.get("tradeID")) == str(trade_id):
                        return {
                            "price": float(
                                reduced.get("price") or tx.get("price") or 0.0
                            ),
                            "realized_pl": float(reduced.get("realizedPL", 0.0)),
                            "reason": str(tx.get("reason", "UNKNOWN")),
                            "time": str(tx.get("time", "")),
                            "last_tx": response.get("lastTransactionID"),
                        }
            return None
        except Exception as exc:
            logger.warning(
                f"Closing-fill lookup for trade {trade_id} failed: "
                f"{summarize_broker_error(exc)}"
            )
            return None

    def _build_close_record(
        self, pos: Position, fill: Optional[Dict[str, Any]]
    ) -> TradeRecord:
        """Build a TradeRecord for a broker-side close.

        Uses the broker-reported exit price and realized P&L when the closing
        fill was found; falls back to the last local price otherwise.
        """
        if fill is not None:
            exit_price = fill["price"]
            exit_time = _parse_oanda_time(fill["time"])
            reason = self._CLOSE_REASON_MAP.get(
                fill["reason"], fill["reason"].lower()
            )
        else:
            exit_price = self._dm.fast_primary.latest_price() or pos.entry_price
            exit_time = datetime.now(timezone.utc)
            reason = "broker_close"

        if pos.side == "LONG":
            gross = (exit_price - pos.entry_price) * pos.quantity
        else:
            gross = (pos.entry_price - exit_price) * pos.quantity
        net = (
            fill["realized_pl"] if fill is not None
            else self._spread.net_pnl(gross, pos.quantity)
        )

        return TradeRecord(
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
            spread_cost=gross - net,
            duration_seconds=(exit_time - pos.entry_time).total_seconds(),
        )

    async def sync_positions_from_oanda(self) -> Optional[List[TradeRecord]]:
        """Reconcile the tracker with OANDA's live positions, event-loop safe.

        The blocking HTTP fetch runs in a worker thread; tracker mutations are
        applied on the event loop under ``_positions_lock`` and touch only
        entries identified from the snapshot, so a concurrently tracked fill
        is never wiped.  Deferred while an order submission is in flight and
        discarded if the tracker changed during the fetch.

        Tracked positions that OANDA no longer reports open are converted to
        ``TradeRecord``s using the broker's closing ORDER_FILL transaction
        (actual exit price, realized P&L, close reason), written to the trade
        log, and passed to ``trade_close_listener`` so the session can feed
        the IC / Kalman learning loop.

        Returns:
            The list of broker-close records when the sync applied (possibly
            empty), or ``None`` when the sync was skipped or failed.
        """
        if self._order_in_flight:
            logger.debug("Position sync deferred — order submission in flight")
            return None
        version_before = self._tracker_version
        loop = asyncio.get_running_loop()
        snapshot = await loop.run_in_executor(None, self._fetch_oanda_open_positions)
        if snapshot is None:
            return None
        if self._order_in_flight or self._tracker_version != version_before:
            logger.debug("Position sync discarded — tracker changed during fetch")
            return None

        open_instruments = {
            pos.get("instrument", "")
            for pos in snapshot
            if abs(float(pos.get("long", {}).get("units", 0))) >= 1
            or abs(float(pos.get("short", {}).get("units", 0))) >= 1
        }

        # Entries OANDA no longer reports open were closed broker-side
        closed_entries = [
            (oid, pos) for oid, pos in self._open_positions.items()
            if pos.symbol not in open_instruments
        ]

        closed_records: List[TradeRecord] = []
        for _oid, pos in closed_entries:
            if not pos.trade_id:
                # Synthetic stub from a previous sync — no entry metadata to
                # score against; it is dropped from the tracker below
                continue
            fill = await loop.run_in_executor(
                None, lambda p=pos: self._fetch_closing_fill(p.trade_id, p.order_id)
            )
            record = self._build_close_record(pos, fill)
            self._logger.log_trade(record)
            logger.info(
                f"Trade closed | {record.symbol} {record.side} "
                f"entry={record.entry_price:.5f} exit={record.exit_price:.5f} "
                f"pnl={record.net_pnl:+.2f} reason={record.exit_reason}"
            )
            closed_records.append(record)
            if self.trade_close_listener is not None:
                try:
                    await self.trade_close_listener(record)
                except Exception as exc:
                    logger.error(f"trade_close_listener error: {exc!r}")
            if fill is not None:
                self._note_last_transaction_id(fill.get("last_tx"))
            self._clear_trade_metadata(pos.trade_id)

        async with self._positions_lock:
            for oid, _pos in closed_entries:
                self._open_positions.pop(oid, None)
            if closed_entries:
                self._tracker_version += 1
            # Import positions OANDA holds that we are not tracking — unless
            # an order is mid-flight, in which case they arrive next cycle
            if not self._order_in_flight:
                tracked_instruments = {
                    p.symbol for p in self._open_positions.values()
                }
                missing = [
                    pos for pos in snapshot
                    if pos.get("instrument", "")
                    in (open_instruments - tracked_instruments)
                ]
                if missing:
                    imported = self._build_tracker_from_snapshot(missing)
                    self._open_positions.update(imported)
                    self._tracker_version += 1
                    logger.info(
                        f"Position sync | imported {len(imported)} untracked "
                        f"position(s) from OANDA"
                    )
        return closed_records

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
            logger.warning(
                f"OANDA position count check failed: {summarize_broker_error(exc)}"
            )
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
        stop_distance: float,
        tp_distance: float,
        atr_value: float,
        signal_scores: Dict[str, float],
        weights: Dict[str, float],
        regime_name: str,
        confidence: float = 0.0,
    ) -> Optional[Position]:
        """Submit a market bracket order to OANDA and return a tracked Position.

        Stop-loss and take-profit are sent as *distances*, not absolute prices,
        so OANDA anchors both levels to the actual fill price.  This keeps the
        realized risk/reward ratio exactly ``config.risk_reward_ratio`` no
        matter how much the market moved between quote and fill.

        Args:
            side:          "LONG" or "SHORT".
            qty:           OANDA units (base currency, e.g. 1000 = 1 micro-lot).
            entry_price:   Mid price at submission time (logging / fallback).
            stop_distance: Positive stop-loss distance from the fill price.
            tp_distance:   Positive take-profit distance from the fill price.
            atr_value:     Raw ATR used for stop/tp calculation (logged).
            signal_scores: Signal scores at decision time.
            weights:       Ensemble weights at decision time.
            regime_name:   RegimeState name string.

        Returns:
            ``Position`` on success, ``None`` on any error.
        """
        from oandapyV20.endpoints import orders as oanda_orders

        instrument = config.primary_symbol

        # OANDA rejects non-positive distances outright
        if stop_distance <= 0.0 or tp_distance <= 0.0:
            logger.error(
                f"Refusing order: non-positive bracket distance | "
                f"stop={stop_distance:.5f} tp={tp_distance:.5f}"
            )
            return None

        units = str(qty) if side == "LONG" else str(-qty)
        order_data = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": units,
                "takeProfitOnFill": {
                    "distance": f"{tp_distance:.5f}",
                },
                "stopLossOnFill": {
                    "distance": f"{stop_distance:.5f}",
                    "timeInForce": "GTC",
                },
            }
        }

        logger.info(
            f"Order | {side} {qty} {instrument} @ ~{entry_price:.5f} | "
            f"stop_dist={stop_distance:.5f} tp_dist={tp_distance:.5f} "
            f"| ATR={atr_value:.5f}"
        )

        reconcile_needed = False
        result: Optional[Position] = None
        self._order_in_flight = True
        try:
            loop = asyncio.get_running_loop()
            r = oanda_orders.OrderCreate(config.oanda_account_id, data=order_data)
            response: Dict[str, Any] = await loop.run_in_executor(
                None, lambda: self._oanda_client.request(r)
            )

            # Cooldown starts on every submission attempt; cleared only after
            # a confirmed fill is tracked
            self._last_order_attempt_ts = time.time()

            if not response or not isinstance(response, dict):
                # The request may or may not have reached OANDA — reconcile
                logger.error(f"OANDA returned invalid response: {response!r}")
                reconcile_needed = True

            elif response.get("orderFillTransaction") is None:
                # Order was rejected or cancelled (FOK) — no position opened
                order_id = (
                    response.get("orderCreateTransaction", {}).get("id")
                    or f"oanda_{int(time.time())}"
                )
                logger.warning(
                    f"Order id={order_id} did not result in immediate fill "
                    "— not counting as open position"
                )

            else:
                fill_tx = response["orderFillTransaction"]
                order_id = fill_tx.get("id") or f"oanda_{int(time.time())}"
                trade_id = (fill_tx.get("tradeOpened") or {}).get("tradeID")

                if trade_id is None:
                    # Fill netted against existing exposure instead of opening
                    # a trade — the tracker no longer matches reality
                    logger.warning(
                        f"Order {order_id} filled but opened no new trade "
                        "(netted against existing exposure) — reconciling"
                    )
                    reconcile_needed = True
                else:
                    # Record the broker's actual fill price and derive the
                    # bracket levels OANDA anchored to it
                    fill_price = float(fill_tx.get("price") or entry_price)
                    if side == "LONG":
                        stop_price = fill_price - stop_distance
                        tp_price = fill_price + tp_distance
                    else:
                        stop_price = fill_price + stop_distance
                        tp_price = fill_price - tp_distance

                    # A fill is live money: track it unconditionally
                    position = Position(
                        symbol=instrument,
                        side=side,
                        entry_price=fill_price,
                        quantity=qty,
                        stop_price=round(stop_price, 5),
                        take_profit_price=round(tp_price, 5),
                        entry_time=datetime.now(timezone.utc),
                        order_id=str(order_id),
                        signal_scores_at_entry=dict(signal_scores),
                        weights_at_entry=dict(weights),
                        regime_at_entry=regime_name,
                        trade_id=str(trade_id),
                        confidence_at_entry=confidence,
                    )
                    self._open_positions[str(order_id)] = position
                    self._tracker_version += 1
                    logger.info(
                        f"OANDA order filled: id={order_id} trade={trade_id} "
                        f"fill={fill_price:.5f} stop={stop_price:.5f} "
                        f"tp={tp_price:.5f} regime={regime_name}"
                    )
                    self._last_order_attempt_ts = 0.0
                    self._last_order_time[instrument] = time.time()
                    self._note_last_transaction_id(response.get("lastTransactionID"))
                    self._record_trade_metadata(position)

                    # Confirm OANDA attached the on-fill stop/TP to the trade;
                    # close immediately rather than leave a naked position live
                    if await self._verify_trade_protection(trade_id):
                        result = position
                    else:
                        logger.warning(
                            f"Trade {trade_id} has no stop/TP attached — "
                            "closing immediately for safety"
                        )
                        await self.close_position_by_symbol(
                            instrument, reason="unprotected"
                        )
                        self._last_order_attempt_ts = time.time()

        except Exception as exc:
            # The order may have reached OANDA before the failure — never
            # assume no fill happened; reconcile with OANDA instead
            logger.error(
                f"OANDA order submission failed: {summarize_broker_error(exc)}"
            )
            self._last_order_attempt_ts = time.time()
            reconcile_needed = True
        finally:
            self._order_in_flight = False

        if reconcile_needed:
            await self.sync_positions_from_oanda()
        return result

    async def _verify_trade_protection(self, trade_id: str) -> bool:
        """Confirm OANDA attached stop-loss and take-profit orders to a trade.

        Queries GET /trades/{id} in a worker thread and checks the
        ``takeProfitOrder`` and ``stopLossOrder`` fields of the trade.  Fails
        OPEN on API errors: on-fill dependents are validated atomically by
        OANDA at order acceptance, so a failed *check* is not evidence of a
        naked trade and must not trigger closing a protected position.

        Args:
            trade_id: OANDA trade ID from orderFillTransaction.tradeOpened.

        Returns:
            ``False`` only on positive evidence that protection is missing.
        """
        from oandapyV20.endpoints import trades as oanda_trades

        try:
            loop = asyncio.get_running_loop()
            r = oanda_trades.TradeDetails(
                config.oanda_account_id, tradeID=str(trade_id)
            )
            response: Dict[str, Any] = await loop.run_in_executor(
                None, lambda: self._oanda_client.request(r)
            )
            trade = (response or {}).get("trade", {})
            if trade.get("state") == "CLOSED":
                # Trade already closed (e.g. stop/TP triggered instantly)
                return True
            has_tp = bool(trade.get("takeProfitOrder"))
            has_sl = bool(trade.get("stopLossOrder"))
            if has_tp and has_sl:
                return True
            logger.warning(
                f"Trade {trade_id} protection check: has_tp={has_tp} has_sl={has_sl}"
            )
            return False
        except Exception as exc:
            logger.warning(
                f"Trade protection check failed for {trade_id}: "
                f"{summarize_broker_error(exc)} "
                "— assuming protected (on-fill dependents are atomic)"
            )
            return True

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
        confidence: float = 0.0,
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

        # --- Spread entry gate (forex) ------------------------------------
        if config.is_forex():
            spread = (
                max(0.0, ask - bid) if (bid is not None and ask is not None) else 0.0
            )
            if spread > config.max_entry_spread:
                if not self._spread_gate_blocked:
                    logger.info(
                        f"Entry blocked | spread={spread:.5f} exceeds "
                        f"max={config.max_entry_spread:.5f}"
                    )
                    self._spread_gate_blocked = True
                return None
            if self._spread_gate_blocked:
                logger.info(
                    f"Entry unblocked | spread={spread:.5f} back below "
                    f"max={config.max_entry_spread:.5f}"
                )
                self._spread_gate_blocked = False

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
                stop_distance=round(stop_distance, 5),
                tp_distance=round(take_profit_distance, 5),
                atr_value=atr_value,
                signal_scores=signal_scores,
                weights=weights,
                regime_name=regime_name,
                confidence=confidence,
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
                confidence_at_entry=confidence,
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
            for oid, pos in list(self._open_positions.items()):
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
                    self._clear_trade_metadata(pos.trade_id)
                except Exception as exc:
                    exc_str = str(exc)
                    if "CLOSEOUT_POSITION_DOESNT_EXIST" in exc_str:
                        logger.warning(
                            f"Position {oid} not found on OANDA — removing from tracker"
                        )
                        closed.append(oid)
                    else:
                        logger.error(
                            f"OANDA close position {oid} failed: "
                            f"{summarize_broker_error(exc)}"
                        )
            async with self._positions_lock:
                for oid in closed:
                    self._open_positions.pop(oid, None)
                if closed:
                    self._tracker_version += 1
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

    async def close_position_by_symbol(self, symbol: str, reason: str) -> None:
        """Close the OANDA position for *symbol* and remove it from the tracker.

        Uses OANDA's PositionClose endpoint for the correct direction.  If the
        position is already gone (CLOSEOUT_POSITION_DOESNT_EXIST) the tracker
        entry is removed silently.

        Args:
            symbol: OANDA instrument string, e.g. ``"EUR_USD"``.
            reason: Exit reason string for the TradeRecord log.
        """
        from oandapyV20.endpoints import positions as oanda_positions

        pos_to_close: Optional[Position] = None
        pos_id: Optional[str] = None
        for oid, pos in self._open_positions.items():
            if pos.symbol == symbol:
                pos_to_close = pos
                pos_id = oid
                break

        if pos_to_close is None:
            return

        data = {
            "longUnits": "ALL" if pos_to_close.side == "LONG" else "NONE",
            "shortUnits": "ALL" if pos_to_close.side == "SHORT" else "NONE",
        }

        try:
            r = oanda_positions.PositionClose(
                config.oanda_account_id,
                instrument=symbol,
                data=data,
            )
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: self._oanda_client.request(r))

            price = self._dm.fast_primary.latest_price() or pos_to_close.entry_price
            now = datetime.now(timezone.utc)
            gross = (
                (price - pos_to_close.entry_price) * pos_to_close.quantity
                if pos_to_close.side == "LONG"
                else (pos_to_close.entry_price - price) * pos_to_close.quantity
            )
            net = self._spread.net_pnl(gross, pos_to_close.quantity)
            self._logger.log_trade(TradeRecord(
                symbol=pos_to_close.symbol,
                side=pos_to_close.side,
                entry_price=pos_to_close.entry_price,
                quantity=pos_to_close.quantity,
                stop_price=pos_to_close.stop_price,
                take_profit_price=pos_to_close.take_profit_price,
                entry_time=pos_to_close.entry_time,
                order_id=pos_to_close.order_id,
                signal_scores_at_entry=pos_to_close.signal_scores_at_entry,
                weights_at_entry=pos_to_close.weights_at_entry,
                regime_at_entry=pos_to_close.regime_at_entry,
                exit_price=price,
                exit_time=now,
                exit_reason=reason,
                gross_pnl=gross,
                net_pnl=net,
                spread_cost=gross - net,
                duration_seconds=(now - pos_to_close.entry_time).total_seconds(),
            ))
            async with self._positions_lock:
                self._open_positions.pop(pos_id, None)
                self._tracker_version += 1
            self._clear_trade_metadata(pos_to_close.trade_id)
            logger.info(
                f"Position closed | {symbol} {pos_to_close.side} "
                f"reason={reason} net={net:+.4f}"
            )

        except Exception as exc:
            if "CLOSEOUT_POSITION_DOESNT_EXIST" in str(exc):
                logger.warning(
                    f"Position {pos_id} ({symbol}) not found on OANDA — clearing tracker"
                )
                async with self._positions_lock:
                    self._open_positions.pop(pos_id, None)
                    self._tracker_version += 1
            else:
                # Close may have failed transiently while the position is still
                # live on OANDA — keep it tracked so the FIFO guard holds, set
                # the cooldown, and let the next attempt or monitor retry
                logger.error(
                    f"close_position_by_symbol({symbol}): "
                    f"{summarize_broker_error(exc)} — keeping "
                    "position tracked for retry; cooldown set"
                )
                self._last_order_attempt_ts = time.time()

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
