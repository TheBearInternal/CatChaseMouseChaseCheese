"""Economic event calendar with pre-event risk management.

Maintains a rolling 90-day window of high-impact macro events so the trading
engine can automatically reduce position size and raise signal thresholds in
the minutes around scheduled releases.

Supported markets
-----------------
* equity  — FOMC, Fed Press Conference, NFP, CPI, PPI, GDP Advance
* forex   — all equity events plus ECB, BOE, BOJ decisions
* crypto  — Bitcoin halving only (all others excluded)

Risk multipliers applied during the HIGH IMPACT window (30 min pre / 15 min post)
----------------------------------------------------------------------------------
* HIGH impact  → 0.20× position size, 1.50× signal threshold
* MEDIUM impact → 0.50× position size, 1.25× signal threshold

Usage::

    cal = EventCalendar()
    in_window, event = cal.is_high_impact_window()
    risk_mult = cal.get_risk_multiplier()      # 0.20, 0.50, or 1.0
    threshold_mult = cal.get_threshold_multiplier()  # 1.50, 1.25, or 1.0
    print(cal.next_event_summary())
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

from config import config
from logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Timezones
# ---------------------------------------------------------------------------

EST = ZoneInfo("America/New_York")
CET = ZoneInfo("Europe/Paris")   # ECB uses Frankfurt/CET
GMT = ZoneInfo("Europe/London")  # BOE uses London/GMT
JST = ZoneInfo("Asia/Tokyo")     # BOJ uses Tokyo/JST

# ---------------------------------------------------------------------------
# Pre/post event windows (minutes)
# ---------------------------------------------------------------------------

PRE_EVENT_MINUTES: int = 30
POST_EVENT_MINUTES: int = 15


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class EconomicEvent:
    """A single scheduled macro event.

    Attributes:
        event_time:   UTC datetime of the release.
        name:         Human-readable event name.
        impact:       ``"HIGH"`` or ``"MEDIUM"``.
        market_types: Set of market types this event is relevant for.
                      Empty set means relevant for all markets.
    """

    event_time: datetime
    name: str
    impact: str = "HIGH"
    market_types: Tuple[str, ...] = field(default_factory=tuple)

    def is_relevant(self) -> bool:
        """Return True if this event applies to the current market_type."""
        if not self.market_types:
            return True
        return config.market_type.lower() in self.market_types


# ---------------------------------------------------------------------------
# Calendar builder helpers
# ---------------------------------------------------------------------------


def _first_weekday_of_month(year: int, month: int, weekday: int) -> datetime:
    """Return the first occurrence of *weekday* (0=Mon … 6=Sun) in the month."""
    first = datetime(year, month, 1, tzinfo=EST)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset)


def _last_weekday_of_month(year: int, month: int, weekday: int) -> datetime:
    """Return the last occurrence of *weekday* in the month."""
    last_day = calendar.monthrange(year, month)[1]
    last = datetime(year, month, last_day, tzinfo=EST)
    offset = (last.weekday() - weekday) % 7
    return last - timedelta(days=offset)


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> datetime:
    """Return the *n*-th (1-indexed) occurrence of *weekday* in the month."""
    first = _first_weekday_of_month(year, month, weekday)
    return first + timedelta(weeks=n - 1)


def _to_utc(dt_local: datetime) -> datetime:
    """Convert a timezone-aware local datetime to UTC."""
    return dt_local.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# EventCalendar
# ---------------------------------------------------------------------------


class EventCalendar:
    """Rolling 90-day economic event calendar.

    Args:
        silent: When ``True``, suppress entering/exiting HIGH IMPACT window
                transition log messages.  Set this on secondary instances to
                avoid duplicate logs when multiple components hold a calendar.
    """

    def __init__(self, silent: bool = False) -> None:
        self._silent = silent
        self._events: List[EconomicEvent] = []
        self._in_window: bool = False
        self._rebuild()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_upcoming_events(self, hours_ahead: float = 4.0) -> List[EconomicEvent]:
        """Return all relevant events within the next *hours_ahead* hours."""
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)
        return [
            e for e in self._events
            if now <= e.event_time <= cutoff and e.is_relevant()
        ]

    def is_high_impact_window(
        self,
        pre: int = PRE_EVENT_MINUTES,
        post: int = POST_EVENT_MINUTES,
    ) -> Tuple[bool, Optional[EconomicEvent]]:
        """Return ``(in_window, event)`` for the nearest relevant active event.

        An event is "active" when the current time falls in the half-open
        interval ``[event_time - pre, event_time + post)``.

        Side effect: logs entering/exiting transition when ``silent=False``.
        """
        now = datetime.now(timezone.utc)
        active_event: Optional[EconomicEvent] = None

        for event in self._events:
            if not event.is_relevant():
                continue
            window_start = event.event_time - timedelta(minutes=pre)
            window_end = event.event_time + timedelta(minutes=post)
            if window_start <= now < window_end:
                active_event = event
                break

        currently_in = active_event is not None
        if not self._silent:
            if currently_in and not self._in_window:
                logger.warning(
                    f"CALENDAR | Entering HIGH IMPACT window: {active_event.name} "  # type: ignore[union-attr]
                    f"({active_event.impact}) at "  # type: ignore[union-attr]
                    f"{active_event.event_time.strftime('%H:%M UTC')}"  # type: ignore[union-attr]
                )
            elif not currently_in and self._in_window:
                logger.info("CALENDAR | Exiting HIGH IMPACT window — resuming normal sizing")

        self._in_window = currently_in
        return currently_in, active_event

    def get_risk_multiplier(self) -> float:
        """Position-size multiplier for the current moment.

        Returns:
            0.20 during HIGH impact window, 0.50 during MEDIUM, 1.0 otherwise.
        """
        in_window, event = self.is_high_impact_window()
        if not in_window or event is None:
            return 1.0
        return 0.20 if event.impact == "HIGH" else 0.50

    def get_threshold_multiplier(self) -> float:
        """Signal-threshold multiplier for the current moment.

        Returns:
            1.50 during HIGH impact window, 1.25 during MEDIUM, 1.0 otherwise.
        """
        in_window, event = self.is_high_impact_window()
        if not in_window or event is None:
            return 1.0
        return 1.50 if event.impact == "HIGH" else 1.25

    def next_event_summary(self) -> str:
        """Return a one-line summary of the next upcoming relevant event."""
        now = datetime.now(timezone.utc)
        upcoming = [
            e for e in self._events
            if e.event_time > now and e.is_relevant()
        ]
        if not upcoming:
            return "No upcoming events in calendar window"
        nxt = upcoming[0]
        delta = nxt.event_time - now
        hours = int(delta.total_seconds() // 3600)
        minutes = int((delta.total_seconds() % 3600) // 60)
        return (
            f"{nxt.name} ({nxt.impact}) in {hours}h {minutes}m "
            f"at {nxt.event_time.strftime('%Y-%m-%d %H:%M UTC')}"
        )

    # ------------------------------------------------------------------
    # Calendar construction
    # ------------------------------------------------------------------

    def _rebuild(self, lookforward_days: int = 90) -> None:
        """Populate ``_events`` with all releases in the next 90 days."""
        now = datetime.now(timezone.utc)
        horizon = now + timedelta(days=lookforward_days)

        events: List[EconomicEvent] = []
        events.extend(self._fomc_events(now, horizon))
        events.extend(self._recurring_events(now, horizon))
        events.extend(self._ecb_events(now, horizon))
        events.extend(self._boe_events(now, horizon))
        events.extend(self._boj_events(now, horizon))
        events.extend(self._crypto_events(now, horizon))

        events.sort(key=lambda e: e.event_time)
        self._events = events
        logger.debug(f"CALENDAR | Built {len(events)} events over next {lookforward_days} days")

    # ------------------------------------------------------------------
    # FOMC (equity + forex)
    # ------------------------------------------------------------------

    _FOMC_2026_DATES = [
        (2026, 1, 29), (2026, 3, 19), (2026, 5, 7),  (2026, 6, 18),
        (2026, 7, 30), (2026, 9, 17), (2026, 11, 5),  (2026, 12, 16),
    ]

    def _fomc_events(self, now: datetime, horizon: datetime) -> List[EconomicEvent]:
        events: List[EconomicEvent] = []
        for y, m, d in self._FOMC_2026_DATES:
            decision_dt = _to_utc(datetime(y, m, d, 14, 0, tzinfo=EST))
            presser_dt  = _to_utc(datetime(y, m, d, 14, 30, tzinfo=EST))
            if now <= decision_dt <= horizon:
                events.append(EconomicEvent(
                    event_time=decision_dt,
                    name="FOMC Rate Decision",
                    impact="HIGH",
                    market_types=("equity", "forex"),
                ))
            if now <= presser_dt <= horizon:
                events.append(EconomicEvent(
                    event_time=presser_dt,
                    name="Fed Chair Press Conference",
                    impact="HIGH",
                    market_types=("equity", "forex"),
                ))
        return events

    # ------------------------------------------------------------------
    # Recurring US macro (equity + forex)
    # ------------------------------------------------------------------

    def _recurring_events(self, now: datetime, horizon: datetime) -> List[EconomicEvent]:
        events: List[EconomicEvent] = []
        start_year = now.year
        end_year = horizon.year + 1

        for year in range(start_year, end_year):
            for month in range(1, 13):
                # NFP — first Friday at 08:30 EST
                nfp_dt = _to_utc(
                    _first_weekday_of_month(year, month, 4).replace(hour=8, minute=30)
                )
                if now <= nfp_dt <= horizon:
                    events.append(EconomicEvent(
                        event_time=nfp_dt,
                        name="Non-Farm Payrolls (NFP)",
                        impact="HIGH",
                        market_types=("equity", "forex"),
                    ))

                # CPI — 2nd Wednesday at 08:30 EST
                cpi_dt = _to_utc(
                    _nth_weekday_of_month(year, month, 2, 2).replace(hour=8, minute=30)
                )
                if now <= cpi_dt <= horizon:
                    events.append(EconomicEvent(
                        event_time=cpi_dt,
                        name="CPI Inflation Report",
                        impact="HIGH",
                        market_types=("equity", "forex"),
                    ))

                # PPI — day after CPI at 08:30 EST
                ppi_dt = _to_utc(
                    (_nth_weekday_of_month(year, month, 2, 2) + timedelta(days=1))
                    .replace(hour=8, minute=30)
                )
                if now <= ppi_dt <= horizon:
                    events.append(EconomicEvent(
                        event_time=ppi_dt,
                        name="PPI Producer Prices",
                        impact="MEDIUM",
                        market_types=("equity", "forex"),
                    ))

                # GDP Advance — last Thursday of Jan/Apr/Jul/Oct at 08:30 EST
                if month in (1, 4, 7, 10):
                    gdp_dt = _to_utc(
                        _last_weekday_of_month(year, month, 3).replace(hour=8, minute=30)
                    )
                    if now <= gdp_dt <= horizon:
                        events.append(EconomicEvent(
                            event_time=gdp_dt,
                            name="GDP Advance Estimate",
                            impact="HIGH",
                            market_types=("equity", "forex"),
                        ))

        return events

    # ------------------------------------------------------------------
    # ECB (forex only)
    # ------------------------------------------------------------------

    _ECB_2026_DATES = [
        (2026, 1, 30), (2026, 3, 5),  (2026, 4, 16), (2026, 6, 4),
        (2026, 7, 16), (2026, 9, 10), (2026, 10, 29), (2026, 12, 10),
    ]

    def _ecb_events(self, now: datetime, horizon: datetime) -> List[EconomicEvent]:
        events: List[EconomicEvent] = []
        for y, m, d in self._ECB_2026_DATES:
            ecb_dt = _to_utc(datetime(y, m, d, 14, 15, tzinfo=CET))
            if now <= ecb_dt <= horizon:
                events.append(EconomicEvent(
                    event_time=ecb_dt,
                    name="ECB Rate Decision",
                    impact="HIGH",
                    market_types=("forex",),
                ))
        return events

    # ------------------------------------------------------------------
    # BOE (forex only) — first Thursday of Feb/Mar/May/Jun/Aug/Sep/Nov/Dec at 12:00 GMT
    # ------------------------------------------------------------------

    _BOE_MONTHS = (2, 3, 5, 6, 8, 9, 11, 12)

    def _boe_events(self, now: datetime, horizon: datetime) -> List[EconomicEvent]:
        events: List[EconomicEvent] = []
        start_year = now.year
        end_year = horizon.year + 1

        for year in range(start_year, end_year):
            for month in self._BOE_MONTHS:
                boe_dt = _to_utc(
                    _first_weekday_of_month(year, month, 3).replace(hour=12, minute=0, tzinfo=GMT)
                )
                if now <= boe_dt <= horizon:
                    events.append(EconomicEvent(
                        event_time=boe_dt,
                        name="BOE Rate Decision",
                        impact="HIGH",
                        market_types=("forex",),
                    ))
        return events

    # ------------------------------------------------------------------
    # BOJ (forex only)
    # ------------------------------------------------------------------

    _BOJ_2026_DATES = [
        (2026, 1, 24), (2026, 3, 19), (2026, 4, 30), (2026, 6, 17),
        (2026, 7, 30), (2026, 9, 19), (2026, 10, 29), (2026, 12, 19),
    ]

    def _boj_events(self, now: datetime, horizon: datetime) -> List[EconomicEvent]:
        events: List[EconomicEvent] = []
        for y, m, d in self._BOJ_2026_DATES:
            boj_dt = _to_utc(datetime(y, m, d, 12, 0, tzinfo=JST))
            if now <= boj_dt <= horizon:
                events.append(EconomicEvent(
                    event_time=boj_dt,
                    name="BOJ Rate Decision",
                    impact="HIGH",
                    market_types=("forex",),
                ))
        return events

    # ------------------------------------------------------------------
    # Crypto
    # ------------------------------------------------------------------

    _BITCOIN_HALVING_DATES = [
        datetime(2028, 4, 20, 0, 0, tzinfo=timezone.utc),
    ]

    def _crypto_events(self, now: datetime, horizon: datetime) -> List[EconomicEvent]:
        events: List[EconomicEvent] = []
        for halving_dt in self._BITCOIN_HALVING_DATES:
            if now <= halving_dt <= horizon:
                events.append(EconomicEvent(
                    event_time=halving_dt,
                    name="Bitcoin Halving",
                    impact="HIGH",
                    market_types=("crypto",),
                ))
        return events
