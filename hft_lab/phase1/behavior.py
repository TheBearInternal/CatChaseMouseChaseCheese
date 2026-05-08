"""Behavioral randomization utilities for hft_lab.

Produces timing and sizing variations that mimic organic human trading patterns
rather than mechanical automation.  This is a **pure utility library** — importing
it has no side effects.  All effects occur only when functions or methods are
explicitly called.

Public surface
--------------
gaussian_jitter        : Normal-distribution delay with non-negative clamp.
human_pause            : Awaitable log-normal reaction-time pause.
market_open_delay      : Awaitable session-start gate drawn from BehaviorProfile.
TokenBucket            : Async token-bucket rate limiter.
default_rate_limiter   : Pre-built TokenBucket(rate=3.0, capacity=10.0).
randomize_quantity     : Uniform-jitter order size randomization.
BehaviorProfile        : Per-session deterministic behavioral parameter set.
PacedAPIClient         : Wraps any async callable with jitter + rate limiting.
"""
from __future__ import annotations

import asyncio
import math
import random
import time
from typing import Any, Callable, Optional

import numpy as np

from logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants — all configurable values declared here
# ---------------------------------------------------------------------------

# human_pause distribution parameters
HUMAN_PAUSE_MEAN_MS: float = 250.0
HUMAN_PAUSE_LOGNORMAL_SIGMA: float = 0.6   # produces occasional pauses up to ~2 s
HUMAN_PAUSE_MAX_MS: float = 2000.0

# Trading session length in minutes (9:30 AM – 4:00 PM EST)
TRADING_MINUTES: int = 390

# BehaviorProfile attribute ranges (all derived deterministically from seed)
SESSION_START_OFFSET_MIN_S: float = 0.0
SESSION_START_OFFSET_MAX_S: float = 180.0
POLLING_JITTER_SIGMA_MIN_MS: float = 50.0
POLLING_JITTER_SIGMA_MAX_MS: float = 300.0
ORDER_SIZE_VARIANCE_MIN: float = 0.08
ORDER_SIZE_VARIANCE_MAX: float = 0.20

# Activity rhythm shape: Gaussian peaks and dip expressed in minutes from open
OPEN_PEAK_CENTER_MIN: float = 15.0     # ~9:45 AM
OPEN_PEAK_SIGMA_MIN: float = 20.0
CLOSE_PEAK_CENTER_MIN: float = 375.0   # ~3:45 PM
CLOSE_PEAK_SIGMA_MIN: float = 20.0
MIDDAY_DIP_CENTER_MIN: float = 180.0   # ~12:30 PM
MIDDAY_DIP_SIGMA_MIN: float = 30.0
MIDDAY_DIP_DEPTH: float = 0.5
BACKGROUND_ACTIVITY_WEIGHT: float = 0.3
RHYTHM_FLOOR: float = 0.05             # minimum weight so bot never fully idles

# Default TokenBucket parameters
DEFAULT_RATE: float = 3.0      # tokens per second
DEFAULT_CAPACITY: float = 10.0 # maximum burst tokens

# PacedAPIClient fallback jitter when no profile is provided
PACED_CLIENT_DEFAULT_JITTER_SIGMA_MS: float = 100.0


# ---------------------------------------------------------------------------
# Timing functions
# ---------------------------------------------------------------------------


def gaussian_jitter(base_delay_ms: float, sigma_ms: float) -> float:
    """Return a delay sampled from a normal distribution, clamped to non-negative.

    Designed for per-call API polling randomization.  The returned value is
    always ``>= 0`` so it is safe to pass directly to ``asyncio.sleep`` after
    dividing by 1000.

    Args:
        base_delay_ms: Centre of the normal distribution in milliseconds.
        sigma_ms:      Standard deviation in milliseconds.

    Returns:
        Non-negative delay in milliseconds.
    """
    sample = random.gauss(base_delay_ms, sigma_ms)
    return max(0.0, sample)


async def human_pause() -> None:
    """Sleep for a log-normal duration approximating human reaction time.

    The distribution is centred near 250 ms with a long right tail that
    occasionally produces pauses up to 2 seconds, mirroring empirical data on
    human decision latency.  The coroutine is non-blocking — it yields control
    back to the event loop during the wait.
    """
    mu = math.log(HUMAN_PAUSE_MEAN_MS)
    sample_ms = random.lognormvariate(mu, HUMAN_PAUSE_LOGNORMAL_SIGMA)
    clamped_ms = min(sample_ms, HUMAN_PAUSE_MAX_MS)
    await asyncio.sleep(clamped_ms / 1000.0)


async def market_open_delay(profile: "BehaviorProfile") -> None:
    """Block until the session's randomized pre-market gate has elapsed.

    Should be awaited once at the start of each session, before any market
    data is consumed or orders considered.  The delay originates from the
    profile's ``session_start_offset`` so it is reproducible for a given seed.

    Args:
        profile: BehaviorProfile for the current session.
    """
    offset_s = profile.get_connection_delay()
    logger.info(f"Market open delay: waiting {offset_s:.2f}s before first action")
    await asyncio.sleep(offset_s)


# ---------------------------------------------------------------------------
# Token Bucket rate limiter
# ---------------------------------------------------------------------------


class TokenBucket:
    """Async token-bucket rate limiter.

    Tokens are replenished continuously at ``rate`` tokens per second up to a
    maximum ``capacity``.  Callers that request more tokens than are currently
    available block asynchronously until replenishment satisfies the request.

    Usage::

        bucket = TokenBucket(rate=5.0, capacity=20.0)
        await bucket.consume()          # consume 1 token
        await bucket.consume(tokens=3) # consume 3 tokens
    """

    def __init__(self, rate: float, capacity: float) -> None:
        """Initialise the bucket.

        Args:
            rate:     Token replenishment rate in tokens per second.
            capacity: Maximum number of tokens the bucket can hold (burst limit).
        """
        self._rate: float = rate
        self._capacity: float = capacity
        self._tokens: float = capacity
        self._last_refill_ts: float = time.monotonic()

    def _refill(self) -> None:
        """Credit elapsed-time tokens to the bucket without exceeding capacity."""
        now = time.monotonic()
        elapsed = now - self._last_refill_ts
        gained = elapsed * self._rate
        if gained > 0.0:
            self._tokens = min(self._capacity, self._tokens + gained)
            logger.debug(
                f"TokenBucket refilled +{gained:.4f} → {self._tokens:.4f} tokens"
            )
            self._last_refill_ts = now

    async def consume(self, tokens: float = 1.0) -> None:
        """Block asynchronously until *tokens* are available, then consume them.

        Args:
            tokens: Number of tokens to consume.  Defaults to ``1.0``.
        """
        while True:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                logger.debug(
                    f"TokenBucket consumed {tokens} → {self._tokens:.4f} remaining"
                )
                return
            deficit = tokens - self._tokens
            wait_s = deficit / self._rate
            logger.debug(
                f"TokenBucket: deficit={deficit:.4f}, sleeping {wait_s:.3f}s"
            )
            await asyncio.sleep(wait_s)


# Module-level default rate limiter shared across components that do not
# require a dedicated bucket.
default_rate_limiter: TokenBucket = TokenBucket(
    rate=DEFAULT_RATE, capacity=DEFAULT_CAPACITY
)


# ---------------------------------------------------------------------------
# Order size randomization
# ---------------------------------------------------------------------------


def randomize_quantity(base_qty: int, variance_pct: float = 0.15) -> int:
    """Return an order quantity jittered within ± *variance_pct* of *base_qty*.

    Draws from a uniform distribution so that no two consecutive orders are
    mechanically identical.  Always returns at least 1 share.

    Args:
        base_qty:     Target share count before randomization.
        variance_pct: Fractional half-width of the uniform distribution.
                      Default ``0.15`` produces ±15 % variation.

    Returns:
        Randomized integer share count, minimum 1.
    """
    delta = base_qty * variance_pct
    jitter = random.uniform(-delta, delta)
    return max(1, round(base_qty + jitter))


# ---------------------------------------------------------------------------
# BehaviorProfile
# ---------------------------------------------------------------------------


class BehaviorProfile:
    """Per-session behavioral parameter set derived deterministically from a seed.

    Encapsulates all randomized timing and sizing decisions for one trading
    session.  Seeding with the same integer reproduces identical parameters,
    enabling reproducible debugging while still producing session-to-session
    variation when the seed is changed.

    Attributes
    ----------
    session_start_offset : float
        Seconds to wait after market open before the first action (0 – 180 s).
    activity_rhythm : list[float]
        390 probability weights, one per trading minute (9:30 – 4:00 EST).
        Values range from ``RHYTHM_FLOOR`` to ``1.0``; the maximum is always
        normalised to 1.0.  Elevated at open and close, depressed at midday.
    polling_jitter_sigma : float
        Per-session standard deviation (ms) for API polling delay randomization.
    order_size_variance : float
        Per-session fractional variance for order quantity randomization.
    """

    def __init__(self, seed: int) -> None:
        """Initialise the profile from an integer seed.

        Args:
            seed: Integer controlling all session-level randomization.
                  Sourced from ``Config.behavior_seed``.
        """
        self._seed: int = seed
        self._rng: np.random.Generator = np.random.default_rng(seed)

        self.session_start_offset: float = float(
            self._rng.uniform(SESSION_START_OFFSET_MIN_S, SESSION_START_OFFSET_MAX_S)
        )
        self.activity_rhythm: list[float] = self._build_activity_rhythm()
        self.polling_jitter_sigma: float = float(
            self._rng.uniform(POLLING_JITTER_SIGMA_MIN_MS, POLLING_JITTER_SIGMA_MAX_MS)
        )
        self.order_size_variance: float = float(
            self._rng.uniform(ORDER_SIZE_VARIANCE_MIN, ORDER_SIZE_VARIANCE_MAX)
        )

    def _build_activity_rhythm(self) -> list[float]:
        """Construct a 390-element weight array shaped like human trading attention.

        Returns:
            List of floats in [RHYTHM_FLOOR, 1.0].
        """
        minutes = np.arange(TRADING_MINUTES, dtype=float)

        open_peak = np.exp(
            -0.5 * ((minutes - OPEN_PEAK_CENTER_MIN) / OPEN_PEAK_SIGMA_MIN) ** 2
        )
        close_peak = np.exp(
            -0.5 * ((minutes - CLOSE_PEAK_CENTER_MIN) / CLOSE_PEAK_SIGMA_MIN) ** 2
        )
        midday_dip = MIDDAY_DIP_DEPTH * np.exp(
            -0.5 * ((minutes - MIDDAY_DIP_CENTER_MIN) / MIDDAY_DIP_SIGMA_MIN) ** 2
        )

        rhythm = open_peak + close_peak + BACKGROUND_ACTIVITY_WEIGHT - midday_dip
        rhythm = np.clip(rhythm, RHYTHM_FLOOR, None)
        rhythm = rhythm / rhythm.max()  # normalise so peak weight == 1.0
        return rhythm.tolist()

    def should_act(self, current_minute: int) -> bool:
        """Return whether the bot should act at *current_minute* of the trading day.

        Draws a uniform random sample and compares it against the activity weight
        for that minute.  Minutes with weight 1.0 (open/close peaks) will almost
        always return True; depressed midday minutes will return True less often,
        mirroring how a human trader's attention wanders.

        Args:
            current_minute: Minutes elapsed since market open.  Valid range: 0 – 389.

        Returns:
            ``True`` if the bot should process signals at this minute.
        """
        if current_minute < 0 or current_minute >= TRADING_MINUTES:
            return False
        weight = self.activity_rhythm[current_minute]
        return bool(self._rng.random() < weight)

    def get_connection_delay(self) -> float:
        """Return the session's randomized startup delay in seconds.

        Returns:
            Float in [0, 180] seconds.
        """
        return self.session_start_offset


# ---------------------------------------------------------------------------
# PacedAPIClient
# ---------------------------------------------------------------------------


class PacedAPIClient:
    """Wraps an async callable to enforce jitter and rate limiting before each call.

    All Phase 2 API interactions must route through this wrapper to avoid
    mechanical, clock-aligned request patterns that could be detected as
    automated activity.

    Usage::

        client = PacedAPIClient(some_async_fn, profile=session_profile)
        result = await client.call(arg1, kwarg=value)
    """

    def __init__(
        self,
        callable_fn: Callable[..., Any],
        rate_limiter: Optional[TokenBucket] = None,
        profile: Optional[BehaviorProfile] = None,
    ) -> None:
        """Initialise the paced client.

        Args:
            callable_fn:  Async callable to wrap.  Must return an awaitable.
            rate_limiter: ``TokenBucket`` to enforce call frequency limits.
                          Defaults to ``default_rate_limiter``.
            profile:      ``BehaviorProfile`` supplying ``polling_jitter_sigma``.
                          If ``None``, a fixed 100 ms sigma is used.
        """
        self._fn: Callable[..., Any] = callable_fn
        self._rate_limiter: TokenBucket = rate_limiter or default_rate_limiter
        self._profile: Optional[BehaviorProfile] = profile

    async def call(self, *args: Any, **kwargs: Any) -> Any:
        """Apply jitter, wait for a token, then execute the wrapped callable.

        Sequence of operations per call:
        1. Draw a jitter delay from ``gaussian_jitter`` using the profile's sigma.
        2. Sleep for that jitter duration.
        3. Acquire a token from the rate limiter (blocks if bucket is empty).
        4. Invoke the wrapped callable and return its result.

        Args:
            *args:   Positional arguments forwarded to the wrapped callable.
            **kwargs: Keyword arguments forwarded to the wrapped callable.

        Returns:
            Whatever the wrapped callable returns.
        """
        sigma = (
            self._profile.polling_jitter_sigma
            if self._profile is not None
            else PACED_CLIENT_DEFAULT_JITTER_SIGMA_MS
        )
        jitter_ms = gaussian_jitter(0.0, sigma)
        jitter_s = jitter_ms / 1000.0

        logger.debug(
            f"PacedAPIClient: jitter={jitter_ms:.1f}ms — waiting for rate-limiter token"
        )
        await asyncio.sleep(jitter_s)
        await self._rate_limiter.consume()

        t0 = time.perf_counter()
        result = await self._fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        logger.debug(f"PacedAPIClient: call completed in {elapsed_ms:.2f}ms")
        return result
