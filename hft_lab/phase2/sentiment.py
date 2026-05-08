"""News sentiment analysis for hft_lab Phase 2.

``SentimentAnalyzer`` polls the NewsAPI every ``NEWS_POLL_INTERVAL`` seconds
for headlines mentioning the primary trading symbol.  A lightweight lexicon
scorer produces a headline score in [-1, +1] which is then EMA-smoothed over
the last five headlines into ``current_sentiment``.

The poller runs as an async background task and never blocks the trading loop.
All API errors are caught and logged; the sentiment defaults to 0.0 on failure.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import List, Optional

from config import config
from logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------

BULLISH_WORDS: frozenset[str] = frozenset({
    "surge", "beat", "rally", "upgrade", "growth", "profit", "record",
    "strong", "positive", "gains", "buy", "outperform", "exceed", "raise",
    "boost", "climb", "rise", "advance", "gain", "expand", "soar", "jump",
    "spike", "accelerate", "approve", "breakthrough", "bullish", "confident",
    "demand", "dividend", "earn", "forecast", "grow", "improve", "increase",
    "innovative", "launch", "lead", "milestone", "opportunity", "optimistic",
    "peak", "perform", "potential", "premium", "progress", "recovery",
    "rebound", "revenue", "reward", "robust", "success", "thriving",
})

BEARISH_WORDS: frozenset[str] = frozenset({
    "crash", "miss", "downgrade", "loss", "decline", "fall", "weak",
    "negative", "cut", "underperform", "below", "lower", "reduce", "risk",
    "concern", "drop", "slide", "sell", "plunge", "dive", "tank", "collapse",
    "tumble", "retreat", "sink", "plummet", "disappoint", "warn", "delay",
    "shortfall", "deficit", "fear", "trouble", "crisis", "fraud", "lawsuit",
    "penalty", "ban", "recall", "reject", "suspend", "halt", "freeze",
    "debt", "liability", "probe", "investigation", "violation", "lawsuit",
    "downside", "bearish", "layoff", "writedown", "impairment",
})

# EMA smoothing factor for headline scores (window ≈ 5 headlines)
EMA_ALPHA: float = 2.0 / (5.0 + 1.0)


# ---------------------------------------------------------------------------
# SentimentAnalyzer
# ---------------------------------------------------------------------------


class SentimentAnalyzer:
    """Async news sentiment poller using NewsAPI and a lexicon scorer.

    Usage::

        analyzer = SentimentAnalyzer()
        task = asyncio.create_task(analyzer.run_poll_loop())
        # ...later...
        score = analyzer.current_sentiment
    """

    def __init__(self) -> None:
        self._sentiment: float = 0.0
        self._last_updated: Optional[datetime] = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._ema: Optional[float] = None

    @property
    def current_sentiment(self) -> float:
        """EMA-smoothed sentiment score in [-1.0, +1.0].

        Returns 0.0 before the first successful poll.
        """
        return self._sentiment

    @property
    def last_updated(self) -> Optional[datetime]:
        """UTC timestamp of the most recent successful poll, or None."""
        return self._last_updated

    def stop(self) -> None:
        """Signal the poll loop to exit on its next iteration."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _score_headline(text: str) -> float:
        """Score one headline using the bullish/bearish lexicon.

        Args:
            text: Headline string (case-insensitive).

        Returns:
            Float in [-1.0, +1.0].
        """
        words = text.lower().split()
        bullish = sum(1 for w in words if w.rstrip(".,!?;:") in BULLISH_WORDS)
        bearish = sum(1 for w in words if w.rstrip(".,!?;:") in BEARISH_WORDS)
        denom = bullish + bearish + 1
        return (bullish - bearish) / denom

    def _update_ema(self, score: float) -> None:
        """Update the running EMA with a new headline score.

        Args:
            score: Latest headline score.
        """
        if self._ema is None:
            self._ema = score
        else:
            self._ema = EMA_ALPHA * score + (1.0 - EMA_ALPHA) * self._ema
        self._sentiment = float(max(-1.0, min(1.0, self._ema)))

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    async def _poll(self) -> None:
        """Fetch headlines and update the EMA sentiment score.

        Runs the synchronous NewsAPI call in a thread executor to avoid
        blocking the event loop.
        """
        from newsapi import NewsApiClient

        def _fetch() -> List[str]:
            client = NewsApiClient(api_key=config.news_api_key)
            response = client.get_everything(
                q=f'"{config.primary_symbol}"',
                language="en",
                sort_by="publishedAt",
                page_size=10,
            )
            if response.get("status") != "ok":
                return []
            articles = response.get("articles", [])
            texts = []
            for article in articles:
                title = article.get("title") or ""
                description = article.get("description") or ""
                texts.append(f"{title} {description}")
            return texts

        try:
            loop = asyncio.get_running_loop()
            headlines = await loop.run_in_executor(None, _fetch)

            if headlines:
                for headline in headlines:
                    score = self._score_headline(headline)
                    self._update_ema(score)
                    logger.info(
                        f"Sentiment | score={score:+.3f} "
                        f"ema={self._sentiment:+.3f} | {headline[:80]}"
                    )
            self._last_updated = datetime.now(timezone.utc)

        except Exception as exc:
            logger.error(f"Sentiment poll error: {exc!r} — returning 0.0")
            # Do not modify _sentiment; keep last known value

    async def run_poll_loop(self) -> None:
        """Async background loop: poll NewsAPI every NEWS_POLL_INTERVAL seconds.

        Designed to run as an ``asyncio.Task``.  Exits cleanly when ``stop()``
        is called.
        """
        logger.info(
            f"SentimentAnalyzer starting — polling every {config.news_poll_interval}s "
            f"for '{config.primary_symbol}' headlines"
        )
        while not self._stop_event.is_set():
            await self._poll()
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=float(config.news_poll_interval),
                )
                break
            except asyncio.TimeoutError:
                pass
        logger.info("SentimentAnalyzer stopped")
