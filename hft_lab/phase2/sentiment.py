"""News sentiment analysis for hft_lab Phase 2.

NLP backend priority
--------------------
1. **FinBERT** (``ProsusAI/finbert``) — loaded in a background thread at
   startup.  Returns a score in [-1, +1] mapped from NEGATIVE / NEUTRAL /
   POSITIVE labels.
2. **VADER** (``vaderSentiment``) — fallback when FinBERT is unavailable
   (no ``torch`` or ``transformers`` installed).
3. **Lexicon scorer** — fast keyword fallback used only when both NLP
   libraries are absent.

News sources
------------
* **NewsAPI** — primary source; query is market-type-aware.
* **Reddit** (optional) — enabled via ``REDDIT_SENTIMENT_ENABLED=true`` in
  ``.env``; uses ``praw`` to fetch top posts from relevant subreddits.

Combined score
--------------
``combined = 0.7 × news_score + 0.3 × reddit_score``

When Reddit is disabled, ``combined = news_score``.

Calendar risk delegation
------------------------
``SentimentAnalyzer`` holds an ``EventCalendar(silent=True)`` so callers can
query ``get_risk_multiplier()`` and ``get_threshold_multiplier()`` without
creating a second calendar that logs duplicate transition messages.

Usage::

    analyzer = SentimentAnalyzer()
    task = asyncio.create_task(analyzer.run_poll_loop())
    score = analyzer.current_sentiment
    risk_mult = analyzer.get_risk_multiplier()
"""
from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timezone
from typing import Callable, List, Optional

from config import config
from logger import get_logger
from phase2.news_calendar import EventCalendar

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

EMA_ALPHA: float = 2.0 / (5.0 + 1.0)

# ---------------------------------------------------------------------------
# Fallback lexicon
# ---------------------------------------------------------------------------

_BULLISH: frozenset[str] = frozenset({
    "surge", "beat", "rally", "upgrade", "growth", "profit", "record",
    "strong", "positive", "gains", "buy", "outperform", "exceed", "raise",
    "boost", "climb", "rise", "advance", "gain", "expand", "soar", "jump",
    "spike", "bullish", "breakthrough", "improve", "increase", "recovery",
    "rebound", "revenue", "reward", "robust", "success", "optimistic",
})

_BEARISH: frozenset[str] = frozenset({
    "crash", "miss", "downgrade", "loss", "decline", "fall", "weak",
    "negative", "cut", "underperform", "below", "lower", "reduce", "risk",
    "concern", "drop", "slide", "sell", "plunge", "tank", "collapse",
    "tumble", "retreat", "disappoint", "warn", "crisis", "fraud", "lawsuit",
    "bearish", "layoff", "writedown", "deficit", "probe", "violation",
})


def _lexicon_score(text: str) -> float:
    words = text.lower().split()
    b = sum(1 for w in words if w.rstrip(".,!?;:") in _BULLISH)
    r = sum(1 for w in words if w.rstrip(".,!?;:") in _BEARISH)
    return (b - r) / (b + r + 1)


# ---------------------------------------------------------------------------
# NLP backend loader (runs in background thread at import time)
# ---------------------------------------------------------------------------

_NLP_LOCK = threading.Lock()
_nlp_scorer: Optional[Callable[[str], float]] = None
_nlp_backend_name: str = "lexicon"


def _load_nlp_backend() -> None:
    global _nlp_scorer, _nlp_backend_name
    try:
        from transformers import pipeline as hf_pipeline

        _pipe = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=-1,
            truncation=True,
            max_length=512,
        )
        _label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

        def _finbert_score(text: str) -> float:
            result = _pipe(text[:512])[0]
            label = result["label"].lower()
            score = result["score"]
            return _label_map.get(label, 0.0) * score

        with _NLP_LOCK:
            _nlp_scorer = _finbert_score
            _nlp_backend_name = "FinBERT"
        logger.info("NLP backend: FinBERT loaded (ProsusAI/finbert)")
        return
    except Exception as exc:
        logger.warning(f"FinBERT unavailable ({exc!r}) — trying VADER")

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VADER

        _vader = _VADER()

        def _vader_score(text: str) -> float:
            return float(_vader.polarity_scores(text)["compound"])

        with _NLP_LOCK:
            _nlp_scorer = _vader_score
            _nlp_backend_name = "VADER"
        logger.info("NLP backend: VADER loaded")
        return
    except Exception as exc:
        logger.warning(f"VADER unavailable ({exc!r}) — falling back to lexicon scorer")

    with _NLP_LOCK:
        _nlp_scorer = _lexicon_score
        _nlp_backend_name = "lexicon"
    logger.info("NLP backend: lexicon scorer (keyword fallback)")


# Load in background so engine startup is non-blocking
threading.Thread(target=_load_nlp_backend, daemon=True, name="nlp-loader").start()


def _score_headline(text: str) -> float:
    with _NLP_LOCK:
        scorer = _nlp_scorer
    if scorer is None:
        return _lexicon_score(text)
    return scorer(text)


def get_nlp_backend_name() -> str:
    """Return the name of the active NLP backend (may be 'loading…' briefly)."""
    with _NLP_LOCK:
        if _nlp_scorer is None:
            return "loading…"
        return _nlp_backend_name


# ---------------------------------------------------------------------------
# Market-aware news query
# ---------------------------------------------------------------------------


def _build_query() -> str:
    """Build a NewsAPI search query appropriate for the current market type."""
    sym = config.primary_symbol
    if config.is_crypto():
        # Use readable coin name when possible
        coin_names = {
            "BTC/USD": "Bitcoin",
            "ETH/USD": "Ethereum",
            "SOL/USD": "Solana",
            "BNB/USD": "Binance Coin",
        }
        readable = coin_names.get(sym, sym.split("/")[0])
        return f'"{readable}" OR "{sym}"'
    if config.is_forex():
        # Forex: include both currencies and relevant central banks
        parts = sym.replace("_", "/").split("/")
        terms = list(parts)
        cb_map = {
            "EUR": "ECB", "GBP": "BOE", "JPY": "BOJ",
            "USD": "Fed", "CAD": "BOC", "AUD": "RBA", "NZD": "RBNZ",
        }
        for p in parts:
            if p in cb_map:
                terms.append(cb_map[p])
        return " OR ".join(f'"{t}"' for t in dict.fromkeys(terms))
    # Equity
    return f'"{sym}"'


# ---------------------------------------------------------------------------
# Reddit sentiment
# ---------------------------------------------------------------------------


def _fetch_reddit_sentiment_sync() -> Optional[float]:
    """Fetch top-post sentiment from relevant subreddits via praw.

    Returns a score in [-1, +1] or ``None`` on any error / disabled.
    """
    if not config.reddit_sentiment_enabled:
        return None

    try:
        import praw  # type: ignore

        reddit = praw.Reddit(
            client_id=config.reddit_client_id,
            client_secret=config.reddit_client_secret,
            user_agent="hft_lab/2.0 sentiment bot",
            check_for_async=False,
        )

        subreddit_map = {
            "equity": ["stocks", "investing", "wallstreetbets"],
            "crypto": ["CryptoCurrency", "Bitcoin", "ethereum"],
            "forex": ["Forex", "FXtraders", "algotrading"],
        }
        subs = subreddit_map.get(config.market_type.lower(), ["investing"])
        subreddit = reddit.subreddit("+".join(subs))

        scores: List[float] = []
        for post in subreddit.hot(limit=20):
            text = f"{post.title} {post.selftext[:200]}"
            scores.append(_score_headline(text))

        if not scores:
            return None
        result = sum(scores) / len(scores)
        logger.info(
            f"Sentiment [Reddit] | posts={len(scores)} avg={result:+.3f} "
            f"subs={'+'.join(subs)}"
        )
        return float(result)

    except Exception as exc:
        logger.warning(f"Reddit sentiment error: {exc!r} — skipping")
        return None


# ---------------------------------------------------------------------------
# SentimentAnalyzer
# ---------------------------------------------------------------------------


class SentimentAnalyzer:
    """Async news + Reddit sentiment poller with FinBERT / VADER / lexicon backend.

    Properties
    ----------
    current_sentiment : float
        Combined EMA-smoothed score in [-1.0, +1.0].
    last_updated : datetime | None
        UTC time of last successful poll.
    calendar_risk_active : bool
        True when an economic event's HIGH IMPACT window is active.

    Methods
    -------
    get_risk_multiplier() -> float
        Delegate to internal EventCalendar (silent=True).
    get_threshold_multiplier() -> float
        Delegate to internal EventCalendar (silent=True).
    """

    def __init__(self) -> None:
        self._news_score: float = 0.0
        self._reddit_score: float = 0.0
        self._combined_score: float = 0.0
        self._ema: Optional[float] = None
        self._last_updated: Optional[datetime] = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._calendar: EventCalendar = EventCalendar(silent=True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_sentiment(self) -> float:
        """Combined EMA-smoothed sentiment score in [-1.0, +1.0]."""
        return self._combined_score

    @property
    def last_updated(self) -> Optional[datetime]:
        """UTC timestamp of the most recent successful poll, or None."""
        return self._last_updated

    @property
    def calendar_risk_active(self) -> bool:
        """True when a HIGH or MEDIUM impact event window is currently active."""
        in_window, _ = self._calendar.is_high_impact_window()
        return in_window

    # ------------------------------------------------------------------
    # Calendar delegation
    # ------------------------------------------------------------------

    def get_risk_multiplier(self) -> float:
        """Position-size multiplier: 0.20 (HIGH), 0.50 (MEDIUM), or 1.0."""
        return self._calendar.get_risk_multiplier()

    def get_threshold_multiplier(self) -> float:
        """Signal-threshold multiplier: 1.50 (HIGH), 1.25 (MEDIUM), or 1.0."""
        return self._calendar.get_threshold_multiplier()

    def stop(self) -> None:
        """Signal the poll loop to exit on its next iteration."""
        self._stop_event.set()

    # ------------------------------------------------------------------
    # EMA update
    # ------------------------------------------------------------------

    def _update_ema(self, score: float) -> None:
        if self._ema is None:
            self._ema = score
        else:
            self._ema = EMA_ALPHA * score + (1.0 - EMA_ALPHA) * self._ema
        self._news_score = float(max(-1.0, min(1.0, self._ema)))

    def _update_combined(self) -> None:
        if config.reddit_sentiment_enabled and self._reddit_score != 0.0:
            self._combined_score = 0.7 * self._news_score + 0.3 * self._reddit_score
        else:
            self._combined_score = self._news_score
        self._combined_score = float(max(-1.0, min(1.0, self._combined_score)))

    # ------------------------------------------------------------------
    # News polling
    # ------------------------------------------------------------------

    async def _poll_news(self) -> None:
        """Fetch NewsAPI headlines and update the EMA news score."""
        from newsapi import NewsApiClient

        query = _build_query()
        backend = get_nlp_backend_name()

        def _fetch() -> List[str]:
            client = NewsApiClient(api_key=config.news_api_key)
            response = client.get_everything(
                q=query,
                language="en",
                sort_by="publishedAt",
                page_size=10,
            )
            if response.get("status") != "ok":
                return []
            articles = response.get("articles", [])
            return [
                f"{a.get('title') or ''} {a.get('description') or ''}"
                for a in articles
            ]

        try:
            loop = asyncio.get_running_loop()
            headlines = await loop.run_in_executor(None, _fetch)

            for headline in headlines:
                score = _score_headline(headline)
                self._update_ema(score)
                logger.info(
                    f"Sentiment [{backend}|NewsAPI] | "
                    f"score={score:+.3f} ema={self._news_score:+.3f} | "
                    f"{headline[:80]}"
                )

            self._last_updated = datetime.now(timezone.utc)

        except Exception as exc:
            logger.error(f"News poll error: {exc!r} — keeping last score")

    # ------------------------------------------------------------------
    # Reddit polling
    # ------------------------------------------------------------------

    async def _poll_reddit(self) -> None:
        """Fetch Reddit top-post sentiment and update reddit score."""
        if not config.reddit_sentiment_enabled:
            return
        loop = asyncio.get_running_loop()
        try:
            score = await loop.run_in_executor(None, _fetch_reddit_sentiment_sync)
            if score is not None:
                self._reddit_score = score
                backend = get_nlp_backend_name()
                logger.info(
                    f"Sentiment [{backend}|Reddit] | "
                    f"reddit_score={score:+.3f}"
                )
        except Exception as exc:
            logger.warning(f"Reddit poll error: {exc!r}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run_poll_loop(self) -> None:
        """Async background loop: poll NewsAPI (and Reddit) every interval.

        Designed to run as an ``asyncio.Task``.  Exits cleanly when
        ``stop()`` is called.
        """
        backend = get_nlp_backend_name()
        reddit_status = "enabled" if config.reddit_sentiment_enabled else "disabled"
        logger.info(
            f"SentimentAnalyzer starting | backend={backend} "
            f"reddit={reddit_status} "
            f"interval={config.news_poll_interval}s "
            f"query={_build_query()!r}"
        )

        while not self._stop_event.is_set():
            await self._poll_news()
            await self._poll_reddit()
            self._update_combined()
            logger.info(
                f"Sentiment COMBINED | "
                f"news={self._news_score:+.3f} "
                f"reddit={self._reddit_score:+.3f} "
                f"combined={self._combined_score:+.3f} "
                f"calendar_risk={self.calendar_risk_active}"
            )
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=float(config.news_poll_interval),
                )
                break
            except asyncio.TimeoutError:
                pass

        logger.info("SentimentAnalyzer stopped")
