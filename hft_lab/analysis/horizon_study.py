#!/usr/bin/env python3
"""IC-by-horizon study for GBP_USD 15-minute bars.

Standalone offline analysis — no live trading, no imports from the engine's
session or execution layers.  Reuses only ``phase2.signals`` and
``phase2.regime`` so the scores and regime labels are exactly what the live
engine would compute.

Pipeline
--------
1. Fetch 6+ months of GBP_USD M15 candles from OANDA, paginating past the
   5000-candle response cap, and cache to ``analysis/data/GBP_USD_15Min.csv``.
2. Replay the history bar by bar through a trailing-window stand-in for
   DataManager so every signal and the regime classification see only data up
   to and including the bar being scored (strict no-lookahead).
3. For horizons k ∈ {1, 2, 4, 8, 16, 32, 96} bars, compute the Pearson IC of
   each signal against the k-bar forward return — the same IC definition
   (pearsonr with min-sample and zero-variance guards) as
   ``phase2.ensemble.InformationCoefficient`` — per regime and pooled.
4. Per horizon: mean absolute forward move in pips and its ratio to a 2-pip
   round-trip cost.
5. Write a long-format CSV and a heatmap PNG to ``analysis/output/``, and
   print the top five signal-horizon-regime cells ranked by |IC| (signed IC
   shown; cells with < 100 observations are flagged unreliable and excluded
   from the ranking).

Usage::

    cd hft_lab && python3 analysis/horizon_study.py
"""
from __future__ import annotations

import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

# Make hft_lab root importable when run as `python3 analysis/horizon_study.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

INSTRUMENT: str = "GBP_USD"
GRANULARITY: str = "M15"
BAR_MINUTES: int = 15
MONTHS_BACK: float = 6.5              # fetch a little over 6 months
HORIZONS: List[int] = [1, 2, 4, 8, 16, 32, 96]
PIP: float = 0.0001
ROUND_TRIP_COST_PIPS: float = 2.0
WINDOW_BARS: int = 120                # trailing bars visible to indicators
BURN_IN_BARS: int = 60                # skip scoring until indicators warm
MIN_OBS: int = 100                    # cells below this are unreliable
MAX_FETCH_PAGES: int = 50

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")
OUTPUT_DIR = os.path.join(_HERE, "output")
CACHE_PATH = os.path.join(DATA_DIR, "GBP_USD_15Min.csv")


# ---------------------------------------------------------------------------
# Data fetch (paginated, cached)
# ---------------------------------------------------------------------------


def fetch_candles() -> pd.DataFrame:
    """Return the full candle history, fetching + caching on first run."""
    if os.path.exists(CACHE_PATH):
        df = pd.read_csv(CACHE_PATH, parse_dates=["timestamp"])
        print(f"Loaded {len(df)} cached bars from {CACHE_PATH}")
        return df

    from oandapyV20 import API
    from oandapyV20.endpoints.instruments import InstrumentsCandles
    from config import config

    client = API(
        access_token=config.oanda_api_key,
        environment=config.oanda_environment,
        request_params={"timeout": 30},
    )

    start = datetime.now(timezone.utc) - timedelta(days=int(MONTHS_BACK * 30.5))
    frm = start
    rows: List[Dict] = []
    last_seen: Optional[str] = None

    for page in range(MAX_FETCH_PAGES):
        params = {
            "from": frm.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "granularity": GRANULARITY,
            "count": 5000,
            "price": "M",
        }
        response = client.request(InstrumentsCandles(INSTRUMENT, params=params))
        candles = response.get("candles", [])
        added = 0
        for c in candles:
            if not c.get("complete", False):
                continue
            t = c["time"]
            if last_seen is not None and t <= last_seen:
                continue
            mid = c.get("mid", {})
            rows.append({
                "timestamp": t,
                "open": float(mid.get("o", 0.0)),
                "high": float(mid.get("h", 0.0)),
                "low": float(mid.get("l", 0.0)),
                "close": float(mid.get("c", 0.0)),
                "volume": float(c.get("volume", 1)),
            })
            last_seen = t
            added += 1
        print(f"  page {page + 1}: +{added} bars (total {len(rows)})")
        if not candles or added == 0 or len(candles) < 4999:
            break
        frm = datetime.fromisoformat(
            candles[-1]["time"].replace("Z", "+00:00").split(".")[0] + "+00:00"
        ) + timedelta(seconds=1)
        time.sleep(0.25)  # politeness between pages

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["vwap_contribution"] = df["close"] * df["volume"]

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    print(f"Fetched {len(df)} bars → cached at {CACHE_PATH}")
    return df


# ---------------------------------------------------------------------------
# Trailing-window DataManager stand-in (no-lookahead by construction)
# ---------------------------------------------------------------------------


class _StubFast:
    """FastBuffer stand-in exposing only the latest bar close as a price."""

    def __init__(self) -> None:
        self.price: Optional[float] = None

    def latest_price(self) -> Optional[float]:
        return self.price

    def latest_bid(self) -> Optional[float]:
        return None

    def latest_ask(self) -> Optional[float]:
        return None

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame()   # no L2 → order_book uses (empty) tick history


class _MediumProxy:
    """MediumBuffer stand-in returning the owner's current trailing window."""

    def __init__(self, owner: "StudyDataManager") -> None:
        self._owner = owner

    def to_dataframe(self) -> pd.DataFrame:
        return self._owner.window_df


class StudyDataManager:
    """Read-only DataManager stand-in over a trailing bar window.

    Exposes exactly the surface consumed by SignalEngine and RegimeClassifier:
    ``medium_primary.to_dataframe()``, ``fast_primary`` price accessors,
    ``compute_vwap()``, ``compute_relative_strength()``, ``_tick_history``.
    The window is set externally per bar, so indicators can never see a
    candle later than the bar being scored.
    """

    def __init__(self) -> None:
        self.window_df: pd.DataFrame = pd.DataFrame()
        self.fast_primary = _StubFast()
        self.fast_benchmark = _StubFast()
        self.medium_primary = _MediumProxy(self)
        self._tick_history: List = []

    def set_window(self, window_df: pd.DataFrame) -> None:
        self.window_df = window_df
        self.fast_primary.price = float(window_df["close"].iloc[-1])

    def to_dataframe(self) -> pd.DataFrame:
        return self.window_df

    def compute_vwap(self) -> Optional[float]:
        vol = float(self.window_df["volume"].sum())
        if vol <= 0:
            return None
        return float(self.window_df["vwap_contribution"].sum() / vol)

    def compute_relative_strength(self) -> Optional[float]:
        return None   # no benchmark in this study → signal scores 0.0


# ---------------------------------------------------------------------------
# IC (same definition as phase2.ensemble.InformationCoefficient.get_ic)
# ---------------------------------------------------------------------------


def compute_ic(preds: np.ndarray, actuals: np.ndarray) -> float:
    """Pearson IC with the ensemble's min-sample and zero-variance guards."""
    if len(preds) < 4:
        return 0.0
    if np.std(preds) < 1e-10 or np.std(actuals) < 1e-10:
        return 0.0
    try:
        r, _ = pearsonr(preds, actuals)
        return float(np.clip(r, -1.0, 1.0)) if not math.isnan(r) else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Main study
# ---------------------------------------------------------------------------


def main() -> None:
    # Engine modules log per-bar regime transitions etc. — silence them
    for name in ["phase2.signals", "phase2.regime", "phase2.data"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    from phase2.regime import RegimeClassifier, RegimeState
    from phase2.signals import SIGNAL_NAMES, SignalEngine

    df = fetch_candles()
    n = len(df)
    if n < BURN_IN_BARS + max(HORIZONS) + MIN_OBS:
        print(f"ERROR: only {n} bars available — not enough for the study")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Replay: per-bar scores + regime, trailing window only
    # ------------------------------------------------------------------
    dm = StudyDataManager()
    engine = SignalEngine(dm)
    classifier = RegimeClassifier()

    closes = df["close"].values.astype(float)
    score_rows: List[Dict] = []
    t0 = time.time()
    for i in range(BURN_IN_BARS, n):
        window = df.iloc[max(0, i - WINDOW_BARS + 1): i + 1]
        dm.set_window(window)
        regime = classifier.classify(dm)
        scores = engine.compute_all(regime)
        score_rows.append({"bar": i, "regime": regime.value, **scores})
        if (i - BURN_IN_BARS) % 2000 == 0:
            print(
                f"  scored bar {i}/{n} "
                f"({(time.time() - t0):.0f}s elapsed)"
            )
    scored = pd.DataFrame(score_rows)
    print(f"Replay complete: {len(scored)} bars scored in {time.time() - t0:.0f}s")

    # ------------------------------------------------------------------
    # Forward returns per horizon (vectorized over the full close series)
    # ------------------------------------------------------------------
    fwd_returns: Dict[int, np.ndarray] = {}
    for k in HORIZONS:
        fr = np.full(n, np.nan)
        fr[: n - k] = (closes[k:] - closes[:-k]) / closes[:-k]
        fwd_returns[k] = fr

    # ------------------------------------------------------------------
    # IC matrix: signal × horizon × regime (+ pooled "ALL")
    # ------------------------------------------------------------------
    regimes = [r.value for r in RegimeState] + ["ALL"]
    results: List[Dict] = []
    for k in HORIZONS:
        fr = fwd_returns[k]
        bar_idx = scored["bar"].values
        valid = ~np.isnan(fr[bar_idx])
        for regime_name in regimes:
            if regime_name == "ALL":
                mask = valid
            else:
                mask = valid & (scored["regime"].values == regime_name)
            actuals = fr[bar_idx[mask]]
            for sig in SIGNAL_NAMES:
                preds = scored[sig].values[mask]
                n_obs = int(mask.sum())
                results.append({
                    "signal": sig,
                    "horizon_bars": k,
                    "regime": regime_name,
                    "ic": compute_ic(preds, actuals),
                    "n_obs": n_obs,
                    "reliable": n_obs >= MIN_OBS,
                })
    results_df = pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Cost analysis per horizon
    # ------------------------------------------------------------------
    cost_rows: List[Dict] = []
    for k in HORIZONS:
        fr = fwd_returns[k]
        abs_pips = np.nanmean(np.abs(fr)) * closes[~np.isnan(fr)].mean() / PIP
        cost_rows.append({
            "horizon_bars": k,
            "horizon_hours": k * BAR_MINUTES / 60.0,
            "mean_abs_fwd_pips": abs_pips,
            "cost_ratio_vs_2pip": abs_pips / ROUND_TRIP_COST_PIPS,
        })
    cost_df = pd.DataFrame(cost_rows)

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ic_csv = os.path.join(OUTPUT_DIR, "ic_by_horizon.csv")
    cost_csv = os.path.join(OUTPUT_DIR, "horizon_cost.csv")
    results_df.to_csv(ic_csv, index=False)
    cost_df.to_csv(cost_csv, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(regimes), figsize=(4.2 * len(regimes), 4.5), squeeze=False
    )
    for ax, regime_name in zip(axes[0], regimes):
        sub = results_df[results_df["regime"] == regime_name]
        grid = np.full((len(SIGNAL_NAMES), len(HORIZONS)), np.nan)
        for r_i, sig in enumerate(SIGNAL_NAMES):
            for c_i, k in enumerate(HORIZONS):
                cell = sub[(sub["signal"] == sig) & (sub["horizon_bars"] == k)]
                if len(cell) and cell["reliable"].iloc[0]:
                    grid[r_i, c_i] = cell["ic"].iloc[0]
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-0.15, vmax=0.15, aspect="auto")
        ax.set_xticks(range(len(HORIZONS)), [str(k) for k in HORIZONS])
        ax.set_yticks(range(len(SIGNAL_NAMES)), list(SIGNAL_NAMES))
        n_cell = sub["n_obs"].max() if len(sub) else 0
        ax.set_title(f"{regime_name} (n≤{n_cell})", fontsize=10)
        ax.set_xlabel("horizon (bars)")
        for r_i in range(len(SIGNAL_NAMES)):
            for c_i in range(len(HORIZONS)):
                v = grid[r_i, c_i]
                ax.text(
                    c_i, r_i,
                    "n/a" if np.isnan(v) else f"{v:+.2f}",
                    ha="center", va="center", fontsize=7,
                    color="black",
                )
    fig.suptitle(
        f"{INSTRUMENT} {GRANULARITY} — IC by signal × horizon × regime "
        f"(cells with n<{MIN_OBS} shown as n/a)",
        fontsize=12,
    )
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="IC")
    heatmap_path = os.path.join(OUTPUT_DIR, "ic_heatmap.png")
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")

    # ------------------------------------------------------------------
    # Stdout report
    # ------------------------------------------------------------------
    print(f"\nWrote {ic_csv}")
    print(f"Wrote {cost_csv}")
    print(f"Wrote {heatmap_path}\n")

    print("Horizon cost analysis (2-pip round trip):")
    print(cost_df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    reliable = results_df[results_df["reliable"]].copy()
    reliable["abs_ic"] = reliable["ic"].abs()
    top = reliable.sort_values("abs_ic", ascending=False).head(5)
    print("\nTop 5 signal-horizon-regime combinations by |IC| (signed IC shown):")
    for _, row in top.iterrows():
        print(
            f"  {row['signal']:<18} k={row['horizon_bars']:<3} "
            f"{row['regime']:<15} IC={row['ic']:+.4f}  n={row['n_obs']}"
        )
    unreliable_n = int((~results_df["reliable"]).sum())
    if unreliable_n:
        print(
            f"\n{unreliable_n} cell(s) had n<{MIN_OBS} observations and were "
            "flagged unreliable (excluded from ranking; see CSV)."
        )


if __name__ == "__main__":
    main()
