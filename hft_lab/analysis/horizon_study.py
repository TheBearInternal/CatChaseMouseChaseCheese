#!/usr/bin/env python3
"""IC-by-horizon study — multi-instrument, significance-tested, out-of-sample.

Standalone offline analysis — no live trading, no imports from the engine's
session or execution layers.  Reuses only ``phase2.signals`` and
``phase2.regime`` so scores and regime labels are exactly what the live
engine would compute.

Pipeline
--------
1. Fetch N years (default 3) of M15 candles per instrument from OANDA,
   paginating past the 5000-candle cap; cache per instrument under
   ``analysis/data/`` (a cache shorter than the requested span refetches).
2. Replay each history bar by bar through a trailing-window DataManager
   stand-in — strict no-lookahead by construction.
3. Chronological 70/30 train/test split.  Train observations whose forward
   window crosses the split boundary are dropped so no train cell touches
   test-period prices.
4. Per cell (signal × horizon × regime × split): IC with the same Pearson
   definition as ``phase2.ensemble``, observation count, Newey-West HAC
   standard error with lag = horizon (overlapping forward windows badly
   understate the naive SE), the resulting t-stat, plus a non-overlapping
   (stride = horizon) IC and classical t as a cross-check.  Cells with
   |t_NW| < 2 are flagged not significant; cells with n < 100 unreliable.
5. Per horizon × instrument: mean |forward move| in pips (JPY quotes use
   0.01) and the ratio to a 2-pip round-trip cost.
6. Outputs to ``analysis/output/``: long-format summary CSV, a markdown
   report (per-instrument top tables, cross-instrument consistency view,
   cost tables), and a per-instrument test-IC heatmap.  Stdout prints the
   top five cells by test |IC| (reliable + significant only) with t-stats
   and observation counts.

Usage::

    cd hft_lab && python3 analysis/horizon_study.py \
        [--years 3] [--instruments GBP_USD,EUR_USD,USD_JPY,AUD_USD]
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

# Make hft_lab root importable when run as `python3 analysis/horizon_study.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

DEFAULT_YEARS: float = 3.0
DEFAULT_INSTRUMENTS: str = "GBP_USD,EUR_USD,USD_JPY,AUD_USD"
GRANULARITY: str = "M15"
BAR_MINUTES: int = 15
HORIZONS: List[int] = [1, 2, 4, 8, 16, 32, 96]
ROUND_TRIP_COST_PIPS: float = 2.0
WINDOW_BARS: int = 120                # trailing bars visible to indicators
BURN_IN_BARS: int = 60                # skip scoring until indicators warm
MIN_OBS: int = 100                    # cells below this are unreliable
T_SIGNIFICANT: float = 2.0            # |t_NW| threshold for significance
TRAIN_FRAC: float = 0.70
MAX_FETCH_PAGES: int = 300

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")
OUTPUT_DIR = os.path.join(_HERE, "output")


def pip_size(instrument: str) -> float:
    """Return the pip size for an instrument (JPY quotes use 0.01)."""
    return 0.01 if instrument.upper().endswith("_JPY") else 0.0001


# ---------------------------------------------------------------------------
# Data fetch (paginated, cached per instrument)
# ---------------------------------------------------------------------------


def fetch_candles(instrument: str, years: float) -> pd.DataFrame:
    """Return the candle history for one instrument, fetching + caching.

    A cached file whose span is materially shorter than the requested number
    of years triggers a full refetch.
    """
    cache_path = os.path.join(DATA_DIR, f"{instrument}_{GRANULARITY}.csv")
    needed_days = years * 365.25 - 45   # tolerance for weekends/holidays

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["timestamp"])
        if len(df) > 0:
            span_days = (df["timestamp"].max() - df["timestamp"].min()).days
            if span_days >= needed_days:
                print(
                    f"[{instrument}] loaded {len(df)} cached bars "
                    f"({span_days}d span) from {cache_path}"
                )
                return df
            print(
                f"[{instrument}] cache spans {span_days}d < requested "
                f"{needed_days:.0f}d — refetching"
            )

    from oandapyV20 import API
    from oandapyV20.endpoints.instruments import InstrumentsCandles
    from config import config

    client = API(
        access_token=config.oanda_api_key,
        environment=config.oanda_environment,
        request_params={"timeout": 30},
    )

    start = datetime.now(timezone.utc) - timedelta(days=years * 365.25)
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
        response = client.request(InstrumentsCandles(instrument, params=params))
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
        if page % 5 == 0 or added == 0:
            print(f"[{instrument}]   page {page + 1}: +{added} (total {len(rows)})")
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
    df.to_csv(cache_path, index=False)
    print(f"[{instrument}] fetched {len(df)} bars → cached at {cache_path}")
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

    Exposes exactly the surface consumed by SignalEngine and RegimeClassifier.
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
# Statistics
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


def newey_west_se(preds: np.ndarray, actuals: np.ndarray, lag: int) -> float:
    """HAC standard error of the correlation coefficient, Bartlett kernel.

    Overlapping k-bar forward windows autocorrelate the scoring residuals up
    to ~k lags, so the naive SE (≈1/√n) is badly understated.  This treats
    the correlation as the OLS slope of standardized y on standardized x and
    applies Newey-West with the given lag to the score series u_t = x_t·e_t.
    Sanity: with lag 0 and independent residuals this reduces to the
    classical √((1-r²)/n).
    """
    n = len(preds)
    if n < 8:
        return float("inf")
    sx, sy = float(np.std(preds)), float(np.std(actuals))
    if sx < 1e-10 or sy < 1e-10:
        return float("inf")
    x = (preds - preds.mean()) / sx
    y = (actuals - actuals.mean()) / sy
    r = float(np.mean(x * y))
    e = y - r * x
    u = x * e
    s_var = float(np.mean(u * u))
    max_lag = min(lag, n - 2)
    for l in range(1, max_lag + 1):
        w = 1.0 - l / (max_lag + 1.0)
        gamma = float(np.mean(u[l:] * u[:-l]))
        s_var += 2.0 * w * gamma
    s_var = max(s_var, 1e-12)
    return math.sqrt(s_var / n)


def classical_t(r: float, n: int) -> float:
    """Classical correlation t-stat for independent samples."""
    if n < 4 or abs(r) >= 1.0:
        return 0.0
    return r * math.sqrt(max(n - 2, 1)) / math.sqrt(max(1.0 - r * r, 1e-12))


def cell_stats(
    preds: np.ndarray, actuals: np.ndarray, horizon: int
) -> Dict[str, float]:
    """Full statistics for one IC cell (overlapping + non-overlapping)."""
    n = len(preds)
    ic = compute_ic(preds, actuals)
    se = newey_west_se(preds, actuals, lag=horizon)
    t_nw = ic / se if se not in (0.0, float("inf")) else 0.0
    # Non-overlapping cross-check: stride the sample by the horizon
    preds_no = preds[::horizon]
    actuals_no = actuals[::horizon]
    ic_no = compute_ic(preds_no, actuals_no)
    return {
        "ic": ic,
        "n_obs": n,
        "se_nw": se if se != float("inf") else float("nan"),
        "t_nw": t_nw,
        "ic_nonoverlap": ic_no,
        "n_nonoverlap": len(preds_no),
        "t_nonoverlap": classical_t(ic_no, len(preds_no)),
    }


# ---------------------------------------------------------------------------
# Per-instrument analysis
# ---------------------------------------------------------------------------


def replay_scores(instrument: str, df: pd.DataFrame) -> pd.DataFrame:
    """Replay the history bar by bar and return per-bar scores + regime."""
    from phase2.regime import RegimeClassifier
    from phase2.signals import SignalEngine

    dm = StudyDataManager()
    engine = SignalEngine(dm)
    classifier = RegimeClassifier()

    n = len(df)
    rows: List[Dict] = []
    t0 = time.time()
    for i in range(BURN_IN_BARS, n):
        window = df.iloc[max(0, i - WINDOW_BARS + 1): i + 1]
        dm.set_window(window)
        regime = classifier.classify(dm)
        scores = engine.compute_all(regime)
        rows.append({"bar": i, "regime": regime.value, **scores})
        done = i - BURN_IN_BARS
        if done and done % 10000 == 0:
            rate = done / (time.time() - t0)
            eta = (n - i) / max(rate, 1e-9)
            print(
                f"[{instrument}]   scored {done}/{n - BURN_IN_BARS} bars "
                f"({rate:.0f}/s, ~{eta / 60:.1f} min left)"
            )
    print(
        f"[{instrument}] replay complete: {len(rows)} bars in "
        f"{time.time() - t0:.0f}s"
    )
    return pd.DataFrame(rows)


def analyze_instrument(
    instrument: str, df: pd.DataFrame, scored: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ic_results, cost_results) DataFrames for one instrument."""
    from phase2.regime import RegimeState
    from phase2.signals import SIGNAL_NAMES

    closes = df["close"].values.astype(float)
    n = len(df)
    split_bar = int(n * TRAIN_FRAC)
    pip = pip_size(instrument)

    fwd_returns: Dict[int, np.ndarray] = {}
    for k in HORIZONS:
        fr = np.full(n, np.nan)
        fr[: n - k] = (closes[k:] - closes[:-k]) / closes[:-k]
        fwd_returns[k] = fr

    bar_idx = scored["bar"].values
    regimes = [r.value for r in RegimeState] + ["ALL"]

    results: List[Dict] = []
    for k in HORIZONS:
        fr = fwd_returns[k]
        valid = ~np.isnan(fr[bar_idx])
        # Leak-free split: train forward windows must end before the boundary
        train_mask = valid & (bar_idx + k <= split_bar)
        test_mask = valid & (bar_idx >= split_bar)
        for split_name, split_mask in (("train", train_mask), ("test", test_mask)):
            for regime_name in regimes:
                if regime_name == "ALL":
                    mask = split_mask
                else:
                    mask = split_mask & (scored["regime"].values == regime_name)
                actuals = fr[bar_idx[mask]]
                for sig in SIGNAL_NAMES:
                    preds = scored[sig].values[mask]
                    stats = cell_stats(preds, actuals, k)
                    results.append({
                        "instrument": instrument,
                        "signal": sig,
                        "horizon_bars": k,
                        "regime": regime_name,
                        "split": split_name,
                        **stats,
                        "significant": abs(stats["t_nw"]) >= T_SIGNIFICANT,
                        "reliable": stats["n_obs"] >= MIN_OBS,
                    })

    cost_rows: List[Dict] = []
    for k in HORIZONS:
        fr = fwd_returns[k]
        ok = ~np.isnan(fr)
        abs_pips = float(np.mean(np.abs(fr[ok])) * closes[ok].mean() / pip)
        cost_rows.append({
            "instrument": instrument,
            "horizon_bars": k,
            "horizon_hours": k * BAR_MINUTES / 60.0,
            "mean_abs_fwd_pips": abs_pips,
            "cost_ratio_vs_2pip": abs_pips / ROUND_TRIP_COST_PIPS,
        })
    return pd.DataFrame(results), pd.DataFrame(cost_rows)


# ---------------------------------------------------------------------------
# Cross-instrument consistency
# ---------------------------------------------------------------------------


def build_consistency(results: pd.DataFrame, instruments: List[str]) -> pd.DataFrame:
    """Cross-instrument view: test-split, pooled-regime IC per (signal, horizon).

    A signal is only as good as its worst disagreement — the verdict makes
    single-pair flukes obvious: ``consistent`` needs every significant cell
    to share one sign with at least 3 instruments significant; ``mixed``
    means significant cells disagree in sign; anything else is ``weak``.
    """
    sub = results[
        (results["split"] == "test")
        & (results["regime"] == "ALL")
        & (results["reliable"])
    ]
    rows: List[Dict] = []
    for (sig, k), grp in sub.groupby(["signal", "horizon_bars"]):
        row: Dict = {"signal": sig, "horizon_bars": k}
        sig_signs: List[float] = []
        for inst in instruments:
            cell = grp[grp["instrument"] == inst]
            if len(cell):
                ic = float(cell["ic"].iloc[0])
                t = float(cell["t_nw"].iloc[0])
                row[f"ic_{inst}"] = ic
                row[f"t_{inst}"] = t
                if abs(t) >= T_SIGNIFICANT:
                    sig_signs.append(math.copysign(1.0, ic))
            else:
                row[f"ic_{inst}"] = float("nan")
                row[f"t_{inst}"] = float("nan")
        n_sig = len(sig_signs)
        same_sign = len(set(sig_signs)) <= 1
        if n_sig >= 3 and same_sign:
            verdict = "consistent"
        elif n_sig >= 2 and not same_sign:
            verdict = "mixed (sign flips across pairs — treat as noise)"
        else:
            verdict = "weak"
        row["n_significant"] = n_sig
        row["verdict"] = verdict
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["mean_abs_t"] = out[[f"t_{i}" for i in instruments]].abs().mean(axis=1)
        out = out.sort_values("mean_abs_t", ascending=False)
    return out


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def write_heatmap(
    instrument: str, results: pd.DataFrame, horizons: List[int]
) -> str:
    """Per-instrument heatmap of test-split ICs (unreliable cells = n/a)."""
    from phase2.regime import RegimeState
    from phase2.signals import SIGNAL_NAMES

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regimes = [r.value for r in RegimeState] + ["ALL"]
    sub_all = results[
        (results["instrument"] == instrument) & (results["split"] == "test")
    ]
    fig, axes = plt.subplots(
        1, len(regimes), figsize=(4.2 * len(regimes), 4.5), squeeze=False
    )
    im = None
    for ax, regime_name in zip(axes[0], regimes):
        sub = sub_all[sub_all["regime"] == regime_name]
        grid = np.full((len(SIGNAL_NAMES), len(horizons)), np.nan)
        annot = [["n/a"] * len(horizons) for _ in SIGNAL_NAMES]
        for r_i, sig in enumerate(SIGNAL_NAMES):
            for c_i, k in enumerate(horizons):
                cell = sub[(sub["signal"] == sig) & (sub["horizon_bars"] == k)]
                if len(cell) and cell["reliable"].iloc[0]:
                    ic = float(cell["ic"].iloc[0])
                    grid[r_i, c_i] = ic
                    star = "*" if cell["significant"].iloc[0] else ""
                    annot[r_i][c_i] = f"{ic:+.2f}{star}"
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-0.15, vmax=0.15, aspect="auto")
        ax.set_xticks(range(len(horizons)), [str(k) for k in horizons])
        ax.set_yticks(range(len(SIGNAL_NAMES)), list(SIGNAL_NAMES))
        ax.set_title(regime_name, fontsize=10)
        ax.set_xlabel("horizon (bars)")
        for r_i in range(len(SIGNAL_NAMES)):
            for c_i in range(len(horizons)):
                ax.text(
                    c_i, r_i, annot[r_i][c_i],
                    ha="center", va="center", fontsize=7, color="black",
                )
    fig.suptitle(
        f"{instrument} {GRANULARITY} — TEST-split IC by signal × horizon × "
        f"regime (* = |t_NW| ≥ {T_SIGNIFICANT:.0f}; n/a = n < {MIN_OBS})",
        fontsize=12,
    )
    if im is not None:
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="IC")
    path = os.path.join(OUTPUT_DIR, f"ic_heatmap_{instrument}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def write_markdown(
    results: pd.DataFrame,
    costs: pd.DataFrame,
    consistency: pd.DataFrame,
    instruments: List[str],
    years: float,
) -> str:
    """Write the pasteable markdown report; return its path."""
    lines: List[str] = []
    lines.append("# IC-by-Horizon Study")
    lines.append("")
    lines.append(
        f"Instruments: {', '.join(instruments)} | granularity {GRANULARITY} | "
        f"{years:g} years | train/test {TRAIN_FRAC:.0%}/{1 - TRAIN_FRAC:.0%} "
        f"chronological | NW lag = horizon | significance |t| ≥ "
        f"{T_SIGNIFICANT:g} | min n = {MIN_OBS}"
    )
    lines.append("")

    lines.append("## Top signals by TEST IC per instrument")
    lines.append("")
    lines.append("Only reliable (n ≥ 100) and NW-significant cells are shown.")
    lines.append("")
    header = (
        "| instrument | signal | horizon | regime | test IC | t_NW | n | "
        "train IC | IC non-overlap (t) |"
    )
    lines.append(header)
    lines.append("|" + "---|" * 9)
    test = results[
        (results["split"] == "test")
        & results["reliable"]
        & results["significant"]
    ].copy()
    test["abs_ic"] = test["ic"].abs()
    train = results[results["split"] == "train"]
    for inst in instruments:
        top = test[test["instrument"] == inst].sort_values(
            "abs_ic", ascending=False
        ).head(8)
        for _, row in top.iterrows():
            tr = train[
                (train["instrument"] == inst)
                & (train["signal"] == row["signal"])
                & (train["horizon_bars"] == row["horizon_bars"])
                & (train["regime"] == row["regime"])
            ]
            train_ic = float(tr["ic"].iloc[0]) if len(tr) else float("nan")
            lines.append(
                f"| {inst} | {row['signal']} | {row['horizon_bars']} | "
                f"{row['regime']} | {row['ic']:+.4f} | {row['t_nw']:+.2f} | "
                f"{row['n_obs']} | {train_ic:+.4f} | "
                f"{row['ic_nonoverlap']:+.4f} ({row['t_nonoverlap']:+.2f}) |"
            )
    lines.append("")

    lines.append("## Cross-instrument consistency (TEST split, all regimes pooled)")
    lines.append("")
    lines.append(
        "A signal ranking highly on one pair and poorly on the others is "
        "noise — the verdict column makes that explicit."
    )
    lines.append("")
    ic_cols = " | ".join(f"IC {i}" for i in instruments)
    lines.append(f"| signal | horizon | {ic_cols} | # significant | verdict |")
    lines.append("|" + "---|" * (len(instruments) + 4))
    for _, row in consistency.iterrows():
        ics = " | ".join(
            (
                f"{row[f'ic_{i}']:+.3f}"
                + ("*" if abs(row.get(f"t_{i}", 0.0)) >= T_SIGNIFICANT else "")
            )
            if not math.isnan(row[f"ic_{i}"]) else "n/a"
            for i in instruments
        )
        lines.append(
            f"| {row['signal']} | {row['horizon_bars']} | {ics} | "
            f"{row['n_significant']} | {row['verdict']} |"
        )
    lines.append("")
    lines.append("`*` = NW-significant on that instrument.")
    lines.append("")

    lines.append("## Cost per horizon (mean |forward move| vs 2-pip round trip)")
    lines.append("")
    lines.append("| instrument | horizon (bars) | hours | mean abs move (pips) | ×2-pip cost |")
    lines.append("|" + "---|" * 5)
    for _, row in costs.iterrows():
        lines.append(
            f"| {row['instrument']} | {int(row['horizon_bars'])} | "
            f"{row['horizon_hours']:.1f} | {row['mean_abs_fwd_pips']:.2f} | "
            f"{row['cost_ratio_vs_2pip']:.2f}x |"
        )
    lines.append("")

    path = os.path.join(OUTPUT_DIR, "horizon_study_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="IC-by-horizon study")
    parser.add_argument("--years", type=float, default=DEFAULT_YEARS)
    parser.add_argument("--instruments", type=str, default=DEFAULT_INSTRUMENTS)
    args = parser.parse_args()
    instruments = [s.strip() for s in args.instruments.split(",") if s.strip()]

    # Engine modules log per-bar regime transitions etc. — silence them
    for name in ["phase2.signals", "phase2.regime", "phase2.data"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results: List[pd.DataFrame] = []
    all_costs: List[pd.DataFrame] = []
    for instrument in instruments:
        try:
            df = fetch_candles(instrument, args.years)
            if len(df) < BURN_IN_BARS + max(HORIZONS) + MIN_OBS:
                print(f"[{instrument}] SKIPPED — only {len(df)} bars")
                continue
            scored = replay_scores(instrument, df)
            res, cost = analyze_instrument(instrument, df, scored)
            all_results.append(res)
            all_costs.append(cost)
            heat = write_heatmap(instrument, res, HORIZONS)
            print(f"[{instrument}] heatmap → {heat}")
        except Exception as exc:
            print(f"[{instrument}] FAILED: {exc!r}")

    if not all_results:
        print("No instruments analyzed — nothing to report.")
        sys.exit(1)

    results = pd.concat(all_results, ignore_index=True)
    costs = pd.concat(all_costs, ignore_index=True)
    consistency = build_consistency(results, instruments)

    summary_csv = os.path.join(OUTPUT_DIR, "horizon_study_summary.csv")
    results.to_csv(summary_csv, index=False)
    consistency_csv = os.path.join(OUTPUT_DIR, "horizon_study_consistency.csv")
    consistency.to_csv(consistency_csv, index=False)
    cost_csv = os.path.join(OUTPUT_DIR, "horizon_cost.csv")
    costs.to_csv(cost_csv, index=False)
    report_md = write_markdown(results, costs, consistency, instruments, args.years)

    print(f"\nWrote {summary_csv}")
    print(f"Wrote {consistency_csv}")
    print(f"Wrote {cost_csv}")
    print(f"Wrote {report_md}\n")

    ranked = results[
        (results["split"] == "test")
        & results["reliable"]
        & results["significant"]
    ].copy()
    ranked["abs_ic"] = ranked["ic"].abs()
    top = ranked.sort_values("abs_ic", ascending=False).head(5)
    print(
        "Top 5 signal-horizon-regime combinations by TEST |IC| "
        "(reliable + NW-significant only; signed IC shown):"
    )
    if len(top) == 0:
        print(
            "  none — no cell passed both the n ≥ 100 and |t_NW| ≥ 2 "
            "filters on the test split."
        )
    for _, row in top.iterrows():
        print(
            f"  {row['instrument']:<8} {row['signal']:<18} "
            f"k={row['horizon_bars']:<3} {row['regime']:<15} "
            f"IC={row['ic']:+.4f}  t_NW={row['t_nw']:+.2f}  n={row['n_obs']}"
        )
    n_unreliable = int((~results["reliable"]).sum())
    n_insig = int(
        (results["reliable"] & ~results["significant"]).sum()
    )
    print(
        f"\n{n_unreliable} cell(s) unreliable (n<{MIN_OBS}); "
        f"{n_insig} reliable cell(s) not significant (|t_NW|<{T_SIGNIFICANT:g}) "
        "— see summary CSV."
    )


if __name__ == "__main__":
    main()
