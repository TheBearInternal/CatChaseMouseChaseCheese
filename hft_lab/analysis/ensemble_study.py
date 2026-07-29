#!/usr/bin/env python3
"""Ensemble IC study + cost-aware backtest with parameter sweep.

Builds on ``horizon_study.py`` (reuses its cached candle CSVs, replay
machinery, and Newey-West statistics) and the engine's own
``KalmanEnsemble`` / ``EnsembleDecision._apply_regime_priors`` /
``InformationCoefficient`` classes so the combined score is computed with
exactly the live engine's math (minus sentiment, which has no offline feed).

Parts
-----
1. Ensemble IC: regime-conditional Kalman weights trained on the TRAIN split
   only (chronological walk, IC target = config.ic_forward_bars forward
   return, exactly like live continuous scoring), applied to the TEST split.
   Per instrument × horizon × regime: ensemble IC with NW t-stat, n, and
   effective sample size (n // horizon, the non-overlapping equivalent),
   reported next to the best single-signal IC in the same cell.
2. Pairwise signal correlation matrix per regime (pooled across instruments),
   flagging pairs with |rho| > 0.7 — highly correlated signals mean the
   six-signal ensemble is effectively fewer.
3. Cost-aware backtest on the TEST split: enter when |ensemble| >= threshold
   in a tradeable regime, ATR(14)-based stop and TP at the swept ratio,
   2-pip round-trip cost, one position at a time, same-bar stop+TP resolved
   conservatively as stop-first.  Sweep timeframe {15m, 1h, 4h} ×
   threshold {0.10, 0.15, 0.20, 0.30} × RR {1.0, 1.5, 2.0, 3.0}.
4. Honesty: the sweep is a multiple-comparison exercise — the report states
   the number of configurations, flags the best cell as an upper bound, and
   includes a matched-frequency random-entry baseline (200 seeded trials).

Simplifications vs live (stated in the report): no sentiment modifier, no
session/rollover/spread gates, flat per-regime threshold, constant 2-pip
cost, fills exactly at stop/TP levels.

Usage::

    cd hft_lab && python3 analysis/ensemble_study.py \
        [--years 3] [--instruments GBP_USD,EUR_USD,USD_JPY,AUD_USD]
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from horizon_study import (
    BURN_IN_BARS,
    DEFAULT_INSTRUMENTS,
    DEFAULT_YEARS,
    GRANULARITY,
    HORIZONS,
    MIN_OBS,
    OUTPUT_DIR,
    ROUND_TRIP_COST_PIPS,
    T_SIGNIFICANT,
    TRAIN_FRAC,
    cell_stats,
    compute_ic,
    fetch_candles,
    pip_size,
    replay_scores,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

TIMEFRAMES: Dict[str, int] = {"15m": 1, "1h": 4, "4h": 16}   # M15 aggregation
THRESHOLDS: List[float] = [0.10, 0.15, 0.20, 0.30]
RR_RATIOS: List[float] = [1.0, 1.5, 2.0, 3.0]
ATR_PERIOD_BT: int = 14
CORR_FLAG: float = 0.70
BASELINE_TRIALS: int = 200
RNG_SEED: int = 42
MIN_TRADES: int = 20                  # configs below this are noise


# ---------------------------------------------------------------------------
# Timeframe aggregation
# ---------------------------------------------------------------------------


def aggregate(df: pd.DataFrame, factor: int) -> pd.DataFrame:
    """Aggregate M15 bars into ``factor``-bar candles (positional grouping)."""
    if factor == 1:
        return df.reset_index(drop=True)
    n = (len(df) // factor) * factor
    d = df.iloc[:n]
    g = np.arange(n) // factor
    out = pd.DataFrame({
        "timestamp": d["timestamp"].groupby(g).first().values,
        "open": d["open"].groupby(g).first().values,
        "high": d["high"].groupby(g).max().values,
        "low": d["low"].groupby(g).min().values,
        "close": d["close"].groupby(g).last().values,
        "volume": d["volume"].groupby(g).sum().values,
    })
    out["vwap_contribution"] = out["close"] * out["volume"]
    return out


def wilder_atr(df: pd.DataFrame, period: int = ATR_PERIOD_BT) -> np.ndarray:
    """Causal Wilder ATR series — atr[t] uses bars <= t only."""
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    n = len(df)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    tr[1:] = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
    )
    return pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().values


# ---------------------------------------------------------------------------
# Kalman training (train split only — mirrors live continuous scoring)
# ---------------------------------------------------------------------------


def train_kalman_weights(
    scored: pd.DataFrame, closes: np.ndarray, split_bar: int
) -> Dict[str, np.ndarray]:
    """Walk the train split chronologically and return per-regime weight vectors.

    Feeds the engine's InformationCoefficient with (score, k-forward-return)
    pairs at k = config.ic_forward_bars and updates the engine's
    KalmanEnsemble for the regime at each bar, exactly like the live
    bar-level scoring path.  Weights are returned already passed through
    EnsembleDecision._apply_regime_priors, i.e. the exact vectors the live
    decide() would dot with the signal scores.
    """
    from config import config
    from phase2.ensemble import EnsembleDecision, InformationCoefficient, KalmanEnsemble
    from phase2.regime import RegimeState
    from phase2.signals import SIGNAL_NAMES

    k = max(1, config.ic_forward_bars)
    n = len(closes)
    fwd = np.full(n, np.nan)
    fwd[: n - k] = (closes[k:] - closes[:-k]) / closes[:-k]

    kal = KalmanEnsemble()
    ic = InformationCoefficient()

    bar_idx = scored["bar"].values
    regimes = scored["regime"].values
    score_mat = scored[list(SIGNAL_NAMES)].values

    for row_i in range(len(scored)):
        b = bar_idx[row_i]
        if b + k > split_bar:          # leak-free: forward window inside train
            break
        ret = fwd[b]
        if np.isnan(ret):
            continue
        for s_i, name in enumerate(SIGNAL_NAMES):
            ic.update(name, float(score_mat[row_i, s_i]), float(ret))
        vec = np.array([ic.get_ic(nm) for nm in SIGNAL_NAMES])
        kal.update(vec, RegimeState(regimes[row_i]))

    adjusted: Dict[str, np.ndarray] = {}
    for r in RegimeState:
        w = kal.get_weights(r)
        adj = EnsembleDecision._apply_regime_priors(w, r)
        adjusted[r.value] = np.array([adj[nm] for nm in SIGNAL_NAMES])
    return adjusted


def ensemble_scores(
    scored: pd.DataFrame, weights_by_regime: Dict[str, np.ndarray]
) -> np.ndarray:
    """Per-bar ensemble score = dot(regime-adjusted weights, signal scores)."""
    from phase2.signals import SIGNAL_NAMES

    score_mat = scored[list(SIGNAL_NAMES)].values
    regimes = scored["regime"].values
    out = np.empty(len(scored))
    for i in range(len(scored)):
        out[i] = float(np.dot(weights_by_regime[regimes[i]], score_mat[i]))
    return out


# ---------------------------------------------------------------------------
# Part 1 — ensemble IC vs best single signal (test split)
# ---------------------------------------------------------------------------


def ensemble_ic_table(
    instrument: str,
    df: pd.DataFrame,
    scored: pd.DataFrame,
    ens: np.ndarray,
) -> pd.DataFrame:
    from phase2.regime import RegimeState
    from phase2.signals import SIGNAL_NAMES

    closes = df["close"].values.astype(float)
    n = len(df)
    split_bar = int(n * TRAIN_FRAC)
    bar_idx = scored["bar"].values
    regime_col = scored["regime"].values

    rows: List[Dict] = []
    for k in HORIZONS:
        fwd = np.full(n, np.nan)
        fwd[: n - k] = (closes[k:] - closes[:-k]) / closes[:-k]
        valid = ~np.isnan(fwd[bar_idx]) & (bar_idx >= split_bar)
        for regime_name in [r.value for r in RegimeState] + ["ALL"]:
            mask = valid if regime_name == "ALL" else (
                valid & (regime_col == regime_name)
            )
            actuals = fwd[bar_idx[mask]]
            stats = cell_stats(ens[mask], actuals, k)
            # Best single signal in the same cell
            best_sig, best_ic = "", 0.0
            for sig in SIGNAL_NAMES:
                s_ic = compute_ic(scored[sig].values[mask], actuals)
                if abs(s_ic) > abs(best_ic):
                    best_sig, best_ic = sig, s_ic
            rows.append({
                "instrument": instrument,
                "horizon_bars": k,
                "regime": regime_name,
                "ensemble_ic": stats["ic"],
                "t_nw": stats["t_nw"],
                "n_obs": stats["n_obs"],
                "n_eff": stats["n_obs"] // k,
                "best_single_signal": best_sig,
                "best_single_ic": best_ic,
                "ensemble_beats_single": abs(stats["ic"]) > abs(best_ic),
                "significant": abs(stats["t_nw"]) >= T_SIGNIFICANT,
                "reliable": stats["n_obs"] >= MIN_OBS,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 2 — signal correlation per regime (pooled across instruments)
# ---------------------------------------------------------------------------


def correlation_matrices(all_scored: List[pd.DataFrame]) -> pd.DataFrame:
    from phase2.regime import RegimeState
    from phase2.signals import SIGNAL_NAMES

    pooled = pd.concat(all_scored, ignore_index=True)
    rows: List[Dict] = []
    for regime_name in [r.value for r in RegimeState] + ["ALL"]:
        sub = pooled if regime_name == "ALL" else pooled[
            pooled["regime"] == regime_name
        ]
        if len(sub) < MIN_OBS:
            continue
        for i, a in enumerate(SIGNAL_NAMES):
            for j, b in enumerate(SIGNAL_NAMES):
                if j <= i:
                    continue
                rho = compute_ic(sub[a].values, sub[b].values)
                rows.append({
                    "regime": regime_name,
                    "signal_a": a,
                    "signal_b": b,
                    "correlation": rho,
                    "n": len(sub),
                    "flagged": abs(rho) > CORR_FLAG,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 3 — cost-aware backtest (test split)
# ---------------------------------------------------------------------------


def run_backtest(
    df: pd.DataFrame,
    scored: pd.DataFrame,
    ens: np.ndarray,
    pip: float,
    threshold: float,
    rr: float,
    tradeable: frozenset,
    atr_mult: float,
    entry_bars: Optional[List[int]] = None,
    entry_sides: Optional[List[int]] = None,
) -> List[float]:
    """One-position-at-a-time walk over the test split; returns net pips/trade.

    When ``entry_bars``/``entry_sides`` are given (random baseline), signal
    and regime gating are bypassed and those entries are used instead.
    """
    closes = df["close"].values.astype(float)
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    atr = wilder_atr(df)
    n = len(df)
    split_bar = int(n * TRAIN_FRAC)

    bar_idx = scored["bar"].values
    regimes = scored["regime"].values
    # Map absolute bar -> row in scored (scored starts at BURN_IN_BARS)
    row_of_bar = {int(b): i for i, b in enumerate(bar_idx)}

    trades: List[float] = []
    forced = entry_bars is not None
    forced_iter = iter(sorted(zip(entry_bars or [], entry_sides or [])))
    next_forced = next(forced_iter, None)

    b = split_bar
    while b < n - 1:
        side = 0
        if forced:
            if next_forced is None:
                break
            if b < next_forced[0]:
                b += 1
                continue
            side = next_forced[1]
            next_forced = next(forced_iter, None)
        else:
            row = row_of_bar.get(b)
            if row is None:
                b += 1
                continue
            if regimes[row] in tradeable and abs(ens[row]) >= threshold:
                side = 1 if ens[row] > 0 else -1
            if side == 0:
                b += 1
                continue
        if atr[b] <= 0:
            b += 1
            continue

        entry = closes[b]
        stop_d = atr[b] * atr_mult
        tp_d = stop_d * rr
        stop = entry - side * stop_d
        tp = entry + side * tp_d

        exit_price = None
        j = b + 1
        while j < n:
            if side == 1:
                if lows[j] <= stop:          # stop checked first: conservative
                    exit_price = stop
                    break
                if highs[j] >= tp:
                    exit_price = tp
                    break
            else:
                if highs[j] >= stop:
                    exit_price = stop
                    break
                if lows[j] <= tp:
                    exit_price = tp
                    break
            j += 1
        if exit_price is None:
            exit_price = closes[n - 1]
            j = n - 1
        trades.append(side * (exit_price - entry) / pip - ROUND_TRIP_COST_PIPS)
        b = j + 1
    return trades


def trade_metrics(trades: List[float], years_test: float) -> Dict[str, float]:
    n = len(trades)
    if n == 0:
        return {
            "trades": 0, "win_rate": float("nan"),
            "mean_pips": float("nan"), "total_pips": 0.0,
            "max_dd_pips": float("nan"), "sharpe": float("nan"),
        }
    arr = np.array(trades)
    cum = np.cumsum(arr)
    dd = float(np.max(np.maximum.accumulate(cum) - cum))
    std = float(np.std(arr))
    per_year = n / max(years_test, 1e-9)
    sharpe = (
        float(np.mean(arr)) / std * math.sqrt(per_year) if std > 1e-12
        else float("nan")
    )
    return {
        "trades": n,
        "win_rate": float(np.mean(arr > 0)),
        "mean_pips": float(np.mean(arr)),
        "total_pips": float(np.sum(arr)),
        "max_dd_pips": dd,
        "sharpe": sharpe,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble IC + cost-aware backtest")
    parser.add_argument("--years", type=float, default=DEFAULT_YEARS)
    parser.add_argument("--instruments", type=str, default=DEFAULT_INSTRUMENTS)
    args = parser.parse_args()
    instruments = [s.strip() for s in args.instruments.split(",") if s.strip()]

    for name in ["phase2.signals", "phase2.regime", "phase2.data", "phase2.ensemble"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    from config import config

    tradeable = config.tradeable_regimes or frozenset(
        {"TRENDING", "MEAN_REVERTING"}
    )
    atr_mult = config.atr_stop_multiplier
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = random.Random(RNG_SEED)

    ic_tables: List[pd.DataFrame] = []
    scored_15m_all: List[pd.DataFrame] = []
    sweep_rows: List[Dict] = []
    # Per-(instrument, timeframe) context kept for the baseline rerun
    bt_ctx: Dict[Tuple[str, str], Dict] = {}

    for instrument in instruments:
        try:
            base_df = fetch_candles(instrument, args.years)
        except Exception as exc:
            print(f"[{instrument}] fetch FAILED: {exc!r}")
            continue
        pip = pip_size(instrument)

        for tf_name, factor in TIMEFRAMES.items():
            df = aggregate(base_df, factor)
            if len(df) < BURN_IN_BARS + max(HORIZONS) + MIN_OBS:
                print(f"[{instrument} {tf_name}] SKIPPED — only {len(df)} bars")
                continue
            print(f"[{instrument} {tf_name}] {len(df)} bars — replaying signals")
            scored = replay_scores(f"{instrument}:{tf_name}", df)
            closes = df["close"].values.astype(float)
            split_bar = int(len(df) * TRAIN_FRAC)

            weights = train_kalman_weights(scored, closes, split_bar)
            ens = ensemble_scores(scored, weights)

            ts = pd.to_datetime(df["timestamp"])
            years_test = max(
                (ts.iloc[-1] - ts.iloc[split_bar]).days / 365.25, 1e-6
            )
            bt_ctx[(instrument, tf_name)] = {
                "df": df, "scored": scored, "ens": ens, "pip": pip,
                "years_test": years_test,
            }

            if tf_name == "15m":
                ic_tables.append(ensemble_ic_table(instrument, df, scored, ens))
                scored_15m_all.append(scored)

            for thr in THRESHOLDS:
                for rr in RR_RATIOS:
                    trades = run_backtest(
                        df, scored, ens, pip, thr, rr, tradeable, atr_mult
                    )
                    m = trade_metrics(trades, years_test)
                    sweep_rows.append({
                        "instrument": instrument,
                        "timeframe": tf_name,
                        "threshold": thr,
                        "risk_reward": rr,
                        **m,
                    })

    if not sweep_rows:
        print("Nothing analyzed — no data.")
        sys.exit(1)

    ensemble_ic = (
        pd.concat(ic_tables, ignore_index=True) if ic_tables else pd.DataFrame()
    )
    correlations = correlation_matrices(scored_15m_all) if scored_15m_all else pd.DataFrame()
    sweep = pd.DataFrame(sweep_rows)

    # ------------------------------------------------------------------
    # Random-entry baseline at matched frequency for the best config
    # ------------------------------------------------------------------
    n_configs = len(sweep)
    eligible = sweep[sweep["trades"] >= MIN_TRADES].copy()
    baseline_summary: Dict = {}
    if len(eligible):
        best = eligible.sort_values("sharpe", ascending=False).iloc[0]
        ctx = bt_ctx[(best["instrument"], best["timeframe"])]
        n_bars = len(ctx["df"])
        split_bar = int(n_bars * TRAIN_FRAC)
        totals: List[float] = []
        for _ in range(BASELINE_TRIALS):
            bars = rng.sample(
                range(split_bar, n_bars - 2), min(int(best["trades"]), n_bars - split_bar - 2)
            )
            sides = [rng.choice([1, -1]) for _ in bars]
            t = run_backtest(
                ctx["df"], ctx["scored"], ctx["ens"], ctx["pip"],
                0.0, best["risk_reward"], tradeable, atr_mult,
                entry_bars=bars, entry_sides=sides,
            )
            totals.append(float(np.sum(t)))
        totals_arr = np.array(totals)
        pctile = float(np.mean(totals_arr < best["total_pips"]) * 100.0)
        baseline_summary = {
            "config": (
                f"{best['instrument']} {best['timeframe']} "
                f"thr={best['threshold']:g} rr={best['risk_reward']:g}"
            ),
            "strategy_total_pips": float(best["total_pips"]),
            "baseline_mean_pips": float(totals_arr.mean()),
            "baseline_std_pips": float(totals_arr.std()),
            "strategy_percentile_vs_baseline": pctile,
        }

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    ensemble_ic.to_csv(os.path.join(OUTPUT_DIR, "ensemble_ic.csv"), index=False)
    correlations.to_csv(
        os.path.join(OUTPUT_DIR, "signal_correlations.csv"), index=False
    )
    sweep.to_csv(os.path.join(OUTPUT_DIR, "backtest_sweep.csv"), index=False)

    lines: List[str] = []
    lines.append("# Ensemble IC Study + Cost-Aware Backtest")
    lines.append("")
    lines.append(
        f"Instruments: {', '.join(instruments)} | base {GRANULARITY} | "
        f"{args.years:g}y | train/test {TRAIN_FRAC:.0%}/{1 - TRAIN_FRAC:.0%} "
        f"chronological | Kalman trained on train only "
        f"(IC target = {config.ic_forward_bars} bars, live math incl. regime "
        f"priors) | cost {ROUND_TRIP_COST_PIPS:g} pips RT | "
        f"tradeable regimes: {sorted(tradeable)}"
    )
    lines.append("")
    lines.append(
        "Simplifications vs live: no sentiment modifier, no session/rollover/"
        "spread gates, flat threshold across regimes, fills exactly at "
        "stop/TP, same-bar stop+TP resolved as stop-first (conservative)."
    )
    lines.append("")

    lines.append("## 1. Ensemble IC vs best single signal (TEST split, 15m)")
    lines.append("")
    lines.append(
        "n_eff = n // horizon (non-overlapping equivalent). Cells with "
        f"n < {MIN_OBS} or |t_NW| < {T_SIGNIFICANT:g} are omitted here "
        "(full grid in ensemble_ic.csv)."
    )
    lines.append("")
    lines.append(
        "| instrument | horizon | regime | ensemble IC | t_NW | n | n_eff | "
        "best single | single IC | ensemble wins? |"
    )
    lines.append("|" + "---|" * 10)
    shown = ensemble_ic[
        ensemble_ic["reliable"] & ensemble_ic["significant"]
    ] if len(ensemble_ic) else pd.DataFrame()
    for _, r in shown.iterrows():
        lines.append(
            f"| {r['instrument']} | {r['horizon_bars']} | {r['regime']} | "
            f"{r['ensemble_ic']:+.4f} | {r['t_nw']:+.2f} | {r['n_obs']} | "
            f"{r['n_eff']} | {r['best_single_signal']} | "
            f"{r['best_single_ic']:+.4f} | "
            f"{'yes' if r['ensemble_beats_single'] else 'no'} |"
        )
    if not len(shown):
        lines.append("| — no reliable + significant ensemble cells — " + "|" * 9)
    lines.append("")

    lines.append("## 2. Signal correlations per regime (pooled across instruments)")
    lines.append("")
    lines.append(
        f"Pairs with |rho| > {CORR_FLAG:g} are flagged — the ensemble is "
        "effectively fewer than six signals wherever these appear."
    )
    lines.append("")
    lines.append("| regime | signal A | signal B | rho | flagged |")
    lines.append("|" + "---|" * 5)
    if len(correlations):
        for _, r in correlations.sort_values(
            "correlation", key=lambda s: s.abs(), ascending=False
        ).iterrows():
            lines.append(
                f"| {r['regime']} | {r['signal_a']} | {r['signal_b']} | "
                f"{r['correlation']:+.3f} | {'**YES**' if r['flagged'] else ''} |"
            )
    lines.append("")

    lines.append("## 3. Backtest sweep (TEST split)")
    lines.append("")
    lines.append(
        f"Top 15 of {n_configs} configurations by Sharpe (configs with "
        f"< {MIN_TRADES} trades excluded from ranking; full grid in "
        "backtest_sweep.csv):"
    )
    lines.append("")
    lines.append(
        "| instrument | timeframe | threshold | RR | trades | win rate | "
        "mean pips | total pips | max DD | Sharpe |"
    )
    lines.append("|" + "---|" * 10)
    for _, r in eligible.sort_values("sharpe", ascending=False).head(15).iterrows():
        lines.append(
            f"| {r['instrument']} | {r['timeframe']} | {r['threshold']:g} | "
            f"{r['risk_reward']:g} | {r['trades']} | {r['win_rate']:.1%} | "
            f"{r['mean_pips']:+.2f} | {r['total_pips']:+.1f} | "
            f"{r['max_dd_pips']:.1f} | {r['sharpe']:+.2f} |"
        )
    lines.append("")

    lines.append("## 4. Honest interpretation")
    lines.append("")
    lines.append(
        f"- **{n_configs} configurations were tested.** Selecting the best "
        "after the fact is a multiple-comparison exercise: the top row above "
        "is an **upper bound on expected performance, not an expectation**. "
        "With this many draws, some configuration will look good by chance "
        "even on pure noise."
    )
    if baseline_summary:
        lines.append(
            f"- Random-entry baseline (matched trade count and RR, "
            f"{BASELINE_TRIALS} seeded trials) for the best config "
            f"({baseline_summary['config']}): baseline mean total "
            f"{baseline_summary['baseline_mean_pips']:+.1f} pips "
            f"(σ {baseline_summary['baseline_std_pips']:.1f}); the strategy's "
            f"{baseline_summary['strategy_total_pips']:+.1f} pips sits at the "
            f"{baseline_summary['strategy_percentile_vs_baseline']:.0f}th "
            "percentile of the random distribution. Below ~95 this result is "
            "indistinguishable from luck."
        )
    lines.append(
        "- Test-split ensemble ICs in section 1 are the cleaner evidence; "
        "the backtest adds path/cost effects but also adds selection bias "
        "from the sweep."
    )
    lines.append("")

    report_path = os.path.join(OUTPUT_DIR, "ensemble_study.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nWrote {os.path.join(OUTPUT_DIR, 'ensemble_ic.csv')}")
    print(f"Wrote {os.path.join(OUTPUT_DIR, 'signal_correlations.csv')}")
    print(f"Wrote {os.path.join(OUTPUT_DIR, 'backtest_sweep.csv')}")
    print(f"Wrote {report_path}\n")

    print(f"Configurations tested: {n_configs} (best is an upper bound)")
    if baseline_summary:
        print(
            f"Best config {baseline_summary['config']}: "
            f"{baseline_summary['strategy_total_pips']:+.1f} pips vs random "
            f"baseline {baseline_summary['baseline_mean_pips']:+.1f}±"
            f"{baseline_summary['baseline_std_pips']:.1f} "
            f"(strategy at {baseline_summary['strategy_percentile_vs_baseline']:.0f}th pct)"
        )
    if len(eligible):
        print("\nTop 5 configs by Sharpe:")
        for _, r in eligible.sort_values("sharpe", ascending=False).head(5).iterrows():
            print(
                f"  {r['instrument']:<8} {r['timeframe']:<4} "
                f"thr={r['threshold']:<5g} rr={r['risk_reward']:<4g} "
                f"trades={r['trades']:<5} win={r['win_rate']:.1%} "
                f"mean={r['mean_pips']:+.2f}p total={r['total_pips']:+.1f}p "
                f"sharpe={r['sharpe']:+.2f}"
            )


if __name__ == "__main__":
    main()
