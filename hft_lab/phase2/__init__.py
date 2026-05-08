"""Phase 2 — Autonomous trading engine for hft_lab.

Submodules
----------
data       : Multi-scale rolling data buffers and historical warm-up.
signals    : Six-dimensional signal engine (MACD, RSI, Bollinger, VWAP,
             OrderBook, RelativeStrength).
regime     : Market regime detection via ADX and Hurst exponent.
ensemble   : Kalman-filter ensemble weighting and IC tracking.
sentiment  : Async news sentiment analysis via NewsAPI.
sizing     : GARCH volatility sizing, ATR stops, spread cost analysis.
execution  : Order execution, position tracking, trade logging.
session    : Market calendar and main session lifecycle management.
engine     : Top-level orchestrator and entry point.
"""
