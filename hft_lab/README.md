# hft_lab — Phase 1: Infrastructure & Behavioral Foundation

Phase 1 of a multi-phase algorithmic trading simulation project built on the
[Alpaca Markets](https://alpaca.markets) API.  This phase establishes connection
infrastructure, structured logging, REST latency profiling, and behavioral
randomization.  **No trading logic, signals, or order execution are present.**

---

## Project Structure

```
hft_lab/
├── .env.template        # Credential template — copy to .env and fill in
├── .gitignore
├── requirements.txt
├── config.py            # Environment loading + typed Config dataclass
├── logger.py            # Colored console + daily-rotating file logging
├── phase1/
│   ├── __init__.py
│   ├── connection.py    # Alpaca WebSocket connection manager
│   ├── latency.py       # REST API latency baseline measurement tool
│   └── behavior.py      # Behavioral randomization utilities
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Python 3.10 or later is required.

### 2. Configure credentials

```bash
cp .env.template .env
```

Open `.env` and fill in each value.  **Never commit `.env` to version control —
it is excluded by `.gitignore`.**

#### Obtaining Alpaca API Keys

1. Create a free account at [alpaca.markets](https://alpaca.markets).
2. In the dashboard, navigate to **Paper Trading → API Keys** and generate a
   key pair for safe development use.
3. For live trading, complete identity verification, then navigate to
   **Live Trading → API Keys**.  Keep live credentials separate from paper
   credentials.

#### Paper vs. Live endpoint

Set `ALPACA_BASE_URL` in `.env` to control which environment is used:

| Mode | `ALPACA_BASE_URL` |
|------|-------------------|
| **Paper** (default — no real money) | `https://paper-api.alpaca.markets` |
| **Live** (real funded account) | `https://api.alpaca.markets` |

When the live URL is configured, `connection.py` logs a clearly visible
`WARNING` at startup indicating a funded account is active.  Always develop
and test on paper first.

---

## Running

### WebSocket connection

Connects to Alpaca's real-time market data stream and logs incoming trades
and quotes for `PRIMARY_SYMBOL` and `BENCHMARK_SYMBOL`.

```bash
python -m phase1.connection
```

**What happens:**

1. A randomized startup delay (0 – 180 s, controlled by `BEHAVIOR_SEED`) is
   applied before connecting — simulating non-mechanical session timing.
2. The manager subscribes to live trades and quotes for both configured symbols.
3. Each incoming message is parsed and logged at INFO level.
4. A background health-check coroutine logs connection status every 30 seconds.
5. On any dropout, exponential backoff reconnection begins (1 s → 2 s → 4 s …
   capped at 60 s, maximum 10 attempts).

**Expected console output:**

```
2025-08-01T09:28:07 | INFO     | phase1.connection | Applying randomized startup delay: 63.42s
2025-08-01T09:29:10 | INFO     | phase1.connection | Connecting to Alpaca stream | base_url=https://paper-api.alpaca.markets symbols=[AAPL, SPY]
2025-08-01T09:29:11 | INFO     | phase1.connection | Stream active | subscribed to trades+quotes for AAPL and SPY
2025-08-01T09:29:11 | INFO     | phase1.connection | TRADE | symbol=AAPL price=195.43 timestamp=2025-08-01T13:29:11.123Z
2025-08-01T09:29:11 | INFO     | phase1.connection | QUOTE | symbol=AAPL bid=195.41 ask=195.44 timestamp=2025-08-01T13:29:11.124Z
2025-08-01T09:29:41 | INFO     | phase1.connection | Health | connected=True last_message=0.3s ago
```

Stop cleanly with **Ctrl+C** (SIGINT) or by sending SIGTERM.

---

### Latency baseline

Sends 100 sequential HTTP GET requests to the Alpaca `/v2/clock` endpoint and
reports round-trip statistics.

```bash
python -m phase1.latency
```

**Expected console output:**

```
+------------------------------------------+
|      Alpaca REST Latency Baseline        |
+------------------------------------------+
| Metric               Value               |
+------------------------------------------+
| Minimum                       12.847 ms |
| Maximum                       89.341 ms |
| Mean                          27.503 ms |
| Median                        24.916 ms |
| Std Dev                        9.872 ms |
| 95th Percentile               51.204 ms |
| 99th Percentile               78.619 ms |
+------------------------------------------+
```

**Output files written to `/logs/`:**

| File | Contents |
|------|----------|
| `latency_baseline.png` | Histogram of the full latency distribution with labeled axes and a timestamp in the title |
| `latency_baseline.csv` | All 100 raw measurements with per-request UTC timestamps |

If mean latency exceeds 100 ms, a `WARNING` is logged advising that execution
timing may be impacted and suggesting a host geographically closer to Alpaca's
infrastructure.

---

## BEHAVIOR_SEED

`BEHAVIOR_SEED` is an integer that deterministically controls all session-level
behavioral randomization:

| Parameter | Range | Effect |
|-----------|-------|--------|
| `session_start_offset` | 0 – 180 s | Delay after startup before first market action |
| `activity_rhythm` | per-minute weights | Probability of acting each minute (elevated at open/close, depressed at midday) |
| `polling_jitter_sigma` | 50 – 300 ms | Standard deviation of per-call API timing jitter |
| `order_size_variance` | 8 – 20 % | Half-width of uniform order size randomization |

**Recommended practice:**

- Use any positive integer.  Common starting values: `42`, `137`, `2718`, `31415`.
- Change the seed between sessions to vary timing patterns and avoid
  session-fingerprinting.
- Record which seeds were used if you need to reproduce a specific session's
  behavior for debugging.
- The same seed always produces identical parameters — reproducibility is
  guaranteed.

---

## Logging

All modules write to two sinks simultaneously:

- **Console** — color-coded by level (DEBUG grey, INFO green, WARNING yellow,
  ERROR red, CRITICAL bold red).
- **File** — `logs/hft_lab.log`, rotated daily at midnight, 7 days retained.

The `/logs` directory is created automatically on first run.

---

## Environment Variable Reference

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ALPACA_API_KEY` | str | — | Alpaca API key ID |
| `ALPACA_SECRET_KEY` | str | — | Alpaca secret key |
| `ALPACA_BASE_URL` | str | `https://paper-api.alpaca.markets` | Trading endpoint |
| `ALPACA_DATA_URL` | str | `https://data.alpaca.markets` | Data endpoint |
| `NEWS_API_KEY` | str | — | NewsAPI key |
| `PRIMARY_SYMBOL` | str | `AAPL` | Main symbol |
| `BENCHMARK_SYMBOL` | str | `SPY` | Benchmark symbol |
| `BEHAVIOR_SEED` | int | `42` | Behavioral randomization seed |
| `ACCOUNT_LIMIT` | float | `500` | Maximum capital exposed (USD) |
| `MAX_RISK_PER_TRADE_PCT` | float | `0.02` | Max fraction of account risked per trade |
| `POSITION_MODE` | str | `dynamic` | `dynamic` or `fixed` |
| `FIXED_POSITION_SIZE` | int | `100` | Shares per trade when `POSITION_MODE=fixed` |
| `MIN_POSITION_SIZE` | int | `1` | Absolute minimum position size |
| `SPREAD_ADJUSTMENT_ENABLED` | bool | `true` | Enable dynamic spread-based entry adjustment |
| `SLIPPAGE_FACTOR` | float | `0.0001` | Simulated market-impact fraction |
| `MIN_PROFIT_SPREAD_MULTIPLE` | float | `3` | Minimum profit-to-spread ratio to enter |
