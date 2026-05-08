# CatChaseMouseChaseCheese
## HFT Adversarial Finance Simulation — Phase 1: Infrastructure

---

## Prerequisites

- Python 3.10 or higher
- A free Alpaca Markets account (paper or live)
- Git

---

## Step 1 — Get Your Alpaca API Keys

1. Go to https://alpaca.markets and create a free account
2. Navigate to your dashboard
3. Click **Paper Trading** in the left sidebar
4. Click **Your API Keys** → **Generate New Key**
5. Copy both the API Key and Secret Key — the secret is only shown once
6. If testing on a live funded account: repeat from the **Live Trading** section instead

---

## Step 2 — Clone and Install

```bash
git clone https://github.com/TheBearInternal/CatChaseMouseChaseCheese.git
cd CatChaseMouseChaseCheese
pip install -r requirements.txt
```

---

## Step 3 — Configure Environment

```bash
cp .env.template .env
```

Open `.env` in any text editor and fill in the following:

```
# Alpaca Credentials
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here

# Paper trading (default):
ALPACA_BASE_URL=https://paper-api.alpaca.markets
# Live trading (only when ready):
# ALPACA_BASE_URL=https://api.alpaca.markets

ALPACA_DATA_URL=https://data.alpaca.markets
NEWS_API_KEY=your_newsapi_key_here

# Trading symbols
PRIMARY_SYMBOL=AAPL
BENCHMARK_SYMBOL=SPY

# Behavioral randomization seed — any integer
# Change this each session for natural variation
BEHAVIOR_SEED=42

# Position sizing
ACCOUNT_LIMIT=500
MAX_RISK_PER_TRADE_PCT=0.02
POSITION_MODE=dynamic
FIXED_POSITION_SIZE=100
MIN_POSITION_SIZE=1

# Spread and realism
SPREAD_ADJUSTMENT_ENABLED=true
SLIPPAGE_FACTOR=0.0001
MIN_PROFIT_SPREAD_MULTIPLE=3
```

---

## Step 4 — Verify Config Loads

```bash
python -c "from config import Config; c = Config(); print('Config loaded. Live account:', c.is_live())"
```

Expected output:
```
Config loaded. Live account: False
```

---

## Step 5 — Run Latency Baseline

```bash
python -m phase1.latency
```

This sends 100 requests to Alpaca and measures your round-trip latency.

Expected output: a formatted table showing min, max, mean, median,
standard deviation, 95th and 99th percentile latency in milliseconds.

Two files are saved automatically:
- /logs/latency_baseline.png — histogram of your latency distribution
- /logs/latency_baseline.csv — raw measurements with timestamps

Note your mean latency — it is your execution baseline for the project.
If mean exceeds 100ms a warning will appear.

---

## Step 6 — Run WebSocket Connection

```bash
python -m phase1.connection
```

Expected output:
- Connection confirmed to Alpaca stream
- Subscription confirmed for PRIMARY_SYMBOL and BENCHMARK_SYMBOL
- Live tick data printing to console as market data arrives
- Health check status logged every 30 seconds

To stop: press CTRL+C — clean shutdown will be logged.

Note: If ALPACA_BASE_URL points to the live endpoint a visible
WARNING will appear at startup confirming you are connected to
a funded account.

---

## About BEHAVIOR_SEED

BEHAVIOR_SEED is an integer that seeds the behavioral randomization
system. It determines:
- How long the bot waits after market open before its first action
- The bot's activity pattern throughout the trading day
- The degree of timing jitter applied to API calls
- The variance applied to order sizes

Using the same seed produces the same behavioral profile every run.
Change it between sessions for natural variation day to day.
Recommended: use a different number each week.

---

## Switching Between Paper and Live

To switch to a live funded account change one line in your .env:

```
ALPACA_BASE_URL=https://api.alpaca.markets
```

The system will automatically detect this and display a WARNING
at startup. All risk management, position limits, and stop losses
remain active regardless of account type.

---

## Project Structure

```
hft_lab/
├── .env.template         # Copy to .env and fill in credentials
├── .gitignore
├── requirements.txt
├── config.py             # Typed config loader and validator
├── logger.py             # Structured logging with color console output
├── phase1/
│   ├── behavior.py       # Behavioral randomization utilities
│   ├── connection.py     # WebSocket connection manager
│   └── latency.py        # Latency baseline measurement
└── README.md
```

---

## Logs

All logs are written to the `/logs` directory which is created
automatically on first run. Logs rotate daily with 7-day retention.
The /logs directory is excluded from git.
