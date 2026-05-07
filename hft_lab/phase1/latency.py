"""Alpaca REST API latency baseline measurement.

Sends :data:`REQUEST_COUNT` sequential HTTP GET requests to the Alpaca
``/v2/clock`` endpoint, measures each round-trip with ``time.perf_counter``
(nanosecond resolution), then:

* Prints a formatted summary table of key statistics to the console.
* Saves a labeled histogram to ``/logs/latency_baseline.png``.
* Saves all raw measurements with per-request timestamps to
  ``/logs/latency_baseline.csv``.
* Logs a WARNING if mean latency exceeds :data:`LATENCY_WARN_THRESHOLD_MS`.

Run with::

    python -m phase1.latency
"""
from __future__ import annotations

import asyncio
import csv
import os
import time
from datetime import datetime, timezone
from typing import NamedTuple

import aiohttp
import matplotlib.pyplot as plt
import numpy as np

from config import config
from logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUEST_COUNT: int = 100
LATENCY_WARN_THRESHOLD_MS: float = 100.0

CLOCK_ENDPOINT: str = "/v2/clock"

LOGS_DIR: str = "logs"
HISTOGRAM_FILENAME: str = "latency_baseline.png"
CSV_FILENAME: str = "latency_baseline.csv"

HISTOGRAM_BINS: int = 20
HISTOGRAM_DPI: int = 150
HISTOGRAM_FIGSIZE: tuple[int, int] = (10, 6)

PERCENTILE_95: float = 95.0
PERCENTILE_99: float = 99.0

# Console table column widths
TABLE_LABEL_WIDTH: int = 20
TABLE_VALUE_WIDTH: int = 14


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class RequestRecord(NamedTuple):
    """Single round-trip measurement.

    Attributes
    ----------
    index       : 1-based request sequence number.
    timestamp_utc : ISO-8601 UTC timestamp at moment the request was sent.
    latency_ms  : Measured round-trip duration in milliseconds.
    """

    index: int
    timestamp_utc: str
    latency_ms: float


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


async def _measure_single(
    session: aiohttp.ClientSession,
    index: int,
) -> RequestRecord:
    """Send one GET to /v2/clock and measure the round-trip time.

    Uses ``time.perf_counter`` for the highest-resolution timing available on
    the platform.  The clock is read immediately before the request is sent and
    immediately after the response body is consumed, so network + server
    processing time are both captured.

    Args:
        session: Authenticated ``aiohttp.ClientSession`` to reuse.
        index:   1-based request number used for progress logging.

    Returns:
        ``RequestRecord`` with the UTC send timestamp and round-trip latency.
    """
    url = config.alpaca_base_url.rstrip("/") + CLOCK_ENDPOINT
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    t_start = time.perf_counter()
    async with session.get(url) as response:
        await response.read()  # consume body so connection is not left dangling
    latency_ms = (time.perf_counter() - t_start) * 1000.0

    logger.debug(f"Request {index:>3}/{REQUEST_COUNT}: {latency_ms:.3f} ms")
    return RequestRecord(index=index, timestamp_utc=timestamp_utc, latency_ms=latency_ms)


async def _run_measurements() -> list[RequestRecord]:
    """Execute all latency measurements sequentially over a single HTTP session.

    A single ``aiohttp.ClientSession`` is reused across all requests to keep
    connection-establishment overhead out of the measurement window.

    Returns:
        Ordered list of :class:`RequestRecord` instances, one per request.
    """
    headers = {
        "APCA-API-KEY-ID": config.alpaca_api_key,
        "APCA-API-SECRET-KEY": config.alpaca_secret_key,
    }

    records: list[RequestRecord] = []
    logger.info(
        f"Measuring {REQUEST_COUNT} sequential round-trips to "
        f"{config.alpaca_base_url}{CLOCK_ENDPOINT}"
    )

    async with aiohttp.ClientSession(headers=headers) as session:
        for i in range(1, REQUEST_COUNT + 1):
            record = await _measure_single(session, i)
            records.append(record)

    return records


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _compute_stats(latencies: list[float]) -> dict[str, float]:
    """Compute descriptive statistics from raw latency measurements.

    Args:
        latencies: List of round-trip latencies in milliseconds.

    Returns:
        Dict with keys ``min``, ``max``, ``mean``, ``median``,
        ``std``, ``p95``, ``p99``.
    """
    arr = np.array(latencies, dtype=float)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p95": float(np.percentile(arr, PERCENTILE_95)),
        "p99": float(np.percentile(arr, PERCENTILE_99)),
    }


# ---------------------------------------------------------------------------
# Output: console table
# ---------------------------------------------------------------------------


def _print_summary(stats: dict[str, float]) -> None:
    """Print a formatted latency statistics table to stdout.

    Args:
        stats: Dict produced by :func:`_compute_stats`.
    """
    total_width = TABLE_LABEL_WIDTH + TABLE_VALUE_WIDTH + 5
    border = "+" + "-" * total_width + "+"
    title = "Alpaca REST Latency Baseline"

    rows: list[tuple[str, float]] = [
        ("Minimum",         stats["min"]),
        ("Maximum",         stats["max"]),
        ("Mean",            stats["mean"]),
        ("Median",          stats["median"]),
        ("Std Dev",         stats["std"]),
        ("95th Percentile", stats["p95"]),
        ("99th Percentile", stats["p99"]),
    ]

    print(border)
    print(f"| {title:^{total_width}} |")
    print(border)
    print(f"| {'Metric':<{TABLE_LABEL_WIDTH}} {'Value':>{TABLE_VALUE_WIDTH + 3}} |")
    print(border)
    for label, value in rows:
        print(f"| {label:<{TABLE_LABEL_WIDTH}} {value:>{TABLE_VALUE_WIDTH}.3f} ms |")
    print(border)


# ---------------------------------------------------------------------------
# Output: histogram
# ---------------------------------------------------------------------------


def _save_histogram(latencies: list[float], run_timestamp: str) -> str:
    """Save a histogram of the latency distribution to /logs.

    Args:
        latencies:     Raw latency measurements in milliseconds.
        run_timestamp: Human-readable UTC timestamp included in the chart title.

    Returns:
        Absolute path to the written PNG file.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    output_path = os.path.join(LOGS_DIR, HISTOGRAM_FILENAME)

    fig, ax = plt.subplots(figsize=HISTOGRAM_FIGSIZE)
    ax.hist(latencies, bins=HISTOGRAM_BINS, color="steelblue", edgecolor="black", alpha=0.85)
    ax.set_xlabel("Round-Trip Latency (ms)", fontsize=13)
    ax.set_ylabel("Request Count", fontsize=13)
    ax.set_title(
        f"Alpaca REST API Latency Distribution\n"
        f"{REQUEST_COUNT} sequential requests  —  {run_timestamp}",
        fontsize=14,
    )
    ax.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=HISTOGRAM_DPI)
    plt.close(fig)

    logger.info(f"Histogram saved → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Output: CSV
# ---------------------------------------------------------------------------


def _save_csv(records: list[RequestRecord]) -> str:
    """Save all raw measurements with per-request timestamps to /logs.

    Columns: ``index``, ``timestamp_utc``, ``latency_ms``.

    Args:
        records: Ordered list of :class:`RequestRecord` instances.

    Returns:
        Absolute path to the written CSV file.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
    output_path = os.path.join(LOGS_DIR, CSV_FILENAME)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "timestamp_utc", "latency_ms"])
        for rec in records:
            writer.writerow([rec.index, rec.timestamp_utc, f"{rec.latency_ms:.6f}"])

    logger.info(f"CSV saved → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _main() -> None:
    """Async entry point: measure, report, save, and warn if needed."""
    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info(f"Latency baseline starting — {REQUEST_COUNT} requests — {run_timestamp}")

    records = await _run_measurements()
    latencies = [r.latency_ms for r in records]

    stats = _compute_stats(latencies)
    _print_summary(stats)
    _save_histogram(latencies, run_timestamp)
    _save_csv(records)

    if stats["mean"] > LATENCY_WARN_THRESHOLD_MS:
        logger.warning(
            f"Mean latency {stats['mean']:.2f}ms exceeds the "
            f"{LATENCY_WARN_THRESHOLD_MS:.0f}ms threshold — "
            "execution timing may be impacted. Consider a host closer to Alpaca's servers."
        )

    logger.info("Latency baseline complete.")


if __name__ == "__main__":
    asyncio.run(_main())
