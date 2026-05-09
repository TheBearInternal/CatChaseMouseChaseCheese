"""Structured logging for hft_lab.

Provides a single ``get_logger(name)`` factory used by every module.  Each
logger writes to two sinks simultaneously:

* **Console** — colorized by level via ``colorlog``.
* **Rotating file** — plain-text in ``/logs/hft_lab.log`` with daily rotation
  and 7-day retention.

Color scheme
------------
DEBUG    → grey
INFO     → green
WARNING  → yellow
ERROR    → red
CRITICAL → bold red

Every log entry carries an ISO-8601 timestamp, level name, originating module
name, and the message body.

The ``/logs`` directory is created automatically if it does not exist.
"""
from __future__ import annotations

import logging
import os
from logging.handlers import TimedRotatingFileHandler

import colorlog

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOGS_DIR: str = "logs"
LOG_FILENAME: str = "hft_lab.log"

# Daily rotation at midnight; keep 7 backup files
LOG_ROTATION_WHEN: str = "midnight"
LOG_BACKUP_COUNT: int = 7

DATE_FORMAT: str = "%Y-%m-%dT%H:%M:%S"

CONSOLE_FORMAT: str = (
    "%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s%(reset)s"
)
FILE_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

LOG_COLORS: dict[str, str] = {
    "DEBUG": "white",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}

# ---------------------------------------------------------------------------
# Internal state — prevents duplicate handler registration
# ---------------------------------------------------------------------------

_root_configured: bool = False


def _configure_root_logger() -> None:
    """Attach console and file handlers to the root logger exactly once.

    Idempotent: subsequent calls are no-ops, preventing duplicate output lines
    when multiple modules call ``get_logger`` on the same interpreter.
    """
    global _root_configured
    if _root_configured:
        return

    os.makedirs(LOGS_DIR, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # --- Console handler (colored) -------------------------------------------
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        colorlog.ColoredFormatter(
            CONSOLE_FORMAT,
            datefmt=DATE_FORMAT,
            log_colors=LOG_COLORS,
        )
    )

    # --- File handler (daily rotation, plain text) ---------------------------
    log_path = os.path.join(LOGS_DIR, LOG_FILENAME)
    file_handler = TimedRotatingFileHandler(
        filename=log_path,
        when=LOG_ROTATION_WHEN,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(fmt=FILE_FORMAT, datefmt=DATE_FORMAT)
    )

    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # Suppress verbose DEBUG output from third-party libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)

    # Suppress FinBERT/HuggingFace HTTP noise
    for _noisy in [
        "httpcore", "httpcore.connection", "httpcore.http11",
        "httpx", "huggingface_hub", "huggingface_hub.utils._http",
        "filelock", "urllib3.connectionpool",
        "websockets", "websockets.client",
        "alpaca.data.live.websocket",
    ]:
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    _root_configured = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_logger(name: str) -> logging.Logger:
    """Return a named logger backed by colored console and rotating file output.

    Call once per module, typically at module level::

        from logger import get_logger
        logger = get_logger(__name__)

    Args:
        name: Logger name, conventionally ``__name__`` of the calling module.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    _configure_root_logger()
    return logging.getLogger(name)
