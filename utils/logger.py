"""Structured logging setup using structlog.

Configures structlog with JSON output, ISO timestamps, and log level
from environment. All agents should import `log` from this module.
"""

import logging
import os

import structlog
from dotenv import load_dotenv

load_dotenv()


def setup_logging() -> None:
    """Configure structlog for structured JSON logging.

    Reads LOG_LEVEL from environment (default: INFO). Sets up processors
    for ISO timestamps, log level inclusion, and JSON rendering.
    """
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(message)s",
    )
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


# Run setup on import so all modules get consistent logging
setup_logging()

log: structlog.stdlib.BoundLogger = structlog.get_logger()
