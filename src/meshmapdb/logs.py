import logging
import sys

import structlog


def configure_logging(
    level: str = "INFO",
    *,
    fmt: str = "%(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """
    Configure both the standard library logger and structlog for your application.
    Call once at startup (for example, in your __main__ or wsgi entrypoint).
    """

    # 1) Configure built-in logging (so warnings, filelock, cryptography, etc. get captured)
    logging.basicConfig(
        format=fmt,
        datefmt=datefmt,
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    # 2) Tell structlog to wrap stdlib loggers and emit key-value or JSON events
    structlog.configure(
        processors=[
            # Add the logger name & level to event_dict
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            # Timestamp
            structlog.processors.TimeStamper(fmt="iso"),
            # Render as JSON or as key/value text
            structlog.processors.JSONRenderer(),  # for JSON output
            # structlog.dev.ConsoleRenderer(colors=False)  # for human-readable
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


# Immediately configure on import if not already configured
configure_logging()


# Expose one global entrypoint for loggers
def get_logger(name: str = None):
    """
    Return a structlog logger bound to the given name.
    If no name is provided, the root logger is returned.
    """
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger()


# A module-level default logger
logger = get_logger("combined_json_db")
