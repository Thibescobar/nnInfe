"""Logging configuration for the pipeline.

The pipeline's human-facing progress is emitted through the standard ``logging`` module rather
than ``print`` — which gives levels, and a place to attach extra sinks (a file, a JSON handler,
an orchestrator's telemetry) later — while keeping the **terminal output byte-identical** to the
previous ``print``-based output:

- INFO/DEBUG records go to **stdout**, WARNING and above to **stderr** (same split as before, where
  progress printed to stdout and errors to stderr).
- The formatter is bare ``%(message)s`` — no level name, no timestamp — so ``logger.info(x)``
  renders exactly the line ``print(x)`` produced.
- ``StreamHandler`` flushes on every record, matching the previous ``print(..., flush=True)``.

Modules log through ``logging.getLogger(__name__)`` (all under the ``nninfe`` parent); the CLI
entrypoints call :func:`configure_logging` once. Verbosity is a level knob only — it never changes
the *format* of a line that is shown.
"""

import logging
import sys

_LOGGER_NAME = "nninfe"


class _MaxLevelFilter(logging.Filter):
    """Allow only records at or below *max_level* (keeps WARNING+ off the stdout handler so they
    are not printed twice)."""

    def __init__(self, max_level: int):
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


def configure_logging(level: int = logging.INFO, force: bool = False) -> logging.Logger:
    """Configure the ``nninfe`` logger and return it. Idempotent unless *force* is set (which
    rebuilds the handlers, e.g. to re-bind them to a freshly redirected ``sys.stdout``).

    *level* is a threshold only: at INFO the terminal shows exactly what it showed before; DEBUG
    would add lower-level records without altering any existing line's format."""
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False  # own handlers only; never double-log through the root logger

    if logger.handlers and not force:
        return logger
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    fmt = logging.Formatter("%(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(_MaxLevelFilter(logging.INFO))
    stdout_handler.setFormatter(fmt)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(fmt)

    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    return logger
