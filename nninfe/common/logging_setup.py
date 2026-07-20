"""Logging configuration for the pipeline.

The pipeline's human-facing progress is emitted through the standard ``logging`` module rather
than ``print`` — which gives levels and a place to attach extra sinks (a file, a JSON handler, an
orchestrator's telemetry) later. The console output is kept clean and readable:

- INFO/DEBUG records go to **stdout**, WARNING and above to **stderr** (diagnostics separated from
  progress, so results piped from stdout stay clean).
- The formatter is bare ``%(message)s`` — no level name, no timestamp. For a watched console that
  is the *readable* choice; a richer format (levels/timestamps, JSON) belongs on a separate sink
  added at ingestion time, not on the terminal.
- ``StreamHandler`` flushes on every record.

Modules log through ``logging.getLogger(__name__)`` (all under the ``nninfe`` parent). A
``NullHandler`` is attached at import so that using nninfe as a *library* without calling
:func:`configure_logging` stays silent (standard library behaviour); the CLI entrypoints call
:func:`configure_logging` once to attach the real console handlers.
"""

import logging
import sys

_LOGGER_NAME = "nninfe"
_configured = False

# Library-safe default: no output (and no "no handlers" warning) until an app configures logging.
logging.getLogger(_LOGGER_NAME).addHandler(logging.NullHandler())


def configure_logging(level: int = logging.INFO, force: bool = False) -> logging.Logger:
    """Configure the ``nninfe`` logger for console output and return it. Idempotent unless *force*
    is set (which rebuilds the handlers, e.g. to re-bind them to a freshly redirected
    ``sys.stdout``). *level* is a threshold only; it never changes the format of a shown line."""
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False  # own handlers only; never double-log through the root logger

    if _configured and not force:
        return logger
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    fmt = logging.Formatter("%(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)  # keep WARNING+ off stdout
    stdout_handler.setFormatter(fmt)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(fmt)

    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    _configured = True
    return logger
