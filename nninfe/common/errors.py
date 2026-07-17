"""Error taxonomy and process exit codes for orchestration.

Kept deliberately small and portable — plain exception classes carrying an integer
``exit_code``, plus a stage-wrapping context manager — so the planned C++ port can mirror the
same contract. Each pipeline stage raises a *typed* error; the CLI maps it to a distinct exit
code so an external orchestrator can react per failure class instead of parsing tracebacks.

Exit-code contract (stable — external orchestration depends on it):

======  ============================================================
 code    meaning
======  ============================================================
 0       success (all images processed)
 1       unexpected / uncaught runtime error
 2       usage: bad CLI arguments or configuration
 3       session: ONNX Runtime session creation failed
 4       inference: model execution failed
 5       image I/O: reading/preprocessing an input image failed
 6       export: writing a result file failed
 7       partial: batch finished but at least one image failed
======  ============================================================
"""

import sys
from contextlib import contextmanager
from pathlib import Path

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_USAGE = 2
EXIT_SESSION = 3
EXIT_INFERENCE = 4
EXIT_IMAGE_IO = 5
EXIT_EXPORT = 6
EXIT_PARTIAL = 7


class NnInfeError(Exception):
    """Base class for typed pipeline errors. ``exit_code`` is the process exit code the CLI
    returns when this error reaches the top level for a single image / fatal stage."""

    exit_code = EXIT_RUNTIME


class SessionError(NnInfeError):
    """ONNX Runtime session creation failed (bad model, driver mismatch, missing provider)."""

    exit_code = EXIT_SESSION


class InferenceError(NnInfeError):
    """Model execution failed (GPU OOM, engine incompatibility, malformed I/O binding)."""

    exit_code = EXIT_INFERENCE


class ImageIOError(NnInfeError):
    """Reading or preprocessing an input image failed (unreadable file, bad geometry)."""

    exit_code = EXIT_IMAGE_IO


class ExportError(NnInfeError):
    """Writing a result file failed (unwritable path, encoder error)."""

    exit_code = EXIT_EXPORT


@contextmanager
def translate_errors(error_cls, caught, message: str):
    """Translate *expected* dependency errors into a typed application error, and let everything
    else propagate.

    Only exceptions whose type is in *caught* (a type or tuple of types) are converted to
    *error_cls* — these are the operational failures a dependency is known to raise (bad model,
    unreadable file, disk full, …). Any other exception is a *programming bug* (``TypeError``,
    ``AttributeError``, a logic error) and is deliberately left to propagate so it surfaces as
    :data:`EXIT_RUNTIME` with a full traceback instead of being mislabeled as an operational
    failure — the distinction an orchestrator needs to decide "retry" vs "this is broken, alert".

    *message* is a **stable, sober** description (no interpolated ``str(exc)``): the original cause
    is chained via ``from exc`` for the logs/traceback, but is kept out of the message so a
    dependency detail (e.g. a filesystem path carrying a patient identifier) is never baked into a
    persisted error string. Already-typed :class:`NnInfeError`\\ s pass through unchanged.
    """
    try:
        yield
    except NnInfeError:
        raise
    except caught as exc:
        raise error_cls(message) from exc


def fail_usage(message: str) -> None:
    """Report a pre-flight usage/configuration error and exit with :data:`EXIT_USAGE`.

    Usage errors (bad arguments, missing/ill-formed inputs) are the caller's fault and are not the
    same as a programming bug: they print a clean message (no traceback) and exit ``2`` — distinct
    from an uncaught bug, which exits ``1`` with a traceback. Mirrors argparse's own exit code."""
    print(f"Error: {message}", file=sys.stderr, flush=True)
    raise SystemExit(EXIT_USAGE)


def write_image_status(output_dir, image_name: str, ok: bool, detail: str = "") -> str:
    """Write a per-image completion sentinel for external orchestration: ``{image_name}.done``
    on success or ``{image_name}.failed`` (carrying *detail*) on failure. This replaces the old
    single shared ``.done`` file so a batch reports the outcome of *each* image, not just that the
    last one finished. *detail* should be a sober, non-PHI string (e.g. the error type + a stable
    message); raw dependency messages are kept out of this persisted file. Returns the path."""
    suffix = "done" if ok else "failed"
    path = Path(output_dir) / f"{image_name}.{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(detail or suffix)
    return str(path)
