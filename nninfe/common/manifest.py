"""Per-image run manifest — the audit/traceability record a medical-device pipeline needs.

For every processed image a small JSON ``{name}_manifest.json`` is written next to its results,
capturing *what produced this output*: the nninfe version, the exact model and plan (by SHA-256,
so a result can always be tied back to the binary that made it), the effective parameters, the
backend/providers actually used, the wall-clock duration, and the outcome. On failure it also
records the orchestrator-facing classification (scope + whether a retry could help) that the
coarse process exit code cannot convey.

Kept to the standard library (hashlib/json) and plain dicts so it stays portable to the C++ port.
The run-level context (version, hashes, backend, params) is built **once** per run and reused for
every image, so the model is hashed a single time regardless of batch size.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from nninfe import __version__
from nninfe.common.errors import (
    ExportError,
    ImageIOError,
    InferenceError,
    InputValidationError,
    NnInfeError,
    SessionError,
)

# Orchestrator-facing classification per error type: what is the blast radius, and could a retry
# plausibly succeed? Coarse on purpose — a signal, not a policy engine (the orchestrator decides).
_ERROR_CLASS = {
    ImageIOError: ("image", False),          # unreadable/bad input -> retrying won't help
    InputValidationError: ("image", False),  # invalid image content -> caller must fix the input
    InferenceError: ("batch", True),         # e.g. transient GPU OOM -> a retry (smaller batch) may work
    ExportError: ("image", True),            # e.g. destination momentarily unavailable / disk -> retryable
    SessionError: ("deployment", False),     # bad model / driver -> needs intervention
}


def sha256_file(path, _chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, read in chunks (a large model is hashed without loading it whole)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_error(exc: BaseException) -> dict:
    """Map an exception to ``{type, scope, retryable}`` for the manifest. Unknown / unexpected
    errors (programming bugs) are ``software``-scope and non-retryable."""
    for cls, (scope, retryable) in _ERROR_CLASS.items():
        if isinstance(exc, cls):
            return {"type": type(exc).__name__, "scope": scope, "retryable": retryable}
    is_typed = isinstance(exc, NnInfeError)
    return {"type": type(exc).__name__, "scope": "batch" if is_typed else "software", "retryable": False}


def build_run_context(pipeline: str, model_path, plan_path, backend: str, providers, params: dict) -> dict:
    """Build the run-level manifest fields shared by every image (computed once per run)."""
    return {
        "schema_version": 1,
        "nninfe_version": __version__,
        "pipeline": pipeline,
        "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
        "plan": {"path": str(plan_path), "sha256": sha256_file(plan_path)},
        "backend": backend,
        "providers": list(providers),
        "params": dict(params),
    }


def write_run_manifest(
    output_dir,
    image_name: str,
    run_context: dict,
    outcome: str,
    duration_s=None,
    error: dict = None,
) -> str:
    """Write ``{image_name}_manifest.json`` combining *run_context* with this image's outcome
    (``"ok"`` / ``"failed"``), duration and, on failure, the *error* classification. Returns the
    path written. *error* strings must already be PHI-sober (type/scope only, no raw cause)."""
    manifest = dict(run_context)
    manifest["image"] = image_name
    manifest["outcome"] = outcome
    manifest["duration_seconds"] = None if duration_s is None else round(duration_s, 3)
    manifest["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    if error is not None:
        manifest["error"] = error
    path = Path(output_dir) / f"{image_name}_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return str(path)
