"""Tests for the Phase-2 observability layer: logging configuration (terminal output format /
stream routing) and the per-image run manifest."""

import json
import logging

from nninfe.common.errors import ExportError, ImageIOError, InferenceError
from nninfe.common.logging_setup import configure_logging
from nninfe.common.manifest import build_run_context, classify_error, sha256_file, write_run_manifest


class TestLoggingSetup:
    def test_info_to_stdout_bare_format(self, capsys):
        configure_logging(force=True)
        logging.getLogger("nninfe.some.module").info("      progress line")
        out = capsys.readouterr()
        # Bare message on stdout — no level name / timestamp — so it matches the old print output.
        assert out.out == "      progress line\n"
        assert out.err == ""

    def test_warning_and_error_go_to_stderr(self, capsys):
        configure_logging(force=True)
        logging.getLogger("nninfe.x").warning("a warning")
        logging.getLogger("nninfe.x").error("an error")
        out = capsys.readouterr()
        assert "a warning" in out.err and "an error" in out.err
        assert out.out == ""  # warnings/errors must not double-print on stdout

    def test_level_threshold_hides_debug_by_default(self, capsys):
        configure_logging(force=True)
        logging.getLogger("nninfe.x").debug("noisy debug")
        assert capsys.readouterr().out == ""


class TestManifestHelpers:
    def test_sha256_matches_hashlib(self, tmp_path):
        import hashlib
        f = tmp_path / "blob.bin"
        f.write_bytes(b"nninfe model bytes")
        assert sha256_file(str(f)) == hashlib.sha256(b"nninfe model bytes").hexdigest()

    def test_classify_error_scopes_and_retryable(self):
        assert classify_error(ImageIOError("x")) == {"type": "ImageIOError", "scope": "image", "retryable": False}
        assert classify_error(InferenceError("x")) == {"type": "InferenceError", "scope": "batch", "retryable": True}
        assert classify_error(ExportError("x")) == {"type": "ExportError", "scope": "image", "retryable": True}
        # An unexpected programming bug is software-scoped and never auto-retryable.
        bug = classify_error(TypeError("boom"))
        assert bug == {"type": "TypeError", "scope": "software", "retryable": False}


class TestRunManifest:
    def _context(self, tmp_path):
        model = tmp_path / "model.onnx"
        model.write_bytes(b"fake-onnx")
        plan = tmp_path / "plan.json"
        plan.write_text("{}")
        return build_run_context(
            pipeline="detection", model_path=str(model), plan_path=str(plan),
            backend="cpu", providers=["CPUExecutionProvider"],
            params={"overlap": 0.5, "output_format": "nifti"},
        )

    def test_context_has_versions_hashes_params(self, tmp_path):
        ctx = self._context(tmp_path)
        assert ctx["pipeline"] == "detection"
        assert ctx["nninfe_version"]
        assert len(ctx["model"]["sha256"]) == 64 and len(ctx["plan"]["sha256"]) == 64
        assert ctx["providers"] == ["CPUExecutionProvider"]
        assert ctx["params"]["overlap"] == 0.5

    def test_write_success_manifest(self, tmp_path):
        ctx = self._context(tmp_path)
        out = tmp_path / "out"
        path = write_run_manifest(out, "case01", ctx, "ok", duration_s=1.2345)
        assert path == str(out / "case01_manifest.json")
        data = json.loads((out / "case01_manifest.json").read_text())
        assert data["image"] == "case01"
        assert data["outcome"] == "ok"
        assert data["duration_seconds"] == 1.234  # rounded to 3 decimals
        assert "error" not in data
        assert data["timestamp_utc"].endswith("+00:00")  # UTC

    def test_write_failure_manifest_carries_classification(self, tmp_path):
        ctx = self._context(tmp_path)
        out = tmp_path / "out"
        write_run_manifest(out, "case02", ctx, "failed", duration_s=0.5,
                           error=classify_error(InferenceError("oom")))
        data = json.loads((out / "case02_manifest.json").read_text())
        assert data["outcome"] == "failed"
        assert data["error"] == {"type": "InferenceError", "scope": "batch", "retryable": True}
