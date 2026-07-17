"""Tests for the Phase-1 robustness layer: error taxonomy, narrow error translation (expected
dependency errors become typed; programming bugs propagate), PHI-sober messages, per-image
completion sentinels, and usage vs runtime exit codes."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from nninfe.common.errors import (
    EXIT_INFERENCE,
    EXIT_SESSION,
    EXIT_USAGE,
    ExportError,
    ImageIOError,
    InferenceError,
    NnInfeError,
    SessionError,
    fail_usage,
    translate_errors,
    write_image_status,
)


class TestErrorTaxonomy:
    def test_exit_codes_are_distinct(self):
        codes = [SessionError.exit_code, InferenceError.exit_code,
                 ImageIOError.exit_code, ExportError.exit_code]
        assert len(set(codes)) == len(codes)
        assert SessionError.exit_code == EXIT_SESSION
        assert InferenceError.exit_code == EXIT_INFERENCE

    def test_all_typed_errors_subclass_base(self):
        for cls in (SessionError, InferenceError, ImageIOError, ExportError):
            assert issubclass(cls, NnInfeError)


class TestTranslateErrors:
    def test_translates_only_expected_exceptions(self):
        with pytest.raises(ImageIOError) as exc:
            with translate_errors(ImageIOError, (OSError, ValueError), "cannot read image"):
                raise ValueError("bad geometry at /data/PATIENT_X/scan")
        # Message is sober: our stable text, NOT the raw cause (which could carry a PHI path).
        assert str(exc.value) == "cannot read image"
        assert "PATIENT_X" not in str(exc.value)
        # ...but the original cause is preserved for the logs/traceback.
        assert isinstance(exc.value.__cause__, ValueError)

    def test_unexpected_exception_propagates_as_bug(self):
        # A programming bug (not in `caught`) must NOT be relabeled as an operational error —
        # it propagates so the top level turns it into EXIT_RUNTIME with a traceback.
        with pytest.raises(AttributeError):
            with translate_errors(ImageIOError, (OSError, ValueError), "cannot read image"):
                raise AttributeError("obj has no attribute 'shpae'")

    def test_typed_error_passes_through(self):
        with pytest.raises(InferenceError):
            with translate_errors(ExportError, (OSError,), "exporting"):
                raise InferenceError("inference blew up")

    def test_no_error_is_transparent(self):
        with translate_errors(ExportError, (OSError,), "exporting"):
            value = 1 + 1
        assert value == 2


class TestFailUsage:
    def test_exits_with_usage_code_and_stderr_message(self, capsys):
        with pytest.raises(SystemExit) as exc:
            fail_usage("--overlap must be in [0, 1)")
        assert exc.value.code == EXIT_USAGE
        assert "--overlap must be in" in capsys.readouterr().err


class TestWriteImageStatus:
    def test_success_writes_done(self, tmp_path):
        path = write_image_status(tmp_path, "case01", ok=True)
        assert path == str(tmp_path / "case01.done")
        assert (tmp_path / "case01.done").exists()
        assert not (tmp_path / "case01.failed").exists()

    def test_failure_writes_failed_with_detail(self, tmp_path):
        path = write_image_status(tmp_path, "case02", ok=False, detail="InferenceError: ONNX inference failed")
        assert path == str(tmp_path / "case02.failed")
        assert (tmp_path / "case02.failed").read_text() == "InferenceError: ONNX inference failed"

    def test_creates_missing_output_dir(self, tmp_path):
        nested = tmp_path / "does" / "not" / "exist"
        write_image_status(nested, "case03", ok=True)
        assert (nested / "case03.done").exists()


class TestSessionAndInferenceErrors:
    def test_create_session_raises_session_error_on_bad_model(self, tmp_path):
        from nninfe.common.session import create_session

        bad_model = tmp_path / "not_a_model.onnx"
        bad_model.write_bytes(b"definitely not onnx")
        with pytest.raises(SessionError) as exc:
            create_session(str(bad_model), backend="cpu")
        assert exc.value.exit_code == EXIT_SESSION

    def test_run_inference_wraps_ort_error(self):
        from onnxruntime.capi.onnxruntime_pybind11_state import Fail

        from nninfe.common.session import run_inference

        session = MagicMock()
        session.run.side_effect = Fail("simulated GPU OOM")
        with pytest.raises(InferenceError) as exc:
            run_inference(session, np.zeros((1, 1, 4, 4, 4), np.float32), np.zeros((1, 1, 6), np.float32))
        assert exc.value.exit_code == EXIT_INFERENCE
        assert isinstance(exc.value.__cause__, Fail)  # cause preserved
        assert "simulated GPU OOM" not in str(exc.value)  # sober message

    def test_run_inference_lets_bugs_propagate(self):
        from nninfe.common.session import run_inference

        session = MagicMock()
        session.run.side_effect = TypeError("this is a real code bug")
        # Not an ORT error -> must NOT be wrapped as InferenceError.
        with pytest.raises(TypeError):
            run_inference(session, np.zeros((1, 1, 4, 4, 4), np.float32), np.zeros((1, 1, 6), np.float32))

    def test_segmentation_batch_wraps_ort_error(self):
        from onnxruntime.capi.onnxruntime_pybind11_state import Fail

        from nninfe.segmentation.pipeline import run_sliding_window_segmentation

        session = MagicMock()
        session.get_inputs.return_value = [MagicMock(name="images", shape=[1, 1, 4, 4, 4])]
        session.run.side_effect = Fail("engine incompatible")
        with pytest.raises(InferenceError):
            run_sliding_window_segmentation(
                session=session,
                volume_zyx=np.zeros((4, 4, 4), np.float32),
                patch_size_zyx=(4, 4, 4),
                batch_size=1,
                overlap=0.5,
            )
