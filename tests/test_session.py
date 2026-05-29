"""Tests for session creation, inference, and integration functions."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nndet_onnx.nndet_onnx_inference_sw import (
    apply_nms,
    create_session,
    nms_nndet,
    parse_outputs,
    postprocess,
    run_inference,
)


class TestCreateSession:
    @patch("nndet_onnx.nndet_onnx_inference_sw.ort")
    def test_cpu_backend(self, mock_ort):
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = MagicMock()

        session = create_session("/fake/model.onnx", backend="cpu")

        mock_ort.InferenceSession.assert_called_once()
        assert session == mock_session

    @patch("nndet_onnx.nndet_onnx_inference_sw.ort")
    def test_cuda_backend(self, mock_ort):
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = MagicMock()

        create_session("/fake/model.onnx", backend="cuda")
        call_args = mock_ort.InferenceSession.call_args
        providers = call_args.kwargs.get("providers") or call_args[1].get("providers")
        assert "CUDAExecutionProvider" in providers

    @patch("nndet_onnx.nndet_onnx_inference_sw.ort")
    @patch("nndet_onnx.nndet_onnx_inference_sw.os.makedirs")
    def test_trt_backend_fp16(self, mock_makedirs, mock_ort):
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["TensorrtExecutionProvider"]
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = MagicMock()

        create_session("/fake/dir/model.onnx", backend="trt", trt_fp16=True)
        call_args = mock_ort.InferenceSession.call_args
        provider_options = call_args.kwargs.get("provider_options") or call_args[1].get("provider_options")
        # Find TRT provider options
        trt_opts = [o for o in provider_options if "trt_fp16_enable" in o]
        assert len(trt_opts) == 1
        assert trt_opts[0]["trt_fp16_enable"] == "True"
        assert trt_opts[0]["trt_engine_cache_enable"] == "True"

    @patch("nndet_onnx.nndet_onnx_inference_sw.ort")
    def test_openvino_backend(self, mock_ort):
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["OpenVINOExecutionProvider"]
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.SessionOptions.return_value = MagicMock()

        create_session("/fake/model.onnx", backend="openvino")
        call_args = mock_ort.InferenceSession.call_args
        provider_options = call_args.kwargs.get("provider_options") or call_args[1].get("provider_options")
        ov_opts = [o for o in provider_options if "device_type" in o]
        assert len(ov_opts) == 1
        assert ov_opts[0]["device_type"] == "CPU"


class TestRunInference:
    def test_calls_session_run(self):
        mock_session = MagicMock()
        mock_session.run.return_value = ["output1", "output2"]
        images = np.zeros((4, 1, 64, 96, 96), dtype=np.float32)
        anchors = np.zeros((4, 100, 6), dtype=np.float32)

        result = run_inference(mock_session, images, anchors)

        mock_session.run.assert_called_once_with(
            None, {"images": images, "anchors": anchors},
        )
        assert result == ["output1", "output2"]


class TestParseOutputs:
    def test_parses_batch_of_4(self):
        # 12 outputs: 4 boxes, 4 scores, 4 labels
        outputs = [
            np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32),  # boxes batch 0
            np.array([[7, 8, 9, 10, 11, 12]], dtype=np.float32),  # boxes batch 1
            np.array([[13, 14, 15, 16, 17, 18]], dtype=np.float32),  # boxes batch 2
            np.array([[19, 20, 21, 22, 23, 24]], dtype=np.float32),  # boxes batch 3
            np.array([0.9], dtype=np.float32),  # scores batch 0
            np.array([0.8], dtype=np.float32),  # scores batch 1
            np.array([0.7], dtype=np.float32),  # scores batch 2
            np.array([0.6], dtype=np.float32),  # scores batch 3
            np.array([0], dtype=np.int64),  # labels batch 0
            np.array([1], dtype=np.int64),  # labels batch 1
            np.array([0], dtype=np.int64),  # labels batch 2
            np.array([1], dtype=np.int64),  # labels batch 3
        ]
        detections = parse_outputs(outputs, batch_size=4)
        assert len(detections) == 4
        np.testing.assert_array_equal(detections[0]["boxes"], outputs[0])
        np.testing.assert_array_equal(detections[0]["scores"], outputs[4])
        np.testing.assert_array_equal(detections[0]["labels"], outputs[8])
        np.testing.assert_array_equal(detections[2]["scores"], outputs[6])


class TestApplyNms:
    def test_numpy_backend(self):
        detection = {
            "boxes": np.array([[0, 0, 10, 10, 0, 10]], dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
            "labels": np.array([0], dtype=np.int64),
        }
        result = apply_nms(detection, iou_threshold=0.3, nms_backend="numpy")
        assert len(result["boxes"]) == 1

    def test_nndet_backend_import_error(self):
        detection = {
            "boxes": np.array([[0, 0, 10, 10, 0, 10]], dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
            "labels": np.array([0], dtype=np.int64),
        }
        with patch.dict("sys.modules", {"torch": None, "nndet": None, "nndet.core": None, "nndet.core.boxes": None}):
            with pytest.raises((RuntimeError, ImportError)):
                nms_nndet(detection, iou_threshold=0.3)


class TestPostprocess:
    def test_full_pipeline(self):
        detection = {
            "boxes": np.array([
                [0, 0, 20, 20, 0, 20],   # large, high score → keep
                [0, 0, 1, 1, 0, 1],       # tiny → filtered by size
                [0, 0, 20, 20, 0, 20],    # duplicate → NMS
            ], dtype=np.float32),
            "scores": np.array([0.9, 0.8, 0.85], dtype=np.float32),
            "labels": np.array([0, 0, 0], dtype=np.int64),
        }
        result = postprocess(
            detection,
            spacing_xyz=(1.0, 1.0, 1.0),
            score_thresh=0.5,
            min_size_mm=2.0,
            iou_threshold=0.3,
        )
        # Box 1 filtered by size, box 2 suppressed by NMS (overlaps box 0)
        assert len(result["boxes"]) == 1
        assert result["scores"][0] == pytest.approx(0.9)
