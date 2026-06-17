"""Tests for main() and process_single_image() with mocked I/O."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import SimpleITK as sitk

from nndet_onnx.nndet_onnx_inference_sw import process_single_image


def _make_plan():
    return {
        "patch_size": [8, 8, 8],
        "target_spacing": [1.0, 1.0, 1.0],
        "intensity_properties": {
            "percentile_00_5": 0.0,
            "percentile_99_5": 1000.0,
            "mean": 500.0,
            "std": 200.0,
        },
        "anchors": {
            "width": [4, 8],
            "height": [4, 8],
            "depth": [4, 8],
        },
        "architecture": {
            "strides": [[1, 1, 1], [2, 2, 2]],
            "decoder_levels": [1, 2],
        },
        "inference_plan": {
            "model_iou": 0.1,
            "model_score_thresh": 0.5,
            "model_topk": 100,
            "model_detections_per_image": 100,
            "remove_small_boxes": 2.0,
        },
    }


def _make_nifti(tmp_path, shape=(16, 16, 16), name="test.nii.gz"):
    arr = np.random.uniform(0, 1000, shape).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    path = str(tmp_path / name)
    sitk.WriteImage(img, path)
    return path


class TestProcessSingleImage:
    def test_returns_detection_count(self, tmp_path):
        """process_single_image runs end-to-end and returns int count."""
        nifti_path = _make_nifti(tmp_path)
        plan = _make_plan()
        batch_size = 2
        patch_size = tuple(plan["patch_size"])

        # Mock session
        mock_session = MagicMock()
        # Each run returns: batch_size boxes + batch_size scores + batch_size labels
        mock_session.run.return_value = [
            # boxes per batch element
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0, 6), dtype=np.float32),
            # scores
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            # labels
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        ]

        from nndet_onnx.detection.anchors import compute_anchors
        anchors_batch = compute_anchors(plan, patch_size, batch_size)

        output_dir = tmp_path / "results"

        result = process_single_image(
            session=mock_session,
            image_path=nifti_path,
            output_dir=output_dir,
            plan_inference=plan,
            patch_size=patch_size,
            batch_size=batch_size,
            anchors_batch=anchors_batch,
            iou_threshold=0.1,
            overlap=0.5,
            score_thresh=0.5,
            min_size_mm=2.0,
            nms_backend="numpy",
            no_global_nms=False,
            export_pkl=False,
        )

        assert isinstance(result, int)
        assert result >= 0
        # Output files should exist
        assert (output_dir / "test_mask.nii.gz").exists()
        assert (output_dir / "test_boxes.json").exists()
        assert (output_dir / "test_boxes.csv").exists()

    def test_with_detections(self, tmp_path):
        """process_single_image with actual detections from mock session."""
        nifti_path = _make_nifti(tmp_path)
        plan = _make_plan()
        batch_size = 2
        patch_size = tuple(plan["patch_size"])

        mock_session = MagicMock()

        def fake_run(_, inputs):
            return [
                # boxes: one detection per batch element
                np.array([[1, 1, 5, 5, 1, 5]], dtype=np.float32),
                np.array([[2, 2, 6, 6, 2, 6]], dtype=np.float32),
                # scores
                np.array([0.9], dtype=np.float32),
                np.array([0.85], dtype=np.float32),
                # labels
                np.array([0], dtype=np.int64),
                np.array([0], dtype=np.int64),
            ]

        mock_session.run.side_effect = fake_run

        from nndet_onnx.detection.anchors import compute_anchors
        anchors_batch = compute_anchors(plan, patch_size, batch_size)
        output_dir = tmp_path / "results"

        result = process_single_image(
            session=mock_session,
            image_path=nifti_path,
            output_dir=output_dir,
            plan_inference=plan,
            patch_size=patch_size,
            batch_size=batch_size,
            anchors_batch=anchors_batch,
            iou_threshold=0.1,
            overlap=0.5,
            score_thresh=0.3,
            min_size_mm=0.5,
            nms_backend="numpy",
            no_global_nms=False,
            export_pkl=True,
        )

        assert result > 0
        assert (output_dir / "test_boxes.pkl").exists()

    def test_no_global_nms(self, tmp_path):
        """process_single_image with no_global_nms=True."""
        nifti_path = _make_nifti(tmp_path)
        plan = _make_plan()
        batch_size = 2
        patch_size = tuple(plan["patch_size"])

        mock_session = MagicMock()
        mock_session.run.return_value = [
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        ]

        from nndet_onnx.detection.anchors import compute_anchors
        anchors_batch = compute_anchors(plan, patch_size, batch_size)
        output_dir = tmp_path / "results"

        result = process_single_image(
            session=mock_session,
            image_path=nifti_path,
            output_dir=output_dir,
            plan_inference=plan,
            patch_size=patch_size,
            batch_size=batch_size,
            anchors_batch=anchors_batch,
            iou_threshold=0.1,
            overlap=0.5,
            score_thresh=0.5,
            min_size_mm=2.0,
            nms_backend="numpy",
            no_global_nms=True,
            export_pkl=False,
        )
        assert isinstance(result, int)


class TestMain:
    @patch("nndet_onnx.nndet_onnx_inference_sw.create_session")
    def test_main_single_image(self, mock_create_session, tmp_path):
        """main() with a single image in minimal mode."""
        import json

        from nndet_onnx.nndet_onnx_inference_sw import main

        # Write plan
        plan = _make_plan()
        plan_path = str(tmp_path / "plan.json")
        with open(plan_path, "w") as f:
            json.dump(plan, f)

        # Write fake ONNX (main only checks extension + existence)
        model_path = str(tmp_path / "model.onnx")
        Path(model_path).write_bytes(b"fake")

        # Write NIfTI
        nifti_path = _make_nifti(tmp_path)

        output_dir = str(tmp_path / "out")

        # Mock session
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(shape=[2, 1, 8, 8, 8])]
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_session.run.return_value = [
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        ]
        mock_create_session.return_value = mock_session

        with patch(
            "sys.argv",
            [
                "prog",
                "--model-path", model_path,
                "--plan-path", plan_path,
                "--image-path", nifti_path,
                "--output-dir", output_dir,
                "--backend", "cpu",
            ],
        ):
            main()

        assert Path(output_dir).exists()

    @patch("nndet_onnx.nndet_onnx_inference_sw.create_session")
    def test_main_build_engine_only(self, mock_create_session, tmp_path):
        """main() with --build-engine-only exits early."""
        import json

        from nndet_onnx.nndet_onnx_inference_sw import main

        plan = _make_plan()
        plan_path = str(tmp_path / "plan.json")
        with open(plan_path, "w") as f:
            json.dump(plan, f)

        model_path = str(tmp_path / "model.onnx")
        Path(model_path).write_bytes(b"fake")

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(shape=[2, 1, 8, 8, 8])]
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_create_session.return_value = mock_session

        with patch(
            "sys.argv",
            [
                "prog",
                "--model-path", model_path,
                "--plan-path", plan_path,
                "--backend", "cpu",
                "--build-engine-only",
            ],
        ):
            main()

        # No inference should have been run
        mock_session.run.assert_not_called()

    @patch("nndet_onnx.nndet_onnx_inference_sw.create_session")
    def test_main_batch_mode(self, mock_create_session, tmp_path):
        """main() with --image-dir processes multiple images."""
        import json

        from nndet_onnx.nndet_onnx_inference_sw import main

        plan = _make_plan()
        plan_path = str(tmp_path / "plan.json")
        with open(plan_path, "w") as f:
            json.dump(plan, f)

        model_path = str(tmp_path / "model.onnx")
        Path(model_path).write_bytes(b"fake")

        # Write 2 NIfTI images in a directory
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        _make_nifti(img_dir, name="img1.nii.gz")
        _make_nifti(img_dir, name="img2.nii.gz")

        output_dir = str(tmp_path / "out")

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(shape=[2, 1, 8, 8, 8])]
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_session.run.return_value = [
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0, 6), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        ]
        mock_create_session.return_value = mock_session

        with patch(
            "sys.argv",
            [
                "prog",
                "--model-path", model_path,
                "--plan-path", plan_path,
                "--image-dir", str(img_dir),
                "--output-dir", output_dir,
                "--backend", "cpu",
            ],
        ):
            main()

        # Both images should have outputs
        assert (Path(output_dir) / "img1_boxes.json").exists()
        assert (Path(output_dir) / "img2_boxes.json").exists()

    def test_main_missing_model_exits(self, tmp_path):
        """main() exits if model file doesn't exist."""
        import json

        from nndet_onnx.nndet_onnx_inference_sw import main

        plan = _make_plan()
        plan_path = str(tmp_path / "plan.json")
        with open(plan_path, "w") as f:
            json.dump(plan, f)

        with patch(
            "sys.argv",
            [
                "prog",
                "--model-path", "/nonexistent/model.onnx",
                "--plan-path", plan_path,
                "--image-path", "/fake/image.nii.gz",
                "--output-dir", str(tmp_path / "out"),
            ],
        ):
            with pytest.raises(SystemExit):
                main()

    def test_main_mutual_exclusion(self, tmp_path):
        """main() exits if both --image-path and --image-dir given."""
        import json

        from nndet_onnx.nndet_onnx_inference_sw import main

        plan = _make_plan()
        plan_path = str(tmp_path / "plan.json")
        with open(plan_path, "w") as f:
            json.dump(plan, f)

        model_path = str(tmp_path / "model.onnx")
        Path(model_path).write_bytes(b"fake")

        nifti_path = _make_nifti(tmp_path)
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        with patch(
            "sys.argv",
            [
                "prog",
                "--model-path", model_path,
                "--plan-path", plan_path,
                "--image-path", nifti_path,
                "--image-dir", str(img_dir),
                "--output-dir", str(tmp_path / "out"),
            ],
        ):
            with pytest.raises(SystemExit):
                main()
